#include "benchmark/replay.hpp"
#include "benchmark/results.hpp"
#include "benchmark_policy_adapter.hpp"
#include "runtime/environment.hpp"
#include "runtime/hardware_profile.hpp"
#include "llama.h"

#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>

namespace qb = quickserve::benchmark;

namespace {

void quiet_llama_log(ggml_log_level, const char *, void *) {}

void print_progress(std::uint64_t submitted, std::uint64_t completed,
                    std::uint64_t total,
                    RequestState::TimePoint started,
                    RequestState::TimePoint now) {
  const double elapsed_seconds =
      std::max(0.001, std::chrono::duration<double>(now - started).count());
  const double percent = total == 0 ? 100.0 :
      100.0 * static_cast<double>(completed) / static_cast<double>(total);
  const double completion_rate = static_cast<double>(completed) / elapsed_seconds;
  std::cout << '\r' << "Progress: " << completed << '/' << total << " ("
            << std::fixed << std::setprecision(1) << percent << "%) | submitted "
            << submitted << " | elapsed " << elapsed_seconds / 60.0 << "m | ETA ";
  if (completion_rate > 0.0) {
    const double remaining = static_cast<double>(total - completed);
    std::cout << remaining / completion_rate / 60.0 << 'm';
  } else {
    std::cout << "--";
  }
  std::cout << "          " << std::flush;
}

struct Options {
  std::filesystem::path trace;
  std::filesystem::path model;
  std::filesystem::path output_dir;
  std::filesystem::path policy_config;
  double target_qps{};
  qb::SelectionLimits limits;
  OutputMode output_mode = OutputMode::Natural;
  std::string output_mode_name = "natural";
  bool output_mode_supplied = false;
  std::uint32_t context_size = 4096;
  std::uint32_t batch_capacity = 512;
  std::uint32_t max_sequences = 16;
  std::uint32_t token_budget = 512;
};

std::uint64_t unsigned_value(const std::string &name, const std::string &text,
                             std::uint64_t maximum = std::numeric_limits<std::uint64_t>::max()) {
  if (text.empty() || text.find_first_not_of("0123456789") != std::string::npos)
    throw std::invalid_argument(name + " must be a positive integer");
  std::size_t used = 0;
  const auto value = std::stoull(text, &used);
  if (used != text.size() || value == 0 || value > maximum)
    throw std::invalid_argument(name + " is out of range");
  return value;
}

Options parse_options(int argc, char **argv) {
  Options options;
  for (int i = 1; i < argc; i += 2) {
    if (i + 1 >= argc) throw std::invalid_argument(std::string("missing value for ") + argv[i]);
    const std::string key = argv[i], value = argv[i + 1];
    if (key == "--trace") options.trace = value;
    else if (key == "--model") options.model = value;
    else if (key == "--output-dir") options.output_dir = value;
    else if (key == "--policy-config") options.policy_config = value;
    else if (key == "--target-qps") {
      std::size_t used = 0;
      options.target_qps = std::stod(value, &used);
      if (used != value.size() || !std::isfinite(options.target_qps) || options.target_qps <= 0)
        throw std::invalid_argument("target QPS must be finite and positive");
    } else if (key == "--max-requests") options.limits.max_requests = unsigned_value(key, value);
    else if (key == "--trace-duration") options.limits.trace_duration_ns = qb::parse_duration_ns(value);
    else if (key == "--output-mode") {
      if (value == "natural") options.output_mode = OutputMode::Natural;
      else if (value == "trace-exact") options.output_mode = OutputMode::TraceExact;
      else throw std::invalid_argument("output mode must be trace-exact or natural");
      options.output_mode_name = value;
      options.output_mode_supplied = true;
    } else if (key == "--context-size") options.context_size = unsigned_value(key, value, UINT32_MAX);
    else if (key == "--batch-capacity") options.batch_capacity = unsigned_value(key, value, UINT32_MAX);
    else if (key == "--max-sequences") options.max_sequences = unsigned_value(key, value, UINT32_MAX);
    else if (key == "--token-budget") options.token_budget = unsigned_value(key, value, UINT32_MAX);
    else throw std::invalid_argument("unknown argument: " + key);
  }
  if (options.trace.empty() || options.model.empty() || options.output_dir.empty() || options.target_qps == 0)
    throw std::invalid_argument("--trace, --model, --target-qps, and --output-dir are required");
  if (!options.output_mode_supplied)
    throw std::invalid_argument("--output-mode is required");
  if (!std::filesystem::is_regular_file(options.model)) throw std::invalid_argument("model path is not a regular file");
  return options;
}

const char *error_name(ErrorCode error) {
  switch (error) {
  case ErrorCode::None: return "";
  case ErrorCode::TokenizationFailed: return "tokenization_failed";
  case ErrorCode::ContextCapacityExceeded: return "context_capacity_exceeded";
  case ErrorCode::DecodeFailed: return "decode_failed";
  case ErrorCode::SamplingFailed: return "sampling_failed";
  case ErrorCode::ProtocolViolation: return "protocol_violation";
  case ErrorCode::EnvironmentStopped: return "environment_stopped";
  }
  return "unknown";
}

std::uint64_t ns_since(RequestState::TimePoint start, RequestState::TimePoint value) {
  const auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(value - start).count();
  if (elapsed < 0) throw std::runtime_error("benchmark clock moved backwards");
  return static_cast<std::uint64_t>(elapsed);
}

std::int64_t checked_arrival_lag(std::uint64_t actual,
                                 std::uint64_t scheduled) {
  constexpr auto limit = static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max());
  if (actual > limit || scheduled > limit)
    throw std::overflow_error("arrival time exceeds signed nanosecond range");
  return static_cast<std::int64_t>(actual) -
         static_cast<std::int64_t>(scheduled);
}

std::optional<std::uint64_t> elapsed(RequestState::TimePoint from, bool has_from,
                                     RequestState::TimePoint to, bool has_to) {
  if (!has_from || !has_to || to < from) return std::nullopt;
  return static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(to - from).count());
}

struct ActiveMetadata {
  qb::TraceRecord record;
  std::uint64_t scheduled_ns{};
};

class WorkerGuard {
public:
  WorkerGuard(Handoff &handoff, std::thread &environment,
              std::thread &replay)
      : handoff_(handoff), environment_(environment), replay_(replay) {}
  ~WorkerGuard() {
    handoff_.request_stop();
    if (replay_.joinable()) replay_.join();
    if (environment_.joinable()) environment_.join();
  }
  void join_all() {
    if (replay_.joinable()) replay_.join();
    if (environment_.joinable()) environment_.join();
  }
private:
  Handoff &handoff_;
  std::thread &environment_;
  std::thread &replay_;
};

int run(const Options &options) {
  const std::optional<HardwareProfile> hardware_profile =
      collect_macos_hardware_profile();
  if (!hardware_profile) {
    throw std::runtime_error("failed to collect macOS hardware profile");
  }

  if (options.token_budget > options.batch_capacity)
    throw std::invalid_argument("token budget may not exceed batch capacity");
  qb::TraceReader reader(options.trace);
  const qb::Selection selection = qb::select_prefix(reader, options.limits);
  (void)qb::scale_deadline_ns(selection.final_source_offset_ns, selection, options.target_qps);
  qb::validate_clock_headroom(selection, options.target_qps,
                              RequestState::Clock::now());

  qb::RunMetadata metadata;
  metadata.policy_name = QUICKSERVE_POLICY_NAME;
  metadata.policy_source_sha256 = QUICKSERVE_POLICY_SOURCE_SHA256;
  metadata.policy_header_sha256 = QUICKSERVE_POLICY_HEADER_SHA256;
  metadata.compiler_id = QUICKSERVE_COMPILER_ID;
  metadata.compiler_version = QUICKSERVE_COMPILER_VERSION;
  metadata.build_type = QUICKSERVE_BUILD_TYPE;
  metadata.quickserve_revision = QUICKSERVE_GIT_REVISION;
  metadata.llama_revision = QUICKSERVE_LLAMA_REVISION;
  metadata.model_path = std::filesystem::absolute(options.model).string();
  metadata.trace_path = std::filesystem::absolute(options.trace).string();
  metadata.trace_source_sha256 = reader.header().source_sha256_hex;
  metadata.trace_record_count = reader.header().record_count;
  metadata.trace_first_timestamp_ns = reader.header().first_timestamp_ns;
  metadata.trace_last_timestamp_ns = reader.header().last_timestamp_ns;
  metadata.selected_requests = selection.count;
  metadata.selected_source_duration_ns = selection.final_source_offset_ns;
  metadata.max_requests = options.limits.max_requests;
  metadata.trace_duration_limit_ns = options.limits.trace_duration_ns;
  metadata.last_scheduled_deadline_ns = qb::scale_deadline_ns(
      selection.final_source_offset_ns, selection, options.target_qps);
  metadata.scheduled_duration_ns = metadata.last_scheduled_deadline_ns;
  metadata.target_qps = options.target_qps;
  if (selection.count >= 2)
    metadata.offered_request_qps = qb::offered_qps(selection, options.target_qps);
  metadata.output_mode = options.output_mode_name;
  metadata.context_size = options.context_size;
  metadata.batch_capacity = options.batch_capacity;
  metadata.max_sequences = options.max_sequences;
  metadata.token_budget = options.token_budget;

  qb::AtomicResults results(options.output_dir);
  results.begin(metadata);
  const std::size_t queue_capacity = std::max<std::size_t>(
      64, static_cast<std::size_t>(options.max_sequences) * 4);
  Handoff handoff(options.token_budget, 3, queue_capacity);
  Environment environment(handoff, {options.model.string(), options.context_size,
                                     options.batch_capacity, options.max_sequences});
  std::thread environment_thread([&] { environment.run(); });
  std::thread replay_thread;
  WorkerGuard workers(handoff, environment_thread, replay_thread);
  const EnvironmentStartupResult startup = environment.wait_for_startup();
  if (!startup.success) {
    throw std::runtime_error("model startup failed: " + startup.error);
  }
  if (!startup.model_profile) {
    throw std::runtime_error("model startup succeeded without a model profile");
  }

  std::exception_ptr replay_error;
  std::uint64_t replay_wall_duration_ns = 0;
  replay_thread = std::thread([&] {
    std::unique_ptr<Scheduler> scheduler;
    try {
      const Scheduler::ClockFunction clock = [] { return RequestState::Clock::now(); };
      std::unordered_map<RequestId, ActiveMetadata> active_metadata;
      std::uint64_t submitted = 0;
      std::uint64_t observed = 0;
      std::uint64_t last_submitted_source_offset = 0;
      bool output_ok = true;
      RequestState::TimePoint start{};
      scheduler = quickserve_benchmark_adapter::create(
          handoff, options.token_budget, *startup.model_profile,
          *hardware_profile, options.policy_config);
      if (!scheduler) throw std::runtime_error("policy factory returned null");
      scheduler->set_clock(clock);
      scheduler->set_workload_observer([&](RequestState::TimePoint at,
                                            SchedulerWorkloadCounts counts) {
        results.sample_counts(ns_since(start, at), counts.active, counts.queued);
      });
      scheduler->set_batch_observer([&](const BatchOutcome &outcome) {
        const auto duration = std::max(
            RequestState::Clock::duration::zero(), outcome.duration);
        output_ok = output_ok && results.observe_batch(
            outcome.prefill_tokens, outcome.decode_items,
            static_cast<std::uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(duration)
                    .count()),
            outcome.success);
      });
      scheduler->enable_streaming_retirement([&](const RequestState &state) {
        try {
          const auto found = active_metadata.find(state.id);
          if (found == active_metadata.end()) return false;
          const ActiveMetadata meta = found->second;
          qb::RequestMetrics row;
          row.request_id = state.id;
          row.source_offset_ns = meta.record.arrival_offset_ns;
          row.scheduled_arrival_ns = meta.scheduled_ns;
          row.actual_arrival_ns = ns_since(start, state.arrival_time);
          row.arrival_lag_ns = checked_arrival_lag(row.actual_arrival_ns, row.scheduled_arrival_ns);
          row.input_tokens = meta.record.context_tokens;
          row.executed_input_tokens = state.prefill_position;
          row.requested_output_tokens = meta.record.generated_tokens;
          row.generated_output_tokens = state.decoded_count;
          row.queue_delay_ns = elapsed(state.arrival_time, true, state.start_time, state.start_recorded);
          row.ttft_ns = elapsed(state.arrival_time, true, state.first_token_time, state.first_token_recorded);
          row.prefill_ns = elapsed(state.start_time, state.start_recorded, state.first_token_time, state.first_token_recorded);
          if (state.decoded_count >= 2) {
            row.decode_span_ns = elapsed(state.first_token_time, state.first_token_recorded,
                                         state.last_token_time, state.last_token_recorded);
            if (row.decode_span_ns) row.tpot_ns = (*row.decode_span_ns + (state.decoded_count - 1) / 2) / (state.decoded_count - 1);
          }
          row.e2e_latency_ns = elapsed(state.arrival_time, true, state.finish_time, state.finish_recorded);
          const std::uint64_t total_tokens = static_cast<std::uint64_t>(row.input_tokens) + row.generated_output_tokens;
          if (row.e2e_latency_ns && total_tokens != 0) row.normalized_latency_ns_per_token = static_cast<double>(*row.e2e_latency_ns) / total_tokens;
          row.terminal_error = error_name(state.terminal_error);
          row.disposition = state.terminal_error == ErrorCode::None ? "success" :
              (!state.admission_succeeded ? "admission_rejection" : "execution_failure");
          row.eog_observed = state.eog_observed;
          row.output_mode = options.output_mode_name;
          output_ok = results.observe(row);
          if (output_ok) { active_metadata.erase(found); ++observed; }
          return output_ok;
        } catch (...) {
          output_ok = false;
          return false;
        }
      });

      qb::TraceCursor cursor = reader.cursor();
      qb::TraceRecord next;
      bool has_next = cursor.next(next);
      if (!has_next) throw std::runtime_error("trace changed before replay");
      qb::ReplayEngine replay(selection, options.target_qps, clock);
      start = replay.start();
      auto last_progress = start - std::chrono::seconds(2);
      while (output_ok && (submitted < selection.count || observed < submitted)) {
        auto now = replay.now();
        if (now - last_progress >= std::chrono::seconds(2)) {
          print_progress(submitted, observed, selection.count, start, now);
          last_progress = now;
        }
        std::uint64_t elapsed_ns = replay.elapsed_ns();
        while (output_ok && submitted < selection.count && has_next) {
          const auto deadline_ns = replay.deadline_ns(next);
          if (deadline_ns > elapsed_ns) break;
          const RequestId id = scheduler->submit_synthetic(next.context_tokens, next.generated_tokens, options.output_mode);
          active_metadata.emplace(id, ActiveMetadata{next, deadline_ns});
          last_submitted_source_offset = next.arrival_offset_ns;
          ++submitted;
          if (submitted < selection.count) {
            has_next = cursor.next(next);
            if (!has_next) throw std::runtime_error("trace changed during replay");
          } else {
            has_next = false;
          }
          now = replay.now();
          elapsed_ns = replay.elapsed_ns();
        }
        const auto snapshot = handoff.scheduler_progress_generation();
        (void)scheduler->run_once();
        if (!output_ok) break;
        if (submitted == selection.count && observed == submitted) break;
        auto deadline = replay.now() + std::chrono::milliseconds(10);
        if (submitted < selection.count && has_next) {
          deadline = replay.deadline(next);
        }
        (void)handoff.wait_for_scheduler_progress(snapshot, deadline);
      }
      if (submitted != selection.count ||
          last_submitted_source_offset != selection.final_source_offset_ns)
        throw std::runtime_error("replay trace no longer matches validated prefix");
      if (!output_ok) scheduler->request_stop();
      while (!scheduler->all_terminal()) (void)scheduler->run_once();
      print_progress(submitted, observed, selection.count, start, replay.now());
      std::cout << '\n';
      handoff.request_stop();
      if (!output_ok) throw std::runtime_error("failed writing per-request results");
      replay_wall_duration_ns = replay.elapsed_ns();
    } catch (...) {
      replay_error = std::current_exception();
      if (scheduler) {
        try {
          scheduler->request_stop();
          while (!scheduler->all_terminal()) (void)scheduler->run_once();
        } catch (...) {
          // Preserve the initiating failure; environment stop below is the
          // final escape hatch if cleanup itself cannot make progress.
        }
      }
      handoff.request_stop();
    }
  });
  workers.join_all();
  if (replay_error) std::rethrow_exception(replay_error);
  results.finish(replay_wall_duration_ns);
  std::cout << "Benchmark complete. Results: "
            << std::filesystem::absolute(options.output_dir).string() << '\n';
  return 0;
}

} // namespace

int main(int argc, char **argv) {
  llama_log_set(quiet_llama_log, nullptr);
  try { return run(parse_options(argc, argv)); }
  catch (const std::exception &error) {
    std::cerr << "quickserve_benchmark: " << error.what() << '\n';
    return 1;
  }
}
