#include "benchmark/replay.hpp"
#include "benchmark/results.hpp"
#include "policies/fifo.hpp"

#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <limits>
#include <string>

using namespace quickserve::benchmark;

namespace {

int g_failures = 0;
#define CHECK(condition)                                                       \
  do {                                                                         \
    if (!(condition)) {                                                        \
      std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #condition);       \
      ++g_failures;                                                            \
    }                                                                          \
  } while (false)

std::filesystem::path make_trace(const std::string &stem,
                                 const std::string &rows) {
  const auto csv = std::filesystem::temp_directory_path() / (stem + ".csv");
  const auto trace = std::filesystem::temp_directory_path() / (stem + ".qst");
  std::filesystem::remove(trace);
  std::ofstream out(csv);
  out << "TIMESTAMP,ContextTokens,GeneratedTokens\n" << rows;
  out.close();
  prepare_trace(csv, trace);
  std::filesystem::remove(csv);
  return trace;
}

void test_duration_parser() {
  const auto minutes = parse_duration_ns("15m");
  const auto milliseconds = parse_duration_ns("1ms");
  CHECK(minutes == 900000000000ULL);
  CHECK(milliseconds == 1000000ULL);
  bool threw = false;
  try { (void)parse_duration_ns("0s"); } catch (const std::invalid_argument &) { threw = true; }
  CHECK(threw);
}

void test_prefix_selection_and_scaling() {
  const auto path = make_trace(
      "quickserve-benchmark-test",
      "2026-01-01 00:00:00Z,10,2\n2026-01-01 00:00:01Z,20,3\n"
      "2026-01-01 00:00:02Z,30,4\n2026-01-01 00:00:03Z,40,5\n");
  TraceReader reader(path);
  SelectionLimits limits;
  limits.max_requests = 3;
  limits.trace_duration_ns = 2000000000ULL;
  const Selection selected = select_prefix(reader, limits);
  const auto first_deadline = scale_deadline_ns(1000000000ULL, selected, 4.0);
  const auto last_deadline = scale_deadline_ns(2000000000ULL, selected, 4.0);
  const auto offered = offered_qps(selected, 4.0);
  CHECK(selected.count == 3);
  CHECK(selected.final_source_offset_ns == 2000000000ULL);
  CHECK(first_deadline == 250000000ULL);
  CHECK(last_deadline == 500000000ULL);
  CHECK(std::abs(offered - 4.0) < 1e-9);
  std::filesystem::remove(path);
}

void test_zero_duration_multi_request_rejected() {
  const auto path = make_trace(
      "quickserve-benchmark-zero",
      "2026-01-01 00:00:00Z,1,1\n2026-01-01 00:00:00Z,1,1\n");
  TraceReader reader(path);
  bool threw = false;
  try { (void)select_prefix(reader, {}); } catch (const std::invalid_argument &) { threw = true; }
  CHECK(threw);
  std::filesystem::remove(path);
}

void test_replay_engine_uses_injected_clock_for_due_arrivals() {
  using Clock = std::chrono::steady_clock;
  Clock::time_point now{};
  Selection selection{3, 2000000000ULL};
  ReplayEngine replay(selection, 2.0, [&] { return now; });
  CHECK(replay.due({0, 1, 1}));
  CHECK(!replay.due({1000000000ULL, 1, 1}));
  now += std::chrono::milliseconds(500);
  CHECK(replay.due({1000000000ULL, 1, 1}));
  CHECK(replay.elapsed_ns() == 500000000ULL);
}

void test_qps_scaling_is_ties_even_and_checks_edges() {
  const Selection selection{3, 4};
  const auto tie_even_down = scale_deadline_ns(1, selection, 1000000000.0);
  const auto tie_even_up = scale_deadline_ns(3, selection, 1000000000.0);
  CHECK(tie_even_down == 0);
  CHECK(tie_even_up == 2);
  bool threw = false;
  try { (void)scale_deadline_ns(1, selection, std::numeric_limits<double>::infinity()); }
  catch (const std::invalid_argument &) { threw = true; }
  CHECK(threw);
  bool signed_overflow = false;
  try { (void)scale_deadline_ns(4, selection, 1.5e-10); }
  catch (const std::overflow_error &) { signed_overflow = true; }
  CHECK(signed_overflow);
  bool headroom_overflow = false;
  try {
    using Clock = std::chrono::steady_clock;
    Clock::time_point near_max = Clock::time_point::max() - std::chrono::nanoseconds(1);
    ReplayEngine too_late({2, 1}, 100000000.0, [&] { return near_max; });
    (void)too_late;
  } catch (const std::overflow_error &) { headroom_overflow = true; }
  CHECK(headroom_overflow);
}

void test_log_sketch_records_documented_bounds() {
  LogSketch sketch;
  sketch.add(0);
  sketch.add(1000000000000000001ULL);
  CHECK(sketch.underflow == 1);
  CHECK(sketch.overflow == 1);
  const long double actual_bin_error =
      std::exp(std::log(1000000000000000000.0L) / 4093.0L) - 1.0L;
  CHECK(log_sketch_relative_error() >= actual_bin_error);
  CHECK(log_sketch_relative_error() < 0.011L);
}

void test_publication_refuses_destination_created_after_begin() {
  const auto root = std::filesystem::temp_directory_path() / "quickserve-results-race";
  std::filesystem::remove_all(root);
  bool threw = false;
  {
    AtomicResults results(root);
    RunMetadata metadata;
    results.begin(metadata);
    std::filesystem::create_directory(root);
    try { results.finish(1); } catch (const std::exception &) { threw = true; }
  }
  CHECK(threw);
  std::filesystem::remove_all(root);
}

void complete_one_request(FifoScheduler &scheduler, Handoff &handoff) {
  const bool admission_running = scheduler.run_once();
  CHECK(admission_running);
  Admission admission;
  const bool took_admission = handoff.try_take_admission(admission);
  CHECK(took_admission);
  if (!took_admission || !admission.synthetic_prompt_tokens) return;
  const auto progress_before_publish = handoff.scheduler_progress_generation();
  const bool reported_admission = handoff.try_report_admission(
      {admission.id, *admission.synthetic_prompt_tokens, ErrorCode::None});
  CHECK(reported_admission);
  const bool plan_running = scheduler.run_once();
  CHECK(plan_running);
  const bool noticed_drained_publication = handoff.wait_for_scheduler_progress(
      progress_before_publish, std::chrono::steady_clock::now() +
                                   std::chrono::seconds(1));
  CHECK(noticed_drained_publication);
  Plan *plan = handoff.consume_plan();
  CHECK(plan != nullptr);
  if (plan == nullptr) return;
  CHECK(plan->work.size() == 1);
  if (plan->work.empty()) return;
  const WorkItem work = plan->work.front();
  const bool reported_completion = handoff.try_report_completion(
      {work.id, work.token_end, 1, 7, work.kind, ErrorCode::None, true, false});
  CHECK(reported_completion);
  handoff.retire_plan(plan);
  const bool release_running = scheduler.run_once();
  CHECK(release_running);
  Release release;
  const bool took_release = handoff.try_take_release(release);
  CHECK(took_release);
  if (!took_release) return;
  const bool acknowledged = handoff.try_acknowledge_release({release.id});
  CHECK(acknowledged);
  (void)scheduler.run_once();
}

void test_fake_clock_replay_drives_fifo_lifecycle() {
  using Clock = std::chrono::steady_clock;
  Clock::time_point now{};
  ReplayEngine replay({2, 1000000000ULL}, 1.0, [&] { return now; });
  Handoff handoff(8);
  FifoScheduler scheduler(handoff, 8);
  scheduler.set_clock(replay.clock_function());
  int observed = 0;
  scheduler.enable_streaming_retirement([&](const RequestState &) { ++observed; return true; });
  const TraceRecord first{0, 2, 1}, second{1000000000ULL, 2, 1};
  CHECK(replay.due(first));
  const auto first_id = scheduler.submit_synthetic(first.context_tokens, first.generated_tokens);
  CHECK(first_id == 0);
  complete_one_request(scheduler, handoff);
  CHECK(!replay.due(second));
  now += std::chrono::seconds(1);
  CHECK(replay.due(second));
  const auto second_id = scheduler.submit_synthetic(second.context_tokens, second.generated_tokens);
  CHECK(second_id == 1);
  complete_one_request(scheduler, handoff);
  CHECK(observed == 2);
  CHECK(scheduler.all_terminal());
}

void test_csv_and_summary_are_valid() {
  const auto root = std::filesystem::temp_directory_path() / "quickserve-results-test";
  std::filesystem::remove_all(root);
  AtomicResults results(root);
  RunMetadata run;
  run.policy_source_sha256 = "aa";
  run.policy_header_sha256 = "bb";
  run.policy_name = "fifo";
  run.target_qps = 2.0;
  run.selected_requests = 2;
  run.output_mode = "natural";
  results.begin(run);
  RequestMetrics row;
  row.request_id = 7;
  row.source_offset_ns = 0;
  row.scheduled_arrival_ns = 0;
  row.actual_arrival_ns = 10;
  row.input_tokens = 4;
  row.requested_output_tokens = 2;
  row.generated_output_tokens = 1;
  row.executed_input_tokens = 4;
  row.e2e_latency_ns = 90;
  row.normalized_latency_ns_per_token = 18.0;
  row.terminal_error = "value,with comma";
  row.disposition = "success";
  row.output_mode = "natural";
  const bool observed_row = results.observe(row);
  CHECK(observed_row);
  RequestMetrics rejected = row;
  rejected.request_id = 8;
  rejected.input_tokens = 100;
  rejected.executed_input_tokens = 0;
  rejected.generated_output_tokens = 0;
  rejected.e2e_latency_ns.reset();
  rejected.normalized_latency_ns_per_token.reset();
  rejected.terminal_error = "context_capacity_exceeded";
  rejected.disposition = "admission_rejection";
  const bool observed_rejection = results.observe(rejected);
  CHECK(observed_rejection);
  CHECK(results.observe_batch(8, 0, 100, true));
  CHECK(results.observe_batch(4, 2, 200, true));
  CHECK(results.observe_batch(0, 2, 50, false));
  results.finish(100);
  CHECK(std::filesystem::exists(root / "requests.csv"));
  CHECK(std::filesystem::exists(root / "summary.json"));
  std::ifstream csv(root / "requests.csv");
  std::string text((std::istreambuf_iterator<char>(csv)), {});
  CHECK(text.find("\"value,with comma\"") != std::string::npos);
  std::ifstream json(root / "summary.json");
  text.assign((std::istreambuf_iterator<char>(json)), {});
  CHECK(text.find("\"selected_requests\":2") != std::string::npos);
  CHECK(text.find("\"p50_ns\":90") != std::string::npos);
  CHECK(text.find("\"offered_request_qps\"") != std::string::npos);
  CHECK(text.find("\"achieved_request_qps\"") != std::string::npos);
  CHECK(text.find("\"input_token_throughput\"") != std::string::npos);
  CHECK(text.find("\"arrival_lag\"") != std::string::npos);
  CHECK(text.find("\"queue_delay\"") != std::string::npos);
  CHECK(text.find("\"ttft\"") != std::string::npos);
  CHECK(text.find("\"tpot\"") != std::string::npos);
  CHECK(text.find("\"input_buckets\"") != std::string::npos);
  CHECK(text.find("\"admission_rejections\":1") != std::string::npos);
  CHECK(text.find("\"executed_input_tokens\":4") != std::string::npos);
  CHECK(text.find("\"jain_fairness\":1") != std::string::npos);
  CHECK(text.find("\"failure_request_rate\"") != std::string::npos);
  CHECK(text.find("\"batches\":{\"total\":3") != std::string::npos);
  CHECK(text.find("\"pure_prefill\":1") != std::string::npos);
  CHECK(text.find("\"pure_decode\":1") != std::string::npos);
  CHECK(text.find("\"mixed\":1") != std::string::npos);
  CHECK(text.find("\"failed\":1") != std::string::npos);
  CHECK(text.find("\"prefill_tokens\":12") != std::string::npos);
  CHECK(text.find("\"decode_items\":4") != std::string::npos);
  CHECK(text.find("\"prefill_tokens_per_batch\":{"
                  "\"mean\":6,\"min\":4,\"max\":8}") !=
        std::string::npos);
  CHECK(text.find("\"mean_duration_ns\":{"
                  "\"all\":116.66666666666667,\"pure_prefill\":100,"
                  "\"pure_decode\":50,\"mixed\":200}") !=
        std::string::npos);
  CHECK(text.find("\"rejection_request_rate\"") != std::string::npos);
  CHECK(text.find("\"1_512\":{\"count\":2,\"success\":1,\"failure\":0,\"rejection\":1") != std::string::npos);
  CHECK(text.find("\"mean_ttft_ns\":null") != std::string::npos);
  const std::string sketch_key = "\"sketch_relative_error\":";
  const auto sketch_position = text.find(sketch_key);
  CHECK(sketch_position != std::string::npos);
  if (sketch_position != std::string::npos) {
    const long double reported = std::stold(
        text.substr(sketch_position + sketch_key.size()));
    const long double actual =
        std::exp(std::log(1000000000000000000.0L) / 4093.0L) - 1.0L;
    CHECK(reported >= actual);
  }
  std::filesystem::remove_all(root);
}

} // namespace

int main() {
  test_duration_parser();
  test_prefix_selection_and_scaling();
  test_zero_duration_multi_request_rejected();
  test_replay_engine_uses_injected_clock_for_due_arrivals();
  test_qps_scaling_is_ties_even_and_checks_edges();
  test_log_sketch_records_documented_bounds();
  test_publication_refuses_destination_created_after_begin();
  test_fake_clock_replay_drives_fifo_lifecycle();
  test_csv_and_summary_are_valid();
  if (g_failures != 0) return 1;
  std::printf("all benchmark checks passed\n");
  return 0;
}
