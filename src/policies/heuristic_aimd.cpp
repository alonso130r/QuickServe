#include "heuristic_aimd.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <stdexcept>
#include <unordered_set>

namespace {

void validate(const HeuristicAIMDConfig &config) {
  if (!std::isfinite(config.initial_window) ||
      !std::isfinite(config.minimum_window) ||
      !std::isfinite(config.maximum_window) ||
      !std::isfinite(config.additive_step) ||
      !std::isfinite(config.multiplicative_factor) ||
      config.minimum_window <= 0.0 ||
      config.maximum_window < config.minimum_window ||
      config.initial_window <= 0.0 || config.additive_step <= 0.0 ||
      config.multiplicative_factor <= 0.0 ||
      config.multiplicative_factor >= 1.0 ||
      config.target_batch_duration <= std::chrono::nanoseconds::zero() ||
      config.ttft_target <= std::chrono::nanoseconds::zero() ||
      config.tpot_target <= std::chrono::nanoseconds::zero() ||
      config.starvation_threshold == 0 ||
      config.max_prefill_chunk == 0) {
    throw std::invalid_argument("invalid heuristic AIMD configuration");
  }
}

HeuristicAIMDConfig validated(HeuristicAIMDConfig config) {
  validate(config);
  return config;
}

std::uint32_t window_capacity(double window) {
  constexpr auto maximum = std::numeric_limits<std::uint32_t>::max();
  if (window >= static_cast<double>(maximum)) return maximum;
  return static_cast<std::uint32_t>(std::max(1.0, std::floor(window)));
}

} // namespace

HeuristicAIMD::HeuristicAIMD(Handoff &handoff, std::uint32_t token_budget,
                             const ModelProfile &model_profile,
                             const HardwareProfile &hardware_profile)
    : HeuristicAIMD(handoff, token_budget, model_profile, hardware_profile,
                    HeuristicAIMDConfig{}) {}

HeuristicAIMD::HeuristicAIMD(Handoff &handoff, std::uint32_t token_budget,
                             const ModelProfile &model_profile,
                             const HardwareProfile &hardware_profile,
                             HeuristicAIMDConfig config)
    : Scheduler(handoff, token_budget), model_profile_(model_profile),
      hardware_profile_(hardware_profile), config_(validated(config)),
      prefill_window_(std::clamp(config_.initial_window, config_.minimum_window,
                                 config_.maximum_window)),
      decode_window_(prefill_window_) {}

void HeuristicAIMD::update_window(double sample, std::optional<double> &ewma,
                                  double &window, bool success) {
  if (!success) {
    window = std::max(window * config_.multiplicative_factor,
                      config_.minimum_window);
    return;
  }
  if (!std::isfinite(sample) || sample <= 0.0) return;
  if (!ewma) {
    ewma = sample;
    return;
  }

  constexpr double tolerance = 0.05;
  if (sample <= *ewma * (1.0 + tolerance)) {
    window = std::min(window + config_.additive_step, config_.maximum_window);
  } else {
    window = std::max(window * config_.multiplicative_factor,
                      config_.minimum_window);
  }
  constexpr double alpha = 0.2;
  *ewma = alpha * sample + (1.0 - alpha) * *ewma;
}

void HeuristicAIMD::on_plan_completed(const BatchOutcome &outcome) {
  const double elapsed_ns = static_cast<double>(outcome.duration.count());
  if (outcome.decode_items != 0 &&
      (outcome.prefill_tokens == 0 || !outcome.success)) {
    update_window(elapsed_ns / outcome.decode_items,
                  decode_time_per_item_ewma_, decode_window_, outcome.success);
  }
}

std::vector<HeuristicAIMD::Candidate> HeuristicAIMD::candidates() const {
  std::vector<Candidate> available;
  const RequestState::TimePoint now = policy_now();
  for (const RequestState &request : policy_requests()) {
    const auto bypass = bypasses_.find(request.id);
    const std::uint32_t bypass_count =
        bypass == bypasses_.end() ? 0 : bypass->second;
    if (request.stage == RequestState::Stage::Decode &&
        request.decoded_count > 0) {
      const std::uint64_t end =
          static_cast<std::uint64_t>(request.prompt_length) +
          request.decoded_count;
      if (end <= std::numeric_limits<std::uint32_t>::max()) {
        const RequestState::TimePoint reference =
            request.last_token_recorded
                ? request.last_token_time
                : (request.first_token_recorded
                       ? request.first_token_time
                       : (request.start_recorded ? request.start_time
                                                 : request.arrival_time));
        const auto elapsed = std::max(RequestState::Clock::duration::zero(),
                                      now - reference);
        const double urgency = std::min(
            4.0, std::chrono::duration<double>(elapsed).count() /
                     std::chrono::duration<double>(config_.tpot_target).count());
        available.push_back(
            {{request.id, static_cast<std::uint32_t>(end - 1),
              static_cast<std::uint32_t>(end), WorkKind::Decode},
             static_cast<std::uint32_t>(end), request.prompt_length, urgency,
             bypass_count});
      }
      continue;
    }
    if (request.stage == RequestState::Stage::Prefill &&
        request.prefill_position < request.prompt_length) {
      const std::uint32_t remaining =
          request.prompt_length - request.prefill_position;
      const std::uint32_t count =
          std::min({remaining, config_.max_prefill_chunk, token_budget_});
      const auto elapsed = std::max(RequestState::Clock::duration::zero(),
                                    now - request.arrival_time);
      const double urgency = std::min(
          4.0, std::chrono::duration<double>(elapsed).count() /
                   std::chrono::duration<double>(config_.ttft_target).count());
      available.push_back(
          {{request.id, request.prefill_position,
            request.prefill_position + count, WorkKind::Prefill},
           request.prefill_position + count, request.prompt_length, urgency,
           bypass_count});
    }
  }
  return available;
}

double HeuristicAIMD::utility(const Candidate &candidate) const {
  double value = candidate.work.kind == WorkKind::Decode
                     ? 1.0
                     : 0.1 * candidate.work.token_count();
  if (candidate.work.kind == WorkKind::Prefill &&
      candidate.work.token_end == candidate.prompt_end) {
    value += 0.25;
  }
  value += 2.0 * candidate.urgency + 0.25 * candidate.bypasses;
  if (candidate.bypasses >= config_.starvation_threshold) {
    value += 1000.0;
  }
  return value;
}

long double
HeuristicAIMD::additional_kv_bytes(const Candidate &candidate) const {
  const long double kv_dimension =
      static_cast<long double>(model_profile_.kv_head_count) *
      model_profile_.head_dimension;
  const long double kv_scalar_bytes =
      model_profile_.key_effective_bytes_per_scalar +
      model_profile_.value_effective_bytes_per_scalar;
  const std::uint32_t tokens = candidate.work.kind == WorkKind::Prefill
                                   ? candidate.work.token_count()
                                   : 1;
  return static_cast<long double>(model_profile_.layer_count) * kv_dimension *
         kv_scalar_bytes * tokens;
}

BatchEstimate
HeuristicAIMD::estimate(const std::vector<Candidate> &selected) const {
  BatchEstimate result{};
  if (selected.empty()) return result;

  std::uint64_t tokens = 0;
  const long double parameter_count = model_profile_.parameter_count;
  const long double layers = model_profile_.layer_count;
  const long double embedding = model_profile_.embedding_dimension;
  const long double kv_dimension =
      static_cast<long double>(model_profile_.kv_head_count) *
      model_profile_.head_dimension;
  const long double kv_scalar_bytes =
      model_profile_.key_effective_bytes_per_scalar +
      model_profile_.value_effective_bytes_per_scalar;

  result.memory_bytes = model_profile_.model_bytes;
  for (const Candidate &candidate : selected) {
    const std::uint32_t count = candidate.work.token_count();
    tokens += count;
    result.compute_operations += 2.0L * parameter_count * count;
    if (candidate.work.kind == WorkKind::Prefill) {
      result.compute_operations +=
          4.0L * layers * embedding * count * candidate.context_length;
      const long double kv_write =
          layers * kv_dimension * kv_scalar_bytes * count;
      result.memory_bytes += kv_write;
      result.additional_kv_bytes += kv_write;
    } else {
      result.compute_operations +=
          4.0L * layers * embedding * candidate.context_length;
      result.memory_bytes += layers * kv_dimension * kv_scalar_bytes *
                             candidate.context_length;
      result.additional_kv_bytes +=
          layers * kv_dimension * kv_scalar_bytes;
    }
  }

  result.compute_pressure =
      result.compute_operations / (2.0L * parameter_count);
  result.bandwidth_pressure =
      result.additional_kv_bytes /
      static_cast<long double>(hardware_profile_.total_memory_bytes);
  result.pressure = static_cast<double>(
      std::max(result.compute_pressure, result.bandwidth_pressure));
  result.valid = tokens <= token_budget_ &&
                 std::isfinite(result.compute_operations) &&
                 std::isfinite(result.memory_bytes) &&
                 std::isfinite(result.additional_kv_bytes) &&
                 result.additional_kv_bytes <=
                     static_cast<long double>(hardware_profile_.total_memory_bytes);
  return result;
}

void HeuristicAIMD::build_plan(Plan &out) {
  const std::vector<Candidate> available = candidates();
  std::vector<Candidate> decode;
  std::vector<Candidate> prefill;
  for (const Candidate &candidate : available) {
    (candidate.work.kind == WorkKind::Decode ? decode : prefill)
        .push_back(candidate);
  }
  const auto more_useful = [&](const Candidate &a, const Candidate &b) {
    return utility(a) > utility(b);
  };
  std::stable_sort(decode.begin(), decode.end(), more_useful);
  std::stable_sort(prefill.begin(), prefill.end(), [&](const Candidate &a,
                                                       const Candidate &b) {
    const bool a_starved = a.bypasses >= config_.starvation_threshold;
    const bool b_starved = b.bypasses >= config_.starvation_threshold;
    if (a_starved != b_starved) return a_starved;
    if (a_starved && a.bypasses != b.bypasses)
      return a.bypasses > b.bypasses;
    const bool a_started = a.work.token_begin != 0;
    const bool b_started = b.work.token_begin != 0;
    if (a_started != b_started) return a_started;
    return a.urgency > b.urgency;
  });

  std::vector<Candidate> selected;
  std::uint32_t remaining_tokens = token_budget_;
  std::uint32_t remaining_decode = window_capacity(decode_window_);
  std::uint32_t remaining_prefill = window_capacity(prefill_window_);
  long double selected_kv_bytes = 0.0L;

  for (const Candidate &candidate : decode) {
    if (remaining_tokens == 0 || remaining_decode == 0) break;
    const long double incremental_kv = additional_kv_bytes(candidate);
    if (!std::isfinite(incremental_kv) ||
        selected_kv_bytes + incremental_kv >
            static_cast<long double>(hardware_profile_.total_memory_bytes)) {
      continue;
    }
    selected_kv_bytes += incremental_kv;
    selected.push_back(candidate);
    --remaining_tokens;
    --remaining_decode;
  }

  for (Candidate candidate : prefill) {
    if (remaining_tokens == 0 || remaining_prefill == 0) break;
    const long double kv_per_token =
        additional_kv_bytes(candidate) / candidate.work.token_count();
    const long double remaining_memory =
        static_cast<long double>(hardware_profile_.total_memory_bytes) -
        selected_kv_bytes;
    if (!std::isfinite(kv_per_token) || kv_per_token <= 0.0L ||
        remaining_memory < kv_per_token) {
      continue;
    }
    const long double memory_capacity = remaining_memory / kv_per_token;
    constexpr auto maximum_count =
        std::numeric_limits<std::uint32_t>::max();
    const std::uint32_t memory_tokens =
        memory_capacity >= static_cast<long double>(maximum_count)
            ? maximum_count
            : static_cast<std::uint32_t>(memory_capacity);
    const std::uint32_t count =
        std::min({candidate.work.token_count(), remaining_tokens,
                  remaining_prefill, memory_tokens});
    candidate.work.token_end = candidate.work.token_begin + count;
    candidate.context_length = candidate.work.token_end;
    const long double incremental_kv = additional_kv_bytes(candidate);
    if (!std::isfinite(incremental_kv) ||
        selected_kv_bytes + incremental_kv >
            static_cast<long double>(hardware_profile_.total_memory_bytes)) {
      continue;
    }
    selected_kv_bytes += incremental_kv;
    selected.push_back(candidate);
    remaining_tokens -= count;
    remaining_prefill -= count;
  }

  if (selected.empty() && !available.empty()) {
    Candidate fallback = *std::max_element(
        available.begin(), available.end(), [&](const Candidate &a,
                                                 const Candidate &b) {
          return utility(a) < utility(b);
        });
    if (fallback.work.kind == WorkKind::Prefill) {
      fallback.work.token_end = fallback.work.token_begin + 1;
      fallback.context_length = fallback.work.token_end;
    }
    selected.push_back(fallback);
  }

  std::unordered_set<RequestId> selected_ids;
  for (const Candidate &candidate : selected) {
    selected_ids.insert(candidate.work.id);
  }
  std::unordered_set<RequestId> eligible_ids;
  for (const Candidate &candidate : available) {
    eligible_ids.insert(candidate.work.id);
    if (selected_ids.count(candidate.work.id) != 0) {
      bypasses_[candidate.work.id] = 0;
    } else {
      std::uint32_t &count = bypasses_[candidate.work.id];
      if (count != std::numeric_limits<std::uint32_t>::max()) ++count;
    }
  }
  for (auto it = bypasses_.begin(); it != bypasses_.end();) {
    if (eligible_ids.count(it->first) == 0) {
      it = bypasses_.erase(it);
    } else {
      ++it;
    }
  }

  last_estimate_ = estimate(selected);
  if (!last_estimate_.valid) return;
  for (const Candidate &candidate : selected) {
    out.work.push_back(candidate.work);
  }
}

namespace quickserve_benchmark_policy {
std::unique_ptr<Scheduler> create(Handoff &handoff, std::uint32_t token_budget,
                                  const ModelProfile &model_profile,
                                  const HardwareProfile &hardware_profile) {
  return std::make_unique<HeuristicAIMD>(handoff, token_budget, model_profile,
                                         hardware_profile);
}
} // namespace quickserve_benchmark_policy
