#include "heuristic_aimd.hpp"
#include "batch_utilities.hpp"

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
      config.max_consecutive_decode_batches == 0 ||
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

BatchFeasibility HeuristicAIMD::check_feasibility(
    const std::vector<Candidate> &selected) const {
  std::vector<WorkItem> work;
  work.reserve(selected.size());
  for (const Candidate &candidate : selected)
    work.push_back(candidate.work);
  const auto usage = quickserve::policy::evaluate_batch_resources(
      policy_requests(), work, token_budget_, model_profile_, hardware_profile_);
  return {usage.total_tokens, usage.work_items, usage.resident_kv_bytes,
          usage.required_memory_bytes, usage.valid};
}

void HeuristicAIMD::build_plan(Plan &out) {
  const std::vector<Candidate> available = candidates();
  std::vector<Candidate> decode;
  std::vector<Candidate> prefill;
  for (const Candidate &candidate : available) {
    (candidate.work.kind == WorkKind::Decode ? decode : prefill)
        .push_back(candidate);
  }
  std::stable_sort(decode.begin(), decode.end(), [](const Candidate &a,
                                                    const Candidate &b) {
    if (a.urgency != b.urgency) return a.urgency > b.urgency;
    return a.bypasses > b.bypasses;
  });
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
  const bool prefill_starved =
      std::any_of(prefill.begin(), prefill.end(), [&](const Candidate &item) {
        return item.bypasses >= config_.starvation_threshold;
      });
  const bool force_prefill =
      !prefill.empty() &&
      (prefill_starved || consecutive_decode_batches_ >=
                              config_.max_consecutive_decode_batches);
  const bool decode_mode = !decode.empty() && !force_prefill;

  if (decode_mode) {
    for (const Candidate &candidate : decode) {
      if (remaining_tokens == 0 || remaining_decode == 0) break;
      std::vector<Candidate> proposed = selected;
      proposed.push_back(candidate);
      if (!check_feasibility(proposed).valid) continue;
      selected.push_back(candidate);
      --remaining_tokens;
      --remaining_decode;
    }
  } else {
    for (Candidate candidate : prefill) {
      if (remaining_tokens == 0 || remaining_prefill == 0) break;
      std::uint32_t count = std::min(
          {candidate.work.token_count(), remaining_tokens, remaining_prefill});
      while (count > 0) {
        candidate.work.token_end = candidate.work.token_begin + count;
        candidate.context_length = candidate.work.token_end;
        std::vector<Candidate> proposed = selected;
        proposed.push_back(candidate);
        if (check_feasibility(proposed).valid) break;
        --count;
      }
      if (count == 0) continue;
      selected.push_back(candidate);
      remaining_tokens -= count;
      remaining_prefill -= count;
    }
  }

  if (selected.empty() && !available.empty()) {
    const std::vector<Candidate> &mode_candidates =
        decode_mode ? decode : prefill;
    Candidate fallback = *std::max_element(
        mode_candidates.begin(), mode_candidates.end(), [&](const Candidate &a,
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

  last_feasibility_ = check_feasibility(selected);
  if (!last_feasibility_.valid) return;
  for (const Candidate &candidate : selected) {
    out.work.push_back(candidate.work);
  }
  if (decode_mode) {
    if (consecutive_decode_batches_ !=
        std::numeric_limits<std::uint32_t>::max()) {
      ++consecutive_decode_batches_;
    }
  } else {
    consecutive_decode_batches_ = 0;
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
