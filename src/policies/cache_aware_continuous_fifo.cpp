#include "cache_aware_continuous_fifo.hpp"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <vector>

CacheAwareContinuousFifoScheduler::CacheAwareContinuousFifoScheduler(
    Handoff &handoff, std::uint32_t token_budget, std::size_t locality_window)
    : Scheduler(handoff, token_budget), locality_window_(locality_window) {
  if (locality_window_ == 0)
    throw std::invalid_argument("locality window must be positive");
}

void CacheAwareContinuousFifoScheduler::build_plan(Plan &out) {
  std::vector<const RequestState *> ordered;
  std::vector<const RequestState *> window;
  for (const RequestState &request : policy_requests()) {
    if (request.stage == RequestState::Stage::PendingRelease)
      break;
    if (request.stage != RequestState::Stage::Prefill &&
        request.stage != RequestState::Stage::Decode)
      continue;
    ordered.push_back(&request);
    if (request.stage == RequestState::Stage::Prefill &&
        window.size() < locality_window_)
      window.push_back(&request);
  }
  std::stable_sort(window.begin(), window.end(), [](const auto *left,
                                                     const auto *right) {
    return left->cached_prefix_tokens > right->cached_prefix_tokens;
  });

  std::uint32_t remaining = token_budget_;
  std::size_t prefill_index = 0;
  for (const RequestState *original : ordered) {
    if (remaining == 0)
      return;
    const RequestState *request = original;
    if (original->stage == RequestState::Stage::Prefill &&
        prefill_index < window.size())
      request = window[prefill_index++];

    if (request->stage == RequestState::Stage::Prefill) {
      if (request->prefill_position >= request->prompt_length)
        continue;
      const std::uint32_t count = std::min(
          request->prompt_length - request->prefill_position, remaining);
      out.work.push_back({request->id, request->prefill_position,
                          request->prefill_position + count,
                          WorkKind::Prefill});
      remaining -= count;
    } else {
      if (request->decoded_count == 0)
        continue;
      const std::uint64_t end =
          static_cast<std::uint64_t>(request->prompt_length) +
          request->decoded_count;
      if (end > std::numeric_limits<std::uint32_t>::max())
        return;
      out.work.push_back({request->id, static_cast<std::uint32_t>(end - 1),
                          static_cast<std::uint32_t>(end), WorkKind::Decode});
      --remaining;
    }
  }
}

namespace quickserve_benchmark_policy {
std::unique_ptr<Scheduler> create(Handoff &handoff,
                                  std::uint32_t token_budget,
                                  const ModelProfile &model_profile,
                                  const HardwareProfile &hardware_profile) {
  (void)model_profile;
  (void)hardware_profile;
  return std::make_unique<CacheAwareContinuousFifoScheduler>(handoff,
                                                              token_budget);
}
} // namespace quickserve_benchmark_policy
