#include "continuous_batched_fifo.hpp"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <memory>

ContinuousBatchedFifoScheduler::ContinuousBatchedFifoScheduler(
    Handoff &handoff, std::uint32_t token_budget)
    : Scheduler(handoff, token_budget) {}

void ContinuousBatchedFifoScheduler::build_plan(Plan &out) {
  std::uint32_t remaining = token_budget_;

  for (const RequestState &request : policy_requests()) {
    if (remaining == 0) {
      return;
    }

    switch (request.stage) {
    case RequestState::Stage::PendingAdmission:
    case RequestState::Stage::Terminal:
      continue;

    case RequestState::Stage::PendingRelease:
      return;

    case RequestState::Stage::Prefill: {
      if (request.prefill_position >= request.prompt_length) {
        continue;
      }
      const std::uint32_t available =
          request.prompt_length - request.prefill_position;
      const std::uint32_t count = std::min(available, remaining);
      out.work.push_back({request.id, request.prefill_position,
                          request.prefill_position + count,
                          WorkKind::Prefill});
      remaining -= count;
      break;
    }

    case RequestState::Stage::Decode: {
      if (request.decoded_count == 0) {
        continue;
      }
      const std::uint64_t end =
          static_cast<std::uint64_t>(request.prompt_length) +
          request.decoded_count;
      if (end > std::numeric_limits<std::uint32_t>::max()) {
        return;
      }
      out.work.push_back({request.id, static_cast<std::uint32_t>(end - 1),
                          static_cast<std::uint32_t>(end), WorkKind::Decode});
      --remaining;
      break;
    }
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
  return std::make_unique<ContinuousBatchedFifoScheduler>(handoff,
                                                          token_budget);
}
} // namespace quickserve_benchmark_policy
