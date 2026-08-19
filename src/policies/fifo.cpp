#include "fifo.hpp"

#include <algorithm>
#include <cstdint>
#include <limits>

FifoScheduler::FifoScheduler(Handoff &handoff, std::uint32_t token_budget)
    : Scheduler(handoff, token_budget) {}

void FifoScheduler::build_plan(Plan &out) {
  const RequestState *active = nullptr;
  bool active_release_pending = false;

  for (const RequestState &request : policy_requests()) {
    const bool has_progress =
        request.prefill_position > 0 || request.decoded_count > 0;
    if ((request.stage == RequestState::Stage::Prefill ||
         request.stage == RequestState::Stage::Decode) &&
        has_progress && active == nullptr) {
      active = &request;
    }
    if (request.stage == RequestState::Stage::PendingRelease &&
        has_progress) {
      active_release_pending = true;
    }
  }

  if (active_release_pending) {
    return;
  }
  const RequestState *selected = active;
  if (selected == nullptr) {
    for (const RequestState &request : policy_requests()) {
      if (request.stage == RequestState::Stage::PendingAdmission ||
          request.stage == RequestState::Stage::Terminal) {
        continue;
      }
      if (request.stage == RequestState::Stage::PendingRelease) {
        return;
      }
      if (request.stage == RequestState::Stage::Prefill ||
          request.stage == RequestState::Stage::Decode) {
        selected = &request;
        break;
      }
    }
  }
  if (selected == nullptr) {
    return;
  }

  if (selected->stage == RequestState::Stage::Prefill) {
    const std::uint64_t end = std::min<std::uint64_t>(
        selected->prompt_length,
        static_cast<std::uint64_t>(selected->prefill_position) +
            token_budget_);
    if (selected->prefill_position < end) {
      out.work.push_back(
          {selected->id, selected->prefill_position,
           static_cast<std::uint32_t>(end), WorkKind::Prefill});
    }
    return;
  }

  if (selected->stage == RequestState::Stage::Decode) {
    if (selected->decoded_count == 0) {
      return;
    }
    const std::uint64_t end =
        static_cast<std::uint64_t>(selected->prompt_length) +
        selected->decoded_count;
    if (end > std::numeric_limits<std::uint32_t>::max()) {
      return;
    }
    out.work.push_back({selected->id, static_cast<std::uint32_t>(end - 1),
                        static_cast<std::uint32_t>(end), WorkKind::Decode});
  }
}
