#pragma once

#include "runtime/hardware_profile.hpp"
#include "runtime/model_profile.hpp"
#include "runtime/scheduler.hpp"

#include <cstdint>
#include <list>
#include <vector>

namespace quickserve::policy {

struct BatchResourceUsage {
  std::uint64_t total_tokens{};
  std::uint64_t work_items{};
  long double resident_kv_bytes{};
  long double required_memory_bytes{};
  bool valid{true};
};

[[nodiscard]] BatchResourceUsage evaluate_batch_resources(
    const std::list<RequestState> &requests, const std::vector<WorkItem> &work,
    std::uint32_t token_budget, const ModelProfile &model,
    const HardwareProfile &hardware);

} // namespace quickserve::policy
