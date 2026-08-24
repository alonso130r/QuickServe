#pragma once

#include "runtime/scheduler.hpp"

#include <cstdint>
#include <memory>

namespace quickserve_benchmark_policy {
std::unique_ptr<Scheduler> create(Handoff &handoff, std::uint32_t token_budget);
}
