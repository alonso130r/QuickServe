#pragma once

#include "runtime/scheduler.hpp"
#include "runtime/model_profile.hpp"
#include "runtime/hardware_profile.hpp"

#include <cstdint>
#include <memory>

namespace quickserve_benchmark_policy {
std::unique_ptr<Scheduler> create(Handoff &handoff, std::uint32_t token_budget,
                                  const ModelProfile &model_profile,
                                  const HardwareProfile &hardware_profile);
}
