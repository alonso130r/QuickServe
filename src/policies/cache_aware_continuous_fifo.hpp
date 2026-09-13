#pragma once

#include "runtime/scheduler.hpp"
#include "runtime/hardware_profile.hpp"
#include "runtime/model_profile.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>

class CacheAwareContinuousFifoScheduler final : public Scheduler {
public:
  CacheAwareContinuousFifoScheduler(Handoff &handoff,
                                    std::uint32_t token_budget,
                                    std::size_t locality_window = 8);

protected:
  void build_plan(Plan &out) override;

private:
  std::size_t locality_window_;
};

namespace quickserve_benchmark_policy {
std::unique_ptr<Scheduler> create(Handoff &handoff,
                                  std::uint32_t token_budget,
                                  const ModelProfile &model_profile,
                                  const HardwareProfile &hardware_profile);
}
