#pragma once

#include "runtime/scheduler.hpp"

#include <cstdint>
#include <memory>

class ContinuousBatchedFifoScheduler final : public Scheduler {
public:
  ContinuousBatchedFifoScheduler(Handoff &handoff,
                                 std::uint32_t token_budget);

protected:
  void build_plan(Plan &out) override;
};

namespace quickserve_benchmark_policy {
std::unique_ptr<Scheduler> create(Handoff &handoff,
                                  std::uint32_t token_budget);
}
