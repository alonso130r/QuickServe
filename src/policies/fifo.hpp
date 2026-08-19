#pragma once

#include <cstdint>

#include "runtime/scheduler.hpp"

class FifoScheduler final : public Scheduler {
public:
  FifoScheduler(Handoff &handoff, std::uint32_t token_budget);

protected:
  void build_plan(Plan &out) override;
};
