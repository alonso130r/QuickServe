#include "benchmark/fifo_policy.hpp"

#include "policies/fifo.hpp"

#include <memory>

namespace quickserve_benchmark_policy {
std::unique_ptr<Scheduler> create(Handoff &handoff, std::uint32_t token_budget) {
  return std::make_unique<FifoScheduler>(handoff, token_budget);
}
}
