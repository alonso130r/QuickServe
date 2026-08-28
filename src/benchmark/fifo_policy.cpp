#include "benchmark/fifo_policy.hpp"

#include "policies/fifo.hpp"

#include <memory>

namespace quickserve_benchmark_policy {
std::unique_ptr<Scheduler> create(Handoff &handoff, std::uint32_t token_budget,
                                  const ModelProfile &model_profile,
                                  const HardwareProfile &hardware_profile) {
  (void)model_profile;
  (void)hardware_profile;
  return std::make_unique<FifoScheduler>(handoff, token_budget);
}
}
