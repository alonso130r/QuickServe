#include "optimization/proxqp_solver.hpp"

#include <cmath>
#include <cstdio>

int main() {
  quickserve::optimization::ProxQPInput input;
  input.token_budget = 10;
  input.sequence_capacity = 10;
  input.prefill_available = 10;
  input.decode_available = 10;
  input.chunk_size = 10;
  input.memory_available = 100;
  input.ttft_headroom_ns = 100;
  input.tpot_headroom_ns = 100;
  input.runtime_scale_ns = 1;
  input.reward_prefill = 0.4;
  input.reward_decode = 0.2;
  input.rho_prefill = 1;
  input.rho_decode = 1;
  const auto result = quickserve::optimization::solve_proxqp(input);
  if (!result.success || std::abs(result.prefill - 4.0) > 1e-5 ||
      std::abs(result.decode - 2.0) > 1e-5) {
    std::printf("unexpected ProxQP result: success=%d p=%g d=%g\n",
                result.success, result.prefill, result.decode);
    return 1;
  }

  input.runtime_base_ns = 300;
  input.runtime_scale_ns = 100;
  input.ttft_headroom_ns = 100;
  input.tpot_headroom_ns = 100;
  const auto elastic = quickserve::optimization::solve_proxqp(input);
  if (!elastic.success || elastic.ttft_slack <= 0 ||
      elastic.tpot_slack <= 0) {
    std::printf("impossible headroom was not relaxed: success=%d st=%g sp=%g\n",
                elastic.success, elastic.ttft_slack, elastic.tpot_slack);
    return 1;
  }
  std::puts("all production ProxQP adapter checks passed");
  return 0;
}
