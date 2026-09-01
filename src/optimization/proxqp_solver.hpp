#pragma once

#include <cstdint>
#include <string>

namespace quickserve::optimization {

struct ProxQPInput {
  double token_budget{512};
  double sequence_capacity{64};
  double prefill_available{512};
  double decode_available{64};
  double chunk_size{128};
  double memory_available{1};
  double memory_base{};
  double memory_prefill{};
  double memory_decode{};
  double runtime_base_ns{};
  double runtime_prefill_ns{};
  double runtime_decode_ns{};
  double runtime_margin_ns{};
  double runtime_scale_ns{1};
  double ttft_headroom_ns{1};
  double tpot_headroom_ns{1};
  double batch_duration_scale_ns{1};
  double reward_prefill{1};
  double reward_decode{1};
  double runtime_weight{0.1};
  double rho_prefill{0.2};
  double rho_decode{0.2};
  double previous_prefill{};
  double previous_decode{};
  double boundary_buffer_z{1};
  double boundary_weight{0.25};
};

struct ProxQPResult {
  bool success{};
  double prefill{};
  double decode{};
  double ttft_slack{};
  double tpot_slack{};
  double objective{};
  double solve_ns{};
  double primal_residual{};
  double dual_residual{};
  std::uint64_t iterations{};
  std::string status;
};

[[nodiscard]] double proxqp_objective(const ProxQPInput &input,
                                      double prefill, double decode);
[[nodiscard]] ProxQPResult solve_proxqp(const ProxQPInput &input);

} // namespace quickserve::optimization
