#include "proxqp_solver.hpp"

#include <chrono>
#include <cmath>
#include <proxsuite/proxqp/dense/dense.hpp>

namespace quickserve::optimization {

double proxqp_objective(const ProxQPInput &i, double prefill, double decode) {
  const double runtime = i.runtime_base_ns + i.runtime_margin_ns +
      i.runtime_prefill_ns * prefill + i.runtime_decode_ns * decode;
  const auto slack = [&](double headroom) {
    return std::max(0.0, (runtime - headroom) / i.runtime_scale_ns +
                             i.boundary_buffer_z);
  };
  const double ttft_slack = slack(i.ttft_headroom_ns);
  const double tpot_slack = slack(i.tpot_headroom_ns);
  const double gp = -i.reward_prefill / i.token_budget +
      i.runtime_weight * i.runtime_prefill_ns / i.batch_duration_scale_ns -
      i.rho_prefill * i.previous_prefill /
          (i.token_budget * i.token_budget);
  const double gd = -i.reward_decode / i.sequence_capacity +
      i.runtime_weight * i.runtime_decode_ns / i.batch_duration_scale_ns -
      i.rho_decode * i.previous_decode /
          (i.sequence_capacity * i.sequence_capacity);
  return 0.5 * i.rho_prefill * std::pow(prefill / i.token_budget, 2) +
      0.5 * i.rho_decode * std::pow(decode / i.sequence_capacity, 2) +
      gp * prefill + gd * decode +
      0.5 * i.boundary_weight *
          (ttft_slack * ttft_slack + tpot_slack * tpot_slack);
}

ProxQPResult solve_proxqp(const ProxQPInput &i) {
  using Matrix = proxsuite::proxqp::dense::Mat<double>;
  using Vector = proxsuite::proxqp::dense::Vec<double>;
  Matrix h = Matrix::Zero(4, 4), c = Matrix::Zero(11, 4);
  Vector g = Vector::Zero(4), lower = Vector::Zero(11),
         upper = Vector::Zero(11);
  h(0, 0) = i.rho_prefill / (i.token_budget * i.token_budget);
  h(1, 1) = i.rho_decode /
            (i.sequence_capacity * i.sequence_capacity);
  h(2, 2) = i.boundary_weight;
  h(3, 3) = i.boundary_weight;
  g(0) = -i.reward_prefill / i.token_budget +
         i.runtime_weight * i.runtime_prefill_ns /
             i.batch_duration_scale_ns -
         i.rho_prefill * i.previous_prefill /
             (i.token_budget * i.token_budget);
  g(1) = -i.reward_decode / i.sequence_capacity +
         i.runtime_weight * i.runtime_decode_ns /
             i.batch_duration_scale_ns -
         i.rho_decode * i.previous_decode /
             (i.sequence_capacity * i.sequence_capacity);
  c << -1, 0, 0, 0, 0, -1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0,
      1 / i.chunk_size, 1, 0, 0, i.memory_prefill, i.memory_decode, 0, 0,
      i.runtime_prefill_ns / i.runtime_scale_ns,
      i.runtime_decode_ns / i.runtime_scale_ns, -1, 0,
      i.runtime_prefill_ns / i.runtime_scale_ns,
      i.runtime_decode_ns / i.runtime_scale_ns, 0, -1, 0, 0, -1, 0, 0, 0, 0,
      -1;
  const double maximum_runtime = i.runtime_base_ns + i.runtime_margin_ns +
      i.runtime_prefill_ns * i.token_budget +
      i.runtime_decode_ns * i.sequence_capacity;
  const double slack_limit = std::max(
      1.0, (maximum_runtime -
                std::min(i.ttft_headroom_ns, i.tpot_headroom_ns)) /
                    i.runtime_scale_ns +
                i.boundary_buffer_z + 1.0);
  lower << -i.prefill_available, -i.decode_available, 0, 0, 0, 0, 0,
      -slack_limit, -slack_limit, -slack_limit, -slack_limit;
  const double safe_base = i.runtime_base_ns + i.runtime_margin_ns;
  upper << 0, 0, i.prefill_available, i.decode_available, i.token_budget,
      i.sequence_capacity, i.memory_available - i.memory_base,
      (i.ttft_headroom_ns - safe_base) / i.runtime_scale_ns -
          i.boundary_buffer_z,
      (i.tpot_headroom_ns - safe_base) / i.runtime_scale_ns -
          i.boundary_buffer_z,
      0, 0;
  proxsuite::proxqp::dense::QP<double> qp(4, 0, 11);
  qp.settings.eps_abs = 1e-9;
  qp.settings.eps_rel = 1e-9;
  qp.settings.verbose = false;
  qp.init(h, g, proxsuite::nullopt, proxsuite::nullopt, c, lower, upper);
  const auto start = std::chrono::steady_clock::now();
  qp.solve();
  const auto finish = std::chrono::steady_clock::now();
  const auto &info = qp.results.info;
  ProxQPResult result;
  result.success = info.status == proxsuite::proxqp::QPSolverOutput::PROXQP_SOLVED;
  result.prefill = qp.results.x(0);
  result.decode = qp.results.x(1);
  result.ttft_slack = qp.results.x(2);
  result.tpot_slack = qp.results.x(3);
  result.objective = info.objValue;
  result.solve_ns = std::chrono::duration<double, std::nano>(finish - start).count();
  result.primal_residual = info.pri_res;
  result.dual_residual = info.dua_res;
  result.iterations = info.iter;
  result.status = result.success ? "solved" : "failed";
  return result;
}

} // namespace quickserve::optimization
