#include "solvers.hpp"
#include <Eigen/Core>
#include <Eigen/SparseCore>
#include <chrono>
#include <clarabel.hpp>
#include <limits>
#include <proxsuite/proxqp/dense/dense.hpp>
#include <vector>

namespace sloexp {
namespace {
using Clock = std::chrono::steady_clock;
double ns(Clock::time_point a, Clock::time_point b) {
  return std::chrono::duration<double, std::nano>(b - a).count();
}
double mean(const std::vector<double> &v) {
  double s = 0;
  for (double x : v)
    s += x;
  return v.empty() ? 0 : s / v.size();
}

struct QpData {
  proxsuite::proxqp::dense::Mat<double> H, C;
  proxsuite::proxqp::dense::Vec<double> g, l, u;
};
QpData qp_data(const ProblemInstance &i) {
  QpData q{proxsuite::proxqp::dense::Mat<double>::Zero(4, 4),
           proxsuite::proxqp::dense::Mat<double>::Zero(13, 4),
           proxsuite::proxqp::dense::Vec<double>::Zero(4),
           proxsuite::proxqp::dense::Vec<double>::Zero(13),
           proxsuite::proxqp::dense::Vec<double>::Zero(13)};
  q.H(0, 0) = i.rho_prefill / (i.token_budget * i.token_budget);
  q.H(1, 1) = i.rho_decode / (i.sequence_capacity * i.sequence_capacity);
  q.H(2, 2) = i.risk_weight;
  q.H(3, 3) = i.risk_weight;
  q.g(0) =
      -i.reward_prefill / i.token_budget +
      i.runtime_weight * i.runtime_prefill / i.batch_duration_scale -
      i.rho_prefill * i.previous_prefill / (i.token_budget * i.token_budget);
  q.g(1) = -i.reward_decode / i.sequence_capacity +
           i.runtime_weight * i.runtime_decode / i.batch_duration_scale -
           i.rho_decode * i.previous_decode /
               (i.sequence_capacity * i.sequence_capacity);
  q.C << -1, 0, 0, 0, 0, -1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0,
      1 / i.chunk_size, 1, 0, 0, i.memory_prefill, i.memory_decode, 0, 0,
      i.runtime_prefill / i.ttft_scale, i.runtime_decode / i.ttft_scale, 0, 0,
      i.runtime_prefill / i.tpot_scale, i.runtime_decode / i.tpot_scale, 0, 0,
      i.runtime_prefill / i.ttft_scale, i.runtime_decode / i.ttft_scale, -1, 0,
      i.runtime_prefill / i.tpot_scale, i.runtime_decode / i.tpot_scale, 0, -1,
      0, 0, -1, 0, 0, 0, 0, -1;
  q.l << -i.prefill_available, -i.decode_available, 0, 0, 0, 0, 0, 0, 0,
      -i.risk_buffer_z, -i.risk_buffer_z, -i.risk_buffer_z,
      -i.risk_buffer_z;
  q.u << 0, 0, i.prefill_available, i.decode_available, i.token_budget,
      i.sequence_capacity, i.memory_available - i.memory_base,
      i.ttft_headroom / i.ttft_scale - i.runtime_base / i.ttft_scale -
          i.fixed_margin_z,
      i.tpot_headroom / i.tpot_scale - i.runtime_base / i.tpot_scale -
          i.fixed_margin_z,
      i.ttft_headroom / i.ttft_scale - i.runtime_base / i.ttft_scale -
          i.fixed_margin_z - i.risk_buffer_z,
      i.tpot_headroom / i.tpot_scale - i.runtime_base / i.tpot_scale -
          i.fixed_margin_z - i.risk_buffer_z,
      0, 0;
  return q;
}

clarabel::DefaultSettings<double> clarabel_settings() {
  auto s = clarabel::DefaultSettings<double>::default_settings();
  s.verbose = false;
  s.max_iter = 100;
  s.tol_gap_abs = 1e-9;
  s.tol_gap_rel = 1e-9;
  s.tol_feas = 1e-9;
  return s;
}
std::string clarabel_status(clarabel::SolverStatus s) {
  switch (s) {
  case clarabel::SolverStatus::Solved:
    return "solved";
  case clarabel::SolverStatus::AlmostSolved:
    return "almost_solved";
  case clarabel::SolverStatus::PrimalInfeasible:
    return "primal_infeasible";
  case clarabel::SolverStatus::DualInfeasible:
    return "dual_infeasible";
  case clarabel::SolverStatus::MaxIterations:
    return "max_iterations";
  case clarabel::SolverStatus::MaxTime:
    return "max_time";
  case clarabel::SolverStatus::NumericalError:
    return "numerical_error";
  case clarabel::SolverStatus::InsufficientProgress:
    return "insufficient_progress";
  default:
    return "other";
  }
}
enum class UncertaintyModel { none, diagonal, covariance };

SolveResult clarabel_run(const ProblemInstance &i, UncertaintyModel model,
                         std::size_t repetitions) {
  const auto built0 = Clock::now();
  const bool robust = model != UncertaintyModel::none;
  const double standard_scale = std::sqrt(i.ttft_scale * i.tpot_scale);
  const int n = robust ? 6 : 4; // robust: p,d,t_z,r,u_T,u_D
  Eigen::MatrixXd Pd = Eigen::MatrixXd::Zero(n, n);
  Pd(0, 0) = i.rho_prefill / (i.token_budget * i.token_budget);
  Pd(1, 1) = i.rho_decode / (i.sequence_capacity * i.sequence_capacity);
  Pd(robust ? 4 : 2, robust ? 4 : 2) = i.risk_weight;
  Pd(robust ? 5 : 3, robust ? 5 : 3) = i.risk_weight;
  Eigen::VectorXd q = Eigen::VectorXd::Zero(n);
  q(0) = -i.reward_prefill / i.token_budget -
         i.rho_prefill * i.previous_prefill / (i.token_budget * i.token_budget);
  q(1) = -i.reward_decode / i.sequence_capacity -
         i.rho_decode * i.previous_decode /
             (i.sequence_capacity * i.sequence_capacity);
  if (robust)
    q(2) = i.runtime_weight * standard_scale / i.batch_duration_scale;
  else {
    q(0) += i.runtime_weight * i.runtime_prefill / i.batch_duration_scale;
    q(1) += i.runtime_weight * i.runtime_decode / i.batch_duration_scale;
  }
  const int linear_rows = robust ? 15 : 13;
  const int soc_rows = model == UncertaintyModel::diagonal
                           ? 4
                           : model == UncertaintyModel::covariance ? 6 : 0;
  Eigen::MatrixXd Ad = Eigen::MatrixXd::Zero(linear_rows + soc_rows, n);
  Eigen::VectorXd b = Eigen::VectorXd::Zero(linear_rows + soc_rows);
  Ad.block(0, 0, 7, 2) << -1, 0, 0, -1, 1, 0, 0, 1, 1, 1, 1 / i.chunk_size, 1,
      i.memory_prefill, i.memory_decode;
  b.head(7) << 0, 0, i.prefill_available, i.decode_available, i.token_budget,
      i.sequence_capacity, i.memory_available - i.memory_base;
  if (!robust) {
    Ad(7, 0) = i.runtime_prefill / i.ttft_scale;
    Ad(7, 1) = i.runtime_decode / i.ttft_scale;
    Ad(8, 0) = i.runtime_prefill / i.tpot_scale;
    Ad(8, 1) = i.runtime_decode / i.tpot_scale;
    b(7) = i.ttft_headroom / i.ttft_scale - i.runtime_base / i.ttft_scale -
           i.fixed_margin_z;
    b(8) = i.tpot_headroom / i.tpot_scale - i.runtime_base / i.tpot_scale -
           i.fixed_margin_z;
    Ad(9, 0) = i.runtime_prefill / i.ttft_scale;
    Ad(9, 1) = i.runtime_decode / i.ttft_scale;
    Ad(9, 2) = -1;
    b(9) = b(7) - i.risk_buffer_z;
    Ad(10, 0) = i.runtime_prefill / i.tpot_scale;
    Ad(10, 1) = i.runtime_decode / i.tpot_scale;
    Ad(10, 3) = -1;
    b(10) = b(8) - i.risk_buffer_z;
    Ad(11, 2) = -1;
    Ad(12, 3) = -1;
  }
  if (robust) {
    const double scale = standard_scale;
    Ad(7, 2) = scale / i.ttft_scale;
    b(7) = i.ttft_headroom / i.ttft_scale;
    Ad(8, 2) = scale / i.tpot_scale;
    b(8) = i.tpot_headroom / i.tpot_scale;
    Ad(9, 0) = i.runtime_prefill / scale;
    Ad(9, 1) = i.runtime_decode / scale;
    Ad(9, 2) = -1;
    Ad(9, 3) = model == UncertaintyModel::covariance
                   ? i.covariance_kappa
                   : i.uncertainty_kappa;
    b(9) = -i.runtime_base / scale;
    Ad(10, 3) = -1;
    b(10) = 0;
    Ad(11, 2) = scale / i.ttft_scale;
    Ad(11, 4) = -1;
    b(11) = i.ttft_headroom / i.ttft_scale - i.risk_buffer_z;
    Ad(12, 2) = scale / i.tpot_scale;
    Ad(12, 5) = -1;
    b(12) = i.tpot_headroom / i.tpot_scale - i.risk_buffer_z;
    Ad(13, 4) = -1;
    Ad(14, 5) = -1;
    // The first cone coordinate is r. Remaining coordinates form the
    // allocation-dependent uncertainty norm.
    Ad(15, 3) = -1;
    if (model == UncertaintyModel::diagonal) {
      b(16) = i.uncertainty_base / scale;
      Ad(17, 0) = -i.uncertainty_prefill / scale;
      Ad(18, 1) = -i.uncertainty_decode / scale;
    } else {
      for (int k = 0; k < 4; ++k) {
        b(16 + k) = i.covariance_factor_constant[k] / scale;
        Ad(16 + k, 0) = -i.covariance_factor_prefill[k] / scale;
        Ad(16 + k, 1) = -i.covariance_factor_decode[k] / scale;
      }
      b(20) = i.covariance_residual / scale;
    }
  }
  Eigen::SparseMatrix<double> P = Pd.sparseView(), A = Ad.sparseView();
  P.makeCompressed();
  A.makeCompressed();
  std::vector<clarabel::SupportedConeT<double>> cones;
  cones.emplace_back(clarabel::NonnegativeConeT<double>(linear_rows));
  if (robust)
    cones.emplace_back(clarabel::SecondOrderConeT<double>(soc_rows));
  const auto built1 = Clock::now();
  SolveResult result;
  for (std::size_t k = 0; k < std::max<std::size_t>(1, repetitions); ++k) {
    const auto c0 = Clock::now();
    clarabel::DefaultSolver<double> solver(P, q, A, b, cones,
                                           clarabel_settings());
    const auto c1 = Clock::now();
    const auto a = Clock::now();
    solver.solve();
    const auto z = Clock::now();
    result.cold_setup_samples.push_back(ns(c0, c1));
    result.cold_solve_samples.push_back(ns(a, z));
    auto sol = solver.solution();
    auto info = solver.info();
    result.success = sol.status == clarabel::SolverStatus::Solved ||
                     sol.status == clarabel::SolverStatus::AlmostSolved;
    result.status = clarabel_status(sol.status);
    result.p = sol.x(0);
    result.d = sol.x(1);
    result.objective = sol.obj_val;
    result.iterations = sol.iterations;
    result.primal_residual = info.res_primal;
    result.dual_residual = info.res_dual;
  }
  clarabel::DefaultSolver<double> warm(P, q, A, b, cones, clarabel_settings());
  warm.solve();
  for (std::size_t k = 0; k < std::max<std::size_t>(1, repetitions); ++k) {
    const auto u0 = Clock::now();
    warm.update_q(q);
    const auto u1 = Clock::now();
    warm.solve();
    const auto u2 = Clock::now();
    result.warm_update_samples.push_back(ns(u0, u1));
    result.warm_solve_samples.push_back(ns(u1, u2));
  }
  result.setup_ns = mean(result.cold_setup_samples) + ns(built0, built1);
  result.solve_ns = mean(result.cold_solve_samples);
  result.warm_update_ns = mean(result.warm_update_samples);
  result.warm_solve_ns = mean(result.warm_solve_samples);
  result.total_ns = result.setup_ns + result.solve_ns;
  return result;
}
} // namespace

SolveResult solve_proxqp(const ProblemInstance &i, std::size_t repetitions) {
  const auto data = qp_data(i);
  proxsuite::proxqp::dense::Mat<double> Aeq(0, 4);
  proxsuite::proxqp::dense::Vec<double> beq(0);
  SolveResult r;
  auto configure = [&](auto &qp) {
    qp.settings.eps_abs = 1e-9;
    qp.settings.eps_rel = 1e-9;
    qp.settings.verbose = false;
  };
  for (std::size_t k = 0; k < std::max<std::size_t>(1, repetitions); ++k) {
    const auto c0 = Clock::now();
    proxsuite::proxqp::dense::QP<double> qp(
        4, 0, 13, false, proxsuite::proxqp::HessianType::Diagonal,
        proxsuite::proxqp::DenseBackend::PrimalDualLDLT);
    configure(qp);
    qp.init(data.H, data.g, Aeq, beq, data.C, data.l, data.u);
    const auto c1 = Clock::now();
    const auto x = Clock::now();
    qp.solve();
    const auto z = Clock::now();
    r.cold_setup_samples.push_back(ns(c0, c1));
    r.cold_solve_samples.push_back(ns(x, z));
    r.success = qp.results.info.status ==
                proxsuite::proxqp::QPSolverOutput::PROXQP_SOLVED;
    r.status = r.success
                   ? "solved"
                   : ("failed-" +
                      std::to_string(static_cast<int>(qp.results.info.status)));
    r.p = qp.results.x(0);
    r.d = qp.results.x(1);
    r.objective = qp.results.info.objValue;
    r.iterations = qp.results.info.iter;
    r.primal_residual = qp.results.info.pri_res;
    r.dual_residual = qp.results.info.dua_res;
  }
  proxsuite::proxqp::dense::QP<double> warm(
      4, 0, 13, false, proxsuite::proxqp::HessianType::Diagonal,
      proxsuite::proxqp::DenseBackend::PrimalDualLDLT);
  configure(warm);
  warm.init(data.H, data.g, Aeq, beq, data.C, data.l, data.u);
  warm.solve();
  warm.settings.initial_guess =
      proxsuite::proxqp::InitialGuessStatus::WARM_START_WITH_PREVIOUS_RESULT;
  for (std::size_t k = 0; k < std::max<std::size_t>(1, repetitions); ++k) {
    const auto u0 = Clock::now();
    warm.update(proxsuite::nullopt, data.g, proxsuite::nullopt,
                proxsuite::nullopt, proxsuite::nullopt, proxsuite::nullopt,
                proxsuite::nullopt);
    const auto u1 = Clock::now();
    warm.solve();
    const auto u2 = Clock::now();
    r.warm_update_samples.push_back(ns(u0, u1));
    r.warm_solve_samples.push_back(ns(u1, u2));
  }
  r.setup_ns = mean(r.cold_setup_samples);
  r.solve_ns = mean(r.cold_solve_samples);
  r.warm_update_ns = mean(r.warm_update_samples);
  r.warm_solve_ns = mean(r.warm_solve_samples);
  r.total_ns = r.setup_ns + r.solve_ns;
  return r;
}
SolveResult solve_clarabel_qp(const ProblemInstance &i, std::size_t r) {
  return clarabel_run(i, UncertaintyModel::none, r);
}
SolveResult solve_clarabel_socp(const ProblemInstance &i, std::size_t r) {
  return clarabel_run(i, UncertaintyModel::diagonal, r);
}
SolveResult solve_clarabel_covariance_socp(const ProblemInstance &i,
                                           std::size_t r) {
  return clarabel_run(i, UncertaintyModel::covariance, r);
}
} // namespace sloexp
