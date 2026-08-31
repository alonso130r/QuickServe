#pragma once
#include "experiment.hpp"
#include <cstdint>
#include <string>
#include <vector>

namespace sloexp {
struct SolveResult {
  bool success{};
  double p{}, d{}, objective{};
  double setup_ns{}, solve_ns{}, total_ns{};
  double warm_update_ns{}, warm_solve_ns{};
  std::vector<double> cold_setup_samples, cold_solve_samples;
  std::vector<double> warm_update_samples, warm_solve_samples;
  double primal_residual{}, dual_residual{};
  std::uint64_t iterations{};
  std::string status;
};
SolveResult solve_proxqp(const ProblemInstance &, std::size_t repetitions);
SolveResult solve_clarabel_qp(const ProblemInstance &, std::size_t repetitions);
SolveResult solve_clarabel_socp(const ProblemInstance &, std::size_t repetitions);
SolveResult solve_clarabel_covariance_socp(const ProblemInstance &,
                                           std::size_t repetitions);
} // namespace sloexp
