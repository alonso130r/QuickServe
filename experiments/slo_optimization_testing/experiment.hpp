#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace sloexp {

struct ProblemInstance {
  std::uint64_t id{};
  double token_budget{512};
  double sequence_capacity{64};
  double prefill_available{512};
  double decode_available{64};
  double chunk_size{128};
  double memory_available{1};
  double memory_base{};
  double memory_prefill{0.0005};
  double memory_decode{0.002};
  double runtime_base{0.05};
  double runtime_prefill{0.001};
  double runtime_decode{0.01};
  double ttft_headroom{1};
  double tpot_headroom{1};
  double ttft_scale{0.1};
  double tpot_scale{0.1};
  double batch_duration_scale{1};
  double fixed_margin_z{0.2};
  double risk_buffer_z{1};
  double risk_weight{0.25};
  double uncertainty_kappa{2};
  double uncertainty_base{0.005};
  double uncertainty_prefill{0.0001};
  double uncertainty_decode{0.001};
  double covariance_kappa{2};
  std::array<double, 4> covariance_factor_constant{0.005, 0, 0, 0};
  std::array<double, 4> covariance_factor_prefill{0, 0.0001, 0, 0};
  std::array<double, 4> covariance_factor_decode{0, 0, 0.001, 0};
  double covariance_residual{0.005};
  double reward_prefill{1};
  double reward_decode{1};
  double runtime_weight{0.1};
  double rho_prefill{0.2};
  double rho_decode{0.2};
  double previous_prefill{};
  double previous_decode{};

  bool operator==(const ProblemInstance &other) const;
};

struct ContinuousSolution {
  double p{};
  double d{};
};

struct Candidate {
  std::uint32_t prefill_tokens{};
  std::uint32_t decode_items{};
  double conservative_runtime_ns{};
  std::uint64_t memory_bytes{};
};

struct TimedObservation {
  std::uint64_t timestamp_ns{};
  double value_ns{};
};
struct PrefillRequest {
  std::uint32_t id{};
  std::uint64_t arrival_ns{};
  std::uint32_t remaining_tokens{};
  std::uint32_t allocated_tokens{};
};
struct DecodeRequest {
  std::uint32_t id{};
  std::uint64_t last_token_ns{};
};

struct Snapshot {
  std::uint64_t now_ns{};
  std::uint64_t window_ns{};
  double ttft_target_ns{};
  double tpot_target_ns{};
  std::vector<TimedObservation> historical_ttft;
  std::vector<TimedObservation> historical_tpot;
  std::vector<PrefillRequest> prefills;
  std::vector<DecodeRequest> decodes;
  double previous_prefill{};
  double previous_decode{};
};

struct OneStepScore {
  double ttft_p99_ns{};
  double tpot_p99_ns{};
  double objective{};
  bool ttft_satisfied{};
  bool tpot_satisfied{};
  std::size_t completed_ttft{}, censored_ttft{}, completed_tpot{},
      censored_tpot{};
};

double nearest_rank(std::vector<double> values, double quantile);
double sliding_window_headroom(const Snapshot &snapshot, bool ttft);
double standardized_runtime_slack(double runtime, double headroom,
                                  double scale);
std::vector<ProblemInstance> generate_instances(std::uint64_t seed,
                                                std::size_t count);
std::optional<Candidate>
project_nearest(const ContinuousSolution &target,
                const std::vector<Candidate> &candidates, double token_budget,
                double sequence_capacity);
OneStepScore score_one_step(const Snapshot &snapshot,
                            const Candidate &candidate,
                            std::uint64_t duration_ns, double token_budget,
                            double sequence_capacity, double g_prefill,
                            double g_decode, double runtime_weight,
                            double rho_prefill, double rho_decode,
                            double batch_duration_scale = 1.0);
double qp_objective(const ProblemInstance &instance, double p, double d,
                    double conservative_runtime);
bool resource_feasible(const ProblemInstance &instance, double p, double d,
                       double runtime_bound);

} // namespace sloexp
