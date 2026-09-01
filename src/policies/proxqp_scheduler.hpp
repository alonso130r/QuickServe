#pragma once

#include "optimization/proxqp_solver.hpp"
#include "runtime/hardware_profile.hpp"
#include "runtime/model_profile.hpp"
#include "runtime/scheduler.hpp"

#include <chrono>
#include <cstdint>
#include <deque>
#include <filesystem>
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

struct ProxQPRuntimeTier {
  std::uint32_t context_min{};
  std::uint32_t context_max{};
  double tau_base_ns{};
  double tau_prefill_ns_per_token{};
  double tau_decode_ns_per_item{};
  double runtime_margin_ns{};
  double runtime_scale_ns{1};
  std::uint32_t token_capacity{};
  std::uint32_t sequence_capacity{};
};

struct ProxQPRuntimeProfile {
  std::uint32_t schema_version{1};
  std::string calibration_id;
  std::vector<ProxQPRuntimeTier> tiers;
};

struct ProxQPSchedulerConfig {
  std::chrono::nanoseconds ttft_target{std::chrono::seconds(2)};
  std::chrono::nanoseconds tpot_target{std::chrono::milliseconds(200)};
  std::chrono::nanoseconds window{std::chrono::seconds(60)};
  std::uint32_t max_prefill_chunk{512};
  std::uint32_t starvation_threshold{8};
  std::uint32_t max_considered_requests{4096};
  double base_prefill_reward{1};
  double base_decode_reward{1};
  double decode_urgency_knee{0.8};
  double decode_urgency_steepness{8};
  double decode_urgency_gain{4};
  double runtime_weight{0.1};
  double rho_prefill{0.2};
  double rho_decode{0.2};
  double boundary_buffer_z{1};
  double boundary_weight{0.25};
  double projection_tolerance{1e-12};
};

[[nodiscard]] double proxqp_decode_urgency(
    double normalized_age, const ProxQPSchedulerConfig &config);
[[nodiscard]] double proxqp_decode_reward(
    const std::vector<double> &urgencies,
    const ProxQPSchedulerConfig &config);

struct ProxQPPolicyConfiguration {
  ProxQPRuntimeProfile profile;
  ProxQPSchedulerConfig config;
};

ProxQPPolicyConfiguration
load_proxqp_policy_configuration(const std::filesystem::path &);

struct ProxQPDecisionRecord {
  RequestState::TimePoint timestamp{};
  std::string solver_status;
  std::string fallback_reason;
  double continuous_prefill{};
  double continuous_decode{};
  std::uint32_t projected_prefill{};
  std::uint32_t projected_decode{};
  double conservative_runtime_ns{};
  double solve_ns{};
  bool recovery{};
  bool callback_failed{};
};

struct ProxQPMeasurementRecord {
  BatchOutcome outcome;
  std::uint32_t context_tier_max{};
  double predicted_runtime_ns{};
  double residual_ns{};
};

class ProxQPScheduler final : public Scheduler {
public:
  using DecisionObserver = std::function<void(const ProxQPDecisionRecord &)>;
  using MeasurementObserver =
      std::function<void(const ProxQPMeasurementRecord &)>;

  ProxQPScheduler(Handoff &, std::uint32_t token_budget, const ModelProfile &,
                  const HardwareProfile &, ProxQPRuntimeProfile,
                  ProxQPSchedulerConfig = {});
  void set_decision_observer(DecisionObserver observer);
  void set_measurement_observer(MeasurementObserver observer);

protected:
  void build_plan(Plan &) override;
  void on_plan_completed(const BatchOutcome &) override;
  void on_request_timing(const PolicyTimingEvent &) override;

private:
  struct Observation { RequestState::TimePoint at; double value_ns; };
  struct Candidate { WorkItem work; double urgency{}; std::uint32_t bypass{}; };
  const ProxQPRuntimeTier *select_tier() const;
  double headroom(bool ttft) const;
  std::vector<Candidate> candidates() const;
  bool urgent_fallback(Plan &, std::vector<Candidate>, const std::string &,
                       ProxQPDecisionRecord &);
  void emit_decision(ProxQPDecisionRecord &);

  ModelProfile model_;
  HardwareProfile hardware_;
  ProxQPRuntimeProfile profile_;
  ProxQPSchedulerConfig config_;
  std::deque<Observation> ttft_window_, tpot_window_;
  std::unordered_map<RequestId, std::uint32_t> bypasses_;
  double previous_prefill_{}, previous_decode_{};
  DecisionObserver decision_observer_;
  MeasurementObserver measurement_observer_;
  bool decision_observer_failed_{}, measurement_observer_failed_;
  bool started_{};
  std::uint32_t pending_tier_max_{};
  double pending_prediction_{};
};

namespace quickserve_benchmark_policy {
#define QUICKSERVE_BENCHMARK_POLICY_USES_CONFIG 1
std::unique_ptr<Scheduler> create(Handoff &, std::uint32_t,
                                  const ModelProfile &,
                                  const HardwareProfile &,
                                  const std::filesystem::path &);
}
