#pragma once

#include "runtime/handoff.hpp"
#include "runtime/hardware_profile.hpp"
#include "runtime/model_profile.hpp"
#include "runtime/scheduler.hpp"
#include <chrono>
#include <cstdint>
#include <memory>
#include <optional>
#include <unordered_map>
#include <vector>

struct HeuristicAIMDConfig {
  double initial_window = 32.0;
  double minimum_window = 16.0;
  double maximum_window = 512.0;
  double additive_step = 16.0;
  double multiplicative_factor = 0.8;
  std::chrono::nanoseconds target_batch_duration =
      std::chrono::milliseconds(50);
  std::chrono::nanoseconds ttft_target = std::chrono::seconds(2);
  std::chrono::nanoseconds tpot_target = std::chrono::milliseconds(200);
  std::uint32_t starvation_threshold = 8;
  std::uint32_t max_prefill_chunk = 512;
};

struct BatchEstimate {
  long double compute_operations = 0.0L;
  long double memory_bytes = 0.0L;
  long double additional_kv_bytes = 0.0L;
  long double compute_pressure = 0.0L;
  long double bandwidth_pressure = 0.0L;
  double pressure = 0.0;
  bool valid = true;
};

class HeuristicAIMD final : public Scheduler {
public:
  HeuristicAIMD(Handoff &handoff, std::uint32_t token_budget,
                const ModelProfile &model_profile,
                const HardwareProfile &hardware_profile);
  HeuristicAIMD(Handoff &handoff, std::uint32_t token_budget,
                const ModelProfile &model_profile,
                const HardwareProfile &hardware_profile,
                HeuristicAIMDConfig config);

  [[nodiscard]] const ModelProfile &model_profile() const noexcept {
    return model_profile_;
  }

  [[nodiscard]] const HardwareProfile &hardware_profile() const noexcept {
    return hardware_profile_;
  }

  [[nodiscard]] double pressure_window() const noexcept {
    return prefill_window_;
  }
  [[nodiscard]] double prefill_window() const noexcept {
    return prefill_window_;
  }
  [[nodiscard]] double decode_window() const noexcept { return decode_window_; }
  [[nodiscard]] const BatchEstimate &last_estimate() const noexcept {
    return last_estimate_;
  }

protected:
  void build_plan(Plan &out) override;
  void on_plan_completed(const BatchOutcome &outcome) override;

private:
  struct Candidate {
    WorkItem work;
    std::uint32_t context_length = 0;
    std::uint32_t prompt_end = 0;
    double urgency = 0.0;
    std::uint32_t bypasses = 0;
  };

  [[nodiscard]] std::vector<Candidate> candidates() const;
  [[nodiscard]] BatchEstimate
  estimate(const std::vector<Candidate> &selected) const;
  [[nodiscard]] long double
  additional_kv_bytes(const Candidate &candidate) const;
  [[nodiscard]] double utility(const Candidate &candidate) const;
  void update_window(double sample, std::optional<double> &ewma,
                     double &window, bool success);

  const ModelProfile model_profile_;
  const HardwareProfile hardware_profile_;
  const HeuristicAIMDConfig config_;
  double prefill_window_ = 1.0;
  double decode_window_ = 1.0;
  std::optional<double> prefill_time_per_token_ewma_;
  std::optional<double> decode_time_per_item_ewma_;
  BatchEstimate last_estimate_{};
  std::unordered_map<RequestId, std::uint32_t> bypasses_;
};

namespace quickserve_benchmark_policy {
std::unique_ptr<Scheduler> create(Handoff &handoff, std::uint32_t token_budget,
                                  const ModelProfile &model_profile,
                                  const HardwareProfile &hardware_profile);
}
