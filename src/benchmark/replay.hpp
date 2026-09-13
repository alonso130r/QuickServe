#pragma once

#include "benchmark/trace.hpp"

#include <cstdint>
#include <chrono>
#include <functional>
#include <optional>
#include <string>
#include <unordered_map>

namespace quickserve::benchmark {

struct SelectionLimits {
  std::optional<std::uint64_t> max_requests;
  std::optional<std::uint64_t> trace_duration_ns;
};

struct Selection {
  std::uint64_t count{};
  std::uint64_t final_source_offset_ns{};
};

std::uint64_t parse_duration_ns(const std::string &text);
Selection select_prefix(const TraceReader &reader, const SelectionLimits &limits);
std::uint64_t scale_deadline_ns(std::uint64_t source_offset_ns,
                                const Selection &selection,
                                double target_qps);
double offered_qps(const Selection &selection, double target_qps);
void validate_clock_headroom(const Selection &selection, double target_qps,
                             std::chrono::steady_clock::time_point start);

class CausalReplayGate {
public:
  CausalReplayGate(Selection selection, double target_qps)
      : selection_(selection), target_qps_(target_qps) {}
  [[nodiscard]] std::optional<std::uint64_t>
  deadline_ns(const TraceRecord &record) const;
  void complete(const TraceRecord &record, std::uint64_t completion_ns);

private:
  struct CompletedTurn {
    std::uint32_t turn_index{};
    std::uint64_t source_offset_ns{};
    std::uint64_t completion_ns{};
  };
  Selection selection_;
  double target_qps_{};
  std::unordered_map<std::string, CompletedTurn> completed_;
};

class ReplayEngine {
public:
  using Clock = std::chrono::steady_clock;
  using ClockFunction = std::function<Clock::time_point()>;
  ReplayEngine(Selection selection, double target_qps, ClockFunction clock);
  [[nodiscard]] Clock::time_point now() const { return clock_(); }
  [[nodiscard]] Clock::time_point start() const { return start_; }
  [[nodiscard]] std::uint64_t elapsed_ns() const;
  [[nodiscard]] std::uint64_t deadline_ns(const TraceRecord &record) const;
  [[nodiscard]] Clock::time_point deadline(const TraceRecord &record) const;
  [[nodiscard]] bool due(const TraceRecord &record) const;
  [[nodiscard]] const ClockFunction &clock_function() const { return clock_; }

private:
  Selection selection_;
  double target_qps_;
  ClockFunction clock_;
  Clock::time_point start_;
};

} // namespace quickserve::benchmark
