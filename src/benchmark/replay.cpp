#include "benchmark/replay.hpp"

#include <cmath>
#include <limits>
#include <stdexcept>

namespace quickserve::benchmark {
namespace {

std::uint64_t checked_mul(std::uint64_t value, std::uint64_t factor) {
  if (value > std::numeric_limits<std::uint64_t>::max() / factor) {
    throw std::overflow_error("duration is too large");
  }
  return value * factor;
}

std::uint64_t round_ties_even(long double value) {
  constexpr long double kClockNanosecondsLimit = 9223372036854775807.0L;
  if (!std::isfinite(value) || value < 0 || value > kClockNanosecondsLimit) {
    throw std::overflow_error("scaled arrival deadline is out of range");
  }
  const long double floor_value = std::floor(value);
  const long double fraction = value - floor_value;
  auto result = static_cast<std::uint64_t>(floor_value);
  if (fraction > 0.5L || (fraction == 0.5L && (result & 1U) != 0)) {
    if (result == std::numeric_limits<std::uint64_t>::max()) {
      throw std::overflow_error("scaled arrival deadline is out of range");
    }
    ++result;
  }
  return result;
}

} // namespace

std::uint64_t parse_duration_ns(const std::string &text) {
  std::uint64_t factor = 0;
  std::string number;
  if (text.size() > 2 && text.substr(text.size() - 2) == "ms") {
    factor = 1000000ULL;
    number = text.substr(0, text.size() - 2);
  } else if (text.size() > 1) {
    number = text.substr(0, text.size() - 1);
    switch (text.back()) {
    case 's': factor = 1000000000ULL; break;
    case 'm': factor = 60000000000ULL; break;
    case 'h': factor = 3600000000000ULL; break;
    default: break;
    }
  }
  if (factor == 0 || number.empty() || number.find_first_not_of("0123456789") != std::string::npos) {
    throw std::invalid_argument("trace duration must be a positive integer followed by ms, s, m, or h");
  }
  std::size_t used = 0;
  const auto value = std::stoull(number, &used);
  if (used != number.size() || value == 0) {
    throw std::invalid_argument("trace duration must be positive");
  }
  return checked_mul(value, factor);
}

Selection select_prefix(const TraceReader &reader, const SelectionLimits &limits) {
  if (limits.max_requests && *limits.max_requests == 0) {
    throw std::invalid_argument("max requests must be positive");
  }
  TraceCursor cursor = reader.cursor();
  TraceRecord record;
  Selection result;
  while ((!limits.max_requests || result.count < *limits.max_requests) &&
         cursor.next(record)) {
    if (limits.trace_duration_ns && record.arrival_offset_ns > *limits.trace_duration_ns) break;
    ++result.count;
    result.final_source_offset_ns = record.arrival_offset_ns;
    if (result.count > static_cast<std::uint64_t>(std::numeric_limits<std::uint32_t>::max()) + 1ULL) {
      throw std::overflow_error("selected trace exceeds request ID space");
    }
  }
  if (result.count == 0) throw std::invalid_argument("selected trace is empty");
  if (result.count >= 2 && result.final_source_offset_ns == 0) {
    throw std::invalid_argument("target QPS is undefined for a zero-duration trace");
  }
  return result;
}

std::uint64_t scale_deadline_ns(std::uint64_t source_offset_ns,
                                const Selection &selection,
                                double target_qps) {
  if (!std::isfinite(target_qps) || target_qps <= 0) {
    throw std::invalid_argument("target QPS must be finite and positive");
  }
  if (selection.count == 1) return 0;
  if (selection.count < 1 || selection.final_source_offset_ns == 0) {
    throw std::invalid_argument("invalid trace selection");
  }
  const long double numerator = static_cast<long double>(source_offset_ns) *
      static_cast<long double>(selection.count - 1) * 1000000000.0L;
  const long double denominator =
      static_cast<long double>(selection.final_source_offset_ns) * target_qps;
  const auto result = round_ties_even(numerator / denominator);
  if (source_offset_ns == selection.final_source_offset_ns && result == 0) {
    throw std::invalid_argument("target QPS rounds the selected duration to zero");
  }
  return result;
}

double offered_qps(const Selection &selection, double target_qps) {
  if (selection.count < 2) return std::numeric_limits<double>::quiet_NaN();
  const auto final_ns = scale_deadline_ns(selection.final_source_offset_ns, selection, target_qps);
  return static_cast<double>(selection.count - 1) * 1e9 / static_cast<double>(final_ns);
}

void validate_clock_headroom(const Selection &selection, double target_qps,
                             std::chrono::steady_clock::time_point start) {
  const auto final_ns = scale_deadline_ns(
      selection.final_source_offset_ns, selection, target_qps);
  using Clock = std::chrono::steady_clock;
  using Period = Clock::duration::period;
  const long double headroom_ns =
      (static_cast<long double>(Clock::time_point::max().time_since_epoch().count()) -
       static_cast<long double>(start.time_since_epoch().count())) *
      static_cast<long double>(Period::num) /
      static_cast<long double>(Period::den) * 1000000000.0L;
  if (headroom_ns < 0 || static_cast<long double>(final_ns) > headroom_ns)
    throw std::overflow_error("scaled replay deadline exceeds clock headroom");
}

ReplayEngine::ReplayEngine(Selection selection, double target_qps,
                           ClockFunction clock)
    : selection_(selection), target_qps_(target_qps), clock_(std::move(clock)) {
  if (!clock_) throw std::invalid_argument("replay clock must be callable");
  start_ = clock_();
  validate_clock_headroom(selection_, target_qps_, start_);
}

std::uint64_t ReplayEngine::elapsed_ns() const {
  const auto value = std::chrono::duration_cast<std::chrono::nanoseconds>(clock_() - start_).count();
  if (value < 0) throw std::runtime_error("replay clock moved backwards");
  return static_cast<std::uint64_t>(value);
}

std::uint64_t ReplayEngine::deadline_ns(const TraceRecord &record) const {
  return scale_deadline_ns(record.arrival_offset_ns, selection_, target_qps_);
}

ReplayEngine::Clock::time_point ReplayEngine::deadline(const TraceRecord &record) const {
  return start_ + std::chrono::nanoseconds(deadline_ns(record));
}

bool ReplayEngine::due(const TraceRecord &record) const {
  return clock_() >= deadline(record);
}

} // namespace quickserve::benchmark
