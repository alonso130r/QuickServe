#include "experiment.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <random>
#include <stdexcept>
#include <tuple>

namespace sloexp {

bool ProblemInstance::operator==(const ProblemInstance &o) const {
  return std::tie(id, token_budget, sequence_capacity, prefill_available,
                  decode_available, chunk_size, memory_available, memory_base,
                  memory_prefill, memory_decode, runtime_base, runtime_prefill,
                  runtime_decode, ttft_headroom, tpot_headroom, ttft_scale,
                  tpot_scale, batch_duration_scale, fixed_margin_z,
                  risk_buffer_z, risk_weight, uncertainty_kappa,
                  uncertainty_base, uncertainty_prefill, uncertainty_decode,
                  covariance_kappa, covariance_factor_constant,
                  covariance_factor_prefill, covariance_factor_decode,
                  covariance_residual,
                  reward_prefill, reward_decode, runtime_weight, rho_prefill,
                  rho_decode, previous_prefill, previous_decode) ==
         std::tie(o.id, o.token_budget, o.sequence_capacity,
                  o.prefill_available, o.decode_available, o.chunk_size,
                  o.memory_available, o.memory_base, o.memory_prefill,
                  o.memory_decode, o.runtime_base, o.runtime_prefill,
                  o.runtime_decode, o.ttft_headroom, o.tpot_headroom,
                  o.ttft_scale, o.tpot_scale, o.batch_duration_scale,
                  o.fixed_margin_z, o.risk_buffer_z, o.risk_weight,
                  o.uncertainty_kappa, o.uncertainty_base,
                  o.uncertainty_prefill, o.uncertainty_decode,
                  o.covariance_kappa, o.covariance_factor_constant,
                  o.covariance_factor_prefill, o.covariance_factor_decode,
                  o.covariance_residual, o.reward_prefill,
                  o.reward_decode, o.runtime_weight, o.rho_prefill,
                  o.rho_decode, o.previous_prefill, o.previous_decode);
}

double nearest_rank(std::vector<double> values, double quantile) {
  if (values.empty())
    return 0.0;
  quantile = std::clamp(quantile, 0.0, 1.0);
  std::sort(values.begin(), values.end());
  const auto rank =
      static_cast<std::size_t>(std::ceil(quantile * values.size()));
  return values[std::max<std::size_t>(1, rank) - 1];
}

double sliding_window_headroom(const Snapshot &s, bool ttft) {
  const double target = ttft ? s.ttft_target_ns : s.tpot_target_ns;
  std::vector<double> fixed, ages;
  const auto &history = ttft ? s.historical_ttft : s.historical_tpot;
  for (const auto &v : history)
    if (v.timestamp_ns + s.window_ns >= s.now_ns)
      fixed.push_back(v.value_ns);
  if (ttft) {
    for (const auto &r : s.prefills)
      ages.push_back(static_cast<double>(s.now_ns - r.arrival_ns));
  } else {
    for (const auto &r : s.decodes)
      ages.push_back(static_cast<double>(s.now_ns - r.last_token_ns));
  }
  auto percentile = [&](double delay) {
    auto values = fixed;
    for (double age : ages)
      values.push_back(age + delay);
    return nearest_rank(std::move(values), 0.99);
  };
  if (percentile(0) > target)
    return 0.0;
  double low = 0.0, high = std::max(1.0, target);
  for (unsigned iteration = 0; iteration < 80; ++iteration) {
    const double mid = 0.5 * (low + high);
    if (percentile(mid) <= target)
      low = mid;
    else
      high = mid;
  }
  return low;
}

double standardized_runtime_slack(double runtime, double headroom,
                                  double scale) {
  if (!(scale > 0))
    throw std::invalid_argument("standardization scale must be positive");
  return (runtime - headroom) / scale;
}

std::vector<ProblemInstance> generate_instances(std::uint64_t seed,
                                                std::size_t count) {
  std::mt19937_64 rng(seed);
  std::uniform_real_distribution<double> unit(0.0, 1.0);
  std::vector<ProblemInstance> out;
  out.reserve(count);
  for (std::size_t i = 0; i < count; ++i) {
    ProblemInstance p;
    p.id = i;
    const unsigned pressure = static_cast<unsigned>((i + seed) % 3);
    const unsigned mix = static_cast<unsigned>((i / 3 + seed) % 3);
    const unsigned tightness = static_cast<unsigned>((i / 9 + seed) % 3);
    p.reward_prefill = (1.0 + pressure) * (mix == 0 ? 1.8 : 0.8);
    p.reward_decode = (1.0 + pressure) * (mix == 2 ? 1.8 : 0.8);
    p.ttft_headroom = p.tpot_headroom = 0.35 + 0.18 * (2 - tightness);
    p.ttft_scale = p.tpot_scale = 0.1;
    p.fixed_margin_z = 0.2;
    p.memory_available = 0.55 + 0.2 * (2 - tightness);
    p.uncertainty_base = 0.002 + 0.01 * pressure;
    p.previous_prefill = unit(rng) * p.token_budget;
    p.previous_decode = unit(rng) * p.sequence_capacity;
    p.prefill_available = 64 + std::floor(unit(rng) * 448);
    p.decode_available = 4 + std::floor(unit(rng) * 60);
    out.push_back(p);
  }
  return out;
}

std::optional<Candidate>
project_nearest(const ContinuousSolution &target,
                const std::vector<Candidate> &candidates, double token_budget,
                double sequence_capacity) {
  if (candidates.empty())
    return std::nullopt;
  return *std::min_element(
      candidates.begin(), candidates.end(),
      [&](const Candidate &a, const Candidate &b) {
        auto key = [&](const Candidate &c) {
          const double dp = (c.prefill_tokens - target.p) / token_budget;
          const double dd = (c.decode_items - target.d) / sequence_capacity;
          return std::tuple{
              dp * dp + dd * dd, c.conservative_runtime_ns,
              -static_cast<std::int64_t>(c.prefill_tokens + c.decode_items),
              c.prefill_tokens, c.decode_items};
        };
        return key(a) < key(b);
      });
}

OneStepScore score_one_step(const Snapshot &s, const Candidate &c,
                            std::uint64_t duration_ns, double token_budget,
                            double sequence_capacity, double gp, double gd,
                            double wt, double rp, double rd,
                            double batch_scale) {
  const std::uint64_t end = s.now_ns + duration_ns;
  std::vector<double> ttft, tpot;
  for (const auto &v : s.historical_ttft)
    if (v.timestamp_ns + s.window_ns >= end)
      ttft.push_back(v.value_ns);
  for (const auto &v : s.historical_tpot)
    if (v.timestamp_ns + s.window_ns >= end)
      tpot.push_back(v.value_ns);
  std::uint32_t remaining_p = c.prefill_tokens;
  OneStepScore result;
  for (const auto &r : s.prefills) {
    const std::uint32_t served = std::min(remaining_p, r.allocated_tokens);
    remaining_p -= served;
    const double age = static_cast<double>(end - r.arrival_ns);
    ttft.push_back(age);
    if (served >= r.remaining_tokens)
      ++result.completed_ttft;
    else
      ++result.censored_ttft;
  }
  std::uint32_t remaining_d = c.decode_items;
  for (const auto &r : s.decodes) {
    (void)remaining_d;
    const double age = static_cast<double>(end - r.last_token_ns);
    tpot.push_back(age);
    if (remaining_d) {
      --remaining_d;
      ++result.completed_tpot;
    } else
      ++result.censored_tpot;
  }
  result.ttft_p99_ns = nearest_rank(std::move(ttft), 0.99);
  result.tpot_p99_ns = nearest_rank(std::move(tpot), 0.99);
  result.ttft_satisfied = result.ttft_p99_ns <= s.ttft_target_ns;
  result.tpot_satisfied = result.tpot_p99_ns <= s.tpot_target_ns;
  const double p = c.prefill_tokens, d = c.decode_items;
  result.objective =
      -gp * p / token_budget - gd * d / sequence_capacity +
      wt * duration_ns / batch_scale +
      0.5 * rp * std::pow((p - s.previous_prefill) / token_budget, 2) +
      0.5 * rd * std::pow((d - s.previous_decode) / sequence_capacity, 2);
  return result;
}

double qp_objective(const ProblemInstance &i, double p, double d,
                    double runtime) {
  return -i.reward_prefill * p / i.token_budget -
         i.reward_decode * d / i.sequence_capacity +
         i.runtime_weight * runtime / i.batch_duration_scale +
         0.5 * i.rho_prefill *
             std::pow((p - i.previous_prefill) / i.token_budget, 2) +
         0.5 * i.rho_decode *
             std::pow((d - i.previous_decode) / i.sequence_capacity, 2);
}

bool resource_feasible(const ProblemInstance &i, double p, double d,
                       double runtime) {
  return p >= -1e-8 && d >= -1e-8 && p <= i.prefill_available + 1e-8 &&
         d <= i.decode_available + 1e-8 && p + d <= i.token_budget + 1e-8 &&
         d + p / i.chunk_size <= i.sequence_capacity + 1e-8 &&
         i.memory_base + i.memory_prefill * p + i.memory_decode * d <=
             i.memory_available + 1e-8 &&
         runtime <= std::min(i.ttft_headroom, i.tpot_headroom) + 1e-8;
}

} // namespace sloexp
