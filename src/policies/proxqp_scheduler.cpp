#include "proxqp_scheduler.hpp"
#include "batch_utilities.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <limits>
#include <map>
#include <set>
#include <stdexcept>
#include <unordered_set>

namespace {
double ns(RequestState::Clock::duration value) {
  return std::chrono::duration<double, std::nano>(value).count();
}

double nearest_rank(std::vector<double> values) {
  if (values.empty()) return 0;
  std::sort(values.begin(), values.end());
  return values[std::max<std::size_t>(1, static_cast<std::size_t>(
      std::ceil(0.99 * values.size()))) - 1];
}

double profile_number(const std::map<std::string, std::string> &v,
                      const std::string &key) {
  const auto it = v.find(key);
  if (it == v.end()) throw std::invalid_argument("missing profile key: " + key);
  std::size_t used = 0;
  const double result = std::stod(it->second, &used);
  if (used != it->second.size() || !std::isfinite(result))
    throw std::invalid_argument("invalid profile value: " + key);
  return result;
}

std::uint32_t profile_u32(const std::map<std::string, std::string> &v,
                          const std::string &key) {
  const double result = profile_number(v, key);
  if (result < 0 || result > UINT32_MAX || result != std::floor(result))
    throw std::invalid_argument("invalid profile integer: " + key);
  return static_cast<std::uint32_t>(result);
}
}

double proxqp_decode_urgency(double normalized_age,
                             const ProxQPSchedulerConfig &config) {
  const double age = std::max(0.0, normalized_age);
  const double z = config.decode_urgency_steepness *
                   (age - config.decode_urgency_knee);
  const double softplus = z > 0 ? z + std::log1p(std::exp(-z))
                                : std::log1p(std::exp(z));
  return age + config.decode_urgency_gain * softplus /
                   config.decode_urgency_steepness;
}

double proxqp_decode_reward(const std::vector<double> &urgencies,
                            const ProxQPSchedulerConfig &config) {
  if (urgencies.empty()) return config.base_decode_reward;
  double squared_sum = 0;
  for (const double urgency : urgencies) squared_sum += urgency * urgency;
  return config.base_decode_reward + squared_sum / urgencies.size();
}

ProxQPPolicyConfiguration
load_proxqp_policy_configuration(const std::filesystem::path &path) {
  std::ifstream input(path);
  if (!input)
    throw std::invalid_argument("cannot open ProxQP policy profile: " + path.string());
  std::map<std::string, std::string> values;
  std::string line;
  while (std::getline(input, line)) {
    if (line.empty() || line.front() == '#') continue;
    const auto equal = line.find('=');
    if (equal == std::string::npos || equal == 0 || equal + 1 == line.size())
      throw std::invalid_argument("malformed ProxQP profile line");
    const auto key = line.substr(0, equal);
    if (!values.emplace(key, line.substr(equal + 1)).second)
      throw std::invalid_argument("duplicate ProxQP profile key: " + key);
  }
  const auto tiers = profile_u32(values, "tier_count");
  std::set<std::string> allowed{
      "schema_version", "calibration_id", "ttft_target_ns", "tpot_target_ns",
      "window_ns", "runtime_weight", "rho_prefill", "rho_decode",
      "boundary_buffer_z", "boundary_weight", "tier_count"};
  ProxQPPolicyConfiguration result;
  result.profile.schema_version = profile_u32(values, "schema_version");
  const auto id = values.find("calibration_id");
  if (id == values.end() || id->second.empty())
    throw std::invalid_argument("missing profile key: calibration_id");
  result.profile.calibration_id = id->second;
  const auto duration = [&](const std::string &key) {
    const double value = profile_number(values, key);
    if (value <= 0 || value > double(INT64_MAX))
      throw std::invalid_argument("invalid profile duration: " + key);
    return std::chrono::nanoseconds(static_cast<std::int64_t>(value));
  };
  result.config.ttft_target = duration("ttft_target_ns");
  result.config.tpot_target = duration("tpot_target_ns");
  result.config.window = duration("window_ns");
  result.config.runtime_weight = profile_number(values, "runtime_weight");
  result.config.rho_prefill = profile_number(values, "rho_prefill");
  result.config.rho_decode = profile_number(values, "rho_decode");
  result.config.boundary_buffer_z = profile_number(values, "boundary_buffer_z");
  result.config.boundary_weight = profile_number(values, "boundary_weight");
  for (std::uint32_t i = 0; i < tiers; ++i) {
    const auto p = "tier." + std::to_string(i) + ".";
    const std::vector<std::string> fields{
        "context_min", "context_max", "tau_base_ns",
        "tau_prefill_ns_per_token", "tau_decode_ns_per_item",
        "runtime_margin_ns", "runtime_scale_ns", "token_capacity",
        "sequence_capacity"};
    for (const auto &field : fields) allowed.insert(p + field);
    result.profile.tiers.push_back({
        profile_u32(values, p + "context_min"),
        profile_u32(values, p + "context_max"),
        profile_number(values, p + "tau_base_ns"),
        profile_number(values, p + "tau_prefill_ns_per_token"),
        profile_number(values, p + "tau_decode_ns_per_item"),
        profile_number(values, p + "runtime_margin_ns"),
        profile_number(values, p + "runtime_scale_ns"),
        profile_u32(values, p + "token_capacity"),
        profile_u32(values, p + "sequence_capacity")});
  }
  for (const auto &[key, unused] : values)
    if (!allowed.count(key))
      throw std::invalid_argument("unknown ProxQP profile key: " + key);
  return result;
}

ProxQPScheduler::ProxQPScheduler(Handoff &handoff, std::uint32_t budget,
                                 const ModelProfile &model,
                                 const HardwareProfile &hardware,
                                 ProxQPRuntimeProfile profile,
                                 ProxQPSchedulerConfig config)
    : Scheduler(handoff, budget), model_(model), hardware_(hardware),
      profile_(std::move(profile)), config_(config) {
  if (profile_.schema_version != 1 || profile_.tiers.empty() ||
      config_.ttft_target <= std::chrono::nanoseconds::zero() ||
      config_.tpot_target <= std::chrono::nanoseconds::zero() ||
      config_.window <= std::chrono::nanoseconds::zero() ||
      config_.max_prefill_chunk == 0 || config_.max_considered_requests == 0 ||
      !std::isfinite(config_.decode_urgency_knee) ||
      config_.decode_urgency_knee < 0 ||
      !std::isfinite(config_.decode_urgency_steepness) ||
      config_.decode_urgency_steepness <= 0 ||
      !std::isfinite(config_.decode_urgency_gain) ||
      config_.decode_urgency_gain < 0)
    throw std::invalid_argument("invalid ProxQP scheduler configuration");
  std::sort(profile_.tiers.begin(), profile_.tiers.end(),
            [](const auto &a, const auto &b) { return a.context_min < b.context_min; });
  std::uint32_t previous_max = 0;
  bool first = true;
  for (const auto &t : profile_.tiers) {
    if (t.context_min > t.context_max || !std::isfinite(t.tau_base_ns) ||
        !std::isfinite(t.tau_prefill_ns_per_token) ||
        !std::isfinite(t.tau_decode_ns_per_item) ||
        t.tau_prefill_ns_per_token < 0 || t.tau_decode_ns_per_item < 0 ||
        t.runtime_margin_ns < 0 || !(t.runtime_scale_ns > 0) ||
        t.token_capacity == 0 || t.sequence_capacity == 0 ||
        (!first && t.context_min <= previous_max))
      throw std::invalid_argument("invalid ProxQP runtime profile");
    first = false;
    previous_max = t.context_max;
  }
}

void ProxQPScheduler::set_decision_observer(DecisionObserver observer) {
  if (started_) throw std::logic_error("observer set after scheduling started");
  decision_observer_ = std::move(observer);
}
void ProxQPScheduler::set_measurement_observer(MeasurementObserver observer) {
  if (started_) throw std::logic_error("observer set after scheduling started");
  measurement_observer_ = std::move(observer);
}

std::vector<ProxQPScheduler::Candidate> ProxQPScheduler::candidates() const {
  std::vector<Candidate> out;
  const auto now = policy_now();
  for (const auto &r : policy_requests()) {
    const auto found = bypasses_.find(r.id);
    const auto bypass = found == bypasses_.end() ? 0 : found->second;
    if (r.stage == RequestState::Stage::Prefill && r.prefill_position < r.prompt_length) {
      const auto count = std::min({r.prompt_length - r.prefill_position,
                                   config_.max_prefill_chunk, token_budget_});
      out.push_back({{r.id, r.prefill_position, r.prefill_position + count,
                      WorkKind::Prefill},
                     ns(now - r.arrival_time) / ns(config_.ttft_target), bypass});
    } else if (r.stage == RequestState::Stage::Decode && r.decoded_count > 0) {
      const std::uint64_t end = static_cast<std::uint64_t>(r.prompt_length) +
                                r.decoded_count;
      if (end <= std::numeric_limits<std::uint32_t>::max())
        out.push_back({{r.id, static_cast<std::uint32_t>(end - 1),
                        static_cast<std::uint32_t>(end), WorkKind::Decode},
                       proxqp_decode_urgency(
                           ns(now - r.last_token_time) /
                               ns(config_.tpot_target),
                           config_),
                       bypass});
    }
  }
  return out;
}

const ProxQPRuntimeTier *ProxQPScheduler::select_tier() const {
  std::uint32_t maximum = 0;
  for (const auto &r : policy_requests()) {
    if (r.stage == RequestState::Stage::Prefill)
      maximum = std::max(maximum, std::min(r.prompt_length,
          r.prefill_position + config_.max_prefill_chunk));
    else if (r.stage == RequestState::Stage::Decode)
      maximum = std::max(maximum, r.prompt_length + r.decoded_count + 1);
  }
  for (const auto &tier : profile_.tiers)
    if (maximum >= tier.context_min && maximum <= tier.context_max) return &tier;
  return nullptr;
}

double ProxQPScheduler::headroom(bool ttft) const {
  std::vector<double> fixed, ages;
  const auto now = policy_now();
  const auto &window = ttft ? ttft_window_ : tpot_window_;
  for (const auto &o : window)
    if (o.at + config_.window >= now) fixed.push_back(o.value_ns);
  for (const auto &r : policy_requests()) {
    if (ttft && !r.first_token_recorded)
      ages.push_back(ns(now - r.arrival_time));
    else if (!ttft && r.stage == RequestState::Stage::Decode && r.first_token_recorded)
      ages.push_back(ns(now - r.last_token_time));
  }
  const double target = ns(ttft ? config_.ttft_target : config_.tpot_target);
  if (ages.empty()) return target;
  auto percentile = [&](double delay) {
    auto values = fixed;
    for (double age : ages) values.push_back(age + delay);
    return nearest_rank(std::move(values));
  };
  if (percentile(0) > target) return 0;
  double low = 0, high = target;
  for (int k = 0; k < 80; ++k) {
    const double mid = (low + high) / 2;
    (percentile(mid) <= target ? low : high) = mid;
  }
  return low;
}

void ProxQPScheduler::emit_decision(ProxQPDecisionRecord &record) {
  if (!decision_observer_ || decision_observer_failed_) return;
  try { decision_observer_(record); }
  catch (...) { decision_observer_failed_ = true; record.callback_failed = true; }
}

bool ProxQPScheduler::urgent_fallback(Plan &out, std::vector<Candidate> available,
                                     const std::string &reason,
                                     ProxQPDecisionRecord &record) {
  std::stable_sort(available.begin(), available.end(), [](const auto &a, const auto &b) {
    if (a.urgency != b.urgency) return a.urgency > b.urgency;
    if (a.bypass != b.bypass) return a.bypass > b.bypass;
    return a.work.id < b.work.id;
  });
  for (auto candidate : available) {
    if (candidate.work.kind == WorkKind::Prefill) {
      while (candidate.work.token_count() > 1) {
        std::vector<WorkItem> one{candidate.work};
        if (quickserve::policy::evaluate_batch_resources(policy_requests(), one,
              token_budget_, model_, hardware_).valid) break;
        --candidate.work.token_end;
      }
    }
    std::vector<WorkItem> one{candidate.work};
    if (!quickserve::policy::evaluate_batch_resources(policy_requests(), one,
          token_budget_, model_, hardware_).valid) continue;
    out.work = one;
    record.fallback_reason = reason;
    record.projected_prefill = candidate.work.kind == WorkKind::Prefill
                                   ? candidate.work.token_count() : 0;
    record.projected_decode = candidate.work.kind == WorkKind::Decode ? 1 : 0;
    emit_decision(record);
    return true;
  }
  return false;
}

void ProxQPScheduler::build_plan(Plan &out) {
  started_ = true;
  ProxQPDecisionRecord record;
  record.timestamp = policy_now();
  auto available = candidates();
  if (available.size() > config_.max_considered_requests) {
    urgent_fallback(out, std::move(available), "considered_request_limit_exceeded", record);
    return;
  }
  const auto *tier = select_tier();
  if (!tier) {
    urgent_fallback(out, std::move(available), "profile_not_covered", record);
    return;
  }
  std::vector<Candidate> prefills, decodes;
  for (auto c : available) (c.work.kind == WorkKind::Prefill ? prefills : decodes).push_back(c);
  auto order = [](auto &values) {
    std::stable_sort(values.begin(), values.end(), [](const auto &a, const auto &b) {
      if (a.urgency != b.urgency) return a.urgency > b.urgency;
      if (a.bypass != b.bypass) return a.bypass > b.bypass;
      return a.work.id < b.work.id;
    });
  };
  order(prefills); order(decodes);
  const double gp = config_.base_prefill_reward +
      (prefills.empty() ? 0 : prefills.front().urgency * prefills.front().urgency);
  std::vector<double> decode_urgencies;
  decode_urgencies.reserve(decodes.size());
  for (const auto &decode : decodes)
    decode_urgencies.push_back(decode.urgency);
  const double gd = proxqp_decode_reward(decode_urgencies, config_);
  const long double kv = static_cast<long double>(model_.layer_count) *
      model_.kv_head_count * model_.head_dimension *
      (model_.key_effective_bytes_per_scalar + model_.value_effective_bytes_per_scalar);
  std::uint64_t resident = 0;
  for (const auto &r : policy_requests()) {
    resident += r.prefill_position;
    if (r.decoded_count) resident += r.decoded_count - 1;
  }
  quickserve::optimization::ProxQPInput input;
  input.token_budget = std::min<std::uint64_t>({token_budget_, model_.batch_capacity, tier->token_capacity});
  input.sequence_capacity = std::min<std::uint64_t>(model_.max_sequences, tier->sequence_capacity);
  input.prefill_available = prefills.empty() ? 0 : input.token_budget;
  input.decode_available = std::min<double>(decodes.size(), input.sequence_capacity);
  input.chunk_size = config_.max_prefill_chunk;
  input.memory_available = hardware_.total_memory_bytes;
  input.memory_base = model_.model_bytes + static_cast<double>(kv * resident);
  input.memory_prefill = input.memory_decode = static_cast<double>(kv);
  input.runtime_base_ns = tier->tau_base_ns;
  input.runtime_prefill_ns = tier->tau_prefill_ns_per_token;
  input.runtime_decode_ns = tier->tau_decode_ns_per_item;
  input.runtime_margin_ns = tier->runtime_margin_ns;
  input.runtime_scale_ns = tier->runtime_scale_ns;
  const double maximum_runtime = tier->tau_base_ns + tier->runtime_margin_ns +
      tier->tau_prefill_ns_per_token * input.token_budget +
      tier->tau_decode_ns_per_item * input.sequence_capacity;
  input.ttft_headroom_ns = prefills.empty() ? maximum_runtime : headroom(true);
  input.tpot_headroom_ns = decodes.empty() ? maximum_runtime : headroom(false);
  input.batch_duration_scale_ns = std::max(1.0, tier->tau_base_ns + tier->runtime_margin_ns);
  input.reward_prefill = gp; input.reward_decode = gd;
  input.runtime_weight = config_.runtime_weight;
  input.rho_prefill = config_.rho_prefill; input.rho_decode = config_.rho_decode;
  input.previous_prefill = previous_prefill_; input.previous_decode = previous_decode_;
  input.boundary_buffer_z = config_.boundary_buffer_z;
  input.boundary_weight = config_.boundary_weight;
  const auto solved = quickserve::optimization::solve_proxqp(input);
  record.solver_status = solved.status; record.solve_ns = solved.solve_ns;
  record.continuous_prefill = solved.prefill; record.continuous_decode = solved.decode;
  if (!solved.success || !std::isfinite(solved.prefill) || !std::isfinite(solved.decode)) {
    urgent_fallback(out, std::move(available), "solver_failed", record);
    return;
  }
  struct Prefix { std::uint32_t total{}; std::vector<WorkItem> work; };
  auto prefixes = [&](const std::vector<Candidate> &source, bool prefill) {
    std::vector<Prefix> result{{0,{}}};
    Prefix current;
    for (const auto &c : source) {
      if (current.work.size() >= input.sequence_capacity) break;
      current.work.push_back(c.work);
      current.total += prefill ? c.work.token_count() : 1;
      if (current.total > input.token_budget) break;
      result.push_back(current);
    }
    return result;
  };
  auto pp = prefixes(prefills, true), dp = prefixes(decodes, false);
  auto trim = [&](auto values, double target) {
    std::stable_sort(values.begin(), values.end(), [&](const auto &a, const auto &b) {
      const double da = std::abs(a.total-target), db = std::abs(b.total-target);
      return da == db ? a.total < b.total : da < db;
    });
    if (values.size() > 8) values.resize(8);
    return values;
  };
  pp = trim(std::move(pp), solved.prefill); dp = trim(std::move(dp), solved.decode);
  double best = std::numeric_limits<double>::infinity();
  double best_distance = std::numeric_limits<double>::infinity();
  std::vector<WorkItem> chosen;
  for (const auto &p : pp) for (const auto &d : dp) {
    std::vector<WorkItem> work = p.work;
    work.insert(work.end(), d.work.begin(), d.work.end());
    const auto usage = quickserve::policy::evaluate_batch_resources(
        policy_requests(), work, token_budget_, model_, hardware_);
    const double runtime = tier->tau_base_ns + tier->tau_prefill_ns_per_token*p.total +
        tier->tau_decode_ns_per_item*d.total + tier->runtime_margin_ns;
    if (work.empty() || !usage.valid) continue;
    const double distance = std::pow((p.total-solved.prefill)/input.token_budget,2) +
        std::pow((d.total-solved.decode)/input.sequence_capacity,2);
    const double objective = quickserve::optimization::proxqp_objective(
        input, p.total, d.total);
    if (objective + config_.projection_tolerance < best ||
        (std::abs(objective - best) <= config_.projection_tolerance &&
         distance < best_distance)) {
      best = objective;
      best_distance = distance;
      chosen = std::move(work);
      record.projected_prefill = p.total; record.projected_decode = d.total;
      record.conservative_runtime_ns = runtime;
    }
  }
  if (chosen.empty()) {
    urgent_fallback(out, std::move(available), "projection_empty", record);
    return;
  }
  out.work = std::move(chosen);
  pending_tier_max_ = tier->context_max;
  pending_prediction_ = record.conservative_runtime_ns;
  emit_decision(record);
}

void ProxQPScheduler::on_request_timing(const PolicyTimingEvent &event) {
  if (event.kind == PolicyTimingEventKind::FirstToken)
    ttft_window_.push_back({event.timestamp, ns(event.latency)});
  else if (event.kind == PolicyTimingEventKind::LaterToken)
    tpot_window_.push_back({event.timestamp, ns(event.latency)});
}

void ProxQPScheduler::on_plan_completed(const BatchOutcome &outcome) {
  previous_prefill_ = outcome.prefill_tokens;
  previous_decode_ = outcome.decode_items;
  if (!measurement_observer_ || measurement_observer_failed_) return;
  ProxQPMeasurementRecord record{outcome, pending_tier_max_, pending_prediction_,
                                  ns(outcome.duration)-pending_prediction_};
  try { measurement_observer_(record); }
  catch (...) { measurement_observer_failed_ = true; }
  pending_tier_max_ = 0; pending_prediction_ = 0;
}

namespace quickserve_benchmark_policy {
std::unique_ptr<Scheduler> create(Handoff &handoff, std::uint32_t budget,
                                  const ModelProfile &model,
                                  const HardwareProfile &hardware,
                                  const std::filesystem::path &config_path) {
  if (config_path.empty())
    throw std::invalid_argument("ProxQP scheduler requires --policy-config");
  auto loaded = load_proxqp_policy_configuration(config_path);
  if (loaded.profile.tiers.empty() ||
      loaded.profile.tiers.back().context_max != model.context_capacity)
    throw std::invalid_argument("ProxQP profile context capacity does not match model");
  for (const auto &tier : loaded.profile.tiers)
    if (tier.token_capacity != model.batch_capacity ||
        tier.sequence_capacity != model.max_sequences)
      throw std::invalid_argument("ProxQP profile batch capacities do not match model");
  return std::make_unique<ProxQPScheduler>(handoff, budget, model, hardware,
                                           std::move(loaded.profile), loaded.config);
}
}
