#include "policies/proxqp_scheduler.hpp"

#include <cstdio>
#include <cmath>
#include <filesystem>
#include <fstream>

namespace {
int failures = 0;
#define CHECK(x) do { if (!(x)) { std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #x); ++failures; } } while (false)

ModelProfile model() {
  return {1, 1'000'000, 2, 128, 4, 2, 32, 2, 2, 1, 4096, 64, 16};
}
HardwareProfile hardware() {
  return {"test", 1ULL << 30, 4, 4, 4096};
}
ProxQPRuntimeProfile profile() {
  ProxQPRuntimeProfile p;
  p.tiers.push_back({0, 4096, 1'000, 10, 100, 100, 1'000, 64, 16});
  return p;
}

ModelProfile mixed_model() {
  auto result = model();
  result.batch_capacity = 512;
  result.max_sequences = 16;
  return result;
}

ProxQPRuntimeProfile mixed_profile() {
  ProxQPRuntimeProfile result;
  result.tiers.push_back({0, 4096, 1'000, 10, 100, 100, 1'000, 512, 16});
  return result;
}

void test_default_prefill_chunk_uses_full_token_budget() {
  const ProxQPSchedulerConfig config;
  CHECK(config.max_prefill_chunk == 512);
}

void test_decode_urgency_is_finite_and_strictly_monotone() {
  const ProxQPSchedulerConfig config;
  const double young = proxqp_decode_urgency(0.25, config);
  const double knee = proxqp_decode_urgency(0.8, config);
  const double late = proxqp_decode_urgency(1.5, config);
  CHECK(std::isfinite(young));
  CHECK(std::isfinite(knee));
  CHECK(std::isfinite(late));
  CHECK(young < knee);
  CHECK(knee < late);
  CHECK(std::isfinite(proxqp_decode_urgency(1.0e6, config)));
}

void test_decode_reward_uses_mean_squared_urgency() {
  ProxQPSchedulerConfig config;
  config.base_decode_reward = 2.0;
  const std::vector<double> urgencies{1.0, 3.0};
  CHECK(proxqp_decode_reward(urgencies, config) == 7.0);
  CHECK(proxqp_decode_reward({}, config) == 2.0);
}

void test_smaller_chunk_allows_mixed_prefill_and_decode_plan() {
  Handoff handoff(16);
  RequestState::TimePoint now{};
  ProxQPSchedulerConfig config;
  config.max_prefill_chunk = 256;
  ProxQPScheduler scheduler(handoff, 512, mixed_model(), hardware(),
                            mixed_profile(), config);
  scheduler.set_clock([&] { return now; });
  const auto decode_id = scheduler.submit_synthetic(1, 3);
  const auto prefill_id = scheduler.submit_synthetic(512, 3);

  scheduler.run_once();
  Admission admission{};
  CHECK(handoff.try_take_admission(admission));
  CHECK(handoff.try_report_admission({decode_id, 1, ErrorCode::None}));
  CHECK(handoff.try_take_admission(admission));
  CHECK(handoff.try_report_admission({prefill_id, 512, ErrorCode::None}));
  scheduler.run_once();

  Plan *setup = handoff.consume_plan();
  CHECK(setup != nullptr);
  if (!setup) return;
  CHECK(handoff.try_report_completion(
      {decode_id, 1, 1, 7, WorkKind::Prefill, ErrorCode::None, true, false}));
  for (const auto &work : setup->work) {
    if (work.id == prefill_id && work.kind == WorkKind::Prefill) {
      CHECK(handoff.try_report_completion(
          {prefill_id, work.token_end, 0, 0, WorkKind::Prefill,
           ErrorCode::None, false, false}));
    }
  }
  handoff.retire_plan(setup);

  now += std::chrono::milliseconds(100);
  scheduler.run_once();
  Plan *mixed = handoff.consume_plan();
  CHECK(mixed != nullptr);
  if (!mixed) return;
  bool saw_prefill = false;
  bool saw_decode = false;
  for (const auto &work : mixed->work) {
    saw_prefill |= work.kind == WorkKind::Prefill;
    saw_decode |= work.kind == WorkKind::Decode;
  }
  CHECK(saw_prefill);
  CHECK(saw_decode);
  handoff.retire_plan(mixed);
}

void test_profile_validation() {
  Handoff handoff(8);
  auto invalid = profile();
  invalid.tiers.front().runtime_scale_ns = 0;
  bool threw = false;
  try {
    ProxQPScheduler scheduler(handoff, 8, model(), hardware(), invalid);
  } catch (const std::invalid_argument &) { threw = true; }
  CHECK(threw);

  auto valid_profile = profile();
  ProxQPSchedulerConfig invalid_config;
  invalid_config.decode_urgency_steepness = 0;
  threw = false;
  try {
    ProxQPScheduler scheduler(handoff, 8, model(), hardware(), valid_profile,
                              invalid_config);
  } catch (const std::invalid_argument &) { threw = true; }
  CHECK(threw);
}

void test_profile_file_loads_runtime_and_scheduler_configuration() {
  const auto path = std::filesystem::temp_directory_path() /
                    "quickserve-proxqp-profile-test.conf";
  std::ofstream out(path);
  out << "schema_version=1\ncalibration_id=test-calibration\n"
         "ttft_target_ns=3000000000\ntpot_target_ns=250000000\n"
         "window_ns=45000000000\nruntime_weight=0.2\n"
         "rho_prefill=0.3\nrho_decode=0.4\nboundary_buffer_z=1.5\n"
         "boundary_weight=0.35\ntier_count=1\n"
         "tier.0.context_min=0\ntier.0.context_max=4096\n"
         "tier.0.tau_base_ns=1000\n"
         "tier.0.tau_prefill_ns_per_token=10\n"
         "tier.0.tau_decode_ns_per_item=100\n"
         "tier.0.runtime_margin_ns=200\n"
         "tier.0.runtime_scale_ns=500\n"
         "tier.0.token_capacity=64\ntier.0.sequence_capacity=16\n";
  out.close();
  const auto loaded = load_proxqp_policy_configuration(path);
  CHECK(loaded.profile.calibration_id == "test-calibration");
  CHECK(loaded.profile.tiers.size() == 1);
  CHECK(loaded.profile.tiers.front().runtime_margin_ns == 200);
  CHECK(loaded.config.ttft_target == std::chrono::seconds(3));
  CHECK(loaded.config.tpot_target == std::chrono::milliseconds(250));
  CHECK(loaded.config.window == std::chrono::seconds(45));
  CHECK(loaded.config.rho_decode == 0.4);
  std::filesystem::remove(path);
}

void test_profile_file_rejects_unknown_keys() {
  const auto path = std::filesystem::temp_directory_path() /
                    "quickserve-proxqp-profile-invalid.conf";
  std::ofstream(path) << "schema_version=1\nunknown_key=1\n";
  bool threw = false;
  try { (void)load_proxqp_policy_configuration(path); }
  catch (const std::invalid_argument &) { threw = true; }
  CHECK(threw);
  std::filesystem::remove(path);
}

void test_policy_builds_a_valid_prefill_plan_and_logs_decision() {
  Handoff handoff(8);
  ProxQPSchedulerConfig config;
  config.ttft_target = std::chrono::seconds(2);
  config.tpot_target = std::chrono::seconds(2);
  config.max_prefill_chunk = 4;
  ProxQPScheduler scheduler(handoff, 8, model(), hardware(), profile(), config);
  int decisions = 0;
  scheduler.set_decision_observer([&](const ProxQPDecisionRecord &r) {
    ++decisions;
    CHECK(r.projected_prefill > 0);
  });
  const auto id = scheduler.submit_synthetic(8, 2);
  scheduler.run_once();
  Admission admission;
  CHECK(handoff.try_take_admission(admission));
  CHECK(handoff.try_report_admission({id, 8, ErrorCode::None}));
  scheduler.run_once();
  Plan *plan = handoff.consume_plan();
  CHECK(plan != nullptr);
  CHECK(plan && !plan->work.empty());
  CHECK(decisions == 1);
  if (plan) handoff.retire_plan(plan);
}

void test_inactive_tpot_does_not_make_prefill_projection_empty() {
  Handoff handoff(8);
  auto slow = profile();
  slow.tiers.front().tau_base_ns = 300'000'000;
  slow.tiers.front().runtime_scale_ns = 100'000'000;
  ProxQPScheduler scheduler(handoff, 8, model(), hardware(), slow);
  std::string fallback;
  scheduler.set_decision_observer([&](const ProxQPDecisionRecord &r) {
    fallback = r.fallback_reason;
  });
  const auto id = scheduler.submit_synthetic(8, 2);
  scheduler.run_once();
  Admission admission;
  CHECK(handoff.try_take_admission(admission));
  CHECK(handoff.try_report_admission({id, 8, ErrorCode::None}));
  scheduler.run_once();
  Plan *plan = handoff.consume_plan();
  CHECK(plan != nullptr);
  CHECK(fallback.empty());
  if (plan) handoff.retire_plan(plan);
}

void test_impossible_runtime_uses_elastic_projection() {
  Handoff handoff(16);
  auto slow = mixed_profile();
  slow.tiers.front().tau_base_ns = 300'000'000;
  slow.tiers.front().runtime_scale_ns = 100'000'000;
  RequestState::TimePoint now{};
  ProxQPScheduler scheduler(handoff, 512, mixed_model(), hardware(), slow);
  scheduler.set_clock([&] { return now; });
  ProxQPDecisionRecord decision;
  scheduler.set_decision_observer([&](const ProxQPDecisionRecord &r) {
    decision = r;
  });
  const auto decode_id = scheduler.submit_synthetic(1, 3);
  const auto prefill_id = scheduler.submit_synthetic(512, 3);
  scheduler.run_once();
  Admission admission{};
  CHECK(handoff.try_take_admission(admission));
  CHECK(handoff.try_report_admission({decode_id, 1, ErrorCode::None}));
  CHECK(handoff.try_take_admission(admission));
  CHECK(handoff.try_report_admission({prefill_id, 512, ErrorCode::None}));
  scheduler.run_once();
  Plan *setup = handoff.consume_plan();
  CHECK(setup != nullptr);
  if (!setup) return;
  CHECK(handoff.try_report_completion(
      {decode_id, 1, 1, 7, WorkKind::Prefill, ErrorCode::None, true, false}));
  for (const auto &work : setup->work) {
    if (work.id == prefill_id && work.kind == WorkKind::Prefill) {
      CHECK(handoff.try_report_completion(
          {prefill_id, work.token_end, 0, 0, WorkKind::Prefill,
           ErrorCode::None, false, false}));
    }
  }
  handoff.retire_plan(setup);

  now += std::chrono::milliseconds(100);
  scheduler.run_once();
  Plan *plan = handoff.consume_plan();
  CHECK(plan != nullptr);
  if (!plan) return;
  CHECK(!plan->work.empty());
  CHECK(!decision.recovery);
  CHECK(decision.fallback_reason.empty());
  handoff.retire_plan(plan);
}
}

int main() {
  test_default_prefill_chunk_uses_full_token_budget();
  test_decode_urgency_is_finite_and_strictly_monotone();
  test_decode_reward_uses_mean_squared_urgency();
  test_smaller_chunk_allows_mixed_prefill_and_decode_plan();
  test_profile_validation();
  test_profile_file_loads_runtime_and_scheduler_configuration();
  test_profile_file_rejects_unknown_keys();
  test_policy_builds_a_valid_prefill_plan_and_logs_decision();
  test_inactive_tpot_does_not_make_prefill_projection_empty();
  test_impossible_runtime_uses_elastic_projection();
  if (failures) return 1;
  std::puts("all ProxQP scheduler checks passed");
  return 0;
}
