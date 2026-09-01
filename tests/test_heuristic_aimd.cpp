#include "policies/heuristic_aimd.hpp"
#include "policies/batch_utilities.hpp"

#include <cstdio>
#include <chrono>
#include <vector>

namespace {

int failures = 0;

#define CHECK(condition)                                                       \
  do {                                                                         \
    if (!(condition)) {                                                        \
      std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #condition);       \
      ++failures;                                                              \
    }                                                                          \
  } while (false)

ModelProfile test_profile() {
  return ModelProfile{7'000'000'000ULL, 4'000'000'000ULL, 32, 4096, 32, 8,
                      128, 2.0, 2.0, 4.0 / 7.0, 8192, 512, 16};
}

HardwareProfile test_hardware_profile() {
  return HardwareProfile{"Mac15,7", 32ULL * 1024 * 1024 * 1024, 10, 10,
                         16'384};
}

HeuristicAIMDConfig test_config(double initial_window = 4.0,
                                std::chrono::milliseconds target =
                                    std::chrono::milliseconds(10)) {
  HeuristicAIMDConfig config;
  config.initial_window = initial_window;
  config.minimum_window = 1.0;
  config.maximum_window = 16.0;
  config.additive_step = 16.0;
  config.multiplicative_factor = 0.8;
  config.target_batch_duration = target;
  config.ttft_target = std::chrono::seconds(10);
  config.tpot_target = std::chrono::milliseconds(10);
  config.starvation_threshold = 2;
  config.max_consecutive_decode_batches = 2;
  config.max_prefill_chunk = 4;
  return config;
}

void admit(HeuristicAIMD &scheduler, Handoff &handoff,
           const std::vector<std::uint32_t> &prompt_lengths) {
  scheduler.run_once();
  for (RequestId id = 0; id < prompt_lengths.size(); ++id) {
    Admission admission{};
    CHECK(handoff.try_take_admission(admission));
    CHECK(handoff.try_report_admission(
        {id, prompt_lengths[id], ErrorCode::None}));
  }
  scheduler.run_once();
}

void test_scheduler_owns_an_immutable_profile_copy() {
  Handoff handoff(8);
  ModelProfile source = test_profile();
  HardwareProfile hardware = test_hardware_profile();
  HeuristicAIMD scheduler(handoff, 64, source, hardware);

  source.parameter_count = 1;
  source.context_capacity = 1;
  hardware.total_memory_bytes = 1;

  CHECK(scheduler.model_profile().parameter_count == 7'000'000'000ULL);
  CHECK(scheduler.model_profile().context_capacity == 8192);
  CHECK(scheduler.model_profile().head_dimension == 128);
  CHECK(scheduler.hardware_profile().model_identifier == "Mac15,7");
  CHECK(scheduler.hardware_profile().total_memory_bytes ==
        32ULL * 1024 * 1024 * 1024);
}

void test_shared_resource_accounting_matches_aimd_units() {
  std::list<RequestState> requests;
  RequestState request;
  request.id = 7;
  request.stage = RequestState::Stage::Prefill;
  request.prefill_position = 10;
  requests.push_back(request);
  const std::vector<WorkItem> work{{7, 10, 14, WorkKind::Prefill}};
  const auto usage = quickserve::policy::evaluate_batch_resources(
      requests, work, 64, test_profile(), test_hardware_profile());
  CHECK(usage.total_tokens == 4);
  CHECK(usage.work_items == 1);
  CHECK(usage.valid);
}

void test_empty_policy_does_not_publish_an_illegal_plan() {
  Handoff handoff(8);
  const ModelProfile source = test_profile();
  const HardwareProfile hardware = test_hardware_profile();
  HeuristicAIMD scheduler(handoff, 64, source, hardware);

  CHECK(!scheduler.run_once());
  CHECK(handoff.consume_plan() == nullptr);
}

void test_scheduler_emits_bounded_prefill_work() {
  Handoff handoff(16);
  const ModelProfile model = test_profile();
  const HardwareProfile hardware = test_hardware_profile();
  HeuristicAIMD scheduler(handoff, 8, model, hardware, test_config());
  scheduler.submit_synthetic(8, 2);
  scheduler.submit_synthetic(8, 2);
  admit(scheduler, handoff, {8, 8});

  Plan *plan = handoff.consume_plan();
  CHECK(plan != nullptr);
  if (!plan) return;
  CHECK(!plan->work.empty());
  std::uint32_t tokens = 0;
  for (const WorkItem &work : plan->work) {
    CHECK(work.kind == WorkKind::Prefill);
    CHECK(work.token_count() <= 4);
    tokens += work.token_count();
  }
  CHECK(tokens <= 8);
  CHECK(scheduler.last_feasibility().total_tokens == tokens);
  CHECK(scheduler.last_feasibility().required_memory_bytes >=
        model.model_bytes);
  CHECK(scheduler.last_feasibility().valid);
  handoff.retire_plan(plan);
}

void test_default_prefill_chunk_uses_full_backend_capacity() {
  Handoff handoff(8);
  const ModelProfile model = test_profile();
  const HardwareProfile hardware = test_hardware_profile();
  HeuristicAIMDConfig config;
  HeuristicAIMD scheduler(handoff, 512, model, hardware, config);
  scheduler.submit_synthetic(1024, 2);
  admit(scheduler, handoff, {1024});

  Plan *plan = handoff.consume_plan();
  CHECK(plan != nullptr);
  if (!plan) return;
  CHECK(plan->work.size() == 1);
  CHECK(plan->work.front().token_count() == 512);
  CHECK(scheduler.prefill_window() == 512.0);
  handoff.retire_plan(plan);
}

void test_oversized_window_still_respects_token_budget() {
  Handoff handoff(8);
  const ModelProfile model = test_profile();
  const HardwareProfile hardware = test_hardware_profile();
  HeuristicAIMDConfig config;
  config.initial_window = 1.0e20;
  config.maximum_window = 1.0e20;
  HeuristicAIMD scheduler(handoff, 4, model, hardware, config);
  scheduler.submit_synthetic(8, 2);
  admit(scheduler, handoff, {8});

  Plan *plan = handoff.consume_plan();
  CHECK(plan != nullptr);
  if (!plan) return;
  CHECK(plan->work.size() == 1);
  CHECK(plan->work.front().token_count() == 4);
  handoff.retire_plan(plan);
}

void test_prefill_capacity_stays_fixed_across_batch_timings() {
  Handoff handoff(8);
  RequestState::TimePoint now{};
  const ModelProfile model = test_profile();
  const HardwareProfile hardware = test_hardware_profile();
  HeuristicAIMDConfig config = test_config();
  config.maximum_window = 64.0;
  HeuristicAIMD scheduler(handoff, 8, model, hardware, config);
  scheduler.set_clock([&] { return now; });
  scheduler.submit_synthetic(24, 2);
  admit(scheduler, handoff, {24});

  auto finish_prefill_batch = [&](std::chrono::nanoseconds duration) {
    Plan *plan = handoff.consume_plan();
    CHECK(plan != nullptr);
    if (!plan) return;
    const WorkItem work = plan->work.front();
    CHECK(handoff.try_report_completion(
        {work.id, work.token_end, 0, 0, WorkKind::Prefill,
         ErrorCode::None, false, false}));
    handoff.retire_plan(plan);
    now += duration;
    scheduler.run_once();
  };

  finish_prefill_batch(std::chrono::milliseconds(4));
  CHECK(scheduler.prefill_window() == 4.0);
  CHECK(scheduler.decode_window() == 4.0);

  finish_prefill_batch(std::chrono::milliseconds(4));
  CHECK(scheduler.prefill_window() == 4.0);
  CHECK(scheduler.decode_window() == 4.0);

  finish_prefill_batch(std::chrono::microseconds(4100));
  CHECK(scheduler.prefill_window() == 4.0);
  CHECK(scheduler.decode_window() == 4.0);

  finish_prefill_batch(std::chrono::milliseconds(40));
  CHECK(scheduler.prefill_window() == 4.0);
  CHECK(scheduler.decode_window() == 4.0);
}

void test_scheduler_keeps_decode_and_prefill_in_separate_batches() {
  Handoff handoff(8);
  RequestState::TimePoint now{};
  const ModelProfile model = test_profile();
  const HardwareProfile hardware = test_hardware_profile();
  HeuristicAIMD scheduler(handoff, 4, model, hardware,
                          test_config(4.0, std::chrono::milliseconds(100)));
  scheduler.set_clock([&] { return now; });
  scheduler.submit_synthetic(1, 3);
  scheduler.submit_synthetic(4, 3);
  admit(scheduler, handoff, {1, 4});

  Plan *first = handoff.consume_plan();
  CHECK(first != nullptr);
  if (!first) return;
  CHECK(first->work.size() == 2);
  CHECK(handoff.try_report_completion(
      {0, 1, 1, 7, WorkKind::Prefill, ErrorCode::None, true, false}));
  CHECK(handoff.try_report_completion(
      {1, 3, 0, 0, WorkKind::Prefill, ErrorCode::None, false, false}));
  handoff.retire_plan(first);

  now += std::chrono::milliseconds(20);
  scheduler.run_once();
  Plan *decode = handoff.consume_plan();
  CHECK(decode != nullptr);
  if (!decode) return;
  CHECK(decode->work.size() == 1);
  CHECK(decode->work.front().id == 0);
  CHECK(decode->work.front().kind == WorkKind::Decode);
  handoff.retire_plan(decode);
}

void test_feasibility_accounts_for_model_and_resident_kv_memory() {
  Handoff handoff(8);
  RequestState::TimePoint now{};
  constexpr std::uint64_t kv_bytes_per_token = 32ULL * 8 * 128 * 4;
  ModelProfile model = test_profile();
  model.model_bytes = 10 * kv_bytes_per_token;
  HardwareProfile hardware = test_hardware_profile();
  hardware.total_memory_bytes = model.model_bytes + 3 * kv_bytes_per_token;
  HeuristicAIMD scheduler(handoff, 4, model, hardware, test_config());
  scheduler.set_clock([&] { return now; });
  scheduler.submit_synthetic(1, 3);
  scheduler.submit_synthetic(4, 3);
  admit(scheduler, handoff, {1, 4});

  Plan *setup = handoff.consume_plan();
  CHECK(setup != nullptr);
  if (!setup) return;
  CHECK(setup->work.size() == 2);
  CHECK(handoff.try_report_completion(
      {0, 1, 1, 7, WorkKind::Prefill, ErrorCode::None, true, false}));
  CHECK(handoff.try_report_completion(
      {1, 1, 0, 0, WorkKind::Prefill, ErrorCode::None, false, false}));
  handoff.retire_plan(setup);

  now += std::chrono::milliseconds(1);
  scheduler.run_once();
  Plan *decode = handoff.consume_plan();
  CHECK(decode != nullptr);
  if (!decode) return;
  CHECK(decode->work.size() == 1);
  CHECK(decode->work.front().kind == WorkKind::Decode);
  CHECK(scheduler.last_feasibility().resident_kv_bytes ==
        3 * kv_bytes_per_token);
  CHECK(scheduler.last_feasibility().required_memory_bytes ==
        hardware.total_memory_bytes);
  handoff.retire_plan(decode);
}

void test_prefill_runs_after_bounded_decode_burst() {
  Handoff handoff(8);
  RequestState::TimePoint now{};
  const ModelProfile model = test_profile();
  const HardwareProfile hardware = test_hardware_profile();
  HeuristicAIMDConfig config = test_config(4.0);
  config.max_consecutive_decode_batches = 2;
  HeuristicAIMD scheduler(handoff, 4, model, hardware, config);
  scheduler.set_clock([&] { return now; });
  scheduler.submit_synthetic(1, 8);
  scheduler.submit_synthetic(4, 8);
  admit(scheduler, handoff, {1, 4});

  Plan *setup = handoff.consume_plan();
  CHECK(setup != nullptr);
  if (!setup) return;
  CHECK(handoff.try_report_completion(
      {0, 1, 1, 7, WorkKind::Prefill, ErrorCode::None, true, false}));
  CHECK(handoff.try_report_completion(
      {1, 3, 0, 0, WorkKind::Prefill, ErrorCode::None, false, false}));
  handoff.retire_plan(setup);

  std::uint32_t decoded = 1;
  for (int batch = 0; batch < 2; ++batch) {
    now += std::chrono::milliseconds(20);
    scheduler.run_once();
    Plan *decode = handoff.consume_plan();
    CHECK(decode != nullptr);
    if (!decode) return;
    CHECK(decode->work.size() == 1);
    CHECK(decode->work.front().kind == WorkKind::Decode);
    ++decoded;
    CHECK(handoff.try_report_completion(
        {0, 1, decoded, 7, WorkKind::Decode, ErrorCode::None, true, false}));
    handoff.retire_plan(decode);
  }

  now += std::chrono::milliseconds(20);
  scheduler.run_once();
  Plan *prefill = handoff.consume_plan();
  CHECK(prefill != nullptr);
  if (!prefill) return;
  CHECK(prefill->work.size() == 1);
  CHECK(prefill->work.front().id == 1);
  CHECK(prefill->work.front().kind == WorkKind::Prefill);
  handoff.retire_plan(prefill);
}

void test_starved_prefill_eventually_beats_decode() {
  Handoff handoff(8);
  RequestState::TimePoint now{};
  HeuristicAIMDConfig config = test_config(1.1);
  config.maximum_window = 1.1;
  config.additive_step = 0.5;
  config.ttft_target = std::chrono::hours(1);
  config.tpot_target = std::chrono::hours(1);
  const ModelProfile model = test_profile();
  const HardwareProfile hardware = test_hardware_profile();
  HeuristicAIMD scheduler(handoff, 4, model, hardware, config);
  scheduler.set_clock([&] { return now; });
  scheduler.submit_synthetic(1, 8);
  scheduler.submit_synthetic(4, 8);
  admit(scheduler, handoff, {1, 4});

  Plan *setup = handoff.consume_plan();
  CHECK(setup != nullptr);
  if (!setup) return;
  CHECK(setup->work.size() == 1);
  CHECK(setup->work.front().id == 0);
  CHECK(handoff.try_report_completion(
      {0, 1, 1, 7, WorkKind::Prefill, ErrorCode::None, true, false}));
  handoff.retire_plan(setup);

  std::uint32_t decoded = 1;
  bool prefill_selected = false;
  for (int round = 0; round < 3; ++round) {
    now += std::chrono::milliseconds(1);
    scheduler.run_once();
    Plan *plan = handoff.consume_plan();
    CHECK(plan != nullptr);
    if (!plan) return;
    for (const WorkItem &work : plan->work) {
      if (work.id == 1) prefill_selected = true;
    }
    if (prefill_selected) {
      handoff.retire_plan(plan);
      break;
    }
    const WorkItem work = plan->work.front();
    CHECK(work.id == 0);
    ++decoded;
    CHECK(handoff.try_report_completion(
        {0, 1, decoded, 7, WorkKind::Decode, ErrorCode::None, true, false}));
    handoff.retire_plan(plan);
  }
  CHECK(prefill_selected);
}

void test_started_prefill_keeps_locality_until_completion() {
  Handoff handoff(8);
  const ModelProfile model = test_profile();
  const HardwareProfile hardware = test_hardware_profile();
  HeuristicAIMD scheduler(handoff, 4, model, hardware, test_config());
  scheduler.submit_synthetic(8, 2);
  scheduler.submit_synthetic(8, 2);
  admit(scheduler, handoff, {8, 8});

  Plan *first = handoff.consume_plan();
  CHECK(first != nullptr);
  if (!first) return;
  CHECK(first->work.size() == 1);
  CHECK(first->work.front().id == 0);
  CHECK(handoff.try_report_completion(
      {0, 4, 0, 0, WorkKind::Prefill, ErrorCode::None, false, false}));
  handoff.retire_plan(first);

  scheduler.run_once();
  Plan *second = handoff.consume_plan();
  CHECK(second != nullptr);
  if (!second) return;
  CHECK(second->work.size() == 1);
  CHECK(second->work.front().id == 0);
  CHECK(second->work.front().token_begin == 4);
  CHECK(second->work.front().token_end == 8);
  handoff.retire_plan(second);
}

} // namespace

int main() {
  test_shared_resource_accounting_matches_aimd_units();
  test_scheduler_owns_an_immutable_profile_copy();
  test_empty_policy_does_not_publish_an_illegal_plan();
  test_scheduler_emits_bounded_prefill_work();
  test_default_prefill_chunk_uses_full_backend_capacity();
  test_oversized_window_still_respects_token_budget();
  test_prefill_capacity_stays_fixed_across_batch_timings();
  test_scheduler_keeps_decode_and_prefill_in_separate_batches();
  test_feasibility_accounts_for_model_and_resident_kv_memory();
  test_prefill_runs_after_bounded_decode_burst();
  test_starved_prefill_eventually_beats_decode();
  test_started_prefill_keeps_locality_until_completion();
  if (failures != 0) {
    std::printf("%d check(s) failed\n", failures);
    return 1;
  }
  std::printf("all heuristic AIMD checks passed\n");
  return 0;
}
