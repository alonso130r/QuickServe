#include "policies/heuristic_aimd.hpp"

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
  CHECK(scheduler.last_estimate().compute_operations > 0.0L);
  CHECK(scheduler.last_estimate().memory_bytes >= model.model_bytes);
  CHECK(scheduler.last_estimate().bandwidth_pressure < 1.0L);
  handoff.retire_plan(plan);
}

void test_default_prefill_chunk_uses_full_backend_capacity() {
  Handoff handoff(8);
  const ModelProfile model = test_profile();
  const HardwareProfile hardware = test_hardware_profile();
  HeuristicAIMDConfig config;
  config.initial_window = 512.0;
  HeuristicAIMD scheduler(handoff, 512, model, hardware, config);
  scheduler.submit_synthetic(1024, 2);
  admit(scheduler, handoff, {1024});

  Plan *plan = handoff.consume_plan();
  CHECK(plan != nullptr);
  if (!plan) return;
  CHECK(plan->work.size() == 1);
  CHECK(plan->work.front().token_count() == 512);
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

void test_aimd_uses_ewma_rate_and_separate_windows() {
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
  CHECK(scheduler.prefill_window() == 20.0);
  CHECK(scheduler.decode_window() == 4.0);

  finish_prefill_batch(std::chrono::microseconds(4100));
  CHECK(scheduler.prefill_window() == 36.0);
  CHECK(scheduler.decode_window() == 4.0);

  finish_prefill_batch(std::chrono::milliseconds(40));
  CHECK(scheduler.prefill_window() == 28.8);
  CHECK(scheduler.decode_window() == 4.0);
}

void test_scheduler_mixes_decode_and_prefill() {
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
  Plan *mixed = handoff.consume_plan();
  CHECK(mixed != nullptr);
  if (!mixed) return;
  CHECK(mixed->work.size() == 2);
  if (mixed->work.size() == 2) {
    CHECK(mixed->work[0].id == 0);
    CHECK(mixed->work[0].kind == WorkKind::Decode);
    CHECK(mixed->work[0].token_begin == 1);
    CHECK(mixed->work[0].token_end == 2);
    CHECK(mixed->work[1].id == 1);
    CHECK(mixed->work[1].kind == WorkKind::Prefill);
    CHECK(mixed->work[1].token_begin == 3);
    CHECK(mixed->work[1].token_end == 4);
  }
  CHECK(handoff.try_report_completion(
      {0, 1, 2, 8, WorkKind::Decode, ErrorCode::None, true, false}));
  CHECK(handoff.try_report_completion(
      {1, 4, 1, 9, WorkKind::Prefill, ErrorCode::None, true, false}));
  handoff.retire_plan(mixed);
  now += std::chrono::milliseconds(20);
  scheduler.run_once();
  CHECK(scheduler.prefill_window() == 4.0);
  CHECK(scheduler.decode_window() == 4.0);
}

void test_prefill_is_memory_capped_instead_of_skipped() {
  Handoff handoff(8);
  RequestState::TimePoint now{};
  const ModelProfile model = test_profile();
  HardwareProfile hardware = test_hardware_profile();
  constexpr std::uint64_t kv_bytes_per_token = 32ULL * 8 * 128 * 4;
  hardware.total_memory_bytes = 2 * kv_bytes_per_token;
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
  Plan *mixed = handoff.consume_plan();
  CHECK(mixed != nullptr);
  if (!mixed) return;
  CHECK(mixed->work.size() == 2);
  if (mixed->work.size() == 2) {
    CHECK(mixed->work[0].kind == WorkKind::Decode);
    CHECK(mixed->work[1].kind == WorkKind::Prefill);
    CHECK(mixed->work[1].token_count() == 1);
  }
  handoff.retire_plan(mixed);
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

} // namespace

int main() {
  test_scheduler_owns_an_immutable_profile_copy();
  test_empty_policy_does_not_publish_an_illegal_plan();
  test_scheduler_emits_bounded_prefill_work();
  test_default_prefill_chunk_uses_full_backend_capacity();
  test_oversized_window_still_respects_token_budget();
  test_aimd_uses_ewma_rate_and_separate_windows();
  test_scheduler_mixes_decode_and_prefill();
  test_prefill_is_memory_capped_instead_of_skipped();
  test_starved_prefill_eventually_beats_decode();
  if (failures != 0) {
    std::printf("%d check(s) failed\n", failures);
    return 1;
  }
  std::printf("all heuristic AIMD checks passed\n");
  return 0;
}
