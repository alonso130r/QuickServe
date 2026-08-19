// Protocol tests for the scheduler <-> runtime handoff.
//
// The property under test is that a work unit is executed at most once. A plan
// may be dropped (the scheduler recomputes it from unadvanced state), but it
// must never be executed twice: decoding the same position twice corrupts a
// request's KV cache. Scheduler::run_once() enforces this by refusing to publish
// while a plan is outstanding.

#include "runtime/handoff.hpp"
#include "runtime/scheduler.hpp"

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <set>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace {

int g_failures = 0;

// Not assert(): tests must keep their teeth under NDEBUG.
#define CHECK(cond)                                                            \
  do {                                                                         \
    if (!(cond)) {                                                             \
      std::printf("  FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);            \
      ++g_failures;                                                            \
    }                                                                          \
  } while (0)

// A policy that prefills in fixed chunks, then decodes one token per iteration.
// Deliberately memoryless: every plan is derived from requests_ alone.
class ChunkedProbePolicy : public Scheduler {
public:
  ChunkedProbePolicy(Handoff &handoff, std::uint32_t token_budget,
                     std::uint32_t chunk)
      : Scheduler(handoff, token_budget), chunk_(chunk) {}

protected:
  void build_plan(Plan &out) override {
    const auto &states = policy_requests();
    for (std::uint32_t id = 0; id < states.size(); ++id) {
      const RequestState &state = states[id];
      if (state.stage == RequestState::Stage::Terminal ||
          state.stage == RequestState::Stage::PendingAdmission ||
          state.stage == RequestState::Stage::PendingRelease) {
        continue;
      }
      if (!state.prefill_done()) {
        const std::uint32_t end =
            std::min(state.prompt_length,
                     state.prefill_position + chunk_);
        out.work.push_back(
            {id, state.prefill_position, end, WorkKind::Prefill});
      } else {
        const std::uint32_t pos =
            state.prompt_length + state.decoded_count - 1;
        out.work.push_back({id, pos, pos + 1, WorkKind::Decode});
      }
    }
  }

private:
  std::uint32_t chunk_;
};

// Stand-in for the runtime thread. start() and finish() are separate so a test
// can slip a scheduler step into the window where a plan is mid-execution --
// which is the only interleaving that exposes duplicate execution.
class FakeRuntime {
public:
  void start(Handoff &handoff) {
    Admission admission{};
    while (handoff.try_take_admission(admission)) {
      CHECK(handoff.try_report_admission(AdmissionResult{
          admission.id, static_cast<std::uint32_t>(admission.prompt.size()),
          ErrorCode::None}));
    }
    Release release{};
    while (handoff.try_take_release(release)) {
      CHECK(handoff.try_acknowledge_release(ReleaseAck{release.id}));
    }

    CHECK(in_flight_ == nullptr);
    in_flight_ = handoff.consume_plan();
    if (in_flight_ == nullptr) {
      return;
    }
    ++plans_run_;
    for (const WorkItem &work : in_flight_->work) {
      if (!executed_.insert({work.id, work.token_begin}).second) {
        ++duplicates_;
      }
    }
  }

  // Reports every completion before retiring, per the Handoff contract.
  void finish(Handoff &handoff, const std::vector<RequestState> &table) {
    if (in_flight_ == nullptr) {
      return;
    }
    for (const WorkItem &work : in_flight_->work) {
      Completion completion{};
      completion.id = work.id;
      completion.kind = work.kind;
      if (work.kind == WorkKind::Prefill) {
        completion.prefill_position = work.token_end;
        completion.decoded_tokens = table[work.id].decoded_count;
        if (work.token_end == table[work.id].prompt_length) {
          completion.decoded_tokens += 1;
          completion.token = 42;
          completion.generated_token = true;
        }
      } else {
        completion.prefill_position = table[work.id].prefill_position;
        completion.decoded_tokens = table[work.id].decoded_count + 1;
        completion.token = 42;
        completion.generated_token = true;
      }
      CHECK(handoff.try_report_completion(completion));
    }
    handoff.retire_plan(in_flight_);
    in_flight_ = nullptr;
  }

  void run_once(Handoff &handoff, const std::vector<RequestState> &table) {
    start(handoff);
    finish(handoff, table);
  }

  [[nodiscard]] int duplicates() const { return duplicates_; }
  [[nodiscard]] int plans_run() const { return plans_run_; }

private:
  std::set<std::pair<std::uint32_t, std::uint32_t>> executed_;
  Plan *in_flight_ = nullptr;
  int duplicates_ = 0;
  int plans_run_ = 0;
};

std::string make_prompt(std::size_t prompt_tokens) {
  return std::string(prompt_tokens, 'x');
}

// A request driven to completion when scheduler and runtime strictly alternate.
void test_interleaved_run_completes() {
  std::printf("test_interleaved_run_completes\n");
  Handoff handoff(/*plan_capacity=*/16);
  ChunkedProbePolicy scheduler(handoff, /*token_budget=*/512, /*chunk=*/4);
  const std::uint32_t id =
      scheduler.submit(make_prompt(10), /*max_output_tokens=*/3);

  FakeRuntime runtime;
  for (int i = 0; i < 20; ++i) {
    scheduler.run_once();
    runtime.run_once(handoff, scheduler.requests());
  }

  const RequestState &state = scheduler.requests()[id];
  CHECK(state.stage == RequestState::Stage::Terminal);
  CHECK(state.prefill_position == 10);
  CHECK(state.decoded_count == 3);
  CHECK(state.output_token_ids.size() == 3);
  CHECK(runtime.duplicates() == 0);
}

// The regression this exists for. Stepping the scheduler while the runtime is
// mid-execution must not republish work already in flight: requests_ has not
// advanced yet, so a memoryless policy would re-emit the identical decode.
void test_step_during_execution_does_not_duplicate() {
  std::printf("test_step_during_execution_does_not_duplicate\n");
  Handoff handoff(/*plan_capacity=*/16);
  ChunkedProbePolicy scheduler(handoff, /*token_budget=*/512, /*chunk=*/4);
  const std::uint32_t id =
      scheduler.submit(make_prompt(10), /*max_output_tokens=*/3);

  FakeRuntime runtime;
  for (int i = 0; i < 20; ++i) {
    scheduler.run_once();                    // publish
    runtime.start(handoff);                  // consume; now mid-execution
    scheduler.run_once();                    // must decline to publish
    runtime.finish(handoff, scheduler.requests());
  }

  const RequestState &state = scheduler.requests()[id];
  CHECK(runtime.duplicates() == 0);
  CHECK(state.stage == RequestState::Stage::Terminal);
  CHECK(state.decoded_count == 3);
  // 3 prefill chunks (4/4/2, with the final chunk sampling the first token)
  // + 2 later decodes, and nothing wasted.
  CHECK(runtime.plans_run() == 5);
}

// Several requests share iterations without their work bleeding into one
// another's positions.
void test_multiple_requests_do_not_duplicate() {
  std::printf("test_multiple_requests_do_not_duplicate\n");
  Handoff handoff(/*plan_capacity=*/16);
  ChunkedProbePolicy scheduler(handoff, /*token_budget=*/512, /*chunk=*/3);
  const std::uint32_t a = scheduler.submit(make_prompt(7), 2);
  const std::uint32_t b = scheduler.submit(make_prompt(2), 4);

  FakeRuntime runtime;
  for (int i = 0; i < 40; ++i) {
    scheduler.run_once();
    runtime.start(handoff);
    scheduler.run_once();
    runtime.finish(handoff, scheduler.requests());
  }

  CHECK(runtime.duplicates() == 0);
  CHECK(scheduler.requests()[a].stage == RequestState::Stage::Terminal);
  CHECK(scheduler.requests()[b].stage == RequestState::Stage::Terminal);
  CHECK(scheduler.requests()[a].decoded_count == 2);
  CHECK(scheduler.requests()[b].decoded_count == 4);
}

// An idle scheduler must not hand the runtime empty plans to churn on.
void test_idle_scheduler_publishes_nothing() {
  std::printf("test_idle_scheduler_publishes_nothing\n");
  Handoff handoff(/*plan_capacity=*/16);
  ChunkedProbePolicy scheduler(handoff, /*token_budget=*/512, /*chunk=*/4);

  FakeRuntime runtime;
  for (int i = 0; i < 5; ++i) {
    scheduler.run_once();
    runtime.run_once(handoff, scheduler.requests());
  }
  CHECK(runtime.plans_run() == 0);
}

// A completed request stays completed and generates no further work.
void test_completed_request_is_not_rescheduled() {
  std::printf("test_completed_request_is_not_rescheduled\n");
  Handoff handoff(/*plan_capacity=*/16);
  ChunkedProbePolicy scheduler(handoff, /*token_budget=*/512, /*chunk=*/8);
  scheduler.submit(make_prompt(4), /*max_output_tokens=*/1);

  FakeRuntime runtime;
  for (int i = 0; i < 10; ++i) {
    scheduler.run_once();
    runtime.run_once(handoff, scheduler.requests());
  }
  const int settled = runtime.plans_run();

  for (int i = 0; i < 10; ++i) {
    scheduler.run_once();
    runtime.run_once(handoff, scheduler.requests());
  }
  CHECK(runtime.plans_run() == settled);
  CHECK(runtime.duplicates() == 0);
}

void test_admission_round_trip() {
  std::printf("test_admission_round_trip\n");
  Handoff handoff(/*plan_capacity=*/4);
  Admission admission{17, "owned prompt", 23};

  CHECK(handoff.try_admit(std::move(admission)));

  Admission received{};
  CHECK(handoff.try_take_admission(received));
  CHECK(received.id == 17);
  CHECK(received.prompt == "owned prompt");
  CHECK(received.max_output_tokens == 23);
  CHECK(!handoff.try_take_admission(received));
}

void test_admission_result_round_trip() {
  std::printf("test_admission_result_round_trip\n");
  Handoff handoff(/*plan_capacity=*/4);
  const AdmissionResult result{19, 31, ErrorCode::None};

  CHECK(handoff.try_report_admission(result));

  AdmissionResult received{};
  CHECK(handoff.try_take_admission_result(received));
  CHECK(received.id == 19);
  CHECK(received.prompt_tokens == 31);
  CHECK(received.error == ErrorCode::None);
  CHECK(!handoff.try_take_admission_result(received));
}

void test_owned_output_piece_round_trip() {
  std::printf("test_owned_output_piece_round_trip\n");
  Handoff handoff(/*plan_capacity=*/4);
  OutputPiece piece{29, 41, "multibyte: \xE2\x98\x83"};

  CHECK(handoff.try_report_output(std::move(piece)));

  OutputPiece received{};
  CHECK(handoff.try_take_output(received));
  CHECK(received.id == 29);
  CHECK(received.token == 41);
  CHECK(received.text == "multibyte: \xE2\x98\x83");
  CHECK(!handoff.try_take_output(received));
}

void test_release_and_acknowledgement_round_trip() {
  std::printf("test_release_and_acknowledgement_round_trip\n");
  Handoff handoff(/*plan_capacity=*/4);

  CHECK(handoff.try_release(Release{7}));
  Release release{};
  CHECK(handoff.try_take_release(release));
  CHECK(release.id == 7);

  CHECK(handoff.try_acknowledge_release(ReleaseAck{7}));
  ReleaseAck ack{};
  CHECK(handoff.try_take_release_ack(ack));
  CHECK(ack.id == 7);
}

void test_run_fatal_round_trip() {
  std::printf("test_run_fatal_round_trip\n");
  Handoff handoff(/*plan_capacity=*/4);

  CHECK(handoff.try_report_fatal(RunFatal{ErrorCode::DecodeFailed}));
  RunFatal fatal{};
  CHECK(handoff.try_take_fatal(fatal));
  CHECK(fatal.error == ErrorCode::DecodeFailed);
}

void test_completion_precedes_completed_epoch() {
  std::printf("test_completion_precedes_completed_epoch\n");
  Handoff handoff(/*plan_capacity=*/4);
  Plan &building = handoff.begin();
  building.work.push_back({3, 0, 2, WorkKind::Prefill});
  const std::uint64_t epoch = handoff.commit();
  Plan *plan = handoff.consume_plan();
  CHECK(plan != nullptr);

  const Completion completion{3, 2, 0, 0, WorkKind::Prefill,
                              ErrorCode::None, false, false};
  CHECK(handoff.try_report_completion(completion));
  CHECK(handoff.completed_epoch() < epoch);

  Completion received{};
  CHECK(handoff.try_take_completion(received));
  CHECK(received.id == 3);
  CHECK(received.prefill_position == 2);
  CHECK(handoff.completed_epoch() < epoch);

  handoff.retire_plan(plan);
  CHECK(handoff.completed_epoch() == epoch);
}

void test_two_threads_observe_ordered_completions_before_epochs() {
  std::printf("test_two_threads_observe_ordered_completions_before_epochs\n");
  constexpr std::uint32_t kPlans = 128;
  Handoff handoff(/*plan_capacity=*/1);
  std::atomic<bool> environment_failed{false};

  std::thread environment([&] {
    for (std::uint32_t i = 1; i <= kPlans; ++i) {
      Plan *plan = nullptr;
      while ((plan = handoff.consume_plan()) == nullptr) {
        std::this_thread::yield();
      }

      if (plan->work.size() != 1) {
        environment_failed.store(true, std::memory_order_release);
        handoff.retire_plan(plan);
        continue;
      }
      if (plan->work.front().id != i) {
        environment_failed.store(true, std::memory_order_release);
      }
      const WorkItem &work = plan->work.front();
      const Completion completion{work.id,
                                  work.token_end,
                                  0,
                                  0,
                                  WorkKind::Prefill,
                                  ErrorCode::None,
                                  false,
                                  false};
      if (!handoff.try_report_completion(completion)) {
        environment_failed.store(true, std::memory_order_release);
      }
      handoff.retire_plan(plan);
    }
  });

  for (std::uint32_t i = 1; i <= kPlans; ++i) {
    Plan &plan = handoff.begin();
    plan.work.push_back({i, i - 1, i, WorkKind::Prefill});
    const std::uint64_t epoch = handoff.commit();

    // The acquire observes retire_plan's release only after the environment
    // has published this epoch's completion to its SPSC channel.
    while (handoff.completed_epoch() < epoch) {
      std::this_thread::yield();
    }
    Completion completion{};
    CHECK(handoff.try_take_completion(completion));
    CHECK(completion.id == i);
    CHECK(completion.prefill_position == i);
    CHECK(!handoff.try_take_completion(completion));
  }

  environment.join();
  CHECK(!environment_failed.load(std::memory_order_acquire));
}

void test_full_channel_preserves_messages_and_rejected_owner() {
  std::printf("test_full_channel_preserves_messages_and_rejected_owner\n");
  Handoff handoff(/*plan_capacity=*/4, /*pool_size=*/2,
                  /*queue_capacity=*/2);
  Admission first{1, "first", 1};
  Admission second{2, "second", 2};
  Admission rejected{3, "must remain owned", 3};

  CHECK(handoff.try_admit(std::move(first)));
  CHECK(handoff.try_admit(std::move(second)));
  CHECK(!handoff.try_admit(std::move(rejected)));
  CHECK(rejected.id == 3);
  CHECK(rejected.prompt == "must remain owned");
  CHECK(rejected.max_output_tokens == 3);

  Admission received{};
  CHECK(handoff.try_take_admission(received));
  CHECK(received.id == 1);
  CHECK(received.prompt == "first");
  CHECK(handoff.try_take_admission(received));
  CHECK(received.id == 2);
  CHECK(received.prompt == "second");
  CHECK(!handoff.try_take_admission(received));

  OutputPiece output_first{4, 40, "output first"};
  OutputPiece output_second{5, 50, "output second"};
  OutputPiece output_rejected{6, 60, "output must remain owned"};
  CHECK(handoff.try_report_output(std::move(output_first)));
  CHECK(handoff.try_report_output(std::move(output_second)));
  CHECK(!handoff.try_report_output(std::move(output_rejected)));
  CHECK(output_rejected.id == 6);
  CHECK(output_rejected.token == 60);
  CHECK(output_rejected.text == "output must remain owned");

  OutputPiece output_received{};
  CHECK(handoff.try_take_output(output_received));
  CHECK(output_received.id == 4);
  CHECK(output_received.text == "output first");
  CHECK(handoff.try_take_output(output_received));
  CHECK(output_received.id == 5);
  CHECK(output_received.text == "output second");
  CHECK(!handoff.try_take_output(output_received));
}

void test_stop_is_monotonic() {
  std::printf("test_stop_is_monotonic\n");
  Handoff handoff(/*plan_capacity=*/4);
  CHECK(!handoff.stop_requested());

  std::atomic<bool> environment_started{false};
  std::atomic<bool> environment_observed_stop{false};
  std::thread environment([&] {
    environment_started.store(true, std::memory_order_release);
    while (!handoff.stop_requested()) {
      std::this_thread::yield();
    }
    environment_observed_stop.store(true, std::memory_order_release);
  });
  while (!environment_started.load(std::memory_order_acquire)) {
    std::this_thread::yield();
  }

  handoff.request_stop();
  environment.join();
  CHECK(handoff.stop_requested());
  CHECK(environment_observed_stop.load(std::memory_order_acquire));

  // No operation can clear the flag after both threads have observed it.
  Admission admission{5, "after stop", 1};
  CHECK(handoff.try_admit(std::move(admission)));
  Admission received{};
  CHECK(handoff.try_take_admission(received));
  CHECK(handoff.stop_requested());
}

void test_empty_plan_is_never_published() {
  std::printf("test_empty_plan_is_never_published\n");
  Handoff handoff(/*plan_capacity=*/4);
  Plan &plan = handoff.begin();
  CHECK(plan.work.empty());
  CHECK(handoff.commit() == 0);
  CHECK(handoff.consume_plan() == nullptr);
  CHECK(handoff.completed_epoch() == 0);
}

void test_invalid_queue_capacity_is_rejected_safely() {
  std::printf("test_invalid_queue_capacity_is_rejected_safely\n");
  for (const std::size_t capacity : {std::size_t{0}, std::size_t{1},
                                     std::size_t{3}}) {
    bool queue_threw = false;
    try {
      SPSCQueue<int> queue(capacity);
      (void)queue;
    } catch (const std::invalid_argument &) {
      queue_threw = true;
    }
    CHECK(queue_threw);
  }

  bool threw = false;
  try {
    Handoff handoff(/*plan_capacity=*/4, /*pool_size=*/2,
                    /*queue_capacity=*/3);
    (void)handoff;
  } catch (const std::invalid_argument &) {
    threw = true;
  }
  CHECK(threw);
}

void test_invalid_plan_pool_size_is_rejected_safely() {
  std::printf("test_invalid_plan_pool_size_is_rejected_safely\n");
  bool too_small_threw = false;
  try {
    Handoff handoff(/*plan_capacity=*/4, /*pool_size=*/1,
                    /*queue_capacity=*/4);
    (void)handoff;
  } catch (const std::invalid_argument &) {
    too_small_threw = true;
  }
  CHECK(too_small_threw);

  bool too_large_threw = false;
  try {
    Handoff handoff(/*plan_capacity=*/4, /*pool_size=*/8,
                    /*queue_capacity=*/4);
    (void)handoff;
  } catch (const std::invalid_argument &) {
    too_large_threw = true;
  }
  CHECK(too_large_threw);
}

// This is only a transport harness: it proves the Chunk 1 channels carry a
// fatal in one direction and releases for every supplied ID in the other. The
// production Scheduler reaction belongs to Chunk 2 and is not modeled here.
void transport_fatal_then_send_releases(
    Handoff &handoff, const std::vector<RequestId> &active_ids) {
  RunFatal fatal{};
  if (!handoff.try_take_fatal(fatal)) {
    return;
  }
  for (const RequestId id : active_ids) {
    CHECK(handoff.try_release(Release{id}));
  }
}

void test_transport_carries_fatal_and_releases_for_all_active_ids() {
  std::printf(
      "test_transport_carries_fatal_and_releases_for_all_active_ids\n");
  Handoff handoff(/*plan_capacity=*/4);
  CHECK(handoff.try_report_fatal(RunFatal{ErrorCode::DecodeFailed}));

  // The supplied set includes an ID absent from the hypothetical failed plan;
  // Handoff transports both releases without interpreting either ID.
  transport_fatal_then_send_releases(handoff, {11, 12});
  Release release{};
  CHECK(handoff.try_take_release(release));
  CHECK(release.id == 11);
  CHECK(handoff.try_take_release(release));
  CHECK(release.id == 12);
  CHECK(!handoff.try_take_release(release));
}

} // namespace

int main() {
  test_interleaved_run_completes();
  test_step_during_execution_does_not_duplicate();
  test_multiple_requests_do_not_duplicate();
  test_idle_scheduler_publishes_nothing();
  test_completed_request_is_not_rescheduled();
  test_admission_round_trip();
  test_admission_result_round_trip();
  test_owned_output_piece_round_trip();
  test_release_and_acknowledgement_round_trip();
  test_run_fatal_round_trip();
  test_completion_precedes_completed_epoch();
  test_two_threads_observe_ordered_completions_before_epochs();
  test_full_channel_preserves_messages_and_rejected_owner();
  test_stop_is_monotonic();
  test_empty_plan_is_never_published();
  test_invalid_queue_capacity_is_rejected_safely();
  test_invalid_plan_pool_size_is_rejected_safely();
  test_transport_carries_fatal_and_releases_for_all_active_ids();

  if (g_failures != 0) {
    std::printf("\n%d check(s) failed\n", g_failures);
    return 1;
  }
  std::printf("\nall checks passed\n");
  return 0;
}
