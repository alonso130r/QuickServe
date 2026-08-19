// Protocol tests for the scheduler <-> runtime handoff.
//
// The property under test is that a work unit is executed at most once. A plan
// may be dropped (the scheduler recomputes it from unadvanced state), but it
// must never be executed twice: decoding the same position twice corrupts a
// request's KV cache. Scheduler::step() enforces this by refusing to publish
// while a plan is outstanding.

#include "runtime/scheduler.cpp"

#include <cstdint>
#include <cstdio>
#include <set>
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
// Deliberately memoryless: every plan is derived from table_ alone.
class ChunkedProbePolicy : public Scheduler {
public:
  ChunkedProbePolicy(Handoff &handoff, std::uint32_t token_budget,
                     std::uint32_t chunk)
      : Scheduler(handoff, token_budget), chunk_(chunk) {}

protected:
  void create_reordering(Plan &out) override {
    for (std::uint32_t id = 0; id < table_.size(); ++id) {
      const RequestState &state = table_[id];
      if (state.stage == RequestState::Stage::Complete) {
        continue;
      }
      if (!state.prefill_done()) {
        const std::uint32_t end =
            std::min(state.prompt_len(), state.prefill_pos + chunk_);
        out.work.push_back({id, state.prefill_pos, end, SplitKind::Prefill});
      } else {
        const std::uint32_t pos = state.prefill_pos + state.decoded;
        out.work.push_back({id, pos, pos + 1, SplitKind::Decode});
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
    CHECK(in_flight_ == nullptr);
    in_flight_ = handoff.consume();
    if (in_flight_ == nullptr) {
      return;
    }
    ++plans_run_;
    for (const SplitRequest &work : in_flight_->work) {
      if (!executed_.insert({work.id, work.tok_begin}).second) {
        ++duplicates_;
      }
    }
  }

  // Reports every completion before retiring, per the Handoff contract.
  void finish(Handoff &handoff, const std::vector<RequestState> &table) {
    if (in_flight_ == nullptr) {
      return;
    }
    for (const SplitRequest &work : in_flight_->work) {
      Completion completion{};
      completion.id = work.id;
      completion.kind = work.kind;
      if (work.kind == SplitKind::Prefill) {
        completion.prefill_pos = work.tok_end;
        completion.decoded = table[work.id].decoded;
      } else {
        completion.prefill_pos = table[work.id].prefill_pos;
        completion.decoded = table[work.id].decoded + 1;
        completion.last_token = 42;
      }
      CHECK(handoff.report(completion));
    }
    handoff.retire(in_flight_);
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

Request make_request(std::size_t prompt_tokens) {
  Request request;
  request.num_tokens = prompt_tokens;
  request.tokenized_prompt.assign(prompt_tokens, 7);
  request.time_of_arrival = std::chrono::high_resolution_clock::now();
  return request;
}

// A request driven to completion when scheduler and runtime strictly alternate.
void test_interleaved_run_completes() {
  std::printf("test_interleaved_run_completes\n");
  Handoff handoff(/*plan_capacity=*/16);
  ChunkedProbePolicy scheduler(handoff, /*token_budget=*/512, /*chunk=*/4);
  const std::uint32_t id =
      scheduler.submit(make_request(10), /*max_output_tokens=*/3);

  FakeRuntime runtime;
  for (int i = 0; i < 20; ++i) {
    scheduler.step();
    runtime.run_once(handoff, scheduler.table());
  }

  const RequestState &state = scheduler.table()[id];
  CHECK(state.stage == RequestState::Stage::Complete);
  CHECK(state.prefill_pos == 10);
  CHECK(state.decoded == 3);
  CHECK(state.output.size() == 3);
  CHECK(runtime.duplicates() == 0);
}

// The regression this exists for. Stepping the scheduler while the runtime is
// mid-execution must not republish work already in flight: table_ has not
// advanced yet, so a memoryless policy would re-emit the identical decode.
void test_step_during_execution_does_not_duplicate() {
  std::printf("test_step_during_execution_does_not_duplicate\n");
  Handoff handoff(/*plan_capacity=*/16);
  ChunkedProbePolicy scheduler(handoff, /*token_budget=*/512, /*chunk=*/4);
  const std::uint32_t id =
      scheduler.submit(make_request(10), /*max_output_tokens=*/3);

  FakeRuntime runtime;
  for (int i = 0; i < 20; ++i) {
    scheduler.step();                        // publish
    runtime.start(handoff);                  // consume; now mid-execution
    scheduler.step();                        // must decline to publish
    runtime.finish(handoff, scheduler.table());
  }

  const RequestState &state = scheduler.table()[id];
  CHECK(runtime.duplicates() == 0);
  CHECK(state.stage == RequestState::Stage::Complete);
  CHECK(state.decoded == 3);
  // 3 prefill chunks (4/4/2) + 3 decodes, and nothing wasted.
  CHECK(runtime.plans_run() == 6);
}

// Several requests share iterations without their work bleeding into one
// another's positions.
void test_multiple_requests_do_not_duplicate() {
  std::printf("test_multiple_requests_do_not_duplicate\n");
  Handoff handoff(/*plan_capacity=*/16);
  ChunkedProbePolicy scheduler(handoff, /*token_budget=*/512, /*chunk=*/3);
  const std::uint32_t a = scheduler.submit(make_request(7), 2);
  const std::uint32_t b = scheduler.submit(make_request(2), 4);

  FakeRuntime runtime;
  for (int i = 0; i < 40; ++i) {
    scheduler.step();
    runtime.start(handoff);
    scheduler.step();
    runtime.finish(handoff, scheduler.table());
  }

  CHECK(runtime.duplicates() == 0);
  CHECK(scheduler.table()[a].stage == RequestState::Stage::Complete);
  CHECK(scheduler.table()[b].stage == RequestState::Stage::Complete);
  CHECK(scheduler.table()[a].decoded == 2);
  CHECK(scheduler.table()[b].decoded == 4);
}

// An idle scheduler must not hand the runtime empty plans to churn on.
void test_idle_scheduler_publishes_nothing() {
  std::printf("test_idle_scheduler_publishes_nothing\n");
  Handoff handoff(/*plan_capacity=*/16);
  ChunkedProbePolicy scheduler(handoff, /*token_budget=*/512, /*chunk=*/4);

  FakeRuntime runtime;
  for (int i = 0; i < 5; ++i) {
    scheduler.step();
    runtime.run_once(handoff, scheduler.table());
  }
  CHECK(runtime.plans_run() == 0);
}

// A completed request stays completed and generates no further work.
void test_completed_request_is_not_rescheduled() {
  std::printf("test_completed_request_is_not_rescheduled\n");
  Handoff handoff(/*plan_capacity=*/16);
  ChunkedProbePolicy scheduler(handoff, /*token_budget=*/512, /*chunk=*/8);
  scheduler.submit(make_request(4), /*max_output_tokens=*/1);

  FakeRuntime runtime;
  for (int i = 0; i < 10; ++i) {
    scheduler.step();
    runtime.run_once(handoff, scheduler.table());
  }
  const int settled = runtime.plans_run();

  for (int i = 0; i < 10; ++i) {
    scheduler.step();
    runtime.run_once(handoff, scheduler.table());
  }
  CHECK(runtime.plans_run() == settled);
  CHECK(runtime.duplicates() == 0);
}

} // namespace

int main() {
  test_interleaved_run_completes();
  test_step_during_execution_does_not_duplicate();
  test_multiple_requests_do_not_duplicate();
  test_idle_scheduler_publishes_nothing();
  test_completed_request_is_not_rescheduled();

  if (g_failures != 0) {
    std::printf("\n%d check(s) failed\n", g_failures);
    return 1;
  }
  std::printf("\nall checks passed\n");
  return 0;
}
