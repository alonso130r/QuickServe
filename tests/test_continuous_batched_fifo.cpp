#include "policies/continuous_batched_fifo.hpp"

#include <cstdio>
#include <vector>

namespace {

int failures = 0;

#define CHECK(condition)                                                       \
  do {                                                                         \
    if (!(condition)) {                                                        \
      std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #condition);         \
      ++failures;                                                              \
    }                                                                          \
  } while (false)

void admit(ContinuousBatchedFifoScheduler &scheduler, Handoff &handoff,
           const std::vector<std::uint32_t> &lengths) {
  scheduler.run_once();
  for (RequestId id = 0; id < lengths.size(); ++id) {
    Admission admission{};
    CHECK(handoff.try_take_admission(admission));
    CHECK(admission.id == id);
    CHECK(handoff.try_report_admission(
        AdmissionResult{id, lengths[id], ErrorCode::None}));
  }
  scheduler.run_once();
}

void check_work(const WorkItem &work, RequestId id, std::uint32_t begin,
                std::uint32_t end, WorkKind kind) {
  CHECK(work.id == id);
  CHECK(work.token_begin == begin);
  CHECK(work.token_end == end);
  CHECK(work.kind == kind);
}

void test_fills_budget_in_strict_fifo_order() {
  Handoff handoff(8);
  ContinuousBatchedFifoScheduler scheduler(handoff, 6);
  scheduler.submit("a", 4);
  scheduler.submit("b", 4);
  scheduler.submit("c", 4);
  admit(scheduler, handoff, {2, 3, 5});

  Plan *plan = handoff.consume_plan();
  CHECK(plan != nullptr);
  if (plan != nullptr) {
    CHECK(plan->work.size() == 3);
    if (plan->work.size() == 3) {
      check_work(plan->work[0], 0, 0, 2, WorkKind::Prefill);
      check_work(plan->work[1], 1, 0, 3, WorkKind::Prefill);
      check_work(plan->work[2], 2, 0, 1, WorkKind::Prefill);
    }
    handoff.retire_plan(plan);
  }
}

void test_mixes_decode_and_prefill_without_reordering() {
  Handoff handoff(8);
  ContinuousBatchedFifoScheduler scheduler(handoff, 4);
  scheduler.submit("first", 3);
  scheduler.submit("second", 3);
  admit(scheduler, handoff, {2, 5});

  Plan *first = handoff.consume_plan();
  CHECK(first != nullptr);
  if (first == nullptr) return;
  CHECK(handoff.try_report_completion(Completion{
      0, 2, 1, 42, WorkKind::Prefill, ErrorCode::None, true, false}));
  CHECK(handoff.try_report_completion(Completion{
      1, 2, 0, 0, WorkKind::Prefill, ErrorCode::None, false, false}));
  handoff.retire_plan(first);
  scheduler.run_once();

  Plan *mixed = handoff.consume_plan();
  CHECK(mixed != nullptr);
  if (mixed != nullptr) {
    CHECK(mixed->work.size() == 2);
    if (mixed->work.size() == 2) {
      check_work(mixed->work[0], 0, 2, 3, WorkKind::Decode);
      check_work(mixed->work[1], 1, 2, 5, WorkKind::Prefill);
    }
    handoff.retire_plan(mixed);
  }
}

void test_pending_release_is_a_fifo_barrier() {
  Handoff handoff(8);
  ContinuousBatchedFifoScheduler scheduler(handoff, 4);
  scheduler.submit("first", 1);
  scheduler.submit("second", 3);
  admit(scheduler, handoff, {1, 3});

  Plan *plan = handoff.consume_plan();
  CHECK(plan != nullptr);
  if (plan == nullptr) return;
  CHECK(handoff.try_report_completion(Completion{
      0, 1, 1, 7, WorkKind::Prefill, ErrorCode::None, true, true}));
  CHECK(handoff.try_report_completion(Completion{
      1, 3, 1, 8, WorkKind::Prefill, ErrorCode::None, true, false}));
  handoff.retire_plan(plan);
  scheduler.run_once();

  CHECK(handoff.consume_plan() == nullptr);
  Release release{};
  CHECK(handoff.try_take_release(release));
  CHECK(release.id == 0);
}

} // namespace

int main() {
  test_fills_budget_in_strict_fifo_order();
  test_mixes_decode_and_prefill_without_reordering();
  test_pending_release_is_a_fifo_barrier();
  if (failures != 0) {
    std::printf("%d check(s) failed\n", failures);
    return 1;
  }
  std::printf("all continuous batched FIFO checks passed\n");
  return 0;
}
