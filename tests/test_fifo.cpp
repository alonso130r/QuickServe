#include "policies/fifo.hpp"

#include <cstdint>
#include <cstdio>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {

int g_failures = 0;

#define CHECK(cond)                                                            \
  do {                                                                         \
    if (!(cond)) {                                                             \
      std::printf("  FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);            \
      ++g_failures;                                                            \
    }                                                                          \
  } while (0)

Admission take_admission(Handoff &handoff) {
  Admission admission{};
  CHECK(handoff.try_take_admission(admission));
  return admission;
}

Plan *take_single_item_plan(Handoff &handoff, RequestId id,
                            std::uint32_t token_begin,
                            std::uint32_t token_end, WorkKind kind) {
  Plan *plan = handoff.consume_plan();
  CHECK(plan != nullptr);
  if (plan == nullptr) {
    return nullptr;
  }
  CHECK(plan->work.size() == 1);
  if (plan->work.size() == 1) {
    const WorkItem &work = plan->work.front();
    CHECK(work.id == id);
    CHECK(work.token_begin == token_begin);
    CHECK(work.token_end == token_end);
    CHECK(work.kind == kind);
  }
  return plan;
}

void retire(Handoff &handoff, Plan *plan) {
  if (plan != nullptr) {
    handoff.retire_plan(plan);
  }
}

void admit_all(FifoScheduler &scheduler, Handoff &handoff,
               const std::vector<std::uint32_t> &prompt_lengths) {
  scheduler.run_once();
  for (RequestId id = 0; id < prompt_lengths.size(); ++id) {
    const Admission admission = take_admission(handoff);
    CHECK(admission.id == id);
    CHECK(handoff.try_report_admission(
        AdmissionResult{id, prompt_lengths[id], ErrorCode::None}));
  }
  scheduler.run_once();
}

void check_scheduler_valid(const FifoScheduler &scheduler) {
  CHECK(scheduler.last_error().valid);
  CHECK(scheduler.last_error().code == ErrorCode::None);
}

void test_oldest_admitted_request_is_selected_first() {
  std::printf("test_oldest_admitted_request_is_selected_first\n");
  Handoff handoff(8);
  FifoScheduler scheduler(handoff, 8);
  const RequestId first = scheduler.submit("first", 4);
  const RequestId second = scheduler.submit("second", 4);

  admit_all(scheduler, handoff, {5, 3});
  Plan *plan = take_single_item_plan(handoff, first, 0, 5,
                                    WorkKind::Prefill);
  if (plan != nullptr) {
    CHECK(plan->work.front().id != second);
  }
  check_scheduler_valid(scheduler);
  retire(handoff, plan);
}

void test_prefill_chunks_follow_acknowledged_progress() {
  std::printf("test_prefill_chunks_follow_acknowledged_progress\n");
  Handoff handoff(8);
  FifoScheduler scheduler(handoff, 3);
  const RequestId id = scheduler.submit("chunked", 4);

  admit_all(scheduler, handoff, {8});
  Plan *first = take_single_item_plan(handoff, id, 0, 3,
                                     WorkKind::Prefill);
  CHECK(handoff.try_report_completion(Completion{
      id, 3, 0, 0, WorkKind::Prefill, ErrorCode::None, false, false}));
  retire(handoff, first);
  scheduler.run_once();

  Plan *second = take_single_item_plan(handoff, id, 3, 6,
                                      WorkKind::Prefill);
  CHECK(handoff.try_report_completion(Completion{
      id, 6, 0, 0, WorkKind::Prefill, ErrorCode::None, false, false}));
  retire(handoff, second);
  scheduler.run_once();

  Plan *final = take_single_item_plan(handoff, id, 6, 8,
                                     WorkKind::Prefill);
  check_scheduler_valid(scheduler);
  retire(handoff, final);
}

void test_final_prefill_token_leads_to_exact_decode_range() {
  std::printf("test_final_prefill_token_leads_to_exact_decode_range\n");
  Handoff handoff(8);
  FifoScheduler scheduler(handoff, 4);
  const RequestId id = scheduler.submit("prompt", 4);

  admit_all(scheduler, handoff, {4});
  Plan *prefill = take_single_item_plan(handoff, id, 0, 4,
                                       WorkKind::Prefill);
  CHECK(handoff.try_report_completion(Completion{
      id, 4, 1, 42, WorkKind::Prefill, ErrorCode::None, true, false}));
  retire(handoff, prefill);
  scheduler.run_once();

  Plan *decode =
      take_single_item_plan(handoff, id, 4, 5, WorkKind::Decode);
  check_scheduler_valid(scheduler);
  retire(handoff, decode);
}

void test_pending_admission_and_terminal_requests_are_skipped() {
  std::printf("test_pending_admission_and_terminal_requests_are_skipped\n");
  Handoff handoff(8);
  FifoScheduler scheduler(handoff, 8);
  const RequestId pending = scheduler.submit("pending", 4);
  const RequestId terminal = scheduler.submit("bad", 4);
  const RequestId ready = scheduler.submit("ready", 4);

  scheduler.run_once();
  CHECK(take_admission(handoff).id == pending);
  CHECK(take_admission(handoff).id == terminal);
  CHECK(take_admission(handoff).id == ready);
  CHECK(handoff.try_report_admission(AdmissionResult{
      terminal, 0, ErrorCode::TokenizationFailed}));
  CHECK(handoff.try_report_admission(
      AdmissionResult{ready, 2, ErrorCode::None}));
  scheduler.run_once();

  CHECK(scheduler.requests()[pending].stage ==
        RequestState::Stage::PendingAdmission);
  CHECK(scheduler.requests()[terminal].stage == RequestState::Stage::Terminal);
  Plan *plan =
      take_single_item_plan(handoff, ready, 0, 2, WorkKind::Prefill);
  check_scheduler_valid(scheduler);
  retire(handoff, plan);
}

void test_late_older_admission_does_not_preempt_active_younger_request() {
  std::printf(
      "test_late_older_admission_does_not_preempt_active_younger_request\n");
  Handoff handoff(8);
  FifoScheduler scheduler(handoff, 2);
  const RequestId older = scheduler.submit("older", 4);
  const RequestId younger = scheduler.submit("younger", 4);

  scheduler.run_once();
  CHECK(take_admission(handoff).id == older);
  CHECK(take_admission(handoff).id == younger);
  CHECK(handoff.try_report_admission(
      AdmissionResult{younger, 5, ErrorCode::None}));
  scheduler.run_once();

  Plan *first = take_single_item_plan(handoff, younger, 0, 2,
                                     WorkKind::Prefill);
  CHECK(handoff.try_report_completion(Completion{
      younger, 2, 0, 0, WorkKind::Prefill, ErrorCode::None, false, false}));
  retire(handoff, first);
  CHECK(handoff.try_report_admission(
      AdmissionResult{older, 4, ErrorCode::None}));
  scheduler.run_once();

  Plan *next = take_single_item_plan(handoff, younger, 2, 4,
                                    WorkKind::Prefill);
  check_scheduler_valid(scheduler);
  retire(handoff, next);
}

void test_pending_release_blocks_younger_until_acknowledged() {
  std::printf("test_pending_release_blocks_younger_until_acknowledged\n");
  Handoff handoff(8);
  FifoScheduler scheduler(handoff, 4);
  const RequestId first = scheduler.submit("first", 1);
  const RequestId second = scheduler.submit("second", 4);

  admit_all(scheduler, handoff, {1, 4});
  Plan *first_plan = take_single_item_plan(handoff, first, 0, 1,
                                          WorkKind::Prefill);
  CHECK(handoff.try_report_completion(Completion{
      first, 1, 1, 11, WorkKind::Prefill, ErrorCode::None, true, true}));
  retire(handoff, first_plan);
  scheduler.run_once();

  CHECK(scheduler.requests()[first].stage ==
        RequestState::Stage::PendingRelease);
  CHECK(handoff.consume_plan() == nullptr);
  Release release{};
  CHECK(handoff.try_take_release(release));
  CHECK(release.id == first);

  scheduler.run_once();
  CHECK(handoff.consume_plan() == nullptr);

  CHECK(handoff.try_acknowledge_release(ReleaseAck{first}));
  scheduler.run_once();
  CHECK(scheduler.requests()[first].stage == RequestState::Stage::Terminal);
  Plan *second_plan = take_single_item_plan(handoff, second, 0, 4,
                                           WorkKind::Prefill);
  check_scheduler_valid(scheduler);
  retire(handoff, second_plan);
}

void test_zero_output_release_barrier_respects_fifo_position() {
  std::printf("test_zero_output_release_barrier_respects_fifo_position\n");
  {
    Handoff handoff(8);
    FifoScheduler scheduler(handoff, 2);
    const RequestId cleanup_only = scheduler.submit("cleanup", 0);
    const RequestId younger = scheduler.submit("younger", 4);

    admit_all(scheduler, handoff, {1, 4});
    CHECK(scheduler.requests()[cleanup_only].stage ==
          RequestState::Stage::PendingRelease);
    CHECK(handoff.consume_plan() == nullptr);

    Release release{};
    CHECK(handoff.try_take_release(release));
    CHECK(release.id == cleanup_only);
    CHECK(handoff.try_acknowledge_release(ReleaseAck{cleanup_only}));
    scheduler.run_once();

    Plan *plan = take_single_item_plan(handoff, younger, 0, 2,
                                      WorkKind::Prefill);
    check_scheduler_valid(scheduler);
    retire(handoff, plan);
  }

  {
    Handoff handoff(8);
    FifoScheduler scheduler(handoff, 2);
    const RequestId older = scheduler.submit("older", 4);
    const RequestId cleanup_only = scheduler.submit("cleanup", 0);

    admit_all(scheduler, handoff, {4, 1});
    CHECK(scheduler.requests()[cleanup_only].stage ==
          RequestState::Stage::PendingRelease);
    Plan *plan = take_single_item_plan(handoff, older, 0, 2,
                                      WorkKind::Prefill);
    check_scheduler_valid(scheduler);
    retire(handoff, plan);
  }
}

void test_progressed_release_blocks_other_progressed_work() {
  std::printf("test_progressed_release_blocks_other_progressed_work\n");
  Handoff handoff(8);
  FifoScheduler scheduler(handoff, 2);
  const RequestId cleanup = scheduler.submit("cleanup", 4);
  const RequestId runnable = scheduler.submit("runnable", 4);

  admit_all(scheduler, handoff, {4, 4});
  Plan *first = take_single_item_plan(handoff, cleanup, 0, 2,
                                     WorkKind::Prefill);
  CHECK(handoff.try_report_completion(Completion{
      cleanup, 2, 1, 21, WorkKind::Prefill, ErrorCode::None, true, true}));
  CHECK(handoff.try_report_completion(Completion{
      runnable, 1, 0, 0, WorkKind::Prefill, ErrorCode::None, false, false}));
  retire(handoff, first);
  scheduler.run_once();

  CHECK(scheduler.requests()[cleanup].stage ==
        RequestState::Stage::PendingRelease);
  CHECK(scheduler.requests()[cleanup].prefill_position == 2);
  CHECK(scheduler.requests()[runnable].stage == RequestState::Stage::Prefill);
  CHECK(scheduler.requests()[runnable].prefill_position == 1);
  CHECK(handoff.consume_plan() == nullptr);

  Release release{};
  CHECK(handoff.try_take_release(release));
  CHECK(release.id == cleanup);
  CHECK(handoff.try_acknowledge_release(ReleaseAck{cleanup}));
  scheduler.run_once();

  Plan *resumed = take_single_item_plan(handoff, runnable, 1, 3,
                                       WorkKind::Prefill);
  check_scheduler_valid(scheduler);
  retire(handoff, resumed);
}

void test_late_older_zero_output_admission_does_not_preempt_active_younger() {
  std::printf(
      "test_late_older_zero_output_admission_does_not_preempt_active_younger\n");
  Handoff handoff(8);
  FifoScheduler scheduler(handoff, 2);
  const RequestId older = scheduler.submit("older cleanup", 0);
  const RequestId younger = scheduler.submit("younger active", 4);

  scheduler.run_once();
  CHECK(take_admission(handoff).id == older);
  CHECK(take_admission(handoff).id == younger);
  CHECK(handoff.try_report_admission(
      AdmissionResult{younger, 5, ErrorCode::None}));
  scheduler.run_once();

  Plan *first = take_single_item_plan(handoff, younger, 0, 2,
                                     WorkKind::Prefill);
  CHECK(handoff.try_report_completion(Completion{
      younger, 2, 0, 0, WorkKind::Prefill, ErrorCode::None, false, false}));
  retire(handoff, first);
  CHECK(handoff.try_report_admission(
      AdmissionResult{older, 1, ErrorCode::None}));
  scheduler.run_once();

  CHECK(scheduler.requests()[older].stage ==
        RequestState::Stage::PendingRelease);
  Plan *next = take_single_item_plan(handoff, younger, 2, 4,
                                    WorkKind::Prefill);
  check_scheduler_valid(scheduler);
  retire(handoff, next);
}

void test_complete_first_request_lifecycle_blocks_second_request() {
  std::printf("test_complete_first_request_lifecycle_blocks_second_request\n");
  Handoff handoff(8);
  FifoScheduler scheduler(handoff, 2);
  const RequestId first = scheduler.submit("first", 2);
  const RequestId second = scheduler.submit("second", 2);

  admit_all(scheduler, handoff, {5, 3});
  Plan *chunk_one = take_single_item_plan(handoff, first, 0, 2,
                                          WorkKind::Prefill);
  CHECK(handoff.try_report_completion(Completion{
      first, 2, 0, 0, WorkKind::Prefill, ErrorCode::None, false, false}));
  retire(handoff, chunk_one);
  scheduler.run_once();

  Plan *chunk_two = take_single_item_plan(handoff, first, 2, 4,
                                          WorkKind::Prefill);
  CHECK(handoff.try_report_completion(Completion{
      first, 4, 0, 0, WorkKind::Prefill, ErrorCode::None, false, false}));
  retire(handoff, chunk_two);
  scheduler.run_once();

  Plan *final_prefill = take_single_item_plan(handoff, first, 4, 5,
                                              WorkKind::Prefill);
  CHECK(handoff.try_report_completion(Completion{
      first, 5, 1, 31, WorkKind::Prefill, ErrorCode::None, true, false}));
  retire(handoff, final_prefill);
  scheduler.run_once();

  Plan *decode =
      take_single_item_plan(handoff, first, 5, 6, WorkKind::Decode);
  CHECK(handoff.try_report_completion(Completion{
      first, 5, 2, 32, WorkKind::Decode, ErrorCode::None, true, false}));
  retire(handoff, decode);
  scheduler.run_once();

  CHECK(scheduler.requests()[first].stage ==
        RequestState::Stage::PendingRelease);
  CHECK(scheduler.requests()[second].stage == RequestState::Stage::Prefill);
  CHECK(handoff.consume_plan() == nullptr);
  Release release{};
  CHECK(handoff.try_take_release(release));
  CHECK(release.id == first);

  scheduler.run_once();
  CHECK(handoff.consume_plan() == nullptr);
  CHECK(handoff.try_acknowledge_release(ReleaseAck{first}));
  scheduler.run_once();

  CHECK(scheduler.requests()[first].stage == RequestState::Stage::Terminal);
  Plan *younger = take_single_item_plan(handoff, second, 0, 2,
                                       WorkKind::Prefill);
  check_scheduler_valid(scheduler);
  retire(handoff, younger);
}

void test_zero_budget_is_rejected() {
  std::printf("test_zero_budget_is_rejected\n");
  Handoff handoff(8);
  bool threw = false;
  try {
    FifoScheduler scheduler(handoff, 0);
    (void)scheduler;
  } catch (const std::invalid_argument &) {
    threw = true;
  }
  CHECK(threw);
}

void test_zero_length_prompt_fails_explicitly_and_releases() {
  std::printf("test_zero_length_prompt_fails_explicitly_and_releases\n");
  Handoff handoff(8);
  FifoScheduler scheduler(handoff, 4);
  const RequestId id = scheduler.submit("empty tokens", 2);

  admit_all(scheduler, handoff, {0});
  CHECK(handoff.consume_plan() == nullptr);
  CHECK(!scheduler.last_error().valid);
  CHECK(scheduler.last_error().code == ErrorCode::ProtocolViolation);
  CHECK(!scheduler.last_error().has_offending_work);
  CHECK(scheduler.requests()[id].stage ==
        RequestState::Stage::PendingRelease);
  Release release{};
  CHECK(handoff.try_take_release(release));
  CHECK(release.id == id);
  CHECK(handoff.try_acknowledge_release(ReleaseAck{id}));
  scheduler.run_once();
  CHECK(scheduler.requests()[id].stage == RequestState::Stage::Terminal);
  CHECK(scheduler.requests()[id].terminal_error ==
        ErrorCode::ProtocolViolation);
}

void test_decode_position_overflow_fails_explicitly_and_releases() {
  std::printf("test_decode_position_overflow_fails_explicitly_and_releases\n");
  Handoff handoff(8);
  FifoScheduler scheduler(handoff, 1);
  const RequestId id = scheduler.submit("huge", 2);

  admit_all(scheduler, handoff,
            {std::numeric_limits<std::uint32_t>::max()});
  Plan *prefill = take_single_item_plan(
      handoff, id, 0, 1, WorkKind::Prefill);
  CHECK(handoff.try_report_completion(Completion{
      id, std::numeric_limits<std::uint32_t>::max(), 1, 7,
      WorkKind::Prefill, ErrorCode::None, true, false}));
  retire(handoff, prefill);
  scheduler.run_once();

  CHECK(handoff.consume_plan() == nullptr);
  CHECK(!scheduler.last_error().valid);
  CHECK(scheduler.last_error().code == ErrorCode::ProtocolViolation);
  CHECK(!scheduler.last_error().has_offending_work);
  CHECK(scheduler.requests()[id].stage ==
        RequestState::Stage::PendingRelease);
  Release release{};
  CHECK(handoff.try_take_release(release));
  CHECK(release.id == id);
  CHECK(handoff.try_acknowledge_release(ReleaseAck{id}));
  scheduler.run_once();
  CHECK(scheduler.requests()[id].stage == RequestState::Stage::Terminal);
  CHECK(scheduler.requests()[id].terminal_error ==
        ErrorCode::ProtocolViolation);
}

} // namespace

int main() {
  test_oldest_admitted_request_is_selected_first();
  test_prefill_chunks_follow_acknowledged_progress();
  test_final_prefill_token_leads_to_exact_decode_range();
  test_pending_admission_and_terminal_requests_are_skipped();
  test_late_older_admission_does_not_preempt_active_younger_request();
  test_pending_release_blocks_younger_until_acknowledged();
  test_zero_output_release_barrier_respects_fifo_position();
  test_progressed_release_blocks_other_progressed_work();
  test_late_older_zero_output_admission_does_not_preempt_active_younger();
  test_complete_first_request_lifecycle_blocks_second_request();
  test_zero_budget_is_rejected();
  test_zero_length_prompt_fails_explicitly_and_releases();
  test_decode_position_overflow_fails_explicitly_and_releases();

  if (g_failures != 0) {
    std::printf("\n%d check(s) failed\n", g_failures);
    return 1;
  }
  std::printf("\nall FIFO policy checks passed\n");
  return 0;
}
