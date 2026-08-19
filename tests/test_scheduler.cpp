#include "runtime/scheduler.hpp"

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <limits>
#include <set>
#include <stdexcept>
#include <string>
#include <thread>
#include <type_traits>
#include <utility>
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

class ProbePolicy final : public Scheduler {
public:
  explicit ProbePolicy(Handoff &handoff, std::uint32_t budget = 32)
      : Scheduler(handoff, budget) {}

  std::vector<WorkItem> next_work;
  int builds = 0;

protected:
  void build_plan(Plan &out) override {
    static_assert(std::is_const_v<std::remove_reference_t<
                      decltype(policy_requests())>>,
                  "policy request view must be immutable");
    ++builds;
    const auto &states = policy_requests();
    if (next_work.empty()) {
      for (const RequestState &state : states) {
        if (state.stage == RequestState::Stage::Prefill &&
            state.prefill_position < state.prompt_length) {
          out.work.push_back({state.id, state.prefill_position,
                              state.prefill_position + 1,
                              WorkKind::Prefill});
          return;
        }
        if (state.stage == RequestState::Stage::Decode &&
            state.decoded_count >= 1) {
          const std::uint64_t end =
              static_cast<std::uint64_t>(state.prompt_length) +
              state.decoded_count;
          if (end <= std::numeric_limits<std::uint32_t>::max()) {
            out.work.push_back(
                {state.id, static_cast<std::uint32_t>(end - 1),
                 static_cast<std::uint32_t>(end), WorkKind::Decode});
          }
          return;
        }
      }
      return;
    }
    for (const WorkItem &work : next_work) {
      if (work.id < states.size() &&
          states[work.id].stage != RequestState::Stage::Terminal &&
          states[work.id].stage != RequestState::Stage::PendingAdmission &&
          states[work.id].stage != RequestState::Stage::PendingRelease) {
        out.work.push_back(work);
      }
    }
  }
};

class EmptyPolicy final : public Scheduler {
public:
  explicit EmptyPolicy(Handoff &handoff) : Scheduler(handoff, 32) {}

protected:
  void build_plan(Plan &) override {}
};

class UnitPrefillPolicy final : public Scheduler {
public:
  explicit UnitPrefillPolicy(Handoff &handoff) : Scheduler(handoff, 1) {}

protected:
  void build_plan(Plan &out) override {
    for (const RequestState &state : policy_requests()) {
      if (state.stage == RequestState::Stage::Prefill) {
        out.work.push_back({state.id, state.prefill_position,
                            state.prefill_position + 1, WorkKind::Prefill});
        return;
      }
    }
  }
};

class MaliciousPolicy final : public Scheduler {
public:
  explicit MaliciousPolicy(Handoff &handoff, std::uint32_t budget = 32)
      : Scheduler(handoff, budget) {}

  std::vector<WorkItem> next_work;

  void force_decode_without_generated_token(RequestId id) {
    auto &states = const_cast<std::vector<RequestState> &>(policy_requests());
    states[id].stage = RequestState::Stage::Decode;
    states[id].decoded_count = 0;
  }

protected:
  void build_plan(Plan &out) override {
    if (!next_work.empty()) {
      out.work = next_work;
      return;
    }
    for (const RequestState &state : policy_requests()) {
      if (state.stage == RequestState::Stage::Prefill &&
          state.prefill_position < state.prompt_length) {
        out.work.push_back({state.id, state.prefill_position,
                            state.prefill_position + 1,
                            WorkKind::Prefill});
        return;
      }
      if (state.stage == RequestState::Stage::Decode &&
          state.decoded_count >= 1) {
        const std::uint64_t end =
            static_cast<std::uint64_t>(state.prompt_length) +
            state.decoded_count;
        if (end <= std::numeric_limits<std::uint32_t>::max()) {
          out.work.push_back({state.id,
                              static_cast<std::uint32_t>(end - 1),
                              static_cast<std::uint32_t>(end),
                              WorkKind::Decode});
        }
        return;
      }
    }
  }
};

Admission take_admission(Handoff &handoff) {
  Admission admission{};
  CHECK(handoff.try_take_admission(admission));
  return admission;
}

void admit_success(ProbePolicy &scheduler, Handoff &handoff, RequestId id,
                   std::uint32_t prompt_tokens) {
  scheduler.run_once();
  const Admission admission = take_admission(handoff);
  CHECK(admission.id == id);
  CHECK(handoff.try_report_admission(
      AdmissionResult{id, prompt_tokens, ErrorCode::None}));
  scheduler.run_once();
}

void admit_success(MaliciousPolicy &scheduler, Handoff &handoff, RequestId id,
                   std::uint32_t prompt_tokens) {
  scheduler.run_once();
  const Admission admission = take_admission(handoff);
  CHECK(admission.id == id);
  CHECK(handoff.try_report_admission(
      AdmissionResult{id, prompt_tokens, ErrorCode::None}));
  scheduler.next_work = {{id, 0, 1, WorkKind::Prefill}};
  scheduler.run_once();
  Plan *setup = handoff.consume_plan();
  CHECK(setup != nullptr);
  if (setup != nullptr) {
    handoff.retire_plan(setup);
  }
  scheduler.next_work.clear();
}

void expect_invalid_plan_is_not_published(
    const char *name, std::uint32_t budget,
    const std::function<void(MaliciousPolicy &, Handoff &, RequestId)> &emit) {
  std::printf("  %s\n", name);
  Handoff handoff(8);
  MaliciousPolicy scheduler(handoff, budget);
  const RequestId id = scheduler.submit("prompt", 4);
  admit_success(scheduler, handoff, id, 4);
  emit(scheduler, handoff, id);
  const WorkItem offending = scheduler.next_work.front();
  scheduler.run_once();
  CHECK(handoff.consume_plan() == nullptr);
  CHECK(!scheduler.last_error().valid);
  CHECK(scheduler.last_error().code == ErrorCode::ProtocolViolation);
  CHECK(scheduler.last_error().has_offending_work);
  CHECK(scheduler.last_error().offending_work.id == offending.id);
  CHECK(scheduler.last_error().offending_work.token_begin ==
        offending.token_begin);
  CHECK(scheduler.last_error().offending_work.token_end ==
        offending.token_end);
  CHECK(scheduler.last_error().offending_work.kind == offending.kind);
  CHECK(!scheduler.last_error().detail.empty());
}

void test_invalid_plans_are_never_published() {
  std::printf("test_invalid_plans_are_never_published\n");

  expect_invalid_plan_is_not_published(
      "cost over token budget", 2,
      [](MaliciousPolicy &scheduler, Handoff &, RequestId id) {
        scheduler.next_work = {{id, 0, 3, WorkKind::Prefill}};
      });
  {
    std::printf("  aggregate cost over token budget\n");
    Handoff handoff(8);
    MaliciousPolicy scheduler(handoff, 3);
    const RequestId first = scheduler.submit("first", 4);
    admit_success(scheduler, handoff, first, 2);
    const RequestId second = scheduler.submit("second", 4);
    admit_success(scheduler, handoff, second, 2);
    scheduler.next_work = {{first, 0, 2, WorkKind::Prefill},
                           {second, 0, 2, WorkKind::Prefill}};
    scheduler.run_once();
    CHECK(handoff.consume_plan() == nullptr);
    CHECK(scheduler.last_error().code == ErrorCode::ProtocolViolation);
    CHECK(scheduler.last_error().offending_work.id == second);
  }
  expect_invalid_plan_is_not_published(
      "unknown request", 32,
      [](MaliciousPolicy &scheduler, Handoff &, RequestId) {
        scheduler.next_work = {{999, 0, 1, WorkKind::Prefill}};
      });
  expect_invalid_plan_is_not_published(
      "zero length range", 32,
      [](MaliciousPolicy &scheduler, Handoff &, RequestId id) {
        scheduler.next_work = {{id, 0, 0, WorkKind::Prefill}};
      });
  expect_invalid_plan_is_not_published(
      "reversed range", 32,
      [](MaliciousPolicy &scheduler, Handoff &, RequestId id) {
        scheduler.next_work = {{id, 2, 1, WorkKind::Prefill}};
      });
  expect_invalid_plan_is_not_published(
      "duplicate request", 32,
      [](MaliciousPolicy &scheduler, Handoff &, RequestId id) {
        scheduler.next_work = {{id, 0, 1, WorkKind::Prefill},
                               {id, 0, 1, WorkKind::Prefill}};
      });
  expect_invalid_plan_is_not_published(
      "prefill does not start at acknowledged position", 32,
      [](MaliciousPolicy &scheduler, Handoff &, RequestId id) {
        scheduler.next_work = {{id, 1, 2, WorkKind::Prefill}};
      });
  expect_invalid_plan_is_not_published(
      "prefill ends past prompt", 32,
      [](MaliciousPolicy &scheduler, Handoff &, RequestId id) {
        scheduler.next_work = {{id, 0, 5, WorkKind::Prefill}};
      });
  expect_invalid_plan_is_not_published(
      "decode before first generated token", 32,
      [](MaliciousPolicy &scheduler, Handoff &, RequestId id) {
        scheduler.next_work = {{id, 3, 4, WorkKind::Decode}};
      });
  expect_invalid_plan_is_not_published(
      "decode stage requires an already generated token", 32,
      [](MaliciousPolicy &scheduler, Handoff &, RequestId id) {
        scheduler.force_decode_without_generated_token(id);
        scheduler.next_work = {{id, 3, 4, WorkKind::Decode}};
      });
  expect_invalid_plan_is_not_published(
      "decode range is off by one", 32,
      [](MaliciousPolicy &scheduler, Handoff &handoff, RequestId id) {
        scheduler.next_work = {{id, 3, 4, WorkKind::Decode}};
        CHECK(handoff.try_report_completion(Completion{
            id, 4, 1, 7, WorkKind::Prefill, ErrorCode::None, true, false}));
        scheduler.run_once();
      });
  expect_invalid_plan_is_not_published(
      "decode work must be exactly one token", 32,
      [](MaliciousPolicy &scheduler, Handoff &handoff, RequestId id) {
        scheduler.next_work = {{id, 4, 6, WorkKind::Decode}};
        CHECK(handoff.try_report_completion(Completion{
            id, 4, 1, 7, WorkKind::Prefill, ErrorCode::None, true, false}));
        scheduler.run_once();
      });

  {
    std::printf("  decode expected range arithmetic overflows\n");
    Handoff handoff(8);
    MaliciousPolicy scheduler(handoff);
    const RequestId id = scheduler.submit("prompt", 2);
    admit_success(scheduler, handoff, id,
                  std::numeric_limits<std::uint32_t>::max());
    scheduler.next_work = {
        {id, std::numeric_limits<std::uint32_t>::max() - 1,
         std::numeric_limits<std::uint32_t>::max(), WorkKind::Decode}};
    CHECK(handoff.try_report_completion(Completion{
        id, std::numeric_limits<std::uint32_t>::max(), 1, 7,
        WorkKind::Prefill, ErrorCode::None, true, false}));
    scheduler.run_once();
    scheduler.run_once();
    CHECK(handoff.consume_plan() == nullptr);
    CHECK(scheduler.last_error().code == ErrorCode::ProtocolViolation);
  }
}

void test_exact_decode_range_is_publishable() {
  std::printf("test_exact_decode_range_is_publishable\n");
  Handoff handoff(8);
  MaliciousPolicy scheduler(handoff);
  const RequestId id = scheduler.submit("prompt", 4);
  admit_success(scheduler, handoff, id, 4);
  scheduler.next_work = {{id, 4, 5, WorkKind::Decode}};
  CHECK(handoff.try_report_completion(Completion{
      id, 4, 1, 7, WorkKind::Prefill, ErrorCode::None, true, false}));
  scheduler.run_once();
  scheduler.run_once();
  Plan *plan = handoff.consume_plan();
  CHECK(plan != nullptr);
  if (plan != nullptr) {
    CHECK(plan->work.size() == 1);
    CHECK(plan->work.front().id == id);
    CHECK(plan->work.front().token_begin == 4);
    CHECK(plan->work.front().token_end == 5);
    CHECK(plan->work.front().kind == WorkKind::Decode);
    handoff.retire_plan(plan);
  }
  CHECK(scheduler.last_error().valid);
  CHECK(scheduler.last_error().code == ErrorCode::None);
}

void test_work_for_non_schedulable_stages_is_rejected() {
  std::printf("test_work_for_non_schedulable_stages_is_rejected\n");

  for (const RequestState::Stage stage : {
           RequestState::Stage::PendingAdmission,
           RequestState::Stage::PendingRelease,
           RequestState::Stage::Terminal,
       }) {
    Handoff handoff(8);
    MaliciousPolicy scheduler(handoff);
    const RequestId anchor = scheduler.submit("anchor", 4);
    admit_success(scheduler, handoff, anchor, 4);
    const RequestId victim = scheduler.submit("victim", 1);
    scheduler.run_once();
    take_admission(handoff);

    if (stage == RequestState::Stage::PendingRelease) {
      CHECK(handoff.try_report_admission(
          AdmissionResult{victim, 1, ErrorCode::None}));
      scheduler.run_once();
      CHECK(handoff.try_report_completion(Completion{
          victim, 1, 1, 8, WorkKind::Prefill, ErrorCode::None, true, false}));
      scheduler.run_once();
    } else if (stage == RequestState::Stage::Terminal) {
      CHECK(handoff.try_report_admission(AdmissionResult{
          victim, 0, ErrorCode::TokenizationFailed}));
      scheduler.run_once();
    }

    CHECK(scheduler.requests()[victim].stage == stage);
    Plan *setup = handoff.consume_plan();
    if (setup != nullptr) {
      handoff.retire_plan(setup);
    }
    scheduler.next_work = {{victim, 0, 1, WorkKind::Prefill}};
    scheduler.run_once();
    CHECK(handoff.consume_plan() == nullptr);
    CHECK(scheduler.last_error().code == ErrorCode::ProtocolViolation);
    CHECK(scheduler.last_error().offending_work.id == victim);
  }
}

void test_protocol_violation_stops_admissions_and_drains_races() {
  std::printf("test_protocol_violation_stops_admissions_and_drains_races\n");

  for (const bool admission_succeeds : {false, true}) {
    Handoff handoff(8);
    MaliciousPolicy scheduler(handoff, 1);
    const RequestId owned = scheduler.submit("owned", 4);
    admit_success(scheduler, handoff, owned, 4);
    const RequestId in_flight = scheduler.submit("racing", 4);
    scheduler.next_work = {{owned, 0, 2, WorkKind::Prefill}};
    scheduler.run_once();

    CHECK(take_admission(handoff).id == in_flight);
    CHECK(handoff.consume_plan() == nullptr);
    CHECK(scheduler.last_error().code == ErrorCode::ProtocolViolation);
    CHECK(scheduler.requests()[owned].stage ==
          RequestState::Stage::PendingRelease);
    CHECK(scheduler.requests()[in_flight].admission ==
          RequestState::AdmissionOwnership::InFlight);

    scheduler.run_once();
    Release owned_release{};
    CHECK(handoff.try_take_release(owned_release));
    CHECK(owned_release.id == owned);
    CHECK(handoff.try_acknowledge_release(ReleaseAck{owned}));
    CHECK(handoff.try_report_admission(AdmissionResult{
        in_flight, admission_succeeds ? 3u : 0u,
        admission_succeeds ? ErrorCode::None
                           : ErrorCode::EnvironmentStopped}));
    scheduler.run_once();

    Release raced_release{};
    if (admission_succeeds) {
      CHECK(handoff.try_take_release(raced_release));
      CHECK(raced_release.id == in_flight);
      CHECK(handoff.try_acknowledge_release(ReleaseAck{in_flight}));
      scheduler.run_once();
    } else {
      CHECK(!handoff.try_take_release(raced_release));
    }

    CHECK(scheduler.all_terminal());
    CHECK(scheduler.requests()[owned].terminal_error ==
          ErrorCode::ProtocolViolation);
    CHECK(scheduler.requests()[in_flight].terminal_error ==
          ErrorCode::ProtocolViolation);
    CHECK(!handoff.stop_requested());
    CHECK(handoff.consume_plan() == nullptr);
    scheduler.run();
    CHECK(handoff.stop_requested());
  }
}

void test_protocol_violation_retries_release_backpressure() {
  std::printf("test_protocol_violation_retries_release_backpressure\n");
  Handoff handoff(8, 2, 2);
  MaliciousPolicy scheduler(handoff, 1);
  const RequestId id = scheduler.submit("owned", 4);
  admit_success(scheduler, handoff, id, 4);
  CHECK(handoff.try_release(Release{100}));
  CHECK(handoff.try_release(Release{101}));

  scheduler.next_work = {{id, 0, 2, WorkKind::Prefill}};
  scheduler.run_once();
  scheduler.run_once();
  CHECK(!scheduler.requests()[id].release_sent);

  Release release{};
  CHECK(handoff.try_take_release(release));
  CHECK(release.id == 100);
  scheduler.run_once();
  CHECK(scheduler.requests()[id].release_sent);
  CHECK(handoff.try_take_release(release));
  CHECK(release.id == 101);
  CHECK(handoff.try_take_release(release));
  CHECK(release.id == id);
  CHECK(handoff.try_acknowledge_release(ReleaseAck{id}));
  scheduler.run_once();
  CHECK(scheduler.all_terminal());
}

void test_protocol_violation_fails_notsent_locally_and_stops_flushing() {
  std::printf(
      "test_protocol_violation_fails_notsent_locally_and_stops_flushing\n");
  Handoff handoff(8, 2, 2);
  MaliciousPolicy scheduler(handoff, 1);
  const RequestId owned = scheduler.submit("owned", 4);
  admit_success(scheduler, handoff, owned, 4);

  Admission blocker_a{100, "blocker a", 1};
  Admission blocker_b{101, "blocker b", 1};
  CHECK(handoff.try_admit(std::move(blocker_a)));
  CHECK(handoff.try_admit(std::move(blocker_b)));
  const RequestId not_sent = scheduler.submit("must stay local", 4);
  scheduler.next_work = {{owned, 0, 2, WorkKind::Prefill}};
  scheduler.run_once();

  CHECK(scheduler.requests()[not_sent].stage == RequestState::Stage::Terminal);
  CHECK(scheduler.requests()[not_sent].admission ==
        RequestState::AdmissionOwnership::NotSent);
  CHECK(scheduler.requests()[not_sent].terminal_error ==
        ErrorCode::ProtocolViolation);
  CHECK(take_admission(handoff).id == 100);
  CHECK(take_admission(handoff).id == 101);

  scheduler.run_once();
  Admission admission{};
  CHECK(!handoff.try_take_admission(admission));
}

void test_protocol_violation_remains_the_drain_cause_after_later_fatal() {
  std::printf(
      "test_protocol_violation_remains_the_drain_cause_after_later_fatal\n");
  Handoff handoff(8);
  MaliciousPolicy scheduler(handoff, 1);
  const RequestId id = scheduler.submit("owned", 4);
  admit_success(scheduler, handoff, id, 4);
  scheduler.next_work = {{id, 0, 2, WorkKind::Prefill}};
  scheduler.run_once();
  CHECK(scheduler.requests()[id].terminal_error ==
        ErrorCode::ProtocolViolation);

  CHECK(handoff.try_report_fatal(RunFatal{ErrorCode::DecodeFailed}));
  scheduler.run_once();
  CHECK(scheduler.requests()[id].terminal_error ==
        ErrorCode::ProtocolViolation);
  CHECK(scheduler.last_error().code == ErrorCode::ProtocolViolation);
}

void test_protocol_violation_remains_the_error_after_late_completion() {
  std::printf(
      "test_protocol_violation_remains_the_error_after_late_completion\n");
  Handoff handoff(8);
  MaliciousPolicy scheduler(handoff, 1);
  const RequestId id = scheduler.submit("owned", 4);
  admit_success(scheduler, handoff, id, 4);
  scheduler.next_work = {{id, 0, 2, WorkKind::Prefill}};
  scheduler.run_once();
  CHECK(scheduler.requests()[id].stage ==
        RequestState::Stage::PendingRelease);
  CHECK(scheduler.requests()[id].terminal_error ==
        ErrorCode::ProtocolViolation);

  CHECK(handoff.try_report_completion(Completion{
      id, 0, 0, 0, WorkKind::Prefill, ErrorCode::DecodeFailed, false,
      false}));
  scheduler.run_once();
  CHECK(scheduler.requests()[id].stage ==
        RequestState::Stage::PendingRelease);
  CHECK(scheduler.requests()[id].terminal_error ==
        ErrorCode::ProtocolViolation);
  CHECK(scheduler.last_error().code == ErrorCode::ProtocolViolation);
}

void test_request_error_survives_a_later_run_fatal() {
  std::printf("test_request_error_survives_a_later_run_fatal\n");
  Handoff handoff(8);
  ProbePolicy scheduler(handoff);
  const RequestId failed = scheduler.submit("sampling failure", 4);
  admit_success(scheduler, handoff, failed, 4);
  const RequestId other = scheduler.submit("other owned request", 4);
  admit_success(scheduler, handoff, other, 4);

  CHECK(handoff.try_report_completion(Completion{
      failed, 4, 1, 0, WorkKind::Prefill, ErrorCode::SamplingFailed, false,
      false}));
  scheduler.run_once();
  CHECK(scheduler.requests()[failed].terminal_error ==
        ErrorCode::SamplingFailed);

  CHECK(handoff.try_report_fatal(RunFatal{ErrorCode::DecodeFailed}));
  scheduler.run_once();
  CHECK(scheduler.requests()[failed].terminal_error ==
        ErrorCode::SamplingFailed);
  CHECK(scheduler.requests()[other].terminal_error ==
        ErrorCode::DecodeFailed);
  CHECK(scheduler.requests()[failed].stage ==
        RequestState::Stage::PendingRelease);
  CHECK(scheduler.requests()[other].stage ==
        RequestState::Stage::PendingRelease);
}

void test_queued_fatal_still_drains_admission_flushed_before_observation() {
  std::printf(
      "test_queued_fatal_still_drains_admission_flushed_before_observation\n");
  Handoff handoff(8);
  ProbePolicy scheduler(handoff);
  const RequestId id = scheduler.submit("racing admission", 4);
  CHECK(handoff.try_report_fatal(RunFatal{ErrorCode::DecodeFailed}));

  scheduler.run_once();
  CHECK(scheduler.requests()[id].admission ==
        RequestState::AdmissionOwnership::InFlight);
  CHECK(scheduler.requests()[id].stage ==
        RequestState::Stage::PendingAdmission);

  const Admission admission = take_admission(handoff);
  CHECK(admission.id == id);
  CHECK(handoff.try_report_admission(
      AdmissionResult{id, 0, ErrorCode::EnvironmentStopped}));
  scheduler.run_once();

  CHECK(scheduler.all_terminal());
  CHECK(scheduler.requests()[id].terminal_error == ErrorCode::DecodeFailed);
  Release release{};
  CHECK(!handoff.try_take_release(release));
  CHECK(!handoff.stop_requested());
  scheduler.run();
  CHECK(handoff.stop_requested());
}

void test_submit_assigns_stable_ids_without_touching_handoff() {
  std::printf("test_submit_assigns_stable_ids_without_touching_handoff\n");
  Handoff handoff(8);
  ProbePolicy scheduler(handoff);

  CHECK(scheduler.submit("first", 3) == 0);
  CHECK(scheduler.submit("second", 4) == 1);
  CHECK(scheduler.requests().size() == 2);
  CHECK(scheduler.requests()[0].id == 0);
  CHECK(scheduler.requests()[1].id == 1);
  Admission admission{};
  CHECK(!handoff.try_take_admission(admission));
}

void test_zero_token_budget_is_rejected() {
  std::printf("test_zero_token_budget_is_rejected\n");
  Handoff handoff(8);
  bool threw = false;
  try {
    ProbePolicy scheduler(handoff, 0);
    (void)scheduler;
  } catch (const std::invalid_argument &) {
    threw = true;
  }
  CHECK(threw);
}

void test_empty_policy_plan_fails_all_schedulable_requests_and_cleans_up() {
  std::printf(
      "test_empty_policy_plan_fails_all_schedulable_requests_and_cleans_up\n");
  Handoff handoff(8);
  EmptyPolicy scheduler(handoff);
  const RequestId first = scheduler.submit("first", 4);
  const RequestId second = scheduler.submit("second", 4);

  scheduler.run_once();
  CHECK(take_admission(handoff).id == first);
  CHECK(take_admission(handoff).id == second);
  CHECK(handoff.try_report_admission(
      AdmissionResult{first, 3, ErrorCode::None}));
  CHECK(handoff.try_report_admission(
      AdmissionResult{second, 2, ErrorCode::None}));
  scheduler.run_once();

  CHECK(!scheduler.last_error().valid);
  CHECK(scheduler.last_error().code == ErrorCode::ProtocolViolation);
  CHECK(!scheduler.last_error().has_offending_work);
  CHECK(!scheduler.last_error().detail.empty());
  CHECK(scheduler.requests()[first].stage ==
        RequestState::Stage::PendingRelease);
  CHECK(scheduler.requests()[second].stage ==
        RequestState::Stage::PendingRelease);
  CHECK(handoff.consume_plan() == nullptr);

  Release first_release{};
  Release second_release{};
  CHECK(handoff.try_take_release(first_release));
  CHECK(handoff.try_take_release(second_release));
  CHECK(first_release.id == first);
  CHECK(second_release.id == second);
  CHECK(handoff.try_acknowledge_release(ReleaseAck{first}));
  CHECK(handoff.try_acknowledge_release(ReleaseAck{second}));
  scheduler.run_once();

  CHECK(scheduler.all_terminal());
  CHECK(scheduler.requests()[first].terminal_error ==
        ErrorCode::ProtocolViolation);
  CHECK(scheduler.requests()[second].terminal_error ==
        ErrorCode::ProtocolViolation);
}

void test_run_once_flushes_admissions_with_backpressure() {
  std::printf("test_run_once_flushes_admissions_with_backpressure\n");
  Handoff handoff(8, 2, 2);
  ProbePolicy scheduler(handoff);
  scheduler.submit("zero", 1);
  scheduler.submit("one", 1);
  scheduler.submit("two", 1);

  scheduler.run_once();
  CHECK(scheduler.requests()[0].admission ==
        RequestState::AdmissionOwnership::InFlight);
  CHECK(scheduler.requests()[1].admission ==
        RequestState::AdmissionOwnership::InFlight);
  CHECK(scheduler.requests()[2].admission ==
        RequestState::AdmissionOwnership::NotSent);

  CHECK(take_admission(handoff).id == 0);
  scheduler.run_once();
  CHECK(scheduler.requests()[2].admission ==
        RequestState::AdmissionOwnership::InFlight);
  CHECK(take_admission(handoff).id == 1);
  CHECK(take_admission(handoff).id == 2);
}

void test_admission_transitions_and_errors_without_release() {
  std::printf("test_admission_transitions_and_errors_without_release\n");
  Handoff handoff(8);
  ProbePolicy scheduler(handoff);
  const RequestId ok = scheduler.submit("ok", 2);
  const RequestId bad = scheduler.submit("bad", 2);

  scheduler.run_once();
  CHECK(scheduler.requests()[ok].stage ==
        RequestState::Stage::PendingAdmission);
  CHECK(scheduler.requests()[ok].admission ==
        RequestState::AdmissionOwnership::InFlight);
  take_admission(handoff);
  take_admission(handoff);

  CHECK(handoff.try_report_admission(
      AdmissionResult{ok, 5, ErrorCode::None}));
  CHECK(handoff.try_report_admission(
      AdmissionResult{bad, 0, ErrorCode::TokenizationFailed}));
  scheduler.run_once();

  CHECK(scheduler.requests()[ok].admission ==
        RequestState::AdmissionOwnership::EnvironmentOwned);
  CHECK(scheduler.requests()[ok].stage == RequestState::Stage::Prefill);
  CHECK(scheduler.requests()[ok].prompt_length == 5);
  CHECK(scheduler.requests()[bad].stage == RequestState::Stage::Terminal);
  CHECK(scheduler.requests()[bad].terminal_error ==
        ErrorCode::TokenizationFailed);
  Release release{};
  CHECK(!handoff.try_take_release(release));
}

void test_admission_failures_are_terminal_and_all_terminal_is_exact() {
  std::printf("test_admission_failures_are_terminal_and_all_terminal_is_exact\n");
  Handoff handoff(8);
  ProbePolicy scheduler(handoff);
  const RequestId tokenization = scheduler.submit("bad tokens", 2);
  const RequestId capacity = scheduler.submit("too large", 2);
  CHECK(!scheduler.all_terminal());
  scheduler.run_once();
  take_admission(handoff);
  take_admission(handoff);
  CHECK(handoff.try_report_admission(AdmissionResult{
      tokenization, 0, ErrorCode::TokenizationFailed}));
  CHECK(handoff.try_report_admission(AdmissionResult{
      capacity, 99, ErrorCode::ContextCapacityExceeded}));
  scheduler.run_once();
  CHECK(scheduler.all_terminal());
  CHECK(scheduler.requests()[tokenization].finish_recorded);
  CHECK(scheduler.requests()[capacity].finish_recorded);
  Release release{};
  CHECK(!handoff.try_take_release(release));
}

void test_absolute_completions_are_idempotent_and_prefill_can_generate() {
  std::printf(
      "test_absolute_completions_are_idempotent_and_prefill_can_generate\n");
  Handoff handoff(8);
  ProbePolicy scheduler(handoff);
  const RequestId id = scheduler.submit("prompt", 4);
  admit_success(scheduler, handoff, id, 6);

  const Completion completion{id, 6, 1, 42, WorkKind::Prefill,
                              ErrorCode::None, true, false};
  CHECK(handoff.try_report_completion(completion));
  CHECK(handoff.try_report_completion(completion));
  scheduler.run_once();
  CHECK(scheduler.requests()[id].prefill_position == 6);
  CHECK(scheduler.requests()[id].decoded_count == 1);
  CHECK(scheduler.requests()[id].output_token_ids ==
        std::vector<Token>{42});
  CHECK(scheduler.requests()[id].first_token_recorded);
  CHECK(scheduler.requests()[id].stage == RequestState::Stage::Decode);
}

void test_output_is_appended_before_matching_completion_is_folded() {
  std::printf("test_output_is_appended_before_matching_completion_is_folded\n");
  Handoff handoff(8);
  ProbePolicy scheduler(handoff);
  const RequestId id = scheduler.submit("prompt", 3);
  admit_success(scheduler, handoff, id, 2);

  OutputPiece piece{id, 77, "piece"};
  CHECK(handoff.try_report_output(std::move(piece)));
  CHECK(handoff.try_report_completion(
      Completion{id, 2, 1, 77, WorkKind::Prefill, ErrorCode::None, true,
                 false}));
  scheduler.run_once();
  CHECK(scheduler.requests()[id].output_text == "piece");
  CHECK(scheduler.requests()[id].output_token_ids ==
        std::vector<Token>{77});
}

void test_eos_and_output_limit_release_then_ack_terminal() {
  std::printf("test_eos_and_output_limit_release_then_ack_terminal\n");
  for (const bool eos : {false, true}) {
    Handoff handoff(8);
    ProbePolicy scheduler(handoff);
    const RequestId id = scheduler.submit("prompt", eos ? 8 : 1);
    admit_success(scheduler, handoff, id, 2);
    CHECK(handoff.try_report_completion(
        Completion{id, 2, 1, 9, WorkKind::Prefill, ErrorCode::None, true,
                   eos}));
    scheduler.run_once();
    CHECK(scheduler.requests()[id].stage ==
          RequestState::Stage::PendingRelease);
    CHECK(!scheduler.all_terminal());
    Release release{};
    CHECK(handoff.try_take_release(release));
    CHECK(release.id == id);
    CHECK(handoff.try_acknowledge_release(ReleaseAck{id}));
    scheduler.run_once();
    CHECK(scheduler.requests()[id].stage == RequestState::Stage::Terminal);
    CHECK(scheduler.requests()[id].finish_recorded);
    CHECK(scheduler.all_terminal());
  }
}

void test_release_retries_when_queue_is_full() {
  std::printf("test_release_retries_when_queue_is_full\n");
  Handoff handoff(8, 2, 2);
  ProbePolicy scheduler(handoff);
  const RequestId id = scheduler.submit("prompt", 1);
  admit_success(scheduler, handoff, id, 1);

  CHECK(handoff.try_release(Release{100}));
  CHECK(handoff.try_release(Release{101}));
  CHECK(handoff.try_report_completion(
      Completion{id, 1, 1, 5, WorkKind::Prefill, ErrorCode::None, true,
                 false}));
  scheduler.run_once();
  CHECK(!scheduler.requests()[id].release_sent);

  Release release{};
  CHECK(handoff.try_take_release(release));
  CHECK(release.id == 100);
  scheduler.run_once();
  CHECK(scheduler.requests()[id].release_sent);
  CHECK(handoff.try_take_release(release));
  CHECK(release.id == 101);
  CHECK(handoff.try_take_release(release));
  CHECK(release.id == id);
}

void test_outstanding_epoch_blocks_publication() {
  std::printf("test_outstanding_epoch_blocks_publication\n");
  Handoff handoff(8);
  ProbePolicy scheduler(handoff);
  const RequestId id = scheduler.submit("prompt", 3);
  admit_success(scheduler, handoff, id, 2);
  scheduler.next_work = {{id, 0, 2, WorkKind::Prefill}};

  scheduler.run_once();
  Plan *first = handoff.consume_plan();
  CHECK(first != nullptr);
  scheduler.run_once();
  CHECK(handoff.consume_plan() == nullptr);
  handoff.retire_plan(first);
}

void test_retirement_racing_after_drain_never_republishes_stale_work() {
  std::printf(
      "test_retirement_racing_after_drain_never_republishes_stale_work\n");
  constexpr std::uint32_t kCycles = 256;
  constexpr std::uint32_t kQueueCapacity = 1024;
  Handoff handoff(1, 2, kQueueCapacity);
  UnitPrefillPolicy scheduler(handoff);
  const RequestId id = scheduler.submit("prompt", 1);
  scheduler.run_once();
  take_admission(handoff);
  CHECK(handoff.try_report_admission(
      AdmissionResult{id, kCycles + 8, ErrorCode::None}));
  scheduler.run_once();

  Plan *current = handoff.consume_plan();
  CHECK(current != nullptr);
  std::set<std::uint32_t> executed_positions;
  int duplicates = 0;
  int forced_interleavings = 0;

  for (std::uint32_t cycle = 0; cycle < kCycles && current != nullptr;
       ++cycle) {
    const WorkItem work = current->work.front();
    if (!executed_positions.insert(work.token_begin).second) {
      ++duplicates;
    }

    for (std::uint32_t i = 0; i < kQueueCapacity; ++i) {
      CHECK(handoff.try_acknowledge_release(ReleaseAck{100000 + i}));
    }

    std::atomic<bool> tick_done{false};
    std::thread scheduler_tick([&] {
      scheduler.run_once();
      tick_done.store(true, std::memory_order_release);
    });

    while (!tick_done.load(std::memory_order_acquire)) {
      if (handoff.try_acknowledge_release(ReleaseAck{200000 + cycle})) {
        ++forced_interleavings;
        break;
      }
      std::this_thread::yield();
    }

    CHECK(handoff.try_report_completion(
        Completion{id, work.token_end, 0, 0, WorkKind::Prefill,
                   ErrorCode::None, false, false}));
    handoff.retire_plan(current);
    scheduler_tick.join();

    Plan *raced = handoff.consume_plan();
    if (raced != nullptr) {
      const WorkItem raced_work = raced->work.front();
      if (!executed_positions.insert(raced_work.token_begin).second) {
        ++duplicates;
      }
      CHECK(handoff.try_report_completion(
          Completion{id, raced_work.token_end, 0, 0, WorkKind::Prefill,
                     ErrorCode::None, false, false}));
      handoff.retire_plan(raced);
    }

    scheduler.run_once();
    current = handoff.consume_plan();
  }

  CHECK(forced_interleavings > 0);
  CHECK(duplicates == 0);
}

void test_terminal_requests_are_never_rescheduled() {
  std::printf("test_terminal_requests_are_never_rescheduled\n");
  Handoff handoff(8);
  ProbePolicy scheduler(handoff);
  const RequestId id = scheduler.submit("bad", 1);
  scheduler.run_once();
  take_admission(handoff);
  CHECK(handoff.try_report_admission(
      AdmissionResult{id, 0, ErrorCode::TokenizationFailed}));
  scheduler.next_work = {{id, 0, 1, WorkKind::Prefill}};
  scheduler.run_once();
  CHECK(scheduler.all_terminal());
  CHECK(handoff.consume_plan() == nullptr);
}

void test_run_fatal_covers_all_states_and_waits_for_cleanup() {
  std::printf("test_run_fatal_covers_all_states_and_waits_for_cleanup\n");
  Handoff handoff(8, 2, 2);
  ProbePolicy scheduler(handoff);
  const RequestId owned = scheduler.submit("owned", 4);
  admit_success(scheduler, handoff, owned, 2);
  const RequestId absent_from_failed_plan = scheduler.submit("absent", 4);
  admit_success(scheduler, handoff, absent_from_failed_plan, 2);

  scheduler.next_work = {{owned, 0, 2, WorkKind::Prefill}};
  scheduler.run_once();
  Plan *failed_plan = handoff.consume_plan();
  CHECK(failed_plan != nullptr);
  CHECK(failed_plan->work.size() == 1);
  CHECK(failed_plan->work.front().id == owned);

  const RequestId raced_success = scheduler.submit("race", 4);
  const RequestId raced_error = scheduler.submit("reject", 4);
  const RequestId not_sent = scheduler.submit("local", 4);
  scheduler.run_once();
  const Admission race_admission = take_admission(handoff);
  const Admission reject_admission = take_admission(handoff);
  CHECK(race_admission.id == raced_success);
  CHECK(reject_admission.id == raced_error);

  // Refill the admission channel so not_sent cannot be flushed before the
  // fatal (the scheduler observes fatals after its nonblocking flush).
  Admission blocker_a{100, "blocker a", 1};
  Admission blocker_b{101, "blocker b", 1};
  CHECK(handoff.try_admit(std::move(blocker_a)));
  CHECK(handoff.try_admit(std::move(blocker_b)));
  CHECK(handoff.try_report_fatal(RunFatal{ErrorCode::DecodeFailed}));
  handoff.retire_plan(failed_plan);
  scheduler.run_once();

  // A completion already produced before the fatal can become visible on the
  // following scheduler iteration. It may fold progress, but must not undo
  // cleanup by moving an owned request back to a schedulable stage.
  CHECK(handoff.try_report_completion(Completion{
      absent_from_failed_plan, 1, 0, 0, WorkKind::Prefill, ErrorCode::None,
      false, false}));
  scheduler.run_once();

  CHECK(scheduler.requests()[owned].stage ==
        RequestState::Stage::PendingRelease);
  CHECK(scheduler.requests()[absent_from_failed_plan].stage ==
        RequestState::Stage::PendingRelease);
  CHECK(scheduler.requests()[not_sent].stage == RequestState::Stage::Terminal);
  CHECK(scheduler.requests()[not_sent].terminal_error ==
        ErrorCode::DecodeFailed);
  CHECK(!scheduler.all_terminal());

  Release owned_release{};
  Release absent_release{};
  CHECK(handoff.try_take_release(owned_release));
  CHECK(owned_release.id == owned);
  CHECK(handoff.try_take_release(absent_release));
  CHECK(absent_release.id == absent_from_failed_plan);
  CHECK(handoff.try_acknowledge_release(ReleaseAck{owned}));
  CHECK(handoff.try_acknowledge_release(
      ReleaseAck{absent_from_failed_plan}));
  CHECK(handoff.try_report_admission(AdmissionResult{
      raced_success, 3, ErrorCode::None}));
  CHECK(handoff.try_report_admission(AdmissionResult{
      raced_error, 0, ErrorCode::EnvironmentStopped}));
  scheduler.run_once();

  CHECK(scheduler.requests()[owned].stage == RequestState::Stage::Terminal);
  CHECK(scheduler.requests()[absent_from_failed_plan].stage ==
        RequestState::Stage::Terminal);
  CHECK(scheduler.requests()[raced_success].stage ==
        RequestState::Stage::PendingRelease);
  CHECK(scheduler.requests()[raced_error].stage ==
        RequestState::Stage::Terminal);
  CHECK(!scheduler.all_terminal());
  Release raced_release{};
  CHECK(handoff.try_take_release(raced_release));
  CHECK(raced_release.id == raced_success);
  CHECK(handoff.try_acknowledge_release(ReleaseAck{raced_success}));
  scheduler.run_once();
  CHECK(scheduler.all_terminal());
  CHECK(handoff.consume_plan() == nullptr);
}

void test_request_stop_drains_and_run_requests_environment_stop() {
  std::printf("test_request_stop_drains_and_run_requests_environment_stop\n");
  Handoff handoff(8);
  ProbePolicy scheduler(handoff);
  scheduler.submit("never admitted", 2);
  scheduler.request_stop();
  scheduler.run();
  CHECK(scheduler.all_terminal());
  CHECK(handoff.stop_requested());
  CHECK(scheduler.requests()[0].terminal_error ==
        ErrorCode::EnvironmentStopped);
}

void test_request_stop_waits_for_inflight_admission_resolution() {
  std::printf("test_request_stop_waits_for_inflight_admission_resolution\n");
  for (const bool raced_success : {false, true}) {
    Handoff handoff(8);
    ProbePolicy scheduler(handoff);
    const RequestId id = scheduler.submit("in flight", 2);
    scheduler.run_once();
    take_admission(handoff);

    scheduler.request_stop();
    scheduler.run_once();
    CHECK(!scheduler.all_terminal());
    CHECK(!handoff.stop_requested());

    CHECK(handoff.try_report_admission(AdmissionResult{
        id, raced_success ? 3u : 0u,
        raced_success ? ErrorCode::None : ErrorCode::EnvironmentStopped}));
    scheduler.run_once();

    Release release{};
    if (raced_success) {
      CHECK(scheduler.requests()[id].stage ==
            RequestState::Stage::PendingRelease);
      CHECK(handoff.try_take_release(release));
      CHECK(release.id == id);
      CHECK(!handoff.stop_requested());
      CHECK(handoff.try_acknowledge_release(ReleaseAck{id}));
      scheduler.run_once();
    } else {
      CHECK(!handoff.try_take_release(release));
    }

    CHECK(scheduler.all_terminal());
    CHECK(!handoff.stop_requested());
    scheduler.run();
    CHECK(handoff.stop_requested());
  }
}

void test_request_stop_retries_owned_release_before_final_stop() {
  std::printf("test_request_stop_retries_owned_release_before_final_stop\n");
  Handoff handoff(8, 2, 2);
  ProbePolicy scheduler(handoff);
  const RequestId id = scheduler.submit("owned", 2);
  admit_success(scheduler, handoff, id, 3);

  CHECK(handoff.try_release(Release{100}));
  CHECK(handoff.try_release(Release{101}));
  scheduler.request_stop();
  scheduler.run_once();
  CHECK(scheduler.requests()[id].stage ==
        RequestState::Stage::PendingRelease);
  CHECK(!scheduler.requests()[id].release_sent);
  CHECK(!scheduler.all_terminal());
  CHECK(!handoff.stop_requested());

  Release release{};
  CHECK(handoff.try_take_release(release));
  CHECK(release.id == 100);
  scheduler.run_once();
  CHECK(scheduler.requests()[id].release_sent);
  CHECK(!handoff.stop_requested());
  CHECK(handoff.try_take_release(release));
  CHECK(release.id == 101);
  CHECK(handoff.try_take_release(release));
  CHECK(release.id == id);
  CHECK(handoff.try_acknowledge_release(ReleaseAck{id}));
  scheduler.run_once();
  CHECK(scheduler.all_terminal());
  CHECK(!handoff.stop_requested());
  scheduler.run();
  CHECK(handoff.stop_requested());
}

} // namespace

int main() {
  test_invalid_plans_are_never_published();
  test_exact_decode_range_is_publishable();
  test_work_for_non_schedulable_stages_is_rejected();
  test_protocol_violation_stops_admissions_and_drains_races();
  test_protocol_violation_retries_release_backpressure();
  test_protocol_violation_fails_notsent_locally_and_stops_flushing();
  test_protocol_violation_remains_the_drain_cause_after_later_fatal();
  test_protocol_violation_remains_the_error_after_late_completion();
  test_request_error_survives_a_later_run_fatal();
  test_queued_fatal_still_drains_admission_flushed_before_observation();
  test_submit_assigns_stable_ids_without_touching_handoff();
  test_zero_token_budget_is_rejected();
  test_empty_policy_plan_fails_all_schedulable_requests_and_cleans_up();
  test_run_once_flushes_admissions_with_backpressure();
  test_admission_transitions_and_errors_without_release();
  test_admission_failures_are_terminal_and_all_terminal_is_exact();
  test_absolute_completions_are_idempotent_and_prefill_can_generate();
  test_output_is_appended_before_matching_completion_is_folded();
  test_eos_and_output_limit_release_then_ack_terminal();
  test_release_retries_when_queue_is_full();
  test_outstanding_epoch_blocks_publication();
  test_retirement_racing_after_drain_never_republishes_stale_work();
  test_terminal_requests_are_never_rescheduled();
  test_run_fatal_covers_all_states_and_waits_for_cleanup();
  test_request_stop_drains_and_run_requests_environment_stop();
  test_request_stop_waits_for_inflight_admission_resolution();
  test_request_stop_retries_owned_release_before_final_stop();

  if (g_failures != 0) {
    std::printf("\n%d check(s) failed\n", g_failures);
    return 1;
  }
  std::printf("\nall scheduler checks passed\n");
  return 0;
}
