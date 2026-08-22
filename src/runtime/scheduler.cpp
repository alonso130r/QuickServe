#include "scheduler.hpp"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <thread>
#include <unordered_set>
#include <utility>

Scheduler::Scheduler(Handoff &handoff, std::uint32_t token_budget)
    : token_budget_(token_budget), handoff_(handoff) {
  if (token_budget == 0) {
    throw std::invalid_argument("Scheduler token budget must be positive");
  }
}

RequestId Scheduler::submit(std::string prompt,
                            std::uint32_t max_output_tokens) {
  const RequestId id = static_cast<RequestId>(requests_.size());
  RequestState state{};
  state.id = id;
  state.max_output_tokens = max_output_tokens;
  state.arrival_time = RequestState::Clock::now();
  requests_.push_back(std::move(state));
  pending_admissions_.push_back(
      Admission{id, std::move(prompt), max_output_tokens});
  return id;
}

bool Scheduler::run_once() {
  if (stop_requested_.load(std::memory_order_acquire) && !draining_) {
    enter_draining(ErrorCode::EnvironmentStopped);
  }
  if (!draining_) {
    flush_pending_admissions();
  }

  // Snapshot completion before draining. If retirement races with this run's
  // drains, defer publication until the next iteration, whose acquire will
  // make the corresponding completions visible before policy state is read.
  const bool plan_done = handoff_.completed_epoch() >= published_epoch_;

  drain_admission_results();
  drain_outputs();
  drain_completions();
  drain_release_acks();
  drain_fatals();
  retry_releases();

  if (all_terminal()) {
    return false;
  }
  if (draining_ || !plan_done) {
    return true;
  }

  const bool has_schedulable =
      std::any_of(requests_.begin(), requests_.end(), [](const auto &state) {
        return state.stage == RequestState::Stage::Prefill ||
               state.stage == RequestState::Stage::Decode;
      });
  if (!has_schedulable) {
    return true;
  }
  const bool release_pending =
      std::any_of(requests_.begin(), requests_.end(), [](const auto &state) {
        return state.stage == RequestState::Stage::PendingRelease;
      });

  Plan &plan = handoff_.begin();
  build_plan(plan);
  if (!plan.work.empty()) {
    SchedulerError validation = validate_plan(plan);
    if (!validation.valid) {
      last_error_ = std::move(validation);
      enter_draining(ErrorCode::ProtocolViolation);
    } else {
      const RequestState::TimePoint start_time = RequestState::Clock::now();
      const std::uint64_t published_epoch = handoff_.commit();
      if (published_epoch != 0) {
        published_epoch_ = published_epoch;
        for (const WorkItem &work : plan.work) {
          RequestState &state = requests_[work.id];
          if (!state.start_recorded) {
            state.start_time = start_time;
            state.start_recorded = true;
          }
        }
      }
    }
  } else if (!release_pending) {
    // An empty plan is only legitimate while a policy deliberately waits for
    // cleanup. Otherwise schedulable work has no path to make progress.
    last_error_.valid = false;
    last_error_.code = ErrorCode::ProtocolViolation;
    last_error_.detail =
        "policy produced no work for schedulable requests";
    last_error_.has_offending_work = false;
    enter_draining(ErrorCode::ProtocolViolation);
    retry_releases();
  }
  return true;
}

void Scheduler::run() {
  while (run_once()) {
    std::this_thread::yield();
  }
  handoff_.request_stop();
}

void Scheduler::request_stop() {
  stop_requested_.store(true, std::memory_order_release);
}

bool Scheduler::all_terminal() const {
  return std::all_of(requests_.begin(), requests_.end(), [](const auto &state) {
    return state.stage == RequestState::Stage::Terminal;
  });
}

const std::vector<RequestState> &Scheduler::requests() const {
  return requests_;
}

const SchedulerError &Scheduler::last_error() const { return last_error_; }

void Scheduler::flush_pending_admissions() {
  while (!pending_admissions_.empty()) {
    Admission &admission = pending_admissions_.front();
    const RequestId id = admission.id;
    if (id >= requests_.size() ||
        requests_[id].admission !=
            RequestState::AdmissionOwnership::NotSent ||
        requests_[id].stage != RequestState::Stage::PendingAdmission) {
      pending_admissions_.pop_front();
      continue;
    }
    if (!handoff_.try_admit(std::move(admission))) {
      return;
    }
    requests_[id].admission = RequestState::AdmissionOwnership::InFlight;
    pending_admissions_.pop_front();
  }
}

void Scheduler::drain_admission_results() {
  AdmissionResult result{};
  while (handoff_.try_take_admission_result(result)) {
    if (result.id >= requests_.size()) {
      continue;
    }
    RequestState &state = requests_[result.id];
    if (state.stage == RequestState::Stage::Terminal ||
        state.admission != RequestState::AdmissionOwnership::InFlight) {
      continue;
    }
    if (result.error != ErrorCode::None) {
      const ErrorCode error = state.terminal_error == ErrorCode::None
                                  ? result.error
                                  : state.terminal_error;
      finish_without_release(state, error);
      continue;
    }

    state.admission = RequestState::AdmissionOwnership::EnvironmentOwned;
    state.prompt_length = result.prompt_tokens;
    if (draining_ || state.max_output_tokens == 0) {
      move_to_pending_release(state,
                              draining_ ? drain_error_ : ErrorCode::None);
    } else {
      state.stage = RequestState::Stage::Prefill;
    }
  }
}

void Scheduler::drain_outputs() {
  OutputPiece piece{};
  while (handoff_.try_take_output(piece)) {
    if (piece.id < requests_.size() &&
        requests_[piece.id].stage != RequestState::Stage::Terminal) {
      requests_[piece.id].output_text.append(piece.text);
    }
  }
}

void Scheduler::drain_completions() {
  Completion completion{};
  while (handoff_.try_take_completion(completion)) {
    if (completion.id >= requests_.size()) {
      continue;
    }
    RequestState &state = requests_[completion.id];
    if (state.stage == RequestState::Stage::Terminal ||
        state.admission !=
            RequestState::AdmissionOwnership::EnvironmentOwned) {
      continue;
    }

    state.prefill_position =
        std::max(state.prefill_position, completion.prefill_position);
    if (completion.generated_token && !state.first_token_recorded) {
      state.first_token_time = RequestState::Clock::now();
      state.first_token_recorded = true;
    }
    if (completion.decoded_tokens > state.decoded_count) {
      if (completion.generated_token) {
        state.output_token_ids.push_back(completion.token);
      }
      state.decoded_count = completion.decoded_tokens;
    }

    if (completion.error != ErrorCode::None || completion.eos ||
        state.decoded_count >= state.max_output_tokens) {
      move_to_pending_release(state, completion.error);
    } else if (state.stage != RequestState::Stage::PendingRelease) {
      state.stage = state.prefill_done() ? RequestState::Stage::Decode
                                         : RequestState::Stage::Prefill;
    }
  }
}

void Scheduler::drain_release_acks() {
  ReleaseAck ack{};
  while (handoff_.try_take_release_ack(ack)) {
    if (ack.id >= requests_.size()) {
      continue;
    }
    RequestState &state = requests_[ack.id];
    if (state.stage != RequestState::Stage::PendingRelease ||
        !state.release_sent) {
      continue;
    }
    state.stage = RequestState::Stage::Terminal;
    state.admission = RequestState::AdmissionOwnership::NotSent;
    state.finish_time = RequestState::Clock::now();
    state.finish_recorded = true;
  }
}

void Scheduler::drain_fatals() {
  RunFatal fatal{};
  while (handoff_.try_take_fatal(fatal)) {
    enter_draining(fatal.error == ErrorCode::None
                       ? ErrorCode::EnvironmentStopped
                       : fatal.error);
  }
}

void Scheduler::retry_releases() {
  for (RequestState &state : requests_) {
    if (state.stage == RequestState::Stage::PendingRelease &&
        !state.release_sent && handoff_.try_release(Release{state.id})) {
      state.release_sent = true;
    }
  }
}

SchedulerError Scheduler::validate_plan(const Plan &plan) const {
  std::unordered_set<RequestId> seen;
  std::uint64_t total_cost = 0;

  const auto violation = [](const WorkItem &work, std::string detail) {
    SchedulerError result{};
    result.valid = false;
    result.code = ErrorCode::ProtocolViolation;
    result.detail = std::move(detail);
    result.has_offending_work = true;
    result.offending_work = work;
    return result;
  };

  for (const WorkItem &work : plan.work) {
    if (work.id >= requests_.size()) {
      return violation(work, "plan references an unknown request");
    }
    if (work.token_begin >= work.token_end) {
      return violation(work, "work range must be nonempty and increasing");
    }
    if (!seen.insert(work.id).second) {
      return violation(work, "plan contains duplicate work for a request");
    }

    const std::uint64_t cost =
        static_cast<std::uint64_t>(work.token_end) - work.token_begin;
    total_cost += cost;
    if (total_cost > token_budget_) {
      return violation(work, "plan exceeds the token budget");
    }

    const RequestState &state = requests_[work.id];
    switch (work.kind) {
    case WorkKind::Prefill:
      if (state.stage != RequestState::Stage::Prefill) {
        return violation(work,
                         "prefill work requires a request in prefill");
      }
      if (work.token_begin != state.prefill_position) {
        return violation(
            work, "prefill must start at the acknowledged position");
      }
      if (work.token_end > state.prompt_length) {
        return violation(work, "prefill work extends past the prompt");
      }
      break;

    case WorkKind::Decode: {
      if (state.stage != RequestState::Stage::Decode) {
        return violation(work, "decode work requires a request in decode");
      }
      if (state.decoded_count < 1) {
        return violation(
            work, "decode work requires an already generated token");
      }
      const std::uint64_t expected_end =
          static_cast<std::uint64_t>(state.prompt_length) +
          state.decoded_count;
      if (expected_end > std::numeric_limits<std::uint32_t>::max()) {
        return violation(work, "decode token position overflows");
      }
      if (work.token_begin != expected_end - 1 ||
          work.token_end != expected_end) {
        return violation(work, "decode work must name the exact next range");
      }
      break;
    }

    default:
      return violation(work, "plan contains an unknown work kind");
    }
  }

  return SchedulerError{};
}

void Scheduler::enter_draining(ErrorCode error) {
  if (draining_) {
    return;
  }
  draining_ = true;
  drain_error_ = error == ErrorCode::None ? ErrorCode::EnvironmentStopped
                                          : error;
  for (RequestState &state : requests_) {
    if (state.stage == RequestState::Stage::Terminal) {
      continue;
    }
    if (state.terminal_error == ErrorCode::None) {
      state.terminal_error = drain_error_;
    }
    switch (state.admission) {
    case RequestState::AdmissionOwnership::NotSent:
      finish_without_release(state, state.terminal_error);
      break;
    case RequestState::AdmissionOwnership::InFlight:
      break;
    case RequestState::AdmissionOwnership::EnvironmentOwned:
      move_to_pending_release(state, state.terminal_error);
      break;
    }
  }
}

void Scheduler::move_to_pending_release(RequestState &state, ErrorCode error) {
  if (state.terminal_error == ErrorCode::None && error != ErrorCode::None) {
    state.terminal_error = error;
  }
  state.stage = RequestState::Stage::PendingRelease;
}

void Scheduler::finish_without_release(RequestState &state, ErrorCode error) {
  state.stage = RequestState::Stage::Terminal;
  state.admission = RequestState::AdmissionOwnership::NotSent;
  state.terminal_error = error;
  state.finish_time = RequestState::Clock::now();
  state.finish_recorded = true;
}
