#include "scheduler.hpp"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <thread>
#include <unordered_set>
#include <utility>

Scheduler::Scheduler(Handoff &handoff, std::uint32_t token_budget,
                     ClockFunction clock)
    : token_budget_(token_budget), handoff_(handoff),
      clock_(clock ? std::move(clock)
                   : [] { return RequestState::Clock::now(); }) {
  if (token_budget == 0) {
    throw std::invalid_argument("Scheduler token budget must be positive");
  }
}

RequestId Scheduler::submit(std::string prompt,
                            std::uint32_t max_output_tokens) {
  return submit_admission(
      Admission{0, std::move(prompt), max_output_tokens});
}

RequestId Scheduler::submit_synthetic(std::uint32_t prompt_tokens,
                                      std::uint32_t max_output_tokens,
                                      OutputMode output_mode) {
  return submit_admission(Admission{0, {}, max_output_tokens, prompt_tokens,
                                    output_mode});
}

RequestId Scheduler::submit_admission(Admission admission) {
  begin_nonconst_operation();
  if (execution_started_ && std::this_thread::get_id() != scheduler_thread_) {
    throw std::logic_error(
        "submit must run on the scheduler-owning thread after execution starts");
  }
  if (next_request_id_ > std::numeric_limits<RequestId>::max()) {
    throw std::overflow_error("request ID space exhausted");
  }
  const RequestId id = static_cast<RequestId>(next_request_id_++);
  admission.id = id;
  RequestState state{};
  state.id = id;
  state.max_output_tokens = admission.max_output_tokens;
  state.synthetic_prompt_tokens = admission.synthetic_prompt_tokens;
  state.output_mode = admission.output_mode;
  state.arrival_time = now();
  live_requests_.push_back(std::move(state));
  const auto inserted = std::prev(live_requests_.end());
  try {
    requests_by_id_.emplace(id, inserted);
  } catch (...) {
    live_requests_.erase(inserted);
    throw;
  }
  try {
    pending_admissions_.push_back(std::move(admission));
  } catch (...) {
    requests_by_id_.erase(id);
    live_requests_.erase(inserted);
    throw;
  }
  ++workload_counts_.queued;
  publish_workload_counts(state.arrival_time);
  return id;
}

void Scheduler::enable_streaming_retirement(TerminalObserver observer) {
  begin_nonconst_operation();
  if (execution_started_ || !live_requests_.empty() ||
      !completed_history_.empty()) {
    throw std::logic_error(
        "streaming retirement must be enabled before requests are submitted");
  }
  if (!observer) {
    throw std::invalid_argument("terminal observer must be callable");
  }
  streaming_retirement_ = true;
  observer_enabled_ = true;
  terminal_observer_ = std::move(observer);
}

void Scheduler::set_workload_observer(WorkloadObserver observer) {
  if (execution_started_ || !live_requests_.empty())
    throw std::logic_error("workload observer must be set before execution");
  workload_observer_ = std::move(observer);
}

void Scheduler::set_batch_observer(BatchObserver observer) {
  if (execution_started_)
    throw std::logic_error("batch observer must be set before execution");
  batch_observer_ = std::move(observer);
}

void Scheduler::set_clock(ClockFunction clock) {
  if (execution_started_ || !live_requests_.empty())
    throw std::logic_error("clock must be set before execution");
  if (!clock) throw std::invalid_argument("clock must be callable");
  clock_ = std::move(clock);
}

bool Scheduler::run_once() {
  begin_nonconst_operation();
  if (!execution_started_) {
    execution_started_ = true;
    scheduler_thread_ = std::this_thread::get_id();
  }
  // Acquire this iteration's timestamp before consuming protocol messages.
  // If an injected clock fails, every message remains queued and the entire
  // scheduler state is unchanged, so cleanup or a retry can still proceed.
  current_time_ = now();
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
  retire_terminal_requests();

  if (plan_done && published_epoch_ != 0 &&
      observed_epoch_ < published_epoch_) {
    const BatchOutcome outcome{
        published_epoch_,
        published_plan_start_,
        current_time_,
        current_time_ - published_plan_start_,
        published_prefill_tokens_,
        published_decode_items_,
        published_work_items_,
        published_plan_success_,
    };
    observed_epoch_ = published_epoch_;
    on_plan_completed(outcome);
    if (batch_observer_) batch_observer_(outcome);
  }

  if (all_terminal()) {
    return finish_iteration(false);
  }
  if (draining_ || !plan_done) {
    return finish_iteration(true);
  }

  const bool has_schedulable =
      std::any_of(live_requests_.begin(), live_requests_.end(), [](const auto &state) {
        return state.stage == RequestState::Stage::Prefill ||
               state.stage == RequestState::Stage::Decode;
      });
  if (!has_schedulable) {
    return finish_iteration(true);
  }
  const bool release_pending =
      std::any_of(live_requests_.begin(), live_requests_.end(), [](const auto &state) {
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
      const RequestState::TimePoint start_time = current_time_;
      const std::uint64_t published_epoch = handoff_.commit();
      if (published_epoch != 0) {
        published_epoch_ = published_epoch;
        published_plan_start_ = start_time;
        published_prefill_tokens_ = 0;
        published_decode_items_ = 0;
        published_work_items_ =
            static_cast<std::uint32_t>(plan.work.size());
        published_plan_success_ = true;
        for (const WorkItem &work : plan.work) {
          if (work.kind == WorkKind::Prefill) {
            published_prefill_tokens_ += work.token_count();
          } else {
            ++published_decode_items_;
          }
          RequestState *state = find_request(work.id);
          if (state != nullptr && !state->start_recorded) {
            state->start_time = start_time;
            state->start_recorded = true;
            if (workload_counts_.queued == 0)
              throw std::logic_error("queued workload counter underflow");
            --workload_counts_.queued;
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
  return finish_iteration(true);
}

void Scheduler::run() {
  while (run_once()) {
    std::this_thread::yield();
  }
  handoff_.request_stop();
}

void Scheduler::request_stop() {
  stop_requested_.store(true, std::memory_order_release);
  handoff_.wake_scheduler();
}

bool Scheduler::all_terminal() const {
  return pending_admissions_.empty() &&
         std::all_of(live_requests_.begin(), live_requests_.end(), [](const auto &state) {
    return state.stage == RequestState::Stage::Terminal;
  });
}

const std::vector<RequestState> &Scheduler::requests() const {
  if (streaming_retirement_) {
    throw std::logic_error("requests() is unavailable in streaming mode");
  }
  if (inspection_cache_valid_) {
    return inspection_cache_;
  }
  inspection_cache_ = completed_history_;
  inspection_cache_.insert(inspection_cache_.end(), live_requests_.begin(),
                           live_requests_.end());
  std::sort(inspection_cache_.begin(), inspection_cache_.end(),
            [](const RequestState &a, const RequestState &b) {
              return a.id < b.id;
            });
  inspection_cache_valid_ = true;
  return inspection_cache_;
}

const SchedulerError &Scheduler::last_error() const { return last_error_; }

SchedulerWorkloadCounts Scheduler::workload_counts() const {
  return workload_counts_;
}

bool Scheduler::finish_iteration(bool keep_running) {
  publish_workload_counts(current_time_);
  return keep_running;
}

void Scheduler::publish_workload_counts(RequestState::TimePoint at) {
  if (!workload_observer_) return;
  const auto counts = workload_counts();
  if (counts.active == last_published_counts_.active &&
      counts.queued == last_published_counts_.queued) return;
  workload_observer_(at, counts);
  last_published_counts_ = counts;
}

void Scheduler::flush_pending_admissions() {
  while (!pending_admissions_.empty()) {
    Admission &admission = pending_admissions_.front();
    const RequestId id = admission.id;
    RequestState *state = find_request(id);
    if (state == nullptr ||
        state->admission !=
            RequestState::AdmissionOwnership::NotSent ||
        state->stage != RequestState::Stage::PendingAdmission) {
      pending_admissions_.pop_front();
      continue;
    }
    if (!handoff_.try_admit(std::move(admission))) {
      return;
    }
    state->admission = RequestState::AdmissionOwnership::InFlight;
    pending_admissions_.pop_front();
  }
}

void Scheduler::drain_admission_results() {
  AdmissionResult result{};
  while (handoff_.try_take_admission_result(result)) {
    RequestState *found = find_request(result.id);
    if (found == nullptr) {
      continue;
    }
    RequestState &state = *found;
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
    state.admission_succeeded = true;
    ++workload_counts_.active;
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
    RequestState *state = find_request(piece.id);
    if (state != nullptr && state->stage != RequestState::Stage::Terminal) {
      state->output_text.append(piece.text);
    }
  }
}

void Scheduler::drain_completions() {
  Completion completion{};
  while (handoff_.try_take_completion(completion)) {
    RequestState *found = find_request(completion.id);
    if (found == nullptr) {
      continue;
    }
    RequestState &state = *found;
    if (completion.error != ErrorCode::None) {
      published_plan_success_ = false;
    }
    if (state.stage == RequestState::Stage::Terminal ||
        state.admission !=
            RequestState::AdmissionOwnership::EnvironmentOwned) {
      continue;
    }

    state.prefill_position =
        std::max(state.prefill_position, completion.prefill_position);
    const bool had_first_token = state.first_token_recorded;
    const RequestState::TimePoint previous_token_time = state.last_token_time;
    if (completion.decoded_tokens > state.decoded_count) {
      if (completion.generated_token) {
        const RequestState::TimePoint token_time = current_time_;
        if (!state.first_token_recorded) {
          state.first_token_time = token_time;
          state.first_token_recorded = true;
        }
        state.last_token_time = token_time;
        state.last_token_recorded = true;
        state.output_token_ids.push_back(completion.token);
      }
      state.decoded_count = completion.decoded_tokens;
      if (completion.generated_token) {
        on_request_timing(
            {state.id,
             had_first_token ? PolicyTimingEventKind::LaterToken
                             : PolicyTimingEventKind::FirstToken,
             current_time_,
             had_first_token ? current_time_ - previous_token_time
                             : current_time_ - state.arrival_time,
             completion.error});
      }
    }

    state.eog_observed = state.eog_observed || completion.eos;
    const bool stops_at_eog =
        completion.eos && state.output_mode == OutputMode::Natural;
    if (completion.error != ErrorCode::None || stops_at_eog ||
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
    RequestState *found = find_request(ack.id);
    if (found == nullptr) {
      continue;
    }
    RequestState &state = *found;
    if (state.stage != RequestState::Stage::PendingRelease ||
        !state.release_sent) {
      continue;
    }
    const RequestState::TimePoint finish_time = current_time_;
    state.stage = RequestState::Stage::Terminal;
    state.admission = RequestState::AdmissionOwnership::NotSent;
    state.finish_time = finish_time;
    state.finish_recorded = true;
    on_request_timing({state.id, PolicyTimingEventKind::Terminal, current_time_,
                       current_time_ - state.arrival_time,
                       state.terminal_error});
    if (workload_counts_.active == 0)
      throw std::logic_error("active workload counter underflow");
    --workload_counts_.active;
    if (!state.start_recorded) {
      if (workload_counts_.queued == 0)
        throw std::logic_error("queued workload counter underflow");
      --workload_counts_.queued;
    }
    queue_terminal_marker(state.id);
  }
}

void Scheduler::drain_fatals() {
  RunFatal fatal{};
  while (handoff_.try_take_fatal(fatal)) {
    published_plan_success_ = false;
    enter_draining(fatal.error == ErrorCode::None
                       ? ErrorCode::EnvironmentStopped
                       : fatal.error);
  }
}

void Scheduler::retry_releases() {
  for (RequestState &state : live_requests_) {
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
    const RequestState *state_ptr = find_request(work.id);
    if (state_ptr == nullptr) {
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

    const RequestState &state = *state_ptr;
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
  for (RequestState &state : live_requests_) {
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
  pending_admissions_.clear();
}

void Scheduler::move_to_pending_release(RequestState &state, ErrorCode error) {
  if (state.terminal_error == ErrorCode::None && error != ErrorCode::None) {
    state.terminal_error = error;
  }
  state.stage = RequestState::Stage::PendingRelease;
}

void Scheduler::finish_without_release(RequestState &state, ErrorCode error) {
  const RequestState::TimePoint finish_time = current_time_;
  state.stage = RequestState::Stage::Terminal;
  state.admission = RequestState::AdmissionOwnership::NotSent;
  state.terminal_error = error;
  state.finish_time = finish_time;
  state.finish_recorded = true;
  if (workload_counts_.queued == 0)
    throw std::logic_error("queued workload counter underflow");
  --workload_counts_.queued;
  queue_terminal_marker(state.id);
}

void Scheduler::retire_terminal_requests() {
  recover_terminal_markers();
  while (!terminal_ready_.empty()) {
    const RequestId id = terminal_ready_.front();
    auto found = requests_by_id_.find(id);
    if (found == requests_by_id_.end() ||
        found->second->stage != RequestState::Stage::Terminal) {
      terminal_ready_.pop_front();
      continue;
    }
    LiveIterator it = found->second;

    if (streaming_retirement_) {
      if (!observer_enabled_) {
        terminal_ready_.pop_front();
        continue;
      }
      bool observed = false;
      try {
        observed = terminal_observer_(*it);
      } catch (...) {
        observed = false;
      }
      if (!observed) {
        observer_enabled_ = false;
        last_error_.valid = false;
        last_error_.code = ErrorCode::EnvironmentStopped;
        last_error_.detail = "terminal observer failed";
        last_error_.has_offending_work = false;
        enter_draining(ErrorCode::EnvironmentStopped);
        terminal_ready_.pop_front();
        continue;
      }
      // The callback may submit and rehash requests_by_id_. Never retain its
      // map iterator across that external call. The list node itself remains
      // stable under submission, but relookup also defends future callbacks
      // that legally perturb other scheduler state.
      found = requests_by_id_.find(id);
      if (found == requests_by_id_.end() ||
          found->second->stage != RequestState::Stage::Terminal) {
        terminal_ready_.pop_front();
        continue;
      }
      it = found->second;
    } else {
      completed_history_.push_back(*it);
    }

    requests_by_id_.erase(found);
    live_requests_.erase(it);
    terminal_ready_.pop_front();
  }
}

void Scheduler::recover_terminal_markers() {
  if (!terminal_marker_recovery_needed_) return;
  for (const RequestState &state : live_requests_) {
    if (state.stage != RequestState::Stage::Terminal) continue;
    if (std::find(terminal_ready_.begin(), terminal_ready_.end(), state.id) ==
        terminal_ready_.end()) {
      terminal_ready_.push_back(state.id);
    }
  }
  terminal_marker_recovery_needed_ = false;
}

void Scheduler::queue_terminal_marker(RequestId id) {
  try {
    terminal_ready_.push_back(id);
  } catch (...) {
    terminal_marker_recovery_needed_ = true;
    throw;
  }
}

RequestState *Scheduler::find_request(RequestId id) {
  const auto found = requests_by_id_.find(id);
  return found == requests_by_id_.end() ? nullptr : &*found->second;
}

const RequestState *Scheduler::find_request(RequestId id) const {
  const auto found = requests_by_id_.find(id);
  return found == requests_by_id_.end() ? nullptr : &*found->second;
}

void Scheduler::begin_nonconst_operation() {
  inspection_cache_.clear();
  inspection_cache_valid_ = false;
}

RequestState::TimePoint Scheduler::now() const { return clock_(); }
