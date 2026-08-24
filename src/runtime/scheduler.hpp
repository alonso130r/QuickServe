#pragma once

#include <atomic>
#include <chrono>
#include <cstdint>
#include <deque>
#include <functional>
#include <list>
#include <optional>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "handoff.hpp"

struct RequestState {
  using Clock = std::chrono::steady_clock;
  using TimePoint = Clock::time_point;

  enum class Stage : std::uint8_t {
    PendingAdmission,
    Prefill,
    Decode,
    PendingRelease,
    Terminal,
  };

  enum class AdmissionOwnership : std::uint8_t {
    NotSent,
    InFlight,
    EnvironmentOwned,
  };

  RequestId id = 0;
  Stage stage = Stage::PendingAdmission;
  AdmissionOwnership admission = AdmissionOwnership::NotSent;
  bool admission_succeeded = false;
  std::uint32_t prompt_length = 0;
  std::uint32_t prefill_position = 0;
  std::uint32_t decoded_count = 0;
  std::uint32_t max_output_tokens = 0;
  std::optional<std::uint32_t> synthetic_prompt_tokens;
  OutputMode output_mode = OutputMode::Natural;
  std::vector<Token> output_token_ids;
  std::string output_text;
  TimePoint arrival_time{};
  TimePoint start_time{};
  TimePoint first_token_time{};
  TimePoint last_token_time{};
  TimePoint finish_time{};
  bool start_recorded = false;
  bool first_token_recorded = false;
  bool last_token_recorded = false;
  bool finish_recorded = false;
  bool eog_observed = false;
  ErrorCode terminal_error = ErrorCode::None;
  bool release_sent = false;

  [[nodiscard]] bool prefill_done() const {
    return prefill_position >= prompt_length;
  }
};

struct SchedulerError {
  bool valid = true;
  ErrorCode code = ErrorCode::None;
  std::string detail;
  bool has_offending_work = false;
  WorkItem offending_work{};
};

struct SchedulerWorkloadCounts {
  std::uint64_t active = 0;
  std::uint64_t queued = 0;
};

// Threading contract:
// - submit() is called before the scheduler worker starts or by that same
//   scheduler-owning thread during replay. It is never a concurrent operation.
// - run_once() and run() are called only by the scheduler worker thread.
// - request_stop() may be called by another thread.
// - all_terminal() and requests() are read only by the scheduler thread or by
//   another thread after the scheduler worker has joined.
class Scheduler {
public:
  using ClockFunction = std::function<RequestState::TimePoint()>;
  using TerminalObserver = std::function<bool(const RequestState &)>;
  using WorkloadObserver =
      std::function<void(RequestState::TimePoint, SchedulerWorkloadCounts)>;

  explicit Scheduler(Handoff &handoff, std::uint32_t token_budget,
                     ClockFunction clock = {});
  virtual ~Scheduler() = default;

  Scheduler(const Scheduler &) = delete;
  Scheduler &operator=(const Scheduler &) = delete;
  Scheduler(Scheduler &&) = delete;
  Scheduler &operator=(Scheduler &&) = delete;

  RequestId submit(std::string prompt, std::uint32_t max_output_tokens);
  RequestId submit_synthetic(std::uint32_t prompt_tokens,
                             std::uint32_t max_output_tokens,
                             OutputMode output_mode = OutputMode::Natural);
  void enable_streaming_retirement(TerminalObserver observer);
  void set_workload_observer(WorkloadObserver observer);
  void set_clock(ClockFunction clock);
  bool run_once();
  void run();
  void request_stop();
  [[nodiscard]] bool all_terminal() const;
  [[nodiscard]] const std::vector<RequestState> &requests() const;
  [[nodiscard]] const SchedulerError &last_error() const;
  [[nodiscard]] SchedulerWorkloadCounts workload_counts() const;

protected:
  virtual void build_plan(Plan &out) = 0;
  [[nodiscard]] const std::list<RequestState> &policy_requests() const {
    return live_requests_;
  }

  const std::uint32_t token_budget_;

private:
  RequestId submit_admission(Admission admission);
  void flush_pending_admissions();
  void drain_admission_results();
  void drain_outputs();
  void drain_completions();
  void drain_release_acks();
  void drain_fatals();
  void retry_releases();
  void queue_terminal_marker(RequestId id);
  void recover_terminal_markers();
  void retire_terminal_requests();
  [[nodiscard]] SchedulerError validate_plan(const Plan &plan) const;
  void enter_draining(ErrorCode error);
  void move_to_pending_release(RequestState &state, ErrorCode error);
  void finish_without_release(RequestState &state, ErrorCode error);
  [[nodiscard]] RequestState *find_request(RequestId id);
  [[nodiscard]] const RequestState *find_request(RequestId id) const;
  void begin_nonconst_operation();
  [[nodiscard]] RequestState::TimePoint now() const;
  bool finish_iteration(bool keep_running);
  void publish_workload_counts(RequestState::TimePoint at);

  Handoff &handoff_;
  // Policies receive only policy_requests(); lifecycle state remains owned and
  // mutable exclusively by Scheduler's non-virtual control path.
  using LiveList = std::list<RequestState>;
  using LiveIterator = LiveList::iterator;
  LiveList live_requests_;
  std::unordered_map<RequestId, LiveIterator> requests_by_id_;
  std::vector<RequestState> completed_history_;
  mutable std::vector<RequestState> inspection_cache_;
  mutable bool inspection_cache_valid_ = false;
  std::deque<Admission> pending_admissions_;
  std::deque<RequestId> terminal_ready_;
  bool terminal_marker_recovery_needed_ = false;
  std::uint64_t next_request_id_ = 0;
  std::uint64_t published_epoch_ = 0;
  std::atomic<bool> stop_requested_{false};
  bool draining_ = false;
  ErrorCode drain_error_ = ErrorCode::None;
  SchedulerError last_error_{};
  ClockFunction clock_;
  RequestState::TimePoint current_time_{};
  TerminalObserver terminal_observer_;
  bool streaming_retirement_ = false;
  bool observer_enabled_ = false;
  bool execution_started_ = false;
  WorkloadObserver workload_observer_;
  SchedulerWorkloadCounts last_published_counts_{};
  SchedulerWorkloadCounts workload_counts_{};
  std::thread::id scheduler_thread_{};
};
