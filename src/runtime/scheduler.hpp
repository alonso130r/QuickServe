#pragma once

#include <atomic>
#include <chrono>
#include <cstdint>
#include <deque>
#include <string>
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
  std::uint32_t prompt_length = 0;
  std::uint32_t prefill_position = 0;
  std::uint32_t decoded_count = 0;
  std::uint32_t max_output_tokens = 0;
  std::vector<Token> output_token_ids;
  std::string output_text;
  TimePoint arrival_time{};
  TimePoint start_time{};
  TimePoint first_token_time{};
  TimePoint finish_time{};
  bool start_recorded = false;
  bool first_token_recorded = false;
  bool finish_recorded = false;
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

// Threading contract:
// - submit() is called only before the scheduler worker starts. Supporting
//   concurrent submission later requires synchronization around owned state.
// - run_once() and run() are called only by the scheduler worker thread.
// - request_stop() may be called by another thread.
// - all_terminal() and requests() are read only by the scheduler thread or by
//   another thread after the scheduler worker has joined.
class Scheduler {
public:
  explicit Scheduler(Handoff &handoff, std::uint32_t token_budget);
  virtual ~Scheduler() = default;

  Scheduler(const Scheduler &) = delete;
  Scheduler &operator=(const Scheduler &) = delete;
  Scheduler(Scheduler &&) = delete;
  Scheduler &operator=(Scheduler &&) = delete;

  RequestId submit(std::string prompt, std::uint32_t max_output_tokens);
  bool run_once();
  void run();
  void request_stop();
  [[nodiscard]] bool all_terminal() const;
  [[nodiscard]] const std::vector<RequestState> &requests() const;
  [[nodiscard]] const SchedulerError &last_error() const;

protected:
  virtual void build_plan(Plan &out) = 0;
  [[nodiscard]] const std::vector<RequestState> &policy_requests() const {
    return requests_;
  }

  const std::uint32_t token_budget_;

private:
  void flush_pending_admissions();
  void drain_admission_results();
  void drain_outputs();
  void drain_completions();
  void drain_release_acks();
  void drain_fatals();
  void retry_releases();
  [[nodiscard]] SchedulerError validate_plan(const Plan &plan) const;
  void enter_draining(ErrorCode error);
  void move_to_pending_release(RequestState &state, ErrorCode error);
  void finish_without_release(RequestState &state, ErrorCode error);

  Handoff &handoff_;
  // Policies receive only policy_requests(); lifecycle state remains owned and
  // mutable exclusively by Scheduler's non-virtual control path.
  std::vector<RequestState> requests_;
  std::deque<Admission> pending_admissions_;
  std::uint64_t published_epoch_ = 0;
  std::atomic<bool> stop_requested_{false};
  bool draining_ = false;
  ErrorCode drain_error_ = ErrorCode::None;
  SchedulerError last_error_{};
};
