#pragma once

#include <atomic>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <vector>

#include "protocol.hpp"
#include "spsc.hpp"

// One environment iteration's absolute work intent. Work items describe token
// positions, never deltas, so scheduler state remains the source of truth.
struct Plan {
  std::vector<WorkItem> work;
  std::uint64_t epoch = 0;
};

// The sole object shared by the scheduler and environment threads.
//
// Every message channel is bounded and SPSC. The method groups below identify
// its sole producer and consumer. A false push leaves the caller's message
// untouched; that producer owns the retry buffer and must retry after yielding.
// In particular, no helper thread may become a second producer.
//
// Plans use stable heap addresses and a single atomic publication slot. The
// scheduler keeps at most one epoch outstanding; the environment reports all
// completions before retiring that plan. retire_plan() is the epoch commit
// point, and its release store makes earlier reports visible to the scheduler.
//
// After reporting RunFatal, the environment stops consuming plans but keeps
// draining admissions that the scheduler already queued. It reports
// EnvironmentStopped for an admission not completed before the fatal, and it
// remains the consumer of releases and producer of admission results and
// release acknowledgements until the scheduler calls request_stop(). Handoff
// only transports that state; it does not implement the lifecycle or
// introduce another producer.
class Handoff {
public:
  explicit Handoff(std::size_t plan_capacity, std::size_t pool_size = 3,
                   std::size_t queue_capacity = 64)
      : pool_size_(validate_pool_size(pool_size, queue_capacity)),
        retired_(queue_capacity), admissions_(queue_capacity),
        releases_(queue_capacity), admission_results_(queue_capacity),
        completions_(queue_capacity), outputs_(queue_capacity),
        release_acks_(queue_capacity), fatals_(queue_capacity),
        reserve_(plan_capacity) {
    spare_ = allocate_plan();
    for (std::size_t i = 1; i < pool_size_; ++i) {
      const bool pushed = retired_.try_push(allocate_plan());
      assert(pushed && "retired_ sized too small for the pool");
      (void)pushed;
    }
  }

  Handoff(const Handoff &) = delete;
  Handoff &operator=(const Handoff &) = delete;
  Handoff(Handoff &&) = delete;
  Handoff &operator=(Handoff &&) = delete;

  // --- scheduler thread: sole producer ----------------------------------

  // On false, admission has not been moved from; the scheduler owns it and
  // must retry later.
  [[nodiscard]] bool try_admit(Admission &&admission) {
    return admissions_.try_push(std::move(admission));
  }

  // On false, the scheduler retains release and must retry later.
  [[nodiscard]] bool try_release(const Release &release) {
    return releases_.try_push(release);
  }

  [[nodiscard]] Plan &begin() {
    spare_->work.clear();
    return *spare_;
  }

  // Empty plans are deliberately not assigned an epoch or published.
  [[nodiscard]] std::uint64_t commit() {
    if (spare_->work.empty()) {
      return 0;
    }

    spare_->epoch = ++epoch_;
    const std::uint64_t published = spare_->epoch;

    Plan *old = published_.exchange(spare_, std::memory_order_release);
    if (old != nullptr) {
      spare_ = old;
      return published;
    }

    if (!retired_.try_pop(spare_)) {
      spare_ = allocate_plan();
    }
    return published;
  }

  void request_stop() {
    stop_requested_.store(true, std::memory_order_release);
  }

  // --- environment thread: sole consumer -------------------------------

  [[nodiscard]] bool try_take_admission(Admission &out) {
    return admissions_.try_pop(out);
  }

  [[nodiscard]] bool try_take_release(Release &out) {
    return releases_.try_pop(out);
  }

  [[nodiscard]] Plan *consume_plan() {
    return published_.exchange(nullptr, std::memory_order_acquire);
  }

  // --- environment thread: sole producer -------------------------------

  // Every false return leaves the input untouched. The environment owns and
  // retries the corresponding message before producing a later one on that
  // channel.
  [[nodiscard]] bool try_report_admission(const AdmissionResult &result) {
    return admission_results_.try_push(result);
  }

  [[nodiscard]] bool try_report_completion(const Completion &completion) {
    return completions_.try_push(completion);
  }

  [[nodiscard]] bool try_report_output(OutputPiece &&piece) {
    return outputs_.try_push(std::move(piece));
  }

  [[nodiscard]] bool try_acknowledge_release(const ReleaseAck &ack) {
    return release_acks_.try_push(ack);
  }

  [[nodiscard]] bool try_report_fatal(const RunFatal &fatal) {
    return fatals_.try_push(fatal);
  }

  // CONTRACT: report every completion for plan before calling retire_plan().
  void retire_plan(Plan *plan) {
    assert(plan != nullptr);
    const std::uint64_t epoch = plan->epoch;
    (void)retired_.try_push(plan);
    completed_epoch_.store(epoch, std::memory_order_release);
  }

  // --- scheduler thread: sole consumer ----------------------------------

  [[nodiscard]] bool try_take_admission_result(AdmissionResult &out) {
    return admission_results_.try_pop(out);
  }

  [[nodiscard]] bool try_take_completion(Completion &out) {
    return completions_.try_pop(out);
  }

  [[nodiscard]] bool try_take_output(OutputPiece &out) {
    return outputs_.try_pop(out);
  }

  [[nodiscard]] bool try_take_release_ack(ReleaseAck &out) {
    return release_acks_.try_pop(out);
  }

  [[nodiscard]] bool try_take_fatal(RunFatal &out) {
    return fatals_.try_pop(out);
  }

  [[nodiscard]] std::uint64_t completed_epoch() const {
    return completed_epoch_.load(std::memory_order_acquire);
  }

  // Monotonic and safe to observe from either thread.
  [[nodiscard]] bool stop_requested() const {
    return stop_requested_.load(std::memory_order_acquire);
  }

private:
  static std::size_t validate_pool_size(std::size_t pool_size,
                                        std::size_t queue_capacity) {
    if (pool_size < 2) {
      throw std::invalid_argument("Handoff plan pool must contain two plans");
    }
    if (pool_size > queue_capacity) {
      throw std::invalid_argument(
          "Handoff retired queue must hold the entire plan pool");
    }
    return pool_size;
  }

  Plan *allocate_plan() {
    pool_.push_back(std::make_unique<Plan>());
    Plan *plan = pool_.back().get();
    plan->work.reserve(reserve_);
    return plan;
  }

  const std::size_t pool_size_;
  std::atomic<std::uint64_t> completed_epoch_{0};
  std::atomic<Plan *> published_{nullptr};
  std::atomic<bool> stop_requested_{false};

  SPSCQueue<Plan *> retired_; // environment -> scheduler

  SPSCQueue<Admission> admissions_; // scheduler -> environment
  SPSCQueue<Release> releases_;     // scheduler -> environment

  SPSCQueue<AdmissionResult> admission_results_; // environment -> scheduler
  SPSCQueue<Completion> completions_;             // environment -> scheduler
  SPSCQueue<OutputPiece> outputs_;                 // environment -> scheduler
  SPSCQueue<ReleaseAck> release_acks_;             // environment -> scheduler
  SPSCQueue<RunFatal> fatals_;                     // environment -> scheduler

  // Scheduler-thread only. The environment touches Plan objects through stable
  // pointers but never mutates the owning vector.
  std::vector<std::unique_ptr<Plan>> pool_;
  Plan *spare_ = nullptr;
  std::uint64_t epoch_ = 0;
  const std::size_t reserve_;
};
