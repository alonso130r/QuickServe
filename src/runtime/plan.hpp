#pragma once

#include <atomic>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "request.cpp"
#include "spsc.hpp"

// One iteration's worth of work: the decodes to step and the prefill chunks to
// advance, sized by the caller against the executor's token budget (n_batch).
struct Plan {
  std::vector<SplitRequest> work;
  std::uint64_t epoch = 0;
};

// The only object shared between the scheduler thread and the runtime thread.
//
// Plans travel scheduler -> runtime through a single atomic slot: one exchange
// publishes an entire batch indivisibly, so the runtime observes either the
// whole new plan or the whole previous one, never a seam. Completions travel
// runtime -> scheduler through an SPSC queue.
//
// Publishing overwrites any plan the runtime has not yet picked up, so a plan
// can be skipped entirely. That is safe *only* because a plan states absolute
// intent ("request 7's prefill should reach token 256") rather than a delta
// ("advance request 7 by 256 tokens"): dropping it changes nothing, because the
// next plan is recomputed from the same unadvanced scheduler state. Never let a
// plan encode a delta.
class Handoff {
public:
  // `plan_capacity` is the expected work items per plan; buffers reserve it
  // once so the steady-state publish path never allocates.
  explicit Handoff(std::size_t plan_capacity, std::size_t pool_size = 3,
                   std::size_t queue_capacity = 64)
      : retired_(queue_capacity), completions_(queue_capacity),
        reserve_(plan_capacity) {
    assert(pool_size >= 2 && "a pool of one cannot survive a consumed plan");
    assert(pool_size <= queue_capacity && "retired_ must hold the whole pool");

    spare_ = allocate_plan();
    for (std::size_t i = 1; i < pool_size; ++i) {
      const bool pushed = retired_.try_push(allocate_plan());
      assert(pushed && "retired_ sized too small for the pool");
      (void)pushed;
    }
  }

  Handoff(const Handoff &) = delete;
  Handoff &operator=(const Handoff &) = delete;
  Handoff(Handoff &&) = delete;
  Handoff &operator=(Handoff &&) = delete;

  // --- scheduler thread ---------------------------------------------------

  // Hand back the scheduler-owned buffer, cleared and ready to be filled. The
  // runtime cannot see it until commit().
  [[nodiscard]] Plan &begin() {
    spare_->work.clear();
    return *spare_;
  }

  // Publish the buffer from begin() and re-arm a fresh spare.
  void commit() {
    spare_->epoch = ++epoch_;

    // Release pairs with the runtime's acquire in consume(): it is what carries
    // the vector's contents across, not merely the pointer value. exchange
    // rather than compare-exchange, so there is no ABA hazard.
    Plan *old = published_.exchange(spare_, std::memory_order_release);

    if (old != nullptr) {
      spare_ = old; // runtime never took it; reuse immediately
      return;
    }

    // The runtime took the previous buffer and still owns it. Reclaim a retired
    // one, or grow the pool if it has not come back yet.
    if (!retired_.try_pop(spare_)) {
      spare_ = allocate_plan();
    }
  }

  [[nodiscard]] bool next_completion(Completion &out) {
    return completions_.try_pop(out);
  }

  // --- runtime thread -----------------------------------------------------

  // Take the freshest published plan, or nullptr if none is pending. The caller
  // owns the returned plan until it hands it to retire().
  [[nodiscard]] Plan *consume() {
    return published_.exchange(nullptr, std::memory_order_acquire);
  }

  void retire(Plan *plan) {
    assert(plan != nullptr);
    // A failed push only costs the buffer its place in the pool; the plan stays
    // owned by pool_, and the scheduler allocates a replacement.
    (void)retired_.try_push(plan);
  }

  [[nodiscard]] bool report(const Completion &completion) {
    return completions_.try_push(completion);
  }

private:
  Plan *allocate_plan() {
    // Plans are heap-allocated individually so their addresses stay stable as
    // pool_ grows while the runtime holds a pointer into it.
    pool_.push_back(std::make_unique<Plan>());
    Plan *plan = pool_.back().get();
    plan->work.reserve(reserve_);
    return plan;
  }

  std::atomic<Plan *> published_{nullptr};
  SPSCQueue<Plan *> retired_;        // runtime -> scheduler, spent buffers
  SPSCQueue<Completion> completions_; // runtime -> scheduler

  // Scheduler-thread only. pool_ is grown solely by the scheduler; the runtime
  // never touches it, only the Plan objects it points at.
  std::vector<std::unique_ptr<Plan>> pool_;
  Plan *spare_ = nullptr;
  std::uint64_t epoch_ = 0;
  const std::size_t reserve_;
};
