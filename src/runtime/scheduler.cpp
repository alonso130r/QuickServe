#pragma once

#include <algorithm>
#include <cstdint>
#include <utility>
#include <vector>

#include "plan.hpp"
#include "request.cpp"

// The scheduler's authoritative view of one request. Lives on the scheduler
// thread only; the runtime keeps its own separate view (llama sequence ids, KV
// occupancy) and the two are reconciled solely by messages through Handoff.
struct RequestState {
  enum class Stage : std::uint8_t { Waiting, Prefill, Decoding, Complete, Paused };

  Request req;
  std::vector<llama_token> output;
  std::uint32_t prefill_pos = 0; // prompt tokens prefilled so far
  std::uint32_t decoded = 0;     // tokens generated so far
  std::uint32_t target_out = 0;
  Stage stage = Stage::Waiting;

  [[nodiscard]] std::uint32_t prompt_len() const {
    return static_cast<std::uint32_t>(req.num_tokens);
  }
  [[nodiscard]] bool prefill_done() const {
    return prefill_pos >= prompt_len();
  }
};

// Base for every scheduling policy.
//
// The base owns the loop mechanics and the plan lifetime; subclasses override
// only create_reordering(), which decides which requests decode and which
// prefills advance. Everything the policy needs is in table_.
//
// Threading: table_ and all policy decisions are scheduler-thread only. Handoff
// is the sole object that crosses to the runtime thread, so nothing else here
// needs synchronization.
class Scheduler {
public:
  explicit Scheduler(Handoff &handoff, std::uint32_t token_budget)
      : token_budget_(token_budget), handoff_(handoff) {}

  virtual ~Scheduler() = default;

  Scheduler(const Scheduler &) = delete;
  Scheduler &operator=(const Scheduler &) = delete;
  Scheduler(Scheduler &&) = delete;
  Scheduler &operator=(Scheduler &&) = delete;

  // Admit a request. Scheduler thread only. Returns the id that indexes table_
  // and that SplitRequest/Completion carry.
  std::uint32_t submit(Request req, std::uint32_t max_output_tokens) {
    const auto id = static_cast<std::uint32_t>(table_.size());
    RequestState state;
    state.req = std::move(req);
    state.target_out = max_output_tokens;
    state.stage = RequestState::Stage::Waiting;
    table_.push_back(std::move(state));
    return id;
  }

  // One scheduling iteration: fold in what the runtime finished, decide, and
  // publish. Deliberately non-virtual so a policy cannot reorder these steps.
  void step() {
    drain_completions();
    Plan &plan = handoff_.begin();
    create_reordering(plan);
    handoff_.commit();
  }

  [[nodiscard]] const std::vector<RequestState> &table() const {
    return table_;
  }

protected:
  // Fill `out` with the work for the next iteration. Must derive every entry
  // from table_ alone and keep no memory between calls: a published plan may be
  // superseded before the runtime sees it, and correctness depends on the next
  // call re-emitting the same intent. Total tokens must fit token_budget_.
  virtual void create_reordering(Plan &out) = 0;

  std::vector<RequestState> table_;
  std::uint32_t token_budget_; // executor's per-batch token cap (n_batch)

private:
  // The only place request progress advances. Completions carry absolute
  // positions, so folding with max() makes them idempotent and lets a
  // superseded plan be dropped harmlessly.
  void drain_completions() {
    Completion completion;
    while (handoff_.next_completion(completion)) {
      if (completion.id >= table_.size()) {
        continue; // stale id from a previous table generation
      }

      RequestState &state = table_[completion.id];
      if (state.stage == RequestState::Stage::Complete) {
        continue;
      }

      state.prefill_pos = std::max(state.prefill_pos, completion.prefill_pos);

      if (completion.kind == SplitKind::Decode &&
          completion.decoded > state.decoded) {
        state.output.push_back(completion.last_token);
        state.decoded = completion.decoded;
      }

      if (completion.eos || state.decoded >= state.target_out) {
        state.stage = RequestState::Stage::Complete;
      } else if (state.stage != RequestState::Stage::Paused) {
        state.stage = state.prefill_done() ? RequestState::Stage::Decoding
                                           : RequestState::Stage::Prefill;
      }
    }
  }

  Handoff &handoff_;
};
