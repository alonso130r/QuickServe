# Real-Model Scheduler Foundation Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a C++ demo that runs an interchangeable scheduling policy against a real llama.cpp model, with the executor and every llama.cpp API call confined to the environment thread.

**Architecture:** The scheduler thread owns request and policy state and publishes absolute, token-budgeted plans through the existing handoff. The environment thread owns the llama model, context, vocabulary, samplers, tokenized prompts, sequence IDs, KV cache, batching, decoding, and detokenization. The demo policy is a strict non-preemptive FIFO queue that finishes the oldest request before running the next; the demo uses hardcoded settings and prompts so policy experimentation is the next task.

**Tech Stack:** C++17, CMake, llama.cpp, GGML/Metal on Apple Silicon, CTest, ASan/UBSan for model-free tests.

**Approved design:** `docs/superpowers/specs/2026-08-18-real-model-scheduler-foundation-design.md`

**Git constraint:** Do not create commits. Leave all implementation changes in the working tree for the user to review and commit.

---

## File Map

### Create

- `src/runtime/protocol.hpp` — model-independent request IDs, admissions, work items, completions, errors, and terminal results.
- `src/runtime/handoff.hpp` — the scheduler/environment mailbox and plan-buffer ownership protocol.
- `src/runtime/scheduler.hpp` — scheduler base class and scheduler-owned request state.
- `src/runtime/environment.cpp` — environment-thread lifecycle, llama.cpp ownership, tokenization, batching, decode, sampling, cleanup, and shutdown.
- `src/runtime/environment_state.hpp` — llama-independent environment request records, KV reservations, sequence allocation, and plain batch descriptions.
- `src/runtime/environment_state.cpp` — model-free environment state and plan-translation logic composed by `Environment::Impl`.
- `src/policies/fifo.hpp` — strict FIFO policy declaration.
- `src/policies/fifo.cpp` — strict FIFO policy implementation.
- `src/app/quickserve_demo.cpp` — hardcoded real-model executable and thread orchestration.
- `tests/test_protocol.cpp` — protocol layout/default/error tests.
- `tests/test_scheduler.cpp` — scheduler lifecycle, backpressure, budget, and completion tests using no model.
- `tests/test_fifo.cpp` — exact FIFO plan tests.
- `tests/test_environment_state.cpp` — model-free batch-description and sequence-lifecycle tests.
- `tests/smoke_real_model.cpp` — opt-in real-GGUF end-to-end smoke test.

### Modify

- `src/runtime/environment.hpp` — replace the placeholder with the public environment API and private implementation boundary.
- `src/runtime/plan.hpp` — move the surviving plan type into `handoff.hpp`, then delete this file.
- `src/runtime/request.cpp` — move protocol types into `protocol.hpp`, then delete this file.
- `src/runtime/scheduler.cpp` — split its declaration into `scheduler.hpp` and retain implementation only.
- `tests/test_handoff.cpp` — update includes and extend handoff correctness coverage.
- `tests/CMakeLists.txt` — add model-free test targets and the opt-in model smoke target.
- `CMakeLists.txt` — replace the incomplete Python module with C++ libraries and `quickserve_demo`.
- `README.md` — document the hardcoded model path, build, demo, and smoke-test commands.

Do not add Python bindings, an HTTP layer, runtime argument parsing, cancellation, or advanced policies in this milestone.

---

## Chunk 1: Make the Cross-Thread Contract Explicit

### Task 1: Extract model-independent protocol types

**Files:**
- Create: `src/runtime/protocol.hpp`
- Delete after migration: `src/runtime/request.cpp`
- Test: `tests/test_protocol.cpp`
- Modify: `tests/CMakeLists.txt`

- [ ] **Step 1: Write protocol tests first**

Cover these exact properties:

```cpp
static_assert(std::is_trivially_copyable_v<WorkItem>);
static_assert(std::is_trivially_copyable_v<Completion>);

CHECK(work.token_count() == work.token_end - work.token_begin);
CHECK(decode.token_count() == 1);
Completion failed{};
failed.id = id;
failed.error = ErrorCode::DecodeFailed;
CHECK(failed.error == ErrorCode::DecodeFailed);
```

Also test that default-constructed messages have deterministic values rather than uninitialized booleans or tokens.

- [ ] **Step 2: Register and run the failing test**

Run: `cmake -S . -B build -DQUICKSERVE_TEST_SANITIZERS=ON && cmake --build build --target test_protocol -j2`

Expected: compilation fails because `runtime/protocol.hpp` does not exist.

- [ ] **Step 3: Add the protocol API**

Use standard integer types so model-free tests do not include `llama.h`:

```cpp
using RequestId = std::uint32_t;
using Token = std::int32_t;

enum class WorkKind : std::uint8_t { Prefill, Decode };
enum class ErrorCode : std::uint8_t {
  None,
  TokenizationFailed,
  ContextCapacityExceeded,
  DecodeFailed,
  SamplingFailed,
  ProtocolViolation,
  EnvironmentStopped,
};

struct Admission {
  RequestId id = 0;
  std::string prompt;
  std::uint32_t max_output_tokens = 0;
};

struct AdmissionResult {
  RequestId id = 0;
  std::uint32_t prompt_tokens = 0;
  ErrorCode error = ErrorCode::None;
};

struct WorkItem {
  RequestId id = 0;
  std::uint32_t token_begin = 0;
  std::uint32_t token_end = 0;
  WorkKind kind = WorkKind::Prefill;
  [[nodiscard]] std::uint32_t token_count() const {
    return token_end - token_begin;
  }
};

struct Completion {
  RequestId id = 0;
  std::uint32_t prefill_position = 0;
  std::uint32_t decoded_tokens = 0;
  Token token = 0;
  WorkKind kind = WorkKind::Prefill;
  ErrorCode error = ErrorCode::None;
  bool generated_token = false;
  bool eos = false;
};

struct OutputPiece {
  RequestId id = 0;
  Token token = 0;
  std::string text;
};

struct Release { RequestId id = 0; };
struct ReleaseAck { RequestId id = 0; };
struct RunFatal { ErrorCode error = ErrorCode::None; };
```

Keep `Admission` separate from the trivially-copyable hot-path messages because it owns prompt text.

- [ ] **Step 4: Run the protocol test**

Run: `cmake --build build --target test_protocol -j2 && ./build/tests/test_protocol`

Expected: all protocol checks pass under ASan/UBSan.

### Task 2: Turn the handoff into a lossless scheduler/environment mailbox

**Files:**
- Create: `src/runtime/handoff.hpp`
- Delete after migration: `src/runtime/plan.hpp`
- Modify: `tests/test_handoff.cpp`

- [ ] **Step 1: Add failing tests for all channels**

Preserve the existing no-duplicate-plan tests. Add tests proving:

- admissions move scheduler to environment;
- admission results move environment to scheduler;
- output pieces move environment to scheduler and preserve owned text;
- releases move scheduler to environment and acknowledgements return;
- run-fatal messages move environment to scheduler;
- completions are visible before a plan epoch becomes complete;
- a run-fatal message from a plan that omits another active request causes the
  scheduler to fail and release both requests;
- a full channel returns `false` without overwriting an older message;
- stop is monotonic and visible to both threads;
- an empty plan is never published.

- [ ] **Step 2: Run the handoff test and confirm failure**

Run: `cmake --build build --target test_handoff -j2`

Expected: compilation fails for the new mailbox methods.

- [ ] **Step 3: Implement the mailbox API**

Keep the current stable plan pool and single atomic published-plan slot. Add bounded SPSC queues and these directional methods:

```cpp
// scheduler -> environment (scheduler is the sole producer)
bool try_admit(Admission &&admission);
bool try_release(const Release &release);
Plan *consume_plan();
void request_stop();

// environment -> scheduler (environment is the sole producer)
bool try_take_admission(Admission &out);
bool try_take_release(Release &out);
bool try_report_admission(const AdmissionResult &result);
bool try_report_completion(const Completion &completion);
bool try_report_output(OutputPiece &&piece);
bool try_acknowledge_release(const ReleaseAck &ack);
bool try_report_fatal(const RunFatal &fatal);
void retire_plan(Plan *plan);

// scheduler reads
bool try_take_admission_result(AdmissionResult &out);
bool try_take_completion(Completion &out);
bool try_take_output(OutputPiece &out);
bool try_take_release_ack(ReleaseAck &out);
bool try_take_fatal(RunFatal &out);
std::uint64_t completed_epoch() const;
bool stop_requested() const;
```

Do not silently ignore failed pushes. Callers must retain the message and retry after yielding; add comments identifying which thread owns each retry buffer. After sending `RunFatal`, environment stops consuming plans but continues consuming every release and producing release acknowledgements until scheduler requests final stop.

- [ ] **Step 4: Run protocol tests repeatedly**

Run: `cmake --build build --target test_handoff -j2 && for i in {1..100}; do ./build/tests/test_handoff >/dev/null || exit 1; done`

Expected: 100 clean passes with no sanitizer findings.

---

## Chunk 2: Establish the Policy Boundary

### Task 3: Split and harden the scheduler base

**Files:**
- Create: `src/runtime/scheduler.hpp`
- Modify: `src/runtime/scheduler.cpp`
- Create: `tests/test_scheduler.cpp`
- Modify: `tests/CMakeLists.txt`

- [ ] **Step 1: Write failing scheduler lifecycle tests**

Create a probe policy and fake environment. Test:

- `submit()` assigns stable increasing IDs before the worker starts without
  touching the scheduler-producer SPSC queue;
- pending admissions flush from `run_once()` after environment starts and a
  full admission queue never blocks `submit()`;
- admission state advances `NotSent -> InFlight -> EnvironmentOwned`, with an
  admission error terminating from `InFlight` without a release;
- a request stays `PendingAdmission` until its admission result arrives;
- tokenization failure makes only that request terminal;
- tokenization and capacity admission failures create no environment-owned
  resource, transition directly to `Terminal`, send no release, and allow
  `all_terminal()` to become true;
- absolute completions fold idempotently with `max()`;
- EOS and output limit make a request terminal;
- a terminal request is never rescheduled;
- `run_once()` will not publish while an epoch is outstanding;
- `all_terminal()` is true only when every submitted request is terminal.
- output pieces append before the matching completion is folded;
- release messages retry until queued and `ReleaseAck` completes cleanup;
- `RunFatal` marks every nonterminal request failed, including one absent from
  the failed plan, and the loop exits only after every release acknowledgement.

- [ ] **Step 2: Run the new target and verify failure**

Run: `cmake --build build --target test_scheduler -j2`

Expected: compilation fails because the split scheduler API is absent.

- [ ] **Step 3: Define scheduler-owned state**

Use request stages `PendingAdmission`, `Prefill`, `Decode`, `PendingRelease`, and `Terminal`, plus a separate admission ownership state `NotSent`, `InFlight`, or `EnvironmentOwned`. Store prompt length, absolute prefill position, decoded count, output token IDs and text, arrival/start/first-token/finish timestamps, output limit, and terminal error. Do not store prompt tokens or llama sequence IDs. The scheduler allocates request IDs; main reads request state only after joining the scheduler thread.

Expose only:

```cpp
RequestId submit(std::string prompt, std::uint32_t max_output_tokens);
bool run_once();
void run();
void request_stop();
bool all_terminal() const;
const std::vector<RequestState> &requests() const; // only read after join
```

Keep `build_plan(Plan &out)` as the sole virtual policy hook.

- [ ] **Step 4: Implement admission and completion folding**

`submit()` creates scheduler state and stores its `Admission` in a scheduler-owned pending deque; it never writes an SPSC queue from main. `run_once()` runs on the scheduler thread, flushes pending admissions without blocking, then drains admission results, output pieces, completions, release acknowledgements, and run-fatal messages. It refuses to republish an outstanding epoch, asks the policy for work, validates the plan, and commits nonempty plans.

On EOS, output limit, or an error after successful admission, move to `PendingRelease`, retain and retry the release message, then record `Terminal` and finish time after the environment acknowledges cleanup. An admission error has no environment resource, so it transitions directly to `Terminal`. Record first-token time when the first completion with `generated_token` arrives, including a final-prefill completion.

- [ ] **Step 5: Run scheduler and existing handoff tests**

Run: `cmake --build build --target test_scheduler test_handoff -j2 && ctest --test-dir build --output-on-failure`

Expected: all model-free tests pass.

### Task 4: Enforce plan invariants centrally

**Files:**
- Modify: `src/runtime/scheduler.hpp`
- Modify: `src/runtime/scheduler.cpp`
- Modify: `tests/test_scheduler.cpp`

- [ ] **Step 1: Add failing malicious-policy tests**

Make probe policies emit each invalid case: token total over budget, unknown request ID, zero-length range, prefill range past prompt end, decode before the first token has been generated, decode with any range other than `[prompt_length + decoded - 1, prompt_length + decoded)`, duplicate request work in one plan, and work for a terminal request.

- [ ] **Step 2: Verify each test fails for the intended reason**

Run: `cmake --build build --target test_scheduler -j2 && ./build/tests/test_scheduler`

Expected: new invariant checks fail because invalid plans currently publish.

- [ ] **Step 3: Add `validate_plan()` before publishing**

Return a structured validation result rather than relying on debug-only `assert`. Convert a violation to `ErrorCode::ProtocolViolation`, stop publishing plans and flushing `NotSent` admissions, enter draining mode, and fail `NotSent` requests locally. Continue draining an admission result for every `InFlight` request: an `EnvironmentStopped` result terminates without release, while a success racing with shutdown immediately enters `PendingRelease`. Retry releases for every `EnvironmentOwned` request, await every release acknowledgement, and only then finish. Add a test with a second admission already in flight when another request's policy emits an invalid plan; assert that the race either rejects or releases it and no environment state is stranded. In tests, expose the last scheduler error for exact assertions.

- [ ] **Step 4: Run scheduler tests**

Run: `cmake --build build --target test_scheduler -j2 && ./build/tests/test_scheduler`

Expected: every malformed plan is rejected and no invalid plan reaches the fake environment.

### Task 5: Add the strict FIFO demo policy

**Files:**
- Create: `src/policies/fifo.hpp`
- Create: `src/policies/fifo.cpp`
- Create: `tests/test_fifo.cpp`
- Modify: `tests/CMakeLists.txt`

- [ ] **Step 1: Write exact-plan tests first**

Cover: arrival-order selection, prefill chunking to the token budget, one-token decode work, pending/terminal skipping, and advancement to the second request only after the first request's release is acknowledged. Assert that every nonempty plan contains work for exactly one request and that a younger request is never scheduled while an older request remains active.

- [ ] **Step 2: Run and verify the missing-policy failure**

Run: `cmake --build build --target test_fifo -j2`

Expected: compilation fails because `policies/fifo.hpp` does not exist.

- [ ] **Step 3: Implement the minimal policy**

Constructor inputs are the handoff and total batch-token budget. `build_plan()` finds the oldest admitted request that is not terminal or pending release. It emits one prefill chunk for that request, bounded by the token budget, or one decode item after prefill. It emits no work for any younger request in the same plan and retains no private progress state.

- [ ] **Step 4: Run all model-free policy tests**

Run: `cmake --build build --target test_fifo -j2 && ./build/tests/test_fifo && ctest --test-dir build --output-on-failure`

Expected: exact single-request FIFO plans match and all earlier tests remain green.

---

## Chunk 3: Put llama.cpp Entirely on the Environment Thread

### Task 6: Add environment startup, ownership, and admission tokenization

**Files:**
- Modify: `src/runtime/environment.hpp`
- Create: `src/runtime/environment.cpp`
- Create: `src/runtime/environment_state.hpp`
- Create: `src/runtime/environment_state.cpp`
- Modify: `CMakeLists.txt`
- Create: `tests/test_environment_state.cpp`

- [ ] **Step 1: Write model-independent lifecycle tests**

Implement a small `EnvironmentRequestTable` in `environment_state.hpp/.cpp` and compose it inside `Environment::Impl`. Test stable request-to-sequence mapping, duplicate admission rejection, capacity rejection, terminal cleanup marking, and sequence-ID reuse only after cleanup. Test aggregate KV reservation with two requests that fit individually but exceed capacity together. The test must not include or link llama.cpp.

- [ ] **Step 2: Run and confirm failure**

Run: `cmake --build build --target test_environment_state -j2`

Expected: compilation fails because the environment request table is absent.

- [ ] **Step 3: Define the public environment API**

Keep llama types out of `environment.hpp` using a private implementation:

```cpp
struct EnvironmentConfig {
  std::string model_path;
  std::uint32_t context_tokens = 4096;
  std::uint32_t batch_tokens = 512;
  std::uint32_t max_sequences = 8;
  std::uint32_t seed = 0;
};

class Environment {
public:
  Environment(Handoff &, EnvironmentConfig);
  ~Environment();
  Environment(const Environment &) = delete;
  void run(); // call only on the environment thread
  bool wait_until_started();
  std::string startup_error() const;
private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};
```

- [ ] **Step 4: Initialize llama only inside `run()`**

Inside the environment thread: call `llama_backend_init()`, load the model with `llama_model_load_from_file`, obtain its vocabulary, create a context with the configured `n_ctx`, `n_batch`, and sequence capacity, then allocate a reusable `llama_batch`. Signal startup success only after all objects exist. Scope all RAII guards inside `run()` and explicitly reset batch, samplers, context, model, and backend before `run()` returns. `Environment::~Environment()` must observe null/empty llama handles and perform no llama.cpp call on main. In a test build, record the owner thread ID and assert all create/free wrappers run there.

- [ ] **Step 5: Consume and tokenize admissions**

Use the two-call `llama_tokenize()` sizing pattern with BOS and special-token parsing enabled. Store tokens only in the environment request record. Retry the corresponding `AdmissionResult` until the scheduler accepts it. Conservatively reserve aggregate KV demand as `prompt_tokens + (max_output_tokens == 0 ? 0 : max_output_tokens - 1)` and reject an admission when that reservation, the existing reservations, or the sequence count exceeds context capacity. Release the reservation only after sequence cleanup is acknowledged.

- [ ] **Step 6: Run model-free tests and compile the environment**

Run: `cmake --build build --target test_environment_state quickserve_runtime -j2 && ctest --test-dir build --output-on-failure`

Expected: lifecycle tests pass and the environment compiles against the repository's pinned llama.cpp API.

### Task 7: Build and execute llama batches

**Files:**
- Modify: `src/runtime/environment.cpp`
- Modify: `src/runtime/environment_state.hpp`
- Modify: `src/runtime/environment_state.cpp`
- Modify: `tests/test_environment_state.cpp`

- [ ] **Step 1: Add pure batch-description tests**

Before touching `llama_batch`, translate a `Plan` plus environment records into a vector of plain `BatchToken` descriptions containing token ID, absolute position, sequence ID, and logits flag. Test chunked prefill positions, decode input token selection, multiple sequences in one batch, and exactly one logits position for each decode operation.

- [ ] **Step 2: Run and verify failure**

Run: `cmake --build build --target test_environment_state -j2 && ./build/tests/test_environment_state`

Expected: batch-description checks fail because the builder is absent.

- [ ] **Step 3: Implement the pure plan translator**

For prefill `[begin,end)`, append the stored prompt tokens at those absolute positions. Request logits on the last token only when this prefill completes the prompt and its output limit is nonzero. For decode, feed the request's most recently sampled token at `prefill_length + decoded_tokens - 1` and request logits for that row.

Handle the first generated token explicitly: immediately after the batch containing the final prompt token succeeds, sample from that row, emit its output piece unless it is EOS, and report `decoded_tokens = 1` on the prefill completion. Never retain a logits pointer across another `llama_decode` call and never decode the final prompt token twice. Each later decode item feeds the newest sampled token into KV, samples its successor, and reports the new absolute decoded count.

- [ ] **Step 4: Populate and execute `llama_batch`**

Reset `batch.n_tokens`, copy each plain description into the batch arrays, call `llama_decode`, then use the recorded logits rows. A nonzero decode return is fatal to the entire run because a mixed-sequence batch has no safe per-request rollback contract: retain and retry one `RunFatal{DecodeFailed}` message, retire the plan only after it is visible, and enter draining shutdown. Scheduler responds by marking every nonterminal request failed and releasing every environment-owned request, including requests absent from the failed plan.

- [ ] **Step 5: Add one sampler per active request**

Create a sampler chain during admission using source-level constants (temperature plus distribution sampler, or greedy sampling for deterministic smoke tests). Call `llama_sampler_sample` with the exact logits row; in the pinned llama.cpp API this function already accepts the sampled token into sampler history, so do not call `llama_sampler_accept` again. Use `llama_vocab_is_eog` for EOS detection and cover two successive samples in the real-model smoke test.

- [ ] **Step 6: Report completions before retiring the plan**

Create one absolute `Completion` per work item. For a generated non-EOS token, call `llama_token_to_piece` on the environment thread using its sizing/retry contract: begin with a small owned buffer, and when the return is negative resize to the reported required byte count and retry. Enqueue the complete owned `OutputPiece` before its completion and suppress EOS text. Retain any output or completion that encounters a full queue and retry it before calling `retire_plan()`. Only retirement advances the completed epoch.

- [ ] **Step 7: Clean terminal sequences**

Do not make terminal decisions in the environment. When a release arrives from the scheduler, call `llama_memory_seq_rm(llama_get_memory(ctx), seq_id, -1, -1)`, free the request sampler, release the sequence ID, erase prompt/generated-token storage only after any printable token piece has been emitted, and acknowledge the release. EOS and errors are facts reported to the scheduler; the scheduler decides the terminal transition.

- [ ] **Step 8: Run model-free tests and compile**

Run: `cmake --build build --target test_environment_state quickserve_runtime -j2 && ctest --test-dir build --output-on-failure`

Expected: all pure state/batch tests pass and the llama-backed target compiles.

---

## Chunk 4: Produce the First Real-Model Run

### Task 8: Add the two-thread C++ demo

**Files:**
- Create: `src/app/quickserve_demo.cpp`
- Modify: `CMakeLists.txt`

- [ ] **Step 1: Add the executable target and verify link failure**

Create a `quickserve_demo` target linked to `quickserve_runtime`, `quickserve_policies`, llama, GGML, and threads.

Run: `cmake --build build --target quickserve_demo -j2`

Expected: link or compilation fails until `main()` exists.

- [ ] **Step 2: Implement hardcoded demo configuration**

At the top of `quickserve_demo.cpp`, define one clearly marked model-path constant and two short prompts. Set small, explicit context, batch, sequence, prefill-chunk, and output limits. Validate the file exists before starting threads and print the exact constant that the developer must edit when it does not.

- [ ] **Step 3: Orchestrate startup safely**

Construct the handoff, environment, and FIFO scheduler. Submit both requests before starting the scheduler worker. Start the environment thread first and wait for its startup result. Start the scheduler thread only on success. On failure, request stop, join the environment, and return nonzero. Verify in the demo output that request 1 begins only after request 0 finishes and its environment resources are released.

- [ ] **Step 4: Implement normal completion and shutdown**

As it consumes `OutputPiece` messages, the scheduler thread appends them to request state and writes them to the terminal; it is the only writer during generation. The scheduler loop exits only after all requests are terminal and all releases are acknowledged, then requests environment stop and returns. Main joins scheduler then environment. Read scheduler results only after the scheduler join and print request ID, accumulated text, time-to-first-token, total latency, decoded-token count, and terminal error.

- [ ] **Step 5: Compile and run the demo**

Run: `cmake --build build --target quickserve_demo -j2 && ./build/quickserve_demo`

Expected with the configured model present: two prompts generate at least one token each, then both threads join cleanly. Expected without it: a concise path error and nonzero exit, with no crash or hanging thread.

### Task 9: Add an opt-in real-model smoke test

**Files:**
- Create: `tests/smoke_real_model.cpp`
- Modify: `tests/CMakeLists.txt`

- [ ] **Step 1: Write the smoke-test contract**

Use the same production environment and FIFO policy. Read `QUICKSERVE_TEST_MODEL` only in this test so a developer can select a small local GGUF without changing source. If unset, print `SKIP` and return CTest's configured skip code. Require two prompts to produce at least one token each, terminate within their output limits, preserve FIFO execution order, and join both threads.

- [ ] **Step 2: Add a non-default CMake target**

Build `smoke_real_model`, but label its CTest entry `model;smoke` and disable it by default unless `QUICKSERVE_ENABLE_MODEL_TESTS=ON` is configured. Set a finite CTest timeout.

- [ ] **Step 3: Run it against the developer's model**

Run:

```bash
QUICKSERVE_TEST_MODEL=/absolute/path/to/model.gguf \
ctest --test-dir build -L model --output-on-failure
```

Expected: one model smoke test passes without duplicate positions, sanitizer errors, or shutdown hangs.

### Task 10: Document and verify the fun-stuff boundary

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Document the developer workflow**

Include prerequisites, the single model-path line to edit, configure/build commands, demo command, opt-in smoke-test command, expected terminal output, and the ownership rule: no llama.cpp call may occur outside the environment thread.

- [ ] **Step 2: Document how to add a policy**

Show that a new policy subclasses `Scheduler`, implements only `build_plan(Plan &)`, derives decisions solely from scheduler request state, stays inside the token budget, and adds model-free exact-plan tests.

- [ ] **Step 3: Run format and static checks**

Run: `git diff --check`

Expected: no whitespace errors.

If `clang-format` is available, run it only over QuickServe-owned C++ files changed by this plan, then inspect the diff.

- [ ] **Step 4: Run all model-free verification**

Run: `cmake -S . -B build -DQUICKSERVE_TEST_SANITIZERS=ON && cmake --build build -j2 && ctest --test-dir build --output-on-failure -LE model`

Expected: the full build succeeds and every model-free test passes under ASan/UBSan.

- [ ] **Step 5: Run the real-model verification**

Run the demo and the opt-in smoke test against the local GGUF. Confirm both prompts generate, token positions never repeat, output limits are exact, KV sequences are removed, and both workers join.

---

## Definition of Done

- [ ] The environment thread is the only thread that loads, calls, or frees llama.cpp objects.
- [ ] The scheduler and environment communicate solely through the documented handoff.
- [ ] No correctness-bearing admission or completion can be silently dropped.
- [ ] The scheduler rejects invalid or over-budget plans.
- [ ] Strict FIFO handles at least two queued requests, fully completing and releasing the first before scheduling the second.
- [ ] The hardcoded C++ demo generates text from a local GGUF model and exits cleanly.
- [ ] Time-to-first-token and total latency are printed per request.
- [ ] All model-free tests pass under ASan/UBSan.
- [ ] The opt-in real-model smoke test passes.
- [ ] A new scheduling policy can be added without modifying environment code.
