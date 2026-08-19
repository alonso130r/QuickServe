# Real llama.cpp Scheduler Backend Implementation Plan

> **For agentic workers:** Follow the tasks in order. Keep changes uncommitted
> and do not add per-task review rounds.

**Goal:** Complete the C++ backend so FIFO and future scheduler policies can
execute requests end to end against a real llama.cpp model.

**Architecture:** The scheduler publishes token-budgeted plans through the
existing handoff. A dedicated environment thread owns llama.cpp, translates all
work items in each plan into a batch, returns output and absolute progress, and
cleans up released requests. FIFO is the first policy but is not embedded in
the executor.

**Tech Stack:** C++17, CMake, llama.cpp, CTest.

**Design:** `docs/superpowers/specs/2026-08-18-real-model-scheduler-foundation-design.md`

**Git constraint:** Do not stage or commit changes.

---

## Current State

The policy-independent foundation is already implemented:

- [x] Model-independent protocol messages.
- [x] Lossless scheduler/environment handoff channels.
- [x] Scheduler request lifecycle, plan validation, cleanup, and shutdown.
- [x] Stateless non-preemptive FIFO policy.
- [x] Model-free protocol, handoff, scheduler, and FIFO tests.

Do not expand or rewrite this completed work unless the real executor exposes a
specific integration bug.

## File Map

### Create

- `src/runtime/environment.cpp` — real llama.cpp environment-thread executor.
- `tests/test_real_backend.cpp` — one two-request real-model FIFO test.

### Modify

- `src/runtime/environment.hpp` — replace the placeholder with the environment
  configuration and public thread API.
- `CMakeLists.txt` — build the C++ runtime and FIFO policy libraries.
- `tests/CMakeLists.txt` — build the real-backend test when a model path is
  supplied.

No demo executable, separate environment-state library, fake executor, metrics,
README expansion, or additional test targets are part of this plan.

## Chunk 3: Implement the Real Environment

### Task 6: Add environment lifecycle and admissions

**Files:**

- Modify: `src/runtime/environment.hpp`
- Create: `src/runtime/environment.cpp`
- Modify: `CMakeLists.txt`

- [ ] **Step 1: Define the small public API**

Add `EnvironmentConfig` with model path, per-request context size, batch-token
capacity, and maximum sequence count. Add an `Environment` constructed with a
`Handoff` and config. Its `run()` method is called only on the environment
thread. Provide a startup result so callers do not start the scheduler after a
model-load failure.

Keep llama types out of the public header. `Environment` is non-copyable.

- [ ] **Step 2: Build the C++ libraries**

Replace the incomplete Python-module target with:

- `quickserve_runtime`, containing scheduler and environment sources and linked
  to llama.cpp and threads;
- `quickserve_policies`, containing FIFO and linked to the runtime API.

Run:

```bash
cmake -S . -B build -DQUICKSERVE_TEST_SANITIZERS=ON
cmake --build build --target quickserve_runtime quickserve_policies -j2
```

Expected: the targets build against the repository's pinned llama.cpp API.

- [ ] **Step 3: Initialize and destroy llama.cpp inside `run()`**

On the environment thread, initialize the backend, load the GGUF model, create
the context, vocabulary access, reusable batch storage, and sequence-ID pool.
Signal startup success only after initialization finishes. On failure, publish
the startup error and return cleanly.

All llama-owned objects must be destroyed inside `run()` before that thread
returns; the main-thread destructor must not call llama.cpp.

- [ ] **Step 4: Process admissions and releases**

For each admission:

1. tokenize with llama.cpp's sizing/retry contract;
2. reject an empty or oversized request;
3. allocate one reusable sequence ID;
4. create a greedy sampler and store prompt/generated-token state;
5. retry the `AdmissionResult` until the handoff accepts it.

For each release, remove the sequence from llama memory, destroy its sampler,
erase the request record, return the sequence ID to the pool, and retry the
`ReleaseAck` until accepted.

### Task 7: Execute scheduler plans through llama.cpp

**Files:**

- Modify: `src/runtime/environment.cpp`

- [ ] **Step 1: Translate a plan into one llama batch**

For every work item in the plan:

- Prefill `[begin, end)` adds that range of stored prompt tokens using their
  absolute positions and request sequence ID.
- The final prompt token requests logits when output is requested.
- Decode adds the most recently sampled token at
  `prompt_length + decoded_tokens - 1` and requests logits for that row.

Reject malformed or locally inconsistent work as `ProtocolViolation`. The
scheduler already enforces the token budget; the environment only checks that
the batch fits its configured storage.

- [ ] **Step 2: Decode and sample**

Call `llama_decode` once for the batch. For each logits row, greedily sample one
token with `llama_sampler_sample`; do not separately call
`llama_sampler_accept`. Detect end-of-generation with
`llama_vocab_is_eog`.

The final prefill operation immediately samples the first output token and
reports `decoded_tokens = 1`. Each later decode operation samples the next
token and reports the new absolute decoded count.

- [ ] **Step 3: Report output and completion before retirement**

Convert each non-EOG token with `llama_token_to_piece`, retrying with a larger
buffer when its return value reports the required size. Send the owned
`OutputPiece` before its matching `Completion`. Retry full output/completion
queues without dropping messages, then retire the plan.

If batch decode or sampling fails, publish one `RunFatal`, stop consuming new
plans, and continue handling releases until scheduler cleanup finishes.

- [ ] **Step 4: Verify existing tests still pass**

Run:

```bash
cmake --build build -j2
ctest --test-dir build --output-on-failure
```

Expected: the runtime builds and the existing model-free tests remain green.

## Chunk 4: Verify One Real End-to-End Run

### Task 8: Add a single real-backend FIFO test

**Files:**

- Create: `tests/test_real_backend.cpp`
- Modify: `tests/CMakeLists.txt`

- [ ] **Step 1: Add conditional model-test wiring**

Add a `QUICKSERVE_TEST_MODEL` CMake cache path. Build and register
`test_real_backend` only when that path is provided. Pass the absolute path to
the test; do not add a second configuration mechanism or demo executable.

- [ ] **Step 2: Run two FIFO requests**

The test constructs `Handoff`, the production `Environment`, and
`FifoScheduler`; submits two short prompts; starts the environment thread;
waits for successful model startup; runs the scheduler; and joins both threads.

Assert only the essential behavior:

- both requests reach `Terminal` without an error;
- both generate at least one output token and nonempty text;
- request 1 does not begin execution until request 0 is complete and released;
- environment and scheduler threads exit normally.

- [ ] **Step 3: Run the complete verification**

```bash
cmake -S . -B build \
  -DQUICKSERVE_TEST_SANITIZERS=ON \
  -DQUICKSERVE_TEST_MODEL=/absolute/path/to/model.gguf
cmake --build build -j2
ctest --test-dir build --output-on-failure
```

Expected: existing model-free tests pass and the two real FIFO requests produce
text, become terminal, release their state, and shut down cleanly.

## Definition of Done

- [ ] The production environment performs real llama.cpp inference.
- [ ] Every llama.cpp call, including cleanup, occurs on the environment thread.
- [ ] The executor handles all work items in a valid plan and is not FIFO-specific.
- [ ] Two FIFO requests complete end to end against a local GGUF model.
- [ ] Existing model-free tests still pass.
- [ ] No demo, fake executor, unrelated framework, staged file, or commit is added.
