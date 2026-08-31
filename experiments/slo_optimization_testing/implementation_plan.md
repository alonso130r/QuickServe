# QP and SOCP Experiment Implementation Plan

> **For agentic workers:** Execute this single tightly coupled task inline. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a standalone, cacheable two-tier comparison of ProxQP/QP and Clarabel/SOCP without implementing a scheduler policy.

**Architecture:** A local CMake target builds one experiment executable and one focused test executable. Solver adapters consume shared two-variable problem instances; pure experiment logic generates scenarios, projects solutions, computes one-step scores, and persists line-oriented cache/results. A standalone lower-level llama.cpp collector executes fixed batch specifications directly, without `Scheduler`, `Environment`, or `Handoff`.

**Tech Stack:** C++17, Eigen, ProxSuite/ProxQP, Clarabel.cpp, QuickServe runtime, CMake.

## Chunk 1: Standalone experiment

### Task 1: Implement and verify the complete experiment

**Files:**

- Create: `experiments/slo_optimization_testing/CMakeLists.txt`
- Create: `experiments/slo_optimization_testing/experiment.hpp`
- Create: `experiments/slo_optimization_testing/experiment.cpp`
- Create: `experiments/slo_optimization_testing/solvers.hpp`
- Create: `experiments/slo_optimization_testing/solvers.cpp`
- Create: `experiments/slo_optimization_testing/backend_cache.hpp`
- Create: `experiments/slo_optimization_testing/backend_cache.cpp`
- Create: `experiments/slo_optimization_testing/main.cpp`
- Create: `experiments/slo_optimization_testing/tests.cpp`
- Create: `experiments/slo_optimization_testing/README.md`
- Modify: `experiments/slo_optimization_testing/implementation_plan.md`

- [x] **Step 1: Write failing focused tests**

Test deterministic scenario generation, nearest-rank p99, projection tie-breaking, one-step scoring, atomic append-safe cache reuse and deduplication, shared-QP solver agreement, and native QP/SOCP feasibility.

- [x] **Step 2: Run tests and verify the expected build or assertion failure**

Configure only the experiment subdirectory and confirm failure is caused by missing experiment implementation.

- [x] **Step 3: Implement shared experiment data and scoring**

Implement two-variable instances, deterministic stratified scenario generation, normalized projection distance, exact resource checks, right-censored one-step TTFT/TPOT scoring, nearest-rank p99, realized objective, and CSV output.

- [x] **Step 4: Implement solver adapters**

Implement ProxQP native QP solves, Clarabel shared-QP solves, and Clarabel native SOCP solves with matched tolerances, normalized status, residuals, iterations, objective, cold/setup/solve timing, and warm repeated timing.

- [x] **Step 5: Implement the cache and backend collector**

Implement versioned cache records with stable composition keys, raw durations, observation IDs, timestamps, run metadata, per-key locking, atomic replacement, crash preservation, and deduplication. Add a direct llama.cpp driver that creates deterministic token and KV state from a fixed measurement specification, executes the requested mixed batch, records duration, and shuts down cleanly. Do not use `Scheduler`, `Environment`, or `Handoff`.

- [x] **Step 6: Implement CLI modes**

Provide `synthetic`, `collect`, and `evaluate` modes. `synthetic` emits Tier 1 raw CSV. `collect` reuses cached keys and appends only missing repetitions. `evaluate` uses a deterministic stratified 60/20/20 cell split, fits nominal models only on training cells, calibrates both bounds to the same 99% target on calibration cells, reports calibration and held-out coverage, uses deterministic disjoint oracle-selection and evaluation repetitions, solves immutable offline snapshots, projects to cached candidates, and emits paired decision-quality CSV.

- [x] **Step 7: Document exact commands and output schemas**

Document configuration, reduced smoke commands, full collection commands, cache identity limitations, and the explicit no-scheduler-policy boundary.

- [x] **Step 8: Run focused verification**

Build in Release mode, run the focused tests, run a reduced synthetic smoke benchmark, confirm both solvers report successful feasible solutions, and run `git diff --check`. Run the backend cache smoke only when an absolute GGUF model path is supplied.

- [x] **Step 9: Stop**

Do not integrate the experiment into the root build, add a scheduler policy, run full repository tests, or alter unrelated staged changes.
# Offline Runner Addition

- [x] Add a failing wrapper behavior test using a fake experiment binary.
- [x] Resolve the main test model from `QUICKSERVE_TEST_MODEL` or its CMake cache.
- [x] Collect only when the measurement cache is absent or empty.
- [x] Always run offline evaluation and document the wrapper.
- [x] Run the focused wrapper test and shell syntax check.

# Z-Scored Sliding-Window Constraints

- [x] Add failing tests for exact window headroom and unit-invariant slack.
- [x] Replace the guessed runtime ceiling with TTFT and TPOT headroom fields.
- [x] Scale QP and SOCP hard constraints into dimensionless form.
- [x] Generate feasible measured-latency-relative snapshots and record their targets.
- [x] Emit solver status and standardized slacks in offline results.
- [x] Run focused tests and regenerate the offline evaluation.

# Boundary-Risk Objective and Informative Evaluation

- [x] Add failing integration checks for boundary-pressure output and full-grid candidates.
- [x] Add squared-hinge risk auxiliaries to the linear QP and SOCP.
- [x] Split repetitions within every cell for train, calibration, and held-out evaluation.
- [x] Generate standardized pressure strata near the constraint boundary.
- [x] Record boundary penalties, reference pressure, and paired decision keys.
- [x] Update the formulations and run focused tests without regenerating user results.
