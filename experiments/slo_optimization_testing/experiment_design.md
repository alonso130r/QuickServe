# QP and SOCP Optimization Experiment

## Purpose

This experiment compares two next-batch optimization setups without constructing or integrating a scheduler policy:

1. The linearly constrained QP from [formulations.md](formulations.md), solved with ProxQP.
2. The uncertainty-aware SOCP from [formulations.md](formulations.md), solved with Clarabel.

The experiment answers three separate questions:

- How quickly and reliably does each solver return a solution?
- How different are the continuous and projected solutions?
- Does the SOCP's allocation-dependent uncertainty produce better decisions than the QP's fixed safety margin?

Solver performance and formulation quality must be isolated. A direct ProxQP/QP versus Clarabel/SOCP comparison alone would confound solver effects with formulation effects.

## Scope

The experiment consists of two tiers:

- Tier 1 is a synthetic solver microbenchmark.
- Tier 2 is an offline decision-quality experiment backed by cached llama.cpp batch measurements.

Neither tier creates a `Scheduler` subclass, publishes plans through `Handoff`, or runs an online scheduling loop. Both formulations receive identical immutable problem instances and produce allocation records for analysis.

Request ordering, pressure coefficients, candidate compositions, sliding-window contents, and recovery-mode labels are fixed fields in the experiment input. The experiment does not rank requests, make admission decisions, dispatch work, or transition persistent request state. It only solves, projects, and scores isolated allocations.

## Experimental conditions

Build and run both solvers in the same process when practical, using:

- Release optimization settings.
- One solver thread.
- Identical floating-point precision.
- Equivalent feasibility and optimality tolerances.
- A monotonic high-resolution clock.
- Fixed random seeds.
- Alternating solver order.

Record the QuickServe revision, compiler, build type, ProxSuite version, Clarabel revision, hardware identifier, and operating-system version with every result set.

Normalize solver statuses and residual definitions in the output schema. Treat infeasible, iteration-limit, numerical-error, and timeout results as distinct outcomes rather than dropping them from timing summaries.

## Tier 1: synthetic solver microbenchmark

Tier 1 has two comparisons.

### Tier 1A: identical QP

Both solvers receive the same strictly convex QP:

\[
\min_x \frac12x^\top Hx+c^\top x
\]

subject to:

\[
Ax\le b.
\]

ProxQP solves the QP natively. Clarabel receives the same quadratic objective with the linear inequalities represented by its nonnegative cone. This comparison isolates solver behavior.

For every instance, record:

- Model-construction time.
- Solver setup time.
- Cold solve time.
- Warm update time.
- Warm solve time.
- Total decision time.
- Iteration count.
- Termination status.
- Primal residual.
- Dual residual.
- Objective value.
- Continuous solution.

Measure normalized solution disagreement:

\[
\Delta_x=
\left\|
\begin{bmatrix}
(p_{\mathrm{prox}}-p_{\mathrm{clar}})/B_{\mathrm{tok}}\\
(d_{\mathrm{prox}}-d_{\mathrm{clar}})/S_{\max}
\end{bmatrix}
\right\|_2.
\]

Measure relative objective disagreement:

\[
\Delta_J=
\frac{|J_{\mathrm{prox}}-J_{\mathrm{clar}}|}
{\max(1,|J_{\mathrm{prox}}|,|J_{\mathrm{clar}}|)}.
\]

Tier 1A determines whether the solvers agree and which solver has lower overhead on the shared mathematical problem.

### Tier 1B: native formulations

Compare:

- ProxQP solving the fixed-margin QP.
- Clarabel solving the allocation-uncertainty SOCP.

Record the Tier 1A metrics plus:

\[
\Delta_p=p_{\mathrm{socp}}-p_{\mathrm{qp}},
\]

\[
\Delta_d=d_{\mathrm{socp}}-d_{\mathrm{qp}},
\]

and the difference in conservative runtime estimates.

Tier 1B measures the computational price of the richer robust formulation. It does not, by itself, determine which formulation makes better decisions.

### Synthetic instance families

Generate deterministic instances across these factors:

| Factor | Levels |
|---|---|
| Deadline pressure | low, medium, high |
| Demand mix | prefill-heavy, balanced, decode-heavy |
| Runtime capacity | loose, moderate, tight |
| Memory capacity | loose, moderate, tight |
| Runtime uncertainty | low, medium, high |
| Optimum location | interior, boundary, nearly infeasible |
| Previous allocation | near, moderate, far |

Use a capped, stratified design so every level and important interaction is represented without materializing the full Cartesian product. Reject generated instances that are numerically invalid. Retain feasible, nearly infeasible, and deliberately infeasible cases as separately labelled groups. Choose the final instance and repetition counts from a pilot, stopping when the bootstrap interval widths for median and p99 solve time meet predeclared precision targets.

### Timing protocol

For each retained instance:

1. Build solver-independent problem data.
2. Convert it to each solver's representation while timing that conversion separately.
3. Perform untimed warm-up solves.
4. Measure repeated warm updates and solves.
5. Measure repeated cold setup and solves.
6. Alternate the first solver between repetitions.
7. Store every raw timing observation.

A cold solve starts from a newly constructed solver object with no retained factorization or iterate. A warm solve reuses the prior solver object and factorization, updates only the permitted problem vectors or bounds, and starts from the previous solution when supported. Report solve-only time and end-to-end conversion, setup, update, and solve time.

Use enough repetitions for stable tail estimates. Start with 1,000 warm repetitions per instance and reduce only if a pilot demonstrates that fewer repetitions produce stable confidence intervals.

Report:

\[
p50,\quad p90,\quad p99,\quad \max
\]

for setup, solve, and total decision time. The primary latency statistic is p99 total decision time.

## Tier 2: cached backend decision-quality experiment

Tier 2 evaluates both formulations against measured backend behavior while remaining offline from the scheduler.

### Standalone measurement collector

Create a standalone experiment executable that executes specified batch compositions directly against the llama.cpp backend. It must not implement a scheduling policy. Its input is a deterministic batch-measurement specification, and its output is an append-safe measurement cache.

Each measurement describes:

- Prefill-token allocation \(p\).
- Decode-item allocation \(d\).
- Number of sequences.
- Context-length bucket or explicit context-length vector.
- Model and backend settings.
- Warm-up repetitions.
- Recorded repetitions.
- Input-token generator and seed.
- Decode KV-cache state description.

Randomize composition execution order within each collection run to reduce correlation between batch size and thermal drift.

### Cache identity

A cache key must include every input capable of changing runtime:

- Model file hash.
- QuickServe revision.
- llama.cpp revision.
- Collector schema version.
- Compiler and build type.
- Hardware identifier.
- Accelerator model, driver, and runtime versions.
- Power or clock mode when controlled.
- Backend configuration.
- Context size.
- Batch capacity.
- Threading configuration.
- Model quantization and exact backend parameters.
- Input-token generator version and seed.
- Decode KV-cache layout and state hash.
- Context-length description.
- Prefill allocation.
- Decode allocation.
- Sequence count.
- Collector input-schema hash.

Store raw repetition durations:

\[
\{\tau_1,\tau_2,\ldots,\tau_R\}.
\]

Do not cache only summary statistics. Raw observations allow uncertainty models and confidence intervals to be recomputed without rerunning the backend.

Write cache entries atomically under a per-key lock. A terminated collection must not invalidate previously completed entries. Repeated runs must deduplicate valid observations and collect only the missing repetition count. Requested repetition count is append progress, not part of cache identity. Store an observation identifier, monotonic timestamp, wall-clock timestamp, randomized execution position, and collection-run identifier with every duration.

### Initial composition grid

Begin with:

\[
p\in\{0,16,32,64,128,256,384,512\},
\]

\[
d\in\{0,1,2,4,8,16,32,64\}.
\]

Filter combinations through exact backend token, sequence, context, and memory limits. Include pure prefill, pure decode, mixed, and capacity-boundary compositions.

Repeat the grid across context-length buckets:

\[
[1,512],\quad[513,2048],\quad[2049,4096],\quad[4097,8192].
\]

If the selected model has a smaller context capacity, truncate the bucket set accordingly.

### Training and held-out data

Split complete composition cells rather than individual repetitions:

- 60% of cells for model fitting.
- 20% of cells for risk calibration.
- 20% of cells for held-out oracle and scoring.

A cell is the complete tuple of allocation, sequence composition, and context bucket. Stratify the split by context bucket, demand mix, and capacity region. Keeping all repetitions from a cell in one split prevents model-fitting leakage.

Fit the nominal runtime models only on training cells:

- QP affine runtime model.
- SOCP affine nominal model.
- SOCP uncertainty shape \(L_\tau\).

Calibrate \(\delta_\tau\) and \(\kappa\) only on calibration cells against the same predeclared one-sided runtime coverage target, initially 99%. Use the smallest nonnegative value for each formulation that reaches the target empirical coverage:

\[
P(\tau_{\mathrm{observed}}\le\tau_{\mathrm{bound}})\ge0.99.
\]

Report achieved calibration and held-out coverage for both formulations. This makes their risk tolerances comparable. Select all model hyperparameters before inspecting held-out decision results.

Within each held-out cell, deterministically split raw repetitions into two disjoint groups:

- Oracle-selection repetitions define the empirical runtime distribution used to select \(B_{\mathrm{oracle}}\).
- Evaluation repetitions score the QP, SOCP, and previously selected oracle.

Never use an evaluation repetition to select the oracle. As a sensitivity analysis, repeat this split under fixed resampling seeds and reselect the oracle inside each resample.

### Offline workload snapshots

Create deterministic workload-state snapshot files from benchmark traces or a factorial snapshot generator before running this experiment. Snapshot creation is a data-preparation step, not part of either optimization setup. A snapshot contains only fixed optimization inputs:

- Current time.
- Queued prefill requests.
- Active decode requests.
- Arrival times.
- Last-token times.
- Prompt lengths.
- Context lengths.
- Previous aggregate allocation.
- Sliding TTFT observations.
- Sliding TPOT observations.
- Resource limits.
- Prefill and decode request order.
- Precomputed \(g_P\) and \(g_D\).
- Permitted cached candidate compositions.
- Recovery-mode label.

For every snapshot:

1. Read the shared deadline-pressure coefficients and fixed candidate set.
2. Solve the QP with ProxQP.
3. Solve the SOCP with Clarabel.
4. Project each continuous solution to nearby cached integer compositions.
5. Enforce exact resource and sliding-window percentile constraints.
6. Score each projected composition using held-out backend measurements.

Both formulations must receive exactly the same snapshot, neighborhood-construction rule, projection distance, and exact feasibility checks. Their resulting neighborhood members may differ because their continuous optima may differ.

### Matched native objectives

Tier 1B and Tier 2 hold all shared quantities identical:

- \(g_P,g_D,w_\tau,\rho_P,\rho_D\).
- Token, sequence, memory, and availability bounds.
- Nominal affine runtime coefficients.
- Previous allocation.
- Projection weights and candidate set.
- Solver tolerances and stopping limits.

The only formulation difference is runtime protection: the QP uses \(\delta_\tau\), while the SOCP uses \(\kappa r_\tau\). To match the stability objective in [formulations.md](formulations.md), set the SOCP coefficient to:

\[
w_\Delta=\frac12.
\]

This makes \(w_\Delta q\) equal to the QP's two \(\rho/2\) stability terms when the rotated-cone epigraph is tight.

### Projection without interpolation bias

Restrict the initial experiment's discrete candidates to compositions present in the held-out cache. This ensures decision quality is evaluated with measured runtime rather than an interpolated surrogate.

Define \(\mathcal N(x^\star)\) as the at most \(K=16\) cached feasible compositions with the smallest normalized projection distance. Break equal-distance ties by smaller conservative runtime, then larger useful work, then lexicographic \((p,d)\) order. If no composition survives exact feasibility checks, record an empty-neighborhood outcome and apply the fixed recovery rule supplied by the snapshot. A snapshot is labelled recovery mode only when its pre-existing sliding window already violates an SLO before adding a candidate outcome. Do not expand the neighborhood adaptively for only one formulation.

For continuous optimum \(x^\star=(p^\star,d^\star)\), select:

\[
B^\star=
\arg\min_{B\in\mathcal N(x^\star)}
\left\|
\begin{bmatrix}
\sqrt{\omega_P}(p(B)-p^\star)/B_{\mathrm{tok}}\\
\sqrt{\omega_D}(d(B)-d^\star)/S_{\max}
\end{bmatrix}
\right\|_2
\]

subject to exact resource and predicted sliding-window p99 constraints.

### One-step realized scoring

For each snapshot, candidate composition, and sampled cached duration \(\tau\), set:

\[
t'=t_{\mathrm{now}}+\tau.
\]

The snapshot supplies the fixed request order and the mapping from an aggregate composition to served request identifiers and token ranges. No ranking decision occurs inside the experiment.

For every selected prefill whose allocated range completes its prompt, append:

\[
\mathrm{TTFT}_i=t'-t_{\mathrm{arrival},i}
\]

to a temporary copy of the timestamped TTFT window. For an unfinished prefill, add its current right-censored lower bound \(t'-t_{\mathrm{arrival},i}\) to the temporary constraint-check multiset, but not to the completed-observation output window.

For every selected decode, append:

\[
\mathrm{TPOT}_j=t'-t_{\mathrm{lastToken},j}
\]

to a temporary copy of the timestamped TPOT window. For an unselected active decode, add the right-censored lower bound \(t'-t_{\mathrm{lastToken},j}\) to the temporary constraint-check multiset. Evict timestamped historical observations older than \(W\) before calculating either percentile.

This augmented constraint multiset prevents an allocation from appearing feasible merely because it postpones an overdue request without producing a completed observation.

For a finite multiset of \(N\) sorted observations, use the nearest-rank convention:

\[
Q_{0.99}=y_{(\lceil0.99N\rceil)}.
\]

Define useful work as:

\[
U(B)=\eta_P\frac{p(B)}{B_{\mathrm{tok}}}
+\eta_D\frac{d(B)}{S_{\max}}.
\]

Define realized one-step cost using the shared objective terms and sampled runtime:

\[
\begin{aligned}
J_{\mathrm{realized}}(B;\tau)
= {}&-g_P\frac{p(B)}{B_{\mathrm{tok}}}
-g_D\frac{d(B)}{S_{\max}}
+w_\tau\frac{\tau}{T_{\mathrm{batch}}}\\
&+\frac{\rho_P}{2}
\left(\frac{p(B)-p_{\mathrm{prev}}}{B_{\mathrm{tok}}}\right)^2
+\frac{\rho_D}{2}
\left(\frac{d(B)-d_{\mathrm{prev}}}{S_{\max}}\right)^2.
\end{aligned}
\]

Average cost and SLO outcomes across the independent evaluation repetitions for that cell. All durations use nanoseconds internally.

### Measurement-backed oracle

For each snapshot, enumerate all held-out cached compositions that satisfy exact resource limits. Select the oracle using only oracle-selection repetitions and the same realized objective and percentile transition rules. A candidate is oracle-feasible when both empirical percentile constraints hold in at least 99% of its oracle-selection repetitions.

Define:

\[
B_{\mathrm{oracle}}=
\arg\min_{B\in\mathcal C_{\mathrm{heldout}}}
J_{\mathrm{realized}}(B)
\]

subject to realized:

\[
Q_{0.99}^{W}(\mathrm{TTFT}\mid B)\le T_{\mathrm{TTFT}},
\]

\[
Q_{0.99}^{W}(\mathrm{TPOT}\mid B)\le T_{\mathrm{TPOT}}.
\]

If no candidate satisfies the current percentile constraints, the snapshot must already be labelled recovery mode. In recovery mode, rank candidates using this fixed lexicographic rule:

1. Minimum total percentile violation.
2. Minimum realized objective.

### Decision-quality metrics

Realized objective regret:

\[
R_J(B)=J_{\mathrm{realized}}(B)-J_{\mathrm{realized}}(B_{\mathrm{oracle}}).
\]

Normalized allocation error:

\[
R_x(B)=
\left\|
\begin{bmatrix}
(p(B)-p(B_{\mathrm{oracle}}))/B_{\mathrm{tok}}\\
(d(B)-d(B_{\mathrm{oracle}}))/S_{\max}
\end{bmatrix}
\right\|_2.
\]

Also record:

- TTFT p99 violation indicator and magnitude.
- TPOT p99 violation indicator and magnitude.
- Joint SLO satisfaction.
- Useful work selected.
- Realized runtime headroom.
- Projection distance.
- Empty feasible-neighborhood rate.
- QP and SOCP allocation disagreement.
- Solver failure or timeout rate.

Measure the SOCP's protective value:

\[
P(\text{QP violates and SOCP satisfies})
\]

and the reverse outcome:

\[
P(\text{SOCP violates and QP satisfies}).
\]

Measure unnecessary conservatism only on snapshots where both satisfy the SLOs:

\[
U_{\mathrm{lost}}=U(B_{\mathrm{oracle}})-U(B_{\mathrm{selected}}).
\]

## Statistical analysis

All primary comparisons are paired because both formulations receive the same instances and snapshots.

Use:

- Bootstrap confidence intervals for median and p99 timing differences.
- Paired bootstrap confidence intervals for regret differences.
- McNemar's test for paired SLO-violation outcomes, clustered or aggregated at the independent snapshot level when snapshots share a source trace.
- Effect sizes and confidence intervals alongside significance tests.

Report results for the complete set and separately for:

- Low-uncertainty snapshots.
- High-uncertainty snapshots.
- Prefill-heavy snapshots.
- Decode-heavy snapshots.
- Tight-capacity snapshots.
- Recovery-mode snapshots.

Do not select the preferred formulation from aggregate averages alone.

Predeclare equivalence margins before collecting Tier 2 results. Initial margins are:

- Decision-time difference: 1% of median measured batch duration.
- Joint-SLO violation-rate difference: 0.5 percentage points.
- Normalized regret difference: 1% of the oracle objective interquartile range.

Revise these values only from the Tier 1 pilot and record the revision before inspecting held-out Tier 2 outcomes.

## Decision rules

Prefer the QP when:

- Its p99 total decision time is materially lower.
- Its SLO violation rate is not materially worse.
- Its realized oracle regret is similar.
- The SOCP loses useful work without improving SLO satisfaction.

Prefer the SOCP when:

- It materially reduces held-out SLO violations.
- It has lower regret in high-uncertainty regions.
- Its advantage persists across context-length buckets.
- Its solve overhead is negligible relative to backend batch time.

Use this overhead target:

\[
p99(t_{\mathrm{decision}})
\le 0.01\,p50(t_{\mathrm{batch}}).
\]

This limits optimization and projection to 1% of a typical batch interval.

## Output artifacts

The experiment should produce:

- A versioned backend measurement cache.
- Raw Tier 1 timing records.
- Raw Tier 2 snapshot decisions.
- Versioned snapshot schema with explicit units.
- Versioned cache schema and collector-input schema.
- Model-fit parameters and diagnostics.
- Per-instance solver residuals and statuses.
- A machine-readable summary.
- Tables comparing timing, violations, regret, and useful work.

Raw outputs must be sufficient to regenerate every reported aggregate without rerunning the solvers or backend.

## Interpretation boundary

The experiment compares optimization setups for isolated next-batch decisions. It does not establish end-to-end scheduler performance, queue stability, long-horizon fairness, or admission-control behavior. Those require a separate online policy experiment and are outside this design.
