# SLO Optimization Conclusions

## Decision

Use the linearly constrained convex quadratic program solved by **ProxQP** as
the continuous optimizer for next-batch allocation.

This choice is based on the current offline experiment, which compares four
categories on the same snapshots, measurements, calibration split, projection
procedure, and held-out scoring:

1. `qp`: the selected QP solved by ProxQP.
2. `clarabel_qp`: the same QP solved by Clarabel.
3. `socp`: an SOCP with a fitted allocation-dependent residual shape.
4. `covariance_socp`: an SOCP using coefficient covariance plus residual
   execution variance.

The recommendation is not that QPs are universally superior to conic models.
It is that the additional conic structure tested here does not improve the
decisions on the available data, while ProxQP solves the selected formulation
with the lowest latency.

The optimizer remains an isolated next-batch composition optimizer. It does
not define request ordering, admission control, queue evolution, fairness, or
a long-horizon scheduler policy.

## Selected decision variables

For each next-batch decision, optimize

\[
x=\begin{bmatrix}p\\d\end{bmatrix},
\]

where:

- \(p\) is the continuous number of prefill tokens to process.
- \(d\) is the continuous number of decode items to process.
- \(p_{\mathrm{prev}}\) and \(d_{\mathrm{prev}}\) are the previous batch's
  allocations.

The continuous values are not directly executable. After solving, the
experiment projects \((p^\star,d^\star)\) onto a measured discrete batch
composition. Request selection itself follows fixed, externally supplied
prefill and decode orderings.

## Selected objective

The selected objective is

\[
\boxed{
\begin{aligned}
\min_{p,d,u_T,u_P}\quad
&-g_P\frac{p}{B_{\mathrm{tok}}}
-g_D\frac{d}{S_{\max}}
+w_\tau\frac{\widehat\tau(p,d)}{T_{\mathrm{batch}}}
\\
&+\frac{\rho_P}{2}
\left(\frac{p-p_{\mathrm{prev}}}{B_{\mathrm{tok}}}\right)^2
+\frac{\rho_D}{2}
\left(\frac{d-d_{\mathrm{prev}}}{S_{\max}}\right)^2
\\
&+\frac{\lambda_R}{2}\left(u_T^2+u_P^2\right).
\end{aligned}
}
\]

The affine runtime predictor is

\[
\widehat\tau(p,d)=\tau_0+\tau_Pp+\tau_Dd.
\]

### Deadline-pressure reward

The first two terms are useful-work rewards:

\[
-g_P\frac{p}{B_{\mathrm{tok}}}
-g_D\frac{d}{S_{\max}}.
\]

Because the problem is minimized, increasing useful prefill or decode work
reduces the objective. The normalization prevents the raw scales of prefill
tokens and decode sequences from determining their relative importance.

The coefficients \(g_P\) and \(g_D\) are computed before the solve from the
current request state. Conceptually, they combine deadline pressure with a
base useful-work reward. Older requests and requests approaching their SLO
deadlines therefore increase the incentive to allocate work to the
corresponding side of the mixed batch.

This term makes the optimizer responsive to the current queue without making
individual request identities optimization variables.

### Runtime cost

The term

\[
w_\tau\frac{\widehat\tau(p,d)}{T_{\mathrm{batch}}}
\]

penalizes predicted batch duration. It prevents the useful-work reward from
always pushing the solution to the largest resource-feasible batch. Dividing
by \(T_{\mathrm{batch}}\) makes the term dimensionless and gives its weight a
stable interpretation across machines and time units.

Runtime appears in both the objective and the hard constraints for different
reasons. In the objective, it expresses a preference among feasible
allocations. In the constraints, it prevents an allocation from consuming
more sliding-window headroom than is currently available.

### Stability cost

The terms

\[
\frac{\rho_P}{2}
\left(\frac{p-p_{\mathrm{prev}}}{B_{\mathrm{tok}}}\right)^2
+\frac{\rho_D}{2}
\left(\frac{d-d_{\mathrm{prev}}}{S_{\max}}\right)^2
\]

penalize abrupt changes in batch composition. This has three useful effects:

- It discourages unnecessary oscillation between prefill-heavy and
  decode-heavy allocations.
- It selects a stable point when several allocations have similar immediate
  work and runtime value.
- With \(\rho_P>0\) and \(\rho_D>0\), it makes the objective strictly convex in
  \((p,d)\), giving a unique continuous optimum whenever the feasible set is
  nonempty.

The normalization again matters. A change of one prefill token is not treated
as equivalent to a change of one decode sequence. Each change is measured as
a fraction of its corresponding capacity.

### Soft boundary-risk cost

For each SLO \(j\in\{T,P\}\), define the standardized conservative slack

\[
z_j(p,d)=
\frac{\widehat\tau(p,d)-H_j}{s_j}+z_{\mathrm{QP}},
\]

where:

- \(H_T\) is the current TTFT sliding-window headroom.
- \(H_P\) is the current TPOT sliding-window headroom.
- \(s_j>0\) is a training-only runtime standardization scale.
- \(z_{\mathrm{QP}}\) is the calibrated, unitless upper-margin correction.

The auxiliary variables satisfy

\[
u_j\ge z_j(p,d)+b,
\qquad
u_j\ge0.
\]

The objective then adds

\[
\frac{\lambda_R}{2}(u_T^2+u_P^2).
\]

This is a squared-hinge penalty near the hard SLO boundaries. It distinguishes
an allocation deep inside the feasible region from one that is technically
feasible but has little remaining margin. The hard constraints still decide
feasibility. The boundary term only ranks feasible continuous solutions.

The auxiliaries preserve the QP structure because their constraints are
linear and their objective contribution is quadratic.

## Selected hard constraints

The continuous QP is subject to

\[
0\le p\le P_{\mathrm{available}},
\]

\[
0\le d\le D_{\mathrm{available}},
\]

\[
p+d\le B_{\mathrm{tok}},
\]

\[
d+\frac{p}{C_{\mathrm{chunk}}}\le S_{\max},
\]

\[
M_0+m_Pp+m_Dd\le M_{\mathrm{available}},
\]

and the two standardized SLO constraints

\[
\frac{\widehat\tau(p,d)-H_{\mathrm{TTFT}}}
{s_{\mathrm{TTFT}}}
+z_{\mathrm{QP}}\le0,
\]

\[
\frac{\widehat\tau(p,d)-H_{\mathrm{TPOT}}}
{s_{\mathrm{TPOT}}}
+z_{\mathrm{QP}}\le0.
\]

All of these constraints are affine in the optimization variables. The
feasible region is therefore polyhedral.

The TTFT and TPOT headrooms come from their separate sliding-window p99 states.
The current experiment uses one measured batch-runtime response under both
headroom constraints. It does not claim that TTFT and TPOT are the same metric.
It claims that the next batch's duration consumes both independently computed
headrooms.

## Calibration of the fixed QP margin

Fit the affine runtime model using only the training split. For calibration
observation \(i\), define the one-sided residual

\[
e_i^+=\max\{0,\tau_i-\widehat\tau(p_i,d_i)\}.
\]

The current experiment uses

\[
\delta_{\mathrm{QP}}=Q_{0.98}(e_i^+)
\]

and converts it to a unitless constraint margin:

\[
z_{\mathrm{QP}}=rac{\delta_{\mathrm{QP}}}{s_\tau}.
\]

For the regenerated results,

\[
\delta_{\mathrm{QP}}=2.33567\times10^6\ \mathrm{ns},
\]

\[
s_\tau=1.81312\times10^6\ \mathrm{ns},
\]

and therefore

\[
z_{\mathrm{QP}}=1.28821.
\]

The p98 value is a predictive-envelope calibration choice. It is not the SLO
percentile. TTFT and TPOT remain sliding-window p99 SLOs.

The fixed margin is intentionally simple. It is appropriate here because the
tested decision-dependent uncertainty models did not improve held-out
coverage or realized SLO outcomes.

## Discrete projection and final hard checks

The continuous optimum is only a target. The executable batch is selected from
the measured grid for the current context by normalized nearest-neighbor
distance:

\[
\operatorname{dist}(B,x^\star)^2=
\left(\frac{p(B)-p^\star}{B_{\mathrm{tok}}}\right)^2
+\left(\frac{d(B)-d^\star}{S_{\max}}\right)^2.
\]

For a measured candidate \(B\), projection uses the smaller of the global
model bound and its calibration-only cell bound:

\[
\tau_{\mathrm{bound}}(B)=
\min\left\{
\widehat\tau(B)+\delta_{\mathrm{QP}},
Q_{0.98}^{\mathrm{cell,cal}}(\tau\mid B)
\right\}.
\]

It then checks exact resource feasibility and applies the candidate outcome to
temporary copies of the TTFT and TPOT sliding windows. A candidate is accepted
only if both resulting nearest-rank p99 values satisfy their targets.

This separation is deliberate:

- The continuous QP supplies a fast, smooth allocation target.
- Projection restricts execution to actual measurable batch compositions.
- The sliding-window checks enforce the operational percentile semantics.

The calibration bound is not a formal guarantee. One of 96 nonempty decisions
violated the held-out SLO in the current experiment. The limitation belongs to
the predictive bound and finite calibration data, not to ProxQP's numerical
solution of the continuous QP.

## Experimental evidence

The final offline run uses:

- 32 measured allocation cells.
- 30 backend repetitions per cell.
- 576 training observations.
- 192 calibration observations.
- 192 held-out observations.
- 100 paired next-batch snapshots.
- A 98th-percentile predictive upper envelope.

All formulations receive the same snapshots and use the same projection and
held-out scoring procedure.

### Solver and decision results

| Category | Solver | Median solve | p99 solve | Empty | Violating decisions | Mean regret | Mean useful work |
|---|---|---:|---:|---:|---:|---:|---:|
| `qp` | ProxQP | 10.17 us | 17.46 us | 4/100 | 1/96 | 0.03204 | 0.40039 |
| `clarabel_qp` | Clarabel | 29.54 us | 40.75 us | 4/100 | 1/96 | 0.02975 | 0.40169 |
| `socp` | Clarabel | 46.38 us | 68.29 us | 4/100 | 1/96 | 0.03127 | 0.40039 |
| `covariance_socp` | Clarabel | 59.33 us | 77.33 us | 4/100 | 1/96 | 0.03127 | 0.40039 |

All four categories solved all 100 continuous problems. All produced four empty
projections and one held-out violating decision among the 96 nonempty
decisions. Mean joint-SLO satisfaction was 99.65% for every category.

The measured bound coverage was:

| Bound | Calibration | Held-out | Leave-cell-out |
|---|---:|---:|---:|
| Fixed QP | 98.44% | 96.35% | 95.31% |
| Residual-shape SOCP | 98.44% | 96.35% | 95.31% |
| Coefficient-covariance SOCP | 98.44% | 96.35% | 94.79% |

The conic bounds therefore did not improve out-of-sample coverage.

## Why ProxQP is selected

### It solves the selected mathematical problem fastest

ProxQP has the lowest median and p99 solve times. Relative to Clarabel solving
the identical QP, ProxQP is approximately 2.9 times faster at the median and
2.3 times faster at p99.

The problem is particularly suitable for a small dense QP solver. The
implemented QP has four variables \((p,d,u_T,u_P)\), a positive semidefinite
quadratic objective, and 13 linear constraint rows. Introducing a general
conic solver for this geometry adds machinery without adding feasible-set
expressiveness.

### Clarabel validates the QP solution

The ProxQP and Clarabel QP categories solve exactly the same objective and
constraints. Their continuous solutions are identical at the precision
recorded in the CSV for all 100 snapshots.

They project identically on 95 of the 96 nonempty snapshots. The only
disagreement occurs at snapshot 82, where the reported continuous solution is

\[
(p^\star,d^\star)=(96,52).
\]

The measured prefill candidates 64 and 128 are equidistant from 96. Hidden
floating-point differences place the solvers on opposite sides of this exact
projection tie. ProxQP selects \((64,16)\), while Clarabel selects \((128,16)\).
The latter happens to be the oracle choice for that snapshot.

This single tie explains Clarabel QP's slightly lower mean regret and slightly
higher mean useful work. It is not evidence of a superior continuous optimum.
It is evidence that projection needs an explicit distance tolerance if
solver-independent tie behavior is required.

### The SOCPs do not improve safety

Neither SOCP reduces:

- The number of empty projections.
- The number of held-out SLO violations.
- The joint-SLO violation rate.
- The p95 regret.
- Lost useful work.

The existing SOCP and covariance SOCP select the same projected allocation on
all 96 nonempty snapshots. Their realized objective, regret, useful work, and
SLO results are consequently identical.

### The covariance model collapses toward a fixed margin

For the coefficient-covariance model, the estimated residual variance is

\[
\widehat\sigma_\varepsilon^2=3.31038\times10^{12}\ \mathrm{ns}^2,
\]

so

\[
\widehat\sigma_\varepsilon\approx1.82\ \mathrm{ms}.
\]

With

\[
\kappa_{\mathrm{cov}}=1.27883,
\]

the residual term alone contributes approximately

\[
1.27883(1.82\ \mathrm{ms})\approx2.33\ \mathrm{ms}.
\]

This is almost the same as the QP's fixed 2.336 ms margin. The
decision-dependent coefficient-covariance contribution is too small to change
the feasible region materially. The more expressive mathematical model thus
behaves like the simpler fixed-margin model on this dataset.

In addition, every selected covariance-SOCP decision records `cell_p98` as its
active projection-bound source. The covariance global bound is not the active
bound for any selected discrete candidate.

### ProxQP preserves the simplest operational path

Selecting ProxQP avoids maintaining conic uncertainty factors, covariance
decompositions, and additional cone variables in the production decision
path. The required runtime inputs are limited to:

- Affine runtime coefficients.
- Current TTFT and TPOT headrooms.
- One calibrated residual margin.
- Current resource capacities.
- Current and previous allocation state.

This simplicity matters because prediction and projection, rather than convex
solve time, are now the dominant sources of decision error.

## What the result does not establish

The experiment does not prove that the chosen optimizer satisfies a production
p99 SLO with 99% probability. In particular:

- The held-out evaluation has limited repetitions per allocation cell.
- A p99 tail cannot be estimated precisely from such a small per-cell sample.
- The predictive envelope achieves 96.35% held-out coverage, below its 98%
  calibration target.
- Three of the four empty projections occur when the held-out oracle contains
  a feasible candidate.
- The experiment evaluates isolated next-batch decisions, not queue dynamics
  over time.
- Candidate projection currently has floating-point sensitivity at exact
  nearest-neighbor ties.

These limitations argue for improving calibration and projection. They do not
provide evidence that a more expensive continuous solver would improve the
outcome.

## Conditions that would justify revisiting the decision

Reconsider an SOCP or another robust formulation if future measurements show
all of the following:

1. Runtime residual scale changes materially and predictably with allocation.
2. A decision-dependent uncertainty model improves held-out or leave-cell-out
   upper-bound coverage.
3. That coverage improvement reduces realized SLO violations or false empty
   projections.
4. The improvement persists across context tiers and independent measurement
   runs.
5. The additional solve cost remains below the decision-time budget.

Also revisit the model if separate next-batch TTFT and TPOT response predictors
become available. A genuine multi-output uncertainty model could then capture
information that the current single batch-duration response cannot represent.

## Recommended implementation boundary

The production-oriented next-batch optimization path should therefore be:

1. Compute fixed request orderings and deadline-pressure coefficients.
2. Compute exact TTFT and TPOT sliding-window headrooms.
3. Fit or update the affine runtime and memory models outside the hot solve.
4. Supply the calibrated fixed residual margin.
5. Solve the four-variable linearly constrained QP with ProxQP.
6. Project onto the complete measured candidate grid for the current context.
7. Apply exact resource and predicted sliding-window p99 checks.
8. Emit an explicit empty or recovery result when no candidate survives.
9. Log the continuous target, projected candidate, active bound source, solver
   status, solve latency, and resulting SLO margins.

The SOCP implementations should remain in the offline experiment as comparison
baselines. Clarabel QP should also remain as a numerical cross-check. Neither
needs to be part of the selected hot path unless new evidence changes the
decision.
