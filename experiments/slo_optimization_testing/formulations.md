# Next-Batch Multi-SLO Optimization Formulations

This note specifies two continuous next-batch allocation models:

1. A linearly constrained, strictly convex quadratic program (QP).
2. A second-order cone program (SOCP) with allocation-dependent runtime uncertainty.

Both models optimize aggregate mixed-batch composition. Exact request selection and sliding-window percentile enforcement occur during discrete projection.

## Shared decision variables

Let

\[
x =
\begin{bmatrix}
p\\
d
\end{bmatrix},
\]

where:

- \(p\) is the continuous prefill-token allocation.
- \(d\) is the continuous decode-item allocation.
- \(x_{\mathrm{prev}}=[p_{\mathrm{prev}},d_{\mathrm{prev}}]^\top\) is the previous allocation.

Requests are ranked before the solve. Prefill requests are ordered by TTFT pressure, and decode requests are ordered by TPOT pressure. The continuous optimizer chooses only the aggregate allocation along those fixed orders.

## Deadline-pressure coefficients

For prefill request \(i\), define normalized TTFT age:

\[
a_i^T =
\frac{t_{\mathrm{now}}-t_{\mathrm{arrival},i}}
     {T_{\mathrm{TTFT}}}.
\]

For decode request \(j\), define normalized TPOT age:

\[
a_j^P =
\frac{t_{\mathrm{now}}-t_{\mathrm{lastToken},j}}
     {T_{\mathrm{TPOT}}}.
\]

Using the squared hinge

\[
h_\gamma(a)=\left[\max(0,a-\gamma)\right]^2,
\]

calculate the allocation rewards once before each solve:

\[
g_P =
w_T\frac{1}{|\mathcal P|}
\sum_{i\in\mathcal P}h_{\gamma_T}(a_i^T)
+w_U\eta_P,
\]

\[
g_D =
w_P\frac{1}{|\mathcal D|}
\sum_{j\in\mathcal D}h_{\gamma_P}(a_j^P)
+w_U\eta_D.
\]

An empty request set contributes zero to its corresponding average.

## 1. Linearly constrained QP

Use the affine runtime model

\[
\widehat\tau(p,d)=\tau_0+\tau_Pp+\tau_Dd
\]

and affine memory model

\[
\widehat M(p,d)=M_0+m_Pp+m_Dd.
\]

The optimization problem is

\[
\boxed{
\begin{aligned}
x^\star=(p^\star,d^\star)
=\arg\min_{p,d}\quad
&-g_P\frac{p}{B_{\mathrm{tok}}}
-g_D\frac{d}{S_{\max}}
+w_\tau
\frac{\tau_0+\tau_Pp+\tau_Dd}{T_{\mathrm{batch}}}
\\
&+\frac{\rho_P}{2}
\left(\frac{p-p_{\mathrm{prev}}}{B_{\mathrm{tok}}}\right)^2
+\frac{\rho_D}{2}
\left(\frac{d-d_{\mathrm{prev}}}{S_{\max}}\right)^2
\\[4pt]
\text{subject to}\quad
&0\le p\le P_{\mathrm{available}},
\\
&0\le d\le D_{\mathrm{available}},
\\
&p+d\le B_{\mathrm{tok}},
\\
&d+\frac{p}{C_{\mathrm{chunk}}}\le S_{\max},
\\
&\frac{\tau_0+\tau_Pp+\tau_Dd-H_{\mathrm{TTFT}}}
        {s_{\mathrm{TTFT}}}+z_{\mathrm{QP}}\le0,
\\
&\frac{\tau_0+\tau_Pp+\tau_Dd-H_{\mathrm{TPOT}}}
        {s_{\mathrm{TPOT}}}+z_{\mathrm{QP}}\le0,
\\
&M_0+m_Pp+m_Dd\le M_{\mathrm{available}}.
\end{aligned}
}
\]

The QP Hessian is

\[
H=
\begin{bmatrix}
\rho_P/B_{\mathrm{tok}}^2 & 0\\
0 & \rho_D/S_{\max}^2
\end{bmatrix}.
\]

Choosing \(\rho_P>0\) and \(\rho_D>0\) makes \(H\succ0\). The objective is then strictly convex, the feasible region is polyhedral, and the continuous optimum is unique whenever the problem is feasible.

The headrooms \(H_{\mathrm{TTFT}}\) and \(H_{\mathrm{TPOT}}\) are computed from the current sliding-window percentile state. The positive scales \(s_{\mathrm{TTFT}}\) and \(s_{\mathrm{TPOT}}\) are estimated only from training residuals. The unitless QP margin is calibrated as

\[
z_{\mathrm{QP}}=\frac{Q_{0.99}(e)}{s_\tau},
\]

where

\[
e=\tau_{\mathrm{observed}}-\widehat\tau(p,d).
\]

To distinguish safe interior allocations from allocations near either hard boundary, introduce \(u_T,u_D\ge0\) and add

\[
\frac{\lambda_R}{2}(u_T^2+u_D^2)
\]

to the QP objective, subject to

\[
u_j\ge z_j^{\mathrm{QP}}(p,d)+b,
\qquad u_j\ge0,
\qquad j\in\{\mathrm{TTFT},\mathrm{TPOT}\}.
\]

Here \(b\) is the dimensionless boundary buffer. These are linear constraints, so the formulation remains a linearly constrained convex QP.

## 2. Allocation-uncertainty SOCP

The SOCP retains the affine nominal runtime but adds allocation-dependent uncertainty:

\[
\widehat\tau_{\mathrm{robust}}(p,d)
=\tau_0+\tau_Pp+\tau_Dd
+\kappa
\left\|
L_\tau
\begin{bmatrix}
1\\p\\d
\end{bmatrix}
\right\|_2.
\]

Introduce:

- \(r_\tau\), the runtime uncertainty radius.
- \(t\), the conservative runtime bound.
- \(q\), the squared allocation-change epigraph.

The conic problem is

\[
\boxed{
\begin{aligned}
\min_{p,d,t,r_\tau,q}\quad
&-g_P\frac{p}{B_{\mathrm{tok}}}
-g_D\frac{d}{S_{\max}}
+w_\tau\frac{t}{T_{\mathrm{batch}}}
+w_\Delta q
\\[4pt]
\text{subject to}\quad
&0\le p\le P_{\mathrm{available}},
\\
&0\le d\le D_{\mathrm{available}},
\\
&p+d\le B_{\mathrm{tok}},
\\
&d+\frac{p}{C_{\mathrm{chunk}}}\le S_{\max},
\\
&M_0+m_Pp+m_Dd\le M_{\mathrm{available}},
\\
&\left\|
L_\tau
\begin{bmatrix}
1\\p\\d
\end{bmatrix}
\right\|_2\le r_\tau,
\\
&\tau_0+\tau_Pp+\tau_Dd+\kappa r_\tau\le t,
\\
&\frac{t-H_{\mathrm{TTFT}}}{s_{\mathrm{TTFT}}}\le0,
\\
&\frac{t-H_{\mathrm{TPOT}}}{s_{\mathrm{TPOT}}}\le0,
\\
&\left(
q,\frac12,
\begin{bmatrix}
\sqrt{\rho_P}\dfrac{p-p_{\mathrm{prev}}}{B_{\mathrm{tok}}}\\[5pt]
\sqrt{\rho_D}\dfrac{d-d_{\mathrm{prev}}}{S_{\max}}
\end{bmatrix}
\right)\in\mathcal Q_r.
\end{aligned}
}
\]

The rotated second-order cone is

\[
\mathcal Q_r=
\left\{(u,v,w):2uv\ge\|w\|_2^2,\ u\ge0,\ v\ge0\right\}.
\]

The final cone constraint implies

\[
q\ge
\rho_P\left(\frac{p-p_{\mathrm{prev}}}{B_{\mathrm{tok}}}\right)^2
+\rho_D\left(\frac{d-d_{\mathrm{prev}}}{S_{\max}}\right)^2.
\]

This formulation remains convex. It is useful when runtime prediction error changes materially across prefill-heavy, decode-heavy, and mixed allocations.

The SOCP uses the same squared-hinge auxiliaries, replacing \(z_j^{\mathrm{QP}}\) with the allocation-dependent robust score

\[
z_j^{\mathrm{SOCP}}(p,d)=
\frac{\widehat\tau(p,d)+\kappa r_\tau-H_j}{s_j}.
\]

The quadratic auxiliary penalty and second-order uncertainty constraint together remain an SOCP with a convex quadratic objective.

## Coefficient-covariance SOCP

The third offline category, `covariance_socp`, models uncertainty in the fitted
runtime coefficients rather than fitting a diagonal residual-shape heuristic.
For allocation (x=(p,d)) and context (c), define

\[
\phi(x,c)=\begin{bmatrix}1&p&d&c\end{bmatrix}^{\mathsf T},
\qquad
\widehat\mu(x,c)=\widehat\beta^{\mathsf T}\phi(x,c).
\]

Using only the training split, ordinary least squares gives

\[
\widehat\beta=(X^{\mathsf T}X)^{-1}X^{\mathsf T}y,
\qquad
\widehat\sigma_\varepsilon^2=
\frac{\lVert y-X\widehat\beta\rVert_2^2}{n-k},
\]

where (k=4). The estimated coefficient covariance is

\[
\widehat\Sigma_\beta=
\widehat\sigma_\varepsilon^2(X^{\mathsf T}X)^{-1}.
\]

Let (L_\beta L_\beta^{\mathsf T}=\widehat\Sigma_\beta). The full predictive
standard deviation includes uncertainty in the fitted mean and irreducible
backend timing variation:

\[
\widehat\sigma_{\mathrm{pred}}(x,c)=
\sqrt{
\phi(x,c)^{\mathsf T}\widehat\Sigma_\beta\phi(x,c)
+\widehat\sigma_\varepsilon^2
}
=
\left\|
\begin{bmatrix}
L_\beta^{\mathsf T}\phi(x,c)\\
\widehat\sigma_\varepsilon
\end{bmatrix}
\right\|_2.
\]

The calibration split determines the one-sided multiplier

\[
\kappa_{\mathrm{cov}}=
Q_{0.98}\left(
\frac{\max\{0,y_i-\widehat\mu(x_i,c_i)\}}
{\widehat\sigma_{\mathrm{pred}}(x_i,c_i)}
\right).
\]

For the fixed context of a next-batch decision, the hard continuous SLO
constraints are

\[
\boxed{
\widehat\mu(x,c)+
\kappa_{\mathrm{cov}}
\left\|
\begin{bmatrix}
L_\beta^{\mathsf T}\phi(x,c)\\
\widehat\sigma_\varepsilon
\end{bmatrix}
\right\|_2
\le H_j,
\qquad
j\in\{\mathrm{TTFT},\mathrm{TPOT}\}.
}
\]

Introducing (r_{\mathrm{cov}}) gives the explicit conic form

\[
\begin{aligned}
\min_{p,d,r_{\mathrm{cov}},u_T,u_P}\quad
&J_{\mathrm{work}}(p,d)+J_{\mathrm{stability}}(p,d)
+\frac{\lambda_R}{2}(u_T^2+u_P^2)
\\
\text{subject to}\quad
&\left\|
\begin{bmatrix}
L_\beta^{\mathsf T}\phi(x,c)\\
\widehat\sigma_\varepsilon
\end{bmatrix}
\right\|_2\le r_{\mathrm{cov}},
\\
&\widehat\mu(x,c)+\kappa_{\mathrm{cov}}r_{\mathrm{cov}}
\le H_{\mathrm{TTFT}},
\\
&\widehat\mu(x,c)+\kappa_{\mathrm{cov}}r_{\mathrm{cov}}
\le H_{\mathrm{TPOT}},
\\
&u_j\ge
\frac{\widehat\mu(x,c)+\kappa_{\mathrm{cov}}r_{\mathrm{cov}}-H_j}{s_j}
+b,
\qquad u_j\ge0,
\\
&x\in\mathcal F_{\mathrm{resources}}.
\end{aligned}
\]

The experiment currently has one measured batch-duration response. It applies
that same conservative runtime bound to the distinct TTFT and TPOT sliding-window
headrooms. It does not claim to estimate a two-output TTFT/TPOT covariance
matrix. During discrete projection, the candidate runtime bound is

\[
\min\left\{
Q_{0.98}^{\mathrm{cell,cal}},
\widehat\mu(x,c)+
\kappa_{\mathrm{cov}}\widehat\sigma_{\mathrm{pred}}(x,c)
\right\}.
\]

## Shared nearest-feasible projection

The continuous optimum is not itself executable. Generate a small neighborhood \(\mathcal N(x^\star)\) of integer allocations around \((p^\star,d^\star)\), construct actual batches in the fixed request orders, and solve

\[
\boxed{
\begin{aligned}
B^\star
=\arg\min_{B\in\mathcal N(x^\star)}\quad
&\left\|
\begin{bmatrix}
\sqrt{\omega_P}\dfrac{p(B)-p^\star}{B_{\mathrm{tok}}}\\[6pt]
\sqrt{\omega_D}\dfrac{d(B)-d^\star}{S_{\max}}
\end{bmatrix}
\right\|_2
\\[4pt]
\text{subject to}\quad
&Q_{0.99}^{W}(\mathrm{TTFT}\mid B)
\le T_{\mathrm{TTFT}},
\\
&Q_{0.99}^{W}(\mathrm{TPOT}\mid B)
\le T_{\mathrm{TPOT}},
\\
&B\in\mathcal F_{\mathrm{resources}}.
\end{aligned}
}
\]

The exact empirical percentile constraints stay outside the continuous problem because they are staircase-shaped and non-convex. Projection searches the complete measured context grid. For each measured cell it uses the smaller of the global model upper bound and the calibration-only per-cell upper quantile, then performs exact token, sequence, KV-cache, and memory checks. Held-out repetitions never influence candidate selection.

If the current sliding window already violates an SLO, replace the infeasible percentile constraint temporarily with lexicographic recovery: first minimize the violating percentile, then minimize distance to \(x^\star\).

## Symbols

- \(B_{\mathrm{tok}}\): token capacity
- \(S_{\max}\): sequence capacity
- \(C_{\mathrm{chunk}}\): nominal prefill chunk
- \(P_{\mathrm{available}}\): available prefill tokens
- \(D_{\mathrm{available}}\): available decode items
- \(g_P,g_D\): deadline-pressure rewards
- \(w_\tau\): runtime weight
- \(w_\Delta\): stability weight
- \(\rho_P,\rho_D\): stability scaling
- \(H_{\mathrm{TTFT}},H_{\mathrm{TPOT}}\): exact remaining sliding-window headrooms
- \(s_{\mathrm{TTFT}},s_{\mathrm{TPOT}}\): training-only standardization scales
- \(z_{\mathrm{QP}}\): dimensionless calibrated QP margin
- \(b\): dimensionless boundary-risk buffer
- \(\lambda_R\): squared-hinge boundary-risk weight
- \(L_\tau\): uncertainty factor
- \(\kappa\): uncertainty multiplier
- \(W\): sliding observation window
- \(\mathcal F_{\mathrm{resources}}\): exactly feasible batches

## Initial choice

Use the QP first. It has two variables, a unique optimum, linear constraints, and low fixed overhead. Use the SOCP when measurements show that runtime residual variance depends strongly on batch composition and a fixed QP margin is either unsafe or unnecessarily conservative.

The offline output reports the same linear-constrained QP twice to isolate
solver effects: `qp` uses ProxQP and `clarabel_qp` uses Clarabel. Both categories
share the fixed calibrated margin, objective, constraints, projection, and
held-out scoring. Any difference between them is therefore numerical or
solver-overhead related rather than a formulation difference.
