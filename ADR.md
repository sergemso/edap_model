# ADR.md — Architecture Decision Records for EDAP Model

## Purpose
This file documents all key design decisions for the EDAP model, with context, alternatives considered, rationale, and consequences. It serves as a permanent record of *why* the model is built the way it is.

Decisions about *how to communicate* the model's results belong in `GUARDRAILS.md`.

---

## ADR-001: Model Scope — Dissipation Only
**Status:** Accepted
**Date:** 2026-05-17
**Context:** Model could aim to explain both growth and collapse of civilizations.
**Decision:** Model explicitly scoped to dissipation only. Does not predict when T rises.
**Alternatives:**
- Full-cycle model (growth + collapse): Rejected — requires exogenous growth mechanisms (scientific revolutions, resource bonanzas) outside model scope. Would add uncalibratable parameters.
- Purely statistical model: Rejected — loses causal mechanism.
**Consequences:**
- Positive: Clear boundary conditions, honest about limitations, avoids overfitting to growth periods.
- Negative: Cannot explain why some civilizations temporarily flourish. Requires complementary theory.

---

## ADR-002: Variable Definitions
**Status:** Accepted
**Date:** 2026-05-17
**Context:** T, K, C need operational definitions that are measurable and non-normative.
**Decision:**
- T = fraction of scientifically realizable technologies deployed. T=1 = full utilization. T<1 = de-technologization.
- K = fraction extracted by elites without productive contribution. No distinction between political systems.
- C = generalized trust + institutional capacity. Not formal institutions per se, but the underlying cooperative potential.
**Alternatives:**
- T as absolute technology level: Rejected — incomparable across epochs.
- K as Gini coefficient: Rejected — measures inequality, not extraction.
- C as democracy index: Rejected — politically normative, excludes non-democratic cooperation.
**Consequences:**
- Positive: Universal definitions applicable from Rome to modernity.
- Negative: Requires proxy estimation; measurement error acknowledged.

---

## ADR-003: Number of Civilizations
**Status:** Accepted
**Date:** 2026-05-17
**Context:** How many historical cases needed for calibration and validation.
**Decision:** Four civilizations: Rome + USSR (train, n=2), Russia + USA (test, n=2).
**Alternatives:**
- 8-10 civilizations: Rejected for v4.0 — requires extensive proxy data collection. Planned for v5.0.
- 2 civilizations only: Rejected — no test set.
- Synthetic data: Rejected — defeats purpose of empirical grounding.
**Consequences:**
- Positive: Feasible scope, all four have documented proxy data.
- Negative: Small sample (12 points total). Overfitting risk acknowledged.

---

## ADR-004: Calibration Method
**Status:** Accepted
**Date:** 2026-05-18
**Context:** Choice of optimization algorithm for parameter estimation.
**Decision:** Constrained differential evolution (scipy) with penalty for K,C ∉ [0,1]. GPU-accelerated via Numba CUDA.
**Alternatives:**
- Bayesian MCMC: Rejected for v4.0 — requires likelihood function specification, prior elicitation, convergence diagnostics. Planned for v5.0.
- Manual tuning: Rejected — non-reproducible.
- Gradient-based optimization: Rejected — objective function non-smooth (turn accuracy is discrete).
**Consequences:**
- Positive: Fast, parallelizable, handles constraints naturally.
- Negative: Point estimates only, no uncertainty quantification.

---

## ADR-005: GPU Acceleration — Numba CUDA
**Status:** Superseded by ADR-021
**Date:** 2026-05-18
**Superseded by:** ADR-021 (Numerical Methods, Solver Choice, and GPU/CPU Dispatch)

---

## ADR-006: Filter Claim Strength
**Status:** Accepted
**Date:** 2026-05-19
**Context:** Whether to present plateau-to-filter transition as a finding or a hypothesis.
**Decision:** Present as qualitative hypothesis, not quantitative prediction. Explicitly state: model does not contain N(t), technosignature, or detectability variables. Filter transition is an implication of model structure requiring dedicated future work.
**Alternatives:**
- Present as confirmed finding: Rejected — unsupported by current model. Would be rightfully criticized.
- Omit entirely: Rejected — filter transition is the intellectual payoff connecting EDAP to Fermi Paradox.
**Consequences:**
- Positive: Honest, defensible. Invites future work rather than rejecting criticism.
- Negative: Weakens the "super-bomb" claim. Paper is "framework + illustration" not "proof."

---

## ADR-007: Statistical Claims
**Status:** Accepted
**Date:** 2026-05-19
**Context:** How to present USA turn accuracy given results may not significantly exceed chance.
**Decision:** Report accuracy with binomial test result. Explicitly state if null cannot be rejected. Do not claim predictive power when statistically unsupported. Present as observational alignment.
**Alternatives:**
- Omit significance test: Rejected — would be perceived as hiding negative result.
- Claim "model predicts turns": Rejected — statistically unsupported if p > 0.05.
- Drop USA entirely: Rejected — removes test set, weakens empirical contribution.
**Consequences:**
- Positive: Methodological honesty builds trust. Preempts reviewer criticism.
- Negative: Weakens apparent empirical success.

---

## ADR-008: Compensation Hypothesis Framing
**Status:** Accepted
**Date:** 2026-05-19
**Context:** Whether elite interventions (New Deal, QE, etc.) are presented as causal responses to model-detected tension.
**Decision:** Present as temporal coincidence and hypothesis-generating observation. Explicitly deny causal claim. Require forward-looking predictions for validation.
**Alternatives:**
- Causal claim: Rejected — no statistical evidence, circular reasoning risk.
- Remove entirely: Rejected — this is the most engaging narrative element.
**Consequences:**
- Positive: Falsifiable proposition stated. Future work can test it.
- Negative: Narrative weaker. Reviewer may still criticize as post-hoc.

---

## ADR-009: Proxy Validation
**Status:** Accepted
**Date:** 2026-05-19
**Context:** Whether to validate proxies against external expert indices.
**Decision:** Acknowledge as limitation. For v4.0, rely on face validity and documented sources. Correlation analysis against V-Dem, Polity IV planned for v5.0.
**Alternatives:**
- Full validation now: Rejected — delays submission.
- Ignore issue: Rejected — reviewer will flag it.
**Consequences:**
- Positive: Honest about current limitation. Planned remediation.
- Negative: Proxies remain unvalidated in v4.0. Reviewer may request validation before acceptance.

---

## ADR-010: Sensitivity Analysis
**Status:** Accepted
**Date:** 2026-05-19
**Context:** Whether to include formal sensitivity analysis for functional forms.
**Decision:** Reference existing Figure 3 panels E-F (exp and log alternatives) as preliminary evidence. Full sensitivity analysis planned for v5.0.
**Alternatives:**
- Full analysis now: Rejected — 2 days work. Defer to v5.0.
- No mention: Rejected — reviewer will demand it.
**Consequences:**
- Positive: Shows awareness. Panels E-F provide some evidence of robustness.
- Negative: Not systematic. Reviewer may still request full analysis.

---

## ADR-011: Metric Selection — Turn Accuracy + BSS Only
**Status:** Accepted
**Date:** 2026-05-18
**Context:** Choosing evaluation metrics for model performance.
**Decision:** Primary metric = turn accuracy (fraction of correctly predicted signs of dT/dt). Secondary = Brier Skill Score for binary K > K_crit prediction. Do not use RMSE, MAE, or R² against absolute T levels.
**Alternatives:**
- RMSE against T(t): Rejected — model explicitly does not predict absolute T level.
- Correlation with linear extrapolation: Rejected — causal model should not be benchmarked against statistical trend line.
- Multi-metric ensemble: Rejected — dilutes the clear signal of what model does well (direction) vs poorly (level).
**Consequences:**
- Positive: Metrics aligned with model's stated scope. Honest evaluation.
- Negative: Harder to compare with other models that report RMSE.

---

## ADR-012: Baseline Model — Climatological, Not Linear Extrapolation
**Status:** Accepted
**Date:** 2026-05-18
**Context:** What should EDAP be compared against to demonstrate added value.
**Decision:** Use Brier Skill Score against climatological baseline (always predict most common outcome). Explicitly do NOT compare against linear extrapolation of T(t).
**Alternatives:**
- Linear extrapolation baseline: Rejected — inappropriate for causal model of dissipation.
- Random walk baseline: Rejected — too weak; even climatology beats it.
- No baseline: Rejected — reviewers demand comparative evaluation.
**Consequences:**
- Positive: Appropriate baseline for binary prediction task.
- Negative: BSS may be negative, requiring explanation as model conservatism.

---

## ADR-013: Calibration Trade-off Acknowledgment
**Status:** Accepted
**Date:** 2026-05-19
**Context:** Calibration optimizing for turn accuracy may eliminate K-crossing lead time.
**Decision:** Explicitly document the trade-off. Do not claim K-crossing lead time as a result when using calibrated parameters that sacrifice it. Present lead time only for parameter sets that preserve it, as supplementary evidence.
**Alternatives:**
- Multi-objective calibration: Rejected for v4.0 — adds complexity. Planned for v5.0.
- Report both metrics at both parameter sets: Considered — clutters results section.
- Hide the trade-off: Rejected — dishonest.
**Consequences:**
- Positive: Transparent about model limitations. Preempts reviewer criticism.
- Negative: Weakens the "early warning system" narrative.

---

## ADR-014: GPU Framework — Numba CUDA
**Status:** Superseded by ADR-021
**Date:** 2026-05-18
**Superseded by:** ADR-021

---

## ADR-015: Population Dynamics Deferred to v5.0
**Status:** Accepted
**Date:** 2026-05-19
**Context:** Whether to add N(t) equation in v4.0 to quantitatively model the filter transition.
**Decision:** Defer to v5.0. v4.0 presents filter as qualitative hypothesis. Explicitly acknowledge that without N(t), filter claim is structural implication, not verified prediction.
**Alternatives:**
- Add N(t) now: Rejected — requires 3+ weeks for model extension, re-calibration, validation.
- Remove filter discussion entirely: Rejected — filter transition is the intellectual payoff.
**Consequences:**
- Positive: v4.0 ships on time. Filter hypothesis generates interest and motivates v5.0.
- Negative: Central claim of paper is "we hypothesize" not "we demonstrate."

---

## ADR-016: Functional Form of K_crit — Quadratic Upturn
**Status:** Accepted
**Date:** 2026-05-18
**Context:** Choice of functional form for technofeudal upturn in K_crit(T).
**Decision:** Quadratic form: ε_tech * max(0, T - T_sing)². Simplest polynomial satisfying boundary conditions: K_crit declines linearly, then upturns smoothly at T_sing.
**Alternatives:**
- Sigmoid: Rejected — adds parameter (steepness). Preliminary sensitivity shows similar bifurcation behavior.
- Step function: Rejected — non-differentiable, problematic for ODE solver.
- Exponential: Considered — preliminary analysis shows bifurcation preserved. Quadratic chosen for simplicity.
**Consequences:**
- Positive: Simple, 2 parameters, easy to interpret.
- Negative: Specific functional form drives the filter hypothesis. Full sensitivity analysis deferred.

---

## ADR-017: T_max ≡ 1.0 (Constant)
**Status:** Accepted
**Date:** 2026-05-18
**Context:** Whether T_max should be a free parameter or a constant.
**Decision:** T_max ≡ 1.0 always. T measures the fraction of scientifically realizable technologies that a civilization deploys. T=1 means full utilization of known science. T<1 means de-technologization.
**Alternatives:**
- T_max as free parameter: Rejected — would require different values per epoch, incommensurable across civilizations.
- T_max as function of scientific paradigm: Rejected — adds complexity without clear calibration strategy.
**Consequences:**
- Positive: Universal scale. Rome at T=0.33 and USA at T=0.76 are directly comparable.
- Negative: Requires careful definition in paper to avoid misinterpretation.

---

## ADR-018: η Removal and Restoration
**Status:** Accepted
**Date:** 2026-05-17 (removed), 2026-05-18 (restored)
**Context:** Whether external shocks S(t) should directly boost cooperation C.
**Decision:** Restore η term in both dT/dt and dC/dt after brief removal in v2.5.
**Rationale:** "Common enemy" effect is empirically well-documented (war mobilization, pandemic response, New Deal). Removing it eliminated a mechanism with strong historical support.
**Alternatives:**
- Keep η removed: Rejected — lost explanatory power for post-crisis recoveries.
- Make η civilization-specific: Considered — adds parameters. Deferred to v5.0.
**Consequences:**
- Positive: Model captures documented historical phenomenon.
- Negative: Adds one more parameter to already-large set.

---

## ADR-019: K_max(T) Evolution — Multiple Iterations
**Status:** Accepted
**Date:** 2026-05-17 to 2026-05-18
**Context:** How to model the physical ceiling on resource extraction.
**Decision (final):** K_max(T) = κ T/(T+T_h) for T < T_auto, smooth transition to 1.0 for T ≥ T_auto.
**History:**
- v2.3-v2.4: K_max(T) with T_automation → 1.0. Worked well.
- v2.5: Replaced by (1-K) logistic. Caused K → 0.92, worse.
- v2.6: Restored K_max(T). Kept since.
**Alternatives considered:**
- (1-K) logistic self-limitation: Rejected — eliminated T-dependence, caused extreme K values.
- No ceiling: Rejected — unphysical (K can exceed 1.0).
**Consequences:**
- Positive: Physically grounded. T_auto transition enables filter mechanism.
- Negative: Piecewise definition with smooth transition is mathematically inelegant but functionally necessary.

---

## ADR-020: T_min Clamping
**Status:** Accepted
**Date:** 2026-05-18
**Context:** Model was producing T < T_min despite T_min=0.08.
**Decision:** Add gradient clamping: if T < T_min and dT/dt < 0, force dT/dt = 0. Same for C at boundaries.
**Alternatives:**
- Stronger restoring force: Considered — would distort dynamics near floor.
- Accept T < T_min: Rejected — violates physical assumption that basic technologies persist.
**Consequences:**
- Positive: Model respects its own assumptions. T never falls below 0.08 in clamped version.
- Negative: Clamping is a hard constraint, not a smooth mechanism.

---

## ADR-021: Numerical Methods — Solver Choice and GPU/CPU Dispatch (Supersedes ADR-005, ADR-014)
**Status:** Accepted
**Date:** 2026-05-19
**Supersedes:** ADR-005, ADR-014
**Context:** Choice of numerical integration method, hardware target, and GPU framework for different use cases.
**Decision:**
- **Single-trajectory simulation:** `scipy.integrate.solve_ivp` with RK45, rtol=1e-8. Used for figures and final evaluation.
- **Batch computation (Monte Carlo, calibration):** Euler with fixed step dt=0.05. Used where speed dominates.
- **Attractor identification:** Euler, vectorized in numpy. Used where only final state matters.
- **GPU framework:** Numba CUDA via @cuda.jit kernels. One thread per simulation trajectory. 128 threads per block. CPU fallback via @njit.
- **Code organization:** All GPU code in dynamics.py. HAS_CUDA checked once at import. Model never requires CUDA.
**Alternatives:**
- RK45 everywhere: Rejected — calibration would take hours.
- Euler everywhere: Rejected — single trajectories benefit from adaptive error control.
- JAX, PyTorch: Rejected — no GPU on target; overkill for this system.
- GPU-only: Rejected — violates portability.
**Consequences:**
- Positive: Appropriate algorithm per use case. 500x GPU speedup for batches. Graceful CPU fallback.
- Negative: Two code paths must be synchronized. Three numerical schemes must produce consistent attractor identity.