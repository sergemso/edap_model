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

---

## ADR-022: v4.1 Revision Scope — Hardening Plus Selective v5.0 Pull-Forward
**Status:** Accepted
**Date:** 2026-10-18
**Context:** A 2026-10-18 audit of v4.0 found integrity defects (paper does not describe the simulated model; GPU/CPU paths diverge; see ADR-023/024) and statistical weaknesses (chance-level test accuracy, no baselines/CIs, boundary-constrained calibration; see ADR-025). ROADMAP planned these fixes for v5.0 (Q4 2026), but submission-readiness requires them now.
**Decision:** Execute a v4.1 cycle: fix all integrity and statistical defects, engage Fermi prior art (ADR-029), and pull forward three v5.0 items — sample expansion (Byzantium + Japan load-bearing; Tang/Song China + Britain stretch), Bayesian calibration (ADR-026), proxy validation study (ADR-027). N(t) population dynamics stay deferred to v5.0. Target venue: *International Journal of Astrobiology*. Implementation plan: `docs/plans/v4_1_hardening.md`.
**Alternatives:**
- v4.1 hardening only (prose + statistics, no new data): Rejected — the 12-point sample remains the weakest flank; new civilizations are the strongest possible answer.
- Full v5.0 now: Rejected — N(t) requires 3+ weeks (ADR-015) and changes the paper's central claim from hypothesis to simulation result; too much for one cycle.
- Critique memo only, no changes: Rejected — defects are already public in the v4.0 release; leaving them unaddressed is worse than fixing.
**Consequences:**
- Positive: Submittable manuscript with defensible statistics; LOO-CV replaces the arbitrary train/test split as headline evidence.
- Negative: Some v4.0 metrics will get worse when honestly evaluated (baselines, LOO); the release notes must explain deltas, not hide them. Cycle length grows by the proxy-construction effort for 2–4 civilizations.

---

## ADR-023: Reset/Recovery Mechanism — Ablate, Then Document
**Status:** Accepted
**Date:** 2026-10-18
**Context:** The published equations (model.tex, eqs. 1–3) describe a deterministic 3-variable ODE system. The simulated system (edap_model/dynamics.py) additionally contains: a 4th state variable `recovery_potential`, a stochastic reset mechanism (`p_reset` = product of two sigmoids, RNG-gated, that overrides dT/dK/dC), and a per-civilization `historical_T_peak` recovery scaling. None of these appear in the paper, THESES.md, or ADR. The published equations therefore cannot reproduce the published numbers.
**Decision:** First run an ablation grid (reset on/off × shock mode historical/OU/none × all civilizations; plan step 0.3) to establish whether the hidden mechanism is load-bearing (threshold: ΔTA ≥ 0.1 on any civilization). Then document the mechanism fully in the model section (equations transcribed exactly from code) together with the ablation table and a hand-set-inputs disclosure table, regardless of the ablation outcome. Ablation informs emphasis, not whether to disclose.
**Alternatives:**
- Document as-is without ablation: Rejected — we would not know whether the mechanism does real work in the published numbers, and a reviewer may ask.
- Remove the mechanism from code and recalibrate to match the published equations: Rejected — premature before knowing its contribution; recalibration happens anyway in v4.1 (ADR-025/026) and removal remains possible later if the ablation shows it is inert.
**Consequences:**
- Positive: Reproducibility gap closed with evidence; converts a hidden degree-of-freedom objection into a documentation strength.
- Negative: The model section becomes longer and the "simple 3-variable system" story is complicated; some reviewers will see the reset mechanism as ad hoc.

---

## ADR-024: GPU/CPU Numerical Parity — Unify Semantics, Enforce by Test
**Status:** Accepted
**Date:** 2026-10-18
**Context:** GUARDRAILS requires the GPU path to produce results identical to the CPU path within numerical tolerance. It currently does not: the CPU path (`dynamics_cpu_single`) uses an `in_reset_zone` latch and `recovery_potential += d_rp * 0.1`; the GPU kernel uses no latch, `d_rp * 0.05`, and `recovery_potential *= 0.7` after reset. Published metrics come from the CPU path while calibration/Monte Carlo run on GPU — the two regimes are currently different models.
**Decision:** Unify both paths on a single canonical semantics (the CPU path's, since published metrics and `solve_ivp` runs use it), and add a permanent parity test (`tests/test_parity.py`) that asserts step-for-step trajectory equality at identical seeds. Extends ADR-021.
**Alternatives:**
- Keep both, document the divergence: Rejected — violates an existing guardrail and makes calibration and evaluation subtly inconsistent.
- Unify on GPU semantics: Rejected — evaluation path (CPU) defines the published numbers; changing it would silently alter all results.
**Consequences:**
- Positive: Calibration and evaluation simulate the same model; guardrail restored and mechanically enforced.
- Negative: Any behavior change in one path must now be mirrored in the other (already a guardrail; now tested).

---

## ADR-025: Statistical Evaluation Protocol v2 — Baselines, CIs, LOO-CV, Bifurcation Sensitivity
**Status:** Accepted
**Date:** 2026-10-18
**Context:** v4.0 evaluation weaknesses: (i) test turn accuracy averages 0.516 ≈ chance while the abstract leads with in-sample 0.70; (ii) the binomial-vs-0.5 null assumes i.i.d. signs, but T-series are strongly trending, so an "always up" predictor could match reported accuracy; (iii) only the USA gets a significance test; no metric carries a confidence interval; (iv) the train/test split (2+2 civilizations) is arbitrary; (v) calibrated α_K_base = 0.103 sits at the lower bound of its (0.10, 0.90) box, making the "precisely at bifurcation 0.12" claim suspect of being a box artifact. Extends ADR-011/012; supersedes their evaluation protocol while keeping the metric definitions.
**Decision:** Protocol v2: (a) report turn accuracy against persistence and majority-sign baselines with a block-permutation null (respects serial correlation); (b) exact binomial tests and Wilson 95% CIs for every civilization; (c) leave-one-civilization-out cross-validation as the headline empirical number (requires ≥6 civilizations — ADR-022); (d) bifurcation-threshold sensitivity to ±20% perturbations of the other calibrated parameters, plus recalibration with α_K_base bounds widened to (0.01, 0.90); (e) abstract must not lead with in-sample numbers (also a GUARDRAILS addition).
**Alternatives:**
- Keep v4.0 protocol, add caveats: Rejected — caveats don't fix a wrong null or a boundary-constrained headline claim.
- Switch to full likelihood-based inference immediately: Rejected for the point estimate — covered instead by the Bayesian layer (ADR-026); frequentist checks remain necessary for turn-pattern accuracy.
**Consequences:**
- Positive: Every reported number carries the right null and interval; headline claim becomes held-out performance.
- Negative: Honest evaluation will likely show the model near its baselines out-of-sample; the paper must own that (framework contribution first, empirical alignment second).

---

## ADR-026: Bayesian (ABC-Style) Calibration Alongside Differential Evolution
**Status:** Accepted
**Date:** 2026-10-18
**Context:** ADR-004 chose differential evolution: fast, but point estimates only — "no uncertainty quantification" was a listed consequence and is now a standing reviewer objection (6 parameters, ~10 informative turns per fold). A full-likelihood MCMC is intractable because the system has a stochastic reset mechanism and a discrete turn-pattern objective.
**Decision:** Add an ABC-style layer (emcee ensemble sampler): Gaussian-kernel pseudo-likelihood on the mean per-civilization (1 − turn accuracy) distance over batch simulations; uniform priors equal to the widened DE bounds; output posterior medians/90% CrI for parameters and posterior-predictive turn-accuracy distributions. Differential evolution remains the point-estimate path; both are reported. Includes an identifiability review (posterior-vs-prior K-S tests, rank correlations) with the rule: if ≥3 of 6 parameters are unidentified, fix them at expert values in calibration and state so in the paper. Extends ADR-004.
**Alternatives:**
- Full PyMC model with ODE solver in the graph: Rejected — gradient-free, slow for batch ODEs with stochastic resets; overkill.
- Bootstrap over DE restarts: Rejected — resamples optimizer noise, not parameter uncertainty.
- Defer to v5.0 as originally planned: Rejected — pull-forward decided in ADR-022; "existence proof" language in THESES §7 cannot survive review without uncertainty.
**Consequences:**
- Positive: Credible intervals on parameters and predictions; identifiability becomes a known, reportable property.
- Negative: Second calibration code path to maintain; ABC tolerance (ε) choice must be documented and defended.

---

## ADR-027: Proxy Disclosure and Validation Package
**Status:** Accepted
**Date:** 2026-10-18
**Context:** ADR-009 deferred proxy validation to v5.0. Remaining objections that v4.1 must answer: (i) per-civilization hand-set inputs (T0/K0/C0, shock dates and magnitudes, historical_T_peak, decay_factor) are undisclosed degrees of freedom; (ii) T/K/C indices are differently composed per civilization (Rome lacks R&D proxies, etc.), so ADR-017's "directly comparable" claim is asserted, not quantified; (iii) proxies are author interpolations from cited sources with no uncertainty labels and no external validation.
**Decision:** Three-part package: (a) hand-set inputs disclosure table in the paper (every per-civilization value with its source); (b) normalization sensitivity analysis (weights ±30%, drop-one-proxy) quantifying worst-case |ΔT|,|ΔK|,|ΔC|, with a comparability caveat added against ADR-017's claim; (c) modern-era validation correlating USA/Russia proxies against external indices (WID.world, V-Dem, WVS, OWID energy, World Bank R&D) plus a face-validity table with uncertainty grades (A/B/C) for Rome/USSR/Byzantium. Extends ADR-009.
**Alternatives:**
- Full V-Dem/Polity correlation study for all civilizations: Rejected — no coverage for ancient cases; the modern-era subset is the honest, feasible scope now.
- Status quo with limitation text: Rejected — ADR-022 made validation in-scope; proxy construction for new civilizations (ADR-022) without construction rules would compound the problem.
**Consequences:**
- Positive: Proxies become defensible-by-documentation; new civilizations inherit the construction rules (now in GUARDRAILS).
- Negative: Validation may show weak correlations for some proxies — reported honestly, possibly weakening some claims; uncertainty grades C will dominate ancient indicators.

---

## ADR-028: Terminology — Keep "Intraspecific Economic Cannibalism", Add Defense and Neutral Synonym
**Status:** Accepted
**Date:** 2026-10-18
**Context:** The core term is provocative and pattern-matches to normative framing that GUARDRAILS prohibit; reviewers may read it as political. But the model and v4.0 release (Zenodo DOI, GitHub) are published under this term, and it is the memorable brand.
**Decision:** Keep the term. Add: (i) an explicit defense paragraph in the introduction — the biological metaphor's rationale (extraction consumes conspecifics' adaptive fitness, analogous to intraspecific parasitism), the non-normative structural definition up front; (ii) first-mention pairing with the neutral synonym "elite rent extraction", used at least once in the abstract and interchangeably thereafter.
**Alternatives:**
- Rename to "elite rent extraction" everywhere: Rejected — breaks continuity with the published v4.0 release and the software citation.
- Keep with no defense: Rejected — leaves the normative-reading objection fully open at review.
**Consequences:**
- Positive: Brand preserved; objection blunted; guardrail-consistent.
- Negative: Some reviewers will still object to the term regardless of defense.

---

## ADR-029: Fermi Positioning — Engage Prior Art, Add Universality Argument
**Status:** Accepted
**Date:** 2026-10-18
**Context:** The endogenous-stagnation idea has published prior art: Haqq-Misra & Baum (2009, Int. J. Astrobiol.) argue nearly the "Great Plateau" (the sustainability solution); Kinouchi (2001) models persistent uncivilizations; Landis (1999) percolation; Ćirković's Great Silence corpus frames the field. Seshat (Turchin et al. 2018) is the empirical benchmark cliodynamics uses. v4.0 cites none of these; claiming the plateau concept as novel risks desk rejection at the target venue. Separately, the Fermi link assumes without argument that elite extraction is a convergent property of any complex civilization (alien biochemistries, eusocial intelligences, AI singletons are unaddressed).
**Decision:** (a) Add a related-work subsection positioning the Great Plateau against Haqq-Misra & Baum (2009), Kinouchi (2001), Landis (1999), Ćirković (2018); reposition EDAP's novelty as the formal, calibrated T-K-C dynamical system, not the stagnation concept; one paragraph on Seshat as the cliodynamic benchmark and why EDAP's proxy approach differs. (b) Add a universality subsection (labeled conjecture per GUARDRAILS): convergence argued from surplus-creation enabling capture, information asymmetry favoring extractors, and coalition stability of extractors over dispersed cooperators; eusocial, singleton, and post-scarcity cases treated explicitly as boundary cases where the model does not apply.
**Alternatives:**
- Cliodynamics-primary reframe (Fermi as appendix): Rejected — the astrobiology venue and the paper's identity depend on the Fermi frame; engagement, not retreat, is the fix.
- Minimal citation additions without positioning: Rejected — citations without positioning read as checkbox compliance and leave the novelty claim ambiguous.
**Consequences:**
- Positive: Novelty claim becomes precise and defensible; desk-reject risk lowered; universality gap closed as conjecture.
- Negative: Concedes that the plateau idea itself is not new — the paper's originality now rests entirely on the formalization and its (modest, honestly reported) empirical alignment.