# GUARDRAILS.md — Argumentation Constraints for EDAP Model

## Purpose
Version-invariant constraints for any text describing the EDAP model. Load before writing or editing the paper, abstract, discussion, or any public communication.

Violating these guardrails compromises scientific integrity.

---

## Permitted Claims

### Mathematical Properties
- Claims about model structure: equations, attractors, bifurcations, parameter roles
- Claims about qualitative behavior: "the system exhibits bistability," "K_crit defines a threshold"
- Claims about mathematical relationships: "α_K_eff is modulated by C"

### Empirical Alignment
- Claims about historical trajectories supported by calibration metrics
- Claims of consistency: "the model is consistent with observed patterns"
- Claims about specific civilizations when backed by reported metrics

### Theoretical Implications
- Claims about what the model implies if its assumptions hold
- Claims about candidate mechanisms for observed phenomena
- Claims that generate falsifiable hypotheses for future testing

### Academic Framing (Explicitly Permitted)
- "We propose" — standard academic framing of a hypothesis
- "We argue" — standard academic presentation of a position
- "We suggest" — appropriate for qualitative implications
- "We hypothesize" — appropriate for untested claims
- "The model identifies" — description of model output
- "The model suggests" — appropriate for qualitative implications
- "Our results indicate" — appropriate when backed by metrics
- "Our analysis reveals" — appropriate for analytical findings
- "is consistent with" — appropriate for observational alignment
- "aligns with" — appropriate for temporal coincidence
- "supports the hypothesis" — appropriate for directional evidence

---

## Prohibited Claims

### Predictive Without Statistical Support
- Never claim predictive power without reporting a significance test
- Never use "predicts," "forecasts," "demonstrates" for results that fail to reject the null
- Never present point estimates without uncertainty quantification

### Causal Without Evidence
- Never claim that modeled tension "caused" or "triggered" historical events
- Never present temporal coincidence as causal relationship
- Never attribute intentions to actors based on model output

### Overclaiming
- Never claim the model "solves" the Fermi Paradox
- Never claim the model is "validated" without specifying metrics and limitations
- Never claim the model is an "early warning system" without demonstrating lead time with significance

### Normative
- Never claim the model proves one political or economic system is superior
- Never use the model to advocate specific policies
- Never present "elites" as morally culpable based on model dynamics

---

## Required Caveats

The following must appear somewhere in every paper. Not all in the abstract — but the full set across the manuscript.

- Model scope: dissipation theory, not growth theory
- Calibration sample size and its implications for generalizability
- Proxy measurement error is unquantified
- Elite faction competition is not modeled
- Functional form sensitivity analysis is preliminary or deferred
- Filter mechanism is qualitative unless population dynamics are explicitly modeled
- Statistical significance of key results is honestly reported
- Trade-offs in calibration objectives are disclosed

---

## Tone

### Confidence Calibration
- **High confidence:** Mathematical properties of the model itself
- **Moderate confidence:** Empirical alignment with historical data
- **Low confidence:** Forward-looking implications, filter mechanism, compensation hypothesis
- **Never:** Triumphalist, dismissive of alternative explanations, politically normative

### Framing
- Present as "theoretical framework" or "formal model," not "proven theory"
- Use "suggests," "is consistent with," "aligns with" — not "demonstrates," "proves," "confirms"
- Acknowledge limitations proactively, not defensively
- Invite future work rather than claiming closure

### Phrase Replacements

| Instead of | Use |
|-----------|-----|
| "The model predicts" | "The model identifies" or "The model is consistent with" |
| "The model demonstrates" | "The model suggests" |
| "The model solves" | "The model offers a candidate mechanism for" |
| "Elites responded to" | "Interventions coincided with" |
| "Validated on" | "Calibrated on" or "Evaluated on" |
| "Early warning system" | "Structural tension detection framework" |

---

## Section-Specific Requirements

### Abstract
- Must state model type (dissipation theory)
- Must indicate whether filter claim is qualitative or quantitative
- Must honestly characterize empirical support

### Results
- Report all primary metrics, not just favorable ones
- Report statistical significance for any claimed predictive power
- If calibration involves trade-offs, disclose them

### Discussion
- Label hypotheses as hypotheses, not findings
- Distinguish "consistent with" from "evidence for"
- Discuss limitations before conclusions

### Limitations
- Must include: sample size, proxy uncertainty, unmodeled mechanisms, scope boundaries
- Must NOT downplay limitations or promise they will be "easily addressed in future work"

### Conclusion
- End on theoretical contribution, not on predictive claims
- Frame as "formal framework for investigation," not "solution"
- Mention future directions if appropriate

---

## GPU Usage

- Batch computations (Monte Carlo ≥ 100 simulations, calibration) SHOULD target NVIDIA/CUDA
- Single-trajectory simulation MAY remain on CPU
- GPU path must produce results identical to CPU path within numerical tolerance
- Performance claims must specify GPU model and CUDA version
- Graceful fallback to CPU is REQUIRED for machines without CUDA
- Any equation change MUST update both CPU and GPU implementations
- Test suite must pass on both backends

---

## Numerical Methods

- Single-trajectory simulation: adaptive-step ODE solver (e.g., RK45) for accuracy
- Batch computation (Monte Carlo, calibration): fixed-step method acceptable for speed
- Attractor identification (bifurcation analysis): fixed-step sufficient
- When using different methods for different purposes, document and verify attractor consistency

---

## Model–Paper Disclosure (v4.1+)

- Every state variable, mechanism, stochastic element, reset rule, or per-civilization hand-set input that affects simulated trajectories MUST appear in the paper's model description — no exceptions, regardless of whether ablation shows it is load-bearing
- Every parameter key in `data/parameters.json` (default_model and civilizations blocks) MUST be traceable to a passage, equation, or table in the paper; the CI parameter-disclosure check enforces this mechanically
- Ablation results for any documented mechanism MUST be reported alongside its description (emphasis may follow contribution; disclosure may not)
- The model section's equations MUST match `edap_model/dynamics.py` exactly; if they diverge, the code is wrong or the paper is wrong — fix before release

## Statistical Reporting (v4.1+)

- Never lead the abstract or conclusions with an in-sample (calibration-set) number; out-of-sample (leave-one-civilization-out) results are the headline empirical statistics
- Never report a turn accuracy without its persistence and majority-sign baseline comparison
- Never use the i.i.d. binomial null as the only test for turn accuracy; serial correlation must be respected (block permutation) or the weakness stated
- Every civilization's turn accuracy carries an exact binomial p-value and a Wilson 95% CI
- Point parameter estimates must be accompanied by their posterior/credible intervals once the Bayesian layer exists; "point estimate" claims without intervals are not permitted
- Brier Skill Scores must be reported for what they are (model vs climatology on the binary K-crossing task), never reframed as success when negative

## Proxy Data Construction Rules (v4.1+)

Apply to every new civilization added to `data/raw/`:

- Every indicator in `T_proxies`/`K_proxies`/`C_proxies` MUST have an entry in the file's `data_sources` naming at least one citable source
- Every data point MUST carry an `estimate_basis` note stating how the values were derived (documented value / interpolation between sourced anchors / expert estimate)
- Reuse existing indicator names and normalization weights wherever possible; a new indicator requires a matching entry in `data/normalization_params.json` (weight, min_global, max_global, description)
- Contested datings or values MUST be flagged in the file's `notes` field
- Uncertainty is graded A (sourced quantitative), B (sourced qualitative range), C (expert estimate) and recorded for the proxy-validity table
- Minimum 8 data points per civilization; anchor spacing must reflect the civilization's actual turning points, not convenient round dates

---

## Quick Reference

**Before writing any sentence about EDAP, ask:**
- Is this claim about model structure, historical alignment, or forward implication?
- Am I using the right confidence level?
- Is there a caveat I should attach?
- Would a hostile reviewer say "this is overstated"?

**If unsure, err toward:**
- Lower confidence
- More caveats
- "Suggests" over "demonstrates"
- "Framework" over "solution"

---

## Related Documents

- `ADR.md` — Architecture Decision Records (why decisions were made)
- `INDEX.md` — Project file map (for AI/RAG systems)
- `README.md` — Human-facing project overview
- `paper/sections/` — Article source, split into 8 independent sections for modular editing and section-specific guardrail checks
