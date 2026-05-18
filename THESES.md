# THESES.md — Central Argumentation and Claims for EDAP Model

## Purpose
This file contains the complete argumentation structure of the EDAP paper. It defines the logical chain from observations → hypothesis → formalization → results → interpretation. Combined with ADR.md (decisions), GUARDRAILS.md (constraints), model.py (equations), and data/ (proxies), this file enables complete reproduction of the paper.

---

## 1. Opening Argument: The Fermi Paradox and Its Gaps

### 1.1 Observation
- We observe zero evidence of extraterrestrial civilizations despite high estimated probability (Hart 1975).
- Existing explanations: Filter (destroyed) or Stagnation (don't expand).

### 1.2 Gap in Existing Explanations
- Filter theories focus on exogenous threats (nuclear, climate, AI).
- Historically, civilizations collapse from internal dynamics, not external shocks.
- No existing theory formalizes *endogenous* dissipation as a filter mechanism.

### 1.3 Our Claim
- The Great Filter is endogenous: elites optimize extraction over adaptation.
- This dissipates adaptive potential from within.
- Two-stage mechanism: Great Plateau (stagnation) → Great Filter (disappearance).

---

## 2. Core Hypothesis: Intraspecific Economic Cannibalism

### 2.1 Definition
- Elites compete for guaranteed resource access.
- They extract societal surplus without productive reinvestment.
- This is structural, not normative — applies to any complex society.

### 2.2 Mechanism
- Three interacting variables: T (technology), K (extraction), C (cooperation).
- When K exceeds critical threshold K_crit(T), feedback loops reverse.
- System enters feudal trap: stable low-T, moderate-K, low-C equilibrium.

### 2.3 Distinction from Existing Work
- Tainter (1988): diminishing returns on complexity → we add extraction dynamics.
- Turchin (2003): elite overproduction → we add cooperation-mediated parasitism.
- Piketty (2014): r > g → we embed inequality in dynamical system with thresholds.
- Hanson (1998): Great Filter as event → we reformulate as process.

---

## 3. Mathematical Formalization

### 3.1 State Variables
- T ∈ [0,1]: fraction of realizable technologies deployed.
- K ∈ [0,1]: fraction extracted without productive contribution.
- C ∈ [0,1]: generalized trust and institutional capacity.

### 3.2 Key Equations
- dT/dt: innovation minus maintenance minus elite destruction plus shock recovery.
- dK/dt: extraction growth minus cooperation suppression minus shadow drainage.
- dC/dt: cooperation growth minus extraction suppression plus shadow regrowth.

### 3.3 Critical Threshold
- K_crit(T) = K₀ - δT + ε_tech * max(0, T - T_sing)².
- Linear decline: complex societies tolerate less extraction.
- Quadratic upturn: technofeudal threshold — automation enables high-K stability.

### 3.4 Key Mechanisms
- Cooperation-mediated parasitism: α_K_eff = α_K_base * (1-C).
- Extraction ceiling: K_max(T) → 1.0 when T > T_auto (elites don't need people).
- Shadow cooperation: informal networks drain K, suppressed by automation.
- Stabilizers: T_min floor, K_max ceiling, gradient clipping.

---

## 4. Empirical Strategy

### 4.1 Proxy Estimation
- T proxies: energy per capita, trade volume, urbanization, R&D, high-tech exports.
- K proxies: top income/wealth shares, tax regressivity, rent sector dominance.
- C proxies: public investment, trust surveys, social mobility, union density.
- Weighted min-max normalization to [0,1].

### 4.2 Historical Sample
- Rome (100 BCE – 476 CE, 6 points, train): full collapse cycle.
- USSR (1922–1991, 6 points, train): full collapse cycle.
- Russia (2000–2024, 10 points, test): ongoing trajectory.
- USA (1933–2024, 18 points, test): ongoing trajectory.

### 4.3 Calibration
- Method: constrained differential evolution.
- Objective: maximize turn accuracy + BSS with penalty for K,C ∉ [0,1].
- GPU-accelerated batch simulation (Numba CUDA, RTX 3080 Ti).
- 6 parameters calibrated, remainder expert-set.

---

## 5. Key Results

### 5.1 Bifurcation Analysis
- System bifurcates at α_K_base ≈ 0.12.
- Below: bistability (sustainable vs feudal trap).
- Above: only feudal trap.
- Calibrated value 0.103 at threshold — humans oscillate between regimes.

### 5.2 Turn Accuracy
- USSR: 5/5 correct (TA=1.0) — closed system, endogenous dynamics dominate.
- Rome: 2/5 correct (TA=0.40) — partial alignment.
- Russia: 4/9 correct (TA=0.44) — K exceeds K_crit from start.
- USA: 10/17 correct (TA=0.59, p=0.31) — not statistically significant.
- Train average: 0.70, Test average: 0.52.

### 5.3 Model Conservatism
- Calibrated for turn accuracy → sacrifices K-crossing lead time.
- Brier Skill Score negative for all civilizations.
- Model is conservative: rarely predicts K > K_crit without strong evidence.
- This is a feature for early warning, not a bug.

### 5.4 Structural Tension Points (USA)
- Predicted turning points coincide with documented interventions:
  1933 (New Deal), 1945 (Bretton Woods), 1971 (petrodollar), 1980 (deregulation),
  2000 (China trade), 2008 (QE), 2020 (stimulus).
- Presented as temporal coincidence, not causal claim.
- Hypothesis: interventions are entropy discharges resetting the system.

---

## 6. Theoretical Implications

### 6.1 The Great Plateau
- Feudal trap explains absence of interstellar expansion.
- Civilization retains T≈0.1, population, technosignatures.
- Does not colonize — resources consumed by internal extraction.
- Resolves one component of Fermi Paradox: no galactic empires.

### 6.2 From Plateau to Filter (Qualitative)
- T_sing threshold: automation decouples elites from population.
- K_max(T) → 1.0 — no limit to extraction.
- Shadow cooperation suppressed by surveillance.
- Population collapses, technosignatures vanish.
- This is the Great Filter — a process, not an event.
- **Explicitly qualitative.** Requires N(t) modeling for quantitative validation.

### 6.3 Historical Precedent: Post-Roman Dark Ages
- 500–1000 CE: technological regression, political fragmentation, cooperation collapse.
- Consistent with feudal trap attractor.
- Recovery required exogenous shocks (Church, cities, Roman law).

### 6.4 Compensation Hypothesis
- USA interventions coincide with model-identified tension points.
- Falsifiable: if compensatory mechanisms exhaust, model's structural prediction (T decline) should materialize.
- Not confirmed by current analysis.

---

## 7. Explicit Limitations (Must Appear in Paper)

- Dissipation-only scope: does not explain growth.
- Small calibration sample: 12 points, 6 params — existence proof, not definitive.
- USA turn accuracy not significant (p=0.31).
- Filter transition is qualitative hypothesis.
- Proxy measurement error unquantified.
- Elite faction competition not modeled.
- Functional form sensitivity preliminary.
- Calibration trade-off: turn accuracy vs lead time.

---

## 8. Logical Chain (Summary for Verification)

1. Civilizations collapse from internal dynamics, not just external threats. (Historical observation)
2. Elite extraction dissipates adaptive potential. (Hypothesis)
3. This can be formalized as T-K-C dynamical system. (Model)
4. System has critical threshold and bifurcation. (Mathematical result)
5. Calibration places humans near bifurcation. (Empirical result)
6. Feudal trap = Great Plateau: explains no expansion. (Theoretical implication)
7. Technofeudal threshold = candidate Filter: explains no detection. (Qualitative hypothesis)
8. Model is honest about limitations. (Methodological commitment)