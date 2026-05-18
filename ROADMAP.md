# ROADMAP.md — EDAP Model Development Plan

## Current Version: v4.0 (May 2026)

### What v4.0 Delivers
- Three-variable ODE system (T, K, C) with 30+ parameters
- GPU-accelerated simulation and calibration (Numba CUDA)
- Calibration on Rome + USSR, evaluation on Russia + USA
- Turn accuracy: 0.70 (train), 0.52 (test)
- Qualitative Great Plateau/Filter hypothesis
- Full paper with 5 figures, 4 tables, 24 references

---

## v5.0: Quantitative Filter Model (Target: Q4 2026)

### Core Extensions

**1. Population dynamics N(t)**
- New equation: dN/dt = r(K, C, shadow_growth) * N * (1 - N/N_max(T)) - δ_K * K * N
- Three reproductive regimes: clan (high shadow → high fertility), atomized (low shadow → low fertility), civilizational (high C → moderate fertility)
- Enables quantitative distinction between plateau (N stable) and filter (N → 0)

**2. Technosignature variable S_tech(t)**
- S_tech = f(N, T, K) — detectable signal strength
- Threshold S_min for astronomical detection (JWST, future telescopes)
- Filter defined as S_tech < S_min for cosmologically significant duration

**3. Bayesian calibration**
- Replace differential evolution with MCMC (emcee or PyMC)
- Posterior distributions for all parameters
- Predictive intervals for N(t), S_tech(t)
- Proper uncertainty quantification

**4. Expanded historical sample**
- Byzantium (500–1453 CE): 8-10 data points
- Dynastic China (Tang/Song/Ming): 5-7 points per dynasty
- Japan (1868–2024): 8-10 points
- Enables cross-validation and reduces overfitting

**5. Proxy validation**
- Correlation analysis against V-Dem, Polity IV, Economic Freedom Index
- Face validity documentation for ancient proxies
- Sensitivity analysis to proxy weights

### Expected Results (v5.0)
- Quantitative P(filter) for USA, China, Russia, EU
- Galactic ensemble: distribution of civilizational outcomes
- Forward predictions with uncertainty
- Falsifiable filter hypothesis

### Target Journal
- *Astrobiology* or *International Journal of Astrobiology*

---

## v6.0: Multi-Elite Dynamics (Target: 2027)

### Extensions
- Elite factions with replicator dynamics
- Coalition formation and dissolution
- Institutional constraints on extraction
- Agent-based validation

### Expected Contribution
- Bridge between EDAP and cliodynamics (Turchin)
- Explain why some societies restrain K while others don't
- Policy-relevant: what institutional designs prevent the feudal trap?

---

## v7.0: Spatial and Inter-Civilizational Dynamics (Target: 2028)

### Extensions
- Spatial diffusion of T, K, C across regions
- Inter-civilizational competition and trade
- Multi-planet extension for interstellar colonization scenarios

### Expected Contribution
- Full Fermi Paradox resolution with quantitative predictions
- Testable against exoplanet atmospheric data

---

## Ongoing: Real-Time K-Index Monitoring

### Concept
- Operationalize K as a real-time index from publicly available data
- Track K/K_crit ratio for major nations
- Open-source dashboard
- Early warning: "Country X approaching feudal trap threshold"

### Status
- Conceptual stage
- Requires stable model parameters and validated proxies
- Target: v5.0 launch alongside paper

---

## Contributing

See `BASE_FILES.md` for the canonical source files from which the project can be reconstructed. All contributions should update the relevant base files, not just generated code.

Pull requests should include:
- Updates to affected ADR entries
- Updates to THESES if argumentation changes
- Passing test suite
- Passing guardrail check
