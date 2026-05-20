# INDEX.md — EDAP Model v4.0 Project Structure

## Overview

EDAP (Endogenous Dissipation of Adaptive Potential) is a mathematical model of civilization dynamics. It simulates how resource extraction by elites (economic cannibalism) erodes cooperation and technology, trapping societies in a feudal equilibrium. The model addresses the Fermi Paradox through the Great Plateau/Filter hypothesis.

**Version:** 4.0
**Language:** Python 3.10+ with Numba CUDA
**License:** CC-BY-4.0
**Repository:** github.com/sergemso/edap_model

---

## Directory Structure

```
edap_model/
├── .zenodo.json                         # Zenodo archive metadata
├── ADR.md                               # Architecture Decision Records & Guardrails
├── BASE_FILES.md                        # Canonical source files for AI/colleague reconstruction
├── CITATION.cff                         # Citation metadata
├── GUARDRAILS.md                        # Version-invariant argumentation constraints for paper writing
├── INDEX.md                             # This file
├── INTERFACES.md                        # System architecture, module contracts, build pipeline (C4 diagrams)
├── LICENSE                              # CC-BY-4.0
├── README.md                            # Project overview, installation, usage
├── ROADMAP.md                           # Future development plan
├── THESES.md                            # Central argumentation chain and claims for the paper
├── calibrate.py                         # Parameter calibration via differential evolution
├── requirements.txt                     # Python dependencies
├── run_model.py                         # Main entry point — full analysis pipeline
├── .github/
│   ├── workflows/
│   │   └── release.yml                  # CI pipeline: test, build, guardrail check
│   └── scripts/
│       ├── guardrail_check.py           # LangChain guardrail + consistency check
│       ├── guardrail_models.json        # model list
│       ├── guardrail_system_prompt.txt  # system prompt template
│       └── diff_metrics.py              # Compare metrics with previous release
├── paper/
│   └── article.tex                      # LaTeX paper (v4.0 final)
├── edap_model/
│   ├── __init__.py                      # Package init, re-exports key classes
│   ├── model.py                         # EDAPModel class — ODE system, simulation
│   ├── dynamics.py                      # Numba CUDA kernels + CPU fallback
│   ├── cycles.py                        # CycleManager — recovery potential, reset logic
│   ├── normalizer.py                    # Proxy → T,K,C normalization
│   ├── metrics.py                       # Turn accuracy, Brier Skill Score, lead time
│   ├── calibration.py                   # Differential evolution calibration loop
│   ├── latex_export.py                  # Auto-generates LaTeX tables from results.json
│   ├── export.py                        # results.json export
│   ├── utils.py                         # load_json, path constants
│   └── figures/
│       ├── fig1_phase.py                # Phase portrait with streamlines + trajectories
│       ├── fig2_timeseries.py           # Historical data with K_crit overlay
│       ├── fig3_bifurcation.py          # Bifurcation sweeps + phase portraits
│       ├── fig4_montecarlo.py           # GPU Monte Carlo: P(K>K_crit), P(dT/dt<0)
│       └── fig5_summary.py              # Bar chart: K vs K_crit for 14 civilizations
├── data/
│   ├── parameters.json                  # Default parameters, civilization configs, Monte Carlo settings
│   ├── normalization_params.json        # Weights, min/max for proxy → T,K,C
│   ├── summary_table.json               # Expert estimates for 14 civilizations (Figure 5)
│   └── raw/
│       ├── rome_proxies.json            # 100 BCE – 476 CE, 6 data points
│       ├── ussr_proxies.json            # 1922–1991, 6 data points
│       ├── russia_proxies.json          # 2000–2024, 10 data points
│       └── usa_proxies.json             # 1933–2024, 18 data points
├── tests/
│   └── test_model.py                    # pytest suite: 25 tests covering model, cycles, metrics, simulation
└── output/                              # All generated files (created by run_model.py)
    ├── fig1_phase_portrait.png
    ├── fig2_time_series.png
    ├── fig3_bifurcation.png
    ├── fig4_monte_carlo.png
    ├── fig5_summary_table.png
    ├── results.json                     # Full numerical results + metrics
    ├── calibration_results.json         # Output from calibrate.py
    ├── table_parameters.tex
    ├── table_turn_accuracy.tex
    ├── table_metrics.tex
    └── table_turns_*.tex                # Turning points per civilization
```

---

## File Descriptions

### Root Level

#### `run_model.py`
**Purpose:** Main entry point. Runs the full analysis pipeline.
**What it does:**
1. Loads parameters from `data/parameters.json`
2. Initializes Normalizer with `data/normalization_params.json`
3. Generates 5 figures (phase portrait, time series, bifurcation, Monte Carlo, summary)
4. Exports `output/results.json` with all metrics
5. Generates LaTeX tables from results
**Usage:** `python run_model.py`
**Dependencies:** All edap_model modules, data files.

#### `calibrate.py`
**Purpose:** Run parameter calibration via constrained differential evolution.
**What it does:**
1. Loads training data (Rome + USSR)
2. Defines objective function: maximize TA + BSS with penalty for K,C ∉ [0,1]
3. Runs differential evolution (scipy) with GPU-accelerated batch simulation
4. Saves best parameters to `output/calibration_results.json`
**Usage:** `python calibrate.py`
**Dependencies:** edap_model.calibration, edap_model.dynamics (GPU).

#### `ADR.md`
**Purpose:** Architecture Decision Records — documents all key design decisions with alternatives, rationale, and consequences. Defines argumentation guardrails for the paper (permitted/prohibited claims, required caveats).

#### `BASE_FILES.md`
**Purpose:** Defines the minimum set of 13 files from which the entire project can be reconstructed.
**What it contains:**
- List of canonical base files (THESES, ADR, GUARDRAILS, INTERFACES, INDEX, ROADMAP, 7 data files)
- Mapping from base files to generated artefacts
- Step-by-step reconstruction guide for AI or researchers
- Verification checklist
**When to use:** When an AI assistant or collaborating researcher needs to understand what the source of truth is. Also serves as a discipline mechanism: any change to the model must update the relevant base file.

#### `GUARDRAILS.md`
**Purpose:** Version-invariant argumentation constraints for any text describing the EDAP model.
**What it contains:**
- Permitted claims (by confidence level: high/mod/low)
- Prohibited claims (predictive without support, causal without evidence, overclaiming, normative)
- Required caveats (must appear in every paper)
- Tone guidelines (confidence calibration, phrase replacements)
- Section-specific requirements (Abstract, Results, Discussion, Limitations, Conclusion)
- Quick reference card for writers
**When to use:** Load before writing or editing any public communication about EDAP.
**Related:** `ADR.md` documents why specific decisions were made; `GUARDRAILS.md` constrains how they are communicated.

#### `INDEX.md`
**Purpose:** This file. Complete project map for AI assistants and RAG systems.

#### `INTERFACES.md`
**Purpose:** Defines system architecture and module contracts for the EDAP model.
**What it contains:**
- C4 context and container diagrams (Mermaid)
- Module contracts: model→dynamics, model→cycles, calibration→model, normalizer→data, figures→output
- Build pipeline: 7-step sequence from raw data to compiled paper
- Data flow diagram (ASCII)
- GPU path diagram
- Key architectural invariants
**When to use:** When reconstructing the system from base documents, or when modifying module boundaries.

#### `ROADMAP.md`
**Purpose:** Future development plan for the EDAP model.
**What it contains:**
- v5.0: Quantitative filter model (N(t), S_tech, Bayesian calibration, expanded sample)
- v6.0: Multi-elite dynamics
- v7.0: Spatial and inter-civilizational dynamics
- Ongoing: Real-time K-index monitoring
- Contribution guidelines
**When to use:** When planning future work, writing grant proposals, or onboarding new collaborators.

#### `THESES.md`
**Purpose:** Central argumentation chain for the EDAP paper.
**What it contains:**
- Opening argument: Fermi Paradox gaps
- Core hypothesis: intraspecific economic cannibalism
- Mathematical formalization summary
- Empirical strategy (proxies, sample, calibration)
- Key results (bifurcation, turn accuracy, model conservatism)
- Theoretical implications (Great Plateau, Great Filter, Dark Ages)
- Explicit limitations
- Verifiable logical chain from observation to conclusion
**When to use:** When writing or regenerating the paper from base documents.

---

### `edap_model/` Package

#### `model.py`
**Purpose:** Core EDAPModel class.
**Key class:** `EDAPModel(params)`
- Holds all parameters (30+)
- `dynamics(t, y)` — calls numba kernel or CPU fallback
- `simulate(T0, K0, C0, t_span)` — solves ODE via scipy.integrate.solve_ivp (RK45)
- `K_critical(T)` — computes K_crit with technofeudal upturn
- `set_seed(seed)` — reproducible stochastic simulations
- Manages shock history and cycle state via CycleManager

#### `dynamics.py`
**Purpose:** Numba JIT-compiled ODE right-hand side. Both CPU and GPU versions.
**Key functions:**
- `dynamics_cpu_single(...)` — @njit, single-point CPU dynamics used by solve_ivp
- `simulate_on_gpu_kernel(...)` — @cuda.jit, batch GPU simulation for Monte Carlo
- `simulate_gpu_batch(model, T0, K0, C0, n_sims, ...)` — host function that launches GPU kernel
- Exports: `dynamics_numba` (CPU), `HAS_NUMBA`, `HAS_CUDA`

#### `cycles.py`
**Purpose:** CycleManager — tracks recovery_potential, reset logic.
**Key class:** `CycleManager(params)`
- `recovery_potential` — starts at 1.0, decays on resets, replenishes during good times
- `in_reset_zone` — flag preventing repeated resets
- `apply_decay()` — multiplies recovery_potential by decay_factor
- `tuple_for_numba()` — packs cycle parameters for GPU kernel

#### `normalizer.py`
**Purpose:** Converts raw proxy data to normalized T, K, C values.
**Key class:** `Normalizer(params_path)`
- Loads normalization parameters (weights, min/max) from JSON
- `normalize(raw_point)` → (T, K, C)
- `normalize_civilization(civ_data)` → list of dicts with year, T, K, C, label

#### `metrics.py`
**Purpose:** Computes model evaluation metrics.
**Key functions:**
- `compute_turn_accuracy(norm_data, sim_result, sim_years)` → accuracy, details
- `compute_k_crossing_lead(norm_data, sim_result, sim_years, model)` → lead_years
- `compute_brier_skill_score(norm_data, sim_result, sim_years, model)` → BSS
- `compute_all_working_metrics(...)` → dict with all three

#### `calibration.py`
**Purpose:** Differential evolution calibration loop.
**Key function:** `calibrate_all(normalizer, model_params, seed)`
- Preloads training data
- Defines objective with penalty for out-of-bounds K,C
- Runs scipy.optimize.differential_evolution
- Returns best parameters + train/test evaluation

#### `latex_export.py`
**Purpose:** Generates LaTeX tables from results.json.
**Key function:** `generate_latex_tables(results_json_path, output_dir)`
- Table 1: All parameters (auto-split into two halves if needed)
- Table 2: Turn accuracy by civilization
- Table 3: Working metrics (TA, BSS, lead time, final T)
- Table 4: Turning points for each civilization (auto-detected from data)

#### `export.py`
**Purpose:** Exports results.json with all metrics.
**Key function:** `export_results_json(model_params, normalizer, filepath)`
- Simulates each civilization
- Computes working metrics
- Adds compensation analysis (USA, Russia)
- Writes JSON

#### `utils.py`
**Purpose:** Utilities.
- `load_json(filename)` — loads JSON with error handling
- Path constants: `DATA_DIR`, `RAW_DIR`, `OUTPUT_DIR`

#### `figures/fig1_phase.py`
**Purpose:** Figure 1 — Phase portrait.
- Computes vector field on (T,K) grid at C=0.25
- Plots streamlines, K_crit(T) line, K_max(T) line
- Overlays historical civilization trajectories from simulation

#### `figures/fig2_timeseries.py`
**Purpose:** Figure 2 — Time series of T, K, C with K_crit overlay.
- Reads proxy files from data/raw/
- Plots actual data as solid lines
- Overlays K_crit(T) as dash-dot line
- Shades regions where K > K_crit

#### `figures/fig3_bifurcation.py`
**Purpose:** Figure 3 — Bifurcation analysis.
- Panel A: α_K_base sweep
- Panel B: δ sweep
- Panel C: Phase portraits at three α_K values (streamlines)
- Panel D: Basin stability heatmap

#### `figures/fig4_montecarlo.py`
**Purpose:** Figure 4 — GPU Monte Carlo forecast.
- Runs n_simulations via GPU batch
- Panel 1: P(K > K_crit) over time with 90% CI
- Panel 2: P(dT/dt < 0) over time with 90% CI
- Separate rows for modern civilizations

#### `figures/fig5_summary.py`
**Purpose:** Figure 5 — Civilizational risk assessment bar chart.
- Loads summary_table.json (14 civilizations)
- Plots actual K vs K_crit at peak T
- Red bars: K > K_crit, Green: K < K_crit
- Warning symbols for current exceedance

---

### `data/` Directory

#### `parameters.json`
**Purpose:** Central configuration file.
**Structure:**
- `default_model`: all 30+ parameters with values
- `civilizations`: Rome, USSR, Russia, USA — each with T0, K0, C0, t_span, color, label, shocks, historical_T_peak, decay_factor
- `monte_carlo`: n_simulations, t_forecast, start_year
- `bifurcation`: parameter ranges for sweeps

#### `normalization_params.json`
**Purpose:** Proxy normalization configuration.
**Structure:**
- `T_normalization`: proxies (energy_per_capita_GJ, trade_volume_index, ...), each with weight, min_global, max_global
- `K_normalization`: proxies (top_1_percent_income_share, tax_regressivity_index, ...)
- `C_normalization`: proxies (public_infrastructure_share, generalized_trust_proxy, ...)

#### `summary_table.json`
**Purpose:** Expert estimates for 14 civilizations (Shumer through modern).
**Structure:** Array of objects with name, period, T_peak, K_peak, K_crit_at_T_peak, K_exceeded, outcome, expert_estimate flag.

#### `raw/*.json`
**Purpose:** Historical proxy data for each civilization.
**Structure per file:**
- `civilization`: name string
- `period`: {start, end}
- `data_points`: array of {year, label, T_proxies: {...}, K_proxies: {...}, C_proxies: {...}}
- `data_sources`: object mapping each proxy to its literature source
- `notes`: historical context and mechanisms (for Russia and USA)

---

### `tests/` Directory

#### `test_model.py`
**Purpose:** pytest test suite (25 tests).
**Test classes:**
- `TestModelInstantiation`: default construction, custom params, cycle manager creation
- `TestCycleManager`: initial state, reset, apply_decay, decay floor, recovery replenishment
- `TestKCritical`: linear at low T, bend above singularity
- `TestAlphaKInhibition`: high C inhibits K, low C allows K
- `TestTechnologyDynamics`: T grows in good conditions, declines when K > K_crit, eta boosts recovery, T stays in bounds
- `TestShadowCooperation`: shadow drain at high K
- `TestRecoveryDynamics`: recovery potential in results, bounded
- `TestSimulation`: correct structure, reproducibility with seed, C never negative, bounds
- `TestAttractors`: feudal trap exists, sustainable with low alpha_K, bistability

---

### `.github/` Directory

#### `workflows/release.yml`
**Purpose:** CI pipeline triggered on version tags (`v*`).
**What it does:**
1. Bumps version in article.tex, CITATION.cff, .zenodo.json, model.py
2. Runs test suite
3. Runs full model pipeline
4. Compiles paper PDF
5. Checks for placeholder values
6. Runs guardrail and consistency check via LangChain + Groq
7. Diffs metrics against previous release
8. Creates draft GitHub Release with artifacts
**When it runs:** On push of version tag (`v4.0.0`, `v4.1.0`, etc.)

#### `scripts/guardrail_check.py`
**Purpose:** Validates article against GUARDRAILS.md and results.json for consistency.
**Features:**
- LangChain agent with Groq (free tier)
- Model fallback on consecutive errors (3 models)
- Retry on rate limit (429) with exponential backoff
- Temperature=0 for deterministic output
- Strict: any violation = fail
- Checks both guardrail compliance AND numerical consistency
**Usage:** `python .github/scripts/guardrail_check.py article.tex GUARDRAILS.md results.json`

#### `scripts/diff_metrics.py`
**Purpose:** Compares current results.json with previous release to detect metric drift.
**What it does:**
- Extracts results.json from previous git tag
- Computes delta for turn accuracy and BSS per civilization
- Reports significant changes (>0.001)
**Usage:** `python .github/scripts/diff_metrics.py output/results.json`

---

### `paper/` Directory

#### `article.tex`
**Purpose:** LaTeX source for the academic paper (v4.0 final).
**Compilation:**
```bash
pdflatex article.tex
bibtex article
pdflatex article.tex
pdflatex article.tex
```
**Structure:**
- Abstract
- Introduction (Fermi Paradox, endogenous filters, intraspecific cannibalism, plateau→filter)
- Model Description (state variables, ODE system, key mechanisms, parameters)
- Data & Calibration (proxy variables, historical civilizations, calibration procedure)
- Results (turn accuracy, structural tension detection, working metrics, phase portrait, bifurcation, Monte Carlo, risk assessment)
- Discussion (dissipation theory, Great Plateau, Dark Ages, filter threshold, compensation hypothesis, limitations)
- Conclusion
- Appendix (parameter table)
- Bibliography (24 references)

---


## Reproducibility

This project is designed for full reproducibility. The canonical source of truth is defined in [BASE_FILES.md](BASE_FILES.md) — 13 files from which all code, figures, tables, and the article itself can be reconstructed.

To verify reproducibility:
```bash
python run_model.py
python -m pytest tests/test_model.py -v
```

For AI-assisted reconstruction, see [BASE_FILES.md](BASE_FILES.md).

---

## Data Flow

```
data/raw/*.json
    │
    ▼
Normalizer.normalize_civilization()
    │
    ▼
EDAPModel.simulate(T0, K0, C0)
    │
    ▼
metrics.compute_all_working_metrics()
    │
    ├──► output/results.json
    │         │
    │         ▼
    │    latex_export.generate_latex_tables()
    │         │
    │         ▼
    │    output/table_*.tex ──► article.tex ──► article.pdf
    │
    └──► figures/fig1-5_*.py
              │
              ▼
         output/fig*.png ──► article.tex ──► article.pdf
```

---

## Key Equations (for quick reference)

```
dT/dt = α_T (T-T_min)(1-T) C (1-K/K_crit) - β_T (T-T_min) - γ_T K T + η S (1-T/T_peak)

dK/dt = α_K_eff K (1-T-C) (1-K/K_max) - β_K (T C) K - λ_eff max(K-K_shadow, 0) K

dC/dt = α_C (T-K) C (1-C) + η S (1-C) + μ_eff max(K-K_shadow, 0) (1-C) - γ_C K C

K_crit(T) = K₀ - δ T + ε_tech max(0, T-T_sing)²

α_K_eff = α_K_base (1 - C)

K_max(T) = { κ T/(T+T_h)                    if T < T_auto
           { smooth transition to 1.0       if T ≥ T_auto
```

---

## Parameters Quick Reference

| Symbol | Name | v4.0 Value | Meaning |
|--------|------|-----------|---------|
| α_T | alpha_T | 0.52 | Innovation speed under ideal conditions |
| β_T | beta_T | 0.167 | Complexity maintenance cost |
| γ_T | gamma_T | 0.078 | Direct technology destruction by elites |
| T_min | T_min | 0.08 | Technological floor |
| α_K_base | alpha_K_base | 0.103 | Base attractiveness of parasitism |
| β_K | beta_K | 0.18 | Suppression of K by cooperation |
| κ | K_max_coeff | 0.95 | Extraction ceiling coefficient |
| T_h | K_half | 0.05 | Half-saturation for K_max |
| T_auto | T_automation | 0.85 | Automation threshold |
| α_C | alpha_C | 0.31 | Cooperation growth rate |
| γ_C | gamma_C | 0.20 | Cooperation suppression by K |
| η | eta | 0.781 | External shock impact |
| K₀ | K0 | 0.55 | Base K_crit at T=0 |
| δ | delta | 0.25 | K_crit sensitivity to T |
| σ | sigma | 0.12 | Shock volatility |
| T_sing | T_singularity | 0.65 | Technofeudal threshold |
| ε_tech | epsilon_tech | 0.40 | Technofeudal amplification |
| K_shadow | K_shadow_threshold | 0.50 | Shadow activation threshold |
| λ_shadow | lambda_shadow | 0.050 | Shadow economy drain |
| μ_shadow | mu_shadow | 0.382 | Shadow cooperation growth |
| T_peak | historical_T_peak | varies | Historical maximum T per civilization |
| decay_factor | decay_factor | varies | Recovery depletion per cycle |

---

## Target Journals

- *Journal of Artificial Societies and Social Simulation* (JASSS) — primary
- *International Journal of Astrobiology* — alternative
- *Physica A* — if emphasis on statistical mechanics

---

## Contact

Sergey A. Strebulaev
ORCID: 0009-0006-2710-8910
Email: strebulaev@gmail.com