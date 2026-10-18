# Plan: EDAP v4.1 — Hardening + Selective v5.0 Pull-Forward

## Why this plan exists (read this first, cold session)

EDAP v4.0 (released May 2026, Zenodo DOI 10.5281/zenodo.20298963) is a 3-variable ODE model (T = technology, K = elite extraction, C = cooperation) explaining civilizational stagnation ("feudal trap") and, qualitatively, the Fermi Paradox. A full audit on 2026-10-18 found integrity defects and statistical weaknesses that must be fixed before submitting v4.1 to the *International Journal of Astrobiology* (decided in ADR-022):

1. **The paper does not describe the model that is simulated.** `paper/sections/model.tex` presents a deterministic 3-variable ODE, but `edap_model/dynamics.py` additionally simulates a hidden 4th state (`recovery_potential`), a stochastic reset mechanism (`p_reset` = product of two sigmoids gated by an RNG), jump overrides of dT/dK/dC, and a per-civilization `historical_T_peak` recovery scaling. None of this is in the paper (ADR-023 resolves: ablate, then document).
2. **GPU and CPU dynamics paths diverge** (CPU: `in_reset_zone` latching, `recovery_potential += d_rp*0.1`; GPU: no latching, `d_rp*0.05`, `recovery_potential *= 0.7` after reset), violating the parity guardrail in `GUARDRAILS.md` (ADR-024).
3. **Statistics are fragile**: test turn accuracy averages 0.516 (≈ chance); the 0.70 headline is in-sample; the null (i.i.d. binomial, p=0.5) ignores serial correlation; no baselines, no CIs, no cross-validation; calibrated `alpha_K_base=0.103` sits at the lower bound of its (0.10, 0.90) calibration box (ADR-025).
4. **Missing prior art**: Haqq-Misra & Baum (2009) is the "Great Plateau" idea published in astrobiology; Ćirković, Landis, Kinouchi absent; no universality argument for why alien civilizations would share elite-extraction dynamics (ADR-029).

Decided scope (ADR-022…029): fix all of the above; add Byzantium + Japan (load-bearing) and Tang/Song China + Britain (stretch) civilizations; add Bayesian (ABC-style) calibration; add a proxy validation study; keep N(t) deferred; keep the term "intraspecific economic cannibalism" with an added defense paragraph; target venue *International Journal of Astrobiology*.

Key repo facts for a cold session:
- Entry point: `python run_model.py` → `output/results.json`, `output/fig*.png`, `output/table_*.tex`.
- Calibration: `python calibrate.py` / `edap_model/calibration.py` (differential evolution, 6 free params: alpha_K_base, beta_T, gamma_T, lambda_shadow, mu_shadow, eta).
- Current civilizations: `data/raw/{rome,ussr,russia,usa}_proxies.json` (train: Rome+USSR; test: Russia+USA). Train TA 0.70 (in-sample), test 0.516.
- Paper: `paper/article.tex` inputs `paper/sections/{abstract,introduction,model,results,discussion,conclusion,appendix,bibliography}.tex`. Compile: `cd paper/ && pdflatex article.tex` ×3.
- Tests: `python -m pytest tests/test_model.py -v`. Guardrail check: `python .github/scripts/guardrail_check.py abstract paper/sections/ GUARDRAILS.md output/results.json`.
- Knowledge files (root): ADR.md, GUARDRAILS.md, THESES.md, ROADMAP.md, INDEX.md, BASE_FILES.md. Any change to equations must update CPU and GPU paths in `edap_model/dynamics.py` together, plus both base-file docs.

Legend: `[ ]` pending · `[~]` in progress · `[x]` done · `[!]` blocked · `[>]` moved to another plan · `‖` after the marker = can run in parallel with its siblings.

---

### 0.1 Unify CPU/GPU dynamics semantics + parity test — [ ]
Class: implement · Needs: none · Check: `python -m pytest tests/test_parity.py -v`
Context: GUARDRAILS.md requires the GPU path to produce results identical to the CPU path within numerical tolerance. Today it does not: `dynamics_cpu_single` (edap_model/dynamics.py:7-136) uses an `in_reset_zone` latch and `recovery_potential += d_rp * 0.1`; `simulate_on_gpu_kernel` (dynamics.py:153-291) has no latch, uses `d_rp * 0.05`, and multiplies `recovery_potential *= 0.7` after a reset. Published metrics come from the CPU path (`EDAPModel.simulate`), Monte Carlo/calibration from the GPU path — so the two regimes are currently different models.
Do: Pick one semantics (recommended: the CPU path's, since published metrics and `solve_ivp` single-trajectory runs use it) and make both paths implement it exactly. Both paths already share the same LCG RNG (`rng_state*1103515245+12345 mod 2^31`), so with unified math and identical seeds the trajectories must match step-for-step at dt=0.05. Create `tests/test_parity.py`: run N≥20 batch simulations on both paths with identical seeds and parameters, assert max |T_gpu−T_cpu|, |K_gpu−K_cpu|, |C_gpu−C_cpu| < 1e-8; include the no-CUDA skip decorator so CI (CPU-only) still exercises the CPU side.
Done-when: `tests/test_parity.py` passes on a CUDA machine and is skipped-but-collected on CPU-only machines; a code comment in dynamics.py names the chosen canonical semantics.

### 0.2 Ablation switches (reset mechanism, shock mode) — [ ] ‖
Class: implement · Needs: none · Check: `python -m pytest tests/test_model.py tests/test_parity.py -v`
Context: ADR-023 requires ablating (a) the reset/recovery mechanism and (b) the shock term before documenting them in the paper. There is currently no way to turn either off.
Do: Thread two options through `EDAPModel` (edap_model/model.py) into both dynamics paths (edap_model/dynamics.py): `use_reset: bool` (default True — when False, skip the `p_reset` block and the `recovery_potential` update entirely) and `shock_mode: "historical" | "ou" | "none"` (default "historical" = current behavior: OU process plus discrete hand-dated events; "ou" = OU only; "none" = S(t)=0). Defaults must preserve current outputs bit-for-bit.
Done-when: with default flags `run_model.py` regenerates `output/results.json` identical to the committed v4.0 values; with `use_reset=False` and `shock_mode="none"` the trajectories demonstrably change (spot-check one civ).

### 0.3 Run ablation grid and record what the hidden mechanism contributes — [ ]
Class: implement · Needs: 0.1, 0.2 · Check: `python -c "import json; r=json.load(open('output/ablation_results.json')); assert len(r['configs'])==6"`
Context: Before writing the model-section documentation (5.2) we must know whether the reset/recovery mechanism and hand-dated shocks are load-bearing for the published fit, or negligible. This is the evidence base ADR-023 calls for.
Do: Add a script `run_ablations.py` (project root, modeled on `run_model.py`): for each of {reset on, off} × {shock historical, ou, none} (6 configs) × all civilizations currently in `data/raw/`, simulate with the committed calibrated parameters (data/parameters.json `default_model`), and record turn accuracy, BSS, final (T,K,C). Write `output/ablation_results.json` and `output/table_ablation.tex` (via a new template in `edap_model/latex_export.py`).
Done-when: `output/ablation_results.json` holds 6 configs × 4 civs with all three metrics; `table_ablation.tex` compiles inside `paper/sections/results.tex` context (test with a scratch `\input`); a short written interpretation (3–5 sentences) is appended to the JSON as `interpretation` stating whether the reset mechanism materially changes turn accuracy (threshold: ΔTA ≥ 0.1 on any civilization = "load-bearing").

### 1.1 Persistence/trend baselines + block-permutation null — [ ]
Class: implement · Needs: none · Check: `python -m pytest tests/test_metrics.py -v`
Context: ADR-025: the current binomial-vs-0.5 null assumes i.i.d. signs, but T-series are strongly trending (USA T_trend "increasing"), so an "always up" predictor could match the reported 0.588. Every TA number must be reported against persistence and trend baselines with a serial-correlation-robust test.
Do: In `edap_model/metrics.py` add: `compute_persistence_accuracy(norm_data)` (predicted sign = previous observed sign), `compute_majority_accuracy(norm_data)` (predicted sign = majority sign of the data), and `block_permutation_test(norm_data, sim_result, sim_years, n_perm=2000, block=2)` (circularly permute the observed sign sequence in blocks to build the null distribution of match rate; returns p-value). Wire all three into `compute_all_working_metrics`. Create `tests/test_metrics.py` with hand-checkable cases (e.g., a monotone series where persistence scores 1.0).
Done-when: `compute_all_working_metrics` output contains `persistence_accuracy`, `majority_accuracy`, `perm_p_vs_model` for each civilization; tests pass.

### 1.2 Per-civilization exact tests + Wilson CIs — [ ] ‖
Class: implement · Needs: none · Check: `python -m pytest tests/test_metrics.py -v`
Context: ADR-025/ADR-007: currently only the USA gets a binomial test; no metric carries a confidence interval. With ~5–17 turns per civilization, Wilson intervals are the honest representation.
Do: In `edap_model/metrics.py` add `exact_binomial_p(n_correct, n_turns, p_null=0.5)` (scipy.stats.binomtest) and `wilson_ci(n_correct, n_turns, z=1.96)`. Evaluate p both against 0.5 and against the persistence baseline's accuracy (the stricter, fairer null). Include in `compute_all_working_metrics`.
Done-when: Every civilization row in `output/results.json` carries `binom_p`, `binom_p_vs_persistence`, `wilson_ci_low`, `wilson_ci_high`.

### 1.3 Bifurcation sensitivity + widened-bound recalibration — [ ]
Class: implement · Needs: 0.2 · Check: `python -c "import json; s=json.load(open('output/bifurcation_sensitivity.json')); assert s['threshold_range'] and s['recalibrated']['alpha_K_base'] > 0.02"`
Context: ADR-025: the claim "calibrated α_K_base = 0.103 sits precisely at the bifurcation point ≈0.12" is vulnerable twice over — the calibration box was (0.10, 0.90), so the optimizer was pressed against the boundary, and the 0.12 threshold's dependence on the other 25+ parameters is unknown.
Do: (a) Extend the bifurcation routine (edap_model/figures/fig3_bifurcation.py, reuse its attractor-identification core into a callable that returns the bifurcation threshold in α_K_base for a given parameter vector). Sweep each of beta_T, gamma_T, lambda_shadow, mu_shadow, eta ±20% around calibrated values; record how the threshold moves. Write `output/bifurcation_sensitivity.json`. (b) In `edap_model/calibration.py` widen `alpha_K_base` bounds to (0.01, 0.90) and re-run `python calibrate.py`; save recalibrated parameters to `output/recalibrated_params.json`.
Done-when: `bifurcation_sensitivity.json` contains threshold values for ≥5 perturbation axes plus the nominal; `recalibrated_params.json` exists with `alpha_K_base` strictly inside bounds. If the recalibrated value drifts below 0.05 or the threshold range spans [0.08, 0.20], flag in the JSON (`verdict: "near-bifurcation claim NOT robust"`) so step 5.8 rewrites that passage; otherwise `verdict: "robust"`.

### 1.4 Regenerate results and tables with the new statistics — [ ]
Class: implement · Needs: 1.1, 1.2, 1.3, 0.1 · Check: `python run_model.py && python -c "import json; r=json.load(open('output/results.json')); assert 'persistence_accuracy' in r['civilizations']['United States of America']['working_metrics']"`
Context: The paper's auto-generated tables (output/table_*.tex ← `edap_model/latex_export.py` ← results.json) must carry the new statistics; THESES.md §5.2/5.3 and every prose number in the paper depend on them.
Do: Extend `edap_model/latex_export.py` so `table_turn_accuracy.tex` gains columns for persistence baseline, permutation p, and Wilson CI, and `table_metrics.tex` gains the ablation and sensitivity references. Re-run `python run_model.py` (which regenerates results.json and all tables).
Done-when: Regenerated tables compile in the paper (scratch-compile `paper/article.tex`); no metric is reported anywhere in `output/` without its baseline and interval.

### 2.1 Byzantium proxy series (load-bearing) — [ ] ‖
Class: implement · Effort: high · Needs: none · Check: `python -c "import json; d=json.load(open('data/raw/byzantium_proxies.json')); assert len(d['data_points'])>=8 and len(d['data_sources'])>=10 and all(p.get('estimate_basis') for p in d['data_points'])"`
Context: ADR-022/025: the sample must grow for leave-one-out CV to mean anything. Byzantium (500–1453) adds a second full collapse-and-recovery arc with a documented fiscal-institutional history. Follow the Proxy Data Construction Rules in GUARDRAILS.md (every indicator has `data_sources`; every point has an `estimate_basis` note; 8–10 points).
Do: Create `data/raw/byzantium_proxies.json` matching the exact schema of `data/raw/rome_proxies.json` (civilization, period, data_points[{year,label,T_proxies,K_proxies,C_proxies,estimate_basis}], data_sources, notes, shocks). Suggested anchor points: 500 (Anastasian fiscal strength), 565 (Justinian peak), 650 (seventh-century crisis), 775 (Isaurian recovery), 850 (Macedonian rise), 1025 (Basil II apex), 1071 (Manzikert), 1180 (Komnenian restoration), 1282 (Palaiologan decline), 1453 (fall). Sources: Treadgold (1997) *A History of the Byzantine State and Society*; Haldon (2016) *The Empire That Would Not Die*; Laiou & Morrisson (2007) *The Byzantine Economy*; Whittow (1996); Cheynet (2006) on aristocratic land concentration. Reuse existing indicator names where possible (energy_per_capita_GJ, urbanization_rate, trade_volume_index; elite wealth-share and rent-sector indicators adapted from Rome's set) so normalization works unchanged; if a new indicator is unavoidable, add it to `data/normalization_params.json` with weight and bounds per the existing format.
Done-when: Schema check above passes; `Normalizer.normalize_civilization` runs on the file without error and produces T, K, C in [0,1] with a plausible trajectory (peak T mid-Macedonian, terminal decline).

### 2.2 Japan proxy series (load-bearing) — [ ] ‖
Class: implement · Effort: high · Needs: none · Check: `python -c "import json; d=json.load(open('data/raw/japan_proxies.json')); assert len(d['data_points'])>=8 and all(p.get('estimate_basis') for p in d['data_points'])"`
Context: ADR-022/025: Japan 1868–2024 is the differential-outcome case — elite coordination that repeatedly restrained extraction (Meiji state-building, postwar compact) with sustained high C. It gives the model a genuine chance to be falsified, which is scientifically the most valuable addition.
Do: Create `data/raw/japan_proxies.json` (same schema and rules as 2.1). Suggested points: 1868 (Meiji restoration), 1885 (land-tax consolidation), 1905 (Russo-Japanese war mobilization), 1920 (Taishō liberalism/zaibatsu concentration), 1945 (defeat/trough), 1960 (high-growth onset), 1975 (income-doubling consolidated), 1990 (bubble peak), 2005 (post-bubble stagnation), 2020 (Abenomics era). Sources: Ohkawa & Rosovsky (1973) *Japanese Economic Growth*; WID.world Japan series for top shares; Gordon (2003) *A Modern History of Japan*; V-Dem/JIPDEC as relevant; OECD for R&D and energy. Use the modern proxy set already defined for USA/Russia where possible.
Done-when: Schema check passes; normalized trajectory shows the postwar C-peak and K-trough that motivate choosing Japan.

### 2.3 Normalization extension + weight/subset sensitivity — [ ]
Class: implement · Needs: 2.1, 2.2 · Check: `python -c "import json; s=json.load(open('output/normalization_sensitivity.json')); assert set(s.keys()) >= {'weight_perturbation','drop_one_proxy'}"`
Context: ADR-027: Rome lacks R&D/high-tech indicators, USA has them — so T for different civilizations is a differently-composed weighted index, and ADR-017's "directly comparable" claim needs quantification, not assertion.
Do: Write `run_normalization_sensitivity.py`: (a) perturb every proxy weight in `data/normalization_params.json` by ±30% (one axis at a time) and record max |ΔT|, |ΔK|, |ΔC| across all points of all civilizations; (b) drop-one-proxy-out per component per civilization and record the same. Write `output/normalization_sensitivity.json`. Add the resulting comparability caveat sentence (with numbers) to the plan for 5.1 Methods.
Done-when: JSON exists with both blocks; the worst-case |Δ| per component across all perturbations is computed (this number goes into the Methods caveat).

### 2.4 Leave-one-civilization-out cross-validation — [ ]
Class: implement · Needs: 2.1, 2.2, 1.1, 0.1 · Check: `python -c "import json; r=json.load(open('output/loocv_results.json')); assert len(r['folds'])==6 and all('heldout_turn_accuracy' in f for f in r['folds'].values())"`
Context: ADR-025: with 6 civilizations, LOO-CV replaces the arbitrary 2-train/2-test split and becomes the paper's headline empirical number (GUARDRAILS: no in-sample number leads the abstract).
Do: Add `loocv()` to `edap_model/calibration.py` (or a root script `run_loocv.py`): for each of the 6 civilizations, calibrate the 6 free parameters on the other 5 (same DE protocol/bounds as `calibrate.py`, widened α_K bounds per 1.3), then evaluate held-out turn accuracy vs persistence and majority baselines. Aggregate: mean held-out TA, Wilson CI over folds, count of folds beating persistence.
Done-when: `output/loocv_results.json` has all 6 folds with per-fold params, held-out TA, baselines, and the aggregate; a `table_loocv.tex` is generated for the paper.

### 2.5 Stretch: Tang/Song China proxy series — [ ] ‖
Class: implement · Effort: high · Needs: none · Check: `python -c "import json; d=json.load(open('data/raw/china_proxies.json')); assert len(d['data_points'])>=10"`
Context: ADR-022: third-priority civilization (run only if 2.1–2.4 land on schedule). Longest continuous bureaucratic record; proxy dates contested — every point must carry its `estimate_basis` and the `notes` field must state which dates are contested.
Do: Create `data/raw/china_proxies.json` covering 618–1279 (Tang through Song), ~5 points per dynasty: 618, 755 (An Lushan), 800, 900, 960 (Song founding), 1000, 1100 (Song economic revolution), 1141 (Jin-Song settlement), 1200, 1279 (Mongol conquest). Sources: Hartwell (1982) on demography and iron production; Shiba (1970) on Song commerce; Ebrey (1999); Qi Xia on Song fiscal history; Maddison-type population/GDP reconstructions. Follow the GUARDRAILS construction rules exactly.
Done-when: Schema passes; `notes` documents contested dating; normalizer runs clean.

### 2.6 Stretch: Britain 1688–2024 proxy series — [ ] ‖
Class: implement · Needs: none · Check: `python -c "import json; d=json.load(open('data/raw/britain_proxies.json')); assert len(d['data_points'])>=10"`
Context: ADR-022: fourth-priority civilization; a second "no-trap" differential case (aristocratic decline, institutional restraints on extraction) — least added outcome-variance but the easiest data.
Do: Create `data/raw/britain_proxies.json`. Suggested points: 1688 (Glorious Revolution), 1710, 1780 (Industrial Revolution onset), 1815 (post-Napoleonic peak inequality), 1867 (Second Reform Act), 1910 (land-aristocracy terminal decline), 1926 (general strike/trough), 1950 (postwar settlement), 1980 (Thatcher deregulation), 2008 (financial crisis), 2020. Sources: Lindert (1986) on land shares; Piketty (2014) UK series; WID.world; Bank of England historical statistics; Lindert & Williamson on living standards.
Done-when: Schema passes; normalizer runs clean.

### 2.7 Re-run LOO-CV on the full sample — [ ]
Class: implement · Needs: 2.4, 2.5, 2.6 · Check: `python -c "import json; r=json.load(open('output/loocv_results.json')); assert len(r['folds'])==8"`
Context: If the stretch civilizations land, the headline LOO number must cover all 8; if they do not land, this step is closed as `[x]` with a note and 2.4's 6-fold result stands.
Do: Re-run the LOO machinery from 2.4 with all civilizations present in `data/raw/`.
Done-when: `output/loocv_results.json` reflects the full sample; `table_loocv.tex` regenerated.

### 3.1 ABC-style Bayesian posterior implementation — [ ]
Class: implement · Needs: 0.2 · Check: `python -m pytest tests/test_bayes.py -v`
Context: ADR-026: differential evolution gives point estimates only; every parameter and metric needs uncertainty. Full-likelihood MCMC is intractable for a stochastic-reset ODE system, so an ABC-style pseudo-likelihood on turn-pattern distance using the existing Numba batch simulator is the pragmatic route.
Do: Create `edap_model/bayes.py`: emcee ensemble sampler (add `emcee` to `requirements.txt`); parameters = the same 6 free params with uniform priors = the widened DE bounds; pseudo-likelihood `log L = -d(params)²/(2ε²)` where `d = mean over training civilizations of (1 − turn accuracy)` computed on the existing batch simulation path (GPU if available, CPU fallback); ε chosen as the 10th percentile of d over a 200-draw prior sample. ~32 walkers, ≥2000 steps, discard 500 as burn-in. Save chain to `output/posterior_samples.npz`. Create `tests/test_bayes.py` with a 50-step smoke run on synthetic short series.
Done-when: smoke test passes; a full run completes and Gelman-Rubin r̂ < 1.1 for all 6 params (report in 3.2 output).

### 3.2 Posterior summaries + predictive intervals — [ ]
Class: implement · Needs: 3.1, 2.4 · Check: `python -c "import json; r=json.load(open('output/results.json')); assert 'bayesian' in r"`
Context: The paper reports credible intervals on parameters and posterior-predictive distributions of held-out turn accuracy — this is what upgrades "existence proof" language in THESES.md §7.
Do: From the chain in `output/posterior_samples.npz`: posterior medians and 90% CrI for the 6 parameters; posterior-predictive TA distribution per civilization (simulate 200 posterior draws per civilization); probability that held-out TA exceeds the persistence baseline (ties into LOO folds where the training set matches). Extend `output/results.json` with a `bayesian` block and generate `table_posterior.tex`.
Done-when: `results.json` bayesian block present; `table_posterior.tex` compiles; r̂ values recorded.

### 3.3 Identifiability review — [ ]
Class: review · Needs: 3.2 · Check: `python -c "import json; s=json.load(open('output/posterior_samples.npz'), allow_pickle=False)" is None or True`
Context: 6 parameters on ≤10 informative turns per fold may leave some parameters unidentified (posterior ≈ prior). ADR-026 requires knowing this before claiming parameter estimates mean anything.
Do: Compare each marginal posterior to its uniform prior (K-S test); compute Spearman rank correlations among posterior parameter pairs. Draft 4–6 sentences for the paper (Methods + Limitations): which parameters are identified, which are not, and the consequence (scenarios, not point predictions). If ≥3 of 6 parameters are unidentified, record the decision to fix them at expert values in calibration and say so in the paper.
Done-when: Paragraph with numbers drafted in `output/identifiability_note.md`; decision (fix-or-report) recorded there and mirrored in THESES.md §7 during 5.9.

### 4.1 Modern-era proxy validation against external indices — [ ] ‖
Class: explore · Needs: none · Check: `python -c "import json; v=json.load(open('output/proxy_validation.json')); assert len(v['correlations'])>=9"`
Context: ADR-027: proxies are currently unvalidated (ADR-009 deferral). For USA/Russia the overlap with external indices is direct and checkable now — this converts a known reviewer objection into a result.
Do: Build `data/validation/external_indices.json` (manually transcribed decadal series, each with source): WID.world top-1% income share and top-10% wealth share (K proxies), V-Dem polyarchy/corruption indices and WVS generalized-trust series (C proxies), Our World in Data energy per capita and World Bank R&D % GDP (T proxies). Write `run_proxy_validation.py`: Spearman correlation of each external series against the corresponding proxy trajectory values at matching years for USA and Russia. Output `output/proxy_validation.json` with per-indicator ρ, n, p.
Done-when: JSON has ≥9 correlation entries (≥3 per component across the two countries); interpretation sentences (which proxies track external data, which do not) included in the JSON.

### 4.2 Face-validity table for ancient/estimated proxies — [ ] ‖
Class: implement · Needs: 2.1 · Check: `test -f output/table_proxy_validity.tex`
Context: ADR-027: Rome/USSR/Byzantium indicators cannot be correlated with external indices at the required depth; the honest treatment is a documented face-validity table with per-indicator uncertainty grades.
Do: Create `output/table_proxy_validity.tex` (script `run_proxy_validity.py` or extend `latex_export.py`): rows = every indicator used by Rome/USSR/Byzantium; columns = indicator, source(s) (from `data_sources`), estimate type (documented value / interpolation / expert estimate), uncertainty grade (A: sourced quantitative, B: sourced qualitative range, C: expert estimate). Grade each honestly — the `estimate_basis` fields from 2.1/2.2 are the input.
Done-when: Table compiles; every indicator of the three civilizations appears exactly once; grades are traceable to `estimate_basis` text.

### 5.1 Add Methods section — [ ]
Class: implement · Needs: 2.3, 2.4, 3.2, 4.1 · Check: `grep -c "subsection" paper/sections/methods.tex`
Context: v4.0 has no Methods section — calibration, normalization, metrics, and the train/test protocol are absent from the manuscript (a journal referee will not reconstruct them from the appendix parameter table). IJA expects full methodological transparency.
Do: Create `paper/sections/methods.tex` with subsections: (i) Proxy construction and normalization — the weighted min-max scheme, per-civilization proxy subsets, the incommensurability caveat with the quantified numbers from 2.3, uncertainty grades from 4.2; (ii) Calibration — differential evolution setup (bounds, objective, penalty) and the Bayesian ABC extension from 3.1; (iii) Evaluation protocol — turn accuracy, baselines (persistence/majority), block-permutation null, Wilson CIs, LOO-CV design; (iv) Data and code availability (GitHub + Zenodo DOI). Add `\input{sections/methods}` to `paper/article.tex` between model and results. Update `paper/article.tex` and `BASE_FILES.md` (23rd base file).
Done-when: Section compiles; every metric reported in `results.tex` is defined in `methods.tex`; parameter-table reference moved to Methods/Appendix consistently.

### 5.2 Document the reset/recovery mechanism + ablations + hand-set inputs — [ ]
Class: implement · Needs: 0.3, 5.1 · Check: `python .github/scripts/guardrail_check.py abstract paper/sections/ GUARDRAILS.md output/results.json`
Context: ADR-023: after ablation evidence exists, the paper's model section must describe the system actually simulated: the `recovery_potential` state, the stochastic reset (sigmoid product + RNG gate + jump overrides), the `historical_T_peak` scaling, and every hand-set per-civilization input (T0/K0/C0, shock dates and magnitudes, peaks, decay factors) — currently none of these appear.
Do: Edit `paper/sections/model.tex`: add a subsection "Recovery dynamics and stochastic resets" with equations transcribed exactly from `dynamics.py` (update THESES.md §3 accordingly in 5.9); include the ablation table (`table_ablation.tex`) with the interpretation from 0.3; add the hand-set inputs disclosure table (per-civilization block with sources for each hand-set value — data from `data/parameters.json` `civilizations` section and proxy-file `shocks`).
Done-when: Every parameter key in `data/parameters.json` (default_model + civilizations) appears somewhere in the paper — verified by the extended guardrail check (6.1); ablation numbers in the text match `output/ablation_results.json`.

### 5.3 Related-work engagement and novelty repositioning — [ ] ‖
Class: implement · Needs: none · Check: `grep -c "haqqmisra2009\|landis\|cir" paper/sections/bibliography.tex`
Context: ADR-029: the endogenous-stagnation idea has published prior art (Haqq-Misra & Baum 2009 "sustainability solution"; Kinouchi 2001; Landis 1999; Ćirković's Great Silence corpus) and empirical cliodynamics has Seshat (Turchin et al. 2018). Claiming the plateau idea as novel without engaging these invites desk rejection; the true novelty is the calibrated formal dynamical model.
Do: Edit `paper/sections/introduction.tex`: add a subsection "Stagnation solutions and prior formalizations" — position the Great Plateau relative to Haqq-Misra & Baum (2009), Kinouchi (2001), Landis (1999); state explicitly that EDAP's contribution is the formal T-K-C dynamical system with empirical calibration, not the stagnation concept; one paragraph on Seshat as the empirical benchmark cliodynamics uses and why EDAP's proxy approach differs. Add to `paper/sections/bibliography.tex`: Haqq-Misra & Baum (2009) Int. J. Astrobiol. 8:213; Kinouchi (2001) Europhys. Lett. 54:786; Landis (1999) J. Brit. Interplanetary Soc. 52:332; Ćirković (2018) *The Great Silence* (MIT Press); Turchin et al. (2018) Nature 559 (Seshat); plus Brown (1971) and Heather (2005) for 5.6.
Done-when: New subsection compiles; all six references present and cited; no sentence anywhere claims novelty for the stagnation idea itself.

### 5.4 Universality argument subsection — [ ] ‖
Class: implement · Needs: none · Check: `grep -c "universality" paper/sections/introduction.tex paper/sections/discussion.tex`
Context: ADR-029: the Fermi link currently assumes, without argument, that elite extraction is a convergent property of any complex civilization. A referee will ask about eusocial intelligences or AI singletons.
Do: Add a subsection (introduction or discussion — recommend discussion, before "From Plateau to Filter"): "On the universality of extraction dynamics" — argue convergence from (i) surplus creation enabling capture, (ii) information asymmetry favoring extractors, (iii) coalition stability of extractors over dispersed cooperators (Olson logic); explicitly treat counterexample classes (eusocial, singleton/AI, post-scarcity) as boundary cases where the model does not apply; label the whole argument as conjecture per GUARDRAILS confidence rules.
Done-when: Subsection compiles; counterexample classes addressed; guardrail check passes (no overclaiming language).

### 5.5 Terminology defense paragraph + neutral synonym — [ ] ‖
Class: implement · Needs: none · Check: `grep -c "rent extraction" paper/sections/abstract.tex paper/sections/introduction.tex`
Context: ADR-028: "intraspecific economic cannibalism" stays (the model is published under the term), but the paper must define it non-normatively up front and carry the neutral synonym "elite rent extraction" alongside, blunting the normative-reading objection the GUARDRAILS already guard against.
Do: Edit `paper/sections/introduction.tex` §Intraspecific Economic Cannibalism: add a defense paragraph — the biological metaphor's rationale (extraction consumes the adaptive fitness of conspecifics, analogous to intraspecific parasitism), the non-normative structural definition, and first-mention pairing "intraspecific economic cannibalism (elite rent extraction)". Use the neutral synonym at least once in `abstract.tex`.
Done-when: Defense paragraph present; abstract uses the neutral synonym; tone check passes.

### 5.6 Dark Ages historiography caveat — [ ] ‖
Class: implement · Needs: none · Check: `grep -c "transformation" paper/sections/discussion.tex`
Context: The post-Roman Dark Ages case is presented as uncontested, but the collapse-vs-transformation debate (Ward-Perkins 2005 vs Brown 1971 / Heather 2005) is live; "recovery required exogenous shocks" also blurs endogenous/exogenous (Church and cities were European developments).
Do: Edit `paper/sections/discussion.tex` §Historical Precedent: add 2–3 sentences acknowledging the transformation school and stating that EDAP's claim is compatibility with the extractive-collapse reading, not adjudication of the debate; soften the exogenous-recovery sentence to note the endogenous/exogenous boundary is itself model-relative.
Done-when: Caveat sentences present; Brown (1971) and Heather (2005) cited; guardrail check passes.

### 5.7 Rewrite abstract — [ ]
Class: implement · Needs: 1.4, 2.4, 3.2, 5.3 · Check: `grep -c "leave-one-out\|cross-valid" paper/sections/abstract.tex`
Context: GUARDRAILS (new in this cycle): no in-sample number leads; the abstract currently leads with train 0.70. The headline empirical number is now the LOO-CV held-out result with its CI.
Do: Rewrite `paper/sections/abstract.tex`: (i) theoretical contribution first; (ii) LOO-CV held-out TA with Wilson CI and baseline comparison as the empirical headline; (iii) Bayesian uncertainty mentioned; (iv) filter transition kept qualitative; (v) neutral synonym once. Numbers must come from final `output/results.json` / `output/loocv_results.json`.
Done-when: Abstract contains no in-sample-only number; all numbers in it exist in the output JSONs; guardrail check on abstract passes.

### 5.8 Rewrite results + Bifurcation/conservatism passages — [ ]
Class: implement · Needs: 1.4, 2.4, 3.2, 0.3 · Check: `python .github/scripts/guardrail_check.py abstract paper/sections/ GUARDRAILS.md output/results.json`
Context: Results must now carry: LOO-CV table, baselines and CIs per civilization, ablation table, posterior intervals, Byzantium/Japan(/China/Britain) evaluations. The "conservatism is a feature" BSS spin and the "precisely at bifurcation" claim must be rewritten per 1.3's verdict.
Do: Rewrite `paper/sections/results.tex` around the regenerated tables; add subsections "Leave-one-out cross-validation" and "Ablations"; rewrite the BSS paragraph as a stated limitation with the multi-objective future work; rewrite the bifurcation sentence using `bifurcation_sensitivity.json` verdict (keep the near-bifurcation claim only if verdict is "robust").
Done-when: Every number in results.tex traceable to an output JSON; the two vulnerable passages rewritten; guardrail check passes.

### 5.9 Sync THESES.md to the revised argumentation — [ ]
Class: implement · Needs: 5.2, 5.3, 5.7, 5.8 · Check: `grep -c "LOO\|leave-one-out" THESES.md`
Context: THESES.md is the paper's argumentation base file; it must mirror the revised chain or the knowledge system desyncs from the paper.
Do: Update THESES.md: §4.2 sample table (new civilizations); §4.3 calibration (add Bayesian block); §5.2 replace train/test TA with LOO-CV results; §5.3 rewrite BSS framing to the limitation form; §5.5 add ablation results; §6 add universality + prior-art positioning; §2.1 add terminology defense note; §7 update limitations (identifiability from 3.3, normalization sensitivity numbers, hand-set inputs disclosure).
Done-when: Every claim in THESES.md matches the paper and output JSONs; no v4.0-era numbers remain where superseded.

### 5.10 IJA format and statements pass — [ ]
Class: review · Needs: 5.7, 5.8 · Check: `grep -c "Data availability\|Competing interests" paper/sections/*.tex paper/article.tex`
Context: IJA (Cambridge) requires data-availability, code-availability, competing-interests, author-contributions statements and specific figure formats; v4.0 has none of these.
Do: Check current IJA author guidelines (Cambridge site); add the required statements (data: proxy JSONs + Zenodo DOI; code: GitHub repo + Zenodo software DOI 10.5281/zenodo.20298963); verify figures meet resolution/format rules; add keywords; verify reference style matches the journal's.
Done-when: Statements present; a checklist against the IJA guidelines recorded in `output/ija_checklist.md`.

### 6.1 Extend guardrail_check.py with parameter-disclosure check — [ ]
Class: implement · Needs: none · Check: `python .github/scripts/guardrail_check.py abstract paper/sections/ GUARDRAILS.md output/results.json && echo PASSED`
Context: The v4.0 defect (undisclosed mechanisms/parameters) was possible because no check compares `data/parameters.json` against the paper. GUARDRAILS now requires full disclosure; CI should enforce it mechanically.
Do: Edit `.github/scripts/guardrail_check.py`: add a check that extracts every parameter key from `data/parameters.json` (`default_model` plus every entry under `civilizations`) and asserts each appears in the concatenation of `paper/sections/model.tex + methods.tex + appendix.tex + output/table_parameters.tex`; report missing keys with exit code 1.
Done-when: Check fails on the current v4.0 paper (demonstrating it catches the defect) and passes after 5.2/5.1 land.

### 6.2 Update knowledge and packaging files — [ ]
Class: implement · Needs: 5.9 · Check: `grep -c "methods.tex\|byzantium" BASE_FILES.md INDEX.md`
Context: BASE_FILES.md, INDEX.md, README.md describe the file set and headline numbers; they desync if untouched (project rule: base files are the source of truth).
Do: Update `BASE_FILES.md` (add `paper/sections/methods.tex`, new proxy JSONs, new run_*.py scripts to the base/generated tables as appropriate); `INDEX.md` (new files + plan reference); `README.md` (v4.1 headline = LOO-CV result; new civilizations listed); `ROADMAP.md` (mark v4.1 delivered, adjust v5.0 remainder — N(t), galactic ensemble); `CITATION.cff` / `.zenodo.json` (version 4.1).
Done-when: All five files reference the v4.1 file set and numbers; no stale v4.0-only claims.

### 6.3 Full pipeline regeneration + compile + tests — [ ]
Class: implement · Needs: 6.2, 5.8 · Check: `python run_model.py && python -m pytest tests/ -v && cd paper && pdflatex -interaction=nonstopmode article.tex`
Context: Everything regenerates from base files; the final PDF must compile against the final tables/figures.
Do: Run `python run_model.py`, `run_ablations.py`, `run_loocv.py`, `run_normalization_sensitivity.py`, `run_proxy_validation.py`; `python -m pytest tests/ -v`; compile the paper three times (`pdflatex article.tex`); visually inspect `paper/article.pdf` (or `output/article.pdf` per the working-directory convention in CI).
Done-when: All scripts exit 0; tests green; PDF compiles with no missing references (`grep -c "undefined" article.log` → 0).

### 6.4 Final review: guardrails, metric diff, PDF read — [ ]
Class: review · Needs: 6.3, 6.1 · Check: `python .github/scripts/diff_metrics.py output/results.json <previous release results.json>`
Context: Before tagging v4.1, verify no guardrail violations, understand metric movement vs v4.0 (some numbers will get worse — that is acceptable and must be explainable, not hidden), and read the compiled PDF end to end.
Do: Run the guardrail check and diff_metrics against the v4.0 results (commit 5ddb606); read the full PDF; fix stragglers; write a 10-line summary of metric deltas and why.
Done-when: Guardrail check green; deltas summarized in `output/v41_release_notes.md`; PDF read completed.

### 6.5 Tag v4.1 release — [ ]
Class: implement · Needs: 6.4 · Check: `git tag -l "v4.1*"`
Context: Release discipline: version tag triggers the CI release workflow (.github/workflows/release.yml) and Zenodo archiving.
Do: Commit all changes with a conventional message; tag `v4.1`; push; confirm the GitHub release workflow passes and Zenodo mints the versioned DOI.
Done-when: Tag exists; CI green; Zenodo record shows v4.1.
