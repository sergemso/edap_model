# EDAP Model v4.0

**Endogenous Dissipation of Adaptive Potential: A Mathematical Model of Civilization Dynamics and the Fermi Paradox**

## Overview

The EDAP model formalizes a mechanism of civilizational collapse through *intraspecific economic cannibalism* — the progressive concentration of resources by elites who optimize extraction over adaptation.

The model describes the co-evolution of three variables:
- **T** (Technology): fraction of realizable technologies actually deployed
- **K** (Cannibalism): fraction of resources extracted by elites without productive contribution
- **C** (Cooperation): capacity for collective action and institutional trust

When K exceeds a critical threshold K_crit(T), adaptive potential dissipates, and civilization enters a *feudal trap* — a stable low-technology equilibrium.

## Key Results

- **Turn accuracy:** 0.70 (train), 0.52 (test)
- **Perfect prediction:** Soviet Union (5/5 directional changes correct)
- **Structural tension detection:** Model identifies turning points that align with documented elite compensatory interventions (New Deal, Bretton Woods, QE, etc.)
- **Bifurcation:** System bifurcates at α_K ≈ 0.12 — below this, both sustainable development and feudal trap are possible; above, only the trap remains

## Installation

```bash
git clone https://github.com/sergemso/edap_model.git
cd edap-v4
pip install -r requirements.txt
```

## Usage

### Run the full analysis pipeline

```bash
python run_model.py
```

This generates:
- `output/fig1_phase_portrait.png` — Phase portrait with historical trajectories
- `output/fig2_time_series.png` — Time series of T, K, C with K_crit overlay
- `output/fig3_bifurcation.png` — Bifurcation analysis and phase portraits
- `output/fig4_monte_carlo.png` — Monte Carlo forecasts (GPU-accelerated)
- `output/fig5_summary_table.png` — Civilizational risk assessment
- `output/results.json` — All numerical results
- `output/table_*.tex` — LaTeX tables for the paper

### Run parameter calibration

```bash
python calibrate.py
```

Requires NVIDIA GPU with CUDA for acceleration. Falls back to CPU if unavailable.

### Run tests

```bash
python -m pytest tests/test_model.py -v
```

## Paper

The paper source is split into sections for modular editing:

```
paper/
├── article.tex          # Skeleton with preamble and \input{sections/...}
└── sections/
    ├── abstract.tex
    ├── introduction.tex
    ├── model.tex
    ├── results.tex
    ├── discussion.tex
    ├── conclusion.tex
    ├── appendix.tex
    └── bibliography.tex
```

Compile: `cd paper/ && pdflatex article.tex` (run three times for cross-references).

## Requirements

- Python 3.10+
- NumPy, SciPy, Matplotlib, Numba
- NVIDIA CUDA + Numba CUDA (optional, for GPU acceleration)
- See `requirements.txt` for full list

## Data

Proxy data for four civilizations are provided in `data/raw/`:
- Roman Empire (100 BCE – 476 CE)
- Soviet Union (1922–1991)
- Russian Federation (2000–2024)
- United States (1933–2024)

Full proxy documentation with historical sources is included in the proxy files.

## Citation

If you use this model or data in your research, please cite:

```bibtex
@software{strebulaev2026edap,
  author       = {Strebulaev, S. A.},
  title        = {EDAP Model v4.0: Endogenous Dissipation of Adaptive Potential},
  year         = {2026},
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.20298963},
  url          = {https://github.com/sergemso/edap_model}
}
```

## License

This project is licensed under the Creative Commons Attribution 4.0 International License (CC-BY-4.0). See `LICENSE` for details.

## Contact

Sergey A. Strebulaev — strebulaev@gmail.com
