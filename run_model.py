"""Run EDAP Model v3.1 analysis pipeline."""

import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(__file__))

from edap_model import EDAPModel, Normalizer, load_json, DATA_DIR, OUTPUT_DIR
from edap_model.figures.fig1_phase import figure1_phase_portrait
from edap_model.figures.fig2_timeseries import figure2_time_series
from edap_model.figures.fig3_bifurcation import figure3_bifurcation
from edap_model.figures.fig4_montecarlo import figure4_monte_carlo
from edap_model.figures.fig5_summary import figure5_summary_table
from edap_model.export import export_results_json

if __name__ == "__main__":
    print("=" * 60)
    print("EDAP Model v3.1 — Analysis Pipeline")
    print("=" * 60)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("\nLoading configuration...")
    model_params = load_json(os.path.join(DATA_DIR, 'parameters.json'))
    normalizer = Normalizer(os.path.join(DATA_DIR, 'normalization_params.json'))
    figures = [
        ('Figure 1: Phase Portrait', figure1_phase_portrait, 'fig1_phase_portrait.png'),
        ('Figure 2: Time Series', figure2_time_series, 'fig2_time_series.png'),
        ('Figure 3: Bifurcation', figure3_bifurcation, 'fig3_bifurcation.png'),
        ('Figure 4: Monte Carlo Forecast', figure4_monte_carlo, 'fig4_monte_carlo.png'),
        ('Figure 5: Summary Table', figure5_summary_table, 'fig5_summary_table.png'),
    ]
    for name, func, fname in figures:
        print(f"\n{'─' * 40}")
        print(f"  {name}")
        try:
            if func in [figure1_phase_portrait, figure2_time_series, figure4_monte_carlo]:
                func(normalizer, model_params, os.path.join(OUTPUT_DIR, fname))
            elif func == figure3_bifurcation:
                func(model_params, os.path.join(OUTPUT_DIR, fname))
            else:
                func(os.path.join(OUTPUT_DIR, fname))
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
    print(f"\n{'─' * 40}")
    try:
        export_results_json(model_params, normalizer, os.path.join(OUTPUT_DIR, 'results.json'))
        from edap_model.latex_export import generate_latex_tables
        generate_latex_tables(os.path.join(OUTPUT_DIR, 'results.json'), OUTPUT_DIR)
    except Exception as e:
        print(f"  ERROR: {e}")
    print("\n" + "=" * 60)
    print("Done. Outputs in 'output/'")
    print("=" * 60)