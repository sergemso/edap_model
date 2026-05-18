"""Run GPU calibration for EDAP v4.0."""

import json, os, sys
sys.path.insert(0, os.path.dirname(__file__))

from edap_model import Normalizer, load_json, DATA_DIR
from edap_model.calibration import calibrate_all

if __name__ == "__main__":
    print("=" * 60)
    print("EDAP v4.0 — GPU Calibration")
    print("=" * 60)
    
    model_params = load_json(os.path.join(DATA_DIR, 'parameters.json'))
    normalizer = Normalizer(os.path.join(DATA_DIR, 'normalization_params.json'))
    
    results = calibrate_all(normalizer, model_params, seed=42)
    
    os.makedirs('output', exist_ok=True)
    with open('output/calibration_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print("\nResults saved to output/calibration_results.json")