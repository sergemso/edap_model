"""Export results to JSON with working metrics (no absolute T prediction)."""

import json
import os
import numpy as np
from edap_model.model import EDAPModel
from edap_model.metrics import compute_all_working_metrics
from edap_model.utils import load_json, RAW_DIR


def export_results_json(model_params, normalizer, filepath="output/results.json"):
    print("  Exporting results.json with working metrics...")
    results = {
        "model_version": "3.1",
        "parameters": model_params.get("default_model", {}),
        "train_test_split": {"train": ["Rome", "USSR"], "test": ["Russia", "USA"]},
        "civilizations": {},
    }

    pfs = sorted(
        [
            f
            for f in os.listdir(RAW_DIR)
            if f.endswith("_proxies.json")
            and not f.startswith("~")
            and os.path.getsize(os.path.join(RAW_DIR, f)) > 0
        ]
    )

    all_metrics = {}

    for fn in pfs:
        raw = load_json(os.path.join(RAW_DIR, fn))
        name = raw.get(
            "civilization", fn.replace("_proxies.json", "").replace("_", " ").title()
        )
        nd = normalizer.normalize_civilization(raw)

        dps = [
            {
                "year": p["year"],
                "label": p.get("label", ""),
                "T": round(float(p["T"]), 4),
                "K": round(float(p["K"]), 4),
                "C": round(float(p["C"]), 4),
            }
            for p in nd
        ]

        T0, K0, C0 = nd[0]["T"], nd[0]["K"], nd[0]["C"]
        tsp = nd[-1]["year"] - nd[0]["year"] + 20

        model = EDAPModel(model_params.get("default_model", {}))
        model.shock_history = {
            float(k) - nd[0]["year"]: v for k, v in raw.get("shocks", {}).items()
        }
        model.cycles.historical_T_peak = max(p["T"] for p in nd)
        model.set_seed(42)

        sim = model.simulate(T0, K0, C0, t_span=tsp, n_points=500)
        sy = [nd[0]["year"] + t for t in sim["t"]]

        sd = [
            {
                "year": round(sy[i], 1),
                "T": round(float(sim["T"][i]), 4),
                "K": round(float(sim["K"][i]), 4),
                "C": round(float(sim["C"][i]), 4),
                "K_crit": round(float(sim["K_crit"][i]), 4),
            }
            for i in range(0, len(sy), 50)
        ]

        aT = np.array([p["T"] for p in nd])
        aK = np.array([p["K"] for p in nd])

        role = "train" if name in ["Roman Empire", "Soviet Union"] else "test"
        wm = compute_all_working_metrics(nd, sim, sy, model)

        results["civilizations"][name] = {
            "file": fn,
            "period": raw.get("period", {}),
            "role": role,
            "data_points": dps,
            "simulation_subsampled": sd,
            "simulation_final_state": {
                "T": round(float(sim["T"][-1]), 4),
                "K": round(float(sim["K"][-1]), 4),
                "C": round(float(sim["C"][-1]), 4),
                "K_crit": round(float(sim["K_crit"][-1]), 4),
            },
            "cycle_info": {
                "final_recovery_potential": round(
                    float(sim.get("final_recovery_potential", 0)), 4
                )
            },
            "working_metrics": wm,
            "statistics": {
                "T_peak": round(float(aT.max()), 4),
                "T_min": round(float(aT.min()), 4),
                "K_peak": round(float(aK.max()), 4),
                "K_min": round(float(aK.min()), 4),
                "T_trend": "increasing" if aT[-1] > aT[0] else "decreasing",
                "K_trend": "increasing" if aK[-1] > aK[0] else "decreasing",
            },
        }

        all_metrics[name] = {"wm": wm, "role": role}

    # Summary
    results["metrics_summary"] = {}
    for role in ["train", "test"]:
        role_data = {k: v for k, v in all_metrics.items() if v["role"] == role}
        if role_data:
            wm_list = [v["wm"] for v in role_data.values()]
            results["metrics_summary"][role] = {
                "avg_turn_accuracy": round(
                    np.mean([w["turn_accuracy"] for w in wm_list]), 3
                ),
                "avg_k_crossing_lead_years": round(
                    np.mean(
                        [
                            w["k_crossing_lead_years"]
                            for w in wm_list
                            if w["k_crossing_lead_years"] is not None
                        ]
                    ),
                    1,
                ),
                "avg_brier_skill_score": round(
                    np.mean(
                        [
                            w["brier_skill_score"]
                            for w in wm_list
                            if w["brier_skill_score"] is not None
                        ]
                    ),
                    3,
                ),
                "positive_bss_count": sum(
                    1 for w in wm_list if (w.get("brier_skill_score") or -999) > 0
                ),
                "negative_bss_count": sum(
                    1 for w in wm_list if (w.get("brier_skill_score") or -999) <= 0
                ),
            }

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"  -> Saved to {filepath}")
