"""Working metrics for EDAP model evaluation.

Metrics that make sense for a causal model of dissipation:
  1. Turn accuracy: does model predict sign of dT/dt correctly?
  2. K-crossing lead time: how early does model warn of K > K_crit?
  3. Brier skill score: relative to climatological baseline.
"""

import numpy as np


def compute_turn_accuracy(norm_data, sim_result, sim_years):
    """Fraction of consecutive data points where model correctly predicts sign of dT/dt."""
    years = np.array([p["year"] for p in norm_data])
    T_data = np.array([p["T"] for p in norm_data])
    T_model = np.interp(years, sim_years, sim_result["T"])

    data_sign = np.sign(np.diff(T_data))
    model_sign = np.sign(np.diff(T_model))

    mask = data_sign != 0
    if mask.sum() == 0:
        return 1.0, 0, 0

    correct = int((data_sign[mask] == model_sign[mask]).sum())
    total = int(mask.sum())

    return float(correct / total), total, correct


def compute_k_crossing_lead(norm_data, sim_result, sim_years, model):
    """How many years before actual K > K_crit did model predict it?"""
    years = np.array([p["year"] for p in norm_data])
    K_data = np.array([p["K"] for p in norm_data])
    T_data = np.array([p["T"] for p in norm_data])

    K_model = np.interp(years, sim_years, sim_result["K"])
    T_model = np.interp(years, sim_years, sim_result["T"])
    Kc_model = model.K_critical(T_model)

    Kc_data = model.K_critical(T_data)
    data_exceed = K_data > Kc_data
    if not data_exceed.any():
        return None, None, None

    actual_crossing = float(years[data_exceed.argmax()])

    model_exceed = K_model > Kc_model
    if not model_exceed.any():
        return None, actual_crossing, None

    model_crossing = float(years[model_exceed.argmax()])
    lead_years = float(actual_crossing - model_crossing)

    return lead_years, actual_crossing, model_crossing


def compute_brier_skill_score(norm_data, sim_result, sim_years, model):
    """Brier Skill Score relative to climatological baseline. Positive = better than baseline."""
    years = np.array([p["year"] for p in norm_data])
    K_data = np.array([p["K"] for p in norm_data])
    T_data = np.array([p["T"] for p in norm_data])

    K_model = np.interp(years, sim_years, sim_result["K"])
    T_model = np.interp(years, sim_years, sim_result["T"])
    Kc_model = model.K_critical(T_model)

    Kc_data = model.K_critical(T_data)
    actual = (K_data > Kc_data).astype(float)
    predicted = (K_model > Kc_model).astype(float)

    brier_model = float(np.mean((predicted - actual) ** 2))
    p_clim = float(actual.mean())
    brier_clim = float(np.mean((p_clim - actual) ** 2))

    if brier_clim == 0:
        return 0.0 if brier_model == 0 else -float("inf")

    return float(1.0 - brier_model / brier_clim)


def compute_all_working_metrics(norm_data, sim_result, sim_years, model):
    """Compute all working metrics for one civilization."""
    turn_acc, n_turns, n_correct = compute_turn_accuracy(
        norm_data, sim_result, sim_years
    )

    lead_result = compute_k_crossing_lead(norm_data, sim_result, sim_years, model)
    lead_years, actual_cross, model_cross = (
        lead_result if lead_result else (None, None, None)
    )

    bss = compute_brier_skill_score(norm_data, sim_result, sim_years, model)

    return {
        "turn_accuracy": round(turn_acc, 3),
        "turn_details": f"{n_correct}/{n_turns} turns correct",
        "k_crossing_lead_years": (
            round(lead_years, 1) if lead_years is not None else None
        ),
        "k_crossing_actual_year": (
            round(actual_cross, 1) if actual_cross is not None else None
        ),
        "k_crossing_model_year": (
            round(model_cross, 1) if model_cross is not None else None
        ),
        "brier_skill_score": round(bss, 3) if bss != -float("inf") else None,
    }
