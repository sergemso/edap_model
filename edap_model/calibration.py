"""GPU-accelerated parameter calibration for EDAP v4.0.
Constrained: penalties for K∉[0,1], C∉[0,1].
Calibrates turn_accuracy + BSS — metrics the model is designed for.
"""

import numpy as np
from scipy.optimize import differential_evolution
from edap_model.model import EDAPModel
from edap_model.dynamics import simulate_gpu_batch, HAS_CUDA
from edap_model.metrics import compute_turn_accuracy, compute_brier_skill_score
from edap_model.utils import load_json, RAW_DIR
import os
import time


def _py(val):
    if isinstance(val, (np.integer,)):
        return int(val)
    if isinstance(val, (np.floating,)):
        return float(val)
    if isinstance(val, np.ndarray):
        return val.tolist()
    return val


def calibrate_all(normalizer, model_params, seed=42):
    # Parameter bounds — same as before
    param_bounds = {
        "alpha_K_base": (0.10, 0.90),
        "beta_T": (0.02, 0.30),
        "gamma_T": (0.05, 0.50),
        "lambda_shadow": (0.02, 0.30),
        "mu_shadow": (0.02, 0.40),
        "eta": (0.10, 0.80),
    }

    train_civs = ["rome_proxies.json", "ussr_proxies.json"]
    test_civs = ["russia_proxies.json", "usa_proxies.json"]

    results = {"train": {}, "test": {}, "best_params": None}

    # Preload train data
    train_data = []
    for fn in train_civs:
        raw = load_json(os.path.join(RAW_DIR, fn))
        nd = normalizer.normalize_civilization(raw)
        train_data.append((fn, raw, nd))

    bounds = [
        param_bounds[k]
        for k in [
            "alpha_K_base",
            "beta_T",
            "gamma_T",
            "lambda_shadow",
            "mu_shadow",
            "eta",
        ]
    ]

    # Faster convergence
    maxiter = 50
    popsize = 20
    n_gpu_sims = 50
    penalty_weight = 5.0  # Strong penalty for out-of-bounds

    print(f"Constrained GPU calibration")
    print(f"Bounds: {param_bounds}")
    print(f"maxiter={maxiter}, popsize={popsize}")
    print(f"Penalty weight for K,C out of [0,1]: {penalty_weight}")

    start_time = time.time()
    eval_count = [0]
    best_score = [float("inf")]

    def objective(params):
        alpha_K_base, beta_T, gamma_T, lambda_shadow, mu_shadow, eta = params
        total_score = 0.0
        total_penalty = 0.0

        for fn, raw, nd in train_data:
            years = np.array([p["year"] for p in nd])
            T_data = np.array([p["T"] for p in nd])
            K_data = np.array([p["K"] for p in nd])
            T0, K0, C0 = nd[0]["T"], nd[0]["K"], nd[0]["C"]
            t_span = years[-1] - years[0] + 20
            historical_T_peak = max(p["T"] for p in nd)

            model = EDAPModel(
                {
                    "alpha_K_base": alpha_K_base,
                    "beta_T": beta_T,
                    "gamma_T": gamma_T,
                    "lambda_shadow": lambda_shadow,
                    "mu_shadow": mu_shadow,
                    "eta": eta,
                }
            )
            model.cycles.historical_T_peak = historical_T_peak

            if HAS_CUDA:
                n_steps = int(t_span / 0.05)
                T_arr, K_arr, C_arr = simulate_gpu_batch(
                    model, T0, K0, C0, n_gpu_sims, t_span, n_steps, seed=seed
                )

                # Penalty for out-of-bounds
                penalty_K = np.sum(np.maximum(-K_arr, 0)) + np.sum(
                    np.maximum(K_arr - 1, 0)
                )
                penalty_C = np.sum(np.maximum(C_arr - 1, 0)) + np.sum(
                    np.maximum(-C_arr, 0)
                )
                total_penalty += (
                    penalty_weight
                    * (penalty_K + penalty_C)
                    / (n_gpu_sims * K_arr.shape[1])
                )

                # Metrics
                ta_sum = 0.0
                bss_sum = 0.0
                sim_years = np.linspace(years[0], years[0] + t_span, T_arr.shape[1])

                for i in range(n_gpu_sims):
                    T_model = np.interp(years, sim_years, T_arr[i, :])
                    K_model = np.interp(years, sim_years, K_arr[i, :])

                    data_sign = np.sign(np.diff(T_data))
                    model_sign = np.sign(np.diff(T_model))
                    mask = data_sign != 0
                    if mask.sum() > 0:
                        ta_sum += np.mean(data_sign[mask] == model_sign[mask])
                    else:
                        ta_sum += 1.0

                    Kc_model = model.K_critical(T_model)
                    Kc_data = model.K_critical(T_data)
                    actual = (K_data > Kc_data).astype(float)
                    predicted = (K_model > Kc_model).astype(float)
                    brier = np.mean((predicted - actual) ** 2)
                    p_clim = np.mean(actual)
                    brier_clim = np.mean((p_clim - actual) ** 2)
                    if brier_clim > 0:
                        bss_sum += 1.0 - brier / brier_clim

                ta = ta_sum / n_gpu_sims
                bss = bss_sum / n_gpu_sims
            else:
                model.set_seed(seed)
                sim = model.simulate(T0, K0, C0, t_span=t_span, n_points=500)
                sy = years[0] + sim["t"]

                # Penalty for out-of-bounds
                K_arr = sim["K"]
                C_arr = sim["C"]
                penalty_K = np.sum(np.maximum(-K_arr, 0)) + np.sum(
                    np.maximum(K_arr - 1, 0)
                )
                penalty_C = np.sum(np.maximum(C_arr - 1, 0)) + np.sum(
                    np.maximum(-C_arr, 0)
                )
                total_penalty += penalty_weight * (penalty_K + penalty_C) / len(K_arr)

                ta, _, _ = compute_turn_accuracy(nd, sim, sy)
                bss = compute_brier_skill_score(nd, sim, sy, model)

            total_score += _py(ta) + _py(bss)

        eval_count[0] += 1
        score = -total_score + total_penalty  # Minimise: - (TA+BSS) + penalty
        elapsed = time.time() - start_time

        if eval_count[0] % 5 == 0 or score < best_score[0]:
            marker = " *** NEW BEST" if score < best_score[0] else ""
            print(
                f"[{elapsed:6.1f}s] Eval {eval_count[0]:4d}: score={score:.4f} "
                f"(TA+BSS={total_score:.4f}, penalty={total_penalty:.2f}), "
                f"α_K={alpha_K_base:.3f}, η={eta:.3f}{marker}"
            )

        if score < best_score[0]:
            best_score[0] = score

        return score

    result = differential_evolution(
        objective,
        bounds,
        seed=seed,
        maxiter=maxiter,
        popsize=popsize,
        disp=False,
        polish=False,
    )

    elapsed = time.time() - start_time
    print(f"\nCalibration finished in {elapsed:.1f}s")

    best_params = {
        "alpha_K_base": float(result.x[0]),
        "beta_T": float(result.x[1]),
        "gamma_T": float(result.x[2]),
        "lambda_shadow": float(result.x[3]),
        "mu_shadow": float(result.x[4]),
        "eta": float(result.x[5]),
    }
    results["best_params"] = best_params
    print(f"Best params: {best_params}")
    print(f"Best objective: {result.fun:.4f}")

    # Evaluate on train
    for fn, raw, nd in train_data:
        name = raw.get("civilization", fn)
        years = np.array([p["year"] for p in nd])
        T0, K0, C0 = nd[0]["T"], nd[0]["K"], nd[0]["C"]
        t_span = years[-1] - years[0] + 20

        model = EDAPModel(best_params)
        model.shock_history = {
            float(k) - years[0]: v for k, v in raw.get("shocks", {}).items()
        }
        model.cycles.historical_T_peak = max(p["T"] for p in nd)
        model.set_seed(seed)
        sim = model.simulate(T0, K0, C0, t_span=t_span, n_points=500)
        sy = years[0] + sim["t"]

        ta, n_turns, n_correct = compute_turn_accuracy(nd, sim, sy)
        bss = compute_brier_skill_score(nd, sim, sy, model)

        # Check bounds
        K_ok = (sim["K"].min() >= 0) and (sim["K"].max() <= 1)
        C_ok = (sim["C"].min() >= 0) and (sim["C"].max() <= 1)

        results["train"][name] = {
            "params": best_params,
            "turn_accuracy": _py(ta),
            "brier_skill_score": _py(bss),
            "final_T": _py(sim["T"][-1]),
            "final_K": _py(sim["K"][-1]),
            "final_C": _py(sim["C"][-1]),
            "K_in_bounds": K_ok,
            "C_in_bounds": C_ok,
        }
        print(f"  Train {name}: TA={ta:.3f}, BSS={bss:.3f}, K_ok={K_ok}, C_ok={C_ok}")

    # Evaluate on test
    for fn in test_civs:
        raw = load_json(os.path.join(RAW_DIR, fn))
        name = raw.get("civilization", fn)
        nd = normalizer.normalize_civilization(raw)
        years = np.array([p["year"] for p in nd])
        T0, K0, C0 = nd[0]["T"], nd[0]["K"], nd[0]["C"]
        t_span = years[-1] - years[0] + 20

        model = EDAPModel(best_params)
        model.shock_history = {
            float(k) - years[0]: v for k, v in raw.get("shocks", {}).items()
        }
        model.cycles.historical_T_peak = max(p["T"] for p in nd)
        model.set_seed(seed)
        sim = model.simulate(T0, K0, C0, t_span=t_span, n_points=500)
        sy = years[0] + sim["t"]

        ta, n_turns, n_correct = compute_turn_accuracy(nd, sim, sy)
        bss = compute_brier_skill_score(nd, sim, sy, model)

        K_ok = (sim["K"].min() >= 0) and (sim["K"].max() <= 1)
        C_ok = (sim["C"].min() >= 0) and (sim["C"].max() <= 1)

        results["test"][name] = {
            "params": best_params,
            "turn_accuracy": _py(ta),
            "brier_skill_score": _py(bss),
            "final_T": _py(sim["T"][-1]),
            "final_K": _py(sim["K"][-1]),
            "final_C": _py(sim["C"][-1]),
            "K_in_bounds": K_ok,
            "C_in_bounds": C_ok,
        }
        print(f"  Test {name}: TA={ta:.3f}, BSS={bss:.3f}, K_ok={K_ok}, C_ok={C_ok}")

    return results
