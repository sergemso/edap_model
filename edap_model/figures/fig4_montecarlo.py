"""Figure 4: Monte Carlo — GPU-accelerated."""

import os
import numpy as np
import matplotlib.pyplot as plt
from edap_model.model import EDAPModel
from edap_model.dynamics import simulate_gpu_batch, HAS_CUDA
from edap_model.utils import load_json, RAW_DIR
import time


def figure4_monte_carlo(normalizer, model_params, save_path):
    print("  Figure 4: Monte Carlo — P(K > K_crit) forecast...")
    ns = model_params.get("monte_carlo", {}).get("n_simulations", 200)
    tf = model_params.get("monte_carlo", {}).get("t_forecast", 25)

    pfs = sorted(
        [
            f
            for f in os.listdir(RAW_DIR)
            if f.endswith("_proxies.json")
            and not f.startswith("~")
            and os.path.getsize(os.path.join(RAW_DIR, f)) > 0
        ]
    )
    modern = []
    for fn in pfs:
        raw = load_json(os.path.join(RAW_DIR, fn))
        nd = normalizer.normalize_civilization(raw)
        if nd and nd[-1]["year"] >= 1900:
            modern.append((fn, raw, nd))

    if not modern:
        print("  No modern civilizations")
        return

    nm = len(modern)
    fig, axes = plt.subplots(nm, 2, figsize=(14, 5 * nm))
    if nm == 1:
        axes = np.array([axes])
    colors = ["#FF4500", "#006400", "#1f77b4"]

    for row, (fn, raw, nd) in enumerate(modern):
        name = raw.get(
            "civilization", fn.replace("_proxies.json", "").replace("_", " ").title()
        )
        color = colors[row % len(colors)]
        last = nd[-1]
        T0, K0, C0 = last["T"], last["K"], last["C"]
        ly = last["year"]

        model = EDAPModel(model_params.get("default_model", {}))
        model.cycles.historical_T_peak = max(p["T"] for p in nd)

        print(f"    {name}: running {ns} GPU simulations...")
        t_start = time.time()

        if HAS_CUDA:
            # GPU batch simulation
            n_steps = int(tf / 0.05)  # dt=0.05
            T_arr, K_arr, C_arr = simulate_gpu_batch(
                model, T0, K0, C0, ns, tf, n_steps, seed=42
            )
            # Reconstruct full trajectories (linear interpolation between stored points)
            t = np.linspace(ly, ly + tf, T_arr.shape[1])
        else:
            # CPU fallback
            T_traj, K_traj, C_traj = [], [], []
            for si in range(ns):
                model_cpu = EDAPModel(model_params.get("default_model", {}))
                model_cpu.set_seed(42 + si)
                model_cpu.cycles.historical_T_peak = max(p["T"] for p in nd)
                r = model_cpu.simulate(T0, K0, C0, t_span=tf, n_points=300)
                T_traj.append(r["T"])
                K_traj.append(r["K"])
                C_traj.append(r["C"])
            T_arr = np.array(T_traj)
            K_arr = np.array(K_traj)
            C_arr = np.array(C_traj)
            t = np.linspace(ly, ly + tf, T_arr.shape[1])

        elapsed = time.time() - t_start
        print(f"      Completed in {elapsed:.1f}s")

        # Compute K_crit for each trajectory
        Kc_arr = np.zeros_like(K_arr)
        for i in range(K_arr.shape[0]):
            Kc_arr[i, :] = model.K_critical(T_arr[i, :])

        K_exceed_arr = (K_arr > Kc_arr).astype(float)
        dT_arr = np.diff(T_arr, axis=1)
        decline_arr = (dT_arr < 0).astype(float)

        # Panel 1: P(K > K_crit)
        ax1 = axes[row, 0]
        p_exceed = K_exceed_arr.mean(axis=0)
        lower = np.percentile(K_exceed_arr, 5, axis=0)
        upper = np.percentile(K_exceed_arr, 95, axis=0)
        ax1.plot(t, p_exceed, color=color, lw=2.5, label="P(K > K_crit)")
        ax1.fill_between(t, lower, upper, color=color, alpha=0.15, label="90% CI")
        ax1.axhline(
            y=0.5, color="gray", ls=":", lw=1.0, alpha=0.5, label="50% threshold"
        )
        ax1.set_ylabel("P(K > K_crit)")
        ax1.set_ylim([0, 1.05])
        ax1.set_xlabel("Year")
        ax1.set_title(f"{name}: Probability of K > K_crit", fontweight="bold")
        ax1.legend(fontsize=8)
        ax1.grid(True, alpha=0.3)

        # Panel 2: P(dT/dt < 0)
        ax2 = axes[row, 1]
        p_decline = decline_arr.mean(axis=0)
        lower_d = np.percentile(decline_arr, 5, axis=0)
        upper_d = np.percentile(decline_arr, 95, axis=0)
        t_mid = (t[:-1] + t[1:]) / 2

        ax2.plot(t_mid, p_decline, color=color, lw=2.5, label="P(dT/dt < 0)")
        ax2.fill_between(
            t_mid, lower_d, upper_d, color=color, alpha=0.15, label="90% CI"
        )
        ax2.axhline(
            y=0.5, color="gray", ls=":", lw=1.0, alpha=0.5, label="50% threshold"
        )
        ax2.set_ylabel("P(Decline)")
        ax2.set_ylim([0, 1.05])
        ax2.set_xlabel("Year")
        ax2.set_title(f"{name}: Probability of Technology Decline", fontweight="bold")
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3)

    gpu_str = "GPU" if HAS_CUDA else "CPU"
    plt.suptitle(
        f"EDAP Model v3.1: Monte Carlo Forecast ({ns} simulations, 90% CI, {gpu_str})\n"
        "Predicting K-crossing and direction, not absolute T level",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"  -> Saved to {save_path}")
