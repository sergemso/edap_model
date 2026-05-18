"""Figure 2: Time series — data only with K_crit overlay, no absolute T prediction."""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from edap_model.model import EDAPModel
from edap_model.utils import load_json, RAW_DIR


def figure2_time_series(normalizer, model_params, save_path):
    print("  Figure 2: Time series (data + K_crit only)...")
    pfs = sorted(
        [
            f
            for f in os.listdir(RAW_DIR)
            if f.endswith("_proxies.json")
            and not f.startswith("~")
            and not f.startswith(".")
            and os.path.getsize(os.path.join(RAW_DIR, f)) > 0
        ]
    )
    if not pfs:
        print("  WARNING: No proxy files found")
        return

    colors = [
        "#8B0000",
        "#CC0000",
        "#FF4500",
        "#006400",
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
    ]
    n = len(pfs)
    fig, axes = plt.subplots(n, 3, figsize=(18, 4 * n))
    if n == 1:
        axes = np.array([axes])

    for row, fname in enumerate(pfs):
        color = colors[row % len(colors)]
        raw = load_json(os.path.join(RAW_DIR, fname))
        name = raw.get(
            "civilization", fname.replace("_proxies.json", "").replace("_", " ").title()
        )
        nd = normalizer.normalize_civilization(raw)

        ya = np.array([p["year"] for p in nd])
        Ta = np.array([p["T"] for p in nd])
        Ka = np.array([p["K"] for p in nd])
        Ca = np.array([p["C"] for p in nd])

        model = EDAPModel(model_params.get("default_model", {}))
        Kca = model.K_critical(Ta)
        Kca_tech = model.K_critical(np.ones_like(Ta) * 0.65)  # K_crit at T_sing

        for col, (yl, ylim, data, kc_data, ylabel) in enumerate(
            [
                (Ta, [0, 1.05], Ta, None, "Technology (T)"),
                (Ka, [0, 1.05], Ka, Kca, "Cannibalism (K)"),
                (Ca, [0, 1.05], Ca, None, "Cooperation (C)"),
            ]
        ):
            ax = axes[row, col]
            ax.plot(ya, data, color=color, lw=2.5, label="Data")

            if kc_data is not None:
                ax.plot(
                    ya,
                    kc_data,
                    "-.",
                    color="red",
                    lw=1.5,
                    alpha=0.8,
                    label=r"$K_{crit}(T)$",
                )
                ax.fill_between(
                    ya,
                    data,
                    kc_data,
                    where=(data > kc_data),
                    alpha=0.15,
                    color="red",
                    label="K > K_crit",
                )
                # Mark crossing point
                exceed = data > kc_data
                if exceed.any():
                    cross_idx = exceed.argmax()
                    ax.axvline(x=ya[cross_idx], color="red", ls=":", lw=1.0, alpha=0.5)

            ax.set_ylabel(yl)
            ax.set_ylim(ylim)
            ax.set_title(f"{name}: {ylabel}", fontweight="bold")
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=7)
            ax.xaxis.set_major_formatter(
                FuncFormatter(
                    lambda x, _: f"{abs(int(x))} BCE" if x < 0 else str(int(x))
                )
            )
            ax.tick_params(axis="x", rotation=45, labelsize=8)

    plt.suptitle(
        "EDAP Model v3.1: Historical Data with Critical Threshold\n"
        "No absolute T prediction — model predicts direction and K-crossing, not level",
        fontsize=16,
        fontweight="bold",
        y=1.01,
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"  -> Saved to {save_path}")
