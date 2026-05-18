"""Figure 3: Bifurcation analysis."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from edap_model.model import EDAPModel


def figure3_bifurcation(model_params, save_path):
    print("  Figure 3: Bifurcation...")
    d = model_params.get('default_model', {})
    K0v, dd = d.get('K0', 0.55), d.get('delta', 0.25)
    aKd = d.get('alpha_K_base', 0.67)
    Tm, Kmc, Kh = d.get('T_min', 0.08), d.get('K_max_coeff', 0.95), d.get('K_half', 0.05)
    Ts, et = d.get('T_singularity', 0.65), d.get('epsilon_tech', 0.40)
    ls, ms = d.get('lambda_shadow', 0.10), d.get('mu_shadow', 0.15)
    Kst, Ta = d.get('K_shadow_threshold', 0.50), d.get('T_automation', 0.85)
    aT, bT, gT = d.get('alpha_T', 0.52), d.get('beta_T', 0.12), d.get('gamma_T', 0.20)
    bK, aC, gC, eta = d.get('beta_K', 0.18), d.get('alpha_C', 0.31), d.get('gamma_C', 0.20), d.get('eta', 0.40)
    np_ = 35
    ni = 20
    rng = np.random.RandomState(42)
    T0l, K0l, C0l = [], [], []
    while len(T0l) < ni:
        t0 = rng.uniform(0.02, 0.65)
        k0 = rng.uniform(0.02, 0.85)
        c0 = rng.uniform(0.02, 0.55)
        if t0 + k0 + c0 <= 0.7:
            T0l.append(t0)
            K0l.append(k0)
            C0l.append(c0)
    T0a = np.array(T0l)
    K0a = np.array(K0l)
    C0a = np.array(C0l)

    def Kcs(T, dv):
        return (K0v - dv * T) + et * np.maximum(T - Ts, 0.0) ** 2

    def Kms(T):
        b = Kmc * np.maximum(T, 0.0) / (np.maximum(T, 0.0) + Kh)
        tr = np.maximum(T - Ta, 0.0) / (1.0 - Ta)
        return b + (1.0 - b) * tr

    def iterate(T, K, C, aKv, dv, st=1200):
        for _ in range(st):
            T = np.clip(T, 0.01, 0.99)
            K = np.clip(K, 0.01, 0.99)
            C = np.clip(C, 0.01, 0.99)
            Kc = np.maximum(Kcs(T, dv), 0.01)
            Km = np.maximum(Kms(T), 0.01)
            Te = np.maximum(T - Tm, 0.001)
            aKe = aKv * (1.0 - C)
            lam = ls * np.maximum(0.0, 1.0 - T / Ta)
            dT = aT * Te * (1.0 - T) * C * (1 - K / Kc) - bT * (T - Tm) - gT * K * T + eta * 0.05 * (1 - T / 0.5)
            dK = aKe * K * (1 - T - C) * (1 - K / Km) - bK * (T * C) * K - lam * np.maximum(K - Kst, 0.0) * K
            mu = ms * np.maximum(0.0, 1.0 - T / Ta)
            dC = aC * (T - K) * C * (1 - C) + eta * 0.05 * (1 - C) + mu * np.maximum(K - Kst, 0.0) * (1 - C) - gC * K * C
            dC = np.where((C <= 0.011) & (dC < 0), 0.0, dC)
            md = np.maximum(np.maximum(np.abs(dT), np.abs(dK)), np.abs(dC))
            edt = np.where(md > 0.5, 0.025 / np.maximum(md, 0.01), 0.05)
            T += dT * edt
            K += dK * edt
            C += dC * edt
        return T, K

    fig = plt.figure(figsize=(18, 12))
    ax1 = fig.add_subplot(2, 2, 1)
    print("    Panel A: α_K sweep...")
    ar = np.linspace(0.02, 0.50, np_)
    T = np.tile(T0a, (np_, 1))
    K = np.tile(K0a, (np_, 1))
    C = np.tile(C0a, (np_, 1))
    T, K = iterate(T, K, C, ar[:, np.newaxis], dd)
    for i in range(ni):
        ax1.scatter(ar, T[:, i], s=2, alpha=0.35, color='blue', rasterized=True)
        ax1.scatter(ar, K[:, i], s=2, alpha=0.35, color='red', rasterized=True)
    ax1.scatter([], [], s=15, color='blue', label='Final T')
    ax1.scatter([], [], s=15, color='red', label='Final K')
    ax1.set_xlabel('α_K_base')
    ax1.set_ylabel('Steady State')
    ax1.set_title('A: α_K_base bifurcation', fontweight='bold')
    ax1.legend(fontsize=7)
    ax1.set_ylim([-0.05, 1.05])
    ax1.grid(True, alpha=0.3)

    ax2 = fig.add_subplot(2, 2, 2)
    print("    Panel B: δ sweep...")
    dr = np.linspace(0.02, 0.30, np_)
    T = np.tile(T0a, (np_, 1))
    K = np.tile(K0a, (np_, 1))
    C = np.tile(C0a, (np_, 1))
    T, K = iterate(T, K, C, aKd, dr[:, np.newaxis])
    for i in range(ni):
        ax2.scatter(dr, T[:, i], s=2, alpha=0.35, color='blue', rasterized=True)
        ax2.scatter(dr, K[:, i], s=2, alpha=0.35, color='red', rasterized=True)
    ax2.scatter([], [], s=15, color='blue', label='Final T')
    ax2.scatter([], [], s=15, color='red', label='Final K')
    ax2.set_xlabel('δ')
    ax2.set_title('B: δ bifurcation', fontweight='bold')
    ax2.legend(fontsize=7)
    ax2.set_ylim([-0.05, 1.05])
    ax2.grid(True, alpha=0.3)

    ax3 = fig.add_subplot(2, 2, 3)
    print("    Panel C: Phase portrait slices...")
    sc = [('green', 'α_K=0.05'), ('orange', 'α_K=0.12'), ('red', 'α_K=0.30')]
    for av, lb in [(0.05, sc[0][1]), (0.12, sc[1][1]), (0.30, sc[2][1])]:
        col = sc[[c[1] for c in sc].index(lb)][0]
        Tv, Kv = np.meshgrid(np.linspace(0.02, 0.68, 22), np.linspace(0.02, 0.88, 22))
        dTg = np.zeros_like(Tv)
        dKg = np.zeros_like(Kv)
        for i in range(22):
            for j in range(22):
                Tp, Kp = Tv[i, j], Kv[i, j]
                Te = max(Tp - Tm, 0.001)
                Kc = max(Kcs(Tp, dd), 0.01)
                Km = max(Kms(Tp), 0.01)
                aKe = av * (1.0 - 0.25)
                lam = ls * max(0.0, 1.0 - Tp / Ta)
                dTg[i, j] = aT * Te * (1.0 - Tp) * 0.25 * (1 - Kp / Kc) - bT * (Tp - Tm) - gT * Kp * Tp
                dKg[i, j] = aKe * Kp * (1 - Tp - 0.25) * (1 - Kp / Km) - bK * (Tp * 0.25) * Kp - lam * max(Kp - Kst, 0.0) * Kp
        M = np.hypot(dTg, dKg)
        M[M == 0] = 1.0
        ax3.streamplot(Tv, Kv, dTg / M, dKg / M, density=1.2, color=col, linewidth=0.8)
    le = [Line2D([0], [0], color=c, lw=2, label=l) for c, l in sc]
    Tv = np.linspace(0, 0.7, 100)
    ax3.plot(Tv, Kcs(Tv, dd), 'k--', lw=1.5, label='K_crit')
    for v, lb, c in [(Tm, 'T_min', 'gray'), (Ts, 'T_sing', 'orange'), (Ta, 'T_auto', 'purple')]:
        ax3.axvline(x=v, color=c, ls=':', lw=1.0, alpha=0.5)
        ax3.text(v + 0.01, 0.87, lb, fontsize=7, color=c, rotation=90)
    ax3.set_xlabel('Technology (T)')
    ax3.set_ylabel('Cannibalism (K)')
    ax3.set_xlim([0, 0.7])
    ax3.set_ylim([0, 0.9])
    ax3.set_title('C: Phase portraits (C=0.25)', fontweight='bold')
    ax3.legend(handles=le, fontsize=7, loc='upper right')
    ax3.grid(True, alpha=0.2)

    ax4 = fig.add_subplot(2, 2, 4)
    print("    Panel D: Heatmap...")
    ag = np.linspace(0.05, 0.45, 25)
    dg = np.linspace(0.05, 0.30, 25)
    st = np.zeros((len(dg), len(ag)))
    for i, dv in enumerate(dg):
        aa = ag[:, np.newaxis]
        T = np.tile(T0a[:10], (len(ag), 1))
        K = np.tile(K0a[:10], (len(ag), 1))
        C = np.tile(C0a[:10], (len(ag), 1))
        Tf, Kf = iterate(T, K, C, aa, dv)
        st[i, :] = np.mean(Tf > 0.3, axis=1)
    im = ax4.contourf(ag, dg, st, levels=np.linspace(0, 1, 11), cmap='RdYlGn', vmin=0, vmax=1)
    ax4.contour(ag, dg, st, levels=[0.5], colors='black', linewidths=1.5)
    ax4.set_xlabel('α_K_base')
    ax4.set_ylabel('δ')
    ax4.set_title('D: Basin stability', fontweight='bold')
    plt.colorbar(im, ax=ax4, label='Fraction sustainable')
    plt.suptitle('EDAP Model v3.1: Structural Stability', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=250)
    plt.close()
    print(f"  -> Saved to {save_path}")