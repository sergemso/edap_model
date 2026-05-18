"""Figure 1: Phase portrait."""

import numpy as np
import matplotlib.pyplot as plt
from edap_model.model import EDAPModel
from edap_model.utils import load_json, RAW_DIR


def figure1_phase_portrait(normalizer, model_params, save_path):
    print("  Figure 1: Phase portrait...")
    model = EDAPModel(model_params.get('default_model', {}))
    model.set_seed(42)

    fig, ax = plt.subplots(figsize=(10, 8))
    Tr, Kr = np.linspace(0.02, 0.68, 30), np.linspace(0.02, 0.88, 30)
    Tg, Kg = np.meshgrid(Tr, Kr)
    dTg, dKg = np.zeros_like(Tg), np.zeros_like(Kg)

    osig = model.sigma
    model.sigma = 0.0
    orig_rp = model.cycles.recovery_potential
    model.cycles.recovery_potential = 1.0
    for i in range(30):
        for j in range(30):
            dT, dK, _ = model.dynamics(0, [Tg[i, j], Kg[i, j], 0.25])
            dTg[i, j] = dT
            dKg[i, j] = dK
    model.sigma = osig
    model.cycles.recovery_potential = orig_rp

    M = np.hypot(dTg, dKg)
    M[M == 0] = 1.0
    ax.streamplot(Tg, Kg, dTg / M, dKg / M, density=1.8, color='gray', linewidth=0.6)

    Tv = np.linspace(0, 0.7, 100)
    ax.plot(Tv, model.K_critical(Tv), 'r--', lw=2.5, label=r'$K_{crit}(T)$', zorder=5)
    ax.axvline(x=model.T_singularity, color='orange', ls=':', lw=1.5, alpha=0.7, label=r'$T_{sing}$')
    ax.axvline(x=model.T_min, color='gray', ls=':', lw=1, alpha=0.5, label=r'$T_{min}$')
    ax.axvline(x=model.T_automation, color='purple', ls=':', lw=1, alpha=0.5, label=r'$T_{auto}$')
    ax.fill_between(Tv, model.K_critical(Tv), 0.9, alpha=0.08, color='red')
    ax.fill_between(Tv, 0, model.K_critical(Tv), alpha=0.05, color='green')

    for name, cfg in model_params.get('civilizations', {}).items():
        if name in ['Russia', 'USA']:
            raw = load_json(RAW_DIR + f'/{name.lower()}_proxies.json')
            nd = normalizer.normalize_civilization(raw)
            T0, K0, C0 = nd[0]['T'], nd[0]['K'], nd[0]['C'] if nd else (cfg['T0'], cfg['K0'], cfg['C0'])
        else:
            T0, K0, C0 = cfg['T0'], cfg['K0'], cfg['C0']
        r = model.simulate(T0, K0, C0, t_span=cfg['t_span'], n_points=500)
        T, K = r['T'], r['K']
        ax.plot(T, K, color=cfg['color'], lw=2.2, label=cfg['label'], zorder=8)
        ax.scatter(T[0], K[0], color=cfg['color'], s=80, marker='o', zorder=15, edgecolors='white', lw=1.5)
        ax.scatter(T[-1], K[-1], color=cfg['color'], s=120, marker='X', zorder=15, edgecolors='black', lw=1.5)
        mid = len(T) // 2
        ax.annotate('', xy=(T[mid + 5], K[mid + 5]), xytext=(T[mid - 5], K[mid - 5]),
                    arrowprops=dict(arrowstyle='->', color=cfg['color'], lw=1.8))

    ax.set_xlabel('Technology Level (T)')
    ax.set_ylabel('Cannibalism Index (K)')
    ax.set_xlim([0, 0.7])
    ax.set_ylim([0, 0.9])
    ax.set_title('EDAP Model v3.1 Phase Portrait', fontweight='bold')
    ax.legend(loc='upper left', fontsize=7, framealpha=0.9)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"  -> Saved to {save_path}")