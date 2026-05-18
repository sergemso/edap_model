"""Figure 5: Summary table."""

import numpy as np
import matplotlib.pyplot as plt
from edap_model.utils import load_json, DATA_DIR


def figure5_summary_table(save_path):
    print("  Figure 5: Summary table...")
    data = load_json(DATA_DIR + '/summary_table.json')
    civs = data['civilizations']
    names = [c['name'] for c in civs]
    Kv = [c['K_peak'] for c in civs]
    Kcv = [c['K_crit_at_T_peak'] for c in civs]
    outcomes = [c['outcome'] for c in civs]
    fig, ax = plt.subplots(figsize=(14, 8))
    xp = np.arange(len(names))
    w = 0.35
    bc = ['#d9534f' if k > kc else '#5cb85c' for k, kc in zip(Kv, Kcv)]
    ax.bar(xp - w / 2, Kv, w, label='Actual K', color=bc, alpha=0.75, edgecolor='black', lw=0.5)
    ax.bar(xp + w / 2, Kcv, w, label='K_crit', color='gray', alpha=0.4, hatch='//', edgecolor='black', lw=0.5)
    for i, (k, kc) in enumerate(zip(Kv, Kcv)):
        if k > kc:
            ax.annotate('\u26A0', (xp[i], max(k, kc) + 0.02), ha='center', fontsize=14, color='red')
    ax.set_xticks(xp)
    ax.set_xticklabels(names, rotation=45, ha='right', fontsize=9, fontfamily='sans-serif')
    ax.set_ylabel('Index Value')
    ax.set_ylim([0, 1.0])
    ax.set_title('EDAP Model v3.1: Civilizational Risk Assessment', fontweight='bold', fontsize=14)
    ax.legend(loc='upper left')
    ax.grid(True, axis='y', alpha=0.3)
    for i, oc in enumerate(outcomes):
        ax.text(xp[i], -0.08, oc, ha='center', fontsize=7, transform=ax.get_xaxis_transform(),
                fontweight='bold', fontfamily='sans-serif',
                color='darkred' if any(w in oc for w in ['Collapse', 'RISK', 'TRAPPED']) else 'darkgreen')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"  -> Saved to {save_path}")