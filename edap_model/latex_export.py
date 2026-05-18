"""Generate LaTeX tables from results.json. No hardcoded civilization names."""

import json
import os


def generate_latex_tables(results_json_path="output/results.json", output_dir="output"):
    with open(results_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    os.makedirs(output_dir, exist_ok=True)
    civs = data["civilizations"]

    # ============================================================
    # Table 1: Parameters
    # ============================================================
    params = data["parameters"]

    param_names = {
        "alpha_T": r"$\alpha_T$",
        "beta_T": r"$\beta_T$",
        "gamma_T": r"$\gamma_T$",
        "T_min": r"$T_{\min}$",
        "alpha_K_base": r"$\alpha_K^{\text{base}}$",
        "beta_K": r"$\beta_K$",
        "K_max_coeff": r"$\kappa$",
        "K_half": r"$T_h$",
        "T_automation": r"$T_{\text{auto}}$",
        "alpha_C": r"$\alpha_C$",
        "gamma_C": r"$\gamma_C$",
        "eta": r"$\eta$",
        "K0": r"$K_0$",
        "delta": r"$\delta$",
        "sigma": r"$\sigma$",
        "T_singularity": r"$T_{\text{sing}}$",
        "epsilon_tech": r"$\epsilon_{\text{tech}}$",
        "K_shadow_threshold": r"$K_{\text{shadow}}$",
        "lambda_shadow": r"$\lambda_{\text{shadow}}$",
        "mu_shadow": r"$\mu_{\text{shadow}}$",
        "historical_T_peak": r"$T_{\text{peak}}$",
        "decay_factor": "decay factor",
        "recovery_potential_0": "rec. pot. init.",
        "T_reset_threshold": r"$T_{\text{reset}}$",
        "sigma_T": r"$\sigma_T$",
        "sigma_K": r"$\sigma_K$",
        "delta_T_reset": r"$\Delta T_{\text{reset}}$",
        "delta_K_reset": r"$\Delta K_{\text{reset}}$",
        "delta_C_reset": r"$\Delta C_{\text{reset}}$",
        "alpha_recovery_base": r"$\alpha_{\text{recovery}}$",
        "epsilon_recovery": r"$\epsilon_{\text{recovery}}$",
    }

    with open(
        os.path.join(output_dir, "table_parameters.tex"), "w", encoding="utf-8"
    ) as f:
        f.write(r"\begin{table}[H]" + "\n")
        f.write(r"\centering" + "\n")
        f.write(r"\footnotesize" + "\n")
        f.write(r"\caption{Model parameters (v4.0)}" + "\n")
        f.write(r"\label{tab:params}" + "\n")
        f.write(r"\resizebox{\textwidth}{!}{" + "\n")
        f.write(r"\begin{tabular}{lcl}" + "\n")
        f.write(r"\toprule" + "\n")
        f.write(r"Parameter & Symbol & Value \\" + "\n")
        f.write(r"\midrule" + "\n")
        for key, value in params.items():
            symbol = param_names.get(key, key)
            safe_key = key.replace('_', '\\_')
            f.write(f"{symbol} & {safe_key} & {value} \\\\\n")
        f.write(r"\bottomrule" + "\n")
        f.write(r"\end{tabular}" + "\n")
        f.write(r"}" + "\n")
        f.write(r"\end{table}" + "\n")
    print(f"  -> {output_dir}/table_parameters.tex")

    # ============================================================
    # Table 2: Turn accuracy
    # ============================================================
    with open(
        os.path.join(output_dir, "table_turn_accuracy.tex"), "w", encoding="utf-8"
    ) as f:
        f.write(r"\begin{table}[htbp]" + "\n")
        f.write(r"\centering" + "\n")
        f.write(r"\caption{Turn accuracy by civilization}" + "\n")
        f.write(r"\label{tab:turns}" + "\n")
        f.write(r"\begin{tabular}{lcccc}" + "\n")
        f.write(r"\toprule" + "\n")
        f.write(
            r"Civilization & Role & Turns correct & Total turns & Accuracy \\" + "\n"
        )
        f.write(r"\midrule" + "\n")
        for name, civ in civs.items():
            wm = civ.get("working_metrics", {})
            td = wm.get("turn_details", "")
            parts = td.replace(" turns correct", "").split("/")
            correct = parts[0] if len(parts) == 2 else "--"
            total = parts[1] if len(parts) == 2 else "--"
            acc = wm.get("turn_accuracy")
            acc_str = f"{acc:.3f}" if isinstance(acc, (int, float)) else "--"
            role = civ.get("role", "--")
            f.write(f"{name} & {role} & {correct} & {total} & {acc_str} \\\\\n")
        f.write(r"\midrule" + "\n")
        summary = data.get("metrics_summary", {})
        for role in ["train", "test"]:
            rd = summary.get(role, {})
            acc = rd.get("avg_turn_accuracy")
            acc_str = f"{acc:.3f}" if isinstance(acc, (int, float)) else "--"
            label = "Training average" if role == "train" else "Test average"
            f.write(f"{label} & -- & -- & -- & {acc_str} \\\\\n")
        f.write(r"\bottomrule" + "\n")
        f.write(r"\end{tabular}" + "\n")
        f.write(r"\end{table}" + "\n")
    print(f"  -> {output_dir}/table_turn_accuracy.tex")

    # ============================================================
    # Table 3: Working metrics
    # ============================================================
    with open(
        os.path.join(output_dir, "table_metrics.tex"), "w", encoding="utf-8"
    ) as f:
        f.write(r"\begin{table}[htbp]" + "\n")
        f.write(r"\centering" + "\n")
        f.write(r"\caption{Working metrics by civilization}" + "\n")
        f.write(r"\label{tab:metrics}" + "\n")
        f.write(r"\begin{tabular}{lcccc}" + "\n")
        f.write(r"\toprule" + "\n")
        f.write(
            r"Civilization & Turn accuracy & Brier skill score & K-crossing lead (yrs) & Final $T$ \\"
            + "\n"
        )
        f.write(r"\midrule" + "\n")
        for name, civ in civs.items():
            wm = civ.get("working_metrics", {})
            ta = wm.get("turn_accuracy")
            bss = wm.get("brier_skill_score")
            lead = wm.get("k_crossing_lead_years")
            final = civ.get("simulation_final_state", {}).get("T")
            ta_str = f"{ta:.3f}" if isinstance(ta, (int, float)) else "--"
            bss_str = f"{bss:.3f}" if isinstance(bss, (int, float)) else "--"
            lead_str = f"{lead:.0f}" if isinstance(lead, (int, float)) else "--"
            final_str = f"{final:.3f}" if isinstance(final, (int, float)) else "--"
            f.write(f"{name} & {ta_str} & {bss_str} & {lead_str} & {final_str} \\\\\n")
        f.write(r"\bottomrule" + "\n")
        f.write(r"\end{tabular}" + "\n")
        f.write(r"\end{table}" + "\n")
    print(f"  -> {output_dir}/table_metrics.tex")

    # ============================================================
    # Table 4: Turning points
    # ============================================================
    for name, civ in civs.items():
        dps = civ.get("data_points", [])
        if len(dps) < 5:
            continue

        years = [p["year"] for p in dps]
        T_vals = [p["T"] for p in dps]

        # Find turning points in data
        turns = []
        for i in range(1, len(T_vals) - 1):
            d1 = T_vals[i] - T_vals[i - 1]
            d2 = T_vals[i + 1] - T_vals[i]
            if d1 * d2 < 0:  # sign change = turning point
                direction = (
                    r"$\nearrow \rightarrow \searrow$"
                    if d1 > 0
                    else r"$\searrow \rightarrow \nearrow$"
                )
                turns.append((years[i], direction))

        if not turns:
            continue

        filename = f'table_turns_{name.lower().replace(" ", "_")}.tex'
        with open(os.path.join(output_dir, filename), "w", encoding="utf-8") as f:
            f.write(r"\begin{table}[htbp]" + "\n")
            f.write(r"\centering" + "\n")
            f.write(r"\caption{Turning points: " + name + "}" + "\n")
            f.write(r"\label{tab:turns_" + name.lower().replace(" ", "_") + "}" + "\n")
            f.write(r"\begin{tabular}{cc}" + "\n")
            f.write(r"\toprule" + "\n")
            f.write(r"Year & Direction \\" + "\n")
            f.write(r"\midrule" + "\n")
            for year, direction in turns:
                f.write(f"{year} & {direction} \\\\\n")
            f.write(r"\bottomrule" + "\n")
            f.write(r"\end{tabular}" + "\n")
            f.write(r"\end{table}" + "\n")
        print(f"  -> {output_dir}/{filename}")


if __name__ == "__main__":
    generate_latex_tables()
