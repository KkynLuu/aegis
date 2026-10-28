#!/usr/bin/env python3
"""
Helper script to format the Multi-Degree Regularity Sweep (k in {2, 3, 4, 6})
into a publication-ready Markdown/LaTeX table for docs/manuscript_draft_tifs.md.
"""

import os
import json

def generate_regularity_table(json_path="benchmarks/regularity_sweep_ctu13_10_results.json"):
    if not os.path.exists(json_path):
        print(f"[!] {json_path} does not exist yet.")
        return None

    with open(json_path) as f:
        data = json.load(f)

    k_values = data["metadata"]["k_values"]
    models = ["GCN", "GIN", "RWPE-GIN", "MagNet", "QUASAR"]
    summary = data["summary"]
    stat_tests = data.get("statistical_tests", {})

    lines = []
    lines.append("**TABLE IV: Impact of Adversarial Overlay Regularity Degree $k \\in \\{2, 3, 4, 6\\}$ Under Full Camouflage ($\\alpha = 1.00$, CTU-13 Scenario 10 Across 10 Seeds)**\n")
    header_cols = [f"$k={k}$ Overlay" for k in k_values]
    lines.append("| Model | Clean ($\\alpha=0.00$) | " + " | ".join(header_cols) + " | Mean Evasion F1 [↑] |")
    lines.append("| :--- | :---: | " + " | ".join([":---:" for _ in k_values]) + " | :---: |")

    for m in models:
        if m not in summary:
            continue
        clean_mean = summary[m]["clean"]["f1_mean"]
        clean_std = summary[m]["clean"]["f1_std"]
        
        k_cells = []
        k_means = []
        for k in k_values:
            k_str = str(k)
            km = summary[m][k_str]["f1_mean"]
            ks = summary[m][k_str]["f1_std"]
            k_means.append(km)
            sig_str = ""
            if m != "QUASAR" and k_str in stat_tests and m in stat_tests[k_str]:
                sig = stat_tests[k_str][m].get("sig", "ns")
                if sig != "ns":
                    sig_str = f"^{{ {sig} }}"
            k_cells.append(f"${km:.2f} \\pm {ks:.2f}{sig_str}$")

        avg_k_mean = float(np_mean := sum(k_means) / len(k_means))

        if m == "QUASAR":
            m_label = f"**{m} (Ours)**"
            clean_str = f"**${clean_mean:.2f} \\pm {clean_std:.2f}$**"
            k_cells_str = [f"**{c}**" for c in k_cells]
            avg_str = f"**${avg_k_mean:.2f}\\%$**"
        else:
            m_label = f"**{m}**"
            clean_str = f"${clean_mean:.2f} \\pm {clean_std:.2f}$"
            k_cells_str = k_cells
            avg_str = f"${avg_k_mean:.2f}\\%$"

        lines.append(f"| {m_label} | {clean_str} | " + " | ".join(k_cells_str) + f" | {avg_str} |")

    return "\n".join(lines)

if __name__ == "__main__":
    tbl = generate_regularity_table()
    if tbl:
        print(tbl)
