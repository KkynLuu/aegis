#!/usr/bin/env python3
"""
Updates Section V-C and Tables II & III in docs/manuscript_draft_tifs.md
with the verified 10-seed Multi-Dataset Topological Camouflage results.
"""

import os
import json
import re

def generate_markdown_tables():
    datasets = [
        ("CTU-13 Scenario 10", "Sparse Decentralized P2P", "benchmarks/rq2_camouflage_ctu13_10_results.json"),
        ("NF-BoT-IoT", "IoT Distributed Telemetry", "benchmarks/rq2_camouflage_nf_bot_iot_results.json"),
        ("CTU-13 Scenario 11", "Centralized IRC Botnet", "benchmarks/rq2_camouflage_ctu13_11_results.json"),
        ("CTU-13 Scenario 9", "Dense Volumetric Spam", "benchmarks/rq2_camouflage_ctu13_9_results.json"),
    ]

    models_order = ["GCN", "GIN", "RWPE-GIN", "MagNet", "QUASAR"]

    # --- TABLE II: Multi-Dataset Macro-F1 across Alphas ---
    table2_lines = []
    table2_lines.append("**TABLE II: Multi-Dataset Topological Camouflage Stress Test: Macro-F1 Score (%) Across Increasing Evasion Intensity $\\alpha$ (10 Independent Seeds)**\n")
    table2_lines.append("| Dataset & Traffic Regime | Model | $\\alpha=0.00$ (Clean) | $\\alpha=0.25$ | $\\alpha=0.50$ (Rewired) | $\\alpha=0.75$ | $\\alpha=1.00$ (Full Camouflage) | $\\Delta_{\\text{F1}} (0.00 \\to 1.00)$ |")
    table2_lines.append("| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |")

    # --- TABLE III: Multi-Dataset Critical Evasion Threshold (Alpha = 1.00) ---
    table3_lines = []
    table3_lines.append("**TABLE III: Performance at Critical Evasion Threshold ($\\alpha = 1.00$: Full Feature Mimicry + Degree-Regular Overlay Across 10 Seeds)**\n")
    table3_lines.append("| Dataset & Traffic Regime | Model | Macro-F1 (%) [↑] | Precision (%) | Recall (%) | FAR (%) [↓] | ROC-AUC (%) | Wilcoxon vs QUASAR |")
    table3_lines.append("| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |")

    for ds_name, ds_desc, json_path in datasets:
        if not os.path.exists(json_path):
            print(f"Skipping {ds_name}: {json_path} not found.")
            continue

        with open(json_path) as f:
            data = json.load(f)

        summary = data["summary"]
        stat_tests = data.get("statistical_tests", {})

        # Table II entries
        m_rows_t2 = []
        for m in models_order:
            if m not in summary:
                continue
            f1_0 = summary[m]["0.00"]["f1_mean"]
            f1_0_std = summary[m]["0.00"]["f1_std"]
            f1_25 = summary[m]["0.25"]["f1_mean"]
            f1_25_std = summary[m]["0.25"]["f1_std"]
            f1_50 = summary[m]["0.50"]["f1_mean"]
            f1_50_std = summary[m]["0.50"]["f1_std"]
            f1_75 = summary[m]["0.75"]["f1_mean"]
            f1_75_std = summary[m]["0.75"]["f1_std"]
            f1_100 = summary[m]["1.00"]["f1_mean"]
            f1_100_std = summary[m]["1.00"]["f1_std"]
            delta = f1_100 - f1_0

            if m == "QUASAR":
                m_str = f"**{m} (Ours)**"
                f1_0_str = f"**${f1_0:.2f} \\pm {f1_0_std:.2f}$**"
                f1_100_str = f"**${f1_100:.2f} \\pm {f1_100_std:.2f}$**"
            else:
                m_str = f"**{m}**"
                f1_0_str = f"${f1_0:.2f} \\pm {f1_0_std:.2f}$"
                f1_100_str = f"${f1_100:.2f} \\pm {f1_100_std:.2f}$"

            m_rows_t2.append((
                m_str,
                f1_0_str,
                f"${f1_25:.2f} \\pm {f1_25_std:.2f}$",
                f"${f1_50:.2f} \\pm {f1_50_std:.2f}$",
                f"${f1_75:.2f} \\pm {f1_75_std:.2f}$",
                f1_100_str,
                f"${delta:+.2f}\\%$"
            ))

        ds_cell = f"**{ds_name}**<br>*( {ds_desc} )*"
        models_cell = "<br>".join([r[0] for r in m_rows_t2])
        f1_0_cell = "<br>".join([r[1] for r in m_rows_t2])
        f1_25_cell = "<br>".join([r[2] for r in m_rows_t2])
        f1_50_cell = "<br>".join([r[3] for r in m_rows_t2])
        f1_75_cell = "<br>".join([r[4] for r in m_rows_t2])
        f1_100_cell = "<br>".join([r[5] for r in m_rows_t2])
        delta_cell = "<br>".join([r[6] for r in m_rows_t2])

        table2_lines.append(f"| {ds_cell} | {models_cell} | {f1_0_cell} | {f1_25_cell} | {f1_50_cell} | {f1_75_cell} | {f1_100_cell} | {delta_cell} |")

        # Table III entries (Critical Alpha = 1.00)
        m_rows_t3 = []
        for m in models_order:
            if m not in summary:
                continue
            s100 = summary[m]["1.00"]
            f1 = s100["f1_mean"]
            f1_std = s100["f1_std"]
            pr = s100["precision_mean"]
            pr_std = s100["precision_std"]
            rc = s100["recall_mean"]
            rc_std = s100["recall_std"]
            far = s100["far_mean"]
            far_std = s100["far_std"]
            auc = s100["auc_mean"]
            auc_std = s100["auc_std"]

            if m == "QUASAR":
                sig_str = "—"
                m_str = f"**{m} (Ours)**"
                f1_str = f"**${f1:.2f} \\pm {f1_std:.2f}$**"
            else:
                m_str = f"**{m}**"
                f1_str = f"${f1:.2f} \\pm {f1_std:.2f}$"
                if "1.00" in stat_tests and m in stat_tests["1.00"]:
                    q_val = stat_tests["1.00"][m].get("bh_q", 1.0)
                    sig = stat_tests["1.00"][m].get("sig", "ns")
                    if sig == "***":
                        sig_str = f"$q = {q_val:.4f}^{{***}}$"
                    elif sig == "**":
                        sig_str = f"$q = {q_val:.4f}^{{**}}$"
                    elif sig == "*":
                        sig_str = f"$q = {q_val:.4f}^*$"
                    else:
                        sig_str = f"$q = {q_val:.4f}^{{\\text{{ns}}}}$"
                else:
                    sig_str = "—"

            m_rows_t3.append((
                m_str,
                f1_str,
                f"${pr:.2f} \\pm {pr_std:.2f}$",
                f"${rc:.2f} \\pm {rc_std:.2f}$",
                f"${far:.2f} \\pm {far_std:.2f}$",
                f"${auc:.2f} \\pm {auc_std:.2f}$",
                sig_str
            ))

        models_cell_t3 = "<br>".join([r[0] for r in m_rows_t3])
        f1_cell_t3 = "<br>".join([r[1] for r in m_rows_t3])
        pr_cell_t3 = "<br>".join([r[2] for r in m_rows_t3])
        rc_cell_t3 = "<br>".join([r[3] for r in m_rows_t3])
        far_cell_t3 = "<br>".join([r[4] for r in m_rows_t3])
        auc_cell_t3 = "<br>".join([r[5] for r in m_rows_t3])
        sig_cell_t3 = "<br>".join([r[6] for r in m_rows_t3])

        table3_lines.append(f"| {ds_cell} | {models_cell_t3} | {f1_cell_t3} | {pr_cell_t3} | {rc_cell_t3} | {far_cell_t3} | {auc_cell_t3} | {sig_cell_t3} |")

    return "\n".join(table2_lines), "\n".join(table3_lines)

if __name__ == "__main__":
    t2, t3 = generate_markdown_tables()
    print("=== UPDATED TABLE III ===")
    print(t3)
