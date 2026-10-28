#!/usr/bin/env python3
"""
Generates the publication-quality 4-panel Figure for Experiment 2 (RQ2):
Multi-Dataset Topological Camouflage Stress Test for IEEE TIFS.

Panels:
  (a) CTU-13 Scenario 10 (Sparse Decentralized P2P)
  (b) NF-BoT-IoT (IoT Botnet & Distributed Telemetry)
  (c) CTU-13 Scenario 9 (Dense Volumetric Spam)
  (d) CTU-13 Scenario 11 (Rbot IRC / Small Centralized C&C)
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATASETS = [
    {
        "name": "CTU-13 Scenario 10",
        "subtitle": "Sparse Decentralized P2P (Neris)",
        "json_path": "benchmarks/rq2_camouflage_ctu13_10_results.json",
        "panel": "(a)",
    },
    {
        "name": "NF-BoT-IoT",
        "subtitle": "IoT Botnet Telemetry (Bojarski et al.)",
        "json_path": "benchmarks/rq2_camouflage_nf_bot_iot_results.json",
        "panel": "(b)",
    },
    {
        "name": "CTU-13 Scenario 9",
        "subtitle": "Dense Volumetric Spam (Neris)",
        "json_path": "benchmarks/rq2_camouflage_ctu13_9_results.json",
        "panel": "(c)",
    },
    {
        "name": "CTU-13 Scenario 11",
        "subtitle": "Rbot Centralized C&C (Sparse Overlay)",
        "json_path": "benchmarks/rq2_camouflage_ctu13_11_results.json",
        "panel": "(d)",
    },
]

MODEL_STYLES = {
    "AEGIS":    {"color": "#AA3377", "marker": "P", "linewidth": 2.4, "markersize": 7, "zorder": 6, "label": "AEGIS (Ours)"},
    "QUASAR":   {"color": "#004488", "marker": "D", "linewidth": 2.0, "markersize": 6, "zorder": 5, "label": "QUASAR (Closed CTQW)"},
    "MagNet":   {"color": "#33BBEE", "marker": "s", "linewidth": 1.6, "markersize": 5, "zorder": 4, "label": "MagNet"},
    "GCN":      {"color": "#009988", "marker": "^", "linewidth": 1.6, "markersize": 5, "zorder": 3, "label": "GCN"},
    "GIN":      {"color": "#EE7733", "marker": "o", "linewidth": 1.6, "markersize": 5, "zorder": 2, "label": "GIN"},
    "RWPE-GIN": {"color": "#CC3311", "marker": "x", "linewidth": 1.6, "markersize": 6, "zorder": 2, "label": "RWPE-GIN"},
}

def plot_multidataset_camouflage(
    output_png: str = "benchmarks/figure_multidataset_camouflage.png",
    output_pdf: str = "benchmarks/figure_multidataset_camouflage.pdf"
):
    fig, axes = plt.subplots(2, 2, figsize=(13, 9.5), sharex=True, sharey=False)
    axes = axes.flatten()

    for idx, ds_info in enumerate(DATASETS):
        ax = axes[idx]
        path = ds_info["json_path"]

        if not os.path.exists(path):
            ax.text(0.5, 0.5, f"Data pending:\n{path}", ha="center", va="center", transform=ax.transAxes, color="gray")
            ax.set_title(f"{ds_info['panel']} {ds_info['name']}\n[{ds_info['subtitle']}]", fontsize=11, fontweight="bold")
            continue

        with open(path) as f:
            data = json.load(f)

        alphas = [float(a) for a in data["metadata"]["alphas"]]
        summary = data["summary"]

        for model_name, style in MODEL_STYLES.items():
            model_key = None
            if model_name in summary:
                model_key = model_name
            elif model_name == "AEGIS" and "AEGIS-MoE (Ours)" in summary:
                model_key = "AEGIS-MoE (Ours)"
            elif model_name == "QUASAR" and "Unitary CTQW (Closed)" in summary:
                model_key = "Unitary CTQW (Closed)"

            if not model_key:
                continue
            f1_means = [summary[model_key][f"{a:.2f}"]["f1_mean"] for a in alphas]
            f1_stds = [summary[model_key][f"{a:.2f}"]["f1_std"] for a in alphas]

            ax.plot(
                alphas,
                f1_means,
                color=style["color"],
                marker=style["marker"],
                linewidth=style["linewidth"],
                markersize=style["markersize"],
                label=style["label"],
                zorder=style["zorder"]
            )
            ax.fill_between(
                alphas,
                [m - s for m, s in zip(f1_means, f1_stds)],
                [m + s for m, s in zip(f1_means, f1_stds)],
                color=style["color"],
                alpha=0.12,
                zorder=style["zorder"] - 1
            )

        # Baseline 50% random guessing floor
        ax.axhline(y=50.0, color="#888888", linestyle="--", linewidth=1.0, alpha=0.6, zorder=1)
        ax.set_title(f"{ds_info['panel']} {ds_info['name']}\n({ds_info['subtitle']})", fontsize=11, fontweight="bold", pad=8)
        ax.set_ylabel("Macro-F1 Score (%)", fontsize=10, fontweight="bold")
        ax.set_xticks(alphas)
        ax.set_xticklabels([f"{a:.2f}" for a in alphas], fontsize=9.5)
        ax.grid(True, linestyle="--", alpha=0.45)
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(20, 100)

    # Set common X-label for bottom plots
    axes[2].set_xlabel(r"Camouflage Intensity $\alpha$ ($\mathbf{x} \to \boldsymbol{\mu}_{\text{benign}}$, $k$-regular overlay)", fontsize=10.5, fontweight="bold")
    axes[3].set_xlabel(r"Camouflage Intensity $\alpha$ ($\mathbf{x} \to \boldsymbol{\mu}_{\text{benign}}$, $k$-regular overlay)", fontsize=10.5, fontweight="bold")

    # Unified Legend at the top
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=5, fontsize=11, frameon=True, framealpha=0.95, edgecolor="#cccccc", bbox_to_anchor=(0.5, 0.99))

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    os.makedirs(os.path.dirname(os.path.abspath(output_png)), exist_ok=True)
    plt.savefig(output_png, dpi=300)
    plt.savefig(output_pdf)
    # Also synchronize to paper/figures/
    paper_fig_png = "paper/figures/figure_multidataset_camouflage.png"
    paper_fig_pdf = "paper/figures/figure_multidataset_camouflage.pdf"
    plt.savefig(paper_fig_png, dpi=300)
    plt.savefig(paper_fig_pdf)
    plt.close()
    print(f"[✓] Multi-dataset camouflage figure saved to:\n    - {output_png}\n    - {output_pdf}\n    - {paper_fig_png}\n    - {paper_fig_pdf}")

if __name__ == "__main__":
    plot_multidataset_camouflage()
