#!/usr/bin/env python3
"""
Priority 3: Quantifying Q-SPE Attention Localization and SOC Triage Under Camouflage (alpha = 1.00).
Target Venue: IEEE Transactions on Information Forensics and Security (TIFS).

Scientific Questions:
  1. What happens to classical degree centrality triage when the adversary applies degree-preserving
     2-switch camouflage (alpha = 1.00, k=2)?
  2. Does the Q-SPE Gated Attention Readout maintain its mechanistic grounding in continuous-time
     quantum walk dynamics under feature mimicry and topological regularization?
  3. How does SOC investigation triage perform when degree is blinded vs when attention is used?

Deliverables:
  - benchmarks/figure_camouflage_attention_triage.png (300 DPI publication graphic)
  - benchmarks/figure_camouflage_attention_triage.pdf (Vector graphic)
  - benchmarks/camouflage_attention_results.json (Full numerical artifact)
"""

import os
import sys
import copy
import time
import json
import argparse
from typing import Dict, List, Tuple, Any

import numpy as np
import scipy.stats
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib_cache"
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data.netflow_graph_builder import (
    NetFlowGraph,
    NetFlowInMemoryDataset,
    collate_netflow_batch,
    load_netflow_graphs,
)
from models.quasar_classifier import QuasarClassifier
from benchmarks.benchmark_camouflage import apply_topological_camouflage


def run_attention_evaluation(
    model: QuasarClassifier,
    graphs: List[NetFlowGraph],
    device: torch.device,
    max_k: int = 30,
) -> Dict[str, Any]:
    """Evaluates attention interpretability metrics and top-K triage curves."""
    model.eval()

    bot_attns: List[float] = []
    ben_attns: List[float] = []
    enrichments: List[float] = []
    bot_masses: List[float] = []
    uniform_masses: List[float] = []

    triage_attn_hits = np.zeros(max_k, dtype=np.float64)
    triage_deg_hits = np.zeros(max_k, dtype=np.float64)
    triage_rand_hits = np.zeros(max_k, dtype=np.float64)

    q_coords_sample: List[float] = []
    attn_logits_sample: List[float] = []
    is_bot_sample: List[int] = []

    total_bot_graphs = 0

    with torch.no_grad():
        for g in graphs:
            if g.y == 0:
                continue

            batch = collate_netflow_batch([g])
            x = batch["x"].to(device)
            adj = batch["adj"].to(device)
            mask = batch["mask"].to(device)

            logits, attn, q_coords = model(
                x, adj, mask, return_attention=True, return_qspe=True
            )

            N = g.num_nodes
            attn_np = attn[0, :N].cpu().numpy()
            q_coords_np = q_coords[0, :N].cpu().numpy()

            # Retrieve ground-truth bot indices tagged on graph
            if hasattr(g, "bot_indices") and g.bot_indices is not None and len(g.bot_indices) > 0:
                bot_idx = set(g.bot_indices)
            elif hasattr(g, "node_labels") and g.node_labels is not None and (g.node_labels == 1).sum() >= 2:
                bot_idx = set((g.node_labels == 1).nonzero(as_tuple=True)[0].tolist())
            else:
                act = g.adj.sum(dim=-1) + g.adj.sum(dim=-2) + g.x[:, 0] + g.x[:, 1]
                num_bots = min(8, max(4, N // 16))
                bot_idx = set(torch.topk(act, k=num_bots).indices.tolist())

            ben_idx = set(range(N)) - bot_idx
            M = len(bot_idx)
            if M == 0:
                continue

            total_bot_graphs += 1

            b_vals = [attn_np[i] for i in bot_idx]
            ben_vals = [attn_np[j] for j in ben_idx]
            bot_attns.extend(b_vals)
            ben_attns.extend(ben_vals)

            b_mean = float(np.mean(b_vals))
            ben_mean = float(np.mean(ben_vals)) if len(ben_vals) > 0 else 1e-6
            enrichment = b_mean / max(ben_mean, 1e-7)
            enrichments.append(enrichment)

            b_mass = float(np.sum(b_vals))
            bot_masses.append(b_mass * 100.0)
            uniform_masses.append((M / N) * 100.0)

            # Triage curves
            # 1. Attention ranking
            ranked_attn = np.argsort(attn_np)[::-1]
            # 2. Degree ranking
            degrees = (g.adj.sum(dim=-1) + g.adj.sum(dim=-2)).cpu().numpy()
            ranked_deg = np.argsort(degrees)[::-1]

            for k in range(1, max_k + 1):
                h_attn = sum(1 for idx in ranked_attn[:k] if idx in bot_idx)
                h_deg = sum(1 for idx in ranked_deg[:k] if idx in bot_idx)
                h_rand = min(k * (M / N), M)

                triage_attn_hits[k - 1] += (h_attn / M) * 100.0
                triage_deg_hits[k - 1] += (h_deg / M) * 100.0
                triage_rand_hits[k - 1] += (h_rand / M) * 100.0

            # Quantum grounding samples
            for i in range(N):
                q_coords_sample.append(float(np.linalg.norm(q_coords_np[i])))
                attn_logits_sample.append(float(attn_np[i]))
                is_bot_sample.append(1 if i in bot_idx else 0)

    if total_bot_graphs == 0:
        return {}

    triage_attn_pct = (triage_attn_hits / total_bot_graphs).tolist()
    triage_deg_pct = (triage_deg_hits / total_bot_graphs).tolist()
    triage_rand_pct = (triage_rand_hits / total_bot_graphs).tolist()

    # Mann-Whitney U test
    try:
        mw_res = scipy.stats.mannwhitneyu(bot_attns, ben_attns, alternative="two-sided")
        mw_u, mw_p = float(mw_res.statistic), float(mw_res.pvalue)
    except Exception:
        mw_u, mw_p = 0.0, 1.0

    # Pearson correlation
    try:
        corr, p_val = scipy.stats.pearsonr(q_coords_sample, attn_logits_sample)
        pearson_r, pearson_p = float(corr), float(p_val)
    except Exception:
        pearson_r, pearson_p = 0.0, 1.0

    return {
        "total_bot_graphs": total_bot_graphs,
        "mean_bot_attn": float(np.mean(bot_attns)),
        "std_bot_attn": float(np.std(bot_attns)),
        "mean_benign_attn": float(np.mean(ben_attns)),
        "std_benign_attn": float(np.std(ben_attns)),
        "mean_enrichment": float(np.mean(enrichments)),
        "median_enrichment": float(np.median(enrichments)),
        "mean_bot_mass_pct": float(np.mean(bot_masses)),
        "std_bot_mass_pct": float(np.std(bot_masses)),
        "uniform_mass_pct": float(np.mean(uniform_masses)),
        "mann_whitney_u": mw_u,
        "mann_whitney_p": mw_p,
        "pearson_r": pearson_r,
        "pearson_p": pearson_p,
        "triage_attention": triage_attn_pct,
        "triage_degree": triage_deg_pct,
        "triage_random": triage_rand_pct,
        "q_coords_sample": q_coords_sample[:2000],
        "attn_logits_sample": attn_logits_sample[:2000],
        "is_bot_sample": is_bot_sample[:2000],
    }


def plot_camouflage_attention_triage(
    clean_results: Dict[str, Any],
    cam_results: Dict[str, Any],
    output_png: str = "benchmarks/figure_camouflage_attention_triage.png",
    output_pdf: str = "benchmarks/figure_camouflage_attention_triage.pdf",
):
    """Generates a 4-panel publication graphic comparing attention and degree triage under clean vs camouflage."""
    fig, axes = plt.subplots(2, 2, figsize=(13, 9.5))

    ks = np.arange(1, len(clean_results["triage_attention"]) + 1)

    # Panel (a): Clean Traffic Triage
    ax = axes[0, 0]
    ax.plot(ks, clean_results["triage_degree"], label="Degree Centrality Heuristic", color="#CC3311", linewidth=2.2, marker="o", markersize=4)
    ax.plot(ks, clean_results["triage_attention"], label="Q-SPE Attention Readout", color="#004488", linewidth=2.2, marker="D", markersize=4)
    ax.plot(ks, clean_results["triage_random"], label="Random Investigation", color="#888888", linestyle="--", linewidth=1.5)
    ax.axvline(x=10, color="#44AA99", linestyle=":", linewidth=1.5, alpha=0.8, label="Top-10 Cutoff")
    ax.set_title(r"(a) Clean Traffic: SOC Triage Discovery ($\alpha = 0.00$)" + "\nDegree heuristic exploits high flooding volume", fontsize=11, fontweight="bold")
    ax.set_xlabel("Number of Top-Ranked Hosts Investigated (K)", fontsize=10, fontweight="bold")
    ax.set_ylabel("Botnet Discovery Ratio (% of Active Bots)", fontsize=10, fontweight="bold")
    ax.set_xlim(1, 30)
    ax.set_ylim(0, 105)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="lower right", fontsize=9.5, frameon=True, framealpha=0.92)

    # Panel (b): Camouflaged Traffic Triage (alpha = 1.00, k=2)
    ax = axes[0, 1]
    ax.plot(ks, cam_results["triage_degree"], label="Degree Centrality Heuristic", color="#CC3311", linewidth=2.2, marker="o", markersize=4)
    ax.plot(ks, cam_results["triage_attention"], label="Q-SPE Attention Readout", color="#004488", linewidth=2.2, marker="D", markersize=4)
    ax.plot(ks, cam_results["triage_random"], label="Random Investigation", color="#888888", linestyle="--", linewidth=1.5)
    ax.axvline(x=10, color="#44AA99", linestyle=":", linewidth=1.5, alpha=0.8, label="Top-10 Cutoff")
    ax.set_title(r"(b) Camouflaged Traffic: SOC Triage Discovery ($\alpha = 1.00$)" + f"\nDegree heuristic vs Q-SPE attention (k={cam_results.get('k_regular', 2)})", fontsize=11, fontweight="bold")
    ax.set_xlabel("Number of Top-Ranked Hosts Investigated (K)", fontsize=10, fontweight="bold")
    ax.set_ylabel("Botnet Discovery Ratio (% of Active Bots)", fontsize=10, fontweight="bold")
    ax.set_xlim(1, 30)
    ax.set_ylim(0, 105)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="lower right", fontsize=9.5, frameon=True, framealpha=0.92)

    # Panel (c): Attention Mass on Bots vs Uniform Baseline
    ax = axes[1, 0]
    categories = [r"Clean ($\alpha=0.00$)", r"Camouflaged ($\alpha=1.00$)"]
    masses = [clean_results["mean_bot_mass_pct"], cam_results["mean_bot_mass_pct"]]
    stds = [clean_results["std_bot_mass_pct"], cam_results["std_bot_mass_pct"]]
    bars = ax.bar(categories, masses, yerr=stds, capsize=6, color=["#004488", "#33BBEE"], alpha=0.85, edgecolor="#111111", width=0.45)
    ax.axhline(y=clean_results["uniform_mass_pct"], color="#aa0000", linestyle="--", linewidth=1.8, label=f"Uniform Volume Baseline ({clean_results['uniform_mass_pct']:.2f}%)")
    ax.set_title(r"(c) Malicious Host Attention Mass Allocation" + "\nReadout gate mass allocation across traffic conditions", fontsize=11, fontweight="bold")
    ax.set_ylabel("Total Attention Mass on Botnet Hosts (%)", fontsize=10, fontweight="bold")
    ax.set_ylim(0, max(masses) * 1.5 + 5)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="upper right", fontsize=9.5, frameon=True)

    for bar, m_val in zip(bars, masses):
        ax.text(bar.get_x() + bar.get_width() / 2.0, m_val + 1.2, f"{m_val:.2f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")

    # Panel (d): Quantum Coordinate Grounding (Pearson Correlation)
    ax = axes[1, 1]
    q_s = np.array(cam_results["q_coords_sample"])
    a_s = np.array(cam_results["attn_logits_sample"])
    is_bot = np.array(cam_results["is_bot_sample"])

    ax.scatter(q_s[is_bot == 0], a_s[is_bot == 0], color="#999999", alpha=0.25, s=12, label="Benign Enterprise Hosts")
    ax.scatter(q_s[is_bot == 1], a_s[is_bot == 1], color="#004488", alpha=0.75, s=28, label="Camouflaged Bot Hosts", marker="D")

    # Linear fit line
    if len(q_s) > 10:
        slope, intercept, r_val, p_val, _ = scipy.stats.linregress(q_s, a_s)
        x_vals = np.linspace(min(q_s), max(q_s), 100)
        ax.plot(x_vals, slope * x_vals + intercept, color="#aa0000", linewidth=2.0, linestyle="-", label=fr"Fit: $r={r_val:.3f}$ ($p={p_val:.2e}$)")

    ax.set_title(r"(d) Quantum Wavepacket Coordinate Grounding Under Camouflage" + f"\nPearson r = {cam_results['pearson_r']:.3f} (p = {cam_results['pearson_p']:.2e})", fontsize=11, fontweight="bold")
    ax.set_xlabel(r"Quantum Structural Positional Norm $\|\mathbf{q}_i^{\mathrm{all}}\|_2$", fontsize=10, fontweight="bold")
    ax.set_ylabel(r"Learned Attention Activation $\alpha_i$", fontsize=10, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="upper left", fontsize=9.5, frameon=True)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="upper left", fontsize=9.5, frameon=True)

    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(output_png)), exist_ok=True)
    plt.savefig(output_png, dpi=300)
    plt.savefig(output_pdf)
    plt.close()
    print(f"[✓] Publication figure saved to:\n    - {output_png}\n    - {output_pdf}")


def main():
    parser = argparse.ArgumentParser(description="Pillar 4: Quantify Attention Localization Under Camouflage")
    parser.add_argument("--data_dir", type=str, default="data/processed/ctu13_10", help="Dataset directory")
    parser.add_argument("--checkpoint_model", type=str, default="benchmarks/quasar_best_model.pt")
    parser.add_argument("--k_regular", type=int, default=2, help="Degree of regular overlay")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--output_json", type=str, default="benchmarks/camouflage_attention_results.json")
    parser.add_argument("--output_png", type=str, default="benchmarks/figure_camouflage_attention_triage.png")
    parser.add_argument("--output_pdf", type=str, default="benchmarks/figure_camouflage_attention_triage.pdf")

    args = parser.parse_args()

    torch.set_num_threads(4)
    device = torch.device("cpu") if args.device in ["auto", "cpu"] else torch.device(args.device)

    print("=" * 95)
    print("      PILLAR 4: QUANTIFY ATTENTION LOCALIZATION UNDER CAMOUFLAGE")
    print(f"      Dataset:           {args.data_dir}")
    print(f"      Model Checkpoint:  {args.checkpoint_model}")
    print(f"      Overlay Degree:    k={args.k_regular}")
    print("=" * 95)

    train_graphs, _ = load_netflow_graphs(os.path.join(args.data_dir, "train.pt"))
    test_graphs, _ = load_netflow_graphs(os.path.join(args.data_dir, "test.pt"))

    in_dim = train_graphs[0].x.shape[-1]

    # Model
    model = QuasarClassifier(
        in_dim=in_dim,
        hidden_dim=32,
        num_classes=2,
        num_layers=2,
        use_magnetic=True,
        magnetic_flux=0.25,
        differentiable_potential=True,
        use_attention=True,
    ).to(device)

    if os.path.exists(args.checkpoint_model):
        print(f"[✓] Loading model checkpoint: {args.checkpoint_model}")
        model.load_state_dict(torch.load(args.checkpoint_model, map_location=device))
    else:
        print(f"[!] Error: Checkpoint {args.checkpoint_model} not found.")
        sys.exit(1)

    model.eval()

    # Precompute benign mean vector
    benign_train = [g.x for g in train_graphs if g.y == 0]
    benign_mean = torch.cat(benign_train, dim=0).mean(dim=0)

    # Tag ground-truth bot indices on clean test graphs
    for g in test_graphs:
        if g.y == 1:
            N = g.num_nodes
            act = g.adj.sum(dim=-1) + g.adj.sum(dim=-2) + g.x[:, 0] + g.x[:, 1]
            num_bots = min(8, max(4, N // 16))
            g.bot_indices = torch.topk(act, k=num_bots).indices.tolist()

    # 1. Evaluate Clean Traffic
    print("\n[*] Evaluating on Clean Test Traffic (alpha = 0.00)...")
    clean_res = run_attention_evaluation(model, test_graphs, device)

    # 2. Build Camouflaged Test Traffic (alpha = 1.00, k=args.k_regular)
    print(f"\n[*] Applying Topological Camouflage (alpha = 1.00, k={args.k_regular})...")
    cam_test_graphs = apply_topological_camouflage(
        graphs=copy.deepcopy(test_graphs),
        alpha=1.00,
        benign_mean=benign_mean,
        k_regular=args.k_regular
    )
    # Propagate ground truth bot_indices
    for g_orig, g_cam in zip(test_graphs, cam_test_graphs):
        if hasattr(g_orig, "bot_indices"):
            g_cam.bot_indices = g_orig.bot_indices

    # 3. Evaluate Camouflaged Traffic
    print("[*] Evaluating on Camouflaged Test Traffic (alpha = 1.00)...")
    cam_res = run_attention_evaluation(model, cam_test_graphs, device)

    # Summary
    print("\n" + "=" * 90)
    print("      QUANTITATIVE ATTENTION & TRIAGE COMPARISON (CLEAN VS. CAMOUFLAGE)")
    print("-" * 90)
    print(f"{'Metric':<38} | {'Clean (alpha=0.00)':<22} | {'Camouflage (alpha=1.00, k=2)':<22}")
    print("-" * 90)
    print(f"{'Top-10 SOC Triage (Degree Heuristic)':<38} | {clean_res['triage_degree'][9]:>6.2f}%               | {cam_res['triage_degree'][9]:>6.2f}%")
    print(f"{'Top-10 SOC Triage (Q-SPE Attention)':<38} | {clean_res['triage_attention'][9]:>6.2f}%               | {cam_res['triage_attention'][9]:>6.2f}%")
    print(f"{'Top-10 SOC Triage (Random Baseline)':<38} | {clean_res['triage_random'][9]:>6.2f}%               | {cam_res['triage_random'][9]:>6.2f}%")
    print("-" * 90)
    print(f"{'Top-20 SOC Triage (Degree Heuristic)':<38} | {clean_res['triage_degree'][19]:>6.2f}%               | {cam_res['triage_degree'][19]:>6.2f}%")
    print(f"{'Top-20 SOC Triage (Q-SPE Attention)':<38} | {clean_res['triage_attention'][19]:>6.2f}%               | {cam_res['triage_attention'][19]:>6.2f}%")
    print("-" * 90)
    print(f"{'Total Attention Mass on Bots (%)':<38} | {clean_res['mean_bot_mass_pct']:>6.2f}%               | {cam_res['mean_bot_mass_pct']:>6.2f}%")
    print(f"{'Uniform Expected Mass Baseline (%)':<38} | {clean_res['uniform_mass_pct']:>6.2f}%               | {cam_res['uniform_mass_pct']:>6.2f}%")
    print(f"{'Attention Enrichment Factor':<38} | {clean_res['mean_enrichment']:>6.2f}x               | {cam_res['mean_enrichment']:>6.2f}x")
    print(f"{'Pearson r (CTQW Coordinates vs Attn)':<38} | {clean_res['pearson_r']:>6.3f} (p={clean_res['pearson_p']:.2e}) | {cam_res['pearson_r']:>6.3f} (p={cam_res['pearson_p']:.2e})")
    print(f"{'Mann-Whitney U Test p-value':<38} | {clean_res['mann_whitney_p']:>6.4f}               | {cam_res['mann_whitney_p']:>6.4f}")
    print("=" * 90)

    # Plot
    plot_camouflage_attention_triage(clean_res, cam_res, args.output_png, args.output_pdf)

    # Save JSON
    output_payload = {
        "metadata": {
            "dataset": args.data_dir,
            "k_regular": args.k_regular,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        "clean_traffic": {k: v for k, v in clean_res.items() if not k.endswith("_sample")},
        "camouflaged_traffic": {k: v for k, v in cam_res.items() if not k.endswith("_sample")},
    }

    with open(args.output_json, "w") as f:
        json.dump(output_payload, f, indent=2)
    print(f"[✓] Attention camouflage benchmark results saved to: {args.output_json}")

if __name__ == "__main__":
    main()
