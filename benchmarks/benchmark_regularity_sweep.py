#!/usr/bin/env python3
"""
Experiment: Multi-Degree Regularity Sweep (k in {2, 3, 4, 6}) Stress Test.
Target Venue: IEEE Transactions on Information Forensics and Security (TIFS).

Scientific Goal:
  Investigate whether GCN's high performance on CTU-13 Scenario 10 under k=2 camouflage
  is an accidental artifact of a simple cyclic spectral watermark, or a genuine general defense.
  We evaluate how detection performance scales as the adversary varies the regularity degree
  k in {2, 3, 4, 6} (from simple 2-regular cycle to 3-regular cubic Möbius graph,
  4-regular chordal ring, and 6-regular dense overlay) under full feature mimicry (alpha = 1.00).

Hypothesis:
  As k increases (k >= 3), the eigenvalues of the regular overlay become dense and quasi-continuous,
  destroying the isolated cyclic ring watermark lambda = 2 cos(2 pi j / M). Consequently, GCN's
  detection collapses toward random guessing, while QUASAR's unitary phase dynamics and
  Dirichlet energy floor sustain robust representation discriminability across all degrees k.
"""

import os
import sys
import copy
import time
import json
import argparse
from typing import List, Dict, Tuple, Optional, Any
import numpy as np
import scipy.stats
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"
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
from models.baseline_gnn import (
    GCNClassifier,
    GINClassifier,
    RWPE_GINClassifier,
    MagNetClassifier,
)
from models.quasar_classifier import QuasarClassifier
from benchmarks.benchmark_camouflage import (
    compute_roc_auc_numpy,
    evaluate_model_on_loader,
    apply_topological_camouflage,
    train_single_model,
)

MODEL_COLORS = {
    "AEGIS":    {"color": "#AA3377", "marker": "P", "linewidth": 2.4, "label": "AEGIS (Ours)"},
    "QUASAR":   {"color": "#004488", "marker": "D", "linewidth": 2.0, "label": "QUASAR (Closed CTQW)"},
    "MagNet":   {"color": "#33BBEE", "marker": "s", "linewidth": 1.6, "label": "MagNet"},
    "GCN":      {"color": "#009988", "marker": "^", "linewidth": 1.6, "label": "GCN"},
    "GIN":      {"color": "#EE7733", "marker": "o", "linewidth": 1.6, "label": "GIN"},
    "RWPE-GIN": {"color": "#CC3311", "marker": "x", "linewidth": 1.6, "label": "RWPE-GIN"},
}

def plot_regularity_sweep(
    k_values: List[int],
    summary: Dict[str, Dict[str, Dict[str, float]]],
    models: List[str],
    dataset_name: str,
    output_path: str
):
    """Plots Macro-F1 vs Regularity Degree k with shaded error bands."""
    plt.figure(figsize=(8.5, 5.5))

    for m in models:
        if m not in summary:
            continue
        cfg = MODEL_COLORS.get(m, {"color": "black", "marker": "o", "linewidth": 1.5, "label": m})
        means = [summary[m][str(k)]["f1_mean"] for k in k_values]
        stds = [summary[m][str(k)]["f1_std"] for k in k_values]

        plt.plot(
            k_values,
            means,
            label=cfg["label"],
            color=cfg["color"],
            marker=cfg["marker"],
            linewidth=cfg["linewidth"],
            markersize=7,
            zorder=5 if m == "QUASAR" else 3
        )
        plt.fill_between(
            k_values,
            [m_val - s_val for m_val, s_val in zip(means, stds)],
            [m_val + s_val for m_val, s_val in zip(means, stds)],
            color=cfg["color"],
            alpha=0.12,
            zorder=4 if m == "QUASAR" else 2
        )

    plt.axhline(y=50.0, color="#aa0000", linestyle="--", linewidth=1.0, alpha=0.6)
    plt.text(k_values[0] + 0.1, 50.8, "Random Guess Floor (50%)", fontsize=9, color="#aa0000", style="italic")

    plt.xlabel("Adversarial Overlay Regularity Degree ($k$)", fontsize=11, fontweight="bold")
    plt.ylabel("Detection Performance (Macro-F1 %)", fontsize=11, fontweight="bold")
    plt.title(f"Impact of Regularity Degree on Camouflage Evasion\nDataset: {dataset_name.upper()} ($\\alpha = 1.00$, 10-Seed Average $\\pm$ 1 Std)",
              fontsize=12, fontweight="bold", pad=12)
    plt.xticks(k_values, [f"k={k}" for k in k_values], fontsize=10)
    plt.yticks(np.arange(30, 101, 10), fontsize=10)
    plt.ylim(25, 98)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="lower left", fontsize=10, frameon=True, framealpha=0.92, edgecolor="#cccccc")

    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    pdf_path = output_path.replace(".png", ".pdf")
    plt.savefig(pdf_path)
    plt.close()
    print(f"[✓] Publication figure saved to:\n    - {output_path}\n    - {pdf_path}")


def run_regularity_sweep(
    dataset_name: str = "ctu13_10",
    data_dir: str = "data/processed/ctu13_10",
    k_values: List[int] = [2, 3, 4, 6],
    alpha: float = 1.00,
    models: List[str] = ["GCN", "GIN", "RWPE-GIN", "MagNet", "QUASAR"],
    seeds: List[int] = [42],
    epochs: int = 35,
    batch_size: int = 32,
    lr: float = 0.003,
    hidden_dim: int = 32,
    num_layers: int = 2,
    device: str = "cpu",
    output_json: Optional[str] = None,
    output_fig: Optional[str] = None,
):
    print("=" * 100)
    print("      EXPERIMENT: MULTI-DEGREE REGULARITY SWEEP (k in {2, 3, 4, 6})")
    print(f"      Dataset:           {dataset_name.upper()} ({data_dir})")
    print(f"      Models:            {', '.join(models)}")
    print(f"      Seeds ({len(seeds)}):        {seeds}")
    print(f"      Regularity k:      {k_values}")
    print(f"      Camouflage alpha:  {alpha:.2f} (Full Mimicry + k-Regular Rewiring)")
    print(f"      Device:            {device.upper()} | Epochs: {epochs} | Batch Size: {batch_size}")
    print("=" * 100)

    train_path = os.path.join(data_dir, "train.pt")
    val_path = os.path.join(data_dir, "val.pt")
    test_path = os.path.join(data_dir, "test.pt")

    train_graphs, _ = load_netflow_graphs(train_path)
    val_graphs, _ = load_netflow_graphs(val_path)
    clean_test_graphs, _ = load_netflow_graphs(test_path)

    in_dim = train_graphs[0].x.shape[1]

    # Compute empirical benign mean vector from training set
    benign_train_nodes = []
    for g in train_graphs:
        if g.y == 0:
            benign_train_nodes.append(g.x)
    if len(benign_train_nodes) > 0:
        benign_mean = torch.cat(benign_train_nodes, dim=0).mean(dim=0)
    else:
        benign_mean = torch.zeros(in_dim)

    # Class weights for loss
    num_benign = sum(1 for g in train_graphs if g.y == 0)
    num_malicious = sum(1 for g in train_graphs if g.y == 1)
    w_benign = 0.5 * (len(train_graphs) / max(num_benign, 1))
    w_malicious = 0.5 * (len(train_graphs) / max(num_malicious, 1))
    class_weights = torch.tensor([w_benign, w_malicious], dtype=torch.float32)

    train_loader = DataLoader(
        NetFlowInMemoryDataset(train_graphs),
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_netflow_batch
    )
    val_loader = DataLoader(
        NetFlowInMemoryDataset(val_graphs),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_netflow_batch
    )

    # Pre-build camouflaged test Loaders for each k
    k_test_loaders: Dict[str, DataLoader] = {}
    for k in k_values:
        cam_graphs = apply_topological_camouflage(
            clean_test_graphs,
            alpha=alpha,
            benign_mean=benign_mean,
            k_regular=k
        )
        k_test_loaders[str(k)] = DataLoader(
            NetFlowInMemoryDataset(cam_graphs),
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_netflow_batch
        )

    # Clean test loader (k="clean")
    clean_loader = DataLoader(
        NetFlowInMemoryDataset(clean_test_graphs),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_netflow_batch
    )

    metrics_list = ["f1", "precision", "recall", "far", "auc", "acc"]
    eval_keys = [str(k) for k in k_values] + ["clean"]

    raw_results: Dict[str, Dict[str, Dict[str, List[float]]]] = {
        m: {k_key: {metric: [] for metric in metrics_list} for k_key in eval_keys}
        for m in models
    }

    default_out_json = output_json or f"benchmarks/regularity_sweep_{dataset_name.lower()}_results.json"
    if os.path.exists(default_out_json):
        try:
            with open(default_out_json, "r") as f:
                prev_data = json.load(f)
            prev_raw = prev_data.get("raw_results", {})
            for m in models:
                if m in prev_raw and m != "AEGIS":
                    valid = True
                    for k_key in eval_keys:
                        if k_key not in prev_raw[m] or len(prev_raw[m][k_key].get("f1", [])) < len(seeds):
                            valid = False
                            break
                    if valid:
                        for k_key in eval_keys:
                            for metric in metrics_list:
                                raw_results[m][k_key][metric] = prev_raw[m][k_key][metric][:len(seeds)]
                        print(f"  [✓] Loaded cached baseline results for {m} from {default_out_json}")
        except Exception as e:
            print(f"  [!] Note: could not load cache: {e}")

    start_time = time.time()

    for seed_idx, seed in enumerate(seeds):
        print(f"\n--- [Seed {seed_idx + 1}/{len(seeds)}: {seed}] ---")
        for m in models:
            k0_key = eval_keys[0]
            if len(raw_results[m][k0_key]["f1"]) >= seed_idx + 1:
                cached_f1s = [f"{k_key}: {raw_results[m][k_key]['f1'][seed_idx]:.1f}%" for k_key in eval_keys]
                print(f"  {m:<10} | (Cached) | F1 across k [{', '.join(cached_f1s)}]")
                continue

            t0 = time.time()
            trained_model = train_single_model(
                model_name=m,
                in_dim=in_dim,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                train_loader=train_loader,
                val_loader=val_loader,
                class_weights=class_weights,
                epochs=epochs,
                lr=lr,
                device=device,
                seed=seed
            )
            t_train = time.time() - t0

            # Evaluate on clean
            res_clean = evaluate_model_on_loader(trained_model, clean_loader, device=device)
            for met in metrics_list:
                raw_results[m]["clean"][met].append(res_clean[met])

            # Evaluate across each k
            k_f1s = []
            for k in k_values:
                k_str = str(k)
                loader = k_test_loaders[k_str]
                res = evaluate_model_on_loader(trained_model, loader, device=device)
                for met in metrics_list:
                    raw_results[m][k_str][met].append(res[met])
                k_f1s.append(f"k={k}: {res['f1']:.1f}%")

            print(f"  {m:<10} | Train: {t_train:>5.2f}s | Clean: {res_clean['f1']:.1f}% | {', '.join(k_f1s)}")

    # Aggregate summary
    summary: Dict[str, Dict[str, Dict[str, float]]] = {
        m: {k_key: {} for k_key in eval_keys} for m in models
    }
    for m in models:
        for k_key in eval_keys:
            for met in metrics_list:
                vals = raw_results[m][k_key][met]
                summary[m][k_key][f"{met}_mean"] = float(np.mean(vals))
                summary[m][k_key][f"{met}_std"] = float(np.std(vals))

    # Statistical significance: paired Wilcoxon signed-rank tests vs QUASAR for each k
    stat_summary: Dict[str, Dict[str, Any]] = {}
    p_records = []
    for k_key in eval_keys:
        stat_summary[k_key] = {}
        quasar_vals = raw_results["QUASAR"][k_key]["f1"]
        for m in models:
            if m == "QUASAR":
                continue
            base_vals = raw_results[m][k_key]["f1"]
            diffs = np.array(quasar_vals) - np.array(base_vals)
            if np.all(diffs == 0) or len(diffs) < 5:
                stat_summary[k_key][m] = {"stat": 0.0, "raw_p": 1.0, "bh_q": 1.0, "sig": "ns"}
            else:
                try:
                    res_w = scipy.stats.wilcoxon(quasar_vals, base_vals, alternative="two-sided")
                    p_val = float(res_w.pvalue)
                    stat_val = float(res_w.statistic)
                except Exception:
                    p_val = 1.0
                    stat_val = 0.0
                p_records.append((k_key, m, stat_val, p_val))

    # Apply Benjamini-Hochberg FDR
    if p_records:
        p_records_sorted = sorted(p_records, key=lambda x: x[3])
        m_total = len(p_records_sorted)
        for rank, (k_key, m, stat_val, p_val) in enumerate(p_records_sorted, start=1):
            q_val = min(1.0, p_val * (m_total / rank))
            sig = "***" if q_val < 0.001 else ("**" if q_val < 0.01 else ("*" if q_val < 0.05 else "ns"))
            stat_summary[k_key][m] = {"stat": stat_val, "raw_p": p_val, "bh_q": q_val, "sig": sig}

    # Print Terminal Table
    print("\n" + "=" * 90)
    print(f"      REGULARITY SWEEP RESULTS: {dataset_name.upper()} (MACRO-F1 % at alpha = {alpha:.2f})")
    print("-" * 90)
    header = f"{'Model':<12} | " + " | ".join([f"k={k:<6}" for k in k_values])
    print(header)
    print("-" * 90)
    for m in models:
        row_strs = []
        for k in k_values:
            k_str = str(k)
            m_f1 = summary[m][k_str]["f1_mean"]
            s_f1 = summary[m][k_str]["f1_std"]
            sig = ""
            if m != "QUASAR" and k_str in stat_summary and m in stat_summary[k_str]:
                s = stat_summary[k_str][m]["sig"]
                if s != "ns":
                    sig = f" [{s}]"
            row_strs.append(f"{m_f1:.1f} ± {s_f1:.1f}{sig}")
        print(f"{m:<12} | " + " | ".join([f"{s:<10}" for s in row_strs]))
    print("=" * 90)

    # Plot
    fig_path = output_fig or f"benchmarks/figure_regularity_sweep_{dataset_name.lower()}.png"
    plot_regularity_sweep(k_values, summary, models, dataset_name, fig_path)

    # Save JSON
    out_json = output_json or f"benchmarks/regularity_sweep_{dataset_name.lower()}_results.json"
    full_payload = {
        "metadata": {
            "dataset_name": dataset_name,
            "data_dir": data_dir,
            "models": models,
            "k_values": k_values,
            "alpha": alpha,
            "seeds": seeds,
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": lr,
            "device": device,
            "total_runtime_sec": time.time() - start_time,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        "summary": summary,
        "statistical_tests": stat_summary,
        "raw_results": raw_results,
    }

    with open(out_json, "w") as f:
        json.dump(full_payload, f, indent=2)
    print(f"[✓] Regularity sweep results saved to: {out_json}")

    return full_payload

def main():
    parser = argparse.ArgumentParser(description="Multi-Degree Regularity Sweep Stress Test (IEEE TIFS)")
    parser.add_argument("--dataset", type=str, default="ctu13_10", help="Dataset name (default: ctu13_10)")
    parser.add_argument("--data_dir", type=str, default=None, help="Dataset folder")
    parser.add_argument("--k_values", nargs="+", type=int, default=[2, 3, 4, 6], help="Degrees k to evaluate")
    parser.add_argument("--alpha", type=float, default=1.00, help="Camouflage alpha (default: 1.00)")
    parser.add_argument("--num_seeds", type=int, default=10, help="Number of random seeds (default: 10)")
    parser.add_argument("--epochs", type=int, default=35, help="Training epochs (default: 35)")
    parser.add_argument("--batch_size", type=int, default=32, help="Mini-batch size (default: 32)")
    parser.add_argument("--lr", type=float, default=0.003, help="Learning rate (default: 0.003)")
    parser.add_argument("--hidden_dim", type=int, default=32, help="Hidden dimension (default: 32)")
    parser.add_argument("--num_layers", type=int, default=2, help="Layer count (default: 2)")
    parser.add_argument("--models", nargs="+", default=["GCN", "GIN", "RWPE-GIN", "MagNet", "QUASAR", "AEGIS"])
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--output_json", type=str, default=None)
    parser.add_argument("--output_fig", type=str, default=None)
    parser.add_argument("--smoke_test", action="store_true", help="Quick 1-seed test mode")
    parser.add_argument("--num_threads", type=int, default=min(4, os.cpu_count() or 4))

    args = parser.parse_args()

    torch.set_num_threads(max(1, args.num_threads))

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    data_dir = args.data_dir or os.path.join("data", "processed", args.dataset)

    if args.smoke_test:
        print("[*] Running in SMOKE TEST mode (1 seed, 35 epochs)...")
        seeds = [42]
    else:
        seeds = [42 + i for i in range(args.num_seeds)]

    run_regularity_sweep(
        dataset_name=args.dataset,
        data_dir=data_dir,
        k_values=args.k_values,
        alpha=args.alpha,
        models=args.models,
        seeds=seeds,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        device=device,
        output_json=args.output_json,
        output_fig=args.output_fig,
    )

if __name__ == "__main__":
    main()
