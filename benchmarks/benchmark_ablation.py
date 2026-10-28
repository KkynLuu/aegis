"""
Benchmark: Ablation Study for QUASAR Components (Table IV Verification)
Target Venue: IEEE TIFS / IEEE TDSC

Evaluates the step-by-step contribution of each novel component in QUASAR:
  1. Base-QUASAR: Pure Adjacency CTQW, single fixed scale (t=1.0), global mean/norm-sum/max pooling.
  2. +OnSitePotential: Adds feature-modulated Hamiltonian on-site potential H = A + diag(V_theta(x)).
  3. +MultiScaleFrequencies: Adds multi-scale log-space wavelet frequencies t in [0.2, 8.0].
  4. +QSPE-Attention (Full): Adds Q-SPE Gated Anomaly Attention Readout (full QUASAR architecture).

Evaluated across 10 independent random seeds on CTU-13 Scenario 10:
  - Clean Test Traffic (alpha = 0.00)
  - Adversarially Camouflaged Traffic (alpha = 1.00, degree-preserving 2-switch rewiring)

Saves results to benchmarks/ablation_results.json and benchmarks/figure_ablation_ctu13_10.png.
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

# Set non-interactive backend and writable cache dir for matplotlib
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Ensure root path is in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data.netflow_graph_builder import (
    NetFlowGraph,
    NetFlowInMemoryDataset,
    collate_netflow_batch,
    load_netflow_graphs,
)
from models.quasar_classifier import QuasarClassifier
from benchmarks.benchmark_camouflage import (
    compute_roc_auc_numpy,
    evaluate_model_on_loader,
    apply_topological_camouflage,
)

VARIANTS = {
    "Base-QUASAR": {
        "display_name": "Base QUASAR (Global Mean/Norm-Sum Pooling)",
        "use_potential": False,
        "use_attention": False,
        "num_time_scales": 1,
        "init_times": (1.0,),
    },
    "+OnSitePotential": {
        "display_name": "+ Inhomogeneous On-Site Potential H = A + diag(V)",
        "use_potential": True,
        "use_attention": False,
        "num_time_scales": 1,
        "init_times": (1.0,),
    },
    "+MultiScaleFrequencies": {
        "display_name": "+ Multi-Scale Wavelets t in [0.2, 8.0]",
        "use_potential": True,
        "use_attention": False,
        "num_time_scales": 4,
        "init_times": (0.2, 0.8, 2.5, 8.0),
    },
    "+QSPE-Attention (Full)": {
        "display_name": "+ Q-SPE Gated Attention Readout (Full QUASAR)",
        "use_potential": True,
        "use_attention": True,
        "num_time_scales": 4,
        "init_times": (0.2, 0.8, 2.5, 8.0),
    },
}


def train_single_ablation_model(
    variant_name: str,
    cfg: Dict[str, Any],
    in_dim: int,
    hidden_dim: int,
    num_layers: int,
    train_loader: DataLoader,
    val_loader: DataLoader,
    class_weights: torch.Tensor,
    epochs: int = 35,
    lr: float = 0.003,
    device: str = "cpu",
    seed: int = 42,
) -> nn.Module:
    """Trains a specific QUASAR ablation variant with validation checkpointing."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = QuasarClassifier(
        in_dim=in_dim,
        hidden_dim=hidden_dim,
        num_classes=2,
        num_layers=num_layers,
        num_time_scales=cfg["num_time_scales"],
        init_times=cfg["init_times"],
        use_potential=cfg["use_potential"],
        use_attention=cfg["use_attention"],
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))

    best_val_f1 = -1.0
    best_weights = copy.deepcopy(model.state_dict())

    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            x = batch["x"].to(device)
            adj = batch["adj"].to(device)
            mask = batch["mask"].to(device)
            y = batch["labels"].to(device)

            optimizer.zero_grad()
            logits = model(x, adj, mask=mask)
            loss = criterion(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()

        scheduler.step()

        # Validation check every epoch
        val_metrics = evaluate_model_on_loader(model, val_loader, device=device)
        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            best_weights = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_weights)
    return model


def run_ablation_benchmark(args: argparse.Namespace) -> None:
    # Set PyTorch thread limit for optimal CPU thermal and execution efficiency
    torch.set_num_threads(args.num_threads)
    print(f"[*] Execution Device: CPU | PyTorch Threads: {args.num_threads} (Thermal & Latency Optimized)")

    if args.smoke_test:
        print("[*] Running in SMOKE TEST mode (2 seeds, 3 epochs)...")
        seeds = [42, 43]
        epochs = 3
    else:
        seeds = [42 + i for i in range(args.num_seeds)]
        epochs = args.epochs

    device = "cpu"

    print("=" * 100)
    print("      EXPERIMENT 3: QUASAR COMPONENT ABLATION STUDY (TABLE IV VERIFICATION)")
    print(f"      Dataset:           CTU13_10 ({args.data_dir})")
    print(f"      Variants:          {list(VARIANTS.keys())}")
    print(f"      Seeds:             {len(seeds)} {seeds}")
    print(f"      Epochs: {epochs} | Batch Size: {args.batch_size} | LR: {args.lr} | Device: {device}")
    print("=" * 100)

    # 1. Load Data
    train_graphs, _ = load_netflow_graphs(os.path.join(args.data_dir, "train.pt"))
    val_graphs, _ = load_netflow_graphs(os.path.join(args.data_dir, "val.pt"))
    test_graphs, _ = load_netflow_graphs(os.path.join(args.data_dir, "test.pt"))

    if args.smoke_test:
        ben_train = [g for g in train_graphs if g.y == 0][:32]
        mal_train = [g for g in train_graphs if g.y == 1][:32]
        train_graphs = ben_train + mal_train

        ben_val = [g for g in val_graphs if g.y == 0][:16]
        mal_val = [g for g in val_graphs if g.y == 1][:16]
        val_graphs = ben_val + mal_val

        ben_test = [g for g in test_graphs if g.y == 0][:16]
        mal_test = [g for g in test_graphs if g.y == 1][:16]
        test_graphs = ben_test + mal_test
        print(f"[!] Smoke test: stratified subset loaded ({len(train_graphs)} train, {len(val_graphs)} val, {len(test_graphs)} test).")

    # Compute empirical benign mean feature vector
    benign_features = [g.x for g in train_graphs if g.y == 0]
    if len(benign_features) > 0:
        all_benign_x = torch.cat(benign_features, dim=0)
        benign_mean = all_benign_x.mean(dim=0, keepdim=True)
    else:
        benign_mean = torch.zeros(1, train_graphs[0].x.shape[-1])

    # Class balance weights for cross-entropy
    train_labels = [g.y for g in train_graphs]
    n_ben = max(1, sum(1 for y in train_labels if y == 0))
    n_mal = max(1, sum(1 for y in train_labels if y == 1))
    class_weights = torch.tensor([n_mal / (n_ben + n_mal), n_ben / (n_ben + n_mal)], dtype=torch.float32)

    train_loader = DataLoader(
        NetFlowInMemoryDataset(train_graphs),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_netflow_batch,
    )
    val_loader = DataLoader(
        NetFlowInMemoryDataset(val_graphs),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_netflow_batch,
    )

    # Pre-generate Clean and Full-Camouflage Test Sets
    test_clean = test_graphs
    test_camouflaged = apply_topological_camouflage(
        test_graphs, alpha=1.00, benign_mean=benign_mean, k_regular=2
    )

    loader_clean = DataLoader(
        NetFlowInMemoryDataset(test_clean),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_netflow_batch,
    )
    loader_camouflaged = DataLoader(
        NetFlowInMemoryDataset(test_camouflaged),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_netflow_batch,
    )

    in_dim = train_graphs[0].x.shape[-1]
    hidden_dim = args.hidden_dim
    num_layers = args.num_layers

    # Storage for per-seed results
    # variant -> {"clean": [metrics], "camouflaged": [metrics]}
    raw_results: Dict[str, Dict[str, List[Dict[str, float]]]] = {
        v: {"clean": [], "camouflaged": []} for v in VARIANTS
    }

    start_total_time = time.time()

    for seed_idx, seed in enumerate(seeds):
        print(f"\n--- [Seed {seed_idx + 1}/{len(seeds)}: {seed}] ---")
        for v_name, cfg in VARIANTS.items():
            t0 = time.time()
            model = train_single_ablation_model(
                variant_name=v_name,
                cfg=cfg,
                in_dim=in_dim,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                train_loader=train_loader,
                val_loader=val_loader,
                class_weights=class_weights,
                epochs=epochs,
                lr=args.lr,
                device=device,
                seed=seed,
            )
            train_dur = time.time() - t0

            # Evaluate on Clean and Full Camouflage
            m_clean = evaluate_model_on_loader(model, loader_clean, device=device)
            m_cam = evaluate_model_on_loader(model, loader_camouflaged, device=device)

            raw_results[v_name]["clean"].append(m_clean)
            raw_results[v_name]["camouflaged"].append(m_cam)

            print(
                f"  {v_name:<24} | Train: {train_dur:>5.2f}s | "
                f"Clean F1: {m_clean['f1']:>5.1f}% (FAR: {m_clean['far']:>4.1f}%) | "
                f"Camouflaged F1: {m_cam['f1']:>5.1f}% (FAR: {m_cam['far']:>4.1f}%)"
            )

    total_runtime = time.time() - start_total_time

    # 4. Aggregate Metrics (Mean, Std)
    summary: Dict[str, Any] = {}
    for v_name in VARIANTS:
        summary[v_name] = {}
        for condition in ["clean", "camouflaged"]:
            metrics_list = raw_results[v_name][condition]
            cond_summary: Dict[str, float] = {}
            for metric in ["f1", "precision", "recall", "far", "auc", "acc"]:
                vals = [m[metric] for m in metrics_list]
                cond_summary[f"{metric}_mean"] = float(np.mean(vals))
                cond_summary[f"{metric}_std"] = float(np.std(vals))
            summary[v_name][condition] = cond_summary

    # 5. Statistical Tests: Compare each variant against Full QUASAR on Clean and Camouflaged
    full_clean_f1 = [m["f1"] for m in raw_results["+QSPE-Attention (Full)"]["clean"]]
    full_cam_f1 = [m["f1"] for m in raw_results["+QSPE-Attention (Full)"]["camouflaged"]]

    stat_tests: Dict[str, Any] = {"clean": {}, "camouflaged": {}}
    for v_name in VARIANTS:
        if v_name == "+QSPE-Attention (Full)":
            continue

        # Clean F1 Wilcoxon
        v_clean_f1 = [m["f1"] for m in raw_results[v_name]["clean"]]
        diff_clean = np.array(full_clean_f1) - np.array(v_clean_f1)
        if np.all(diff_clean == 0) or len(diff_clean) < 3:
            p_clean = 1.0
            stat_clean = 0.0
        else:
            try:
                res_c = scipy.stats.wilcoxon(full_clean_f1, v_clean_f1, alternative="greater")
                stat_clean, p_clean = float(res_c.statistic), float(res_c.pvalue)
            except Exception:
                stat_clean, p_clean = 0.0, 1.0

        # Camouflaged F1 Wilcoxon
        v_cam_f1 = [m["f1"] for m in raw_results[v_name]["camouflaged"]]
        diff_cam = np.array(full_cam_f1) - np.array(v_cam_f1)
        if np.all(diff_cam == 0) or len(diff_cam) < 3:
            p_cam = 1.0
            stat_cam = 0.0
        else:
            try:
                res_m = scipy.stats.wilcoxon(full_cam_f1, v_cam_f1, alternative="greater")
                stat_cam, p_cam = float(res_m.statistic), float(res_m.pvalue)
            except Exception:
                stat_cam, p_cam = 0.0, 1.0

        stat_tests["clean"][v_name] = {"stat": stat_clean, "p_value": p_clean}
        stat_tests["camouflaged"][v_name] = {"stat": stat_cam, "p_value": p_cam}

    # Print Formatted Table IV
    print("\n" + "=" * 115)
    print("                     TABLE IV: QUASAR ARCHITECTURAL ABLATION STUDY (CTU13_10)")
    print("-" * 115)
    print(f"{'Architecture Variant':<45} | {'Clean F1 (%)':<18} | {'Camouflaged F1 (%)':<20} | {'FAR (%) [↓]':<15}")
    print("-" * 115)
    for v_name, cfg in VARIANTS.items():
        disp = cfg["display_name"]
        c_f1 = f"{summary[v_name]['clean']['f1_mean']:.2f} ± {summary[v_name]['clean']['f1_std']:.2f}"
        m_f1 = f"{summary[v_name]['camouflaged']['f1_mean']:.2f} ± {summary[v_name]['camouflaged']['f1_std']:.2f}"
        far = f"{summary[v_name]['clean']['far_mean']:.2f} ± {summary[v_name]['clean']['far_std']:.2f}"
        print(f"{disp:<45} | {c_f1:<18} | {m_f1:<20} | {far:<15}")
    print("=" * 115)

    # Print LaTeX Table
    latex_table = generate_latex_table(summary)
    print("\n--- IEEE TIFS Publication-Ready LaTeX Table IV ---")
    print(latex_table)
    print("-" * 60)

    # Save JSON Artifact
    output_payload = {
        "metadata": {
            "dataset": args.dataset,
            "data_dir": args.data_dir,
            "num_seeds": len(seeds),
            "seeds": seeds,
            "epochs": epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "total_runtime_sec": total_runtime,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        "summary": summary,
        "statistical_tests": stat_tests,
        "raw_results": raw_results,
    }

    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(output_payload, f, indent=2)
    print(f"\n[✓] Verified ablation results saved to: {args.output_json}")

    # Generate Publication Figure
    plot_ablation_figure(summary, args.output_fig)
    print(f"[✓] Publication figure saved to: {args.output_fig}")


def generate_latex_table(summary: Dict[str, Any]) -> str:
    """Generates publication-ready IEEE TIFS LaTeX code for Table IV."""
    rows = []
    for v_name, cfg in VARIANTS.items():
        disp = cfg["display_name"]
        c_mean = summary[v_name]["clean"]["f1_mean"]
        c_std = summary[v_name]["clean"]["f1_std"]
        m_mean = summary[v_name]["camouflaged"]["f1_mean"]
        m_std = summary[v_name]["camouflaged"]["f1_std"]
        far_mean = summary[v_name]["clean"]["far_mean"]
        far_std = summary[v_name]["clean"]["far_std"]

        if v_name == "+QSPE-Attention (Full)":
            disp_str = f"\\textbf{{{disp}}}"
            c_str = f"\\textbf{{{c_mean:.2f} $\\pm$ {c_std:.2f}}}"
            m_str = f"\\textbf{{{m_mean:.2f} $\\pm$ {m_std:.2f}}}"
            far_str = f"\\textbf{{{far_mean:.2f} $\\pm$ {far_std:.2f}}}"
        else:
            disp_str = disp
            c_str = f"{c_mean:.2f} $\\pm$ {c_std:.2f}"
            m_str = f"{m_mean:.2f} $\\pm$ {m_std:.2f}"
            far_str = f"{far_mean:.2f} $\\pm$ {far_std:.2f}"

        rows.append(f"{disp_str} & {c_str} & {m_str} & {far_str} \\\\")

    rows_str = "\n".join(rows)

    return f"""\\begin{{table}}[t]
\\centering
\\caption{{Architectural Ablation Study on CTU-13. Macro-F1 score (\\%) and False Alarm Rate (FAR, \\%) (mean $\\pm$ std across 10 independent seeds) for unperturbed clean traffic and full topological camouflage ($\\alpha=1.00$).}}
\\label{{tab:quasar_ablation}}
\\small
\\begin{{tabular}}{{lccc}}
\\toprule
\\textbf{{Architecture Variant}} & \\textbf{{Clean F1 (\\%)}} & \\textbf{{Camouflaged F1 (\\%)}} & \\textbf{{FAR (\\%) [$\\downarrow$]}} \\\\
\\midrule
{rows_str}
\\bottomrule
\\end{{tabular}}
\\end{{table}}"""


def plot_ablation_figure(summary: Dict[str, Any], output_path: str) -> None:
    """Plots publication-quality bar chart comparing the ablation variants."""
    variant_names = [
        "Base QUASAR\n(Global Mean)",
        "+ On-Site\nPotential",
        "+ Multi-Scale\nWavelets",
        "+ Q-SPE Attention\n(Full QUASAR)",
    ]
    keys = list(VARIANTS.keys())

    clean_f1 = [summary[k]["clean"]["f1_mean"] for k in keys]
    clean_err = [summary[k]["clean"]["f1_std"] for k in keys]

    cam_f1 = [summary[k]["camouflaged"]["f1_mean"] for k in keys]
    cam_err = [summary[k]["camouflaged"]["f1_std"] for k in keys]

    x = np.arange(len(keys))
    width = 0.35

    plt.figure(figsize=(9, 5.5), dpi=300)
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.size"] = 11

    rects1 = plt.bar(
        x - width / 2,
        clean_f1,
        width,
        yerr=clean_err,
        capsize=5,
        label="Clean Traffic ($\\alpha=0.00$)",
        color="#2b5c8f",
        edgecolor="black",
        linewidth=1.2,
        alpha=0.9,
    )
    rects2 = plt.bar(
        x + width / 2,
        cam_f1,
        width,
        yerr=cam_err,
        capsize=5,
        label="Adversarial Camouflage ($\\alpha=1.00$)",
        color="#d95f02",
        edgecolor="black",
        linewidth=1.2,
        alpha=0.9,
    )

    plt.ylabel("Macro-F1 Score (%)", fontweight="bold")
    plt.title("QUASAR Component Ablation Study (CTU-13 Benchmark)", fontweight="bold", pad=14)
    plt.xticks(x, variant_names, fontweight="semibold")
    plt.ylim(50, 100)
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.legend(frameon=True, facecolor="white", edgecolor="none", loc="lower right")

    # Add numeric labels on top of bars
    for rect in rects1:
        height = rect.get_height()
        plt.annotate(
            f"{height:.1f}%",
            xy=(rect.get_x() + rect.get_width() / 2, height),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
            color="#1b3c5f",
        )

    for rect in rects2:
        height = rect.get_height()
        plt.annotate(
            f"{height:.1f}%",
            xy=(rect.get_x() + rect.get_width() / 2, height),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
            color="#a94000",
        )

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="QUASAR Component Ablation Study (IEEE TIFS Table IV Verification)"
    )
    parser.add_argument("--dataset", type=str, default="ctu13_10", choices=["ctu13_10", "ctu13_9", "nf_bot_iot"])
    parser.add_argument("--data_dir", type=str, default="data/processed/ctu13_10")
    parser.add_argument("--num_seeds", type=int, default=10, help="Number of random seeds (default: 10)")
    parser.add_argument("--epochs", type=int, default=35, help="Training epochs per seed (default: 35)")
    parser.add_argument("--batch_size", type=int, default=32, help="Mini-batch size (default: 32)")
    parser.add_argument("--lr", type=float, default=0.003, help="Learning rate (default: 0.003)")
    parser.add_argument("--hidden_dim", type=int, default=32, help="Hidden dimension (default: 32)")
    parser.add_argument("--num_layers", type=int, default=2, help="Number of layers (default: 2)")
    parser.add_argument("--num_threads", type=int, default=4, help="PyTorch CPU threads (default: 4)")
    parser.add_argument("--output_json", type=str, default="benchmarks/ablation_results.json")
    parser.add_argument("--output_fig", type=str, default="benchmarks/figure_ablation_ctu13_10.png")
    parser.add_argument("--smoke_test", action="store_true", help="Quick smoke test (2 seeds, 3 epochs)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_ablation_benchmark(args)
