"""
Benchmark: Magnetic Continuous-Time Quantum Walk (M-CTQW) vs Baselines (Pillar 2).

Evaluates the impact of incorporating directed NetFlow causality via Complex Hermitian
Hamiltonians with Aharonov-Bohm phase modulation:
    H_mag(θ) = A_sym ⊙ exp(i Θ_phase) + diag(θ)
    Θ_phase = 2π q (A - A^T)

Compared Architectures:
    1. MagNet (Zhang et al. NeurIPS 2021, Complex Hermitian Laplacian baseline)
    2. Symmetric-QUASAR (Real symmetric baseline: A_sym + diag(θ))
    3. M-CTQW-Static (Magnetic CTQW with fixed/detached on-site potential)
    4. M-CTQW-Full (Magnetic CTQW with differentiable Hellmann-Feynman optimization & Directional Flux Attention)

Evaluated across multiple independent random seeds on:
    - Clean Enterprise NetFlow Traffic (alpha = 0.00)
    - Full Camouflage (alpha = 1.00, feature mimicry + 2-regular cycle overlay)

Features robust checkpointing: Automatically resumes exactly where it left off
if interrupted, skipping already-completed runs without repeating compute.
"""

import os
import sys

# Set writable MPLCONFIGDIR before importing matplotlib
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib_cache"

import time
import json
import argparse
from typing import Dict, List, Any, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from scipy.stats import wilcoxon
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
from models.baseline_gnn import MagNetClassifier
from benchmarks.benchmark_camouflage import (
    compute_roc_auc_numpy,
    evaluate_model_on_loader,
    apply_topological_camouflage,
)

# Benchmark Variants
VARIANTS = {
    "MagNet": {
        "display_name": "MagNet (Complex Laplacian GNN)",
        "is_quasar": False,
        "type": "magnet",
        "q": 0.25,
    },
    "Symmetric-QUASAR": {
        "display_name": "Symmetric QUASAR (Real Baseline)",
        "is_quasar": True,
        "use_magnetic": False,
        "magnetic_flux": 0.25,
        "use_potential": True,
        "use_attention": True,
        "differentiable_potential": False,
    },
    "M-CTQW-Static": {
        "display_name": "M-CTQW (Static Potential)",
        "is_quasar": True,
        "use_magnetic": True,
        "magnetic_flux": 0.25,
        "use_potential": True,
        "use_attention": True,
        "differentiable_potential": False,
    },
    "M-CTQW-Full": {
        "display_name": "M-CTQW (Differentiable + Chiral Flux)",
        "is_quasar": True,
        "use_magnetic": True,
        "magnetic_flux": 0.25,
        "use_potential": True,
        "use_attention": True,
        "differentiable_potential": True,
    },
}


def build_model(
    v_name: str,
    cfg: Dict[str, Any],
    in_dim: int,
    hidden_dim: int,
    num_layers: int,
    num_classes: int = 2,
    dropout: float = 0.1,
) -> nn.Module:
    """Factory creating the specified model variant."""
    if cfg.get("is_quasar", True):
        return QuasarClassifier(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            num_layers=num_layers,
            dropout=dropout,
            use_potential=cfg["use_potential"],
            use_attention=cfg["use_attention"],
            use_magnetic=cfg["use_magnetic"],
            magnetic_flux=cfg["magnetic_flux"],
            differentiable_potential=cfg["differentiable_potential"],
        )
    elif cfg.get("type") == "magnet":
        return MagNetClassifier(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            num_layers=num_layers,
            q=cfg.get("q", 0.25),
            dropout=dropout,
        )
    else:
        raise ValueError(f"Unknown variant type: {v_name}")


def train_single_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    class_weights: torch.Tensor,
    epochs: int,
    lr: float,
    device: torch.device,
    seed: int,
) -> Tuple[nn.Module, float]:
    """Trains a single model with class weighting and validation tracking."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    best_val_f1 = -1.0
    best_weights = None
    t0 = time.time()

    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            x = batch["x"].to(device)
            adj = batch["adj"].to(device)
            mask = batch["mask"].to(device)
            y = batch["labels"].to(device)

            optimizer.zero_grad()
            logits = model(x, adj, mask)
            loss = criterion(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()

        scheduler.step()

        # Validation
        val_metrics = evaluate_model_on_loader(model, val_loader, device=device)
        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    train_time = time.time() - t0
    if best_weights is not None:
        model.load_state_dict(best_weights)

    return model, train_time


def load_checkpoint(checkpoint_path: str) -> Dict[str, Any]:
    """Loads existing checkpoint dictionary if present, else returns empty structure."""
    if os.path.exists(checkpoint_path):
        try:
            with open(checkpoint_path, "r") as f:
                data = json.load(f)
            if isinstance(data, dict) and "completed_runs" in data:
                return data
        except Exception as e:
            print(f"[!] Warning: Failed to parse checkpoint at {checkpoint_path} ({e}). Starting fresh.")
    return {"completed_runs": {}}


def save_checkpoint_atomic(checkpoint_path: str, data: Dict[str, Any]) -> None:
    """Atomically saves checkpoint to disk to avoid corruption upon interrupt."""
    os.makedirs(os.path.dirname(os.path.abspath(checkpoint_path)), exist_ok=True)
    temp_path = f"{checkpoint_path}.tmp.{os.getpid()}"
    with open(temp_path, "w") as f:
        json.dump(data, f, indent=2)
    os.replace(temp_path, checkpoint_path)


def generate_magnetic_latex_table(
    variants: Dict[str, Dict[str, Any]],
    summary: Dict[str, Dict[str, Dict[str, float]]],
    stat_tests: Dict[str, Any],
) -> str:
    """Generates a publication-grade LaTeX table comparing M-CTQW to baselines."""
    rows = []
    for v_name, cfg in variants.items():
        disp = cfg["display_name"]
        clean = summary[v_name]["clean"]
        cam = summary[v_name]["camouflaged"]

        c_f1 = f"{clean['f1_mean']:.2f} \\pm {clean['f1_std']:.2f}"
        c_far = f"{clean['far_mean']:.2f} \\pm {clean['far_std']:.2f}"
        cam_f1 = f"{cam['f1_mean']:.2f} \\pm {cam['f1_std']:.2f}"
        cam_far = f"{cam['far_mean']:.2f} \\pm {cam['far_std']:.2f}"

        p_clean = stat_tests["clean"].get(v_name, {}).get("p_val", None)
        p_str = f"p={p_clean:.4f}" if p_clean is not None else "---"

        if v_name == "M-CTQW-Full":
            disp = f"\\textbf{{{disp}}}"
            c_f1 = f"\\textbf{{{c_f1}}}"
            c_far = f"\\textbf{{{c_far}}}"
            cam_f1 = f"\\textbf{{{cam_f1}}}"
            cam_far = f"\\textbf{{{cam_far}}}"
            p_str = "\\textbf{Baseline}"

        rows.append(f"{disp} & {c_f1} & {c_far} & {cam_f1} & {cam_far} & {p_str} \\\\")

    table_body = "\n".join(rows)

    return f"""\\begin{{table*}}[t]
\\centering
\\caption{{Pillar 2: Magnetic Continuous-Time Quantum Walk (M-CTQW) Performance on CTU-13. Evaluates directional flow causality via Aharonov-Bohm phase modulation ($\\Theta = 2\\pi q (\\mathbf{{A}} - \\mathbf{{A}}^\\top)$) and Hellmann-Feynman potential optimization across independent seeds (mean $\\pm$ std).}}
\\label{{tab:magnetic_mctqw}}
\\small
\\begin{{tabular}}{{lccccc}}
\\toprule
\\textbf{{Model Variant}} & \\textbf{{Clean F1 (\\%) [$\\uparrow$]}} & \\textbf{{Clean FAR (\\%) [$\\downarrow$]}} & \\textbf{{Camouflaged F1 (\\%) [$\\uparrow$]}} & \\textbf{{Camouflaged FAR (\\%) [$\\downarrow$]}} & \\textbf{{Wilcoxon vs Full}} \\\\
\\midrule
{table_body}
\\bottomrule
\\end{{tabular}}
\\end{{table*}}"""


def plot_magnetic_figure(
    variants: Dict[str, Dict[str, Any]],
    summary: Dict[str, Dict[str, Dict[str, float]]],
    output_path: str,
) -> None:
    """Generates a high-resolution 2-panel publication figure comparing M-CTQW to baselines."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.2), dpi=300)
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.size"] = 11

    v_keys = list(variants.keys())
    labels = [k.replace("M-CTQW-", "M-CTQW\n").replace("Symmetric-", "Symmetric\n") for k in v_keys]
    x_pos = np.arange(len(v_keys))
    width = 0.35

    # Panel (a): Macro-F1: Clean vs Camouflaged
    clean_f1 = [summary[k]["clean"]["f1_mean"] for k in v_keys]
    clean_std = [summary[k]["clean"]["f1_std"] for k in v_keys]
    cam_f1 = [summary[k]["camouflaged"]["f1_mean"] for k in v_keys]
    cam_std = [summary[k]["camouflaged"]["f1_std"] for k in v_keys]

    ax1.bar(x_pos - width / 2, clean_f1, width, yerr=clean_std, capsize=4, label="Clean Traffic ($\\alpha=0.00$)", color="#1f77b4", alpha=0.9, edgecolor="black")
    ax1.bar(x_pos + width / 2, cam_f1, width, yerr=cam_std, capsize=4, label="Full Camouflage ($\\alpha=1.00$)", color="#d62728", alpha=0.9, edgecolor="black")

    ax1.set_ylabel("Macro-F1 Score (%)", fontsize=12, fontweight="bold")
    ax1.set_title("(a) Evasion Robustness: Clean vs 2-Regular Camouflage", fontsize=12, fontweight="bold", pad=10)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(labels, fontsize=10)
    ax1.set_ylim(40, 100)
    ax1.grid(axis="y", linestyle="--", alpha=0.5)
    ax1.legend(loc="lower left", framealpha=0.95)

    # Panel (b): False Alarm Rate (FAR) vs ROC-AUC
    far_vals = [summary[k]["clean"]["far_mean"] for k in v_keys]
    far_std = [summary[k]["clean"]["far_std"] for k in v_keys]
    auc_vals = [summary[k]["clean"]["auc_mean"] for k in v_keys]
    auc_std = [summary[k]["clean"]["auc_std"] for k in v_keys]

    colors = ["#7f7f7f", "#ff7f0e", "#2ca02c", "#9467bd"]
    for i, k in enumerate(v_keys):
        ax2.errorbar(
            far_vals[i],
            auc_vals[i],
            xerr=far_std[i],
            yerr=auc_std[i],
            fmt="o",
            markersize=10,
            capsize=5,
            label=variants[k]["display_name"],
            color=colors[i % len(colors)],
            mec="black",
        )

    ax2.set_xlabel("False Alarm Rate (FAR %) [Lower is Better]", fontsize=12, fontweight="bold")
    ax2.set_ylabel("ROC-AUC (%) [Higher is Better]", fontsize=12, fontweight="bold")
    ax2.set_title("(b) Enterprise Trade-off: Precision vs False Positives", fontsize=12, fontweight="bold", pad=10)
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(loc="lower right", framealpha=0.95, fontsize=9)

    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def run_magnetic_benchmark(args: argparse.Namespace) -> None:
    # 0. Thread Discipline for Apple Silicon thermal stability
    torch.set_num_threads(args.num_threads)
    os.environ["OMP_NUM_THREADS"] = str(args.num_threads)
    os.environ["MKL_NUM_THREADS"] = str(args.num_threads)

    device = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    print(f"[*] Running M-CTQW Benchmark (Pillar 2)")
    print(f"    Device: {device} (PyTorch threads: {torch.get_num_threads()})")
    print(f"    Data: {args.data_dir} | Seeds: {args.seeds} | Epochs: {args.epochs}")
    print(f"    Checkpoint: {args.checkpoint_path}")

    # 1. Load NetFlow Dataset
    train_graphs, _ = load_netflow_graphs(os.path.join(args.data_dir, "train.pt"))
    val_graphs, _ = load_netflow_graphs(os.path.join(args.data_dir, "val.pt"))
    test_graphs, _ = load_netflow_graphs(os.path.join(args.data_dir, "test.pt"))

    # Compute Class Weights
    y_train = [g.y for g in train_graphs]
    n_benign = sum(1 for y in y_train if y == 0)
    n_attack = sum(1 for y in y_train if y == 1)
    class_weights = torch.tensor([1.0, float(n_benign) / max(n_attack, 1)], dtype=torch.float32)

    # Compute Benign Feature Mean for Feature Mimicry
    benign_features = [g.x for g in train_graphs if g.y == 0]
    benign_mean = torch.stack([x.mean(dim=0) for x in benign_features]).mean(dim=0)

    # Build Test Datasets: Clean and Full Camouflage
    test_clean = test_graphs
    test_camouflaged = apply_topological_camouflage(
        test_graphs, alpha=1.00, benign_mean=benign_mean, k_regular=2
    )

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
    loader_clean = DataLoader(
        NetFlowInMemoryDataset(test_clean),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_netflow_batch,
    )
    loader_cam = DataLoader(
        NetFlowInMemoryDataset(test_camouflaged),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_netflow_batch,
    )

    in_dim = train_graphs[0].x.shape[-1]
    hidden_dim = args.hidden_dim
    num_layers = args.num_layers

    # 2. Checkpoint Management ("Continue where we left off")
    if args.force_restart:
        print("[!] --force_restart specified: Clearing prior checkpoints.")
        checkpoint_data = {"completed_runs": {}}
    else:
        checkpoint_data = load_checkpoint(args.checkpoint_path)

    completed_runs = checkpoint_data.get("completed_runs", {})
    num_resumed = len(completed_runs)
    if num_resumed > 0:
        print(f"[✓] Found {num_resumed} completed runs in checkpoint ({args.checkpoint_path}). Resuming execution!")

    # Raw results mapping: variant -> {"clean": [list of metrics], "camouflaged": [list of metrics]}
    raw_results: Dict[str, Dict[str, List[Dict[str, float]]]] = {
        v: {"clean": [], "camouflaged": []} for v in VARIANTS
    }

    start_total_time = time.time()

    # 3. Benchmark Execution Loop
    for seed_idx, seed in enumerate(args.seeds):
        print(f"\n========================================================")
        print(f"=== [Seed {seed_idx + 1}/{len(args.seeds)}: Random Seed {seed}] ===")
        print(f"========================================================")

        for v_name, cfg in VARIANTS.items():
            run_key = f"seed_{seed}_{v_name}"

            if run_key in completed_runs:
                # --- RESUME FROM CHECKPOINT ---
                saved = completed_runs[run_key]
                raw_results[v_name]["clean"].append(saved["clean"])
                raw_results[v_name]["camouflaged"].append(saved["camouflaged"])
                print(
                    f"  [⏩ Resumed] {v_name:<20} | "
                    f"Clean F1: {saved['clean']['f1']:>5.2f}% (FAR: {saved['clean']['far']:>4.1f}%) | "
                    f"Cam F1: {saved['camouflaged']['f1']:>5.2f}% (FAR: {saved['camouflaged']['far']:>4.1f}%)"
                )
                continue

            # --- TRAIN AND EVALUATE NEW RUN ---
            print(f"  [⚡ Training] {v_name:<20} | Initializing model architecture...")
            model = build_model(
                v_name=v_name,
                cfg=cfg,
                in_dim=in_dim,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                dropout=args.dropout,
            )

            model, train_dur = train_single_model(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                class_weights=class_weights,
                epochs=args.epochs,
                lr=args.lr,
                device=device,
                seed=seed,
            )

            # Evaluate on Clean and Full Camouflage
            m_clean = evaluate_model_on_loader(model, loader_clean, device=device)
            m_cam = evaluate_model_on_loader(model, loader_cam, device=device)

            raw_results[v_name]["clean"].append(m_clean)
            raw_results[v_name]["camouflaged"].append(m_cam)

            # Record run in completed dictionary and atomically save checkpoint
            completed_runs[run_key] = {
                "seed": seed,
                "variant": v_name,
                "train_time_sec": float(train_dur),
                "clean": m_clean,
                "camouflaged": m_cam,
            }
            checkpoint_data["completed_runs"] = completed_runs
            save_checkpoint_atomic(args.checkpoint_path, checkpoint_data)

            print(
                f"  [✓ Done]     {v_name:<20} | Train: {train_dur:>5.1f}s | "
                f"Clean F1: {m_clean['f1']:>5.2f}% (FAR: {m_clean['far']:>4.1f}%) | "
                f"Cam F1: {m_cam['f1']:>5.2f}% (FAR: {m_cam['far']:>4.1f}%)"
            )

    total_runtime = time.time() - start_total_time

    # 4. Summary Statistics (Mean ± Std)
    summary: Dict[str, Any] = {}
    for v_name in VARIANTS:
        summary[v_name] = {}
        for cond in ["clean", "camouflaged"]:
            m_list = raw_results[v_name][cond]
            cond_stat: Dict[str, float] = {}
            for metric in ["f1", "precision", "recall", "far", "auc", "acc"]:
                vals = [m[metric] for m in m_list]
                cond_stat[f"{metric}_mean"] = float(np.mean(vals))
                cond_stat[f"{metric}_std"] = float(np.std(vals))
            summary[v_name][cond] = cond_stat

    # 5. Statistical Hypothesis Testing: Wilcoxon Signed-Rank Test vs M-CTQW-Full
    full_clean_f1 = [m["f1"] for m in raw_results["M-CTQW-Full"]["clean"]]
    full_cam_f1 = [m["f1"] for m in raw_results["M-CTQW-Full"]["camouflaged"]]
    stat_tests: Dict[str, Any] = {"clean": {}, "camouflaged": {}}

    for v_name in VARIANTS:
        if v_name == "M-CTQW-Full":
            continue
        v_clean_f1 = [m["f1"] for m in raw_results[v_name]["clean"]]
        v_cam_f1 = [m["f1"] for m in raw_results[v_name]["camouflaged"]]

        # Clean Wilcoxon
        try:
            diff_clean = np.array(full_clean_f1) - np.array(v_clean_f1)
            if np.all(diff_clean == 0):
                p_c = 1.0
            else:
                _, p_c = wilcoxon(full_clean_f1, v_clean_f1)
        except Exception:
            p_c = 1.0

        # Camouflaged Wilcoxon
        try:
            diff_cam = np.array(full_cam_f1) - np.array(v_cam_f1)
            if np.all(diff_cam == 0):
                p_cam = 1.0
            else:
                _, p_cam = wilcoxon(full_cam_f1, v_cam_f1)
        except Exception:
            p_cam = 1.0

        stat_tests["clean"][v_name] = {"p_val": float(p_c)}
        stat_tests["camouflaged"][v_name] = {"p_val": float(p_cam)}

    # 6. Generate LaTeX Table
    latex_table = generate_magnetic_latex_table(VARIANTS, summary, stat_tests)
    print("\n" + "=" * 60)
    print("--- IEEE TIFS / npj QI Publication-Ready LaTeX Table ---")
    print("=" * 60)
    print(latex_table)
    print("=" * 60)

    # 7. Generate Publication Figure
    plot_magnetic_figure(VARIANTS, summary, args.output_fig)
    print(f"\n[✓] Publication figure saved to: {args.output_fig}")

    # 8. Save Final JSON Payload
    output_payload = {
        "metadata": {
            "experiment": "Pillar 2: Magnetic Continuous-Time Quantum Walk (M-CTQW)",
            "dataset": "ctu13_10",
            "data_dir": args.data_dir,
            "variants": list(VARIANTS.keys()),
            "seeds": args.seeds,
            "epochs": args.epochs,
            "hidden_dim": args.hidden_dim,
            "num_layers": args.num_layers,
            "total_runtime_sec": total_runtime,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        "summary": summary,
        "statistical_tests": stat_tests,
        "raw_results": raw_results,
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(output_payload, f, indent=2)
    print(f"[✓] Full experimental results saved to: {args.output_json}")


def main():
    parser = argparse.ArgumentParser(
        description="Pillar 2: Magnetic CTQW (M-CTQW) Benchmark with Resumption"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/processed/ctu13_10",
        help="Path to preprocessed CTU-13 graph dataset.",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[42, 43, 44, 45, 46],
        help="List of random seeds to evaluate.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=30,
        help="Number of training epochs per seed.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for graph data loaders.",
    )
    parser.add_argument(
        "--hidden_dim",
        type=int,
        default=32,
        help="Hidden dimension of GNN layers.",
    )
    parser.add_argument(
        "--num_layers",
        type=int,
        default=2,
        help="Number of quantum walk message-passing layers.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.003,
        help="Learning rate for AdamW optimizer.",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.1,
        help="Dropout probability.",
    )
    parser.add_argument(
        "--num_threads",
        type=int,
        default=4,
        help="Number of CPU threads to preserve thermal headroom.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Compute device ('cpu' or 'cuda').",
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default="benchmarks/magnetic_results.json",
        help="Path for final JSON results.",
    )
    parser.add_argument(
        "--output_fig",
        type=str,
        default="benchmarks/figure_magnetic_mctqw.png",
        help="Path for 2-panel publication figure.",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="benchmarks/magnetic_checkpoint.json",
        help="Path for automated run checkpointing.",
    )
    parser.add_argument(
        "--force_restart",
        action="store_true",
        help="If set, ignores any existing checkpoint and starts from scratch.",
    )

    args = parser.parse_args()
    run_magnetic_benchmark(args)


if __name__ == "__main__":
    main()
