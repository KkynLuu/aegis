"""
Benchmark: Multi-Hop Depth Scaling and Over-Smoothing Dirichlet Energy Analysis
Target Venue: IEEE TIFS / IEEE TDSC / npj Quantum Information

Evaluates models across network depths L in {2, 4, 8, 12, 16}:
  1. GCN (Spatial message passing with symmetric degree normalization)
  2. GIN (Graph Isomorphism Network with sum multiset aggregation)
  3. RWPE-GIN (GIN augmented with Random Walk Positional Encodings)
  4. Unitary CTQW (Closed) / QUASAR (Unitary quantum walk with phase transport)
  5. AEGIS (Open Non-Hermitian Quantum Walk with topological point-gap protection)

Core Hypotheses:
  - In classical GNNs (GCN), Markovian diffusion causes exponential Dirichlet
    energy collapse (over-smoothing):
        lim_{L -> inf} E(H^(L)) = 0
    resulting in identical node representations and catastrophic classification collapse.
  - In Unitary CTQW, unitary evolution is norm-preserving but suffers from phase
    scrambling and false-alarm explosions on enterprise interaction graphs.
  - In AEGIS, non-Hermitian open quantum dynamics preserve a non-vanishing Dirichlet
    energy floor (Theorem 2) via complex spectral point gaps and skin-boundary localization,
    sustaining high-precision multi-hop lateral movement detection across deep layers (L = 16).

Saves results to benchmarks/oversmoothing_results.json and generates publication figures.
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
from models.baseline_gnn import (
    GCNClassifier,
    GINClassifier,
    RWPE_GINClassifier,
)
from models.unitary_ctqw_classifier import UnitaryCTQWClassifier as QuasarClassifier
from models.aegis_classifier import AegisClassifier
from benchmarks.benchmark_camouflage import compute_roc_auc_numpy


def compute_dirichlet_energy_batch(
    h: torch.Tensor,
    adj: torch.Tensor,
    mask: Optional[torch.Tensor] = None
) -> float:
    """
    Computes pairwise Dirichlet Energy of node representations:
        E(H) = (1 / (4 * N_real)) * sum_{i,j} A_sym_{ij} * ||h_i - h_j||^2

    When representations over-smooth (h_i -> h_j for all neighbors),
    E(H) strictly converges to zero.
    """
    B, N, d = h.shape
    energies = []

    for b in range(B):
        n_real = int(mask[b].sum().item()) if mask is not None else N
        if n_real <= 1:
            continue

        h_sub = h[b, :n_real, :]
        adj_sub = adj[b, :n_real, :n_real]
        adj_sym = 0.5 * (adj_sub + adj_sub.t())

        # Vectorized pair distance: ||h_i - h_j||^2 = ||h_i||^2 + ||h_j||^2 - 2 <h_i, h_j>
        norm_sq = (h_sub ** 2).sum(dim=-1, keepdim=True)
        dist_sq = norm_sq + norm_sq.t() - 2.0 * torch.mm(h_sub, h_sub.t())
        dist_sq = torch.clamp(dist_sq, min=0.0)

        energy = 0.25 * torch.sum(adj_sym * dist_sq).item() / max(1, n_real)
        energies.append(energy)

    return float(np.mean(energies)) if len(energies) > 0 else 0.0


def find_optimal_threshold(
    model: nn.Module,
    loader: DataLoader,
    device: str
) -> float:
    """Finds optimal classification threshold on validation set to maximize Macro-F1 with zero test leakage."""
    model.eval()
    all_probs = []
    all_trues = []

    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)
            adj = batch["adj"].to(device)
            mask = batch["mask"].to(device)
            y = batch["labels"].to(device)

            out = model(x, adj, mask=mask)
            if isinstance(out, tuple):
                logits = out[0]
            else:
                logits = out

            probs = torch.softmax(logits, dim=-1)[:, 1]
            all_probs.append(probs.cpu().numpy())
            all_trues.append(y.cpu().numpy())

    probs = np.concatenate(all_probs) if len(all_probs) > 0 else np.array([])
    y_true = np.concatenate(all_trues) if len(all_trues) > 0 else np.array([])
    if len(y_true) == 0:
        return 0.50

    best_t, best_f1, best_far = 0.50, -1.0, 100.0
    for t in np.linspace(0.35, 0.65, 31):
        preds = (probs >= t).astype(int)
        tp = int(((preds == 1) & (y_true == 1)).sum())
        fp = int(((preds == 1) & (y_true == 0)).sum())
        fn = int(((preds == 0) & (y_true == 1)).sum())
        tn = int(((preds == 0) & (y_true == 0)).sum())

        p_mal = tp / max(tp + fp, 1)
        r_mal = tp / max(tp + fn, 1)
        f1_mal = 2.0 * p_mal * r_mal / max(p_mal + r_mal, 1e-6)

        p_ben = tn / max(tn + fn, 1)
        r_ben = tn / max(tn + fp, 1)
        f1_ben = 2.0 * p_ben * r_ben / max(p_ben + r_ben, 1e-6)

        macro_f1 = (f1_mal + f1_ben) / 2.0 * 100.0
        far = (fp / max(fp + tn, 1)) * 100.0

        # Primary objective: Macro-F1; Secondary tie-breaker: lower FAR
        if (macro_f1 > best_f1) or (abs(macro_f1 - best_f1) < 1e-5 and far < best_far):
            best_f1 = macro_f1
            best_far = far
            best_t = float(t)

    return best_t


def evaluate_depth_model(
    model: nn.Module,
    loader: DataLoader,
    device: str,
    threshold: float = 0.50
) -> Dict[str, float]:
    """Evaluates classification metrics and Dirichlet Energy on test DataLoader."""
    model.eval()
    all_preds = []
    all_probs = []
    all_trues = []
    initial_energies = []
    final_energies = []

    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)
            adj = batch["adj"].to(device)
            mask = batch["mask"].to(device)
            y = batch["labels"].to(device)

            out = model(x, adj, mask=mask, return_embeddings=True)
            if isinstance(out, tuple):
                logits = out[0]
                h_final = out[1]
            else:
                logits = out
                h_final = x

            probs = torch.softmax(logits, dim=-1)[:, 1]
            preds = (probs >= threshold).long()

            all_preds.append(preds.cpu().numpy())
            all_probs.append(probs.cpu().numpy())
            all_trues.append(y.cpu().numpy())

            # Dirichlet energy of input features and final layer embeddings
            e_init = compute_dirichlet_energy_batch(x, adj, mask=mask)
            e_fin = compute_dirichlet_energy_batch(h_final, adj, mask=mask)
            initial_energies.append(e_init)
            final_energies.append(e_fin)

    preds = np.concatenate(all_preds) if len(all_preds) > 0 else np.array([])
    probs = np.concatenate(all_probs) if len(all_probs) > 0 else np.array([])
    y_true = np.concatenate(all_trues) if len(all_trues) > 0 else np.array([])

    if len(y_true) == 0:
        return {"f1": 0.0, "far": 0.0, "dirichlet_ratio": 0.0, "dirichlet_final": 0.0}

    acc = float((preds == y_true).mean() * 100.0)
    tp = int(((preds == 1) & (y_true == 1)).sum())
    fp = int(((preds == 1) & (y_true == 0)).sum())
    fn = int(((preds == 0) & (y_true == 1)).sum())
    tn = int(((preds == 0) & (y_true == 0)).sum())

    p_mal = tp / max(tp + fp, 1)
    r_mal = tp / max(tp + fn, 1)
    f1_mal = 2.0 * p_mal * r_mal / max(p_mal + r_mal, 1e-6)

    p_ben = tn / max(tn + fn, 1)
    r_ben = tn / max(tn + fp, 1)
    f1_ben = 2.0 * p_ben * r_ben / max(p_ben + r_ben, 1e-6)

    macro_f1 = (f1_mal + f1_ben) / 2.0 * 100.0
    far = (fp / max(fp + tn, 1)) * 100.0
    auc = compute_roc_auc_numpy(y_true, probs)

    avg_e_init = float(np.mean(initial_energies)) if len(initial_energies) > 0 else 1.0
    avg_e_fin = float(np.mean(final_energies)) if len(final_energies) > 0 else 0.0
    dirichlet_ratio = avg_e_fin / max(1e-8, avg_e_init)

    return {
        "f1": macro_f1,
        "far": far,
        "auc": auc,
        "acc": acc,
        "dirichlet_final": avg_e_fin,
        "dirichlet_init": avg_e_init,
        "dirichlet_ratio": dirichlet_ratio,
        "threshold": threshold,
    }


def train_depth_model(
    model_name: str,
    num_layers: int,
    in_dim: int,
    hidden_dim: int,
    train_loader: DataLoader,
    val_loader: DataLoader,
    class_weights: torch.Tensor,
    epochs: int = 35,
    lr: float = 0.003,
    weight_decay: float = 1e-4,
    device: str = "cpu",
    seed: int = 42,
) -> Tuple[nn.Module, float]:
    """Trains a model with a specified depth L and calibrates the decision threshold."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    if model_name == "GCN":
        model = GCNClassifier(in_dim=in_dim, hidden_dim=hidden_dim, num_classes=2, num_layers=num_layers)
    elif model_name == "GIN":
        model = GINClassifier(in_dim=in_dim, hidden_dim=hidden_dim, num_classes=2, num_layers=num_layers)
    elif model_name == "RWPE-GIN":
        model = RWPE_GINClassifier(in_dim=in_dim, hidden_dim=hidden_dim, num_classes=2, num_layers=num_layers, k_rwpe=8)
    elif model_name in ("QUASAR", "Unitary CTQW (Closed)"):
        model = QuasarClassifier(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            num_classes=2,
            num_layers=num_layers,
            num_time_scales=4,
            init_times=(0.2, 0.8, 2.5, 8.0),
            use_potential=True,
            use_attention=True,
        )
    elif model_name in ("AEGIS", "AEGIS-MoE (Ours)", "AEGIS (Ours)"):
        model = AegisClassifier(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            num_classes=2,
            num_layers=num_layers,
            num_time_scales=4,
            init_times=(0.2, 0.8, 2.5, 8.0),
            base_dissipation=0.2,
            adaptive_dissipation=True,
            dropout=0.1,
            residual=True,
            use_edge_encoder=True,
            use_potential=True,
            use_attention=True,
            use_volumetric=True,
            use_moe=True,
            moe_routing_mode="softmax",
        )
    else:
        raise ValueError(f"Unknown model name: {model_name}")

    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))

    best_val_f1 = -1.0
    best_weights = copy.deepcopy(model.state_dict())

    for epoch in range(1, epochs + 1):
        model.train()
        for batch in train_loader:
            x = batch["x"].to(device)
            adj = batch["adj"].to(device)
            mask = batch["mask"].to(device)
            y = batch["labels"].to(device)

            optimizer.zero_grad()
            out = model(x, adj, mask=mask)
            logits = out[0] if isinstance(out, tuple) else out
            loss = criterion(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        scheduler.step()

        # Validation checkpointing every 2 epochs and final epoch
        if epoch % 2 == 0 or epoch == epochs:
            val_metrics = evaluate_depth_model(model, val_loader, device=device, threshold=0.50)
            if val_metrics["f1"] > best_val_f1:
                best_val_f1 = val_metrics["f1"]
                best_weights = copy.deepcopy(model.state_dict())

    # Load best checkpoint and calibrate optimal decision threshold
    model.load_state_dict(best_weights)
    best_tau = find_optimal_threshold(model, val_loader, device=device)
    return model, best_tau


def run_oversmoothing_benchmark(args: argparse.Namespace) -> None:
    device = args.device
    if device == "mps" and not torch.backends.mps.is_available():
        print("[!] MPS device requested but not available. Falling back to CPU.")
        device = "cpu"

    if device == "cpu":
        torch.set_num_threads(args.num_threads)
        print(f"[*] Execution Device: CPU | PyTorch Threads: {args.num_threads}")
    else:
        print(f"[*] Execution Device: {device.upper()} (Hardware Accelerated)")

    if args.smoke_test:
        print("[*] Running in SMOKE TEST mode (2 seeds, 3 epochs, depths [2, 4])...")
        seeds = [42, 43]
        depths = [2, 4]
        epochs = 3
    else:
        seeds = [42 + i for i in range(args.num_seeds)]
        depths = args.depths
        epochs = args.epochs

    # Map model aliases
    canonical_models = []
    for m in args.models:
        if m in ("QUASAR", "Unitary CTQW", "Unitary CTQW (Closed)"):
            canonical_models.append("QUASAR")
        elif m in ("AEGIS", "AEGIS-MoE (Ours)", "AEGIS (Ours)"):
            canonical_models.append("AEGIS")
        else:
            canonical_models.append(m)

    print("=" * 115)
    print("      EXPERIMENT 3: MULTI-HOP DEPTH SCALING & OVER-SMOOTHING DIRICHLET ENERGY BENCHMARK")
    print(f"      Dataset:     CTU13_10 ({args.data_dir})")
    print(f"      Models:      {canonical_models}")
    print(f"      Depths (L):  {depths}")
    print(f"      Seeds:       {len(seeds)} {seeds}")
    print(f"      Epochs: {epochs} | Batch Size: {args.batch_size} | LR: {args.lr} | Device: {device}")
    print("=" * 115)

    # 1. Load Data
    train_graphs, _ = load_netflow_graphs(os.path.join(args.data_dir, "train.pt"))
    val_graphs, _ = load_netflow_graphs(os.path.join(args.data_dir, "val.pt"))
    test_graphs, _ = load_netflow_graphs(os.path.join(args.data_dir, "test.pt"))

    if args.smoke_test:
        train_graphs = [g for g in train_graphs if g.y == 0][:32] + [g for g in train_graphs if g.y == 1][:32]
        val_graphs = [g for g in val_graphs if g.y == 0][:16] + [g for g in val_graphs if g.y == 1][:16]
        test_graphs = [g for g in test_graphs if g.y == 0][:16] + [g for g in test_graphs if g.y == 1][:16]

    n_tot = len(train_graphs)
    n_neg = max(1, sum(1 - g.y for g in train_graphs))
    n_pos = max(1, sum(g.y for g in train_graphs))
    w_0 = float(n_tot) / (2.0 * float(n_neg))
    w_1 = float(n_tot) / (2.0 * float(n_pos))
    class_weights = torch.tensor([w_0, w_1], dtype=torch.float32)

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
    test_loader = DataLoader(
        NetFlowInMemoryDataset(test_graphs),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_netflow_batch,
    )

    in_dim = train_graphs[0].x.shape[-1]
    hidden_dim = args.hidden_dim

    # 2. Check for existing cached results
    results: Dict[str, Dict[str, List[Dict[str, float]]]] = {}
    summary: Dict[str, Dict[str, Dict[str, float]]] = {}
    if os.path.exists(args.output_json) and not args.force_rerun:
        try:
            with open(args.output_json, "r") as f:
                cached_data = json.load(f)
            if "raw_results" in cached_data:
                results = cached_data["raw_results"]
            if "summary" in cached_data:
                summary = cached_data["summary"]
            print(f"[*] Loaded existing cached benchmark results from {args.output_json}")
            print(f"    Cached models: {list(results.keys())}")
        except Exception as e:
            print(f"[!] Could not load cache: {e}. Starting fresh.")
            results = {}
            summary = {}

    start_total_time = time.time()

    for L in depths:
        str_L = str(L)
        print(f"\n====================== LAYER DEPTH L = {L} ======================")
        for m_name in canonical_models:
            if m_name not in results:
                results[m_name] = {}
            if str_L not in results[m_name]:
                results[m_name][str_L] = []

            # Check if this depth has already been evaluated for all seeds
            if len(results[m_name][str_L]) >= len(seeds) and not args.force_rerun:
                print(f"  --> {m_name:<10} (L={L:>2}): Already evaluated across {len(seeds)} seeds. Skipping.")
                continue

            # Need to evaluate
            results[m_name][str_L] = []
            for seed_idx, seed in enumerate(seeds):
                t0 = time.time()
                model, tau_star = train_depth_model(
                    model_name=m_name,
                    num_layers=L,
                    in_dim=in_dim,
                    hidden_dim=hidden_dim,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    class_weights=class_weights,
                    epochs=epochs,
                    lr=args.lr,
                    weight_decay=args.weight_decay,
                    device=device,
                    seed=seed,
                )
                train_dur = time.time() - t0
                test_metrics = evaluate_depth_model(model, test_loader, device=device, threshold=tau_star)
                results[m_name][str_L].append(test_metrics)

                print(
                    f"  {m_name:<10} (L={L:>2}, Seed {seed_idx + 1}/{len(seeds)}) | "
                    f"Train: {train_dur:>5.1f}s | tau*: {tau_star:.2f} | "
                    f"F1: {test_metrics['f1']:>5.2f}% | "
                    f"FAR: {test_metrics['far']:>4.2f}% | "
                    f"Dirichlet Ratio R_E: {test_metrics['dirichlet_ratio']:>8.4f}",
                    flush=True
                )

            # Recompute summary for this (m_name, L)
            if m_name not in summary:
                summary[m_name] = {}
            m_list = results[m_name][str_L]
            depth_summary = {}
            for metric in ["f1", "far", "auc", "acc", "dirichlet_ratio", "dirichlet_final"]:
                vals = [m[metric] for m in m_list]
                depth_summary[f"{metric}_mean"] = float(np.mean(vals))
                depth_summary[f"{metric}_std"] = float(np.std(vals))
            summary[m_name][str_L] = depth_summary

            # Checkpoint output JSON immediately
            _save_checkpoint(args, results, summary, time.time() - start_total_time)

    total_runtime = time.time() - start_total_time

    # 4. Final Aggregation & Verification
    for m_name in results:
        if m_name not in summary:
            summary[m_name] = {}
        for str_L in results[m_name]:
            m_list = results[m_name][str_L]
            if len(m_list) > 0:
                depth_summary = {}
                for metric in ["f1", "far", "auc", "acc", "dirichlet_ratio", "dirichlet_final"]:
                    vals = [m[metric] for m in m_list]
                    depth_summary[f"{metric}_mean"] = float(np.mean(vals))
                    depth_summary[f"{metric}_std"] = float(np.std(vals))
                summary[m_name][str_L] = depth_summary

    # Save final JSON
    _save_checkpoint(args, results, summary, total_runtime)
    print(f"\n[✓] Results successfully saved to: {args.output_json}")

    # Print Formatted Results Table
    all_summary_models = list(summary.keys())
    print("\n" + "=" * 125)
    print("           MULTI-HOP DEPTH SCALING & OVER-SMOOTHING DIRICHLET ENERGY BENCHMARK")
    print("-" * 125)
    print(f"{'Model':<12} | {'Depth (L)':<10} | {'Macro-F1 (%)':<20} | {'Dirichlet Ratio (E/E0)':<25} | {'FAR (%) [↓]':<15}")
    print("-" * 125)
    for L in depths:
        str_L = str(L)
        for m_name in all_summary_models:
            if str_L in summary[m_name]:
                f1_str = f"{summary[m_name][str_L]['f1_mean']:.2f} ± {summary[m_name][str_L]['f1_std']:.2f}"
                e_str = f"{summary[m_name][str_L]['dirichlet_ratio_mean']:.4f} ± {summary[m_name][str_L]['dirichlet_ratio_std']:.4f}"
                far_str = f"{summary[m_name][str_L]['far_mean']:.2f} ± {summary[m_name][str_L]['far_std']:.2f}"
                disp_name = "Unitary CTQW" if m_name == "QUASAR" else m_name
                print(f"{disp_name:<12} | L={L:<8} | {f1_str:<20} | {e_str:<25} | {far_str:<15}")
        print("-" * 125)
    print("=" * 125)

    # Generate Publication Figure
    plot_oversmoothing_figure(depths, summary, args.output_fig)
    print(f"[✓] Publication figure saved to: {args.output_fig}")
    paper_fig_png = "paper/figures/figure_oversmoothing_depth.png"
    paper_fig_pdf = "paper/figures/figure_oversmoothing_depth.pdf"
    plot_oversmoothing_figure(depths, summary, paper_fig_png)
    plot_oversmoothing_figure(depths, summary, paper_fig_pdf)
    print(f"[✓] Paper figures updated in {paper_fig_png} and {paper_fig_pdf}")

    # Generate LaTeX Table
    latex_table = generate_oversmoothing_latex_table(depths, summary)
    print("\n--- IEEE TIFS Publication-Ready LaTeX Table (Table IV) ---")
    print(latex_table)
    print("-" * 60)


def _save_checkpoint(
    args: argparse.Namespace,
    results: Dict[str, Dict[str, List[Dict[str, float]]]],
    summary: Dict[str, Dict[str, Dict[str, float]]],
    total_runtime: float,
) -> None:
    output_payload = {
        "metadata": {
            "dataset": args.dataset,
            "data_dir": args.data_dir,
            "models": list(results.keys()),
            "depths": args.depths,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "total_runtime_sec": total_runtime,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        "summary": summary,
        "raw_results": results,
    }
    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(output_payload, f, indent=2)


def generate_oversmoothing_latex_table(
    depths: List[int],
    summary: Dict[str, Dict[str, Dict[str, float]]]
) -> str:
    """Generates publication-ready IEEE LaTeX code matching Table IV in main.tex."""
    model_order = [
        ("GCN", "GCN"),
        ("GIN", "GIN"),
        ("RWPE-GIN", "RWPE-GIN"),
        ("QUASAR", "Unitary CTQW (Closed)"),
        ("AEGIS", "\\textbf{AEGIS (Ours)}"),
    ]

    available_models = [m for m in model_order if m[0] in summary]
    col_headers = " & ".join([f"\\textbf{{{name}}}" for _, name in available_models])

    blocks = []
    for L in depths:
        str_L = str(L)
        f1_vals, far_vals, re_vals = [], [], []

        for key, _ in available_models:
            if str_L in summary[key]:
                f1_m = summary[key][str_L]["f1_mean"]
                f1_s = summary[key][str_L]["f1_std"]
                far_m = summary[key][str_L]["far_mean"]
                far_s = summary[key][str_L]["far_std"]
                re_m = summary[key][str_L]["dirichlet_ratio_mean"]
                re_s = summary[key][str_L]["dirichlet_ratio_std"]

                if key == "AEGIS":
                    f1_vals.append(f"\\mathbf{{{f1_m:.2f} \\pm {f1_s:.2f}}}")
                    far_vals.append(f"\\mathbf{{{far_m:.2f} \\pm {far_s:.2f}}}")
                    re_vals.append(f"\\mathbf{{{re_m:.3f} \\pm {re_s:.3f}}}")
                else:
                    f1_vals.append(f"${f1_m:.2f} \\pm {f1_s:.2f}$")
                    far_vals.append(f"${far_m:.2f} \\pm {far_s:.2f}$")
                    re_vals.append(f"${re_m:.3f} \\pm {re_s:.3f}$" if re_m < 10.0 else f"${re_m:.1f}$")
            else:
                f1_vals.append("--")
                far_vals.append("--")
                re_vals.append("--")

        block = f"""$L = {L}$ & Macro-F1 (\\%) [$\\uparrow$] & {' & '.join(f1_vals)} \\\\
& FAR (\\%) [$\\downarrow$] & {' & '.join(far_vals)} \\\\
& Dirichlet Ratio $\\mathcal{{R}}_{{\\mathcal{{E}}}}$ & {' & '.join(re_vals)} \\\\"""
        blocks.append(block)

    table_body = "\n\\midrule\n".join(blocks)

    return f"""\\begin{{table*}}[t]
\\centering
\\caption{{Multi-Hop Depth Scaling and Dirichlet Energy Stability Across Layer Depths $L \\in \\{{{', '.join(map(str, depths))}\\}}$}}
\\label{{tab:depth_scaling}}
\\small
\\resizebox{{\\textwidth}}{{!}}{{
\\begin{{tabular}}{{clccccc}}
\\toprule
\\textbf{{Depth ($L$)}} & \\textbf{{Metric}} & {col_headers} \\\\
\\midrule
{table_body}
\\bottomrule
\\end{{tabular}}
}}
\\end{{table*}}"""


def plot_oversmoothing_figure(
    depths: List[int],
    summary: Dict[str, Dict[str, Dict[str, float]]],
    output_path: str
) -> None:
    """Generates a publication-quality 2-panel figure showing over-smoothing dynamics."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.4), dpi=300)
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.size"] = 11

    colors = {
        "GCN": "#d95f02",
        "GIN": "#7570b3",
        "RWPE-GIN": "#e7298a",
        "QUASAR": "#66a61e",
        "AEGIS": "#1f78b4",
    }
    labels = {
        "GCN": "GCN (Markovian Diffusion)",
        "GIN": "GIN (Xu et al. Multiset)",
        "RWPE-GIN": "RWPE-GIN (Positional Encoding)",
        "QUASAR": "Unitary CTQW (Closed Walk)",
        "AEGIS": "AEGIS (Non-Hermitian Open Walk)",
    }
    markers = {
        "GCN": "s",
        "GIN": "^",
        "RWPE-GIN": "D",
        "QUASAR": "v",
        "AEGIS": "o",
    }

    # Panel A: Dirichlet Energy Ratio vs Depth
    # Note: Focus on bounded/non-exploding models (GCN, Unitary CTQW, AEGIS) for clear visual comparison
    dirichlet_models = [m for m in ["GCN", "QUASAR", "AEGIS"] if m in summary]
    for m in dirichlet_models:
        means = [summary[m][str(L)]["dirichlet_ratio_mean"] for L in depths if str(L) in summary[m]]
        stds = [summary[m][str(L)]["dirichlet_ratio_std"] for L in depths if str(L) in summary[m]]
        eval_depths = [L for L in depths if str(L) in summary[m]]

        ax1.plot(
            eval_depths,
            means,
            marker=markers.get(m, "o"),
            color=colors.get(m, "black"),
            linewidth=2.8 if m == "AEGIS" else 2.0,
            markersize=9 if m == "AEGIS" else 7,
            label=labels.get(m, m),
            zorder=5 if m == "AEGIS" else 3,
        )
        ax1.fill_between(
            eval_depths,
            np.maximum(0, np.array(means) - np.array(stds)),
            np.array(means) + np.array(stds),
            color=colors.get(m, "black"),
            alpha=0.18 if m == "AEGIS" else 0.12,
        )

    ax1.set_xlabel("Network Layer Depth ($L$)", fontweight="bold", fontsize=12)
    ax1.set_ylabel("Dirichlet Energy Ratio $\\mathcal{R}_{\\mathcal{E}} = \\mathcal{E}(\\mathbf{H}^{(L)})/\\mathcal{E}(\\mathbf{H}^{(0)})$", fontweight="bold", fontsize=11)
    ax1.set_title("(a) Over-Smoothing: Dirichlet Energy Stability", fontweight="bold", fontsize=12, pad=12)
    ax1.set_xticks(depths)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(frameon=True, facecolor="white", edgecolor="none", loc="best", fontsize=10)

    # Panel B: Macro-F1 vs Depth across ALL models
    all_models = [m for m in ["GCN", "GIN", "RWPE-GIN", "QUASAR", "AEGIS"] if m in summary]
    for m in all_models:
        means = [summary[m][str(L)]["f1_mean"] for L in depths if str(L) in summary[m]]
        stds = [summary[m][str(L)]["f1_std"] for L in depths if str(L) in summary[m]]
        eval_depths = [L for L in depths if str(L) in summary[m]]

        ax2.plot(
            eval_depths,
            means,
            marker=markers.get(m, "o"),
            color=colors.get(m, "black"),
            linewidth=2.8 if m == "AEGIS" else 2.0,
            markersize=9 if m == "AEGIS" else 7,
            label=labels.get(m, m),
            zorder=5 if m == "AEGIS" else 3,
        )
        ax2.fill_between(
            eval_depths,
            np.array(means) - np.array(stds),
            np.array(means) + np.array(stds),
            color=colors.get(m, "black"),
            alpha=0.18 if m == "AEGIS" else 0.12,
        )

    ax2.set_xlabel("Network Layer Depth ($L$)", fontweight="bold", fontsize=12)
    ax2.set_ylabel("Detection Macro-F1 (%)", fontweight="bold", fontsize=12)
    ax2.set_title("(b) Multi-Hop Lateral Movement Detection", fontweight="bold", fontsize=12, pad=12)
    ax2.set_xticks(depths)
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(frameon=True, facecolor="white", edgecolor="none", loc="lower left", fontsize=10)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Multi-Hop Depth Scaling and Over-Smoothing Dirichlet Energy Benchmark"
    )
    parser.add_argument("--dataset", type=str, default="ctu13_10", choices=["ctu13_10", "ctu13_9", "nf_bot_iot"])
    parser.add_argument("--data_dir", type=str, default="data/processed/ctu13_10")
    parser.add_argument("--models", nargs="+", default=["AEGIS"], help="Models to evaluate (default: AEGIS)")
    parser.add_argument("--depths", nargs="+", type=int, default=[2, 4, 8, 12, 16], help="Depths L to evaluate")
    parser.add_argument("--num_seeds", type=int, default=5, help="Number of random seeds (default: 5)")
    parser.add_argument("--epochs", type=int, default=35, help="Training epochs per seed (default: 35)")
    parser.add_argument("--batch_size", type=int, default=32, help="Mini-batch size (default: 32)")
    parser.add_argument("--lr", type=float, default=0.003, help="Learning rate (default: 0.003)")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay (default: 1e-4)")
    parser.add_argument("--hidden_dim", type=int, default=32, help="Hidden dimension (default: 32)")
    parser.add_argument("--device", type=str, default="mps" if torch.backends.mps.is_available() else "cpu")
    parser.add_argument("--num_threads", type=int, default=8, help="PyTorch CPU threads (default: 8)")
    parser.add_argument("--output_json", type=str, default="benchmarks/oversmoothing_results.json")
    parser.add_argument("--output_fig", type=str, default="benchmarks/figure_oversmoothing_depth.png")
    parser.add_argument("--smoke_test", action="store_true", help="Quick smoke test (2 seeds, 3 epochs, L=[2, 4])")
    parser.add_argument("--force_rerun", action="store_true", help="Force rerun even if results exist in cache")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_oversmoothing_benchmark(args)
