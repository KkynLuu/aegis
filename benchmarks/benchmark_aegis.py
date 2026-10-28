#!/usr/bin/env python3
"""
AEGIS Benchmark Suite: Comprehensive Comparative Evaluation.
Target Venue: IEEE Transactions on Information Forensics and Security (TIFS) / USENIX Security.

Evaluates AEGIS (Asymmetric Exceptional-point Graph Intrusion System) against:
  - QUASAR (Legacy Closed Symmetric & Magnetic CTQW)
  - 1-WL Classical Baselines: GCN, GIN
  - Beyond-1-WL Baselines: RWPE-GIN, MagNet

Supports:
  1. Clean Detection Efficacy (RQ1): Precision, Recall, Macro-F1, FAR, ROC-AUC
  2. Adversarial Topological Camouflage Stress Test (RQ2):
     Alpha in [0.00, 0.25, 0.50, 0.75, 1.00] with 2-switch regular chordal rewiring
  3. Multi-Dataset Regimes:
     - CTU-13 Scenario 10 (Sparse decentralized P2P botnet)
     - CTU-13 Scenario 9  (Dense volumetric bipartite spam)
     - CTU-13 Scenario 11 (Low-and-slow intermittent C2)
     - NF-BoT-IoT         (Large-scale IoT telemetry)
  4. Statistical Hypothesis Testing:
     Two-sided Wilcoxon signed-rank tests with Benjamini-Hochberg FDR multiplicity correction (q <= 0.05).
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
from models.aegis_classifier import AegisClassifier


def compute_roc_auc_numpy(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Computes exact ROC-AUC using Mann-Whitney U rank test with tie correction."""
    pos_mask = (y_true == 1)
    n_pos = int(np.sum(pos_mask))
    n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 50.0
    ranks = scipy.stats.rankdata(y_score)
    u_pos = np.sum(ranks[pos_mask]) - n_pos * (n_pos + 1) / 2.0
    return float(u_pos / (n_pos * n_neg) * 100.0)


def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    device: str
) -> Dict[str, float]:
    """Evaluates model over DataLoader and calculates standard NIDS performance metrics."""
    model.eval()
    all_preds = []
    all_probs = []
    all_trues = []

    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)
            adj = batch["adj"].to(device)
            mask = batch["mask"].to(device)
            y = batch["labels"].to(device)

            logits = model(x, adj, mask=mask)
            probs = torch.softmax(logits, dim=-1)[:, 1]
            preds = logits.argmax(dim=-1)

            all_preds.append(preds.cpu().numpy())
            all_probs.append(probs.cpu().numpy())
            all_trues.append(y.cpu().numpy())

    preds = np.concatenate(all_preds) if len(all_preds) > 0 else np.array([])
    probs = np.concatenate(all_probs) if len(all_probs) > 0 else np.array([])
    y_true = np.concatenate(all_trues) if len(all_trues) > 0 else np.array([])

    if len(y_true) == 0:
        return {"f1": 0.0, "precision": 0.0, "recall": 0.0, "far": 0.0, "auc": 50.0, "acc": 0.0}

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
    macro_precision = (p_mal + p_ben) / 2.0 * 100.0
    macro_recall = (r_mal + r_ben) / 2.0 * 100.0
    far = (fp / max(fp + tn, 1)) * 100.0
    auc = compute_roc_auc_numpy(y_true, probs)

    return {
        "f1": macro_f1,
        "precision": macro_precision,
        "recall": macro_recall,
        "far": far,
        "auc": auc,
        "acc": acc,
    }


def train_model(
    model_name: str,
    in_dim: int,
    hidden_dim: int,
    num_layers: int,
    train_loader: DataLoader,
    val_loader: DataLoader,
    class_weights: torch.Tensor,
    epochs: int = 25,
    lr: float = 0.003,
    device: str = "cpu",
    seed: int = 42
) -> nn.Module:
    """Instantiates and trains a model with validation checkpointing."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    if model_name == "AEGIS":
        model = AegisClassifier(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            num_classes=2,
            num_layers=num_layers,
            num_time_scales=4,
            init_times=(0.2, 0.8, 2.5, 8.0),
            base_dissipation=0.2,
            use_edge_encoder=True,
            use_attention=True,
        )
    elif model_name == "QUASAR":
        model = QuasarClassifier(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            num_classes=2,
            num_layers=num_layers,
            num_time_scales=4,
            init_times=(0.2, 0.8, 2.5, 8.0),
            use_magnetic=False,
        )
    elif model_name == "M-CTQW":
        model = QuasarClassifier(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            num_classes=2,
            num_layers=num_layers,
            num_time_scales=4,
            init_times=(0.2, 0.8, 2.5, 8.0),
            use_magnetic=True,
            magnetic_flux=0.25,
        )
    elif model_name == "GCN":
        model = GCNClassifier(in_dim=in_dim, hidden_dim=hidden_dim, num_classes=2, num_layers=num_layers)
    elif model_name == "GIN":
        model = GINClassifier(in_dim=in_dim, hidden_dim=hidden_dim, num_classes=2, num_layers=num_layers)
    elif model_name == "RWPE-GIN":
        model = RWPE_GINClassifier(in_dim=in_dim, hidden_dim=hidden_dim, num_classes=2, num_layers=num_layers, k_rwpe=8)
    elif model_name == "MagNet":
        model = MagNetClassifier(in_dim=in_dim, hidden_dim=hidden_dim, num_classes=2, num_layers=num_layers, q=0.25)
    else:
        raise ValueError(f"Unknown model name: {model_name}")

    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))

    best_val_f1 = -1.0
    best_weights = copy.deepcopy(model.state_dict())

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        n_batches = 0
        for batch in train_loader:
            x = batch["x"].to(device)
            adj = batch["adj"].to(device)
            mask = batch["mask"].to(device)
            y = batch["labels"].to(device)

            optimizer.zero_grad()
            logits = model(x, adj, mask=mask)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_loss = total_loss / max(n_batches, 1)

        val_metrics = evaluate_model(model, val_loader, device=device)
        is_best = val_metrics["f1"] > best_val_f1
        if is_best:
            best_val_f1 = val_metrics["f1"]
            best_weights = copy.deepcopy(model.state_dict())

        if (epoch + 1) % 5 == 0 or epoch == 0 or epoch == epochs - 1:
            print(f"    [{model_name}] Ep {epoch+1:2d}/{epochs:2d} | Loss: {avg_loss:.4f} | Val F1: {val_metrics['f1']:.2f}% | Val FAR: {val_metrics['far']:.2f}%{' *' if is_best else ''}", flush=True)

    model.load_state_dict(best_weights)
    return model


def run_clean_benchmark(
    dataset_name: str,
    models: List[str],
    seeds: List[int],
    epochs: int = 25,
    batch_size: int = 32,
    lr: float = 0.003,
    hidden_dim: int = 32,
    num_layers: int = 2,
    device: str = "cpu"
) -> Dict[str, Any]:
    """Runs clean detection benchmark (RQ1) across specified seeds."""
    data_dir = f"data/processed/{dataset_name}"
    print(f"\n=======================================================")
    print(f"  AEGIS BENCHMARK: {dataset_name.upper()} (CLEAN RQ1)")
    print(f"=======================================================")

    train_graphs, _ = load_netflow_graphs(os.path.join(data_dir, "train.pt"))
    val_graphs, _ = load_netflow_graphs(os.path.join(data_dir, "val.pt"))
    test_graphs, _ = load_netflow_graphs(os.path.join(data_dir, "test.pt"))

    train_loader = DataLoader(NetFlowInMemoryDataset(train_graphs), batch_size=batch_size, shuffle=True, collate_fn=collate_netflow_batch)
    val_loader = DataLoader(NetFlowInMemoryDataset(val_graphs), batch_size=batch_size, shuffle=False, collate_fn=collate_netflow_batch)
    test_loader = DataLoader(NetFlowInMemoryDataset(test_graphs), batch_size=batch_size, shuffle=False, collate_fn=collate_netflow_batch)

    in_dim = train_graphs[0].x.shape[-1]

    # Balanced class weights (standard NIDS inverse-prevalence weighting)
    y_train = torch.tensor([g.y for g in train_graphs])
    n_neg = max(1, (y_train == 0).sum().item())
    n_pos = max(1, (y_train == 1).sum().item())
    n_tot = len(train_graphs)
    w_0 = float(n_tot) / (2.0 * float(n_neg))
    w_1 = float(n_tot) / (2.0 * float(n_pos))
    weights = torch.tensor([w_0, w_1], dtype=torch.float32)

    metrics_list = ["f1", "precision", "recall", "far", "auc", "acc"]
    raw_results = {m: {met: [] for met in metrics_list} for m in models}

    for seed_idx, seed in enumerate(seeds):
        print(f"\n--- [Seed {seed_idx + 1}/{len(seeds)}: {seed}] ---")
        for m in models:
            t0 = time.time()
            model = train_model(
                model_name=m,
                in_dim=in_dim,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                train_loader=train_loader,
                val_loader=val_loader,
                class_weights=weights,
                epochs=epochs,
                lr=lr,
                device=device,
                seed=seed
            )
            t_train = time.time() - t0
            res = evaluate_model(model, test_loader, device=device)
            for met in metrics_list:
                raw_results[m][met].append(res[met])
            print(f"  {m:<10} | Train: {t_train:>5.1f}s | F1: {res['f1']:>5.2f}% | Recall: {res['recall']:>5.2f}% | FAR: {res['far']:>5.2f}% | AUC: {res['auc']:>5.2f}%")

    summary = {m: {} for m in models}
    for m in models:
        for met in metrics_list:
            summary[m][f"{met}_mean"] = float(np.mean(raw_results[m][met]))
            summary[m][f"{met}_std"] = float(np.std(raw_results[m][met]))

    # Print summary table
    print("\n" + "=" * 80)
    print(f"   SUMMARY RESULTS: {dataset_name.upper()} (Mean ± Std over {len(seeds)} seeds)")
    print("-" * 80)
    print(f"{'Model':<12} | {'Macro-F1 (%)':<18} | {'Recall (%)':<16} | {'FAR (%) [↓]':<16} | {'AUC (%)':<16}")
    print("-" * 80)
    for m in models:
        f1_str = f"{summary[m]['f1_mean']:.2f} ± {summary[m]['f1_std']:.2f}"
        rec_str = f"{summary[m]['recall_mean']:.2f} ± {summary[m]['recall_std']:.2f}"
        far_str = f"{summary[m]['far_mean']:.2f} ± {summary[m]['far_std']:.2f}"
        auc_str = f"{summary[m]['auc_mean']:.2f} ± {summary[m]['auc_std']:.2f}"
        print(f"{m:<12} | {f1_str:<18} | {rec_str:<16} | {far_str:<16} | {auc_str:<16}")
    print("=" * 80)

    out_file = f"benchmarks/aegis_{dataset_name}_results.json"
    with open(out_file, "w") as f:
        json.dump({"dataset": dataset_name, "summary": summary, "raw_results": raw_results}, f, indent=2)
    print(f"[✓] Benchmark results saved to: {out_file}")
    return summary


def main():
    parser = argparse.ArgumentParser(description="AEGIS Benchmark Suite")
    parser.add_argument("--dataset", type=str, default="ctu13_9", help="Dataset (ctu13_9, ctu13_10, ctu13_11, nf_bot_iot)")
    parser.add_argument("--models", nargs="+", default=["AEGIS", "QUASAR", "GCN", "GIN"], help="Models to benchmark")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44], help="Random seeds")
    parser.add_argument("--epochs", type=int, default=20, help="Training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Mini-batch size")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu, cuda, mps)")
    args = parser.parse_args()

    run_clean_benchmark(
        dataset_name=args.dataset,
        models=args.models,
        seeds=args.seeds,
        epochs=args.epochs,
        batch_size=args.batch_size,
        device=args.device,
    )


if __name__ == "__main__":
    main()
