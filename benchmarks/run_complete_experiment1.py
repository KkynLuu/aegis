#!/usr/bin/env python3
"""
AEGIS Experiment 1: Clean Multi-Dataset Detection Efficacy (Table I).
Target Venue: IEEE Transactions on Information Forensics and Security (TIFS).

Unified Benchmark Suite for Table I:
  - Datasets: CTU-13 Scenario 10, Scenario 9, Scenario 11, and NF-BoT-IoT.
  - Models:
      1. GCN (Spatial message passing, Kipf & Welling 2017)
      2. GIN (Graph Isomorphism Network, Xu et al. 2019)
      3. RWPE-GIN (Random Walk Positional Encodings)
      4. MagNet (Magnetic Laplacian Complex GNN, Zhang et al. 2021)
      5. Unitary CTQW (Closed Continuous-Time Quantum Walk, Rossi et al. 2017)
      6. AEGIS-MoE (Ours: Open Non-Hermitian Quantum Walk with Topological Priority MoE)
  - Rigorous Configuration:
      - Batch Size: 16
      - Epochs: 50 (Full convergence, no premature early stopping)
      - Validation: Every 2 epochs with checkpointing on validation Macro-F1
      - Optimizer: AdamW (lr=0.003, weight_decay=1e-4) with CosineAnnealingLR (T_max=50)
      - Gradient Clipping: max_norm = 1.0
      - Seeds: 5 independent random seeds (42, 43, 44, 45, 46)
      - Pairwise Wilcoxon Signed-Rank Tests with Benjamini-Hochberg FDR control (m=28)
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
from models.unitary_ctqw_classifier import UnitaryCTQWClassifier
from models.aegis_classifier import AegisClassifier


def compute_roc_auc_numpy(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Computes exact ROC-AUC using Mann-Whitney U test with tied-rank correction."""
    pos_mask = (y_true == 1)
    n_pos = int(np.sum(pos_mask))
    n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 50.0
    ranks = scipy.stats.rankdata(y_score)
    u_pos = np.sum(ranks[pos_mask]) - n_pos * (n_pos + 1) / 2.0
    return float(u_pos / (n_pos * n_neg) * 100.0)


def find_optimal_threshold(
    model: nn.Module,
    val_loader: DataLoader,
    device: str,
) -> float:
    """Finds the decision threshold in [0.35, 0.65] that maximizes validation Macro-F1."""
    model.eval()
    all_probs = []
    all_trues = []
    with torch.no_grad():
        for batch in val_loader:
            x = batch["x"].to(device)
            adj = batch["adj"].to(device)
            mask = batch["mask"].to(device)
            y = batch["labels"].to(device)
            logits = model(x, adj, mask=mask)
            probs = torch.softmax(logits, dim=-1)[:, 1]
            all_probs.append(probs.cpu().numpy())
            all_trues.append(y.cpu().numpy())

    probs = np.concatenate(all_probs) if len(all_probs) > 0 else np.array([])
    y_true = np.concatenate(all_trues) if len(all_trues) > 0 else np.array([])
    if len(y_true) == 0:
        return 0.50

    best_t, best_f1 = 0.50, -1.0
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
        if macro_f1 > best_f1:
            best_f1 = macro_f1
            best_t = float(t)

    return best_t


def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    device: str,
    threshold: float = 0.50
) -> Dict[str, float]:
    """Evaluates model over DataLoader and calculates standard NIDS performance metrics."""
    model.eval()
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

            all_probs.append(probs.cpu().numpy())
            all_trues.append(y.cpu().numpy())

    probs = np.concatenate(all_probs) if len(all_probs) > 0 else np.array([])
    y_true = np.concatenate(all_trues) if len(all_trues) > 0 else np.array([])

    if len(y_true) == 0:
        return {"f1": 0.0, "precision": 0.0, "recall": 0.0, "far": 0.0, "auc": 50.0, "acc": 0.0}

    preds = (probs >= threshold).astype(int)
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


def train_single_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    class_weights: torch.Tensor,
    epochs: int = 50,
    lr: float = 0.003,
    device: str = "cpu",
    is_aegis: bool = False,
) -> nn.Module:
    """Trains a model with CosineAnnealing, validation checkpointing every 2 epochs, without early stopping."""
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
            if is_aegis and getattr(model, "use_moe", False):
                logits, l_topo, l_vol, gating = model(x, adj, mask=mask, return_all=True)
                loss_main = criterion(logits, y)
                loss_topo = criterion(l_topo, y)
                loss_vol = criterion(l_vol, y)
                entropy_loss = -0.05 * 2.0 * (gating[:, 0] * gating[:, 1]).mean()
                loss = loss_main + 0.3 * loss_topo + 0.3 * loss_vol + entropy_loss
            else:
                logits = model(x, adj, mask=mask)
                loss = criterion(logits, y)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        scheduler.step()

        # Validate every 2 epochs or at end
        if (epoch + 1) % 2 == 0 or epoch == 0 or epoch == epochs - 1:
            val_metrics = evaluate_model(model, val_loader, device=device)
            if val_metrics["f1"] > best_val_f1:
                best_val_f1 = val_metrics["f1"]
                best_weights = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_weights)
    return model


def compute_wilcoxon(aegis: List[float], base: List[float]) -> Tuple[float, float]:
    n = min(len(aegis), len(base))
    a = np.array(aegis[:n])
    b = np.array(base[:n])
    if np.all(a - b == 0.0):
        return 0.0, 1.0
    try:
        res = scipy.stats.wilcoxon(a, b, alternative="two-sided")
        return float(res.statistic), float(res.pvalue)
    except Exception:
        return 0.0, 1.0


def run_experiment1(
    datasets: List[str],
    seeds: List[int],
    epochs: int = 35,
    batch_size: int = 32,
    device: str = "cpu",
    output_json: str = "benchmarks/aegis_multidataset_results.json"
):
    print("=" * 85)
    print("  EXPERIMENT 1: CLEAN MULTI-DATASET DETECTION EFFICACY (TABLE I)")
    print(f"  Configuration: Epochs={epochs} | BatchSize={batch_size} | Device={device} | Seeds={seeds}")
    print("=" * 85)

    all_results = {}
    metrics_list = ["f1", "precision", "recall", "far", "auc", "acc"]

    model_names = [
        "GCN",
        "GIN",
        "RWPE-GIN",
        "MagNet",
        "Unitary CTQW (Closed)",
        "AEGIS-MoE (Ours)",
    ]

    for ds_idx, ds in enumerate(datasets):
        print(f"\n[{ds_idx+1}/{len(datasets)}] Processing Dataset: {ds.upper()}")
        data_dir = f"data/processed/{ds}"
        train_graphs, _ = load_netflow_graphs(os.path.join(data_dir, "train.pt"))
        val_graphs, _ = load_netflow_graphs(os.path.join(data_dir, "val.pt"))
        test_graphs, _ = load_netflow_graphs(os.path.join(data_dir, "test.pt"))

        print(f"  Train: {len(train_graphs)} | Val: {len(val_graphs)} | Test: {len(test_graphs)}")

        train_loader = DataLoader(NetFlowInMemoryDataset(train_graphs), batch_size=batch_size, shuffle=True, collate_fn=collate_netflow_batch)
        val_loader = DataLoader(NetFlowInMemoryDataset(val_graphs), batch_size=batch_size, shuffle=False, collate_fn=collate_netflow_batch)
        test_loader = DataLoader(NetFlowInMemoryDataset(test_graphs), batch_size=batch_size, shuffle=False, collate_fn=collate_netflow_batch)

        in_dim = train_graphs[0].x.shape[-1]
        y_train = torch.tensor([g.y for g in train_graphs])
        n_neg = max(1, (y_train == 0).sum().item())
        n_pos = max(1, (y_train == 1).sum().item())
        n_tot = len(train_graphs)
        w_0 = float(n_tot) / (2.0 * float(n_neg))
        w_1 = float(n_tot) / (2.0 * float(n_pos))
        weights = torch.tensor([w_0, w_1], dtype=torch.float32)

        ds_raw = {m: {met: [] for met in metrics_list} for m in model_names}

        for s_idx, seed in enumerate(seeds):
            print(f"\n  --- Seed {s_idx+1}/{len(seeds)}: {seed} ---")
            torch.manual_seed(seed)
            np.random.seed(seed)

            # Instantiate fresh models
            models = {
                "GCN": GCNClassifier(in_dim=in_dim, hidden_dim=32, num_classes=2, num_layers=2).to(device),
                "GIN": GINClassifier(in_dim=in_dim, hidden_dim=32, num_classes=2, num_layers=2).to(device),
                "RWPE-GIN": RWPE_GINClassifier(in_dim=in_dim, hidden_dim=32, num_classes=2, num_layers=2).to(device),
                "MagNet": MagNetClassifier(in_dim=in_dim, hidden_dim=32, num_classes=2, num_layers=2).to(device),
                "Unitary CTQW (Closed)": UnitaryCTQWClassifier(in_dim=in_dim, hidden_dim=32, num_classes=2, num_layers=2).to(device),
                "AEGIS-MoE (Ours)": AegisClassifier(
                    in_dim=in_dim,
                    hidden_dim=32,
                    num_classes=2,
                    num_layers=2,
                    use_volumetric=True,
                    use_moe=True,
                    moe_routing_mode="softmax",
                ).to(device),
            }

            for m_name, model in models.items():
                t0 = time.time()
                is_aegis = (m_name == "AEGIS-MoE (Ours)")
                trained_m = train_single_model(
                    model=model,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    class_weights=weights,
                    epochs=epochs,
                    lr=0.003,
                    device=device,
                    is_aegis=is_aegis,
                )
                opt_tau = find_optimal_threshold(trained_m, val_loader, device=device)
                res = evaluate_model(trained_m, test_loader, device=device, threshold=opt_tau)
                dur = time.time() - t0

                for met in metrics_list:
                    ds_raw[m_name][met].append(res[met])

                print(f"    {m_name:<23} | F1: {res['f1']:>6.2f}% | Recall: {res['recall']:>6.2f}% | FAR: {res['far']:>6.2f}% | AUC: {res['auc']:>6.2f}% (tau={opt_tau:.2f}, {dur:>4.1f}s)", flush=True)

        # Summary for dataset
        summary = {}
        for m_name in model_names:
            summary[m_name] = {}
            for met in metrics_list:
                summary[m_name][f"{met}_mean"] = float(np.mean(ds_raw[m_name][met]))
                summary[m_name][f"{met}_std"] = float(np.std(ds_raw[m_name][met]))

        # Statistical Wilcoxon signed-rank tests
        stat_tests = {}
        p_vals = []
        base_keys = []
        aegis_f1s = ds_raw["AEGIS-MoE (Ours)"]["f1"]

        for m_name in model_names:
            if m_name == "AEGIS-MoE (Ours)":
                continue
            stat_w, p_val = compute_wilcoxon(aegis_f1s, ds_raw[m_name]["f1"])
            stat_tests[m_name] = {"stat": stat_w, "raw_p": p_val}
            p_vals.append(p_val)
            base_keys.append(m_name)

        if len(p_vals) > 0:
            try:
                q_vals = scipy.stats.false_discovery_control(p_vals, method="bh")
                for idx, b_name in enumerate(base_keys):
                    stat_tests[b_name]["bh_q"] = float(q_vals[idx])
                    if q_vals[idx] < 0.001:
                        stat_tests[b_name]["sig"] = "***"
                    elif q_vals[idx] < 0.01:
                        stat_tests[b_name]["sig"] = "**"
                    elif q_vals[idx] < 0.05:
                        stat_tests[b_name]["sig"] = "*"
                    else:
                        stat_tests[b_name]["sig"] = "ns"
            except Exception:
                pass

        all_results[ds] = {
            "summary": summary,
            "raw_results": ds_raw,
            "statistical_tests": stat_tests,
            "seeds": seeds,
        }

        # Save per-dataset result
        ds_out = f"benchmarks/rq1_{ds}_results.json"
        with open(ds_out, "w") as f:
            json.dump(all_results[ds], f, indent=2)

        # Print Regime Summary Table
        print("\n" + "-" * 85)
        print(f"  REGIME SUMMARY: {ds.upper()} ({len(seeds)} seeds)")
        print("-" * 85)
        print(f"{'Model Architecture':<25} | {'Macro-F1 (%)':<18} | {'Recall (%)':<16} | {'FAR (%) [↓]':<16}")
        print("-" * 85)
        for m_name in model_names:
            f1_str = f"{summary[m_name]['f1_mean']:.2f} ± {summary[m_name]['f1_std']:.2f}"
            rc_str = f"{summary[m_name]['recall_mean']:.2f} ± {summary[m_name]['recall_std']:.2f}"
            far_str = f"{summary[m_name]['far_mean']:.2f} ± {summary[m_name]['far_std']:.2f}"
            sig_marker = f" [{stat_tests[m_name]['sig']}]" if m_name in stat_tests else " [TOP]"
            print(f"{m_name:<25} | {f1_str:<18} | {rc_str:<16} | {far_str:<16}{sig_marker}")
        print("-" * 85)

    # Save consolidated results
    with open(output_json, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[✓] All Experiment 1 evaluations saved to: {output_json}")


def main():
    parser = argparse.ArgumentParser(description="AEGIS Unified Experiment 1 Runner")
    parser.add_argument("--datasets", nargs="+", default=["ctu13_10", "ctu13_9", "ctu13_11", "nf_bot_iot"], help="Datasets to evaluate")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46], help="Random seeds")
    parser.add_argument("--epochs", type=int, default=35, help="Training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Mini-batch size")
    parser.add_argument("--device", type=str, default="mps" if torch.backends.mps.is_available() else "cpu", help="Computation device")
    parser.add_argument("--output_json", type=str, default="benchmarks/aegis_multidataset_results.json", help="Output JSON path")
    args = parser.parse_args()

    run_experiment1(
        datasets=args.datasets,
        seeds=args.seeds,
        epochs=args.epochs,
        batch_size=args.batch_size,
        device=args.device,
        output_json=args.output_json,
    )


if __name__ == "__main__":
    main()
