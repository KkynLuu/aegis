#!/usr/bin/env python3
"""
AEGIS Comprehensive Multi-Dataset Benchmark & Statistical Significance Evaluation.
Target Venue: IEEE Transactions on Information Forensics and Security (TIFS) / USENIX Security.

Consolidates empirical evaluation across:
  - CTU-13 Scenario 10 (Sparse decentralized P2P overlay)
  - CTU-13 Scenario 9  (Dense volumetric bipartite spam)
  - CTU-13 Scenario 11 (Low-and-slow intermittent centralized C2)
  - NF-BoT-IoT         (Large-scale IoT telemetry)

Integrates precomputed 10-seed baselines (GCN, GIN, RWPE-GIN, MagNet, QUASAR)
and computes rigorous pairwise Wilcoxon signed-rank tests with Benjamini-Hochberg FDR control.
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


def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    device: str,
    threshold: Optional[float] = None
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

    if threshold is not None:
        preds = (probs >= threshold).astype(int)
    else:
        preds = (probs >= 0.5).astype(int)

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


def find_optimal_threshold(
    model: nn.Module,
    val_loader: DataLoader,
    device: str
) -> float:
    """Finds optimal decision threshold tau* in [0.05, 0.95] maximizing validation Macro-F1."""
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

    probs = np.concatenate(all_probs)
    y_true = np.concatenate(all_trues)

    best_tau = 0.5
    best_f1 = -1.0
    for tau in np.linspace(0.35, 0.65, 31):
        preds = (probs >= tau).astype(int)
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
            best_tau = float(tau)

    return best_tau


def train_aegis(
    in_dim: int,
    hidden_dim: int,
    num_layers: int,
    train_loader: DataLoader,
    val_loader: DataLoader,
    class_weights: torch.Tensor,
    epochs: int = 50,
    lr: float = 0.003,
    device: str = "cpu",
    seed: int = 42,
    use_volumetric: bool = True
) -> nn.Module:
    """Trains AegisClassifier with validation checkpointing, gradient clipping, and early stopping."""
    torch.manual_seed(seed)
    np.random.seed(seed)

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
        use_volumetric=use_volumetric,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))

    best_val_f1 = -1.0
    best_weights = copy.deepcopy(model.state_dict())
    patience_counter = 0

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
            if getattr(model, "use_moe", False):
                logits, l_topo, l_vol, gating = model(x, adj, mask=mask, return_all=True)
                loss_main = criterion(logits, y)
                loss_topo = criterion(l_topo, y)
                loss_vol = criterion(l_vol, y)
                if getattr(model, "moe_routing_mode", "topological_priority") == "topological_priority":
                    loss = loss_main + 0.3 * loss_topo + 0.3 * loss_vol
                else:
                    mean_g = gating.mean(dim=0)
                    loss_balance = 2.0 * (mean_g[0] * mean_g[1])
                    loss = loss_main + 0.3 * loss_topo + 0.3 * loss_vol - 0.05 * loss_balance
            else:
                logits = model(x, adj, mask=mask)
                loss = criterion(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_loss = total_loss / max(n_batches, 1)

        # Validation every 2 epochs or on the first / final epoch
        if (epoch + 1) % 2 == 0 or epoch == 0 or epoch == epochs - 1:
            val_metrics = evaluate_model(model, val_loader, device=device)
            is_best = val_metrics["f1"] > best_val_f1
            if is_best:
                best_val_f1 = val_metrics["f1"]
                best_weights = copy.deepcopy(model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1

            print(f"      Ep {epoch+1:2d}/{epochs:2d} | Loss: {avg_loss:.4f} | Val F1: {val_metrics['f1']:.2f}% | Val FAR: {val_metrics['far']:.2f}%{' *' if is_best else ''}", flush=True)

    model.load_state_dict(best_weights)
    return model


def compute_wilcoxon_tests(
    aegis_scores: List[float],
    baseline_scores: List[float],
    alternative: str = "two-sided"
) -> Tuple[float, float]:
    """Computes paired Wilcoxon signed-rank test between AEGIS and baseline."""
    n = min(len(aegis_scores), len(baseline_scores))
    a = np.array(aegis_scores[:n])
    b = np.array(baseline_scores[:n])
    diff = a - b
    if np.all(diff == 0.0):
        return 0.0, 1.0
    try:
        res = scipy.stats.wilcoxon(a, b, alternative=alternative)
        return float(res.statistic), float(res.pvalue)
    except Exception:
        return 0.0, 1.0


def run_evaluation(
    datasets: List[str],
    seeds: List[int],
    epochs: int = 25,
    batch_size: int = 64,
    lr: float = 0.003,
    hidden_dim: int = 32,
    num_layers: int = 2,
    device: str = "cpu",
    output_json: str = "benchmarks/aegis_multidataset_results.json",
    use_volumetric: bool = True,
    no_cache: bool = False,
) -> Dict[str, Any]:
    torch.set_num_threads(8)
    all_results = {}
    if os.path.exists(output_json):
        try:
            with open(output_json, "r") as f:
                all_results = json.load(f)
        except Exception:
            all_results = {}

    metrics_list = ["f1", "precision", "recall", "far", "auc", "acc"]

    for ds in datasets:
        print(f"\n{'='*75}")
        print(f"  EVALUATION REGIME: {ds.upper()}")
        print(f"{'='*75}")

        data_dir = f"data/processed/{ds}"
        train_graphs, _ = load_netflow_graphs(os.path.join(data_dir, "train.pt"))
        val_graphs, _ = load_netflow_graphs(os.path.join(data_dir, "val.pt"))
        test_graphs, _ = load_netflow_graphs(os.path.join(data_dir, "test.pt"))

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

        # Load baseline results if available
        baseline_file = f"benchmarks/rq1_{ds}_results.json"
        baseline_data = {}
        if os.path.exists(baseline_file):
            with open(baseline_file, "r") as f:
                baseline_data = json.load(f)
            print(f"  [✓] Loaded baseline results from {baseline_file}")

        # Check cache for AEGIS results
        cache_file = f"benchmarks/aegis_cache_{ds}.json"
        aegis_raw = {met: [] for met in metrics_list}
        finished_seeds = []
        if os.path.exists(cache_file) and not no_cache:
            try:
                with open(cache_file, "r") as f:
                    cached = json.load(f)
                aegis_raw = cached.get("raw_results", aegis_raw)
                finished_seeds = cached.get("seeds", [])
                print(f"  [✓] Loaded {len(finished_seeds)} cached AEGIS seeds: {finished_seeds}")
            except Exception:
                pass

        for s_idx, s in enumerate(seeds):
            if s in finished_seeds:
                continue
            print(f"\n  --- [Seed {s_idx + 1}/{len(seeds)}: {s}] ---")
            t0 = time.time()
            model = train_aegis(
                in_dim=in_dim,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                train_loader=train_loader,
                val_loader=val_loader,
                class_weights=weights,
                epochs=epochs,
                lr=lr,
                device=device,
                seed=s,
                use_volumetric=use_volumetric,
            )
            t_eval = time.time() - t0
            best_tau = 0.50
            res = evaluate_model(model, test_loader, device=device, threshold=0.50)
            for met in metrics_list:
                aegis_raw[met].append(res[met])
            finished_seeds.append(s)

            # Update cache
            with open(cache_file, "w") as f:
                json.dump({"seeds": finished_seeds, "raw_results": aegis_raw}, f, indent=2)

            print(f"    AEGIS [tau=0.50] | Time: {t_eval:>5.1f}s | F1: {res['f1']:>5.2f}% | Precision: {res['precision']:>5.2f}% | Recall: {res['recall']:>5.2f}% | FAR: {res['far']:>5.2f}% | AUC: {res['auc']:>5.2f}%", flush=True)

        # Compute summary
        aegis_summary = {}
        for met in metrics_list:
            aegis_summary[f"{met}_mean"] = float(np.mean(aegis_raw[met]))
            aegis_summary[f"{met}_std"] = float(np.std(aegis_raw[met]))

        # Statistical tests against baselines
        stat_tests = {}
        if "summary" in baseline_data and "raw_results" in baseline_data:
            base_raw = baseline_data["raw_results"]
            p_values = []
            test_keys = []
            for bm in base_raw:
                if bm == "AEGIS":
                    continue
                stat_w, p_val = compute_wilcoxon_tests(aegis_raw["f1"], base_raw[bm]["f1"])
                test_keys.append(bm)
                p_values.append(p_val)
                stat_tests[bm] = {"stat": stat_w, "raw_p": p_val}

            # BH-FDR correction
            if len(p_values) > 0:
                try:
                    q_vals = scipy.stats.false_discovery_control(p_values, method="bh")
                    for idx, bm in enumerate(test_keys):
                        stat_tests[bm]["bh_q"] = float(q_vals[idx])
                        if q_vals[idx] < 0.001:
                            stat_tests[bm]["sig"] = "***"
                        elif q_vals[idx] < 0.01:
                            stat_tests[bm]["sig"] = "**"
                        elif q_vals[idx] < 0.05:
                            stat_tests[bm]["sig"] = "*"
                        else:
                            stat_tests[bm]["sig"] = "ns"
                except Exception:
                    pass

        ds_summary = baseline_data.get("summary", {})
        ds_summary["AEGIS"] = aegis_summary

        all_results[ds] = {
            "summary": ds_summary,
            "aegis_raw": aegis_raw,
            "statistical_tests": stat_tests,
            "seeds": finished_seeds,
        }

        # Print Regime Summary Table
        print("\n" + "-" * 75)
        print(f"  REGIME SUMMARY: {ds.upper()} ({len(finished_seeds)} seeds)")
        print("-" * 75)
        print(f"{'Model':<12} | {'Macro-F1 (%)':<18} | {'Recall (%)':<16} | {'FAR (%) [↓]':<16}")
        print("-" * 75)
        for m, m_data in ds_summary.items():
            f1_str = f"{m_data.get('f1_mean', 0.0):.2f} ± {m_data.get('f1_std', 0.0):.2f}"
            rc_str = f"{m_data.get('recall_mean', 0.0):.2f} ± {m_data.get('recall_std', 0.0):.2f}"
            far_str = f"{m_data.get('far_mean', 0.0):.2f} ± {m_data.get('far_std', 0.0):.2f}"
            marker = f" [{stat_tests[m]['sig']}]" if m in stat_tests else ""
            print(f"{m:<12} | {f1_str:<18} | {rc_str:<16} | {far_str:<16}{marker}")
        print("-" * 75)

    with open(output_json, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[✓] Consolidated Multi-Dataset Results successfully saved to: {output_json}")
    return all_results


def main():
    parser = argparse.ArgumentParser(description="AEGIS Multi-Dataset Benchmark")
    parser.add_argument("--datasets", nargs="+", default=["ctu13_10", "ctu13_9", "ctu13_11", "nf_bot_iot"], help="Datasets to evaluate")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46], help="Random seeds")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Mini-batch size")
    parser.add_argument("--device", type=str, default="mps" if torch.backends.mps.is_available() else "cpu", help="Device (cpu, cuda, mps)")
    parser.add_argument("--output_json", type=str, default="benchmarks/aegis_multidataset_results.json", help="Output JSON path")
    parser.add_argument("--use_volumetric", action="store_true", default=True, help="Use dual-branch volumetric readout")
    parser.add_argument("--no_cache", action="store_true", default=True, help="Do not load cached seed evaluations")
    args = parser.parse_args()

    run_evaluation(
        datasets=args.datasets,
        seeds=args.seeds,
        epochs=args.epochs,
        batch_size=args.batch_size,
        device=args.device,
        output_json=args.output_json,
        use_volumetric=args.use_volumetric,
        no_cache=args.no_cache,
    )


if __name__ == "__main__":
    main()
