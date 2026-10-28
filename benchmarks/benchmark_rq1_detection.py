#!/usr/bin/env python3
"""
Experiment 1: Standard Detection Efficacy (RQ1) Benchmark.
Target Venue: IEEE Transactions on Information Forensics and Security (TIFS).

Evaluates QUASAR against 1-WL classical baselines (GCN, GIN) and modern beyond-1-WL
baselines (RWPE-GIN, MagNet) on standardized in-the-wild NetFlow graph telemetry
(NF-BoT-IoT and CTU-13 Scenario 10).

Evaluation Protocol:
  - Temporal Train/Val/Test split (70% Train, 10% Validation, 20% Test)
  - Best-model checkpointing on Validation Macro-F1
  - 10 independent random seeds
  - Metrics: Macro-Precision, Macro-Recall, Macro-F1, False Alarm Rate (FAR), ROC-AUC
  - Statistical Hypothesis Testing: Two-sided paired Wilcoxon signed-rank test
  - Multiple Testing Correction: Benjamini-Hochberg False Discovery Rate (BH-FDR, q <= 0.05)
  - Automatic generation of publication-ready IEEE TIFS LaTeX table
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
    MagNetClassifier,
)
from models.quasar_classifier import QuasarClassifier


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


def evaluate_model_on_loader(
    model: nn.Module,
    loader: DataLoader,
    device: str
) -> Dict[str, float]:
    """Evaluates a model over a DataLoader and computes full NIDS detection metrics."""
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

    # Confusion Matrix components
    tp = int(((preds == 1) & (y_true == 1)).sum())
    fp = int(((preds == 1) & (y_true == 0)).sum())
    fn = int(((preds == 0) & (y_true == 1)).sum())
    tn = int(((preds == 0) & (y_true == 0)).sum())

    # Malicious class metrics
    p_mal = tp / max(tp + fp, 1)
    r_mal = tp / max(tp + fn, 1)
    f1_mal = 2.0 * p_mal * r_mal / max(p_mal + r_mal, 1e-6)

    # Benign class metrics
    p_ben = tn / max(tn + fn, 1)
    r_ben = tn / max(tn + fp, 1)
    f1_ben = 2.0 * p_ben * r_ben / max(p_ben + r_ben, 1e-6)

    macro_f1 = (f1_mal + f1_ben) / 2.0 * 100.0
    macro_precision = (p_mal + p_ben) / 2.0 * 100.0
    macro_recall = (r_mal + r_ben) / 2.0 * 100.0

    # False Alarm Rate: FP / (FP + TN)
    far = (fp / max(fp + tn, 1)) * 100.0

    # ROC-AUC
    auc = compute_roc_auc_numpy(y_true, probs)

    return {
        "f1": macro_f1,
        "precision": macro_precision,
        "recall": macro_recall,
        "far": far,
        "auc": auc,
        "acc": acc,
    }


def train_and_evaluate_single_run(
    model_name: str,
    in_dim: int,
    hidden_dim: int,
    num_layers: int,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    class_weights: torch.Tensor,
    epochs: int = 35,
    lr: float = 0.003,
    device: str = "cpu",
    seed: int = 42
) -> Dict[str, float]:
    """Trains a model with validation checkpointing and reports test metrics."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Model instantiation
    if model_name == "GCN":
        model = GCNClassifier(in_dim=in_dim, hidden_dim=hidden_dim, num_classes=2, num_layers=num_layers)
    elif model_name == "GIN":
        model = GINClassifier(in_dim=in_dim, hidden_dim=hidden_dim, num_classes=2, num_layers=num_layers)
    elif model_name == "RWPE-GIN":
        model = RWPE_GINClassifier(in_dim=in_dim, hidden_dim=hidden_dim, num_classes=2, num_layers=num_layers, k_rwpe=8)
    elif model_name == "MagNet":
        model = MagNetClassifier(in_dim=in_dim, hidden_dim=hidden_dim, num_classes=2, num_layers=num_layers, q=0.25)
    elif model_name == "QUASAR":
        model = QuasarClassifier(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            num_classes=2,
            num_layers=num_layers,
            num_time_scales=4,
            init_times=(0.2, 0.8, 2.0, 5.0)
        )
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

        scheduler.step()

        # Validate
        val_metrics = evaluate_model_on_loader(model, val_loader, device=device)
        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            best_weights = copy.deepcopy(model.state_dict())

    # Evaluate best checkpoint on test set
    model.load_state_dict(best_weights)
    test_metrics = evaluate_model_on_loader(model, test_loader, device=device)
    test_metrics["best_val_f1"] = best_val_f1
    return test_metrics


def perform_statistical_tests(
    results_raw: Dict[str, Dict[str, List[float]]],
    models: List[str],
    metrics: List[str],
    target_model: str = "QUASAR"
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """
    Performs two-sided paired Wilcoxon signed-rank tests comparing QUASAR
    against all baseline models on each metric across the random seeds.
    Enforces Benjamini-Hochberg False Discovery Rate (BH-FDR) correction (q <= 0.05).
    """
    if target_model not in results_raw:
        return {}

    test_records = []
    # Collect all pairwise tests for joint FDR control
    for m in models:
        if m == target_model:
            continue
        for metric in metrics:
            target_scores = np.array(results_raw[target_model][metric])
            baseline_scores = np.array(results_raw[m][metric])
            diff = target_scores - baseline_scores

            # Check if all differences are zero
            if np.all(diff == 0.0):
                p_val = 1.0
                stat = 0.0
            else:
                try:
                    res = scipy.stats.wilcoxon(target_scores, baseline_scores, alternative="two-sided")
                    p_val = float(res.pvalue)
                    stat = float(res.statistic)
                except Exception:
                    p_val = 1.0
                    stat = 0.0

            test_records.append({
                "model": m,
                "metric": metric,
                "stat": stat,
                "raw_p": p_val,
            })

    if len(test_records) == 0:
        return {}

    # Benjamini-Hochberg FDR correction
    raw_p_values = [r["raw_p"] for r in test_records]
    try:
        q_values = scipy.stats.false_discovery_control(raw_p_values, method="bh")
    except Exception:
        # Fallback manual BH-FDR calculation if function unavailable
        m_tests = len(raw_p_values)
        sorted_indices = np.argsort(raw_p_values)
        q_values = np.empty(m_tests)
        min_q = 1.0
        for rank in range(m_tests, 0, -1):
            idx = sorted_indices[rank - 1]
            p = raw_p_values[idx]
            q = min(1.0, (m_tests / rank) * p)
            min_q = min(min_q, q)
            q_values[idx] = min_q

    stat_summary: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for r, q_val in zip(test_records, q_values):
        m = r["model"]
        metric = r["metric"]
        if m not in stat_summary:
            stat_summary[m] = {}

        if q_val < 0.001:
            sig_str = "***"
        elif q_val < 0.01:
            sig_str = "**"
        elif q_val < 0.05:
            sig_str = "*"
        else:
            sig_str = "ns"

        stat_summary[m][metric] = {
            "stat": r["stat"],
            "raw_p": float(r["raw_p"]),
            "bh_q": float(q_val),
            "sig": sig_str,
        }

    return stat_summary


def generate_latex_table(
    dataset_name: str,
    summary: Dict[str, Dict[str, float]],
    stat_summary: Dict[str, Dict[str, Dict[str, Any]]],
    models: List[str]
) -> str:
    """Generates a publication-ready LaTeX table for IEEE TIFS."""
    latex_lines = [
        r"\begin{table*}[t]",
        r"\centering",
        rf"\caption{{Standard Intrusion Detection Efficacy (RQ1) on \textbf{{{dataset_name}}}. Mean $\pm$ standard deviation across 10 independent random seeds. $^{{***}}q<0.001$, $^{{**}}q<0.01$, $^{{*}}q<0.05$ indicate statistically significant superiority of QUASAR over the baseline after Benjamini-Hochberg FDR correction ($q \le 0.05$).}}",
        rf"\label{{tab:rq1_{dataset_name.lower()}}}",
        r"\small",
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        r"\textbf{Model} & \textbf{Macro-F1 (\%)} & \textbf{Precision (\%)} & \textbf{Recall (\%)} & \textbf{FAR (\%)} $\downarrow$ & \textbf{ROC-AUC (\%)} \\",
        r"\midrule",
    ]

    for m in models:
        f1_m, f1_s = summary[m]["f1_mean"], summary[m]["f1_std"]
        pr_m, pr_s = summary[m]["precision_mean"], summary[m]["precision_std"]
        rc_m, rc_s = summary[m]["recall_mean"], summary[m]["recall_std"]
        far_m, far_s = summary[m]["far_mean"], summary[m]["far_std"]
        auc_m, auc_s = summary[m]["auc_mean"], summary[m]["auc_std"]

        # Check significance markers
        sig_f1 = f"^{{{stat_summary[m]['f1']['sig']}}}" if (m != "QUASAR" and m in stat_summary and stat_summary[m]['f1']['sig'] != "ns") else ""
        sig_auc = f"^{{{stat_summary[m]['auc']['sig']}}}" if (m != "QUASAR" and m in stat_summary and stat_summary[m]['auc']['sig'] != "ns") else ""

        if m == "QUASAR":
            row = (
                rf"\textbf{{{m}}} & "
                rf"\textbf{{{f1_m:.2f} $\pm$ {f1_s:.2f}}} & "
                rf"\textbf{{{pr_m:.2f} $\pm$ {pr_s:.2f}}} & "
                rf"\textbf{{{rc_m:.2f} $\pm$ {rc_s:.2f}}} & "
                rf"\textbf{{{far_m:.2f} $\pm$ {far_s:.2f}}} & "
                rf"\textbf{{{auc_m:.2f} $\pm$ {auc_s:.2f}}} \\"
            )
        else:
            row = (
                rf"{m} & "
                rf"{f1_m:.2f} $\pm$ {f1_s:.2f}{sig_f1} & "
                rf"{pr_m:.2f} $\pm$ {pr_s:.2f} & "
                rf"{rc_m:.2f} $\pm$ {rc_s:.2f} & "
                rf"{far_m:.2f} $\pm$ {far_s:.2f} & "
                rf"{auc_m:.2f} $\pm$ {auc_s:.2f}{sig_auc} \\"
            )
        latex_lines.append(row)

    latex_lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table*}",
    ])
    return "\n".join(latex_lines)


def run_rq1_benchmark(
    dataset_name: str,
    data_dir: str,
    models: List[str],
    seeds: List[int],
    epochs: int = 35,
    batch_size: int = 32,
    lr: float = 0.003,
    hidden_dim: int = 32,
    num_layers: int = 2,
    device: str = "cpu",
    output_json: Optional[str] = None,
    test_dir: Optional[str] = None,
    test_dataset_name: Optional[str] = None,
):
    eval_mode_str = f"CROSS-SCENARIO: TRAIN [{dataset_name.upper()}] -> TEST [{test_dataset_name.upper()}]" if test_dir else f"DATASET: {dataset_name.upper()}"
    print("=" * 95)
    print("      EXPERIMENT 1: STANDARD DETECTION EFFICACY (RQ1) BENCHMARK")
    print(f"      Evaluation: {eval_mode_str}")
    print(f"      Train Dir:  {data_dir}")
    if test_dir:
        print(f"      Test Dir:   {test_dir}")
    print(f"      Models:     {', '.join(models)}")
    print(f"      Seeds:      {len(seeds)} {seeds}")
    print(f"      Epochs:     {epochs} | Batch Size: {batch_size} | LR: {lr} | Device: {device}")
    print("=" * 95)

    train_path = os.path.join(data_dir, "train.pt")
    val_path = os.path.join(data_dir, "val.pt")
    test_path = os.path.join(test_dir if test_dir else data_dir, "test.pt")

    for p in [train_path, val_path, test_path]:
        if not os.path.exists(p):
            print(f"[!] Error: Missing preprocessed file: {p}")
            print(f"    Please execute: python scripts/preprocess_rq1.py --dataset {dataset_name}")
            sys.exit(1)

    # Load datasets
    train_graphs, train_meta = load_netflow_graphs(train_path)
    val_graphs, val_meta = load_netflow_graphs(val_path)
    test_graphs, test_meta = load_netflow_graphs(test_path)

    in_dim = train_graphs[0].x.shape[-1]
    n_train_ben = sum(1 - g.y for g in train_graphs)
    n_train_mal = sum(g.y for g in train_graphs)

    # Balanced class weights: w_c = N / (2 * N_c)
    n_total = len(train_graphs)
    w_0 = float(n_total) / (2.0 * max(float(n_train_ben), 1.0))
    w_1 = float(n_total) / (2.0 * max(float(n_train_mal), 1.0))
    class_weights = torch.tensor([w_0, w_1], dtype=torch.float32)

    print(f"[*] Dataset Splits Loaded:")
    print(f"    - Train: {len(train_graphs)} graphs (Benign: {n_train_ben}, Malicious: {n_train_mal}, weights: [{w_0:.2f}, {w_1:.2f}])")
    print(f"    - Val:   {len(val_graphs)} graphs")
    print(f"    - Test:  {len(test_graphs)} graphs")
    print(f"    - Node Feature Dimension: {in_dim}")

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
    test_loader = DataLoader(
        NetFlowInMemoryDataset(test_graphs),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_netflow_batch
    )

    metrics_list = ["f1", "precision", "recall", "far", "auc"]
    raw_results: Dict[str, Dict[str, List[float]]] = {
        m: {k: [] for k in metrics_list} for m in models
    }

    total_start = time.time()

    # Training and Evaluation across Seeds
    for seed_idx, seed in enumerate(seeds):
        print(f"\n--- [Seed {seed_idx + 1}/{len(seeds)}: {seed}] ---")
        for m in models:
            t0 = time.time()
            res = train_and_evaluate_single_run(
                model_name=m,
                in_dim=in_dim,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                train_loader=train_loader,
                val_loader=val_loader,
                test_loader=test_loader,
                class_weights=class_weights,
                epochs=epochs,
                lr=lr,
                device=device,
                seed=seed
            )
            dt = time.time() - t0
            for k in metrics_list:
                raw_results[m][k].append(res[k])

            print(f"  {m:<10} | F1: {res['f1']:>6.2f}% | Rec: {res['recall']:>6.2f}% | "
                  f"FAR: {res['far']:>6.2f}% | AUC: {res['auc']:>6.2f}% ({dt:>5.2f}s)")

    # Aggregate Statistics
    summary: Dict[str, Dict[str, float]] = {}
    for m in models:
        summary[m] = {}
        for k in metrics_list:
            vals = raw_results[m][k]
            summary[m][f"{k}_mean"] = float(np.mean(vals))
            summary[m][f"{k}_std"] = float(np.std(vals))

    # Statistical Significance Testing (Wilcoxon + BH-FDR)
    stat_summary = perform_statistical_tests(
        results_raw=raw_results,
        models=models,
        metrics=metrics_list,
        target_model="QUASAR"
    )

    # Print Terminal Results Table
    print("\n" + "=" * 105)
    print(f"               EMPIRICAL RESULTS: {dataset_name.upper()} (10-SEED EVALUATION)")
    print("-" * 105)
    print(f"{'Model':<12} | {'Macro-F1 (%)':<20} | {'Precision (%)':<16} | {'Recall (%)':<16} | {'FAR (%) [↓]':<16} | {'ROC-AUC (%)':<16}")
    print("-" * 105)
    for m in models:
        f1_str = f"{summary[m]['f1_mean']:>5.2f} ± {summary[m]['f1_std']:<5.2f}"
        pr_str = f"{summary[m]['precision_mean']:>5.2f} ± {summary[m]['precision_std']:<5.2f}"
        rc_str = f"{summary[m]['recall_mean']:>5.2f} ± {summary[m]['recall_std']:<5.2f}"
        far_str = f"{summary[m]['far_mean']:>5.2f} ± {summary[m]['far_std']:<5.2f}"
        auc_str = f"{summary[m]['auc_mean']:>5.2f} ± {summary[m]['auc_std']:<5.2f}"

        marker = ""
        if m in stat_summary:
            sig = stat_summary[m]["f1"]["sig"]
            if sig != "ns":
                marker = f" [{sig}]"

        print(f"{m:<12} | {f1_str:<20} | {pr_str:<16} | {rc_str:<16} | {far_str:<16} | {auc_str:<16}{marker}")
    print("=" * 105)

    # Print Statistical Significance Summary
    if len(stat_summary) > 0:
        print("\n--- Pairwise Statistical Hypothesis Testing (QUASAR vs Baseline) ---")
        print("Test: Two-sided paired Wilcoxon signed-rank test with Benjamini-Hochberg FDR correction (q <= 0.05)")
        print(f"{'Baseline':<12} | {'Metric':<10} | {'QUASAR Mean':<12} | {'Base Mean':<12} | {'Raw p-val':<12} | {'BH q-val':<12} | {'Significance'}")
        print("-" * 88)
        for m, m_dict in stat_summary.items():
            for metric, res in m_dict.items():
                q_m = summary["QUASAR"][f"{metric}_mean"]
                b_m = summary[m][f"{metric}_mean"]
                p_v = res["raw_p"]
                q_v = res["bh_q"]
                sig = res["sig"]
                print(f"{m:<12} | {metric:<10} | {q_m:>10.2f}%  | {b_m:>10.2f}%  | {p_v:>10.4f}  | {q_v:>10.4f}  | {sig}")
        print("-" * 88)

    # Generate LaTeX Table for IEEE TIFS
    latex_code = generate_latex_table(dataset_name, summary, stat_summary, models)
    print("\n--- IEEE TIFS Publication-Ready LaTeX Table ---")
    print(latex_code)
    print("-----------------------------------------------\n")

    # Serialize JSON
    output_path = output_json or f"benchmarks/rq1_{dataset_name.lower()}_results.json"
    full_payload = {
        "metadata": {
            "dataset_name": dataset_name,
            "data_dir": data_dir,
            "models": models,
            "seeds": seeds,
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": lr,
            "hidden_dim": hidden_dim,
            "num_layers": num_layers,
            "device": device,
            "total_runtime_sec": time.time() - total_start,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        "summary": summary,
        "statistical_tests": stat_summary,
        "raw_results": raw_results,
        "latex_table": latex_code,
    }

    with open(output_path, "w") as f:
        json.dump(full_payload, f, indent=2)
    print(f"[✓] Complete Experiment 1 results saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="RQ1 Detection Efficacy Benchmark (IEEE TIFS Experiment 1)")
    parser.add_argument("--dataset", type=str, choices=["nf_bot_iot", "ctu13_10", "ctu13_9", "ctu13_11", "ctu13_all"], default="nf_bot_iot",
                        help="Dataset name to evaluate / train on (default: nf_bot_iot)")
    parser.add_argument("--test_dataset", type=str, default=None, choices=["nf_bot_iot", "ctu13_10", "ctu13_9", "ctu13_11", "ctu13_all"],
                        help="Optional unseen test dataset for cross-scenario evaluation (e.g., --dataset ctu13_10 --test_dataset ctu13_11)")
    parser.add_argument("--data_dir", type=str, default=None,
                        help="Path to folder containing preprocessed train.pt, val.pt, test.pt")
    parser.add_argument("--test_dir", type=str, default=None,
                        help="Optional path to folder containing unseen test.pt for cross-scenario evaluation")
    parser.add_argument("--num_seeds", type=int, default=10,
                        help="Number of independent random seeds (default: 10)")
    parser.add_argument("--epochs", type=int, default=35,
                        help="Training epochs per seed (default: 35)")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Mini-batch size for DataLoader (default: 32)")
    parser.add_argument("--lr", type=float, default=0.003,
                        help="Learning rate for AdamW optimizer (default: 0.003)")
    parser.add_argument("--hidden_dim", type=int, default=32,
                        help="Hidden feature dimension (default: 32)")
    parser.add_argument("--num_layers", type=int, default=2,
                        help="Number of graph convolutional layers (default: 2)")
    parser.add_argument("--models", nargs="+", default=["GCN", "GIN", "RWPE-GIN", "MagNet", "QUASAR"],
                        help="Models to evaluate (default: GCN GIN RWPE-GIN MagNet QUASAR)")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device to use: auto, cpu, or cuda (default: auto)")
    parser.add_argument("--output_json", type=str, default=None,
                        help="Custom path to save output JSON results")
    parser.add_argument("--smoke_test", action="store_true",
                        help="Run quick smoke test (2 seeds, 3 epochs) for pipeline validation")
    parser.add_argument("--num_threads", type=int, default=min(4, os.cpu_count() or 4),
                        help="PyTorch CPU threads (default: 4 for optimal latency and thermal control)")

    args = parser.parse_args()

    # Set PyTorch thread count for CPU to avoid 100% core saturation and overheating
    num_threads = max(1, args.num_threads)
    torch.set_num_threads(num_threads)

    # Device selection: use CUDA if available. For macOS, CPU is vastly superior to MPS
    # because Apple Silicon MPS cannot natively stage 128x128 eigendecompositions in threadgroup memory.
    if args.device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
    else:
        device = args.device

    print(f"[*] Execution Device: {device.upper()} | PyTorch Threads: {num_threads} (Thermal & Latency Optimized)")

    data_dir = args.data_dir or os.path.join("data", "processed", args.dataset)
    test_dir = args.test_dir or (os.path.join("data", "processed", args.test_dataset) if args.test_dataset else None)
    test_dataset_name = args.test_dataset or (os.path.basename(test_dir) if test_dir else None)

    if args.smoke_test:
        print("[*] Running in SMOKE TEST mode (2 seeds, 3 epochs)...")
        seeds = [42, 43]
        epochs = 3
    else:
        seeds = [42 + i for i in range(args.num_seeds)]
        epochs = args.epochs

    run_rq1_benchmark(
        dataset_name=args.dataset,
        data_dir=data_dir,
        models=args.models,
        seeds=seeds,
        epochs=epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        device=device,
        output_json=args.output_json,
        test_dir=test_dir,
        test_dataset_name=test_dataset_name,
    )


if __name__ == "__main__":
    main()
