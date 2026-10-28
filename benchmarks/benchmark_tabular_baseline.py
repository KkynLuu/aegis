#!/usr/bin/env python3
"""
Benchmark: Classical Non-Graph Tabular Baseline (Gradient Boosted Decision Trees)
Target Venue: IEEE Transactions on Information Forensics and Security (TIFS)

Evaluates HistGradientBoostingClassifier (scikit-learn LightGBM equivalent) across:
  1. Clean Detection (RQ1) across 4 NetFlow operational regimes:
     - CTU-13 Scenario 10 (Sparse Decentralized P2P)
     - CTU-13 Scenario 9 (Dense Volumetric Star Flood)
     - CTU-13 Scenario 11 (Low-and-Slow C2 Beaconing)
     - NF-BoT-IoT (Large-Scale Flow Aggregation)
  2. Adversarial Topological Camouflage (RQ2) on CTU-13 Scenario 10 across:
     alpha in [0.00, 0.25, 0.50, 0.75, 1.00]

Evaluated across 10 independent random seeds (42..51).
Saves output to benchmarks/tabular_baseline_results.json.
"""

import os
import sys
import copy
import json
import argparse
from typing import List, Dict, Tuple, Any
import numpy as np
import torch
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    accuracy_score,
    confusion_matrix,
)

# Ensure project root is in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data.netflow_graph_builder import NetFlowGraph, load_netflow_graphs
from benchmarks.benchmark_camouflage import apply_topological_camouflage


def extract_tabular_graph_features(graphs: List[NetFlowGraph]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extracts statistical summary features from graph-level NetFlow windows:
      - 5 statistical moments across node NetFlow features (mean, std, max, min, sum)
      - Graph-level topological invariants (node count, edge count, density, degree distributions)
    """
    X_list = []
    y_list = []

    for g in graphs:
        x_mat = g.x.numpy() if hasattr(g.x, "numpy") else np.array(g.x)
        adj_mat = g.adj.numpy() if hasattr(g.adj, "numpy") else np.array(g.adj)

        # Node feature moments (10 features * 5 = 50)
        mean_feat = np.mean(x_mat, axis=0)
        std_feat = np.std(x_mat, axis=0)
        max_feat = np.max(x_mat, axis=0)
        min_feat = np.min(x_mat, axis=0)
        sum_feat = np.sum(x_mat, axis=0)

        # Graph-level structural metrics
        N = float(g.num_nodes)
        E = float(adj_mat.sum())
        density = E / (N * (N - 1) + 1e-6)

        deg_out = adj_mat.sum(axis=-1)
        deg_in = adj_mat.sum(axis=-2)
        deg_tot = deg_out + deg_in

        topological_stats = [
            N,
            E,
            density,
            float(np.mean(deg_out)),
            float(np.std(deg_out)),
            float(np.max(deg_out)),
            float(np.mean(deg_in)),
            float(np.std(deg_in)),
            float(np.max(deg_in)),
            float(np.mean(deg_tot)),
            float(np.std(deg_tot)),
            float(np.max(deg_tot)),
        ]

        row = np.concatenate(
            [mean_feat, std_feat, max_feat, min_feat, sum_feat, topological_stats]
        )
        X_list.append(row)
        y_list.append(int(g.y))

    return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.int64)


def evaluate_dataset(
    dataset_name: str,
    seeds: List[int] = [42, 43, 44, 45, 46, 47, 48, 49, 50, 51],
) -> Dict[str, Any]:
    data_dir = f"data/processed/{dataset_name}"
    print(f"\n{'='*70}\n[DATASET] Loading {dataset_name} from {data_dir}...\n{'='*70}")
    train_graphs, _ = load_netflow_graphs(os.path.join(data_dir, "train.pt"))
    val_graphs, _ = load_netflow_graphs(os.path.join(data_dir, "val.pt"))
    test_graphs, _ = load_netflow_graphs(os.path.join(data_dir, "test.pt"))

    # Extract tabular representations
    print("Extracting tabular feature matrices...")
    X_train, y_train = extract_tabular_graph_features(train_graphs)
    X_val, y_val = extract_tabular_graph_features(val_graphs)
    X_test, y_test = extract_tabular_graph_features(test_graphs)

    # Train on train+val for final tree ensemble
    X_full_train = np.vstack([X_train, X_val])
    y_full_train = np.concatenate([y_train, y_val])

    metrics = {
        "f1": [],
        "precision": [],
        "recall": [],
        "far": [],
        "auc": [],
        "acc": [],
    }

    for seed in seeds:
        clf = HistGradientBoostingClassifier(
            max_iter=100,
            learning_rate=0.1,
            random_state=seed,
        )
        clf.fit(X_full_train, y_full_train)

        preds = clf.predict(X_test)
        scores = clf.predict_proba(X_test)[:, 1] if len(np.unique(y_test)) > 1 else preds

        f1 = f1_score(y_test, preds, average="macro") * 100.0
        prec = precision_score(y_test, preds, average="macro", zero_division=0) * 100.0
        rec = recall_score(y_test, preds, average="macro", zero_division=0) * 100.0
        acc = accuracy_score(y_test, preds) * 100.0

        tn, fp, fn, tp = confusion_matrix(y_test, preds).ravel()
        far = (fp / (fp + tn + 1e-6)) * 100.0

        try:
            auc = roc_auc_score(y_test, scores) * 100.0
        except Exception:
            auc = 50.0

        metrics["f1"].append(float(f1))
        metrics["precision"].append(float(prec))
        metrics["recall"].append(float(rec))
        metrics["far"].append(float(far))
        metrics["auc"].append(float(auc))
        metrics["acc"].append(float(acc))

    summary = {
        "f1_mean": float(np.mean(metrics["f1"])),
        "f1_std": float(np.std(metrics["f1"])),
        "precision_mean": float(np.mean(metrics["precision"])),
        "precision_std": float(np.std(metrics["precision"])),
        "recall_mean": float(np.mean(metrics["recall"])),
        "recall_std": float(np.std(metrics["recall"])),
        "far_mean": float(np.mean(metrics["far"])),
        "far_std": float(np.std(metrics["far"])),
        "auc_mean": float(np.mean(metrics["auc"])),
        "auc_std": float(np.std(metrics["auc"])),
        "acc_mean": float(np.mean(metrics["acc"])),
        "acc_std": float(np.std(metrics["acc"])),
    }

    print(
        f"[{dataset_name.upper()} GBDT] Macro-F1: {summary['f1_mean']:.2f} +/- {summary['f1_std']:.2f}% | "
        f"FAR: {summary['far_mean']:.2f} +/- {summary['far_std']:.2f}% | AUC: {summary['auc_mean']:.2f}%"
    )

    return {
        "summary": summary,
        "raw": metrics,
        "seeds": seeds,
    }


def evaluate_camouflage(
    seeds: List[int] = [42, 43, 44, 45, 46, 47, 48, 49, 50, 51],
    alphas: List[float] = [0.00, 0.25, 0.50, 0.75, 1.00],
) -> Dict[str, Any]:
    dataset_name = "ctu13_10"
    data_dir = f"data/processed/{dataset_name}"
    print(f"\n{'='*70}\n[CAMOUFLAGE] Evaluating GBDT under Chordal Camouflage ({dataset_name})...\n{'='*70}")
    train_graphs, _ = load_netflow_graphs(os.path.join(data_dir, "train.pt"))
    val_graphs, _ = load_netflow_graphs(os.path.join(data_dir, "val.pt"))
    test_graphs, _ = load_netflow_graphs(os.path.join(data_dir, "test.pt"))

    # Compute benign mean
    benign_nodes = [g.x for g in train_graphs if g.y == 0]
    benign_mean = torch.cat(benign_nodes, dim=0).mean(dim=0)

    X_train, y_train = extract_tabular_graph_features(train_graphs)
    X_val, y_val = extract_tabular_graph_features(val_graphs)
    X_full_train = np.vstack([X_train, X_val])
    y_full_train = np.concatenate([y_train, y_val])

    cam_results: Dict[str, Any] = {}

    for alpha in alphas:
        a_str = f"{alpha:.2f}"
        print(f"\nEvaluating alpha = {a_str}...")

        if alpha == 0.0:
            eval_graphs = test_graphs
        else:
            eval_graphs = apply_topological_camouflage(
                copy.deepcopy(test_graphs),
                alpha=alpha,
                benign_mean=benign_mean,
            )

        X_test_cam, y_test = extract_tabular_graph_features(eval_graphs)

        f1_scores = []
        for seed in seeds:
            clf = HistGradientBoostingClassifier(
                max_iter=100,
                learning_rate=0.1,
                random_state=seed,
            )
            clf.fit(X_full_train, y_full_train)
            preds = clf.predict(X_test_cam)
            f1 = f1_score(y_test, preds, average="macro") * 100.0
            f1_scores.append(float(f1))

        mean_f1 = float(np.mean(f1_scores))
        std_f1 = float(np.std(f1_scores))
        print(f"Alpha {a_str} -> Macro-F1: {mean_f1:.2f} +/- {std_f1:.2f}%")

        cam_results[a_str] = {
            "mean": mean_f1,
            "std": std_f1,
            "raw": f1_scores,
        }

    clean_f1 = cam_results["0.00"]["mean"]
    max_drop = clean_f1 - cam_results["1.00"]["mean"]
    cam_results["max_drop"] = float(max_drop)
    print(f"\nGBDT Total Camouflage Degradation (alpha 0.0 -> 1.0): -{max_drop:.2f}%")

    return cam_results


def main():
    parser = argparse.ArgumentParser(description="Evaluate GBDT baseline on NetFlow benchmarks.")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["ctu13_10", "ctu13_9", "ctu13_11", "nf_bot_iot"],
        help="Datasets to evaluate.",
    )
    args = parser.parse_args()

    results: Dict[str, Any] = {}
    for ds in args.datasets:
        results[ds] = evaluate_dataset(ds)

    results["camouflage_ctu13_10"] = evaluate_camouflage()

    out_path = "benchmarks/tabular_baseline_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[DONE] Saved tabular baseline results to {out_path}")


if __name__ == "__main__":
    main()
