"""
NetFlow Standard Benchmark (IEEE TIFS Experiment 1).

Evaluates QUASAR against 1-WL (GCN, GIN) and Beyond-1-WL (RWPE-GIN, MagNet) baselines
on standard, in-the-wild NetFlow graphs (CTU-13 and NF-BoT-IoT).
Reports: Macro-Precision, Macro-Recall, Macro-F1, False Alarm Rate (FAR), and ROC-AUC.
"""

import os
import sys
import time
import json
import numpy as np
import torch
import torch.nn as nn
from typing import List, Dict, Tuple
# Ensure root path is in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

def compute_roc_auc_numpy(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Computes exact ROC-AUC using Mann-Whitney U test without sklearn."""
    pos_mask = (y_true == 1)
    n_pos = int(np.sum(pos_mask))
    n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 50.0
    order = np.argsort(y_score)
    ranks = np.empty_like(order)
    ranks[order] = np.arange(1, len(y_score) + 1)
    u_pos = np.sum(ranks[pos_mask]) - n_pos * (n_pos + 1) / 2.0
    return float(u_pos / (n_pos * n_neg) * 100.0)

from data.netflow_graph_builder import CTU13NetFlowDataset, collate_netflow_batch, NetFlowGraph
from models.baseline_gnn import GCNClassifier, GINClassifier, RWPE_GINClassifier, MagNetClassifier
from models.quasar_classifier import QuasarClassifier


def train_and_eval_netflow(
    model: nn.Module,
    train_graphs: List[NetFlowGraph],
    test_graphs: List[NetFlowGraph],
    lr: float = 0.005,
    epochs: int = 50,
    device: str = "cpu"
) -> Dict[str, float]:
    """Trains on train_graphs and evaluates detection metrics on test_graphs."""
    model = model.to(device)
    train_batch = collate_netflow_batch(train_graphs)
    test_batch = collate_netflow_batch(test_graphs)

    x_tr, adj_tr, mask_tr, y_tr = (
        train_batch["x"].to(device),
        train_batch["adj"].to(device),
        train_batch["mask"].to(device),
        train_batch["labels"].to(device)
    )
    x_te, adj_te, mask_te, y_te = (
        test_batch["x"].to(device),
        test_batch["adj"].to(device),
        test_batch["mask"].to(device),
        test_batch["labels"].to(device)
    )

    # Compute class weights to handle natural NIDS class imbalance
    n_neg = float((y_tr == 0).sum().item())
    n_pos = float((y_tr == 1).sum().item())
    weights = torch.tensor([1.0, max(n_neg / max(n_pos, 1.0), 1.0)], device=device, dtype=torch.float32)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        logits = model(x_tr, adj_tr, mask=mask_tr)
        loss = nn.functional.cross_entropy(logits, y_tr, weight=weights)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        logits_te = model(x_te, adj_te, mask=mask_te)
        probs = torch.softmax(logits_te, dim=-1)[:, 1].cpu().numpy()
        preds = logits_te.argmax(dim=-1).cpu().numpy()
        y_true = y_te.cpu().numpy()

        acc = float((preds == y_true).mean() * 100.0)

        # Confusion Matrix components
        tp = int(((preds == 1) & (y_true == 1)).sum())
        fp = int(((preds == 1) & (y_true == 0)).sum())
        fn = int(((preds == 0) & (y_true == 1)).sum())
        tn = int(((preds == 0) & (y_true == 0)).sum())

        p_mal = tp / max(tp + fp, 1)
        r_mal = tp / max(tp + fn, 1)
        f1_mal = 2 * p_mal * r_mal / max(p_mal + r_mal, 1e-6)

        p_ben = tn / max(tn + fn, 1)
        r_ben = tn / max(tn + fp, 1)
        f1_ben = 2 * p_ben * r_ben / max(p_ben + r_ben, 1e-6)

        macro_f1 = (f1_mal + f1_ben) / 2.0 * 100.0
        precision = p_mal * 100.0
        recall = r_mal * 100.0
        far = fp / max(fp + tn, 1) * 100.0 # False Alarm Rate

        try:
            auc = compute_roc_auc_numpy(y_true, probs)
        except Exception:
            auc = 50.0

    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": macro_f1,
        "far": far,
        "auc": auc
    }


def run_netflow_benchmark(
    dataset_name: str = "CTU-13",
    data_path: str = "data/CTU-13/10/capture20110818.binetflow",
    max_graphs: int = 150,
    epochs: int = 40,
    num_runs: int = 5,
    seed: int = 42
):
    print("=" * 85)
    print(f"      NETFLOW DETECTION EFFICACY BENCHMARK (IEEE TIFS EXPERIMENT 1)")
    print(f"      Dataset: {dataset_name} | Max Subgraphs: {max_graphs} | Runs: {num_runs}")
    print("=" * 85)

    if not os.path.exists(data_path):
        print(f"[!] File not found: {data_path}")
        return

    # Load graphs across full capture timeline
    dataset = CTU13NetFlowDataset(data_path, window_sec=60.0, stride_sec=120.0, max_graphs=max_graphs)
    all_graphs = list(dataset)

    # 70% Train, 30% Test (chronological split)
    split_idx = int(0.7 * len(all_graphs))
    train_graphs = all_graphs[:split_idx]
    test_graphs = all_graphs[split_idx:]
    in_dim = train_graphs[0].x.shape[-1]

    models_to_test = ["GCN", "GIN", "RWPE-GIN", "MagNet", "QUASAR"]
    aggregated = {m: {"f1": [], "recall": [], "precision": [], "far": [], "auc": []} for m in models_to_test}

    for run_idx in range(num_runs):
        run_seed = seed + run_idx * 100
        print(f"\n[*] Run {run_idx + 1}/{num_runs} (seed={run_seed})...")

        # 1. GCN
        torch.manual_seed(run_seed)
        gcn = GCNClassifier(in_dim=in_dim, hidden_dim=32, num_classes=2, num_layers=2)
        res_gcn = train_and_eval_netflow(gcn, train_graphs, test_graphs, epochs=epochs)
        for k in aggregated["GCN"]: aggregated["GCN"][k].append(res_gcn[k])

        # 2. GIN
        torch.manual_seed(run_seed)
        gin = GINClassifier(in_dim=in_dim, hidden_dim=32, num_classes=2, num_layers=2)
        res_gin = train_and_eval_netflow(gin, train_graphs, test_graphs, epochs=epochs)
        for k in aggregated["GIN"]: aggregated["GIN"][k].append(res_gin[k])

        # 3. RWPE-GIN
        torch.manual_seed(run_seed)
        rwpe = RWPE_GINClassifier(in_dim=in_dim, hidden_dim=32, num_classes=2, num_layers=2, k_rwpe=8)
        res_rwpe = train_and_eval_netflow(rwpe, train_graphs, test_graphs, epochs=epochs)
        for k in aggregated["RWPE-GIN"]: aggregated["RWPE-GIN"][k].append(res_rwpe[k])

        # 4. MagNet
        torch.manual_seed(run_seed)
        magnet = MagNetClassifier(in_dim=in_dim, hidden_dim=32, num_classes=2, num_layers=2, q=0.25)
        res_mag = train_and_eval_netflow(magnet, train_graphs, test_graphs, epochs=epochs)
        for k in aggregated["MagNet"]: aggregated["MagNet"][k].append(res_mag[k])

        # 5. QUASAR
        torch.manual_seed(run_seed)
        quasar = QuasarClassifier(
            in_dim=in_dim,
            hidden_dim=32,
            num_classes=2,
            num_layers=2,
            num_time_scales=4,
            init_times=(0.2, 0.8, 2.0, 5.0)
        )
        res_qw = train_and_eval_netflow(quasar, train_graphs, test_graphs, epochs=epochs)
        for k in aggregated["QUASAR"]: aggregated["QUASAR"][k].append(res_qw[k])

    # Summary Table
    print("\n" + "=" * 95)
    print(f"{'Model':<16} | {'Macro-F1 (%)':<18} | {'Recall (%)':<15} | {'FAR (%)':<15} | {'ROC-AUC (%)':<15}")
    print("-" * 95)
    summary = {}
    for m in models_to_test:
        f1_mean, f1_std = np.mean(aggregated[m]["f1"]), np.std(aggregated[m]["f1"])
        rec_mean, rec_std = np.mean(aggregated[m]["recall"]), np.std(aggregated[m]["recall"])
        far_mean, far_std = np.mean(aggregated[m]["far"]), np.std(aggregated[m]["far"])
        auc_mean, auc_std = np.mean(aggregated[m]["auc"]), np.std(aggregated[m]["auc"])
        summary[m] = {
            "f1_mean": float(f1_mean), "f1_std": float(f1_std),
            "recall_mean": float(rec_mean), "recall_std": float(rec_std),
            "far_mean": float(far_mean), "far_std": float(far_std),
            "auc_mean": float(auc_mean), "auc_std": float(auc_std),
        }
        print(f"{m:<16} | {f1_mean:>6.2f} ± {f1_std:<8.2f} | {rec_mean:>6.2f} ± {rec_std:<6.2f} | {far_mean:>6.2f} ± {far_std:<6.2f} | {auc_mean:>6.2f} ± {auc_std:<6.2f}")
    print("=" * 95)

    # Save output to JSON
    output_path = os.path.join(os.path.dirname(__file__), "netflow_ctu13_results.json")
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[✓] Benchmark results saved to: {output_path}")


if __name__ == "__main__":
    run_netflow_benchmark(max_graphs=150, epochs=30, num_runs=3)
