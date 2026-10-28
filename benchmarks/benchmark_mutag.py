"""
MUTAG Real-World Molecular Benchmark.

Evaluates QW-GNN vs Classical Baselines (GCN, GIN) on the 188 mutagenic molecules
of the MUTAG dataset across standard 10-fold cross-validation.
Demonstrates that continuous-time quantum walk representations effectively transfer
to real-world molecular chemistry with node attributes and variable graph topologies.
"""

import os
import sys
import time
import json
import torch
import torch.nn as nn
import numpy as np

# Ensure root qw_gnn path is in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data.mutag_dataset import MUTAGDataset, get_mutag_kfold_splits, collate_mutag_batch
from models.baseline_gnn import GCNClassifier, GINClassifier
from models.qw_classifier import QWClassifier

def train_and_eval_molecular_model(
    model: nn.Module,
    train_batch: dict,
    test_batch: dict,
    lr: float = 0.005,
    epochs: int = 60,
    weight_decay: float = 1e-4,
    device: str = "cpu"
) -> float:
    model = model.to(device)
    x_tr = train_batch["x"].to(device)
    adj_tr = train_batch["adj"].to(device)
    mask_tr = train_batch["mask"].to(device)
    y_tr = train_batch["labels"].to(device)
    
    x_te = test_batch["x"].to(device)
    adj_te = test_batch["adj"].to(device)
    mask_te = test_batch["mask"].to(device)
    y_te = test_batch["labels"].to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-4)
    
    for _ in range(epochs):
        model.train()
        optimizer.zero_grad()
        logits = model(x_tr, adj_tr, mask_tr)
        loss = nn.functional.cross_entropy(logits, y_tr)
        loss.backward()
        optimizer.step()
        scheduler.step()
        
    model.eval()
    with torch.no_grad():
        logits_te = model(x_te, adj_te, mask_te)
        preds = logits_te.argmax(dim=-1)
        acc = (preds == y_te).float().mean().item()
        
    return acc * 100.0

def run_mutag_benchmark(epochs: int = 60, num_folds: int = 10, seed: int = 42):
    print("=" * 75)
    print("      MUTAG MOLECULAR GRAPH BENCHMARK (REAL-WORLD CHEMISTRY)")
    print(f"      Comparing Classical GNNs vs QW-GNN ({num_folds}-Fold CV)")
    print("=" * 75)
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    dataset = MUTAGDataset(root="data/TUDataset")
    splits = get_mutag_kfold_splits(dataset, num_folds=num_folds, seed=seed)
    
    results = {
        "GCN": [],
        "GIN": [],
        "QW-GNN": []
    }
    
    print(f"\n[*] Loaded MUTAG: {len(dataset)} molecules, 7 atom types, binary mutagenicity.")
    print(f"[*] Starting {num_folds}-fold cross validation ({epochs} epochs/fold)...\n")
    print(f"{'Fold':<8} | {'GCN Acc (%)':<15} | {'GIN Acc (%)':<15} | {'QW-GNN Acc (%)':<15} | {'Time (s)':<10}")
    print("-" * 75)
    
    t_start = time.time()
    
    for fold_idx in range(num_folds):
        train_idx, test_idx = splits[fold_idx]
        train_batch = collate_mutag_batch([dataset[i] for i in train_idx])
        test_batch = collate_mutag_batch([dataset[i] for i in test_idx])
        
        fold_t0 = time.time()
        fold_seed = seed + fold_idx * 100
        
        # 1. Classical GCN
        torch.manual_seed(fold_seed)
        gcn = GCNClassifier(in_dim=7, hidden_dim=32, num_classes=2, num_layers=3, dropout=0.1)
        acc_gcn = train_and_eval_molecular_model(gcn, train_batch, test_batch, lr=0.01, epochs=epochs)
        results["GCN"].append(acc_gcn)
        
        # 2. Classical GIN
        torch.manual_seed(fold_seed)
        gin = GINClassifier(in_dim=7, hidden_dim=32, num_classes=2, num_layers=3, dropout=0.1)
        acc_gin = train_and_eval_molecular_model(gin, train_batch, test_batch, lr=0.01, epochs=epochs)
        results["GIN"].append(acc_gin)
        
        # 3. Proposed QW-GNN
        torch.manual_seed(fold_seed)
        qw_gnn = QWClassifier(
            in_dim=7,
            hidden_dim=32,
            num_classes=2,
            num_layers=2,
            num_time_scales=4,
            init_times=(0.2, 0.8, 2.0, 4.0),
            dropout=0.1
        )
        acc_qw = train_and_eval_molecular_model(qw_gnn, train_batch, test_batch, lr=0.005, epochs=epochs)
        results["QW-GNN"].append(acc_qw)
        
        fold_duration = time.time() - fold_t0
        print(f"Fold {fold_idx + 1:<3} | {acc_gcn:<15.2f} | {acc_gin:<15.2f} | {acc_qw:<15.2f} | {fold_duration:<10.2f}")
        
    total_time = time.time() - t_start
    print("-" * 75)
    
    summary = {}
    for model_name in ["GCN", "GIN", "QW-GNN"]:
        accs = np.array(results[model_name])
        mean_acc = float(np.mean(accs))
        std_acc = float(np.std(accs))
        summary[model_name] = {
            "mean": mean_acc,
            "std": std_acc,
            "raw": accs.tolist()
        }
        
    print(f"\n{'Model':<12} | {'Mean Accuracy (%)':<20} | {'Status':<25}")
    print("=" * 65)
    print(f"{'GCN':<12} | {summary['GCN']['mean']:>6.2f} ± {summary['GCN']['std']:<10.2f} | Classical 1-WL Baseline")
    print(f"{'GIN':<12} | {summary['GIN']['mean']:>6.2f} ± {summary['GIN']['std']:<10.2f} | Classical 1-WL Baseline")
    print(f"{'QW-GNN':<12} | {summary['QW-GNN']['mean']:>6.2f} ± {summary['QW-GNN']['std']:<10.2f} | Continuous-Time Quantum Walk")
    print("=" * 65)
    print(f"[✓] MUTAG benchmark completed in {total_time:.2f} seconds.")
    
    output_path = os.path.join(os.path.dirname(__file__), "mutag_results.json")
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[✓] Detailed benchmark statistics saved to: {output_path}")

if __name__ == "__main__":
    run_mutag_benchmark(epochs=60, num_folds=10, seed=42)
