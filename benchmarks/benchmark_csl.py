"""
CSL (Circular Skip Links) Expressivity Benchmark.

Evaluates QW-GNN vs Classical 1-WL Baselines (GCN, GIN) on the 10-class CSL benchmark
(Murphy et al., ICML 2019; Dwivedi et al., 2020) across 5-fold cross-validation.
All graphs are 4-regular with N=41 nodes. Classical MPNNs are theoretically bounded
at 10.0% random accuracy. QW-GNN uses Continuous-Time Quantum Walks to surpass 1-WL.
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

from data.csl_dataset import CSLDataset, get_csl_kfold_splits, collate_csl_batch
from models.baseline_gnn import GCNClassifier, GINClassifier, RWPE_GINClassifier
from models.quasar_classifier import QuasarClassifier, QuasarClassifier as QWClassifier

def train_and_eval_model(
    model: nn.Module,
    train_batch: dict,
    test_batch: dict,
    lr: float = 0.01,
    epochs: int = 100,
    device: str = "cpu"
) -> float:
    """Trains a model on train_batch and returns test accuracy on test_batch."""
    model = model.to(device)
    x_tr = train_batch["x"].to(device)
    adj_tr = train_batch["adj"].to(device)
    y_tr = train_batch["labels"].to(device)
    
    x_te = test_batch["x"].to(device)
    adj_te = test_batch["adj"].to(device)
    y_te = test_batch["labels"].to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-3)
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        logits = model(x_tr, adj_tr)
        loss = nn.functional.cross_entropy(logits, y_tr)
        loss.backward()
        optimizer.step()
        scheduler.step()
        
    model.eval()
    with torch.no_grad():
        logits_te = model(x_te, adj_te)
        preds = logits_te.argmax(dim=-1)
        acc = (preds == y_te).float().mean().item()
        
    return acc * 100.0

def run_csl_benchmark(epochs: int = 150, num_folds: int = 5, seed: int = 42):
    print("=" * 75)
    print("      CSL (CIRCULAR SKIP LINKS) GRAPH EXPRESSIVITY BENCHMARK")
    print(f"      Comparing Classical GNNs (1-WL Bounded) vs QW-GNN ({num_folds}-Fold CV)")
    print("=" * 75)
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Initialize CSL Dataset
    dataset = CSLDataset(node_dim=4, seed=seed)
    splits = get_csl_kfold_splits(dataset, num_folds=num_folds, seed=seed)
    
    results = {
        "GCN": [],
        "GIN": [],
        "RWPE-GIN": [],
        "QUASAR (QW-GNN)": []
    }
    
    print(f"\n[*] Loaded CSL Dataset: {len(dataset)} graphs across 10 classes (41 nodes, 4-regular).")
    print(f"[*] Starting {num_folds}-fold cross validation ({epochs} epochs/fold)...\n")
    print(f"{'Fold':<8} | {'GCN Acc (%)':<13} | {'GIN Acc (%)':<13} | {'RWPE-GIN (%)':<13} | {'QUASAR (%)':<13} | {'Time (s)':<10}")
    print("-" * 88)
    
    t_start = time.time()
    
    for fold_idx in range(num_folds):
        train_idx, test_idx = splits[fold_idx]
        train_batch = collate_csl_batch([dataset[i] for i in train_idx])
        test_batch = collate_csl_batch([dataset[i] for i in test_idx])
        
        fold_t0 = time.time()
        fold_seed = seed + fold_idx * 100
        
        # 1. Classical GCN (1-WL baseline)
        torch.manual_seed(fold_seed)
        gcn = GCNClassifier(in_dim=4, hidden_dim=32, num_classes=10, num_layers=3, dropout=0.0)
        acc_gcn = train_and_eval_model(gcn, train_batch, test_batch, lr=0.01, epochs=epochs)
        results["GCN"].append(acc_gcn)
        
        # 2. Classical GIN (most expressive 1-WL baseline)
        torch.manual_seed(fold_seed)
        gin = GINClassifier(in_dim=4, hidden_dim=32, num_classes=10, num_layers=3, dropout=0.0)
        acc_gin = train_and_eval_model(gin, train_batch, test_batch, lr=0.01, epochs=epochs)
        results["GIN"].append(acc_gin)

        # 3. Beyond-1-WL Classical Baseline: RWPE-GIN (Dwivedi et al., 2021)
        torch.manual_seed(fold_seed)
        rwpe_gin = RWPE_GINClassifier(in_dim=4, hidden_dim=32, num_classes=10, num_layers=3, k_rwpe=16, dropout=0.0)
        acc_rwpe = train_and_eval_model(rwpe_gin, train_batch, test_batch, lr=0.01, epochs=epochs)
        results["RWPE-GIN"].append(acc_rwpe)
        
        # 4. Proposed QUASAR (Continuous-Time Quantum Walk)
        torch.manual_seed(fold_seed)
        quasar = QuasarClassifier(
            in_dim=4,
            hidden_dim=32,
            num_classes=10,
            num_layers=2,
            num_time_scales=6,
            init_times=(0.2, 0.5, 1.0, 2.0, 3.5, 5.0),
            dropout=0.0
        )
        acc_qw = train_and_eval_model(quasar, train_batch, test_batch, lr=0.01, epochs=epochs)
        results["QUASAR (QW-GNN)"].append(acc_qw)
        
        fold_duration = time.time() - fold_t0
        print(f"Fold {fold_idx + 1:<3} | {acc_gcn:<13.2f} | {acc_gin:<13.2f} | {acc_rwpe:<13.2f} | {acc_qw:<13.2f} | {fold_duration:<10.2f}")
        
    total_time = time.time() - t_start
    print("-" * 88)
    
    # Calculate statistics
    summary = {}
    for model_name in ["GCN", "GIN", "RWPE-GIN", "QUASAR (QW-GNN)"]:
        accs = np.array(results[model_name])
        mean_acc = float(np.mean(accs))
        std_acc = float(np.std(accs))
        summary[model_name] = {
            "mean": mean_acc,
            "std": std_acc,
            "raw": accs.tolist()
        }
        
    print(f"\n{'Model':<18} | {'Mean Accuracy (%)':<20} | {'Expressivity Status':<25}")
    print("=" * 75)
    print(f"{'GCN':<18} | {summary['GCN']['mean']:>6.2f} ± {summary['GCN']['std']:<10.2f} | 1-WL Bounded (Random Guess 10%)")
    print(f"{'GIN':<18} | {summary['GIN']['mean']:>6.2f} ± {summary['GIN']['std']:<10.2f} | 1-WL Bounded (Random Guess 10%)")
    print(f"{'RWPE-GIN':<18} | {summary['RWPE-GIN']['mean']:>6.2f} ± {summary['RWPE-GIN']['std']:<10.2f} | Classical Beyond-1-WL (Dissipative RW)")
    print(f"{'QUASAR (QW-GNN)':<18} | {summary['QUASAR (QW-GNN)']['mean']:>6.2f} ± {summary['QUASAR (QW-GNN)']['std']:<10.2f} | Unitary Quantum Walk (Phase Interference)")
    print("=" * 75)
    print(f"[✓] Benchmark completed in {total_time:.2f} seconds.")
    
    # Save output to JSON
    output_path = os.path.join(os.path.dirname(__file__), "csl_results.json")
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[✓] Detailed benchmark statistics saved to: {output_path}")

if __name__ == "__main__":
    run_csl_benchmark(epochs=150, num_folds=5, seed=42)
