#!/usr/bin/env python3
"""
Depth Scaling and Dirichlet Energy Stability Evaluation for AEGIS.
Evaluates AEGIS across layer depths L in {2, 4, 8, 12, 16} on CTU-13 Scenario 10.
Measures:
  - Macro-F1 (%)
  - False Alarm Rate (FAR %)
  - Dirichlet Energy Ratio R_E = E(H^(L)) / E(H^(0))
"""

import os
import sys
import copy
import time
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data.netflow_graph_builder import (
    NetFlowInMemoryDataset,
    collate_netflow_batch,
    load_netflow_graphs,
)
from models.aegis_classifier import AegisClassifier
from benchmarks.benchmark_oversmoothing import compute_dirichlet_energy_batch
from benchmarks.benchmark_rq1_detection import evaluate_model_on_loader


def run_depth_scaling_aegis(
    dataset_name: str = "ctu13_10",
    depths: list = [2, 4, 8, 12, 16],
    seeds: list = [42, 43, 44],
    epochs: int = 8,
    batch_size: int = 64,
    device: str = "mps" if torch.backends.mps.is_available() else "cpu"
):
    print(f"Running depth scaling on device: {device}", flush=True)
    torch.set_num_threads(8)
    data_dir = f"data/processed/{dataset_name}"
    train_graphs, _ = load_netflow_graphs(os.path.join(data_dir, "train.pt"))
    val_graphs, _ = load_netflow_graphs(os.path.join(data_dir, "val.pt"))
    test_graphs, _ = load_netflow_graphs(os.path.join(data_dir, "test.pt"))

    train_loader = DataLoader(NetFlowInMemoryDataset(train_graphs), batch_size=batch_size, shuffle=True, collate_fn=collate_netflow_batch)
    val_loader = DataLoader(NetFlowInMemoryDataset(val_graphs), batch_size=batch_size, shuffle=False, collate_fn=collate_netflow_batch)
    test_loader = DataLoader(NetFlowInMemoryDataset(test_graphs), batch_size=batch_size, shuffle=False, collate_fn=collate_netflow_batch)

    in_dim = train_graphs[0].x.shape[-1]
    n_tot = len(train_graphs)
    n_neg = max(1, sum(1 - g.y for g in train_graphs))
    n_pos = max(1, sum(g.y for g in train_graphs))
    w_0 = float(n_tot) / (2.0 * float(n_neg))
    w_1 = float(n_tot) / (2.0 * float(n_pos))
    class_weights = torch.tensor([w_0, w_1], dtype=torch.float32)

    out_path = f"benchmarks/aegis_depth_scaling_{dataset_name}.json"
    results = {}
    if os.path.exists(out_path):
        try:
            with open(out_path, "r") as f:
                results = json.load(f)
            print(f"Loaded existing depth scaling results: {list(results.keys())}", flush=True)
        except Exception:
            results = {}

    for L in depths:
        if str(L) in results and "dirichlet_mean" in results[str(L)]:
            print(f"Depth L={L} already evaluated, skipping.", flush=True)
            continue

        print(f"\n==================================================", flush=True)
        print(f"  Evaluating AEGIS with Depth L = {L} ({epochs} epochs, {len(seeds)} seeds)", flush=True)
        print(f"==================================================", flush=True)
        f1_list, far_list, dirichlet_list = [], [], []

        for seed in seeds:
            torch.manual_seed(seed)
            np.random.seed(seed)

            model = AegisClassifier(
                in_dim=in_dim,
                hidden_dim=32,
                num_classes=2,
                num_layers=L,
                num_time_scales=2,
                init_times=(0.5, 2.0),
                base_dissipation=0.1,
            ).to(device)

            optimizer = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=1e-4)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
            criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))

            best_val_f1 = -1.0
            best_weights = copy.deepcopy(model.state_dict())

            t0 = time.time()
            for ep in range(1, epochs + 1):
                model.train()
                for batch in train_loader:
                    optimizer.zero_grad()
                    logits = model(batch["x"].to(device), batch["adj"].to(device), mask=batch["mask"].to(device))
                    loss = criterion(logits, batch["labels"].to(device))
                    loss.backward()
                    optimizer.step()
                scheduler.step()

                val_res = evaluate_model_on_loader(model, val_loader, device=device)
                if val_res["f1"] > best_val_f1:
                    best_val_f1 = val_res["f1"]
                    best_weights = copy.deepcopy(model.state_dict())

            model.load_state_dict(best_weights)
            test_res = evaluate_model_on_loader(model, test_loader, device=device)

            # Compute Dirichlet energy ratio: R_E = E(H^(L)) / E(H^(0))
            model.eval()
            with torch.no_grad():
                init_energies = []
                final_energies = []
                for batch in test_loader:
                    bx = batch["x"].to(device)
                    badj = batch["adj"].to(device)
                    bmask = batch["mask"].to(device)

                    e_init = compute_dirichlet_energy_batch(bx, badj, mask=bmask)
                    _, h_final = model(bx, badj, mask=bmask, return_embeddings=True)
                    e_final = compute_dirichlet_energy_batch(h_final, badj, mask=bmask)

                    init_energies.append(e_init)
                    final_energies.append(e_final)

                mean_e_init = max(1e-8, float(np.mean(init_energies)))
                mean_e_final = float(np.mean(final_energies))
                r_e = mean_e_final / mean_e_init

            f1_list.append(test_res["f1"])
            far_list.append(test_res["far"])
            dirichlet_list.append(r_e)
            t_seed = time.time() - t0
            print(f"  [Seed {seed}] L={L} ({t_seed:.1f}s) | F1: {test_res['f1']:.2f}% | FAR: {test_res['far']:.2f}% | R_E: {r_e:.4f}", flush=True)

        results[str(L)] = {
            "f1_mean": float(np.mean(f1_list)),
            "f1_std": float(np.std(f1_list)),
            "far_mean": float(np.mean(far_list)),
            "far_std": float(np.std(far_list)),
            "dirichlet_mean": float(np.mean(dirichlet_list)),
            "dirichlet_std": float(np.std(dirichlet_list)),
        }
        print(f"==> L={L} SUMMARY: F1: {results[str(L)]['f1_mean']:.2f} ± {results[str(L)]['f1_std']:.2f}% | "
              f"FAR: {results[str(L)]['far_mean']:.2f} ± {results[str(L)]['far_std']:.2f}% | "
              f"R_E: {results[str(L)]['dirichlet_mean']:.4f} ± {results[str(L)]['dirichlet_std']:.4f}", flush=True)

        # Checkpoint to disk after each depth
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)

    print(f"\n[✓] Depth scaling completed! Results saved to {out_path}", flush=True)
    return results


if __name__ == "__main__":
    run_depth_scaling_aegis()
