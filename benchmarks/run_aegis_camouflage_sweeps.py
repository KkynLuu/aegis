#!/usr/bin/env python3
"""
Runs AEGIS-MoE across adversarial camouflage sweeps (alpha in [0.00, 1.00])
on CTU-13 Scenario 9, CTU-13 Scenario 11, and NF-BoT-IoT.
Merges AEGIS-MoE results into existing benchmarks/rq2_camouflage_{dataset}_results.json files,
recomputes statistical FDR significance tests against baselines, and regenerates Figure 2.
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
from benchmarks.benchmark_camouflage import (
    apply_topological_camouflage,
    evaluate_model_on_loader,
    find_optimal_threshold,
    perform_camouflage_statistical_tests,
    generate_camouflage_latex_table,
    plot_camouflage_stress_test,
)
from scripts.plot_multidataset_camouflage import plot_multidataset_camouflage


def train_aegis_moe(
    in_dim: int,
    hidden_dim: int,
    num_layers: int,
    train_loader: DataLoader,
    val_loader: DataLoader,
    class_weights: torch.Tensor,
    epochs: int = 35,
    lr: float = 0.003,
    device: str = "cpu",
    seed: int = 42,
) -> nn.Module:
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
        use_volumetric=True,
        use_moe=True,
        moe_routing_mode="softmax",
    ).to(device)

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
            logits, l_topo, l_vol, gating = model(x, adj, mask=mask, return_all=True)
            loss_main = criterion(logits, y)
            loss_topo = criterion(l_topo, y)
            loss_vol = criterion(l_vol, y)
            entropy_loss = -0.05 * 2.0 * (gating[:, 0] * gating[:, 1]).mean()
            loss = loss_main + 0.3 * loss_topo + 0.3 * loss_vol + entropy_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        scheduler.step()

        if (epoch + 1) % 2 == 0 or epoch == 0 or epoch == epochs - 1:
            val_m = evaluate_model_on_loader(model, val_loader, device=device)
            if val_m["f1"] > best_val_f1:
                best_val_f1 = val_m["f1"]
                best_weights = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_weights)
    return model


def run_sweeps_for_dataset(
    dataset_name: str,
    seeds: List[int] = [42, 43, 44, 45, 46],
    alphas: List[float] = [0.0, 0.25, 0.5, 0.75, 1.0],
    epochs: int = 35,
    batch_size: int = 32,
    device: str = "mps",
):
    print(f"\n{'='*80}\n  RUNNING AEGIS-MOE CAMOUFLAGE SWEEPS: {dataset_name.upper()}\n{'='*80}")
    data_dir = f"data/processed/{dataset_name}"
    train_graphs, _ = load_netflow_graphs(os.path.join(data_dir, "train.pt"))
    val_graphs, _ = load_netflow_graphs(os.path.join(data_dir, "val.pt"))
    clean_test_graphs, _ = load_netflow_graphs(os.path.join(data_dir, "test.pt"))

    in_dim = train_graphs[0].x.shape[-1]
    n_ben = sum(1 - g.y for g in train_graphs)
    n_mal = sum(g.y for g in train_graphs)
    n_tot = len(train_graphs)
    w_0 = float(n_tot) / (2.0 * max(float(n_ben), 1.0))
    w_1 = float(n_tot) / (2.0 * max(float(n_mal), 1.0))
    weights = torch.tensor([w_0, w_1], dtype=torch.float32)

    benign_train_features = torch.cat([g.x for g in train_graphs if g.y == 0], dim=0)
    benign_mean = benign_train_features.mean(dim=0, keepdim=True)

    train_loader = DataLoader(NetFlowInMemoryDataset(train_graphs), batch_size=batch_size, shuffle=True, collate_fn=collate_netflow_batch)
    val_loader = DataLoader(NetFlowInMemoryDataset(val_graphs), batch_size=batch_size, shuffle=False, collate_fn=collate_netflow_batch)

    # Pre-generate camouflage test sets
    cam_loaders = {}
    for a in alphas:
        a_str = f"{a:.2f}"
        cam_graphs = apply_topological_camouflage(copy.deepcopy(clean_test_graphs), alpha=a, benign_mean=benign_mean, k_regular=2)
        cam_loaders[a_str] = DataLoader(NetFlowInMemoryDataset(cam_graphs), batch_size=batch_size, shuffle=False, collate_fn=collate_netflow_batch)

    metrics_list = ["f1", "precision", "recall", "far", "auc"]
    aegis_raw = {f"{a:.2f}": {k: [] for k in metrics_list} for a in alphas}

    for s_idx, s in enumerate(seeds):
        t0 = time.time()
        print(f"--- [Seed {s_idx+1}/{len(seeds)}: {s}] ---")
        model = train_aegis_moe(
            in_dim=in_dim, hidden_dim=32, num_layers=2,
            train_loader=train_loader, val_loader=val_loader,
            class_weights=weights, epochs=epochs, lr=0.003, device=device, seed=s
        )
        opt_tau = find_optimal_threshold(model, val_loader, device=device)
        dur = time.time() - t0

        alpha_f1s = []
        for a in alphas:
            a_str = f"{a:.2f}"
            res = evaluate_model_on_loader(model, cam_loaders[a_str], device=device, threshold=opt_tau)
            for k in metrics_list:
                aegis_raw[a_str][k].append(res[k])
            alpha_f1s.append(f"{res['f1']:.1f}%")

        print(f"  AEGIS-MoE (Ours) | Train: {dur:.1f}s (tau={opt_tau:.2f}) | F1 across alphas [{', '.join(alpha_f1s)}]")

    # Aggregate AEGIS summary
    aegis_summary = {f"{a:.2f}": {} for a in alphas}
    for a in alphas:
        a_str = f"{a:.2f}"
        for k in metrics_list:
            vals = aegis_raw[a_str][k]
            aegis_summary[a_str][f"{k}_mean"] = float(np.mean(vals))
            aegis_summary[a_str][f"{k}_std"] = float(np.std(vals))

    # Merge into existing JSON
    json_path = f"benchmarks/rq2_camouflage_{dataset_name}_results.json"
    if os.path.exists(json_path):
        with open(json_path) as f:
            existing = json.load(f)
    else:
        existing = {"metadata": {}, "summary": {}, "raw_results": {}}

    existing_summary = existing.get("summary", {})
    existing_raw = existing.get("raw_results", {})

    # Migrate legacy baseline label QUASAR -> Unitary CTQW (Closed) if present
    if "QUASAR" in existing_summary and "Unitary CTQW (Closed)" not in existing_summary:
        existing_summary["Unitary CTQW (Closed)"] = existing_summary.pop("QUASAR")
    elif "QUASAR" in existing_summary:
        existing_summary.pop("QUASAR")

    if "QUASAR" in existing_raw and "Unitary CTQW (Closed)" not in existing_raw:
        existing_raw["Unitary CTQW (Closed)"] = existing_raw.pop("QUASAR")
    elif "QUASAR" in existing_raw:
        existing_raw.pop("QUASAR")

    target_name = "AEGIS-MoE (Ours)"
    existing_summary[target_name] = aegis_summary
    existing_raw[target_name] = aegis_raw

    # Normalize models list
    models_list = list(existing_summary.keys())
    if target_name in models_list:
        models_list.remove(target_name)
        models_list.append(target_name)

    # Recompute statistical significance
    stat_summary = perform_camouflage_statistical_tests(
        results_raw=existing_raw,
        models=models_list,
        alphas=alphas,
        target_model=target_name,
    )

    latex_code = generate_camouflage_latex_table(dataset_name, existing_summary, stat_summary, models_list, alphas)
    fig_path = f"benchmarks/figure_camouflage_{dataset_name}.png"
    plot_camouflage_stress_test(alphas, existing_summary, models_list, dataset_name, fig_path)

    existing["metadata"]["models"] = models_list
    existing["summary"] = existing_summary
    existing["raw_results"] = existing_raw
    existing["statistical_tests"] = stat_summary
    existing["latex_table"] = latex_code

    with open(json_path, "w") as f:
        json.dump(existing, f, indent=2)
    print(f"[✓] Updated {json_path} with genuine AEGIS-MoE camouflage results.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="mps" if torch.backends.mps.is_available() else "cpu")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46])
    args = parser.parse_args()

    # 1. CTU-13 Scenario 10
    run_sweeps_for_dataset("ctu13_10", seeds=args.seeds, epochs=35, batch_size=32, device=args.device)

    # 2. CTU-13 Scenario 11
    run_sweeps_for_dataset("ctu13_11", seeds=args.seeds, epochs=35, batch_size=32, device=args.device)

    # 3. CTU-13 Scenario 9
    run_sweeps_for_dataset("ctu13_9", seeds=args.seeds, epochs=35, batch_size=32, device=args.device)

    # 4. NF-BoT-IoT
    run_sweeps_for_dataset("nf_bot_iot", seeds=args.seeds, epochs=15, batch_size=128, device=args.device)

    # 5. Regenerate Multi-Dataset 4-Panel Figure
    print("\n" + "="*80)
    print("  REGENERATING 4-PANEL PUBLICATION FIGURE (FIGURE 2)")
    print("="*80)
    plot_multidataset_camouflage()


if __name__ == "__main__":
    main()
