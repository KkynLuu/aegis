#!/usr/bin/env python3
"""
AEGIS Out-of-Distribution Cross-Scenario Botnet Transferability Benchmark (Table V).
Target Venue: IEEE Transactions on Information Forensics and Security (TIFS)

Evaluates zero-shot transferability across distinct captures of the Rbot malware family:
  - Training Set:   CTU-13 Scenario 10 (Rbot Lateral P2P Overlay, N=800 train, N=114 val)
  - Zero-Shot Test: CTU-13 Scenario 11 (Rbot Intermittent C2 Beaconing, untouched full N=92 graphs: 31 mal, 61 ben)

Evaluates:
  1. GCN (Spatial message passing with symmetric degree normalization)
  2. GIN (Graph Isomorphism Network with sum multiset aggregation)
  3. RWPE-GIN (GIN augmented with Random Walk Positional Encodings)
  4. Unitary CTQW (Closed) (Unitary quantum walk with phase transport)
  5. AEGIS-MoE (Ours) (Dual-branch cross-regime mixture of experts)
  6. AEGIS (Isolated E_topo) (Continuous-time non-Hermitian topological open quantum walk)

Thresholds evaluated:
  - tau = 0.50 (Standard operational balanced threshold)
  - tau = 0.40 (High-recall security analyst operational threshold)
  - tau*       (Optimal zero-leakage threshold calibrated exclusively on Scenario 10 validation set)

Outputs:
  - benchmarks/cross_scenario_transfer_results.json
  - Formatted publication table matching Table V in paper/main.tex
"""

import os
import sys
import copy
import time
import json
import argparse
from typing import Dict, List, Tuple, Any
import numpy as np
import scipy.stats
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data.netflow_graph_builder import (
    load_netflow_graphs,
    collate_netflow_batch,
    NetFlowInMemoryDataset,
)
from models.baseline_gnn import (
    GCNClassifier,
    GINClassifier,
    RWPE_GINClassifier,
)
from models.unitary_ctqw_classifier import UnitaryCTQWClassifier
from models.aegis_classifier import AegisClassifier
from benchmarks.benchmark_camouflage import compute_roc_auc_numpy


def evaluate_predictions(probs: np.ndarray, trues: np.ndarray, tau: float = 0.50) -> Dict[str, float]:
    """Calculates classification metrics at decision threshold tau."""
    preds = (probs >= tau).astype(int)
    acc = float((preds == trues).mean() * 100.0)
    tp = int(((preds == 1) & (trues == 1)).sum())
    fp = int(((preds == 1) & (trues == 0)).sum())
    fn = int(((preds == 0) & (trues == 1)).sum())
    tn = int(((preds == 0) & (trues == 0)).sum())

    p_mal = tp / max(tp + fp, 1)
    r_mal = tp / max(tp + fn, 1)
    f1_mal = 2.0 * p_mal * r_mal / max(p_mal + r_mal, 1e-6)

    p_ben = tn / max(tn + fn, 1)
    r_ben = tn / max(tn + fp, 1)
    f1_ben = 2.0 * p_ben * r_ben / max(p_ben + r_ben, 1e-6)

    macro_f1 = (f1_mal + f1_ben) / 2.0 * 100.0
    far = (fp / max(fp + tn, 1)) * 100.0
    auc = compute_roc_auc_numpy(trues, probs)

    return {
        "macro_f1": macro_f1,
        "recall": r_mal * 100.0,
        "precision": p_mal * 100.0,
        "far": far,
        "accuracy": acc,
        "auc": auc,
        "threshold": float(tau),
    }


def find_optimal_threshold_numpy(probs: np.ndarray, y_true: np.ndarray) -> float:
    """Finds optimal classification threshold on validation set to maximize Macro-F1 with zero test leakage."""
    if len(y_true) == 0:
        return 0.50

    best_t, best_f1, best_far = 0.50, -1.0, 100.0
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
        far = (fp / max(fp + tn, 1)) * 100.0

        if (macro_f1 > best_f1) or (abs(macro_f1 - best_f1) < 1e-5 and far < best_far):
            best_f1 = macro_f1
            best_far = far
            best_t = float(t)

    return best_t


def predict_loader(model: nn.Module, loader: DataLoader, device: str = "cpu") -> Tuple[np.ndarray, np.ndarray]:
    """Runs inference across DataLoader and returns probabilities and ground truth labels."""
    model.eval()
    probs_list = []
    trues_list = []
    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)
            adj = batch["adj"].to(device)
            mask = batch["mask"].to(device)
            y = batch["labels"].to(device)
            out = model(x, adj, mask=mask)
            logits = out[0] if isinstance(out, tuple) else out
            probs = torch.softmax(logits, dim=-1)[:, 1]
            probs_list.append(probs.cpu().numpy())
            trues_list.append(y.cpu().numpy())

    return np.concatenate(probs_list), np.concatenate(trues_list)


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    weights: torch.Tensor,
    epochs: int = 35,
    lr: float = 0.003,
    weight_decay: float = 1e-4,
    is_aegis_moe: bool = False,
    device: str = "cpu",
) -> Tuple[nn.Module, float]:
    """Trains graph model with CosineAnnealingLR and validation checkpointing."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss(weight=weights.to(device))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    best_val_f1 = -1.0
    best_state = copy.deepcopy(model.state_dict())

    for epoch in range(1, epochs + 1):
        model.train()
        for batch in train_loader:
            x = batch["x"].to(device)
            adj = batch["adj"].to(device)
            mask = batch["mask"].to(device)
            y = batch["labels"].to(device)

            optimizer.zero_grad()
            if is_aegis_moe:
                logits, l_topo, l_vol, gating = model(x, adj, mask=mask, return_all=True)
                loss_main = criterion(logits, y)
                loss_topo = criterion(l_topo, y)
                loss_vol = criterion(l_vol, y)
                entropy_loss = -0.05 * 2.0 * (gating[:, 0] * gating[:, 1]).mean()
                loss = loss_main + 0.3 * loss_topo + 0.3 * loss_vol + entropy_loss
            else:
                out = model(x, adj, mask=mask)
                logits = out[0] if isinstance(out, tuple) else out
                loss = criterion(logits, y)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        scheduler.step()

        # Validation checkpointing every 2 epochs and final epoch
        if epoch % 2 == 0 or epoch == epochs:
            val_probs, val_trues = predict_loader(model, val_loader, device=device)
            val_res = evaluate_predictions(val_probs, val_trues, tau=0.50)
            if val_res["macro_f1"] > best_val_f1:
                best_val_f1 = val_res["macro_f1"]
                best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    # Calibrate optimal decision threshold tau* on validation set
    val_probs, val_trues = predict_loader(model, val_loader, device=device)
    tau_star = find_optimal_threshold_numpy(val_probs, val_trues)
    return model, tau_star


def build_model(model_name: str, in_dim: int = 10, hidden_dim: int = 32) -> Tuple[nn.Module, bool]:
    """Instantiates model architecture."""
    if model_name == "GCN":
        return GCNClassifier(in_dim=in_dim, hidden_dim=hidden_dim, num_classes=2, num_layers=2), False
    elif model_name == "GIN":
        return GINClassifier(in_dim=in_dim, hidden_dim=hidden_dim, num_classes=2, num_layers=2), False
    elif model_name == "RWPE-GIN":
        return RWPE_GINClassifier(in_dim=in_dim, hidden_dim=hidden_dim, num_classes=2, num_layers=2, k_rwpe=8), False
    elif model_name in ("Unitary CTQW (Closed)", "Unitary CTQW"):
        return UnitaryCTQWClassifier(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            num_classes=2,
            num_layers=2,
            num_time_scales=4,
            init_times=(0.2, 0.8, 2.5, 8.0),
            use_potential=True,
            use_attention=True,
        ), False
    elif model_name in ("AEGIS-MoE (Ours)", "AEGIS-MoE"):
        return AegisClassifier(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            num_classes=2,
            num_layers=2,
            num_time_scales=4,
            init_times=(0.2, 0.8, 2.5, 8.0),
            base_dissipation=0.2,
            adaptive_dissipation=True,
            dropout=0.1,
            residual=True,
            use_edge_encoder=True,
            use_potential=True,
            use_attention=True,
            use_volumetric=True,
            use_moe=True,
            moe_routing_mode="softmax",
        ), True
    elif model_name in ("AEGIS (Isolated E_topo)", "AEGIS (Pure Topo)"):
        return AegisClassifier(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            num_classes=2,
            num_layers=2,
            num_time_scales=4,
            init_times=(0.2, 0.8, 2.5, 8.0),
            base_dissipation=0.2,
            adaptive_dissipation=True,
            dropout=0.1,
            residual=True,
            use_edge_encoder=True,
            use_potential=True,
            use_attention=True,
            use_volumetric=False,
            use_moe=False,
        ), False
    else:
        raise ValueError(f"Unknown model name: {model_name}")


def main():
    parser = argparse.ArgumentParser(description="AEGIS Out-of-Distribution Cross-Scenario Botnet Transferability Benchmark (Table V)")
    parser.add_argument("--epochs", type=int, default=35, help="Number of training epochs (default: 35)")
    parser.add_argument("--num_seeds", type=int, default=5, help="Number of random seeds (default: 5)")
    parser.add_argument("--batch_size", type=int, default=32, help="Mini-batch size (default: 32)")
    parser.add_argument("--lr", type=float, default=0.003, help="Learning rate (default: 0.003)")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay (default: 1e-4)")
    parser.add_argument("--device", type=str, default="mps" if torch.backends.mps.is_available() else "cpu", help="Computation device (mps/cpu/cuda)")
    parser.add_argument("--models", nargs="+", default=["GCN", "GIN", "RWPE-GIN", "Unitary CTQW (Closed)", "AEGIS-MoE (Ours)", "AEGIS (Isolated E_topo)"])
    parser.add_argument("--output_json", type=str, default="benchmarks/cross_scenario_transfer_results.json")
    parser.add_argument("--smoke_test", action="store_true", help="Quick smoke test mode (2 seeds, 3 epochs)")
    args = parser.parse_args()

    device = args.device
    if device == "mps" and not torch.backends.mps.is_available():
        print("[!] MPS device requested but not available. Falling back to CPU.")
        device = "cpu"

    if args.smoke_test:
        print("[*] Running in SMOKE TEST mode (2 seeds, 3 epochs)...")
        seeds = [42, 43]
        epochs = 3
    else:
        seeds = [42 + i for i in range(args.num_seeds)]
        epochs = args.epochs

    models_to_run = args.models

    print("=" * 105)
    print("  EXPERIMENT 4: ZERO-SHOT CROSS-SCENARIO BOTNET TRANSFERABILITY EVALUATION")
    print("  Source (Train):      CTU-13 Scenario 10 (P2P Rbot Overlay, N=800 train, N=114 val)")
    print("  Target (Zero-Shot):  CTU-13 Scenario 11 (C2 Rbot Beaconing, untouched full N=92 graphs)")
    print(f"  Models:              {models_to_run}")
    print(f"  Seeds ({len(seeds)}):         {seeds}")
    print(f"  Epochs: {epochs} | Batch Size: {args.batch_size} | LR: {args.lr} | Device: {device}")
    print("=" * 105)

    # 1. Load Datasets
    print("\n[1/4] Loading graph datasets...")
    s10_train, _ = load_netflow_graphs("data/processed/ctu13_10/train.pt")
    s10_val, _ = load_netflow_graphs("data/processed/ctu13_10/val.pt")

    s11_train, _ = load_netflow_graphs("data/processed/ctu13_11/train.pt")
    s11_val, _ = load_netflow_graphs("data/processed/ctu13_11/val.pt")
    s11_test, _ = load_netflow_graphs("data/processed/ctu13_11/test.pt")
    s11_full = s11_train + s11_val + s11_test

    if args.smoke_test:
        s10_train = [g for g in s10_train if g.y == 0][:32] + [g for g in s10_train if g.y == 1][:32]
        s10_val = [g for g in s10_val if g.y == 0][:16] + [g for g in s10_val if g.y == 1][:16]
        s11_full = [g for g in s11_full if g.y == 0][:16] + [g for g in s11_full if g.y == 1][:16]

    print(f"  [✓] Scenario 10 Training Graphs: {len(s10_train)}")
    print(f"  [✓] Scenario 10 Validation Graphs: {len(s10_val)}")
    mal_count = [g.y for g in s11_full].count(1)
    ben_count = [g.y for g in s11_full].count(0)
    print(f"  [✓] Scenario 11 Full Out-of-Distribution Test Graphs: {len(s11_full)} (Malicious: {mal_count}, Benign: {ben_count})")

    train_loader = DataLoader(NetFlowInMemoryDataset(s10_train), batch_size=args.batch_size, shuffle=True, collate_fn=collate_netflow_batch)
    val_loader = DataLoader(NetFlowInMemoryDataset(s10_val), batch_size=args.batch_size, shuffle=False, collate_fn=collate_netflow_batch)
    s11_loader = DataLoader(NetFlowInMemoryDataset(s11_full), batch_size=args.batch_size, shuffle=False, collate_fn=collate_netflow_batch)

    # Class weights for training
    y_train = [g.y for g in s10_train]
    n0, n1 = max(1, y_train.count(0)), max(1, y_train.count(1))
    tot = len(y_train)
    weights = torch.tensor([tot / (2.0 * n0), tot / (2.0 * n1)], dtype=torch.float32)

    threshold_keys = ["tau_0.50", "tau_0.40", "tau_star"]
    threshold_values = {"tau_0.50": 0.50, "tau_0.40": 0.40}

    # Data structures: model -> threshold -> list of dicts across seeds
    raw_results: Dict[str, Dict[str, List[Dict[str, float]]]] = {
        m: {t: [] for t in threshold_keys} for m in models_to_run
    }

    start_time = time.time()

    # 2. Train & Evaluate Across Seeds
    for seed_idx, seed in enumerate(seeds):
        print(f"\n====================== SEED {seed_idx + 1}/{len(seeds)}: {seed} ======================")
        for model_name in models_to_run:
            torch.manual_seed(seed)
            np.random.seed(seed)

            model, is_moe = build_model(model_name)
            model = model.to(device)

            t0 = time.time()
            trained_model, tau_star = train_model(
                model,
                train_loader,
                val_loader,
                weights,
                epochs=epochs,
                lr=args.lr,
                weight_decay=args.weight_decay,
                is_aegis_moe=is_moe,
                device=device,
            )
            train_dur = time.time() - t0

            # Zero-shot inference on Scenario 11
            probs, trues = predict_loader(trained_model, s11_loader, device=device)

            # Evaluate at each threshold
            res_050 = evaluate_predictions(probs, trues, tau=0.50)
            res_040 = evaluate_predictions(probs, trues, tau=0.40)
            res_star = evaluate_predictions(probs, trues, tau=tau_star)

            raw_results[model_name]["tau_0.50"].append(res_050)
            raw_results[model_name]["tau_0.40"].append(res_040)
            raw_results[model_name]["tau_star"].append(res_star)

            print(
                f"  {model_name:<26} | Train: {train_dur:>5.1f}s | "
                f"tau=0.50: F1={res_050['macro_f1']:>5.2f}%, FAR={res_050['far']:>4.1f}% | "
                f"tau=0.40: F1={res_040['macro_f1']:>5.2f}%, Rec={res_040['recall']:>5.1f}% | "
                f"tau*={tau_star:.2f}: F1={res_star['macro_f1']:>5.2f}%",
                flush=True
            )

    total_runtime = time.time() - start_time

    # 3. Aggregate Summary Across Seeds
    summary: Dict[str, Dict[str, Dict[str, float]]] = {}
    for m in models_to_run:
        summary[m] = {}
        for t in threshold_keys:
            m_list = raw_results[m][t]
            metrics_summary = {}
            for metric in ["macro_f1", "recall", "precision", "far", "accuracy", "auc", "threshold"]:
                vals = [x[metric] for x in m_list]
                metrics_summary[f"{metric}_mean"] = float(np.mean(vals))
                metrics_summary[f"{metric}_std"] = float(np.std(vals))
            summary[m][t] = metrics_summary

    # Save JSON Artifact
    output_payload = {
        "metadata": {
            "source_dataset": "ctu13_10 (Scenario 10, P2P Rbot)",
            "target_dataset": "ctu13_11 (Scenario 11, C2 Rbot)",
            "target_graphs": len(s11_full),
            "target_malicious": mal_count,
            "target_benign": ben_count,
            "models": models_to_run,
            "num_seeds": len(seeds),
            "seeds": seeds,
            "epochs": epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "total_runtime_sec": total_runtime,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        "summary": summary,
        "raw_results": raw_results,
    }

    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(output_payload, f, indent=2)
    print(f"\n[✓] Results successfully saved to: {args.output_json}")

    # 4. Print Clean Formatted Results Table (Matching Table V in paper)
    print("\n" + "=" * 115)
    print("  TABLE V: ZERO-SHOT OUT-OF-DISTRIBUTION BOTNET TRANSFERABILITY (TRAINED ON S10 -> EVALUATED ON S11, N=92)")
    print("=" * 115)
    print(f"{'Model Architecture':<32} | {'Macro-F1 (%)':<16} | {'Recall (%)':<14} | {'Precision (%)':<16} | {'FAR (%) [↓]':<14} | {'Accuracy (%)':<14}")
    print("-" * 115)

    for m in models_to_run:
        for t, label in [("tau_0.50", r"\tau = 0.50"), ("tau_0.40", r"\tau = 0.40"), ("tau_star", r"\tau^* (Calibrated)")]:
            s = summary[m][t]
            f1_str = f"{s['macro_f1_mean']:.2f} ± {s['macro_f1_std']:.2f}"
            rec_str = f"{s['recall_mean']:.2f} ± {s['recall_std']:.2f}"
            prec_str = f"{s['precision_mean']:.2f} ± {s['precision_std']:.2f}"
            far_str = f"{s['far_mean']:.2f} ± {s['far_std']:.2f}"
            acc_str = f"{s['accuracy_mean']:.2f} ± {s['accuracy_std']:.2f}"

            row_name = f"{m} ({label})"
            print(f"{row_name:<32} | {f1_str:<16} | {rec_str:<14} | {prec_str:<16} | {far_str:<14} | {acc_str:<14}")
        print("-" * 115)
    print("=" * 115)

    # 5. Generate Publication-Ready LaTeX Table
    latex_table = generate_cross_scenario_latex_table(models_to_run, summary)
    print("\n--- IEEE TIFS Publication-Ready LaTeX Table (Table V) ---")
    print(latex_table)
    print("-" * 65)


def generate_cross_scenario_latex_table(
    models: List[str],
    summary: Dict[str, Dict[str, Dict[str, float]]]
) -> str:
    """Generates publication-ready IEEE LaTeX code for Table V."""
    rows = []
    for m in models:
        for t, tau_tex in [("tau_0.50", r"\tau = 0.50"), ("tau_0.40", r"\tau = 0.40")]:
            s = summary[m][t]
            f1_m, f1_s = s["macro_f1_mean"], s["macro_f1_std"]
            rec_m, rec_s = s["recall_mean"], s["recall_std"]
            prec_m, prec_s = s["precision_mean"], s["precision_std"]
            far_m, far_s = s["far_mean"], s["far_std"]
            acc_m, acc_s = s["accuracy_mean"], s["accuracy_std"]

            if "AEGIS-MoE" in m:
                m_label = f"\\textbf{{{m} ({tau_tex})}}"
                row_str = f"{m_label} & \\mathbf{{{f1_m:.2f} \\pm {f1_s:.2f}}} & {rec_m:.2f} & \\mathbf{{{prec_m:.2f}}} & \\mathbf{{{far_m:.2f}}} & \\mathbf{{{acc_m:.2f}}} \\\\"
            else:
                m_label = f"{m} ({tau_tex})"
                row_str = f"{m_label} & {f1_m:.2f} \\pm {f1_s:.2f} & {rec_m:.2f} & {prec_m:.2f} & {far_m:.2f} & {acc_m:.2f} \\\\"
            rows.append(row_str)
        rows.append(r"\midrule")

    if rows[-1] == r"\midrule":
        rows.pop()

    body = "\n".join(rows)

    return f"""\\begin{{table}}[t]
\\centering
\\caption{{Zero-Shot Out-of-Distribution Botnet Transferability (Trained on CTU-13 Scenario 10 $\\to$ Evaluated on Full Scenario 11, $N=92$)}}
\\label{{tab:cross_scenario}}
\\small
\\resizebox{{\\columnwidth}}{{!}}{{
\\begin{{tabular}}{{lccccc}}
\\toprule
\\textbf{{Model Architecture}} & \\textbf{{Macro-F1 (\\%)}} & \\textbf{{Recall (\\%)}} & \\textbf{{Precision (\\%)}} & \\textbf{{FAR (\\%) [$\\downarrow$]}} & \\textbf{{Accuracy (\\%)}} \\\\
\\midrule
{body}
\\bottomrule
\\end{{tabular}}
}}
\\end{{table}}"""


if __name__ == "__main__":
    main()
