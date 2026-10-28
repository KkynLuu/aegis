#!/usr/bin/env python3
"""
Benchmark: MoE Dual-Expert Ablation Study (Table IV Verification)
Target Venue: IEEE Transactions on Information Forensics and Security (TIFS)

Ablates the contribution of the two physical modalities in AEGIS-MoE:
  1. E_topo Alone: Open Non-Hermitian Continuous-Time Quantum Walk (Head_topo)
  2. E_vol Alone: Directed Multiset Volumetric Accumulation (Head_vol)
  3. Uniform Ensemble: Equal static weighting (g_topo = g_vol = 0.5) without adaptive routing
  4. Full AEGIS-MoE: Dynamic structural gating router

Evaluated across independent seeds (42..46) on:
  - CTU-13 Scenario 10 (Clean Sparse P2P)
  - CTU-13 Scenario 10 Camouflage (alpha = 1.00, Chordal Regularization)
  - CTU-13 Scenario 9 (Dense Volumetric Star Flood)
  - NF-BoT-IoT (Large-Scale Aggregated Traffic)

Saves output to benchmarks/aegis_ablation_results.json.
"""

import os
import sys
import copy
import json
import argparse
from typing import List, Dict, Tuple, Any
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix, roc_auc_score

# Ensure project root is in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data.netflow_graph_builder import (
    NetFlowGraph,
    NetFlowInMemoryDataset,
    collate_netflow_batch,
    load_netflow_graphs,
)
from models.aegis_classifier import AegisClassifier
from benchmarks.run_aegis_evaluation import train_aegis
from benchmarks.benchmark_camouflage import apply_topological_camouflage


def evaluate_ablation_loader(
    model: nn.Module,
    loader: DataLoader,
    device: str,
) -> Dict[str, Dict[str, float]]:
    model.eval()

    y_trues = []
    moe_preds, topo_preds, vol_preds, unif_preds = [], [], [], []
    g_topos = []

    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)
            adj = batch["adj"].to(device)
            mask = batch["mask"].to(device)
            y = batch["labels"].cpu().numpy()

            l_moe, l_topo, l_vol, g = model(x, adj, mask=mask, return_all=True)

            y_trues.extend(y)
            moe_preds.extend(l_moe.argmax(dim=-1).cpu().numpy())
            topo_preds.extend(l_topo.argmax(dim=-1).cpu().numpy())
            vol_preds.extend(l_vol.argmax(dim=-1).cpu().numpy())
            unif_preds.extend((0.5 * l_topo + 0.5 * l_vol).argmax(dim=-1).cpu().numpy())
            g_topos.extend(g[:, 0].cpu().numpy())

    y_trues = np.array(y_trues)

    def compute_metrics(preds):
        f1 = f1_score(y_trues, preds, average="macro") * 100.0
        prec = precision_score(y_trues, preds, average="macro", zero_division=0) * 100.0
        rec = recall_score(y_trues, preds, average="macro", zero_division=0) * 100.0
        tn, fp, fn, tp = confusion_matrix(y_trues, preds).ravel()
        far = (fp / (fp + tn + 1e-6)) * 100.0
        acc = float((preds == y_trues).mean() * 100.0)
        return {"f1": float(f1), "precision": float(prec), "recall": float(rec), "far": float(far), "acc": acc}

    return {
        "AEGIS-MoE": compute_metrics(moe_preds),
        "E_topo": compute_metrics(topo_preds),
        "E_vol": compute_metrics(vol_preds),
        "Uniform-Ensemble": compute_metrics(unif_preds),
        "mean_g_topo": float(np.mean(g_topos)),
    }


def run_ablation(
    seeds: List[int] = [42, 43, 44, 45, 46],
    epochs: int = 35,
) -> Dict[str, Any]:
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Running MoE Dual-Expert Ablation on device: {device}")

    results: Dict[str, Any] = {}

    # =========================================================================
    # 1. CTU-13 Scenario 10: Clean AND Camouflage (alpha = 1.00)
    # =========================================================================
    print(f"\n{'='*75}\n  ABLATION REGIME: CTU-13 SCENARIO 10 (CLEAN & CAMOUFLAGE)\n{'='*75}")
    dir_10 = "data/processed/ctu13_10"
    tr_10, _ = load_netflow_graphs(os.path.join(dir_10, "train.pt"))
    va_10, _ = load_netflow_graphs(os.path.join(dir_10, "val.pt"))
    te_10, _ = load_netflow_graphs(os.path.join(dir_10, "test.pt"))

    # Generate camouflage test set
    benign_nodes = [g.x for g in tr_10 if g.y == 0]
    benign_mean = torch.cat(benign_nodes, dim=0).mean(dim=0)
    cam_10_te = apply_topological_camouflage(copy.deepcopy(te_10), alpha=1.00, benign_mean=benign_mean)

    tr_loader_10 = DataLoader(NetFlowInMemoryDataset(tr_10), batch_size=32, shuffle=True, collate_fn=collate_netflow_batch)
    va_loader_10 = DataLoader(NetFlowInMemoryDataset(va_10), batch_size=32, shuffle=False, collate_fn=collate_netflow_batch)
    te_clean_loader_10 = DataLoader(NetFlowInMemoryDataset(te_10), batch_size=32, shuffle=False, collate_fn=collate_netflow_batch)
    te_cam_loader_10 = DataLoader(NetFlowInMemoryDataset(cam_10_te), batch_size=32, shuffle=False, collate_fn=collate_netflow_batch)

    labels = [g.y for g in tr_10]
    c0, c1 = labels.count(0), labels.count(1)
    tot = len(labels)
    weights_10 = torch.tensor([tot / (2.0 * max(c0, 1)), tot / (2.0 * max(c1, 1))], dtype=torch.float32)

    metric_keys = ["f1", "far", "prec", "rec", "acc"]
    runs_clean_10 = {m: {k: [] for k in metric_keys} for m in ["AEGIS-MoE", "E_topo", "E_vol", "Uniform-Ensemble"]}
    runs_clean_10["g_topo"] = []

    runs_cam_10 = {m: {k: [] for k in metric_keys} for m in ["AEGIS-MoE", "E_topo", "E_vol", "Uniform-Ensemble"]}
    runs_cam_10["g_topo"] = []

    for s in seeds:
        print(f"--- Training seed {s} on ctu13_10 ---")
        model = train_aegis(10, 32, 2, tr_loader_10, va_loader_10, weights_10, epochs=epochs, device=device, seed=s)

        # Clean eval
        m_clean = evaluate_ablation_loader(model, te_clean_loader_10, device=device)
        for mod in ["AEGIS-MoE", "E_topo", "E_vol", "Uniform-Ensemble"]:
            for k in metric_keys:
                runs_clean_10[mod][k].append(m_clean[mod]["precision" if k == "prec" else ("recall" if k == "rec" else k)])
        runs_clean_10["g_topo"].append(m_clean["mean_g_topo"])

        # Camouflage eval
        m_cam = evaluate_ablation_loader(model, te_cam_loader_10, device=device)
        for mod in ["AEGIS-MoE", "E_topo", "E_vol", "Uniform-Ensemble"]:
            for k in metric_keys:
                runs_cam_10[mod][k].append(m_cam[mod]["precision" if k == "prec" else ("recall" if k == "rec" else k)])
        runs_cam_10["g_topo"].append(m_cam["mean_g_topo"])

        print(
            f"Seed {s} | Clean MoE: {m_clean['AEGIS-MoE']['f1']:.2f}% (E_topo: {m_clean['E_topo']['f1']:.2f}%, E_vol: {m_clean['E_vol']['f1']:.2f}%, g_topo: {m_clean['mean_g_topo']:.2f}) | "
            f"Cam MoE: {m_cam['AEGIS-MoE']['f1']:.2f}% (E_topo: {m_cam['E_topo']['f1']:.2f}%, E_vol: {m_cam['E_vol']['f1']:.2f}%, g_topo: {m_cam['mean_g_topo']:.2f})"
        )

    def summarize(runs):
        out = {}
        for mod in ["AEGIS-MoE", "E_topo", "E_vol", "Uniform-Ensemble"]:
            out[mod] = {
                "f1_mean": float(np.mean(runs[mod]["f1"])),
                "f1_std": float(np.std(runs[mod]["f1"])),
                "far_mean": float(np.mean(runs[mod]["far"])),
                "far_std": float(np.std(runs[mod]["far"])),
                "prec_mean": float(np.mean(runs[mod]["prec"])),
                "prec_std": float(np.std(runs[mod]["prec"])),
                "rec_mean": float(np.mean(runs[mod]["rec"])),
                "rec_std": float(np.std(runs[mod]["rec"])),
                "acc_mean": float(np.mean(runs[mod]["acc"])),
                "acc_std": float(np.std(runs[mod]["acc"])),
            }
        out["g_topo_mean"] = float(np.mean(runs["g_topo"]))
        out["g_topo_std"] = float(np.std(runs["g_topo"]))
        return out

    results["ctu13_10"] = {"summary": summarize(runs_clean_10), "raw": runs_clean_10}
    results["ctu13_10_cam_1.00"] = {"summary": summarize(runs_cam_10), "raw": runs_cam_10}

    # =========================================================================
    # 2. CTU-13 Scenario 9 (Dense Volumetric Star Flood)
    # =========================================================================
    print(f"\n{'='*75}\n  ABLATION REGIME: CTU-13 SCENARIO 9 (DENSE VOLUMETRIC SPAM)\n{'='*75}")
    dir_9 = "data/processed/ctu13_9"
    tr_9, _ = load_netflow_graphs(os.path.join(dir_9, "train.pt"))
    va_9, _ = load_netflow_graphs(os.path.join(dir_9, "val.pt"))
    te_9, _ = load_netflow_graphs(os.path.join(dir_9, "test.pt"))

    tr_loader_9 = DataLoader(NetFlowInMemoryDataset(tr_9), batch_size=32, shuffle=True, collate_fn=collate_netflow_batch)
    va_loader_9 = DataLoader(NetFlowInMemoryDataset(va_9), batch_size=32, shuffle=False, collate_fn=collate_netflow_batch)
    te_loader_9 = DataLoader(NetFlowInMemoryDataset(te_9), batch_size=32, shuffle=False, collate_fn=collate_netflow_batch)

    labels_9 = [g.y for g in tr_9]
    c0, c1 = labels_9.count(0), labels_9.count(1)
    tot_9 = len(labels_9)
    weights_9 = torch.tensor([tot_9 / (2.0 * max(c0, 1)), tot_9 / (2.0 * max(c1, 1))], dtype=torch.float32)

    runs_9 = {m: {k: [] for k in metric_keys} for m in ["AEGIS-MoE", "E_topo", "E_vol", "Uniform-Ensemble"]}
    runs_9["g_topo"] = []

    for s in seeds:
        print(f"--- Training seed {s} on ctu13_9 ---")
        model = train_aegis(10, 32, 2, tr_loader_9, va_loader_9, weights_9, epochs=epochs, device=device, seed=s)
        m_9 = evaluate_ablation_loader(model, te_loader_9, device=device)
        for mod in ["AEGIS-MoE", "E_topo", "E_vol", "Uniform-Ensemble"]:
            for k in metric_keys:
                runs_9[mod][k].append(m_9[mod]["precision" if k == "prec" else ("recall" if k == "rec" else k)])
        runs_9["g_topo"].append(m_9["mean_g_topo"])
        print(
            f"Seed {s} on Scenario 9 -> MoE: {m_9['AEGIS-MoE']['f1']:.2f}% | "
            f"E_topo: {m_9['E_topo']['f1']:.2f}% | E_vol: {m_9['E_vol']['f1']:.2f}% | g_topo: {m_9['mean_g_topo']:.2f}"
        )

    results["ctu13_9"] = {"summary": summarize(runs_9), "raw": runs_9}

    # =========================================================================
    # 3. NF-BoT-IoT
    # =========================================================================
    print(f"\n{'='*75}\n  ABLATION REGIME: NF-BOT-IOT\n{'='*75}")
    dir_iot = "data/processed/nf_bot_iot"
    tr_iot, _ = load_netflow_graphs(os.path.join(dir_iot, "train.pt"))
    va_iot, _ = load_netflow_graphs(os.path.join(dir_iot, "val.pt"))
    te_iot, _ = load_netflow_graphs(os.path.join(dir_iot, "test.pt"))

    tr_loader_iot = DataLoader(NetFlowInMemoryDataset(tr_iot), batch_size=128, shuffle=True, collate_fn=collate_netflow_batch)
    va_loader_iot = DataLoader(NetFlowInMemoryDataset(va_iot), batch_size=128, shuffle=False, collate_fn=collate_netflow_batch)
    te_loader_iot = DataLoader(NetFlowInMemoryDataset(te_iot), batch_size=128, shuffle=False, collate_fn=collate_netflow_batch)

    labels_iot = [g.y for g in tr_iot]
    c0, c1 = labels_iot.count(0), labels_iot.count(1)
    tot_iot = len(labels_iot)
    weights_iot = torch.tensor([tot_iot / (2.0 * max(c0, 1)), tot_iot / (2.0 * max(c1, 1))], dtype=torch.float32)

    runs_iot = {m: {k: [] for k in metric_keys} for m in ["AEGIS-MoE", "E_topo", "E_vol", "Uniform-Ensemble"]}
    runs_iot["g_topo"] = []

    for s in seeds:
        print(f"--- Training seed {s} on nf_bot_iot ---")
        model = train_aegis(10, 32, 2, tr_loader_iot, va_loader_iot, weights_iot, epochs=15, device=device, seed=s)
        m_iot = evaluate_ablation_loader(model, te_loader_iot, device=device)
        for mod in ["AEGIS-MoE", "E_topo", "E_vol", "Uniform-Ensemble"]:
            for k in metric_keys:
                runs_iot[mod][k].append(m_iot[mod]["precision" if k == "prec" else ("recall" if k == "rec" else k)])
        runs_iot["g_topo"].append(m_iot["mean_g_topo"])
        print(
            f"Seed {s} on NF-BoT-IoT -> MoE: {m_iot['AEGIS-MoE']['f1']:.2f}% | "
            f"E_topo: {m_iot['E_topo']['f1']:.2f}% | E_vol: {m_iot['E_vol']['f1']:.2f}% | g_topo: {m_iot['mean_g_topo']:.2f}"
        )

    results["nf_bot_iot"] = {"summary": summarize(runs_iot), "raw": runs_iot}

    out_file = "benchmarks/aegis_ablation_results.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[DONE] Saved comprehensive ablation results to {out_file}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46])
    parser.add_argument("--epochs", type=int, default=35)
    args = parser.parse_args()
    run_ablation(seeds=args.seeds, epochs=args.epochs)
