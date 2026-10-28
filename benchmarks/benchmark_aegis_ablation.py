#!/usr/bin/env python3
"""
Benchmark: MoE Dual-Expert Standalone Ablation Study (Table IV Verification)
Target Venue: IEEE Transactions on Information Forensics and Security (TIFS)

Rigorous independent ablation of the two physical modalities in AEGIS:
  1. E_topo Alone: Standalone Open Non-Hermitian Continuous-Time Quantum Walk (AegisClassifier without volumetric/moe)
  2. E_vol Alone: Standalone Directed Multiset Volumetric Aggregator (StandaloneVolumetricClassifier)
  3. Uniform Ensemble: Equal static weighting (0.5 * p_topo + 0.5 * p_vol) with optimal validation threshold calibration
  4. Full AEGIS-MoE: Dynamic topological-priority structural gating router

All models trained completely independently from scratch across independent random seeds (42..46) on:
  - CTU-13 Scenario 10 (Clean Sparse P2P Overlay)
  - CTU-13 Scenario 10 Camouflage (alpha = 1.00, Chordal Regularization)
  - CTU-13 Scenario 9 (Dense Volumetric Bipartite Spam Flood)
  - NF-BoT-IoT (Large-Scale Multi-Class IoT Telemetry)

Saves verified metrics to benchmarks/aegis_ablation_results.json.
"""

import os
import sys
import copy
import json
import time
import argparse
from typing import List, Dict, Tuple, Optional, Any
import numpy as np
import scipy.stats
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix

# Ensure project root is in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data.netflow_graph_builder import (
    NetFlowGraph,
    NetFlowInMemoryDataset,
    collate_netflow_batch,
    load_netflow_graphs,
)
from models.aegis_classifier import AegisClassifier, StandaloneVolumetricClassifier
from benchmarks.benchmark_camouflage import apply_topological_camouflage


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


def compute_metrics(y_true: np.ndarray, y_score: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    """Computes comprehensive NIDS forensic performance metrics."""
    preds = (y_score >= threshold).astype(int)
    macro_f1 = f1_score(y_true, preds, average="macro", zero_division=0) * 100.0
    macro_prec = precision_score(y_true, preds, average="macro", zero_division=0) * 100.0
    macro_rec = recall_score(y_true, preds, average="macro", zero_division=0) * 100.0
    tn, fp, fn, tp = confusion_matrix(y_true, preds, labels=[0, 1]).ravel()
    far = (fp / max(fp + tn, 1e-6)) * 100.0
    acc = float((preds == y_true).mean() * 100.0)
    auc = compute_roc_auc_numpy(y_true, y_score)
    return {
        "f1": float(macro_f1),
        "precision": float(macro_prec),
        "recall": float(macro_rec),
        "far": float(far),
        "acc": float(acc),
        "auc": float(auc),
    }


def predict_probs_and_gating(
    model: nn.Module,
    loader: DataLoader,
    device: str,
    is_aegis_moe: bool = False,
) -> Tuple[np.ndarray, np.ndarray, Optional[float]]:
    """Predicts class 1 probabilities and returns mean g_topo if MoE."""
    model.eval()
    all_probs = []
    all_trues = []
    g_topos = []

    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)
            adj = batch["adj"].to(device)
            mask = batch["mask"].to(device)
            y = batch["labels"].numpy()

            if is_aegis_moe:
                logits, _, _, gating = model(x, adj, mask=mask, return_all=True)
                g_topos.extend(gating[:, 0].cpu().numpy())
            else:
                logits = model(x, adj, mask=mask)

            probs = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
            all_probs.extend(probs)
            all_trues.extend(y)

    probs = np.array(all_probs)
    trues = np.array(all_trues)
    mean_g_topo = float(np.mean(g_topos)) if len(g_topos) > 0 else None
    return probs, trues, mean_g_topo


def calibrate_threshold(y_val: np.ndarray, probs_val: np.ndarray) -> float:
    """Finds optimal decision threshold tau* in [0.35, 0.65] maximizing validation Macro-F1."""
    best_tau, best_f1 = 0.50, -1.0
    for tau in np.linspace(0.35, 0.65, 31):
        preds = (probs_val >= tau).astype(int)
        f1 = f1_score(y_val, preds, average="macro", zero_division=0) * 100.0
        if f1 > best_f1:
            best_f1 = f1
            best_tau = float(tau)
    return best_tau


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    class_weights: torch.Tensor,
    epochs: int = 35,
    lr: float = 0.003,
    device: str = "cpu",
    is_aegis_moe: bool = False,
) -> nn.Module:
    """Trains model with CosineAnnealingLR, validation checkpointing every 2 epochs, gradient clipping."""
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
            if is_aegis_moe:
                logits, l_topo, l_vol, gating = model(x, adj, mask=mask, return_all=True)
                loss_main = criterion(logits, y)
                loss_topo = criterion(l_topo, y)
                loss_vol = criterion(l_vol, y)
                loss = loss_main + 0.3 * loss_topo + 0.3 * loss_vol
            else:
                logits = model(x, adj, mask=mask)
                loss = criterion(logits, y)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        scheduler.step()

        # Validation checkpointing every 2 epochs or at end
        if (epoch + 1) % 2 == 0 or epoch == 0 or epoch == epochs - 1:
            probs_val, y_val, _ = predict_probs_and_gating(model, val_loader, device=device, is_aegis_moe=is_aegis_moe)
            tau = calibrate_threshold(y_val, probs_val)
            val_m = compute_metrics(y_val, probs_val, threshold=tau)
            if val_m["f1"] > best_val_f1:
                best_val_f1 = val_m["f1"]
                best_weights = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_weights)
    return model


def run_ablation(
    seeds: List[int] = [42, 43, 44, 45, 46],
    epochs: int = 35,
    device: Optional[str] = None,
) -> Dict[str, Any]:
    if device is None:
        device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Running MoE Dual-Expert Standalone Ablation on device: {device}")

    metric_keys = ["f1", "precision", "recall", "far", "acc", "auc"]
    modality_keys = ["AEGIS-MoE", "E_topo", "E_vol", "Uniform-Ensemble"]

    results: Dict[str, Any] = {}

    def init_run_dict():
        d = {m: {k: [] for k in metric_keys} for m in modality_keys}
        d["g_topo"] = []
        return d

    def summarize(runs):
        out = {}
        for mod in modality_keys:
            out[mod] = {}
            for k in metric_keys:
                prefix = "prec" if k == "precision" else ("rec" if k == "recall" else k)
                out[mod][f"{prefix}_mean"] = float(np.mean(runs[mod][k]))
                out[mod][f"{prefix}_std"] = float(np.std(runs[mod][k]))
        out["g_topo_mean"] = float(np.mean(runs["g_topo"])) if len(runs["g_topo"]) > 0 else 0.0
        out["g_topo_std"] = float(np.std(runs["g_topo"])) if len(runs["g_topo"]) > 0 else 0.0
        return out

    # =========================================================================
    # 1. CTU-13 Scenario 10: Clean AND Camouflage (alpha = 1.00)
    # =========================================================================
    print(f"\n{'='*75}\n  ABLATION REGIME: CTU-13 SCENARIO 10 (CLEAN & CAMOUFLAGE)\n{'='*75}")
    dir_10 = "data/processed/ctu13_10"
    tr_10, _ = load_netflow_graphs(os.path.join(dir_10, "train.pt"))
    va_10, _ = load_netflow_graphs(os.path.join(dir_10, "val.pt"))
    te_10, _ = load_netflow_graphs(os.path.join(dir_10, "test.pt"))

    # Generate camouflage test set with exact edge-conservation
    benign_nodes = [g.x for g in tr_10 if g.y == 0]
    benign_mean = torch.cat(benign_nodes, dim=0).mean(dim=0)
    cam_10_te = apply_topological_camouflage(copy.deepcopy(te_10), alpha=1.00, benign_mean=benign_mean)

    tr_loader_10 = DataLoader(NetFlowInMemoryDataset(tr_10), batch_size=32, shuffle=True, collate_fn=collate_netflow_batch)
    va_loader_10 = DataLoader(NetFlowInMemoryDataset(va_10), batch_size=32, shuffle=False, collate_fn=collate_netflow_batch)
    te_clean_loader_10 = DataLoader(NetFlowInMemoryDataset(te_10), batch_size=32, shuffle=False, collate_fn=collate_netflow_batch)
    te_cam_loader_10 = DataLoader(NetFlowInMemoryDataset(cam_10_te), batch_size=32, shuffle=False, collate_fn=collate_netflow_batch)

    labels_10 = [g.y for g in tr_10]
    c0, c1 = labels_10.count(0), labels_10.count(1)
    tot_10 = len(labels_10)
    weights_10 = torch.tensor([tot_10 / (2.0 * max(c0, 1)), tot_10 / (2.0 * max(c1, 1))], dtype=torch.float32)

    runs_clean_10 = init_run_dict()
    runs_cam_10 = init_run_dict()

    for s in seeds:
        print(f"\n--- [CTU-13-10] Training seed {s} ---")
        t0 = time.time()

        # 1. E_topo Alone (Standalone non-Hermitian quantum walk)
        torch.manual_seed(s)
        np.random.seed(s)
        m_topo = AegisClassifier(
            in_dim=10, hidden_dim=32, num_classes=2, num_layers=2,
            use_volumetric=False, use_moe=False
        ).to(device)
        m_topo = train_model(m_topo, tr_loader_10, va_loader_10, weights_10, epochs=epochs, lr=0.003, device=device)
        p_topo_val, y_val_10, _ = predict_probs_and_gating(m_topo, va_loader_10, device=device)
        tau_topo = calibrate_threshold(y_val_10, p_topo_val)
        p_topo_clean, y_te_clean, _ = predict_probs_and_gating(m_topo, te_clean_loader_10, device=device)
        p_topo_cam, y_te_cam, _ = predict_probs_and_gating(m_topo, te_cam_loader_10, device=device)
        m_topo_clean = compute_metrics(y_te_clean, p_topo_clean, threshold=tau_topo)
        m_topo_cam = compute_metrics(y_te_cam, p_topo_cam, threshold=tau_topo)

        # 2. E_vol Alone (Standalone directed multiset accumulator)
        torch.manual_seed(s)
        np.random.seed(s)
        m_vol = StandaloneVolumetricClassifier(in_dim=10, hidden_dim=32, num_classes=2, num_layers=2).to(device)
        m_vol = train_model(m_vol, tr_loader_10, va_loader_10, weights_10, epochs=epochs, lr=0.003, device=device)
        p_vol_val, _, _ = predict_probs_and_gating(m_vol, va_loader_10, device=device)
        tau_vol = calibrate_threshold(y_val_10, p_vol_val)
        p_vol_clean, _, _ = predict_probs_and_gating(m_vol, te_clean_loader_10, device=device)
        p_vol_cam, _, _ = predict_probs_and_gating(m_vol, te_cam_loader_10, device=device)
        m_vol_clean = compute_metrics(y_te_clean, p_vol_clean, threshold=tau_vol)
        m_vol_cam = compute_metrics(y_te_cam, p_vol_cam, threshold=tau_vol)

        # 3. Uniform Ensemble (Static 50/50 probability fusion of standalone experts)
        p_unif_val = 0.5 * p_topo_val + 0.5 * p_vol_val
        tau_unif = calibrate_threshold(y_val_10, p_unif_val)
        p_unif_clean = 0.5 * p_topo_clean + 0.5 * p_vol_clean
        p_unif_cam = 0.5 * p_topo_cam + 0.5 * p_vol_cam
        m_unif_clean = compute_metrics(y_te_clean, p_unif_clean, threshold=tau_unif)
        m_unif_cam = compute_metrics(y_te_cam, p_unif_cam, threshold=tau_unif)

        # 4. Full AEGIS-MoE (Dynamic topological-priority structural gating)
        torch.manual_seed(s)
        np.random.seed(s)
        m_moe = AegisClassifier(
            in_dim=10, hidden_dim=32, num_classes=2, num_layers=2,
            use_volumetric=True, use_moe=True, moe_routing_mode="topological_priority"
        ).to(device)
        m_moe = train_model(m_moe, tr_loader_10, va_loader_10, weights_10, epochs=epochs, lr=0.003, device=device, is_aegis_moe=True)
        p_moe_val, _, _ = predict_probs_and_gating(m_moe, va_loader_10, device=device, is_aegis_moe=True)
        tau_moe = calibrate_threshold(y_val_10, p_moe_val)
        p_moe_clean, _, g_clean = predict_probs_and_gating(m_moe, te_clean_loader_10, device=device, is_aegis_moe=True)
        p_moe_cam, _, g_cam = predict_probs_and_gating(m_moe, te_cam_loader_10, device=device, is_aegis_moe=True)
        m_moe_clean = compute_metrics(y_te_clean, p_moe_clean, threshold=tau_moe)
        m_moe_cam = compute_metrics(y_te_cam, p_moe_cam, threshold=tau_moe)

        dur = time.time() - t0
        print(f"  Seed {s} ({dur:.1f}s):")
        print(f"    Clean -> MoE: {m_moe_clean['f1']:.2f}% | E_topo: {m_topo_clean['f1']:.2f}% | E_vol: {m_vol_clean['f1']:.2f}% | Unif: {m_unif_clean['f1']:.2f}% | g_topo: {g_clean:.2f}")
        print(f"    Cam   -> MoE: {m_moe_cam['f1']:.2f}% | E_topo: {m_topo_cam['f1']:.2f}% | E_vol: {m_vol_cam['f1']:.2f}% | Unif: {m_unif_cam['f1']:.2f}% | g_topo: {g_cam:.2f}")

        # Store Clean
        runs_clean_10["g_topo"].append(g_clean)
        for k in metric_keys:
            runs_clean_10["AEGIS-MoE"][k].append(m_moe_clean[k])
            runs_clean_10["E_topo"][k].append(m_topo_clean[k])
            runs_clean_10["E_vol"][k].append(m_vol_clean[k])
            runs_clean_10["Uniform-Ensemble"][k].append(m_unif_clean[k])

        # Store Cam
        runs_cam_10["g_topo"].append(g_cam)
        for k in metric_keys:
            runs_cam_10["AEGIS-MoE"][k].append(m_moe_cam[k])
            runs_cam_10["E_topo"][k].append(m_topo_cam[k])
            runs_cam_10["E_vol"][k].append(m_vol_cam[k])
            runs_cam_10["Uniform-Ensemble"][k].append(m_unif_cam[k])

    results["ctu13_10"] = {"summary": summarize(runs_clean_10), "raw": runs_clean_10}
    results["ctu13_10_cam_1.00"] = {"summary": summarize(runs_cam_10), "raw": runs_cam_10}

    # =========================================================================
    # 2. CTU-13 Scenario 9 (Dense Volumetric Bipartite Spam Flood)
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

    runs_9 = init_run_dict()

    for s in seeds:
        print(f"\n--- [CTU-13-9] Training seed {s} ---")
        t0 = time.time()

        # 1. E_topo Alone
        torch.manual_seed(s)
        np.random.seed(s)
        m_topo = AegisClassifier(
            in_dim=10, hidden_dim=32, num_classes=2, num_layers=2,
            use_volumetric=False, use_moe=False
        ).to(device)
        m_topo = train_model(m_topo, tr_loader_9, va_loader_9, weights_9, epochs=epochs, lr=0.003, device=device)
        p_topo_val, y_val_9, _ = predict_probs_and_gating(m_topo, va_loader_9, device=device)
        tau_topo = calibrate_threshold(y_val_9, p_topo_val)
        p_topo_te, y_te_9, _ = predict_probs_and_gating(m_topo, te_loader_9, device=device)
        m_topo_9 = compute_metrics(y_te_9, p_topo_te, threshold=tau_topo)

        # 2. E_vol Alone
        torch.manual_seed(s)
        np.random.seed(s)
        m_vol = StandaloneVolumetricClassifier(in_dim=10, hidden_dim=32, num_classes=2, num_layers=2).to(device)
        m_vol = train_model(m_vol, tr_loader_9, va_loader_9, weights_9, epochs=epochs, lr=0.003, device=device)
        p_vol_val, _, _ = predict_probs_and_gating(m_vol, va_loader_9, device=device)
        tau_vol = calibrate_threshold(y_val_9, p_vol_val)
        p_vol_te, _, _ = predict_probs_and_gating(m_vol, te_loader_9, device=device)
        m_vol_9 = compute_metrics(y_te_9, p_vol_te, threshold=tau_vol)

        # 3. Uniform Ensemble
        p_unif_val = 0.5 * p_topo_val + 0.5 * p_vol_val
        tau_unif = calibrate_threshold(y_val_9, p_unif_val)
        p_unif_te = 0.5 * p_topo_te + 0.5 * p_vol_te
        m_unif_9 = compute_metrics(y_te_9, p_unif_te, threshold=tau_unif)

        # 4. Full AEGIS-MoE
        torch.manual_seed(s)
        np.random.seed(s)
        m_moe = AegisClassifier(
            in_dim=10, hidden_dim=32, num_classes=2, num_layers=2,
            use_volumetric=True, use_moe=True, moe_routing_mode="topological_priority"
        ).to(device)
        m_moe = train_model(m_moe, tr_loader_9, va_loader_9, weights_9, epochs=epochs, lr=0.003, device=device, is_aegis_moe=True)
        p_moe_val, _, _ = predict_probs_and_gating(m_moe, va_loader_9, device=device, is_aegis_moe=True)
        tau_moe = calibrate_threshold(y_val_9, p_moe_val)
        p_moe_te, _, g_te = predict_probs_and_gating(m_moe, te_loader_9, device=device, is_aegis_moe=True)
        m_moe_9 = compute_metrics(y_te_9, p_moe_te, threshold=tau_moe)

        dur = time.time() - t0
        print(f"  Seed {s} ({dur:.1f}s):")
        print(f"    Scenario 9 -> MoE: {m_moe_9['f1']:.2f}% | E_topo: {m_topo_9['f1']:.2f}% | E_vol: {m_vol_9['f1']:.2f}% | Unif: {m_unif_9['f1']:.2f}% | g_topo: {g_te:.2f}")

        runs_9["g_topo"].append(g_te)
        for k in metric_keys:
            runs_9["AEGIS-MoE"][k].append(m_moe_9[k])
            runs_9["E_topo"][k].append(m_topo_9[k])
            runs_9["E_vol"][k].append(m_vol_9[k])
            runs_9["Uniform-Ensemble"][k].append(m_unif_9[k])

    results["ctu13_9"] = {"summary": summarize(runs_9), "raw": runs_9}

    # =========================================================================
    # 3. NF-BoT-IoT (Large-Scale Multi-Class IoT Telemetry)
    # =========================================================================
    print(f"\n{'='*75}\n  ABLATION REGIME: NF-BOT-IOT (IOT TELEMETRY)\n{'='*75}")
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

    runs_iot = init_run_dict()
    epochs_iot = min(epochs, 20)

    for s in seeds:
        print(f"\n--- [NF-BoT-IoT] Training seed {s} ---")
        t0 = time.time()

        # 1. E_topo Alone
        torch.manual_seed(s)
        np.random.seed(s)
        m_topo = AegisClassifier(
            in_dim=10, hidden_dim=32, num_classes=2, num_layers=2,
            use_volumetric=False, use_moe=False
        ).to(device)
        m_topo = train_model(m_topo, tr_loader_iot, va_loader_iot, weights_iot, epochs=epochs_iot, lr=0.003, device=device)
        p_topo_val, y_val_iot, _ = predict_probs_and_gating(m_topo, va_loader_iot, device=device)
        tau_topo = calibrate_threshold(y_val_iot, p_topo_val)
        p_topo_te, y_te_iot, _ = predict_probs_and_gating(m_topo, te_loader_iot, device=device)
        m_topo_iot = compute_metrics(y_te_iot, p_topo_te, threshold=tau_topo)

        # 2. E_vol Alone
        torch.manual_seed(s)
        np.random.seed(s)
        m_vol = StandaloneVolumetricClassifier(in_dim=10, hidden_dim=32, num_classes=2, num_layers=2).to(device)
        m_vol = train_model(m_vol, tr_loader_iot, va_loader_iot, weights_iot, epochs=epochs_iot, lr=0.003, device=device)
        p_vol_val, _, _ = predict_probs_and_gating(m_vol, va_loader_iot, device=device)
        tau_vol = calibrate_threshold(y_val_iot, p_vol_val)
        p_vol_te, _, _ = predict_probs_and_gating(m_vol, te_loader_iot, device=device)
        m_vol_iot = compute_metrics(y_te_iot, p_vol_te, threshold=tau_vol)

        # 3. Uniform Ensemble
        p_unif_val = 0.5 * p_topo_val + 0.5 * p_vol_val
        tau_unif = calibrate_threshold(y_val_iot, p_unif_val)
        p_unif_te = 0.5 * p_topo_te + 0.5 * p_vol_te
        m_unif_iot = compute_metrics(y_te_iot, p_unif_te, threshold=tau_unif)

        # 4. Full AEGIS-MoE
        torch.manual_seed(s)
        np.random.seed(s)
        m_moe = AegisClassifier(
            in_dim=10, hidden_dim=32, num_classes=2, num_layers=2,
            use_volumetric=True, use_moe=True, moe_routing_mode="topological_priority"
        ).to(device)
        m_moe = train_model(m_moe, tr_loader_iot, va_loader_iot, weights_iot, epochs=epochs_iot, lr=0.003, device=device, is_aegis_moe=True)
        p_moe_val, _, _ = predict_probs_and_gating(m_moe, va_loader_iot, device=device, is_aegis_moe=True)
        tau_moe = calibrate_threshold(y_val_iot, p_moe_val)
        p_moe_te, _, g_te = predict_probs_and_gating(m_moe, te_loader_iot, device=device, is_aegis_moe=True)
        m_moe_iot = compute_metrics(y_te_iot, p_moe_te, threshold=tau_moe)

        dur = time.time() - t0
        print(f"  Seed {s} ({dur:.1f}s):")
        print(f"    NF-BoT-IoT -> MoE: {m_moe_iot['f1']:.2f}% | E_topo: {m_topo_iot['f1']:.2f}% | E_vol: {m_vol_iot['f1']:.2f}% | Unif: {m_unif_iot['f1']:.2f}% | g_topo: {g_te:.2f}")

        runs_iot["g_topo"].append(g_te)
        for k in metric_keys:
            runs_iot["AEGIS-MoE"][k].append(m_moe_iot[k])
            runs_iot["E_topo"][k].append(m_topo_iot[k])
            runs_iot["E_vol"][k].append(m_vol_iot[k])
            runs_iot["Uniform-Ensemble"][k].append(m_unif_iot[k])

    results["nf_bot_iot"] = {"summary": summarize(runs_iot), "raw": runs_iot}

    # Save to JSON
    out_file = "benchmarks/aegis_ablation_results.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[DONE] Saved verified standalone ablation results to {out_file}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46])
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()
    run_ablation(seeds=args.seeds, epochs=args.epochs, device=args.device)
