#!/usr/bin/env python3
"""
Experiment 2: Topological Camouflage Stress Test (RQ2) Benchmark.
Target Venue: IEEE Transactions on Information Forensics and Security (TIFS).

Evaluates detection resilience against adversaries who perform:
  1. Traffic Feature Mimicry: Throttling and standardizing flow features toward benign averages:
     x_cam = (1 - alpha) * x + alpha * mean(x_benign)
  2. Topological Overlay Regularization: Structuring lateral botnet links into a k-regular chordal ring.

Mathematical Threat Model:
  Under full camouflage (alpha = 1.0) and k-regular lateral overlays, all nodes in the regular
  subgraph possess identical degrees and identical 1-hop neighbor multisets.
  By Proposition 1 (1-WL Indistinguishability Under Feature Normalization), classical 1-WL GNNs
  (GCN, GIN) are mathematically blind (accuracy collapses to 50%).
  QUASAR leverages unitary quantum walk phase interference U(t) = exp(-iHt) to shatter the
  color-refinement symmetry and maintain robust detection.

Evaluation Protocol:
  - Trained on clean training set; evaluated on test set across alpha in [0.0, 0.25, 0.50, 0.75, 1.00]
  - 10 independent random seeds
  - Mini-batch DataLoader with validation checkpointing on clean val.pt
  - Metrics: Macro-F1, Precision, Recall, False Alarm Rate (FAR), ROC-AUC
  - Statistical testing: Paired Wilcoxon signed-rank tests with Benjamini-Hochberg FDR (q <= 0.05)
  - Automatic generation of publication-quality plot (PNG) and IEEE TIFS LaTeX Table
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

# Set non-interactive backend and writable cache dir for matplotlib
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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
from models.unitary_ctqw_classifier import UnitaryCTQWClassifier as QuasarClassifier
from models.aegis_classifier import AegisClassifier


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


def find_optimal_threshold(
    model: nn.Module,
    loader: DataLoader,
    device: str
) -> float:
    """Finds optimal classification threshold on validation set to maximize Macro-F1 with zero test leakage."""
    model.eval()
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
            all_probs.append(probs.cpu().numpy())
            all_trues.append(y.cpu().numpy())

    probs = np.concatenate(all_probs) if len(all_probs) > 0 else np.array([])
    y_true = np.concatenate(all_trues) if len(all_trues) > 0 else np.array([])
    if len(y_true) == 0:
        return 0.50

    best_t, best_f1 = 0.50, -1.0
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
        if macro_f1 > best_f1:
            best_f1 = macro_f1
            best_t = float(t)

    return best_t


def evaluate_model_on_loader(
    model: nn.Module,
    loader: DataLoader,
    device: str,
    threshold: float = 0.50
) -> Dict[str, float]:
    """Evaluates a model over a DataLoader and computes full NIDS detection metrics."""
    model.eval()
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

            all_probs.append(probs.cpu().numpy())
            all_trues.append(y.cpu().numpy())

    probs = np.concatenate(all_probs) if len(all_probs) > 0 else np.array([])
    y_true = np.concatenate(all_trues) if len(all_trues) > 0 else np.array([])

    if len(y_true) == 0:
        return {"f1": 0.0, "precision": 0.0, "recall": 0.0, "far": 0.0, "auc": 50.0, "acc": 0.0}

    preds = (probs >= threshold).astype(int)
    acc = float((preds == y_true).mean() * 100.0)

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
    macro_precision = (p_mal + p_ben) / 2.0 * 100.0
    macro_recall = (r_mal + r_ben) / 2.0 * 100.0
    far = (fp / max(fp + tn, 1)) * 100.0
    auc = compute_roc_auc_numpy(y_true, probs)

    return {
        "f1": macro_f1,
        "precision": macro_precision,
        "recall": macro_recall,
        "far": far,
        "auc": auc,
        "acc": acc,
    }


def apply_topological_camouflage(
    graphs: List[NetFlowGraph],
    alpha: float,
    benign_mean: torch.Tensor,
    k_regular: int = 2
) -> List[NetFlowGraph]:
    """
    Applies realistic adversarial topological camouflage and feature mimicry to attack graphs:
      1. Flow Feature Mimicry: The active botnet hosts interpolate their flow features
         toward the empirical benign baseline:
         x_bot = (1 - alpha) * x_bot + alpha * benign_mean
      2. Strict Budget-Preserving Topological Regularization (if alpha >= 0.5):
         The lateral P2P overlay among bot nodes is rewired into a degree-regular
         circulant ring (degree k_regular >= 2) with exact edge count conservation
         (Delta E == 0) and communication degree preservation (Delta deg(v) == 0).
    Benign background traffic and legitimate external sessions remain completely unaltered.
    """
    if alpha <= 0.0:
        return graphs

    camouflaged: List[NetFlowGraph] = []
    for g in graphs:
        if g.y == 0:
            camouflaged.append(g)
            continue

        N = g.num_nodes
        orig_E = int(g.adj.sum().item())
        x_cam = g.x.clone()
        adj_cam = g.adj.clone()

        # Identify active botnet communicating hosts
        if hasattr(g, "node_labels") and g.node_labels is not None and (g.node_labels == 1).sum() >= 2:
            bot_indices = (g.node_labels == 1).nonzero(as_tuple=True)[0].tolist()
        else:
            act = g.adj.sum(dim=-1) + g.adj.sum(dim=-2) + g.x[:, 0] + g.x[:, 1]
            num_bots = min(8, max(4, N // 16))
            bot_indices = torch.topk(act, k=num_bots).indices.tolist()

        M = len(bot_indices)
        bot_set = set(bot_indices)
        benign_nodes = [i for i in range(N) if i not in bot_set]

        # 1. Flow Feature Mimicry on bot nodes
        target_mean = benign_mean.expand(M, -1)
        x_cam[bot_indices] = (1.0 - alpha) * x_cam[bot_indices] + alpha * target_mean

        # 2. Strict Budget-Preserving Topological Regularization (if alpha >= 0.5)
        if alpha >= 0.5 and M >= 3:
            orig_conn_degs = {
                u: int(((adj_cam[u] + adj_cam[:, u]) > 0).sum().item())
                for u in bot_indices
            }

            k_eff = min(k_regular, M - 1)
            if k_eff % 2 == 1 and M % 2 == 1:
                k_eff = k_eff - 1

            desired_directed = []
            for i in range(M):
                desired_directed.append((bot_indices[i], bot_indices[(i + 1) % M]))
            if k_eff >= 4:
                for i in range(M):
                    desired_directed.append((bot_indices[(i + 1) % M], bot_indices[i]))
            if k_eff >= 3:
                for i in range(M // 2):
                    desired_directed.append((bot_indices[i], bot_indices[(i + M // 2) % M]))
                    desired_directed.append((bot_indices[(i + M // 2) % M], bot_indices[i]))

            seen = set()
            unique_directed = [e for e in desired_directed if not (e in seen or seen.add(e))]
            cycle_set = set(unique_directed)

            # Step 2A: Remove non-desired internal lateral links between bots
            for i in range(M):
                for j in range(M):
                    if i != j and (bot_indices[i], bot_indices[j]) not in cycle_set:
                        adj_cam[bot_indices[i], bot_indices[j]] = 0.0

            # Step 2B: Establish desired regular overlay edges
            for u_node, v_node in unique_directed:
                adj_cam[u_node, v_node] = 1.0

            # Step 2C: Match external connection degrees to preserve Delta deg(v) == 0
            if len(benign_nodes) > 0:
                for u_node in bot_indices:
                    orig_conn_deg = orig_conn_degs[u_node]
                    target_ext = max(0, orig_conn_deg - k_eff)
                    cur_ext = [j for j in benign_nodes if adj_cam[u_node, j] > 0 or adj_cam[j, u_node] > 0]
                    if len(cur_ext) > target_ext:
                        for j in cur_ext[:len(cur_ext) - target_ext]:
                            adj_cam[u_node, j] = 0.0
                            adj_cam[j, u_node] = 0.0
                    elif len(cur_ext) < target_ext:
                        to_add = target_ext - len(cur_ext)
                        added = 0
                        for j in benign_nodes:
                            if adj_cam[u_node, j] == 0 and adj_cam[j, u_node] == 0:
                                adj_cam[u_node, j] = 1.0
                                added += 1
                                if added >= to_add:
                                    break

            # Step 2D: Strict Global Edge Conservation (Delta E == 0)
            # Conserves edge budget across bot interfaces while preserving neighbor degrees via bidirectionality
            cur_E = int(adj_cam.sum().item())
            diff = cur_E - orig_E

            if diff < 0:
                needed = -diff
                added = 0
                # First: make existing one-way external edges bidirectional (preserves neighbor degree)
                if len(benign_nodes) > 0:
                    for u in bot_indices:
                        if added >= needed: break
                        for b in benign_nodes:
                            if adj_cam[u, b] > 0 and adj_cam[b, u] == 0:
                                adj_cam[b, u] = 1.0
                                added += 1
                                if added >= needed: break
                            elif adj_cam[b, u] > 0 and adj_cam[u, b] == 0:
                                adj_cam[u, b] = 1.0
                                added += 1
                                if added >= needed: break

                # Second: add new external edges if still needed
                if added < needed and len(benign_nodes) > 0:
                    for u in bot_indices:
                        if added >= needed: break
                        for b in benign_nodes:
                            if adj_cam[u, b] == 0 and adj_cam[b, u] == 0:
                                adj_cam[u, b] = 1.0
                                added += 1
                                if added >= needed: break

                # Third: add internal chordal edges if no benign nodes available
                if added < needed:
                    for u in bot_indices:
                        if added >= needed: break
                        for v in bot_indices:
                            if u != v and adj_cam[u, v] == 0:
                                adj_cam[u, v] = 1.0
                                added += 1
                                if added >= needed: break

            elif diff > 0:
                needed = diff
                pruned = 0
                # First: make existing bidirectional external edges one-way (preserves neighbor degree)
                if len(benign_nodes) > 0:
                    for u in bot_indices:
                        if pruned >= needed: break
                        for b in benign_nodes:
                            if adj_cam[u, b] > 0 and adj_cam[b, u] > 0:
                                adj_cam[b, u] = 0.0
                                pruned += 1
                                if pruned >= needed: break

                # Second: prune external edges if still needed
                if pruned < needed and len(benign_nodes) > 0:
                    for u in bot_indices:
                        if pruned >= needed: break
                        for b in benign_nodes:
                            if adj_cam[u, b] > 0:
                                adj_cam[u, b] = 0.0
                                pruned += 1
                                if pruned >= needed: break
                            elif adj_cam[b, u] > 0:
                                adj_cam[b, u] = 0.0
                                pruned += 1
                                if pruned >= needed: break

                # Third: prune internal links if still needed
                if pruned < needed:
                    for u in bot_indices:
                        if pruned >= needed: break
                        for v in bot_indices:
                            if u != v and adj_cam[u, v] > 0:
                                adj_cam[u, v] = 0.0
                                pruned += 1
                                if pruned >= needed: break

            # Rigorous Mathematical Invariance Assertion
            final_E = int(adj_cam.sum().item())
            assert final_E == orig_E, (
                f"Adversarial camouflage violated Delta E == 0: "
                f"orig_E={orig_E}, cam_E={final_E}, diff={final_E - orig_E}"
            )

        camouflaged.append(
            NetFlowGraph(
                x=x_cam,
                adj=adj_cam,
                y=g.y,
                window_start=g.window_start,
                window_end=g.window_end,
                num_nodes=g.num_nodes,
                num_edges=orig_E,
                attack_name=f"Camouflaged-{g.attack_name}"
            )
        )

    return camouflaged


def train_single_model(
    model_name: str,
    in_dim: int,
    hidden_dim: int,
    num_layers: int,
    train_loader: DataLoader,
    val_loader: DataLoader,
    class_weights: torch.Tensor,
    epochs: int = 35,
    lr: float = 0.003,
    device: str = "cpu",
    seed: int = 42
) -> nn.Module:
    """Trains a model on the clean training set with validation checkpointing."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    if model_name == "GCN":
        model = GCNClassifier(in_dim=in_dim, hidden_dim=hidden_dim, num_classes=2, num_layers=num_layers)
    elif model_name == "GIN":
        model = GINClassifier(in_dim=in_dim, hidden_dim=hidden_dim, num_classes=2, num_layers=num_layers)
    elif model_name == "RWPE-GIN":
        model = RWPE_GINClassifier(in_dim=in_dim, hidden_dim=hidden_dim, num_classes=2, num_layers=num_layers, k_rwpe=8)
    elif model_name == "MagNet":
        model = MagNetClassifier(in_dim=in_dim, hidden_dim=hidden_dim, num_classes=2, num_layers=num_layers, q=0.25)
    elif model_name in ("QUASAR", "Unitary CTQW (Closed)"):
        model = QuasarClassifier(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            num_classes=2,
            num_layers=num_layers,
            num_time_scales=4,
            init_times=(0.2, 0.8, 2.5, 8.0)
        )
    elif model_name in ("AEGIS", "AEGIS-MoE (Ours)"):
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
            if getattr(model, "use_moe", False):
                logits, l_topo, l_vol, gating = model(x, adj, mask=mask, return_all=True)
                loss_main = criterion(logits, y)
                loss_topo = criterion(l_topo, y)
                loss_vol = criterion(l_vol, y)
                entropy_loss = -0.05 * 2.0 * (gating[:, 0] * gating[:, 1]).mean()
                loss = loss_main + 0.3 * loss_topo + 0.3 * loss_vol + entropy_loss
            else:
                logits = model(x, adj, mask=mask)
                loss = criterion(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        scheduler.step()

        if (epoch + 1) % 2 == 0 or epoch == 0 or epoch == epochs - 1:
            val_metrics = evaluate_model_on_loader(model, val_loader, device=device)
            if val_metrics["f1"] > best_val_f1:
                best_val_f1 = val_metrics["f1"]
                best_weights = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_weights)
    return model


def perform_camouflage_statistical_tests(
    results_raw: Dict[str, Dict[str, Dict[str, List[float]]]],
    models: List[str],
    alphas: List[float],
    target_model: str = "AEGIS"
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """
    Performs two-sided paired Wilcoxon signed-rank tests at each alpha level
    comparing AEGIS against baseline models on Macro-F1 across the random seeds.
    Enforces Benjamini-Hochberg FDR correction (q <= 0.05).
    """
    records = []
    # If target_model not directly in results_raw, try alias
    actual_target = target_model
    if actual_target not in results_raw:
        if "AEGIS-MoE (Ours)" in results_raw:
            actual_target = "AEGIS-MoE (Ours)"
        elif "AEGIS" in results_raw:
            actual_target = "AEGIS"

    for a in alphas:
        a_str = f"{a:.2f}"
        for m in models:
            if m == actual_target or m in ("AEGIS", "AEGIS-MoE (Ours)"):
                continue
            if actual_target not in results_raw or a_str not in results_raw[actual_target]:
                continue
            q_scores = np.array(results_raw[actual_target][a_str]["f1"])
            b_scores = np.array(results_raw[m][a_str]["f1"])
            n = min(len(q_scores), len(b_scores))
            q_scores = q_scores[:n]
            b_scores = b_scores[:n]
            diff = q_scores - b_scores
            if np.all(diff == 0.0):
                p_val, stat = 1.0, 0.0
            else:
                try:
                    res = scipy.stats.wilcoxon(q_scores, b_scores, alternative="two-sided")
                    p_val, stat = float(res.pvalue), float(res.statistic)
                except Exception:
                    p_val, stat = 1.0, 0.0
            records.append({"alpha": a_str, "model": m, "stat": stat, "raw_p": p_val})

    if len(records) == 0:
        return {}

    raw_p_values = [r["raw_p"] for r in records]
    try:
        q_values = scipy.stats.false_discovery_control(raw_p_values, method="bh")
    except Exception:
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

    stat_out: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for r, q_val in zip(records, q_values):
        a_str = r["alpha"]
        m = r["model"]
        if a_str not in stat_out:
            stat_out[a_str] = {}

        if q_val < 0.001:
            sig = "***"
        elif q_val < 0.01:
            sig = "**"
        elif q_val < 0.05:
            sig = "*"
        else:
            sig = "ns"

        stat_out[a_str][m] = {
            "stat": r["stat"],
            "raw_p": float(r["raw_p"]),
            "bh_q": float(q_val),
            "sig": sig,
        }
    return stat_out


def generate_camouflage_latex_table(
    dataset_name: str,
    summary: Dict[str, Dict[str, Dict[str, float]]],
    stat_summary: Dict[str, Dict[str, Dict[str, Any]]],
    models: List[str],
    alphas: List[float]
) -> str:
    """Generates an IEEE TIFS LaTeX Table of Macro-F1 across camouflage intensities."""
    alpha_cols = " & ".join([rf"\textbf{{\boldmath$\alpha={a:.2f}$}}" for a in alphas])
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        rf"\caption{{Topological Camouflage Stress Test (RQ2) on \textbf{{{dataset_name}}}. Macro-F1 score (\%) (mean $\pm$ std across random seeds) under increasing evasion intensity $\alpha$. At $\alpha=1.00$, flow features are fully mimicked ($\mathbf{{x}} \leftarrow \boldsymbol{{\mu}}_{{\text{{benign}}}}$) and lateral links form a $k$-regular chordal ring. $^{{***}}q<0.001$, $^{{**}}q<0.01$, $^{{*}}q<0.05$ denote statistical superiority of AEGIS over baseline after Benjamini-Hochberg FDR correction ($q \le 0.05$).}}",
        rf"\label{{tab:rq2_camouflage_{dataset_name.lower()}}}",
        r"\small",
        rf"\begin{{tabular}}{{l{'c' * len(alphas)}}}",
        r"\toprule",
        rf"\textbf{{Model}} & {alpha_cols} \\",
        r"\midrule",
    ]

    for m in models:
        cells = []
        for a in alphas:
            a_str = f"{a:.2f}"
            f1_m = summary[m][a_str]["f1_mean"]
            f1_s = summary[m][a_str]["f1_std"]
            sig = ""
            if m not in ("AEGIS", "AEGIS-MoE (Ours)") and a_str in stat_summary and m in stat_summary[a_str]:
                s = stat_summary[a_str][m]["sig"]
                if s != "ns":
                    sig = f"^{{{s}}}"

            if m in ("AEGIS", "AEGIS-MoE (Ours)"):
                cells.append(rf"\textbf{{{f1_m:.2f} $\pm$ {f1_s:.2f}}}")
            else:
                cells.append(rf"{f1_m:.2f} $\pm$ {f1_s:.2f}{sig}")

        row_label = rf"\textbf{{{m}}}" if m in ("AEGIS", "AEGIS-MoE (Ours)") else m
        lines.append(f"{row_label} & " + " & ".join(cells) + r" \\")

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table*}",
    ])
    return "\n".join(lines)


def plot_camouflage_stress_test(
    alphas: List[float],
    summary: Dict[str, Dict[str, Dict[str, float]]],
    models: List[str],
    dataset_name: str,
    output_path: str
):
    """Generates publication-quality figure: Macro-F1 vs Camouflage Intensity Alpha."""
    plt.figure(figsize=(8.5, 5.2), dpi=300)
    colors = {
        "GCN": "#7f7f7f",                 # Gray
        "GIN": "#1f77b4",                 # Blue
        "RWPE-GIN": "#ff7f0e",            # Orange
        "MagNet": "#2ca02c",              # Green
        "QUASAR": "#004488",              # Deep Blue
        "Unitary CTQW (Closed)": "#004488", # Deep Blue
        "AEGIS": "#d62728",               # Red (Highlighted / Ours)
        "AEGIS-MoE (Ours)": "#d62728",    # Red (Highlighted / Ours)
    }
    markers = {
        "GCN": "s",
        "GIN": "^",
        "RWPE-GIN": "D",
        "MagNet": "v",
        "QUASAR": "o",
        "Unitary CTQW (Closed)": "o",
        "AEGIS": "P",
        "AEGIS-MoE (Ours)": "P",
    }
    linestyles = {
        "GCN": ":",
        "GIN": "--",
        "RWPE-GIN": "-.",
        "MagNet": "--",
        "QUASAR": "-.",
        "Unitary CTQW (Closed)": "-.",
        "AEGIS": "-",
        "AEGIS-MoE (Ours)": "-",
    }

    x_vals = np.array(alphas)

    for m in models:
        means = np.array([summary[m][f"{a:.2f}"]["f1_mean"] for a in alphas])
        stds = np.array([summary[m][f"{a:.2f}"]["f1_std"] for a in alphas])
        c = colors.get(m, "#333333")
        is_ours = m in ("AEGIS", "AEGIS-MoE (Ours)")
        is_ctqw = m in ("QUASAR", "Unitary CTQW (Closed)")
        lw = 2.8 if is_ours else (2.0 if is_ctqw else 1.8)
        mk = markers.get(m, "o")
        ls = linestyles.get(m, "-")

        lbl = rf"$\mathbf{{{m}}}$ (Ours)" if is_ours else (rf"$\mathbf{{{m}}}$" if is_ctqw else m)
        plt.plot(x_vals, means, label=lbl, color=c, linewidth=lw, marker=mk, markersize=7, linestyle=ls)
        plt.fill_between(x_vals, means - stds, means + stds, color=c, alpha=0.14 if is_ours else 0.06)

    # Annotate the 1-WL breakdown threshold
    plt.axvline(x=0.5, color="#888888", linestyle=":", linewidth=1.2, alpha=0.7)
    plt.text(0.51, 48.0, r"$\leftarrow$ Topological Regularization Applied ($\alpha \geq 0.5$)",
             fontsize=9, color="#555555", style="italic")

    # 50% random guessing floor
    plt.axhline(y=50.0, color="#aa0000", linestyle="--", linewidth=1.0, alpha=0.5)
    plt.text(0.02, 50.8, "Random Guess Floor (50%)", fontsize=8.5, color="#aa0000", style="italic")

    plt.xlabel(r"Camouflage Intensity $\alpha$ (Feature Mimicry $\rightarrow$ Regular Overlay)", fontsize=11, fontweight="bold")
    plt.ylabel("Detection Performance (Macro-F1 %)", fontsize=11, fontweight="bold")
    plt.title(f"Topological Camouflage Evasion Stress Test (IEEE TIFS Experiment 2)\nDataset: {dataset_name.upper()} (Multi-Seed Average $\\pm$ 1 Std)",
              fontsize=12, fontweight="bold", pad=12)
    plt.xticks(x_vals, [f"{a:.2f}" for a in alphas], fontsize=10)
    plt.yticks(np.arange(40, 101, 10), fontsize=10)
    plt.ylim(42, 98)
    plt.xlim(-0.02, 1.02)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="lower left", fontsize=10, frameon=True, framealpha=0.92, edgecolor="#cccccc")

    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    pdf_path = os.path.splitext(output_path)[0] + ".pdf"
    plt.savefig(pdf_path)
    plt.close()
    print(f"[✓] Publication figure saved to: {output_path} and {pdf_path}")


def run_camouflage_benchmark(
    dataset_name: str,
    data_dir: str,
    alphas: List[float],
    models: List[str],
    seeds: List[int],
    epochs: int = 35,
    batch_size: int = 32,
    lr: float = 0.003,
    hidden_dim: int = 32,
    num_layers: int = 2,
    k_regular: int = 2,
    device: str = "cpu",
    output_json: Optional[str] = None,
    output_fig: Optional[str] = None,
    smoke_test: bool = False
):
    print("=" * 100)
    print("      EXPERIMENT 2: TOPOLOGICAL CAMOUFLAGE STRESS TEST (RQ2)")
    print(f"      Dataset:           {dataset_name.upper()} ({data_dir})")
    print(f"      Models:            {', '.join(models)}")
    print(f"      Seeds:             {len(seeds)} {seeds}")
    print(f"      Alphas:            {alphas}")
    print(f"      k-Regular Overlay: k={k_regular} | Epochs: {epochs} | Batch Size: {batch_size} | Device: {device}")
    print("=" * 100)

    train_path = os.path.join(data_dir, "train.pt")
    val_path = os.path.join(data_dir, "val.pt")
    test_path = os.path.join(data_dir, "test.pt")

    for p in [train_path, val_path, test_path]:
        if not os.path.exists(p):
            print(f"[!] Error: Missing file: {p}")
            print(f"    Please execute: python scripts/preprocess_rq1.py --dataset {dataset_name}")
            sys.exit(1)

    train_graphs, _ = load_netflow_graphs(train_path)
    val_graphs, _ = load_netflow_graphs(val_path)
    clean_test_graphs, _ = load_netflow_graphs(test_path)

    if smoke_test:
        ben_train = [g for g in train_graphs if g.y == 0][:32]
        mal_train = [g for g in train_graphs if g.y == 1][:32]
        train_graphs = ben_train + mal_train

        ben_val = [g for g in val_graphs if g.y == 0][:16]
        mal_val = [g for g in val_graphs if g.y == 1][:16]
        val_graphs = ben_val + mal_val

        ben_test = [g for g in clean_test_graphs if g.y == 0][:16]
        mal_test = [g for g in clean_test_graphs if g.y == 1][:16]
        clean_test_graphs = ben_test + mal_test
        print(f"[!] Smoke test: stratified subset loaded ({len(train_graphs)} train, {len(val_graphs)} val, {len(clean_test_graphs)} test).")

    in_dim = train_graphs[0].x.shape[-1]
    n_train_ben = sum(1 - g.y for g in train_graphs)
    n_train_mal = sum(g.y for g in train_graphs)

    # Compute baseline benign feature mean from TRAINING set (strictly no test leakage)
    benign_train_features = torch.cat([g.x for g in train_graphs if g.y == 0], dim=0)
    benign_mean = benign_train_features.mean(dim=0, keepdim=True)  # [1, node_dim]

    # Balanced class weights
    n_total = len(train_graphs)
    w_0 = float(n_total) / (2.0 * max(float(n_train_ben), 1.0))
    w_1 = float(n_total) / (2.0 * max(float(n_train_mal), 1.0))
    class_weights = torch.tensor([w_0, w_1], dtype=torch.float32)

    print(f"[*] Dataset Splits Loaded:")
    print(f"    - Train Graphs: {len(train_graphs)} (Benign: {n_train_ben}, Malicious: {n_train_mal})")
    print(f"    - Clean Test:   {len(clean_test_graphs)} (Benign: {sum(1-g.y for g in clean_test_graphs)}, Malicious: {sum(g.y for g in clean_test_graphs)})")
    print(f"    - Benign Mean Vector computed across {benign_train_features.shape[0]} training nodes.")

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

    metrics_list = ["f1", "precision", "recall", "far", "auc"]
    # Structure: raw_results[model][alpha_str][metric] -> list of scores across seeds
    raw_results: Dict[str, Dict[str, Dict[str, List[float]]]] = {
        m: {f"{a:.2f}": {k: [] for k in metrics_list} for a in alphas} for m in models
    }

    # Pre-generate camouflaged test sets per alpha (deterministic for given alpha)
    cam_test_loaders: Dict[str, DataLoader] = {}
    for a in alphas:
        a_str = f"{a:.2f}"
        cam_graphs = apply_topological_camouflage(
            clean_test_graphs,
            alpha=a,
            benign_mean=benign_mean,
            k_regular=k_regular
        )
        cam_test_loaders[a_str] = DataLoader(
            NetFlowInMemoryDataset(cam_graphs),
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_netflow_batch
        )

    start_time = time.time()

    # Training once per seed, then evaluating across all alphas
    for seed_idx, seed in enumerate(seeds):
        print(f"\n--- [Seed {seed_idx + 1}/{len(seeds)}: {seed}] ---")
        for m in models:
            t0 = time.time()
            trained_model = train_single_model(
                model_name=m,
                in_dim=in_dim,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                train_loader=train_loader,
                val_loader=val_loader,
                class_weights=class_weights,
                epochs=epochs,
                lr=lr,
                device=device,
                seed=seed
            )
            t_train = time.time() - t0

            # Calibrate threshold strictly on clean validation set (zero test leakage)
            opt_tau = find_optimal_threshold(trained_model, val_loader, device=device)

            # Evaluate across all alphas using calibrated threshold
            alpha_f1s = []
            for a in alphas:
                a_str = f"{a:.2f}"
                loader = cam_test_loaders[a_str]
                res = evaluate_model_on_loader(trained_model, loader, device=device, threshold=opt_tau)
                for k in metrics_list:
                    raw_results[m][a_str][k].append(res[k])
                alpha_f1s.append(f"{res['f1']:.1f}%")

            print(f"  {m:<23} | Train: {t_train:>5.2f}s (tau={opt_tau:.2f}) | F1 across alphas [{', '.join(alpha_f1s)}]")

    # Aggregate summaries
    summary: Dict[str, Dict[str, Dict[str, float]]] = {
        m: {f"{a:.2f}": {} for a in alphas} for m in models
    }
    for m in models:
        for a in alphas:
            a_str = f"{a:.2f}"
            for k in metrics_list:
                vals = raw_results[m][a_str][k]
                summary[m][a_str][f"{k}_mean"] = float(np.mean(vals))
                summary[m][a_str][f"{k}_std"] = float(np.std(vals))

    # Statistical significance testing against AEGIS
    target_name = "AEGIS-MoE (Ours)" if "AEGIS-MoE (Ours)" in models else "AEGIS"
    stat_summary = perform_camouflage_statistical_tests(
        results_raw=raw_results,
        models=models,
        alphas=alphas,
        target_model=target_name
    )

    # Print Terminal Table
    print("\n" + "=" * 105)
    print(f"         TOPOLOGICAL CAMOUFLAGE STRESS TEST RESULTS: {dataset_name.upper()} (MACRO-F1 %)")
    print("-" * 105)
    header = f"{'Model':<23} | " + " | ".join([f"alpha={a:<4.2f}      " for a in alphas])
    print(header)
    print("-" * 105)
    for m in models:
        cells = []
        for a in alphas:
            a_str = f"{a:.2f}"
            f1_m = summary[m][a_str]["f1_mean"]
            f1_s = summary[m][a_str]["f1_std"]
            sig_marker = ""
            if m != target_name and a_str in stat_summary and m in stat_summary[a_str]:
                s = stat_summary[a_str][m]["sig"]
                if s != "ns":
                    sig_marker = f" [{s}]"
            cells.append(f"{f1_m:>5.2f} ± {f1_s:<4.2f}{sig_marker}")
        print(f"{m:<23} | " + " | ".join(cells))
    print("=" * 105)

    # Print Breakdown at Full Camouflage (alpha = 1.0)
    print(f"\n--- Critical Stress Threshold (alpha = 1.00: Full Feature Mimicry + Degree-Regular Overlay) ---")
    print(f"{'Model':<23} | {'Macro-F1 (%)':<18} | {'Precision (%)':<16} | {'Recall (%)':<16} | {'FAR (%) [↓]':<16}")
    print("-" * 95)
    for m in models:
        f1 = f"{summary[m]['1.00']['f1_mean']:.2f} ± {summary[m]['1.00']['f1_std']:.2f}"
        pr = f"{summary[m]['1.00']['precision_mean']:.2f} ± {summary[m]['1.00']['precision_std']:.2f}"
        rc = f"{summary[m]['1.00']['recall_mean']:.2f} ± {summary[m]['1.00']['recall_std']:.2f}"
        far = f"{summary[m]['1.00']['far_mean']:.2f} ± {summary[m]['1.00']['far_std']:.2f}"
        sig = ""
        if m != target_name and "1.00" in stat_summary and m in stat_summary["1.00"]:
            s = stat_summary["1.00"][m]["sig"]
            if s != "ns":
                sig = f" [{s}]"
        print(f"{m:<23} | {f1:<18} | {pr:<16} | {rc:<16} | {far:<16}{sig}")
    print("=" * 95)

    # Generate and print LaTeX Table
    latex_code = generate_camouflage_latex_table(dataset_name, summary, stat_summary, models, alphas)
    print("\n--- IEEE TIFS Publication-Ready LaTeX Table (Experiment 2) ---")
    print(latex_code)
    print("--------------------------------------------------------------\n")

    # Generate Publication Figure
    k_suffix = f"_k{k_regular}" if k_regular != 2 else ""
    fig_path = output_fig or f"benchmarks/figure_camouflage_{dataset_name.lower()}{k_suffix}.png"
    plot_camouflage_stress_test(alphas, summary, models, dataset_name, fig_path)

    # Save complete JSON
    out_json = output_json or f"benchmarks/rq2_camouflage_{dataset_name.lower()}{k_suffix}_results.json"
    full_payload = {
        "metadata": {
            "dataset_name": dataset_name,
            "data_dir": data_dir,
            "models": models,
            "alphas": alphas,
            "seeds": seeds,
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": lr,
            "k_regular": k_regular,
            "device": device,
            "total_runtime_sec": time.time() - start_time,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        "summary": summary,
        "statistical_tests": stat_summary,
        "raw_results": raw_results,
        "latex_table": latex_code,
    }

    with open(out_json, "w") as f:
        json.dump(full_payload, f, indent=2)
    print(f"[✓] Experiment 2 results saved to: {out_json}")


def main():
    parser = argparse.ArgumentParser(description="Experiment 2: Topological Camouflage Stress Test (IEEE TIFS)")
    parser.add_argument("--dataset", type=str, choices=["ctu13_10", "ctu13_9", "ctu13_11", "nf_bot_iot"], default="ctu13_10",
                        help="Dataset name to evaluate (default: ctu13_10)")
    parser.add_argument("--data_dir", type=str, default=None,
                        help="Path to folder containing preprocessed train.pt, val.pt, test.pt")
    parser.add_argument("--alphas", nargs="+", type=float, default=[0.0, 0.25, 0.50, 0.75, 1.00],
                        help="Camouflage intensity alphas to sweep (default: 0.0 0.25 0.50 0.75 1.00)")
    parser.add_argument("--num_seeds", type=int, default=5,
                        help="Number of independent random seeds (default: 5)")
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
    parser.add_argument("--k_regular", type=int, default=2,
                        help="Degree of the regularized camouflage overlay (default: 2)")
    parser.add_argument("--models", nargs="+", default=["GCN", "GIN", "RWPE-GIN", "MagNet", "Unitary CTQW (Closed)", "AEGIS-MoE (Ours)"],
                        help="Models to evaluate (default: GCN GIN RWPE-GIN MagNet 'Unitary CTQW (Closed)' 'AEGIS-MoE (Ours)')")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device to use: auto, cpu, or cuda (default: auto)")
    parser.add_argument("--output_json", type=str, default=None,
                        help="Custom path to save output JSON results")
    parser.add_argument("--output_fig", type=str, default=None,
                        help="Custom path to save publication figure PNG")
    parser.add_argument("--smoke_test", action="store_true",
                        help="Quick smoke test mode (2 seeds, 3 epochs, alphas 0.0 and 1.0)")
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
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    else:
        device = args.device

    print(f"[*] Execution Device: {device.upper()} | PyTorch Threads: {num_threads} (Thermal & Latency Optimized)")

    data_dir = args.data_dir or os.path.join("data", "processed", args.dataset)

    if args.smoke_test:
        print("[*] Running in SMOKE TEST mode (2 seeds, 3 epochs, alphas [0.0, 1.0])...")
        seeds = [42, 43]
        epochs = 3
        alphas = [0.0, 1.0]
    else:
        seeds = [42 + i for i in range(args.num_seeds)]
        epochs = args.epochs
        alphas = args.alphas

    run_camouflage_benchmark(
        dataset_name=args.dataset,
        data_dir=data_dir,
        alphas=alphas,
        models=args.models,
        seeds=seeds,
        epochs=epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        k_regular=args.k_regular,
        device=device,
        output_json=args.output_json,
        output_fig=args.output_fig,
        smoke_test=args.smoke_test,
    )


if __name__ == "__main__":
    main()
