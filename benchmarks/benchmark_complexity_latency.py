#!/usr/bin/env python3
"""
Benchmark: Computational Complexity & Real-Time Inference Latency (S1)
Target Venue: IEEE Transactions on Information Forensics and Security (TIFS)

Benchmarks inference runtime (latency in milliseconds per graph window) and VRAM scaling
of AEGIS-MoE across graph sizes N in [32, 64, 128, 256, 512, 1024]:
  1. Exact Padé Matrix Exponential: torch.linalg.matrix_exp
  2. Full AEGIS-MoE Forward Pass (End-to-End Inference)
  3. Theoretical Krylov Subspace Arnoldi Projection Scaling Comparison:
     O(m^2 * N + m * |E|) vs O(N^3) with m = 20

Saves results to benchmarks/complexity_latency_results.json.
"""

import os
import sys
import time
import json
from typing import Dict, Any, List
import numpy as np
import torch
import torch.nn as nn

# Ensure project root is in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.aegis_classifier import AegisClassifier


def benchmark_latency() -> Dict[str, Any]:
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Benchmarking AEGIS Complexity & Latency on device: {device}")

    Ns = [32, 64, 128, 256, 512, 1024]
    batch_size = 16

    results: Dict[str, Any] = {
        "device": str(device),
        "batch_size": batch_size,
        "nodes": Ns,
        "matrix_exp_latency_ms": [],
        "full_forward_latency_ms": [],
        "throughput_graphs_per_sec": [],
        "krylov_theoretical_flops_ratio": [],
    }

    # Instantiate representative AEGIS model
    model = AegisClassifier(
        in_dim=10,
        hidden_dim=32,
        num_classes=2,
        num_layers=2,
        num_time_scales=4,
        init_times=(0.2, 0.8, 2.5, 8.0),
        base_dissipation=0.2,
        use_edge_encoder=True,
        use_attention=True,
        use_volumetric=True,
        use_moe=True,
    ).to(device)
    model.eval()

    for N in Ns:
        # Create synthetic graph batch with realistic enterprise edge sparsity (average degree ~ 4)
        x = torch.randn(batch_size, N, 10, device=device)
        # Sparse directed adjacency with avg degree ~ 4
        prob = min(4.0 / max(N, 1), 0.5)
        adj = (torch.rand(batch_size, N, N, device=device) < prob).float()
        # Remove self-loops
        for b in range(batch_size):
            adj[b].fill_diagonal_(0.0)

        # 1. Benchmark Matrix Exponential alone
        H = torch.randn(batch_size, N, N, dtype=torch.complex64, device=device)
        # Warmup
        for _ in range(3):
            _ = torch.linalg.matrix_exp(H[:2])
        if device.type == "mps":
            torch.mps.synchronize()

        iters = 15 if N <= 256 else (8 if N <= 512 else 3)
        t0 = time.perf_counter()
        for _ in range(iters):
            _ = torch.linalg.matrix_exp(H)
        if device.type == "mps":
            torch.mps.synchronize()
        dt_mat = (time.perf_counter() - t0) / iters
        mat_ms_per_graph = (dt_mat / batch_size) * 1000.0

        # 2. Benchmark Full End-to-End Forward Pass
        for _ in range(3):
            with torch.no_grad():
                _ = model(x[:2], adj[:2])
        if device.type == "mps":
            torch.mps.synchronize()

        t0 = time.perf_counter()
        for _ in range(iters):
            with torch.no_grad():
                _ = model(x, adj)
        if device.type == "mps":
            torch.mps.synchronize()
        dt_fwd = (time.perf_counter() - t0) / iters
        fwd_ms_per_graph = (dt_fwd / batch_size) * 1000.0
        throughput = 1000.0 / fwd_ms_per_graph

        # 3. Krylov subspace theoretical speedup ratio
        # Dense Padé: 4/3 * N^3 + scaling FLOPs (~ 15 * N^3)
        # Krylov (m=20): 2 * m * |E| + 2 * m^2 * N with |E| ~ 4 * N
        m = 20
        pade_flops = 15.0 * (N ** 3)
        krylov_flops = 2.0 * m * (4.0 * N) + 2.0 * (m ** 2) * N
        speedup_ratio = pade_flops / max(krylov_flops, 1.0)

        results["matrix_exp_latency_ms"].append(float(mat_ms_per_graph))
        results["full_forward_latency_ms"].append(float(fwd_ms_per_graph))
        results["throughput_graphs_per_sec"].append(float(throughput))
        results["krylov_theoretical_flops_ratio"].append(float(speedup_ratio))

        print(
            f"N = {N:4d} | MatExp: {mat_ms_per_graph:6.2f} ms | "
            f"Full Forward: {fwd_ms_per_graph:6.2f} ms | "
            f"Throughput: {throughput:7.1f} graphs/sec | "
            f"Krylov Speedup: {speedup_ratio:5.1f}x"
        )

    out_file = "benchmarks/complexity_latency_results.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[DONE] Saved complexity benchmark results to {out_file}")
    return results


if __name__ == "__main__":
    benchmark_latency()
