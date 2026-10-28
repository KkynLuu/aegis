#!/usr/bin/env python3
"""
Demo: Quantum Walk Expressivity vs 1-WL.
Demonstrates that QWConv separates C10 and 2xC5 graphs where classical 1-WL fails.
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import numpy as np
from models.qw_conv import QWConv

def get_c10_and_two_c5():
    """Generates adjacency matrices for C10 and 2xC5 (both 10 nodes, all degree 2)."""
    # Graph A: C10
    A_C10 = np.zeros((10, 10))
    for i in range(10):
        A_C10[i, (i + 1) % 10] = 1
        A_C10[i, (i - 1) % 10] = 1

    # Graph B: 2 x C5
    A_2C5 = np.zeros((10, 10))
    for i in range(5):
        A_2C5[i, (i + 1) % 5] = 1
        A_2C5[i, (i - 1) % 5] = 1
    for i in range(5, 10):
        A_2C5[i, 5 + ((i - 5 + 1) % 5)] = 1
        A_2C5[i, 5 + ((i - 5 - 1) % 5)] = 1

    return torch.tensor(A_C10).float(), torch.tensor(A_2C5).float()

def main():
    print("=" * 65)
    print("  QW-GNN EXPRESSIVITY VERIFICATION")
    print("  Evaluating QWConv on 1-WL Equivalent Graphs (C10 vs 2xC5)")
    print("=" * 65)
    
    A_C10, A_2C5 = get_c10_and_two_c5()
    # Batch both graphs: [B=2, N=10, N=10]
    adj_batch = torch.stack([A_C10, A_2C5], dim=0)
    
    # Identical uniform node features (all 1s) - 1-WL condition
    # [B=2, N=10, D=8]
    x_batch = torch.ones(2, 10, 8)
    
    torch.manual_seed(42)
    layer = QWConv(in_dim=8, out_dim=16, num_time_scales=4, init_times=(0.5, 1.5, 3.0, 5.0))
    
    with torch.no_grad():
        out, q_spe = layer(x_batch, adj_batch)
        
    # Graph-level representations via sum pooling
    graph_emb_C10 = out[0].sum(dim=0)
    graph_emb_2C5 = out[1].sum(dim=0)
    
    dist = torch.norm(graph_emb_C10 - graph_emb_2C5).item()
    
    print("\n[1] Classical 1-WL / MPNN Outcome:")
    print("    Degree multiset for C10:  {2: 10}")
    print("    Degree multiset for 2xC5: {2: 10}")
    print("    Classical MPNN graph embedding distance: 0.000000 (Cannot separate)")
    
    print("\n[2] QW-GNN Outcome:")
    print(f"    Graph embedding Euclidean distance: {dist:.6f}")
    print(f"    Q-SPE norm (C10):  {torch.norm(q_spe[0]).item():.4f}")
    print(f"    Q-SPE norm (2xC5): {torch.norm(q_spe[1]).item():.4f}")
    
    print("-" * 65)
    if dist > 1e-4:
        print("[✓] SUCCESS: QW-GNN produces strictly distinct representations")
        print("    for 1-WL equivalent non-isomorphic graphs!")
    else:
        print("[!] Failed to separate.")
    print("=" * 65)

if __name__ == "__main__":
    main()
