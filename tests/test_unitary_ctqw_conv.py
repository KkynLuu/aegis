"""
Unit test for QuasarConv and QuasarClassifier:
1. Validates tensor shapes and forward pass (both unbatched and batched).
2. Validates backward pass (gradients through matrix_exp and log_times).
3. Validates the Topological Camouflage separation on C_10 vs 2 x C_5.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from models.quasar_conv import QuasarConv
from models.quasar_classifier import QuasarClassifier


def build_c10_and_2xc5():
    """Builds adjacency matrices for C_10 (10-cycle) and 2 x C_5 (two disjoint 5-cycles)."""
    # 1. C_10 (single 10-node cycle)
    A_c10 = torch.zeros(10, 10)
    for i in range(10):
        A_c10[i, (i + 1) % 10] = 1.0
        A_c10[(i + 1) % 10, i] = 1.0

    # 2. 2 x C_5 (disjoint 5-cycles: 0-4 and 5-9)
    A_2c5 = torch.zeros(10, 10)
    for i in range(5):
        A_2c5[i, (i + 1) % 5] = 1.0
        A_2c5[(i + 1) % 5, i] = 1.0
        A_2c5[i + 5, 5 + (i + 1) % 5] = 1.0
        A_2c5[5 + (i + 1) % 5, i + 5] = 1.0

    return A_c10, A_2c5


def test_quasar_conv_forward_backward():
    print("[*] Running QuasarConv forward and backward test...")
    torch.manual_seed(42)

    in_dim = 8
    out_dim = 16
    N = 10

    conv = QuasarConv(in_dim=in_dim, out_dim=out_dim, num_time_scales=4)
    x = torch.randn(N, in_dim, requires_grad=True)
    A_c10, _ = build_c10_and_2xc5()

    out, q_spe = conv(x, A_c10)
    assert out.shape == (N, out_dim), f"Expected shape ({N}, {out_dim}), got {out.shape}"
    assert q_spe.shape == (N, 4), f"Expected Q-SPE shape ({N}, 4), got {q_spe.shape}"
    print(f"[✓] Unbatched forward pass successful. Output: {out.shape}, Q-SPE: {q_spe.shape}")

    # Test backward pass with non-uniform target
    target = torch.randn_like(out)
    loss = (out * target).sum()
    loss.backward()

    assert x.grad is not None, "Input x gradient is None!"
    assert conv.log_times.grad is not None, "Time parameter log_times gradient is None!"
    print(f"[✓] Backward pass successful. Gradient on log_times: {conv.log_times.grad.norm().item():.4f}")


def test_topological_camouflage_separation():
    print("[*] Running Topological Camouflage test (C_10 vs 2 x C_5)...")
    torch.manual_seed(42)

    in_dim = 4
    out_dim = 8
    conv = QuasarConv(in_dim=in_dim, out_dim=out_dim, num_time_scales=4)

    # Identical uninformative initial features (degree = 2 for both regular graphs)
    x = torch.ones(10, in_dim)

    A_c10, A_2c5 = build_c10_and_2xc5()

    # Forward through QuasarConv
    h_c10, _ = conv(x, A_c10)
    h_2c5, _ = conv(x, A_2c5)

    # Graph-level pooled embeddings (mean pooling)
    g_c10 = h_c10.mean(dim=0)
    g_2c5 = h_2c5.mean(dim=0)

    # Euclidean distance between graph embeddings
    diff_norm = torch.norm(g_c10 - g_2c5).item()
    print(f"[*] Embedding difference ||g(C_10) - g(2 x C_5)|| = {diff_norm:.6f}")

    # In classical 1-WL GNNs (GCN, GIN), diff_norm is mathematically 0.000000!
    assert diff_norm > 1e-4, f"Failed to separate regular graphs! diff_norm = {diff_norm}"
    print(f"[✓] SUCCESS! QUASAR successfully shatters the 1-WL barrier and separates camouflaged graphs.")


def test_quasar_classifier_batched():
    print("[*] Running QuasarClassifier batched test...")
    B, N, in_dim, num_classes = 4, 10, 4, 10
    model = QuasarClassifier(in_dim=in_dim, hidden_dim=32, num_classes=num_classes, num_layers=2)

    x = torch.randn(B, N, in_dim)
    adj = torch.randint(0, 2, (B, N, N)).float()
    adj = torch.triu(adj, diagonal=1)
    adj = adj + adj.transpose(-1, -2)

    logits = model(x, adj)
    assert logits.shape == (B, num_classes), f"Expected shape ({B}, {num_classes}), got {logits.shape}"
    loss = logits.sum()
    loss.backward()
    print(f"[✓] QuasarClassifier batched test successful. Logits: {logits.shape}")


if __name__ == "__main__":
    test_quasar_conv_forward_backward()
    print("-" * 50)
    test_topological_camouflage_separation()
    print("-" * 50)
    test_quasar_classifier_batched()
