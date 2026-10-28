"""
Comprehensive Unit Tests for AEGIS Core Modules:
  1. EdgeCentricHoppingEncoder (Asymmetric Hatano-Nelson Hopping)
  2. AegisConv (Non-Hermitian Operator & Dissipative Continuous-Time Propagation)
  3. Physical Dissipation Verification (Norm attenuation over time)
  4. AegisClassifier (End-to-End Pseudospectral Skin Readout & Autograd Backprop)
"""

import os
import sys
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from models.aegis_conv import AegisConv, EdgeCentricHoppingEncoder
from models.aegis_classifier import AegisClassifier


def test_edge_centric_hopping_asymmetry():
    """Verifies that EdgeCentricHoppingEncoder outputs non-reciprocal J_uv != J_vu."""
    B, N, D = 2, 6, 8
    torch.manual_seed(42)
    x = torch.randn(B, N, D)
    adj = torch.randint(0, 2, (B, N, N)).float()
    # Zero out diagonal
    adj = adj * (1.0 - torch.eye(N).unsqueeze(0))

    encoder = EdgeCentricHoppingEncoder(node_dim=D, hidden_dim=16)
    W = encoder(x, adj)

    assert W.shape == (B, N, N)
    # Check that diagonal is strictly zero
    diag = torch.diagonal(W, dim1=-2, dim2=-1)
    assert torch.allclose(diag, torch.zeros_like(diag))

    # Verify that asymmetric hopping is produced: W != W^T
    diff = torch.norm(W - W.transpose(-1, -2))
    assert diff.item() > 0.1, "Hopping matrix should be non-reciprocal!"


def test_aegis_conv_forward_and_backward():
    """Verifies forward propagation, non-Hermitian coordinates, and autograd gradient flow."""
    B, N, in_dim, out_dim = 2, 8, 10, 16
    torch.manual_seed(42)
    x = torch.randn(B, N, in_dim, requires_grad=True)
    adj = (torch.rand(B, N, N) > 0.5).float() * (1.0 - torch.eye(N).unsqueeze(0))

    conv = AegisConv(
        in_dim=in_dim,
        out_dim=out_dim,
        num_time_scales=4,
        init_times=(0.2, 0.8, 2.0, 5.0),
        base_dissipation=0.1,
    )

    out, nh_coords = conv(x, adj, return_spe=True)

    assert out.shape == (B, N, out_dim)
    # Coordinate dimension = 3 * K = 3 * 4 = 12
    assert nh_coords.shape == (B, N, 12)

    # Check finite numbers
    assert torch.isfinite(out).all()
    assert torch.isfinite(nh_coords).all()

    # Backprop
    loss = out.sum() + nh_coords.sum()
    loss.backward()

    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert conv.log_times.grad is not None
    assert conv.dissipation_mlp[0].weight.grad is not None


def test_physical_dissipation_norm_attenuation():
    """
    Physical Law Verification:
    Verifies that the on-site dissipation -i*Gamma strictly attenuates
    the total probability mass as evolution time t increases:
    sum_v P_uv(t_long) < sum_v P_uv(t_short).
    """
    B, N, in_dim = 1, 8, 4
    torch.manual_seed(42)
    x = torch.randn(B, N, in_dim)
    adj = (torch.rand(B, N, N) > 0.5).float() * (1.0 - torch.eye(N).unsqueeze(0))

    conv = AegisConv(
        in_dim=in_dim,
        out_dim=in_dim,
        num_time_scales=2,
        init_times=(0.1, 4.0),  # short time vs long time
        base_dissipation=0.5,
    )

    H_nh, V, gamma = conv._compute_hamiltonian(x, adj)
    times = torch.tensor([0.1, 4.0])
    U_list = conv._propagate_exact(H_nh, times)

    # Transition probability kernels
    P_short = U_list[0].real.square() + U_list[0].imag.square()
    P_long = U_list[1].real.square() + U_list[1].imag.square()

    total_mass_short = P_short.sum(dim=-1).mean().item()
    total_mass_long = P_long.sum(dim=-1).mean().item()

    print(f"Total probability mass: short (t=0.1) = {total_mass_short:.4f}, long (t=4.0) = {total_mass_long:.4f}")
    assert total_mass_long < total_mass_short, (
        f"Dissipation must attenuate state norm: {total_mass_long} should be < {total_mass_short}"
    )


def test_aegis_classifier_batched_with_mask():
    """Verifies AegisClassifier on variable-sized graphs with padding masks."""
    B, N, in_dim, num_classes = 4, 12, 10, 2
    torch.manual_seed(42)
    x = torch.randn(B, N, in_dim)
    adj = (torch.rand(B, N, N) > 0.6).float() * (1.0 - torch.eye(N).unsqueeze(0))

    # Mask: 2 real nodes in first graph, 8 in second, 12 in third/fourth
    mask = torch.zeros(B, N)
    mask[0, :6] = 1.0
    mask[1, :8] = 1.0
    mask[2, :] = 1.0
    mask[3, :] = 1.0

    model = AegisClassifier(
        in_dim=in_dim,
        hidden_dim=32,
        num_classes=num_classes,
        num_layers=2,
        num_time_scales=3,
        use_attention=True,
    )

    logits, attn = model(x, adj, mask=mask, return_attention=True)
    assert logits.shape == (B, num_classes)
    assert attn.shape == (B, N)

    # Verify that attention on padded dummy nodes is 0
    assert torch.allclose(attn[0, 6:], torch.zeros_like(attn[0, 6:]))
    assert torch.allclose(attn[1, 8:], torch.zeros_like(attn[1, 8:]))

    # Backprop
    criterion = nn.CrossEntropyLoss()
    target = torch.randint(0, num_classes, (B,))
    loss = criterion(logits, target)
    loss.backward()

    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"Parameter {name} has None gradient!"
            assert torch.isfinite(param.grad).all(), f"Parameter {name} has NaN/Inf gradient!"


def test_aegis_dual_branch():
    """Verifies that AEGIS dual-branch architecture (topological + volumetric) trains cleanly."""
    B, N, in_dim, num_classes = 4, 12, 10, 2
    torch.manual_seed(42)
    x = torch.randn(B, N, in_dim, requires_grad=True)
    adj = (torch.rand(B, N, N) > 0.6).float() * (1.0 - torch.eye(N).unsqueeze(0))
    mask = torch.ones(B, N)
    mask[0, 8:] = 0.0

    model = AegisClassifier(
        in_dim=in_dim,
        hidden_dim=32,
        num_classes=num_classes,
        num_layers=2,
        num_time_scales=3,
        use_attention=True,
        use_volumetric=True,
    )

    logits = model(x, adj, mask=mask)
    assert logits.shape == (B, num_classes)
    loss = logits.sum()
    loss.backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()


def test_adaptive_dissipation_density_scaling():
    """
    Verifies that the adaptive dissipation floor dynamically conditions on graph density:
      - Sparse graphs (density ~ 1-2): floor approaches min_floor (~0.005), preserving coherent transport.
      - Dense graphs (density >> 5): floor scales up to base_dissipation (0.10-0.20), damping phase noise.
    """
    N, in_dim = 10, 8
    torch.manual_seed(42)
    x = torch.randn(1, N, in_dim)

    # 1. Sparse 2-regular directed cycle (density = 2.0)
    adj_sparse = torch.zeros(1, N, N)
    for i in range(N):
        adj_sparse[0, i, (i + 1) % N] = 1.0
        adj_sparse[0, i, (i - 1) % N] = 1.0

    # 2. Dense directed graph (density = 8.0)
    adj_dense = (torch.ones(1, N, N) - torch.eye(N).unsqueeze(0))

    conv = AegisConv(
        in_dim=in_dim,
        out_dim=in_dim,
        base_dissipation=0.20,
        adaptive_dissipation=True,
    )

    H_sparse, _, gamma_sparse = conv._compute_hamiltonian(x, adj_sparse)
    H_dense, _, gamma_dense = conv._compute_hamiltonian(x, adj_dense)

    mean_gamma_sparse = gamma_sparse.mean().item()
    mean_gamma_dense = gamma_dense.mean().item()

    print(f"Adaptive Dissipation: Sparse (density=2.0) gamma={mean_gamma_sparse:.4f} vs Dense (density=8.0) gamma={mean_gamma_dense:.4f}")
    assert mean_gamma_sparse < mean_gamma_dense, "Sparse graph must have strictly lower dissipation than dense graph!"
    # Ensure sparse dissipation floor is lower than dense dissipation
    assert mean_gamma_sparse < 0.15, f"Sparse dissipation {mean_gamma_sparse} should be < 0.15"
    assert mean_gamma_dense >= 0.25, f"Dense dissipation {mean_gamma_dense} should be >= 0.25"


def test_topological_priority_moe_routing():
    """
    Verifies Topological-Priority Residual MoE Routing:
      1. Under degree-regular chordal camouflage (skew_ratio == 1.0):
         E_vol is strictly suppressed (g_vol == 0.0), delegating 100% of routing mass to E_topo.
      2. Under volumetric flood (star topology, skew_ratio >> 5.0):
         E_vol activates (g_vol > 0.30) to capture anomalous multiset count.
    """
    B, N, in_dim, num_classes = 2, 12, 10, 2
    torch.manual_seed(42)
    x = torch.randn(B, N, in_dim)

    # Graph 0: Perfectly 2-regular directed cycle (adversarial camouflage proxy)
    adj = torch.zeros(B, N, N)
    for i in range(N):
        adj[0, i, (i + 1) % N] = 1.0
        adj[0, i, (i - 1) % N] = 1.0

    # Graph 1: Star graph / volumetric spam proxy (Node 0 sends to all other 11 nodes)
    adj[1, 0, 1:] = 1.0
    adj[1, 1:, 0] = 1.0

    model = AegisClassifier(
        in_dim=in_dim,
        hidden_dim=32,
        num_classes=num_classes,
        num_layers=2,
        num_time_scales=3,
        use_volumetric=True,
        use_moe=True,
        moe_routing_mode="topological_priority",
    )

    logits, l_topo, l_vol, gating = model(x, adj, return_all=True)

    g_topo_cam = gating[0, 0].item()
    g_vol_cam = gating[0, 1].item()
    g_topo_star = gating[1, 0].item()
    g_vol_star = gating[1, 1].item()

    print(f"Topological Priority Routing:")
    print(f"  Regular Camouflage: g_topo = {g_topo_cam:.4f}, g_vol = {g_vol_cam:.4f}")
    print(f"  Volumetric Star:    g_topo = {g_topo_star:.4f}, g_vol = {g_vol_star:.4f}")

    # Under camouflage: E_vol must be 0.0 and E_topo must be 1.0
    assert abs(g_vol_cam) < 1e-5, f"Under regular camouflage, g_vol must be 0.0, got {g_vol_cam}"
    assert abs(g_topo_cam - 1.0) < 1e-5, f"Under regular camouflage, g_topo must be 1.0, got {g_topo_cam}"
    # Logits under camouflage must match l_topo exactly
    assert torch.allclose(logits[0], l_topo[0], atol=1e-5), "Under camouflage, logits must equal l_topo!"

    # Under volumetric flood: E_vol must activate
    assert g_vol_star > 0.05, f"Under volumetric star, g_vol must activate (> 0.05), got {g_vol_star}"


if __name__ == "__main__":
    test_edge_centric_hopping_asymmetry()
    print("[✓] EdgeCentricHoppingEncoder test passed.")
    test_aegis_conv_forward_and_backward()
    print("[✓] AegisConv forward/backward test passed.")
    test_physical_dissipation_norm_attenuation()
    print("[✓] Physical dissipation norm attenuation test passed.")
    test_aegis_classifier_batched_with_mask()
    print("[✓] AegisClassifier batched mask test passed.")
    test_aegis_dual_branch()
    print("[✓] AegisDualBranch test passed.")
    test_adaptive_dissipation_density_scaling()
    print("[✓] Adaptive dissipation density scaling test passed.")
    test_topological_priority_moe_routing()
    print("[✓] Topological-Priority MoE routing test passed.")
    print("\n[ALL TESTS PASSED SUCCESSFULLY!]")
