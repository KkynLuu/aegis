"""
QUASAR Graph Classifier (QuasarClassifier).

Deep continuous-time quantum walk architecture for graph-level classification and NIDS.
Combines:
  1. Multi-scale unitary time evolution layers (QuasarConv) supporting:
     - Real Symmetric CTQW
     - Magnetic CTQW (M-CTQW) with Aharonov-Bohm phase modulation
  2. Quantum Structural Positional Encodings (Q-SPE) via return probabilities & directional flux
  3. Residual normalization connections
  4. Quantum-aware multiscale graph pooling (Mean, Sum, Max, and Q-SPE)
  5. Multi-Layer Perceptron (MLP) classification head
"""

from typing import List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

import os
import sys

try:
    from .quasar_conv import QuasarConv
except ImportError:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from models.quasar_conv import QuasarConv


class QuasarClassifier(nn.Module):
    """
    Graph-level Classifier powered by Continuous-Time Quantum Walks (QUASAR)
    and Magnetic Quantum Walks (M-CTQW).
    """

    def __init__(
        self,
        in_dim: int = 1,
        hidden_dim: int = 32,
        num_classes: int = 10,
        num_layers: int = 2,
        num_time_scales: int = 4,
        init_times: Tuple[float, ...] = (0.2, 0.8, 2.5, 8.0),
        dropout: float = 0.1,
        residual: bool = True,
        use_potential: bool = True,
        use_attention: bool = True,
        use_magnetic: bool = False,
        magnetic_flux: float = 0.25,
        differentiable_potential: bool = True,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.dropout = dropout
        self.residual = residual
        self.use_potential = use_potential
        self.use_attention = use_attention
        self.use_magnetic = use_magnetic
        self.magnetic_flux = magnetic_flux
        self.differentiable_potential = differentiable_potential

        # 1. Input Node Feature Encoder
        self.node_encoder = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # 2. QuasarConv Backbone Layers
        self.conv_layers = nn.ModuleList([
            QuasarConv(
                in_dim=hidden_dim,
                out_dim=hidden_dim,
                num_time_scales=num_time_scales,
                init_times=init_times,
                use_potential=use_potential,
                use_magnetic=use_magnetic,
                magnetic_flux=magnetic_flux,
                differentiable_potential=differentiable_potential,
            )
            for _ in range(num_layers)
        ])

        # Layer normalizations over node features
        self.norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers)
        ])

        # 3. Quantum-SPE Gated Anomaly Attention Pooling
        # Attends to localized sub-topologies exhibiting anomalous quantum return dynamics
        q_dim_per_layer = num_time_scales * (2 if use_magnetic else 1)
        total_qspe_dim = num_layers * q_dim_per_layer
        if use_attention:
            self.attn_mlp = nn.Sequential(
                nn.Linear(hidden_dim + total_qspe_dim, hidden_dim // 2),
                nn.SiLU(),
                nn.Linear(hidden_dim // 2, 1),
            )
            pooled_dim = 4 * hidden_dim + 2 * total_qspe_dim
        else:
            self.attn_mlp = None
            pooled_dim = 4 * hidden_dim + total_qspe_dim

        # 4. Classification Head MLP
        self.classifier = nn.Sequential(
            nn.Linear(pooled_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(
        self,
        x: torch.Tensor,
        adj: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_embeddings: bool = False,
        return_attention: bool = False,
        return_qspe: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        """
        Forward pass for batched graphs.

        Args:
            x: Node features [B, N, in_dim]
            adj: Adjacency matrix [B, N, N]
            mask: Optional node mask [B, N] (1 for real node, 0 for padded dummy)
            return_embeddings: If True, returns (logits, node_embeddings_h)

        Returns:
            logits: Class logits [B, num_classes] (or tuple with node embeddings)
        """
        B, N, _ = x.shape
        h = self.node_encoder(x)  # [B, N, hidden_dim]

        if mask is not None:
            mask_float = mask.float().unsqueeze(-1)  # [B, N, 1]
            node_counts = (
                mask.sum(dim=1, keepdim=True).float().clamp(min=1.0)
            )  # [B, 1]
        else:
            mask_float = None
            node_counts = None

        all_q_spe = []
        raw_q_spe_list = []
        for i, (conv, norm) in enumerate(zip(self.conv_layers, self.norms)):
            h_next, q_spe = conv(
                h, adj
            )  # h_next: [B, N, hidden_dim], q_spe: [B, N, q_dim_per_layer]
            if self.residual:
                h = norm(h + h_next)
            else:
                h = norm(h_next)
            h = F.silu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)

            raw_q_spe_list.append(q_spe)
            if mask is not None:
                h = h * mask_float
                q_spe_mean = (q_spe * mask_float).sum(dim=1) / node_counts
                all_q_spe.append(q_spe_mean)
            else:
                all_q_spe.append(q_spe.mean(dim=1))

        # Node-level Q-SPE tensor across all layers [B, N, total_qspe_dim]
        q_node_all = torch.cat(raw_q_spe_list, dim=-1)

        # Standard Multiscale Pooling
        if mask is not None:
            h_mean = (h * mask_float).sum(dim=1) / node_counts
            h_norm_sum = (h * mask_float).sum(dim=1) / (node_counts**0.5)
            h_masked = h.masked_fill(~mask.bool().unsqueeze(-1), -1e9)
            h_max = h_masked.max(dim=1).values
        else:
            h_mean = h.mean(dim=1)  # [B, hidden_dim]
            h_norm_sum = h.sum(dim=1) / (N**0.5)  # Normalized sum pooling
            h_max = h.max(dim=1).values  # [B, hidden_dim]

        q_spe_summary = torch.cat(all_q_spe, dim=-1)  # [B, total_qspe_dim]

        if self.use_attention:
            # Compute Q-SPE Gated Anomaly Attention weights
            attn_in = torch.cat([h, q_node_all], dim=-1)  # [B, N, hidden_dim + total_qspe_dim]
            attn_logits = self.attn_mlp(attn_in).squeeze(-1)  # [B, N]

            if mask is not None:
                attn_logits = attn_logits.masked_fill(~mask.bool(), -1e9)
                attn_weights = F.softmax(attn_logits, dim=-1).unsqueeze(-1)  # [B, N, 1]
                attn_weights = attn_weights * mask_float
                attn_weights = attn_weights / (attn_weights.sum(dim=1, keepdim=True).clamp(min=1e-6))
            else:
                attn_weights = F.softmax(attn_logits, dim=-1).unsqueeze(-1)  # [B, N, 1]

            h_attn = (h * attn_weights).sum(dim=1)  # [B, hidden_dim]
            q_attn = (q_node_all * attn_weights).sum(dim=1)  # [B, total_qspe_dim]

            # Multiscale Graph Representation combining macroscopic topology and localized quantum attention
            graph_repr = torch.cat(
                [h_mean, h_norm_sum, h_attn, h_max, q_spe_summary, q_attn], dim=-1
            )
        else:
            # Baseline Global Pooling (Mean, Norm Sum, Max, and Mean Q-SPE)
            if mask is not None:
                h_raw_sum = (h * mask_float).sum(dim=1)
                attn_weights = mask_float / node_counts.unsqueeze(-1)
            else:
                h_raw_sum = h.sum(dim=1)
                attn_weights = torch.full((B, N, 1), 1.0 / N, device=x.device)
            graph_repr = torch.cat(
                [h_mean, h_norm_sum, h_raw_sum, h_max, q_spe_summary], dim=-1
            )

        # Final logits
        logits = self.classifier(graph_repr)
        returns = [logits]
        if return_embeddings:
            returns.append(h)
        if return_attention:
            returns.append(attn_weights.squeeze(-1))
        if return_qspe:
            returns.append(q_node_all)

        if len(returns) == 1:
            return logits
        return tuple(returns)


if __name__ == "__main__":
    print("Testing QuasarClassifier with both Real Symmetric and Magnetic CTQW...")
    B, N, in_dim, num_classes = 4, 10, 4, 2
    x = torch.randn(B, N, in_dim)
    adj = torch.randint(0, 2, (B, N, N)).float()
    mask = torch.ones(B, N)

    # 1. Real Symmetric Baseline
    model_real = QuasarClassifier(
        in_dim=in_dim, hidden_dim=32, num_classes=num_classes, num_layers=2, use_magnetic=False
    )
    logits_real = model_real(x, adj, mask)
    print(f"Real Symmetric Logits: {logits_real.shape}")
    logits_real.sum().backward()

    # 2. Magnetic CTQW
    model_mag = QuasarClassifier(
        in_dim=in_dim, hidden_dim=32, num_classes=num_classes, num_layers=2, use_magnetic=True, magnetic_flux=0.25
    )
    logits_mag = model_mag(x, adj, mask)
    print(f"Magnetic CTQW Logits: {logits_mag.shape}")
    logits_mag.sum().backward()

    print("[✓] Forward and backward passes verified for both Real and Magnetic CTQW!")
