"""
AEGIS Graph Classifier (AegisClassifier).
Asymmetric Exceptional-point Graph Intrusion System (AEGIS).

Deep Non-Hermitian Quantum Walk Architecture for Evasion-Resilient Network Intrusion Detection:
    1. Edge-Centric Asymmetric Hopping & Microscopic Non-Reciprocity (Hatano-Nelson Model).
    2. Inhomogeneous Potential Landscapes & Controlled Environmental Dissipation.
    3. Non-Hermitian Continuous-Time Propagation:
       U(t) = exp(-i * H_AEGIS * t)
    4. Topological Point-Gap Symmetry Shattering & Non-Hermitian Skin Effect (NHSE).
    5. Pseudospectral & Skin-Boundary Gated Anomaly Readout Pooling.
    6. Fully Autograd-Differentiable End-to-End Spectral Optimization.
"""

from typing import List, Optional, Tuple, Union
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .aegis_conv import AegisConv
except ImportError:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from models.aegis_conv import AegisConv


class DirectedMultisetVolumetricBlock(nn.Module):
    """
    Directed Multiset Volumetric Feature Accumulator.
    Computes unnormalized directed fan-out and fan-in multiset messages (Xu et al. GIN formulation for directed graphs):
        h_u^{(l+1)} = MLP( (1 + eps) * h_u^{(l)} + A @ h^{(l)} + A^T @ h^{(l)} )
    Preserves exact volumetric multiplicity and degree skew required for
    volumetric spam/DoS anomaly detection (CTU-13 Scenario 9) without spectral degree normalization.
    """
    def __init__(self, in_dim: int, out_dim: int, learn_eps: bool = True):
        super().__init__()
        self.learn_eps = learn_eps
        if learn_eps:
            self.eps = nn.Parameter(torch.zeros(1))
        else:
            self.register_buffer("eps", torch.zeros(1))

        self.mlp = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.SiLU(),
            nn.Linear(out_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.SiLU(),
        )

    def forward(self, v: torch.Tensor, adj: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, N, D = v.shape
        fwd = torch.bmm(adj, v)
        bwd = torch.bmm(adj.transpose(-1, -2), v)
        agg = (1.0 + self.eps) * v + fwd + bwd
        out = self.mlp(agg.view(B * N, D)).view(B, N, -1)
        if mask is not None:
            out = out * mask.unsqueeze(-1)
        return out


class AegisClassifier(nn.Module):
    """
    Graph-level Intrusion Detection Classifier powered by Open Non-Hermitian
    Quantum Dynamics and Pseudospectral Skin Readout (AEGIS).
    Supports Dual-Branch Cross-Regime Gated Readout (Topological Phase Transport + Volumetric Multiset Counting).
    """

    def __init__(
        self,
        in_dim: int = 10,
        hidden_dim: int = 32,
        num_classes: int = 2,
        num_layers: int = 2,
        num_time_scales: int = 4,
        init_times: Tuple[float, ...] = (0.2, 0.8, 2.5, 8.0),
        base_dissipation: float = 0.1,
        adaptive_dissipation: bool = True,
        dropout: float = 0.1,
        residual: bool = True,
        use_edge_encoder: bool = True,
        use_potential: bool = True,
        use_attention: bool = True,
        use_volumetric: bool = True,
        use_moe: bool = True,
        moe_routing_mode: str = "topological_priority",
    ):
        """
        Args:
            in_dim: Input node feature dimension (e.g. 10 for NetFlow statistics).
            hidden_dim: Hidden representation dimension.
            num_classes: Output class count (2 for binary NIDS detection).
            num_layers: Number of AegisConv backbone layers.
            num_time_scales: Number of evolution scales K.
            init_times: Initial continuous-time evolution parameters.
            base_dissipation: Maximum absorption floor gamma_0 > 0.
            adaptive_dissipation: If True, dynamically conditions dissipation floor on graph interaction density.
            dropout: Dropout probability.
            residual: Whether to use residual skip-connections.
            use_edge_encoder: If True, parameterizes asymmetric hopping from edge connectivity.
            use_potential: If True, learns on-site potential landscape V_u.
            use_attention: If True, uses Pseudospectral & Skin-Boundary Gated Attention Pooling.
            use_volumetric: If True, integrates parallel directed multiset volumetric branch.
            use_moe: If True, uses Adaptive Mixture of Experts router between topological & volumetric experts.
            moe_routing_mode: "topological_priority" (anchors E_topo and gates E_vol by skew) or "softmax".
        """
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.dropout = dropout
        self.residual = residual
        self.use_attention = use_attention
        self.num_time_scales = num_time_scales
        self.base_dissipation = base_dissipation
        self.adaptive_dissipation = adaptive_dissipation
        self.use_moe = use_moe and use_volumetric
        self.moe_routing_mode = moe_routing_mode

        # 1. Input Node Feature Normalization & Encoder
        self.in_norm = nn.LayerNorm(in_dim)
        self.node_encoder = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # 2. AegisConv Backbone Layers
        self.conv_layers = nn.ModuleList([
            AegisConv(
                in_dim=hidden_dim,
                out_dim=hidden_dim,
                num_time_scales=num_time_scales,
                init_times=init_times,
                base_dissipation=base_dissipation,
                adaptive_dissipation=adaptive_dissipation,
                use_edge_encoder=use_edge_encoder,
                use_spe=True,
                use_potential=use_potential,
                propagator_mode="exact",
            )
            for _ in range(num_layers)
        ])

        # Layer normalizations over node features
        self.norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers)
        ])

        # 3. Pseudospectral & Skin-Boundary Gated Anomaly Readout Pooling
        self.raw_coords_per_layer = num_time_scales * 3
        self.summary_coords_per_layer = num_time_scales * 7
        self.total_raw_coord_dim = num_layers * self.raw_coords_per_layer
        self.total_summary_coord_dim = num_layers * self.summary_coords_per_layer

        if use_attention:
            self.skin_attn_mlp = nn.Sequential(
                nn.Linear(hidden_dim + self.total_raw_coord_dim, hidden_dim // 2),
                nn.SiLU(),
                nn.Linear(hidden_dim // 2, 1),
            )
            # 5 * hidden_dim: h_mean, h_norm_sum, h_sum (multiset counting), h_max, h_skin
            self.topo_dim = 5 * hidden_dim + self.total_summary_coord_dim + self.total_raw_coord_dim
        else:
            self.skin_attn_mlp = None
            self.topo_dim = 4 * hidden_dim + self.total_summary_coord_dim

        # Expert 1 Head (Non-Hermitian Topological Operator)
        self.head_topo = nn.Sequential(
            nn.Linear(self.topo_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

        # 4. Volumetric Branch & Cross-Regime Integration
        self.use_volumetric = use_volumetric
        if use_volumetric:
            self.vol_blocks = nn.ModuleList([
                DirectedMultisetVolumetricBlock(in_dim if i == 0 else hidden_dim, hidden_dim)
                for i in range(num_layers)
            ])
            self.vol_dim = num_layers * hidden_dim

            # Expert 2 Head (Directed Multiset Volumetric Aggregator)
            self.head_vol = nn.Sequential(
                nn.Linear(self.vol_dim, hidden_dim * 2),
                nn.LayerNorm(hidden_dim * 2),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, num_classes),
            )

            # Adaptive Mixture of Experts Router
            if self.use_moe:
                self.struct_dim = 6 + 2 * in_dim
                self.topo_router_proj = nn.Linear(self.topo_dim, hidden_dim // 2)
                self.vol_router_proj = nn.Linear(self.vol_dim, hidden_dim // 2)
                router_in = self.struct_dim + hidden_dim
                self.router = nn.Sequential(
                    nn.Linear(router_in, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.SiLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, 2),  # [g_topo, g_vol]
                )
            else:
                self.router = None

            fused_dim = self.topo_dim + self.vol_dim
        else:
            self.vol_blocks = None
            self.head_vol = None
            self.router = None
            fused_dim = self.topo_dim

        # Fallback Classifier Head (only used when not self.use_moe)
        if not self.use_moe:
            self.classifier = nn.Sequential(
                nn.Linear(fused_dim, hidden_dim * 2),
                nn.LayerNorm(hidden_dim * 2),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, num_classes),
            )
        else:
            self.classifier = None

    def extract_structural_features(
        self,
        x: torch.Tensor,
        adj: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Extracts graph-level topological descriptors and degree-skew invariants for the MoE Router."""
        B, N, _ = x.shape
        if mask is not None:
            mask_float = mask.float()
            node_counts = mask_float.sum(dim=1).clamp(min=1.0)
            adj_masked = adj * mask_float.unsqueeze(-1) * mask_float.unsqueeze(-2)
        else:
            mask_float = torch.ones((B, N), device=x.device)
            node_counts = torch.full((B,), float(N), device=x.device)
            adj_masked = adj

        d_out = adj_masked.sum(dim=-1)  # [B, N]
        total_edges = d_out.sum(dim=-1)  # [B]
        mean_deg = total_edges / node_counts

        if mask is not None:
            d_out_masked = d_out.masked_fill(~mask.bool(), -1e9)
            max_deg = d_out_masked.max(dim=-1).values.clamp(min=0.0)
            var_deg = ((d_out - mean_deg.unsqueeze(-1))**2 * mask_float).sum(dim=-1) / node_counts
        else:
            max_deg = d_out.max(dim=-1).values
            var_deg = ((d_out - mean_deg.unsqueeze(-1))**2).mean(dim=-1)

        skew_ratio = max_deg / (mean_deg + 1e-4)
        log_n = torch.log1p(node_counts)
        log_e = torch.log1p(total_edges)
        struct_stats = torch.stack([log_n, log_e, mean_deg, max_deg, var_deg, skew_ratio], dim=-1)

        if mask is not None:
            mean_x = (x * mask_float.unsqueeze(-1)).sum(dim=1) / node_counts.unsqueeze(-1)
            max_x = x.masked_fill(~mask.bool().unsqueeze(-1), -1e9).max(dim=1).values
        else:
            mean_x = x.mean(dim=1)
            max_x = x.max(dim=1).values

        return torch.cat([struct_stats, mean_x, max_x], dim=-1)

    def forward(
        self,
        x: torch.Tensor,
        adj: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_embeddings: bool = False,
        return_attention: bool = False,
        return_coords: bool = False,
        return_all: bool = False,
        return_gating: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        """
        Forward pass for batched directed graphs.

        Args:
            x: Node features [B, N, in_dim]
            adj: Directed adjacency matrix [B, N, N]
            mask: Optional boolean or float mask [B, N] (1 for real node, 0 for padded dummy)
            return_embeddings: If True, returns (logits, node_embeddings_h)
            return_attention: If True, returns (logits, skin_attention_weights)
            return_coords: If True, returns (logits, non_hermitian_coordinates)

        Returns:
            logits: Class logits [B, num_classes] (or tuple with metadata)
        """
        B, N, _ = x.shape
        x_norm = self.in_norm(x)
        h = self.node_encoder(x_norm)  # [B, N, hidden_dim]

        if mask is not None:
            mask_float = mask.float().unsqueeze(-1)  # [B, N, 1]
            node_counts = mask.sum(dim=1, keepdim=True).float().clamp(min=1.0)  # [B, 1]
            mask_bool_unsq = mask.bool().unsqueeze(-1)
            h = h * mask_float
        else:
            mask_float = None
            node_counts = None
            mask_bool_unsq = None

        all_q_summary = []
        raw_q_list = []
        K = self.num_time_scales

        # Sequential Non-Hermitian propagation across layers
        for conv, norm in zip(self.conv_layers, self.norms):
            h_next, q_coords = conv(h, adj, mask=mask, return_spe=True)

            if self.residual:
                h = norm(h + h_next)
            else:
                h = norm(h_next)

            h = F.silu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)

            raw_q_list.append(q_coords)

            # Decompose multi-scale coordinates: [B, N, 3*K]
            q_ret = q_coords[:, :, :K]
            q_flux = q_coords[:, :, K:2*K]
            q_skin = q_coords[:, :, 2*K:]

            if mask is not None:
                h = h * mask_float
                q_ret_mean = (q_ret * mask_float).sum(dim=1) / node_counts
                q_ret_max = q_ret.masked_fill(~mask_bool_unsq, -1e9).max(dim=1).values
                q_flux_max = q_flux.masked_fill(~mask_bool_unsq, -1e9).max(dim=1).values
                q_flux_min = q_flux.masked_fill(~mask_bool_unsq, 1e9).min(dim=1).values
                q_flux_abs = (q_flux.abs() * mask_float).sum(dim=1) / node_counts
                q_skin_max = q_skin.masked_fill(~mask_bool_unsq, -1e9).max(dim=1).values
                q_skin_mean = (q_skin * mask_float).sum(dim=1) / node_counts
            else:
                q_ret_mean = q_ret.mean(dim=1)
                q_ret_max = q_ret.max(dim=1).values
                q_flux_max = q_flux.max(dim=1).values
                q_flux_min = q_flux.min(dim=1).values
                q_flux_abs = q_flux.abs().mean(dim=1)
                q_skin_max = q_skin.max(dim=1).values
                q_skin_mean = q_skin.mean(dim=1)

            q_layer_sum = torch.cat(
                [q_ret_mean, q_ret_max, q_flux_max, q_flux_min, q_flux_abs, q_skin_max, q_skin_mean], dim=-1
            )
            all_q_summary.append(q_layer_sum)

        # Full Non-Hermitian structural & flux coordinates across all layers
        q_node_all = torch.cat(raw_q_list, dim=-1)           # [B, N, total_raw_coord_dim]
        q_summary = torch.cat(all_q_summary, dim=-1)         # [B, total_summary_coord_dim]

        # Multi-Granular Topological Pooling
        if mask is not None:
            h_sum = (h * mask_float).sum(dim=1)
            h_mean = h_sum / node_counts
            h_norm_sum = h_sum / (node_counts**0.5)
            h_masked = h.masked_fill(~mask_bool_unsq, -1e9)
            h_max = h_masked.max(dim=1).values
        else:
            h_sum = h.sum(dim=1)
            h_mean = h.mean(dim=1)
            h_norm_sum = h.sum(dim=1) / (N**0.5)
            h_max = h.max(dim=1).values

        if self.use_attention:
            # Pseudospectral & Skin-Boundary Gated Attention Pooling
            attn_in = torch.cat([h, q_node_all], dim=-1)  # [B, N, hidden_dim + total_raw_coord_dim]
            attn_logits = self.skin_attn_mlp(attn_in).squeeze(-1)  # [B, N]

            if mask is not None:
                attn_logits = attn_logits.masked_fill(~mask.bool(), -1e9)
                attn_weights = F.softmax(attn_logits, dim=-1).unsqueeze(-1)  # [B, N, 1]
                attn_weights = attn_weights * mask_float
                attn_weights = attn_weights / (attn_weights.sum(dim=1, keepdim=True).clamp(min=1e-6))
            else:
                attn_weights = F.softmax(attn_logits, dim=-1).unsqueeze(-1)  # [B, N, 1]

            h_skin = (h * attn_weights).sum(dim=1)            # [B, hidden_dim]
            q_skin = (q_node_all * attn_weights).sum(dim=1)    # [B, total_raw_coord_dim]

            # Fused Topological Graph Representation: [h_mean, h_norm_sum, h_sum, h_max, h_skin, q_summary, q_skin]
            graph_topo = torch.cat(
                [h_mean, h_norm_sum, h_sum, h_max, h_skin, q_summary, q_skin], dim=-1
            )
        else:
            graph_topo = torch.cat(
                [h_mean, h_norm_sum, h_sum, h_max, q_summary], dim=-1
            )
            attn_weights = mask_float / node_counts.unsqueeze(-1) if mask is not None else torch.full((B, N, 1), 1.0 / N, device=x.device)

        # Cross-Regime Mixture of Experts Readout
        logits_topo = self.head_topo(graph_topo)

        if self.use_volumetric:
            v = x
            if mask is not None:
                v = v * mask_float
            vol_readouts = []
            for block in self.vol_blocks:
                v = block(v, adj, mask=mask)
                if mask is not None:
                    v_sum = (v * mask_float).sum(dim=1)
                else:
                    v_sum = v.sum(dim=1)
                vol_readouts.append(v_sum)
            graph_vol = torch.cat(vol_readouts, dim=-1)
            logits_vol = self.head_vol(graph_vol)

            if self.use_moe:
                struct_feats = self.extract_structural_features(x, adj, mask)
                topo_proj = F.silu(self.topo_router_proj(graph_topo))
                vol_proj = F.silu(self.vol_router_proj(graph_vol))
                router_in = torch.cat([struct_feats, topo_proj, vol_proj], dim=-1)

                router_logits = self.router(router_in)

                if self.moe_routing_mode == "topological_priority":
                    # Topological-Priority Smooth Residual Routing:
                    # 1. Degree-Skew Gating (Graph-Theoretic Invariant):
                    #    struct_feats[:, 5:6] is skew_ratio = max(d_out) / (mean_deg + eps)
                    #    On regular graphs (e.g. 2-regular chordal cycles): skew_ratio -> 1.0, so skew_gate == 0.0 exactly.
                    #    On star graphs / dense floods: skew_ratio >> 5.0, so skew_gate -> 1.0 fully.
                    skew_ratio = struct_feats[:, 5:6]
                    skew_gate = torch.relu(torch.tanh((skew_ratio - 1.0) / 2.0))  # [B, 1] in [0.0, 1.0]

                    # 2. Topological Benign Authority:
                    #    E_topo operates on the non-Hermitian Hamiltonian H = W + V - i*Gamma.
                    #    When traffic is benign, spectral winding and skin condensation are absent (p_topo[0] >= 0.58).
                    #    Under this topological certificate, classical volumetric false alarms are quenched.
                    p_topo = F.softmax(logits_topo, dim=-1)
                    p_vol = F.softmax(logits_vol, dim=-1)
                    p_topo_benign = p_topo[:, 0:1]
                    p_topo_attack = p_topo[:, 1:2]
                    p_vol_attack = p_vol[:, 1:2]

                    # Smooth sigmoidal veto: ~1.0 when p_topo[0] > 0.58, ~0.0 when uncertain/attack
                    topo_benign_veto = torch.sigmoid(15.0 * (p_topo_benign - 0.58))

                    # 3. Evasion-Resilient Confidence Gating:
                    #    When an adversary disguises flow statistics (camouflage), E_vol is deceived (p_vol[1] << 0.5)
                    #    while E_topo detects point-gap topological winding (p_topo[1] >> 0.5).
                    #    Under such evasion attempts (p_topo[1] > p_vol[1] + 0.08), trust in E_vol is quenched.
                    evasion_gap = p_topo_attack - p_vol_attack
                    evasion_quench = torch.sigmoid(15.0 * (evasion_gap - 0.08))

                    # 4. Dynamic Volumetric Trust Factor:
                    #    E_vol is trusted ONLY when:
                    #    (a) The graph topology is non-regular (skew_gate > 0.0), AND
                    #    (b) E_topo does NOT certify the traffic as benign (topo_benign_veto ~ 0.0), AND
                    #    (c) The traffic is NOT an adversarial camouflage evasion (evasion_quench ~ 0.0).
                    vol_trust = skew_gate * (1.0 - topo_benign_veto) * (1.0 - evasion_quench)  # [B, 1] in [0.0, 1.0]

                    # 5. Gating Weight Allocation:
                    #    - On dense volumetric floods (Scenario 9): skew_gate -> 1.0, topo_benign_veto ~ 0,
                    #      evasion_quench ~ 0, so vol_trust -> 1.0, allowing g_vol to expand up to 0.75.
                    #    - On regular camouflage: skew_gate == 0.0 or evasion_quench -> 1.0, forcing g_vol == 0.0 (g_topo == 1.0).
                    #    - On benign traffic: topo_benign_veto -> 1.0, forcing g_vol -> 0.0 (g_topo -> 1.0),
                    #      eliminating classical volumetric false alarms.
                    raw_g_vol = torch.sigmoid(router_logits[:, 1:2])
                    max_vol_capacity = 0.75
                    g_vol = max_vol_capacity * raw_g_vol * vol_trust  # [B, 1] in [0.0, 0.75]
                    g_topo = 1.0 - g_vol  # [B, 1] in [0.25, 1.00]

                    logits = g_topo * logits_topo + g_vol * logits_vol
                    gating_weights = torch.cat([g_topo, g_vol], dim=-1)
                else:
                    gating_weights = F.softmax(router_logits, dim=-1)  # [B, 2]: [g_topo, g_vol]
                    g_topo = gating_weights[:, 0:1]
                    g_vol = gating_weights[:, 1:2]
                    logits = g_topo * logits_topo + g_vol * logits_vol
            else:
                fused_repr = torch.cat([graph_topo, graph_vol], dim=-1)
                logits = self.classifier(fused_repr)
                gating_weights = torch.full((B, 2), 0.5, device=x.device)
        else:
            logits = logits_topo
            logits_vol = logits_topo
            gating_weights = torch.zeros((B, 2), device=x.device)
            gating_weights[:, 0] = 1.0

        if return_all:
            return logits, logits_topo, logits_vol, gating_weights

        if return_gating:
            return logits, gating_weights

        returns = [logits]
        if return_embeddings:
            returns.append(h)
        if return_attention:
            returns.append(attn_weights.squeeze(-1))
        if return_coords:
            returns.append(q_node_all)

        if len(returns) == 1:
            return logits
        return tuple(returns)


if __name__ == "__main__":
    print("Testing AegisClassifier with Non-Hermitian Quantum Dynamics...")
    B, N, in_dim, num_classes = 4, 16, 10, 2
    x = torch.randn(B, N, in_dim)
    # Directed random adjacency
    adj = (torch.rand(B, N, N) > 0.7).float()
    mask = torch.ones(B, N)

    model = AegisClassifier(
        in_dim=in_dim,
        hidden_dim=32,
        num_classes=num_classes,
        num_layers=2,
        num_time_scales=4,
        base_dissipation=0.1,
    )

    logits, attn = model(x, adj, mask=mask, return_attention=True)
    print(f"Logits shape: {logits.shape}")
    print(f"Skin Attention shape: {attn.shape}")

    loss = logits.sum()
    loss.backward()

    print("[✓] Forward pass and backward autograd gradient flow successfully verified!")
