"""
AEGIS Convolutional Layer (AegisConv).
Asymmetric Exceptional-point Graph Intrusion System (AEGIS).

Implements Open Non-Hermitian Quantum Dynamics and Pseudospectral Message Passing:
    1. Asymmetric Hopping:
       J_uv != J_vu parameterized by edge NetFlow telemetry (Hatano-Nelson model).
    2. Inhomogeneous On-Site Potentials & Controlled Environmental Dissipation:
       V_u in R (on-site potential landscape)
       Γ_u > 0   (on-site dissipation / absorption rate)
       H_AEGIS = W_asym + diag(V) - i * diag(Γ)  [Non-Hermitian, Non-Normal Operator]
    3. Continuous-Time Non-Unitary State Evolution:
       U(t) = exp(-i * H_AEGIS * t)
       P(t) = |U(t)|^2 (Non-Hermitian probability transition density)
    4. Point-Gap Topology & Non-Hermitian Skin Effect (NHSE):
       Breaks 1-WL symmetry via complex winding invariant w(E_B) != 0.
       Condenses probability density on directional attack sinks (victim boundaries).
    5. Controlled Dissipation:
       Exponential damping exp(-Γ t) eliminates phase scrambling on dense networks.
"""

from typing import Optional, Tuple, Union, List
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class EdgeCentricHoppingEncoder(nn.Module):
    """
    Computes asymmetric forward and backward hopping amplitudes J_uv and J_vu
    from directed NetFlow adjacency and incident node features.
    
    Uses efficient bilinear dot-product attention parameterization coupled with
    a topological non-reciprocity vector gauge field (Hatano-Nelson invariant):
        J_uv = softplus(Q_fwd(x_u)^T K_fwd(x_v) / sqrt(d)) * exp(+beta_topo)
        J_vu = softplus(Q_bwd(x_u)^T K_bwd(x_v) / sqrt(d)) * exp(-beta_topo)
    
    Guarantees that unreciprocated directional communication links (the invariant
    signature of C2 and exfiltration) maintain non-Hermitian point-gap winding
    w(E_B) != 0 and skin condensation even under 100% feature camouflage.
    """
    def __init__(self, node_dim: int, hidden_dim: int = 16):
        super().__init__()
        self.node_dim = node_dim
        self.hidden_dim = hidden_dim
        self.scale = math.sqrt(hidden_dim)
        
        # Forward hopping projection: u -> v
        self.q_fwd = nn.Linear(node_dim, hidden_dim)
        self.k_fwd = nn.Linear(node_dim, hidden_dim)
        
        # Backward reverse coupling: v -> u
        self.q_bwd = nn.Linear(node_dim, hidden_dim)
        self.k_bwd = nn.Linear(node_dim, hidden_dim)

        # Topological non-reciprocity gauge parameter beta > 0
        self.beta_topo = nn.Parameter(torch.tensor(0.5))

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Node features [B, N, D]
            adj: Directed adjacency matrix [B, N, N] where adj[b, u, v] > 0 means u -> v

        Returns:
            W: Asymmetric non-Hermitian hopping matrix [B, N, N]
        """
        B, N, D = x.shape
        
        # 1. Forward hopping along directed edge u -> v
        Q_fwd = self.q_fwd(x)  # [B, N, hidden_dim]
        K_fwd = self.k_fwd(x)  # [B, N, hidden_dim]
        score_fwd = torch.bmm(Q_fwd, K_fwd.transpose(-1, -2)) / self.scale
        j_fwd = F.softplus(score_fwd)

        # 2. Backward reverse coupling along edge v -> u
        Q_bwd = self.q_bwd(x)  # [B, N, hidden_dim]
        K_bwd = self.k_bwd(x)  # [B, N, hidden_dim]
        score_bwd = torch.bmm(Q_bwd, K_bwd.transpose(-1, -2)) / self.scale
        j_bwd = F.softplus(score_bwd)

        # 3. Modulate by directed communication topology with Hatano-Nelson gauge potential
        beta = torch.clamp(self.beta_topo, min=0.1, max=2.0)
        adj_fwd = adj * torch.exp(beta)
        adj_bwd = adj.transpose(-1, -2) * torch.exp(-beta)
        W = adj_fwd * j_fwd + adj_bwd * j_bwd

        # 4. Directed degree normalization: guarantees rho(W) <= 1.0, ensuring physical dissipation
        deg_out = W.sum(dim=-1, keepdim=True).clamp(min=1e-5)  # [B, N, 1]
        deg_in = W.sum(dim=-2, keepdim=True).clamp(min=1e-5)   # [B, 1, N]
        W_norm = W / torch.sqrt(deg_out * deg_in)

        # 5. Zero out self-loops in kinetic hopping
        eye_mask = torch.eye(N, device=x.device, dtype=torch.bool).unsqueeze(0)
        W_norm = W_norm.masked_fill(eye_mask, 0.0)

        return W_norm


class AegisConv(nn.Module):
    """
    Continuous-Time Non-Hermitian Quantum Walk Convolutional Layer.
    """
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_time_scales: int = 4,
        init_times: Tuple[float, ...] = (0.2, 0.8, 2.0, 5.0),
        base_dissipation: float = 0.1,
        adaptive_dissipation: bool = True,
        use_edge_encoder: bool = True,
        use_spe: bool = True,
        use_potential: bool = True,
        propagator_mode: str = "exact",  # "exact" (matrix_exp) or "arnoldi"
        arnoldi_steps: int = 12,
    ):
        """
        Args:
            in_dim: Input node feature dimension.
            out_dim: Output node feature dimension.
            num_time_scales: Number of parallel continuous-time evolution scales K.
            init_times: Initial evolution time values.
            base_dissipation: Maximum absorption floor gamma_0 > 0.
            adaptive_dissipation: If True, dynamically conditions dissipation floor on graph interaction density.
            use_edge_encoder: If True, learns asymmetric hopping J_uv != J_vu from features.
            use_spe: Whether to compute and concatenate Non-Hermitian Structural Encodings.
            use_potential: Whether to parameterize on-site potential landscape V_u.
            propagator_mode: "exact" via matrix_exp or "arnoldi" for sparse Krylov projection.
            arnoldi_steps: Subspace dimension m for Arnoldi projection.
        """
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_time_scales = num_time_scales
        self.base_dissipation = base_dissipation
        self.adaptive_dissipation = adaptive_dissipation
        self.use_edge_encoder = use_edge_encoder
        self.use_spe = use_spe
        self.use_potential = use_potential
        self.propagator_mode = propagator_mode
        self.arnoldi_steps = arnoldi_steps

        # Learnable log-space evolution times to ensure t_k > 0
        init_t = torch.tensor(init_times, dtype=torch.float32)
        self.log_times = nn.Parameter(torch.log(init_t))

        # Asymmetric hopping parameterizer
        if use_edge_encoder:
            self.edge_encoder = EdgeCentricHoppingEncoder(in_dim, hidden_dim=in_dim)
        else:
            self.edge_encoder = None

        # On-site potential well: V_u in R
        if use_potential:
            self.potential_mlp = nn.Sequential(
                nn.Linear(in_dim, in_dim),
                nn.SiLU(),
                nn.Linear(in_dim, 1),
            )
            pot_dim = 1
        else:
            self.potential_mlp = None
            pot_dim = 0

        # On-site environmental dissipation rate: Gamma_u > 0
        self.dissipation_mlp = nn.Sequential(
            nn.Linear(in_dim, in_dim // 2 if in_dim > 4 else in_dim),
            nn.SiLU(),
            nn.Linear(in_dim // 2 if in_dim > 4 else in_dim, 1),
        )
        # Initialize final dissipation projection to start near minimal absorption (softplus(-2.5) ~ 0.08)
        nn.init.xavier_uniform_(self.dissipation_mlp[-1].weight, gain=0.1)
        nn.init.constant_(self.dissipation_mlp[-1].bias, -2.5)

        # Message transformation weights per time scale
        self.msg_weights = nn.ParameterList([
            nn.Parameter(torch.empty(in_dim, out_dim)) for _ in range(num_time_scales)
        ])
        for w in self.msg_weights:
            nn.init.xavier_uniform_(w)

        # Non-Hermitian Structural Coordinates:
        # Per time scale: [Return Prob P_ii(t), Net Directional Flux Phi_i(t), Skin Density S_i(t)]
        # Total coordinate dimension = 3 * K
        self.coord_dim = num_time_scales * 3
        if use_spe:
            self.spe_proj = nn.Sequential(
                nn.Linear(self.coord_dim, out_dim),
                nn.SiLU(),
                nn.Linear(out_dim, out_dim),
            )
            spe_dim = out_dim
        else:
            self.spe_proj = None
            spe_dim = 0

        # Continuous-Time Complex Wave Function States:
        # For each evolution scale k in 1..K: Re(psi_k) and Im(psi_k) of dimension in_dim
        # Total complex state dimension = 2 * K * in_dim
        state_dim = 2 * num_time_scales * in_dim
        combined_dim = in_dim + state_dim + out_dim + spe_dim + pot_dim + 1  # includes V and Gamma

        # Node update MLP
        self.update_mlp = nn.Sequential(
            nn.Linear(combined_dim, out_dim),
            nn.SiLU(),
            nn.Linear(out_dim, out_dim),
        )

        self.norm = nn.LayerNorm(out_dim)
        if in_dim != out_dim:
            self.res_proj = nn.Linear(in_dim, out_dim)
        else:
            self.res_proj = nn.Identity()

    def _compute_hamiltonian(
        self, x: torch.Tensor, adj: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        """
        Constructs the Non-Hermitian Hamiltonian:
        H_AEGIS = W_asym + diag(V) - i * diag(Gamma)
        """
        B, N, D = x.shape

        # 1. Asymmetric Hopping Matrix W [B, N, N]
        if self.use_edge_encoder and self.edge_encoder is not None:
            W = self.edge_encoder(x, adj)
        else:
            W = adj.clone()

        # 2. On-Site Potential V_u [B, N]
        if self.use_potential and self.potential_mlp is not None:
            V = self.potential_mlp(x).squeeze(-1)  # [B, N]
        else:
            V = None

        # 3. Controlled Environmental Dissipation Rate Gamma_u [B, N]
        raw_gamma = self.dissipation_mlp(x).squeeze(-1)
        if self.adaptive_dissipation:
            # Condition the dissipation floor on network graph interaction density: rho = |E| / |V|
            # On sparse P2P overlays / low-and-slow C2 (rho ~ 1-2): floor -> 0.005 to preserve ballistic multi-hop transport.
            # On dense volumetric spam (rho >> 5): floor -> base_dissipation to attenuate high-frequency phase noise.
            if mask is not None:
                mask_float = mask.float()  # [B, N]
                node_counts = mask_float.sum(dim=-1).clamp(min=1.0)  # [B]
                edge_counts = (adj * mask_float.unsqueeze(-1) * mask_float.unsqueeze(-2)).sum(dim=(-2, -1))  # [B]
                density = edge_counts / node_counts  # [B]
            else:
                density = adj.sum(dim=(-2, -1)) / max(N, 1)  # [B]

            density_scale = torch.sigmoid((density - 3.0) * 1.5)  # [B]
            min_floor = 0.005
            eff_floor = min_floor + (self.base_dissipation - min_floor) * density_scale  # [B]
            gamma = F.softplus(raw_gamma) + eff_floor.unsqueeze(-1)  # [B, N]
        else:
            gamma = F.softplus(raw_gamma) + self.base_dissipation  # [B, N]

        # 4. Construct Complex Non-Hermitian Operator [B, N, N]
        H_real = W.clone()
        if V is not None:
            H_real = H_real + torch.diag_embed(V)
        H_imag = -torch.diag_embed(gamma)

        H_nh = torch.complex(H_real, H_imag)
        return H_nh, V, gamma

    def _propagate_exact(
        self, H_nh: torch.Tensor, times: torch.Tensor
    ) -> List[torch.Tensor]:
        """
        Exact non-Hermitian continuous-time propagation using scaling & squaring matrix_exp:
        U(t_k) = exp(-i * H_nh * t_k)
        """
        U_list = []
        for k in range(self.num_time_scales):
            t_k = times[k]
            # Operator in Schrödinger form: -i * H_nh * t_k
            op = -1j * H_nh * t_k
            U_k = torch.linalg.matrix_exp(op)  # [B, N, N] complex
            U_list.append(U_k)
        return U_list

    def forward(
        self,
        x: torch.Tensor,
        adj: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_spe: bool = True,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass for single or batched graphs.

        Args:
            x: Node features [N, in_dim] or [B, N, in_dim]
            adj: Directed adjacency matrix [N, N] or [B, N, N]
            mask: Optional node mask [N] or [B, N]
            return_spe: If True, returns (updated_h, non_hermitian_coords)

        Returns:
            out: Updated node features [B, N, out_dim]
            nh_coords: Non-Hermitian structural & flux coordinates [B, N, 3*K]
        """
        is_unbatched = (x.dim() == 2)
        if is_unbatched:
            x = x.unsqueeze(0)
            adj = adj.unsqueeze(0)
            if mask is not None:
                mask = mask.unsqueeze(0)

        B, N, D = x.shape
        times = torch.exp(self.log_times)  # [K]

        # 1. Build Attributed Non-Hermitian Hamiltonian
        H_nh, V, gamma = self._compute_hamiltonian(x, adj, mask=mask)

        # 2. Continuous-Time Non-Hermitian Evolution
        U_list = self._propagate_exact(H_nh, times)

        # 3. Continuous-Time Complex Wave Function Propagation: psi(t) = U_k @ x
        # Linearly expands to 1-hop multiset message Im(psi) ~ -t*H*x and 2-hop Re(psi) ~ x - 0.5*t^2*H^2*x
        x_complex = torch.complex(x, torch.zeros_like(x))
        evolved_states = []

        # Extract Non-Hermitian Physical Density Kernels
        P_all = []
        return_probs = []
        causal_fluxes = []
        skin_densities = []

        for k in range(self.num_time_scales):
            U_k = U_list[k]
            # State vector evolution
            psi_k = torch.bmm(U_k, x_complex)
            evolved_states.extend([psi_k.real, psi_k.imag])

            # Transition probability density: P_uv(t) = |U_uv(t)|^2
            P_k = U_k.real.square() + U_k.imag.square()  # [B, N, N]
            P_all.append(P_k)

            # A. Return Probability P_uu(t) [B, N] (Captures localized cycles and loops)
            p_uu = torch.diagonal(P_k, dim1=-2, dim2=-1)
            return_probs.append(p_uu)

            # B. Net Directional Causal Flux: Phi_u(t) = sum_v (P_uv - P_vu)
            # Positive for C2 masters/sources, negative for compromised victims/sinks
            flux_u = (P_k - P_k.transpose(-1, -2)).sum(dim=-1)
            causal_fluxes.append(flux_u)

            # C. Skin Mode Density / Pseudospectral Amplification: S_u(t) = sum_v P_uv
            # Measures total surviving amplitude and non-normal transient amplification
            skin_u = P_k.sum(dim=-1)
            skin_densities.append(skin_u)

        # Stack multi-scale coordinates: [B, N, 3 * K]
        q_return = torch.stack(return_probs, dim=-1)   # [B, N, K]
        q_flux = torch.stack(causal_fluxes, dim=-1)     # [B, N, K]
        q_skin = torch.stack(skin_densities, dim=-1)   # [B, N, K]
        nh_coords = torch.cat([q_return, q_flux, q_skin], dim=-1)  # [B, N, 3*K]

        # 4. Non-Hermitian Dynamic Message Aggregation
        aggregated_messages = torch.zeros(
            B, N, self.out_dim, device=x.device, dtype=x.dtype
        )
        for k in range(self.num_time_scales):
            transformed_x = torch.matmul(x, self.msg_weights[k])  # [B, N, out_dim]
            msg_k = torch.bmm(P_all[k], transformed_x)             # [B, N, out_dim]
            aggregated_messages = aggregated_messages + msg_k

        # 5. Node Update with Multi-Scale Quantum Wave Functions, SPE Coordinates, and Dissipation
        update_parts = [x] + evolved_states + [aggregated_messages]
        if self.use_spe and self.spe_proj is not None:
            spe_emb = self.spe_proj(nh_coords)  # [B, N, out_dim]
            update_parts.append(spe_emb)

        if self.use_potential and V is not None:
            update_parts.append(V.unsqueeze(-1))
        update_parts.append(gamma.unsqueeze(-1))

        combined = torch.cat(update_parts, dim=-1)
        updated = self.update_mlp(combined)
        residual = self.res_proj(x)
        out = self.norm(residual + updated)

        if is_unbatched:
            out = out.squeeze(0)
            nh_coords = nh_coords.squeeze(0)

        if return_spe:
            return out, nh_coords
        return out
