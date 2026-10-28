"""
QUASAR Convolutional Layer (QuasarConv).
Implements Continuous-Time Quantum Walk (CTQW) and Magnetic CTQW (M-CTQW) message passing on attributed graphs.

Evolves node state amplitudes under parameterized network Hamiltonians:
    1. Real Symmetric CTQW:
       H(θ) = A_sym + diag(θ)
       U(t) = exp(-i H(θ) t)
       P(t) = |U(t)|^2
    2. Magnetic CTQW (M-CTQW, Aharonov-Bohm Effect):
       H_mag(θ) = A_sym ⊙ exp(i Θ_phase) + diag(θ)
       where Θ_phase = 2π q (A - A^T) breaks time-reversal symmetry to encode directed NetFlow causality.
       U(t) = exp(-i H_mag t)
       P(t) = |U(t)|^2 (Chiral bistochastic transition probability matrix)

Supports both single graph [N, D] and batched graphs [B, N, D].
Off-diagonal spectral interference shatters the 1-WL expressivity barrier.
"""

from typing import Optional, Tuple, Union
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class QuasarConv(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_time_scales: int = 4,
        init_times: Tuple[float, ...] = (0.2, 0.8, 2.0, 5.0),
        use_spe: bool = True,
        use_potential: bool = True,
        use_magnetic: bool = False,
        magnetic_flux: float = 0.25,
        differentiable_potential: bool = True,
    ):
        """
        Args:
            in_dim: Input node feature dimension.
            out_dim: Output node feature dimension.
            num_time_scales: Number of parallel quantum walk evolution time scales K.
            init_times: Tuple of initial time values for learnable time parameters.
            use_spe: Whether to concatenate Quantum Structural Positional Encodings.
            use_potential: Whether to include feature-modulated on-site potential landscape.
            use_magnetic: If True, uses Magnetic CTQW with Complex Hermitian Hamiltonian to preserve directed causality.
            magnetic_flux: Magnetic flux parameter q (e.g. 0.25 for ±π/2 Aharonov-Bohm phase shifts).
            differentiable_potential: If True, uses Hellmann-Feynman path for end-to-end differentiable potential optimization.
        """
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_time_scales = num_time_scales
        self.use_spe = use_spe
        self.use_potential = use_potential
        self.use_magnetic = use_magnetic
        self.magnetic_flux = magnetic_flux
        self.differentiable_potential = differentiable_potential

        # Learnable evolution time parameters log(t) to ensure positivity: t = exp(log_t)
        init_t = torch.tensor(init_times, dtype=torch.float32)
        self.log_times = nn.Parameter(torch.log(init_t))

        # On-site potential generator: theta_i = MLP_pot(x_i)
        if use_potential:
            self.potential_mlp = nn.Sequential(
                nn.Linear(in_dim, in_dim // 2 if in_dim > 4 else in_dim),
                nn.SiLU(),
                nn.Linear(in_dim // 2 if in_dim > 4 else in_dim, 1),
            )
            pot_dim = 1
        else:
            self.potential_mlp = None
            pot_dim = 0

        # Message transformation weights per time scale
        self.msg_weights = nn.ParameterList([
            nn.Parameter(torch.empty(in_dim, out_dim)) for _ in range(num_time_scales)
        ])
        for w in self.msg_weights:
            nn.init.xavier_uniform_(w)

        # Dedicated Q-SPE non-linear encoder
        # If magnetic, includes both return probability P_ii(t) and net directional flux Phi_i(t)
        q_dim = num_time_scales * (2 if use_magnetic else 1)
        if use_spe:
            self.qspe_proj = nn.Sequential(
                nn.Linear(q_dim, out_dim),
                nn.SiLU(),
                nn.Linear(out_dim, out_dim),
            )
            combined_dim = in_dim + out_dim + out_dim + pot_dim
        else:
            self.qspe_proj = None
            combined_dim = in_dim + out_dim + pot_dim

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

    def forward(
        self,
        x: torch.Tensor,
        adj: torch.Tensor,
        return_qspe: bool = True,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass for single or batched graphs.

        Args:
            x: Node features [N, in_dim] or [B, N, in_dim]
            adj: Adjacency matrix [N, N] or [B, N, N] (directed NetFlow)
            return_qspe: If True, returns tuple (out, q_spe)

        Returns:
            out: Updated node features [N, out_dim] or [B, N, out_dim]
            q_spe: (Optional) Return probabilities and directional flux coordinates
        """
        is_unbatched = (x.dim() == 2)
        if is_unbatched:
            x = x.unsqueeze(0)
            adj = adj.unsqueeze(0)

        B, N, D = x.shape
        times = torch.exp(self.log_times)  # [K]

        # 1. Compute on-site potentials theta [B, N] (Rossi et al. PRA 2017)
        adj_sym = 0.5 * (adj + adj.transpose(-1, -2))
        if self.use_potential:
            theta = F.softplus(self.potential_mlp(x).squeeze(-1))  # [B, N]
            diag_theta = torch.diag_embed(theta)                  # [B, N, N]
        else:
            theta = None
            diag_theta = None

        if self.use_magnetic:
            # 2A. Magnetic Continuous-Time Quantum Walk (M-CTQW)
            # Aharonov-Bohm phase matrix: Theta_uv = 2pi * q * (A_uv - A_vu)
            skew_adj = adj - adj.transpose(-1, -2)
            theta_phase = 2.0 * math.pi * self.magnetic_flux * skew_adj

            H_real = adj_sym * torch.cos(theta_phase)
            H_imag = adj_sym * torch.sin(theta_phase)
            if diag_theta is not None:
                H_real = H_real + diag_theta

            H_mag = torch.complex(H_real, H_imag)  # Strictly Complex Hermitian: H = H^dagger
            eigvals, V = torch.linalg.eigh(H_mag)   # Real eigvals [B, N], Complex V [B, N, N]

            # Differentiable Hellmann-Feynman path:
            # Detach eigenvectors to bypass 1/(lambda_i - lambda_j) singularity while keeping
            # energy eigenvalues differentiable with d(lambda_k)/d(V_u) = |v_k(u)|^2
            V_used = V.detach()
            eig_used = eigvals if self.differentiable_potential else eigvals.detach()

            P_all = []
            q_spe_list = []
            flux_list = []

            for k in range(self.num_time_scales):
                t_k = times[k]
                phase = torch.complex(torch.cos(-eig_used * t_k), torch.sin(-eig_used * t_k))
                U_k = torch.bmm(V_used * phase.unsqueeze(1), V_used.mH)  # [B, N, N]

                # Transition probability matrix P_jk = |U_jk|^2 (chiral bistochastic)
                P_k = U_k.real.square() + U_k.imag.square()              # [B, N, N]
                P_all.append(P_k)

                # Return probability P_ii(t)
                q_spe_k = torch.diagonal(P_k, dim1=-2, dim2=-1)          # [B, N]
                q_spe_list.append(q_spe_k)

                # Net directional quantum probability flux: Phi_i(t) = sum_j (P_ij - P_ji)
                flux_k = (P_k - P_k.transpose(-1, -2)).sum(dim=-1)       # [B, N]
                flux_list.append(flux_k)

            q_spe = torch.stack(q_spe_list, dim=-1)  # [B, N, K]
            q_flux = torch.stack(flux_list, dim=-1)  # [B, N, K]
            q_coords = torch.cat([q_spe, q_flux], dim=-1)  # [B, N, 2K]
        else:
            # 2B. Real Symmetric CTQW Baseline
            if diag_theta is not None:
                H = (adj_sym + diag_theta).detach()
            else:
                H = adj_sym.detach()

            # For MPS, compute eigh on CPU to avoid threadgroup memory size limits
            if H.device.type == "mps":
                eigvals, V = torch.linalg.eigh(H.cpu())
                eigvals = eigvals.to(H.device)
                V = V.to(H.device)
            else:
                eigvals, V = torch.linalg.eigh(H)  # [B, N], [B, N, N]
            V_t = V.transpose(-1, -2)          # [B, N, N]

            P_all = []
            q_spe_list = []

            for k in range(self.num_time_scales):
                t_k = times[k]
                cos_phase = torch.cos(eigvals * t_k).unsqueeze(1)  # [B, 1, N]
                sin_phase = torch.sin(eigvals * t_k).unsqueeze(1)  # [B, 1, N]

                Re_U = torch.bmm(V * cos_phase, V_t)  # [B, N, N]
                Im_U = torch.bmm(V * sin_phase, V_t)  # [B, N, N]

                P_k = Re_U.square() + Im_U.square()   # [B, N, N]
                P_all.append(P_k)

                q_spe_k = torch.diagonal(P_k, dim1=-2, dim2=-1)  # [B, N]
                q_spe_list.append(q_spe_k)

            q_spe = torch.stack(q_spe_list, dim=-1)  # [B, N, K]
            q_coords = q_spe

        # 3. Quantum-Gated Message Aggregation
        aggregated_messages = torch.zeros(
            B, N, self.out_dim, device=x.device, dtype=x.dtype
        )
        for k in range(self.num_time_scales):
            transformed_x = torch.matmul(x, self.msg_weights[k])  # [B, N, out_dim]
            msg_k = torch.bmm(P_all[k], transformed_x)             # [B, N, out_dim]
            aggregated_messages = aggregated_messages + msg_k

        # 4. Node Update with Quantum Structural Positional Encodings and On-Site Potential
        update_parts = [x, aggregated_messages]
        if self.use_spe:
            q_emb = self.qspe_proj(q_coords)  # [B, N, out_dim]
            update_parts.append(q_emb)
        if self.use_potential and theta is not None:
            update_parts.append(theta.unsqueeze(-1))
        combined = torch.cat(update_parts, dim=-1)

        updated = self.update_mlp(combined)
        residual = self.res_proj(x)
        out = self.norm(residual + updated)

        if is_unbatched:
            out = out.squeeze(0)
            q_coords = q_coords.squeeze(0)

        if return_qspe:
            return out, q_coords
        return out
