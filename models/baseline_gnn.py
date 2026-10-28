"""
Classical Graph Neural Network Baselines: GIN and GCN.

Implements standard Graph Isomorphism Network (GIN, Xu et al., ICLR 2019)
and Graph Convolutional Network (GCN, Kipf & Welling, ICLR 2017)
to empirically demonstrate their fundamental 1-WL expressivity barrier.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union

class GINLayer(nn.Module):
    """
    Graph Isomorphism Network (GIN) Layer.
    h_i^{(l+1)} = MLP( (1 + eps) * h_i^{(l)} + sum_{j in N(i)} h_j^{(l)} )
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
            nn.SiLU()
        )
        
    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        x: [B, N, in_dim]
        adj: [B, N, N]
        """
        B, N, _ = x.shape
        # Neighbor sum: adj @ x
        neighbor_sum = torch.bmm(adj, x) # [B, N, in_dim]
        out = (1.0 + self.eps) * x + neighbor_sum
        
        # Apply MLP per node
        out_flat = out.view(B * N, -1)
        out_flat = self.mlp(out_flat)
        return out_flat.view(B, N, -1)

class GINClassifier(nn.Module):
    """
    Full GIN Graph Classifier with Jumping Knowledge connections.
    """
    def __init__(
        self,
        in_dim: int = 1,
        hidden_dim: int = 32,
        num_classes: int = 10,
        num_layers: int = 3,
        dropout: float = 0.1
    ):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        
        self.layers = nn.ModuleList([
            GINLayer(in_dim if i == 0 else hidden_dim, hidden_dim)
            for i in range(num_layers)
        ])
        
        # Readout from each layer (Jumping Knowledge sum pooling)
        pooled_dim = hidden_dim * num_layers
        self.classifier = nn.Sequential(
            nn.Linear(pooled_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
        
    def forward(
        self,
        x: torch.Tensor,
        adj: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_embeddings: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        h = x
        layer_pools = []
        if mask is not None:
            mask_float = mask.float().unsqueeze(-1)
            h = h * mask_float
        for layer in self.layers:
            h = layer(h, adj)
            if mask is not None:
                h = h * mask_float
            h = F.dropout(h, p=self.dropout, training=self.training)
            if mask is not None:
                layer_pools.append((h * mask_float).sum(dim=1))
            else:
                layer_pools.append(h.sum(dim=1))  # Sum pooling
            
        graph_repr = torch.cat(layer_pools, dim=-1)
        logits = self.classifier(graph_repr)
        if return_embeddings:
            return logits, h
        return logits

class GCNClassifier(nn.Module):
    """
    Standard GCN Graph Classifier.
    """
    def __init__(
        self,
        in_dim: int = 1,
        hidden_dim: int = 32,
        num_classes: int = 10,
        num_layers: int = 3,
        dropout: float = 0.1
    ):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        
        self.weights = nn.ParameterList([
            nn.Parameter(torch.empty(in_dim if i == 0 else hidden_dim, hidden_dim))
            for i in range(num_layers)
        ])
        for w in self.weights:
            nn.init.xavier_uniform_(w)
            
        self.norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers)
        ])
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
        
    def forward(
        self,
        x: torch.Tensor,
        adj: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_embeddings: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        B, N, _ = x.shape
        # Normalized adjacency with self-loops: A_tilde = D^{-1/2} (A + I) D^{-1/2}
        adj_self = adj + torch.eye(N, device=adj.device, dtype=adj.dtype).unsqueeze(0)
        deg = adj_self.sum(dim=-1)
        deg_inv_sqrt = torch.pow(deg.clamp(min=1e-6), -0.5)
        norm_adj = deg_inv_sqrt.unsqueeze(-1) * adj_self * deg_inv_sqrt.unsqueeze(-2)
        
        h = x
        if mask is not None:
            mask_float = mask.float().unsqueeze(-1)
            node_counts = mask.sum(dim=1, keepdim=True).float().clamp(min=1.0)
            h = h * mask_float
        for w, norm in zip(self.weights, self.norms):
            transformed = torch.matmul(h, w)
            h = torch.bmm(norm_adj, transformed)
            h = norm(h)
            h = F.silu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
            if mask is not None:
                h = h * mask_float
            
        if mask is not None:
            h_mean = (h * mask_float).sum(dim=1) / node_counts
            h_sum = (h * mask_float).sum(dim=1) / (node_counts ** 0.5)
        else:
            h_mean = h.mean(dim=1)
            h_sum = h.sum(dim=1)
        graph_repr = torch.cat([h_mean, h_sum], dim=-1)
        logits = self.classifier(graph_repr)
        if return_embeddings:
            return logits, h
        return logits


def compute_rwpe(adj: torch.Tensor, k_steps: int = 16) -> torch.Tensor:
    """
    Computes Random Walk Positional Encodings (RWPE, Dwivedi et al., 2021).
    Calculates landing probabilities on self: (M^k)_{ii} for k in 1..k_steps
    where M = D^{-1} A.

    Args:
        adj: Adjacency matrix [B, N, N]
        k_steps: Number of random walk steps.

    Returns:
        rwpe: Positional encodings [B, N, k_steps]
    """
    B, N, _ = adj.shape
    deg = adj.sum(dim=-1).clamp(min=1.0) # [B, N]
    # M = D^{-1} A
    M = adj / deg.unsqueeze(-1) # [B, N, N]

    M_k = M.clone()
    pe_list = []
    for k in range(k_steps):
        # Extract diagonal return probabilities
        diag_probs = torch.diagonal(M_k, dim1=-2, dim2=-1) # [B, N]
        pe_list.append(diag_probs)
        if k < k_steps - 1:
            M_k = torch.bmm(M_k, M)

    return torch.stack(pe_list, dim=-1) # [B, N, k_steps]


class RWPE_GINClassifier(nn.Module):
    """
    Graph Isomorphism Network augmented with Random Walk Positional Encodings (RWPE-GIN).
    Represents modern beyond-1-WL classical baseline (Dwivedi et al., 2021).
    """
    def __init__(
        self,
        in_dim: int = 1,
        hidden_dim: int = 32,
        num_classes: int = 10,
        num_layers: int = 3,
        k_rwpe: int = 16,
        dropout: float = 0.1
    ):
        super().__init__()
        self.k_rwpe = k_rwpe
        self.rwpe_proj = nn.Sequential(
            nn.Linear(k_rwpe, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.node_proj = nn.Linear(in_dim, hidden_dim)
        
        self.gin = GINClassifier(
            in_dim=hidden_dim * 2,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            num_layers=num_layers,
            dropout=dropout
        )

    def forward(
        self,
        x: torch.Tensor,
        adj: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_embeddings: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        rwpe = compute_rwpe(adj, k_steps=self.k_rwpe) # [B, N, k_rwpe]
        rwpe_emb = self.rwpe_proj(rwpe) # [B, N, hidden_dim]
        node_emb = self.node_proj(x) # [B, N, hidden_dim]
        x_aug = torch.cat([node_emb, rwpe_emb], dim=-1)
        return self.gin(x_aug, adj, mask=mask, return_embeddings=return_embeddings)


class MagNetConv(nn.Module):
    """
    Complex-valued Magnetic Graph Convolutional Layer (Zhang et al., NeurIPS 2021).
    Utilizes the Hermitian Magnetic Laplacian H^(q) = A_s * exp(i * 2 * pi * q * (A - A^T))
    to capture asymmetric directional flow patterns in directed interaction graphs.
    """
    def __init__(self, in_dim: int, out_dim: int, q: float = 0.25):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.q = q
        
        # Complex weights: W = W_real + 1j * W_imag
        self.weight_real = nn.Parameter(torch.empty(in_dim, out_dim))
        self.weight_imag = nn.Parameter(torch.empty(in_dim, out_dim))
        nn.init.xavier_uniform_(self.weight_real)
        nn.init.xavier_uniform_(self.weight_imag)
        
        self.norm_real = nn.LayerNorm(out_dim)
        self.norm_imag = nn.LayerNorm(out_dim)

    def forward(
        self,
        x_real: torch.Tensor,
        x_imag: torch.Tensor,
        adj: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x_real: [B, N, in_dim]
        x_imag: [B, N, in_dim]
        adj: Directed adjacency matrix [B, N, N]
        """
        B, N, _ = adj.shape
        device = adj.device
        
        # 1. Construct Symmetrized Adjacency A_s and Phase Matrix Theta
        adj_t = adj.transpose(-1, -2)
        A_s = 0.5 * (adj + adj_t) # [B, N, N]
        
        # Directed phase difference: Theta = 2 * pi * q * (A - A^T)
        theta = 2.0 * math.pi * self.q * (adj - adj_t) # [B, N, N]
        
        # Add self-loops to symmetrized graph for normalization
        I = torch.eye(N, device=device, dtype=adj.dtype).unsqueeze(0)
        A_s_self = A_s + I
        
        # Complex Hermitian Adjacency: H = A_s * (cos(theta) + 1j * sin(theta)) + I
        H_real = A_s * torch.cos(theta) + I # [B, N, N]
        H_imag = A_s * torch.sin(theta)     # [B, N, N]
        
        # Degree normalization: D^{-1/2} H D^{-1/2}
        deg = A_s_self.sum(dim=-1).clamp(min=1e-6) # [B, N]
        deg_inv_sqrt = torch.pow(deg, -0.5) # [B, N]
        
        norm_scale = deg_inv_sqrt.unsqueeze(-1) * deg_inv_sqrt.unsqueeze(-2) # [B, N, N]
        H_real_norm = H_real * norm_scale
        H_imag_norm = H_imag * norm_scale
        
        # 2. Complex Matrix Multiplication with Learnable Weights
        # Z_trans = (X_real + 1j*X_imag) * (W_real + 1j*W_imag)
        # Real: X_R * W_R - X_I * W_I
        # Imag: X_R * W_I + X_I * W_R
        trans_real = torch.matmul(x_real, self.weight_real) - torch.matmul(x_imag, self.weight_imag)
        trans_imag = torch.matmul(x_real, self.weight_imag) + torch.matmul(x_imag, self.weight_real)
        
        # 3. Message Passing with Hermitian Magnetic Operator: H_norm @ Z_trans
        # Real: H_R @ trans_R - H_I @ trans_I
        # Imag: H_R @ trans_I + H_I @ trans_R
        out_real = torch.bmm(H_real_norm, trans_real) - torch.bmm(H_imag_norm, trans_imag)
        out_imag = torch.bmm(H_real_norm, trans_imag) + torch.bmm(H_imag_norm, trans_real)
        
        # 4. Complex Non-Linearity (SiLU on real and imaginary components)
        out_real = F.silu(self.norm_real(out_real))
        out_imag = F.silu(self.norm_imag(out_imag))
        
        return out_real, out_imag


class MagNetClassifier(nn.Module):
    """
    Full Magnetic Graph Neural Network Classifier (MagNet, Zhang et al., NeurIPS 2021).
    SOTA complex-valued baseline for directed network intrusion graphs.
    """
    def __init__(
        self,
        in_dim: int = 1,
        hidden_dim: int = 32,
        num_classes: int = 10,
        num_layers: int = 2,
        q: float = 0.25,
        dropout: float = 0.1
    ):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        
        # Initial real node projection
        self.node_proj = nn.Linear(in_dim, hidden_dim)
        
        # Stacked MagNet layers
        self.layers = nn.ModuleList([
            MagNetConv(hidden_dim, hidden_dim, q=q)
            for _ in range(num_layers)
        ])
        
        # Readout: combines real and imaginary representations across Mean, Sum, and Max
        # Real (3*hidden_dim) + Imag (3*hidden_dim) = 6*hidden_dim
        pooled_dim = 6 * hidden_dim
        
        self.classifier = nn.Sequential(
            nn.Linear(pooled_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(
        self,
        x: torch.Tensor,
        adj: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        B, N, _ = x.shape
        x_real = self.node_proj(x)
        x_imag = torch.zeros_like(x_real)
        
        if mask is not None:
            mask_float = mask.float().unsqueeze(-1)
            node_counts = mask.sum(dim=1, keepdim=True).float().clamp(min=1.0)
            x_real = x_real * mask_float
        else:
            mask_float = None
            node_counts = None
            
        for layer in self.layers:
            x_real, x_imag = layer(x_real, x_imag, adj)
            x_real = F.dropout(x_real, p=self.dropout, training=self.training)
            x_imag = F.dropout(x_imag, p=self.dropout, training=self.training)
            if mask is not None:
                x_real = x_real * mask_float
                x_imag = x_imag * mask_float
                
        # Multiscale Graph Pooling across Real and Imaginary channels
        if mask is not None:
            r_mean = (x_real * mask_float).sum(dim=1) / node_counts
            r_sum = (x_real * mask_float).sum(dim=1) / (node_counts ** 0.5)
            r_max = x_real.masked_fill(~mask.bool().unsqueeze(-1), -1e9).max(dim=1).values
            
            i_mean = (x_imag * mask_float).sum(dim=1) / node_counts
            i_sum = (x_imag * mask_float).sum(dim=1) / (node_counts ** 0.5)
            i_max = x_imag.masked_fill(~mask.bool().unsqueeze(-1), -1e9).max(dim=1).values
        else:
            r_mean = x_real.mean(dim=1)
            r_sum = x_real.sum(dim=1) / (N ** 0.5)
            r_max = x_real.max(dim=1).values
            
            i_mean = x_imag.mean(dim=1)
            i_sum = x_imag.sum(dim=1) / (N ** 0.5)
            i_max = x_imag.max(dim=1).values
            
        graph_repr = torch.cat([r_mean, r_sum, r_max, i_mean, i_sum, i_max], dim=-1)
        return self.classifier(graph_repr)


if __name__ == "__main__":
    B, N, in_dim, num_classes = 4, 41, 4, 10
    x = torch.randn(B, N, in_dim)
    adj = torch.randint(0, 2, (B, N, N)).float()
    
    gin = GINClassifier(in_dim=in_dim, hidden_dim=32, num_classes=num_classes)
    gcn = GCNClassifier(in_dim=in_dim, hidden_dim=32, num_classes=num_classes)
    rwpe_gin = RWPE_GINClassifier(in_dim=in_dim, hidden_dim=32, num_classes=num_classes, k_rwpe=16)
    magnet = MagNetClassifier(in_dim=in_dim, hidden_dim=32, num_classes=num_classes, q=0.25)
    
    out_gin = gin(x, adj)
    out_gcn = gcn(x, adj)
    out_rwpe = rwpe_gin(x, adj)
    out_magnet = magnet(x, adj)
    print("GIN output shape:     ", out_gin.shape)
    print("GCN output shape:     ", out_gcn.shape)
    print("RWPE-GIN output shape:", out_rwpe.shape)
    print("MagNet output shape:  ", out_magnet.shape)
    print("[✓] All 4 baseline models (1-WL & beyond-1-WL) verified successfully!")

