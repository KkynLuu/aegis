# Research Proposal: Surpassing the Weisfeiler–Lehman Expressivity Limit via Continuous-Time Quantum Walks

**Target Venue:** *npj Quantum Information* (Nature Portfolio)  
**Lead Author:** Nguyen Trong Luu (PTIT, Hanoi)  
**Primary Focus:** Quantum Computing, Geometric Deep Learning, Graph Expressivity Theory  

---

## 1. Executive Summary & Core Motivation

Standard Message Passing Neural Networks (MPNNs, GCN, GIN, and equivariant models such as EGNN and PRISM) suffer from a fundamental theoretical barrier: their expressive capacity is strictly bounded by the **1-Weisfeiler–Lehman (1-WL) color refinement test** (Morris et al., AAAI 2019; Xu et al., ICLR 2019). Consequently, classical GNNs cannot distinguish simple non-isomorphic graphs (e.g., $C_{10}$ vs. $2 \times C_5$, or strongly regular graphs of identical parameters). Classical solutions (such as $k$-WL higher-order tensor networks) require $\mathcal{O}(N^k)$ memory, rendering them impractical for scaling.

In this project, we exploit **Continuous-Time Quantum Walks (CTQW)** to design **QW-GNN** (Quantum Walk Graph Neural Network). By evolving quantum state amplitudes unitarily under graph Hamiltonians:
$$|\psi(t)\rangle = e^{-i \mathcal{H} t} |\psi(0)\rangle$$
constructive and destructive phase interference naturally computes non-local topological invariants across all path lengths simultaneously. We aim to prove both theoretically and empirically that quantum walk-based message passing strictly surpasses the 1-WL expressivity limit with polynomial complexity $\mathcal{O}(N^3)$ (single-particle) and maps directly to near-term quantum processors for interacting multi-particle walks.

---

## 2. Mathematical Framework

### 2.1 The 1-WL Failure Mode
Classical 1-WL assigns initial colors $c_v^{(0)} = 1$ and iteratively updates them via:
$$c_v^{(t+1)} = \text{HASH}\left(c_v^{(t)}, \{\{ c_u^{(t)} : u \in \mathcal{N}(v) \}\}\right)$$
For regular graphs of identical degree (e.g., a 10-node cycle $C_{10}$ versus two disjoint 5-node cycles $2 \times C_5$), the multisets of neighbors are identical ($\{\{2, 2\}\}$) at all iterations. Classical MPNNs therefore produce identical node embeddings and identical graph representations.

### 2.2 Continuous-Time Quantum Walk Operator
Let $G = (V, E)$ with adjacency matrix $\mathbf{A} \in \{0, 1\}^{N \times N}$ and degree matrix $\mathbf{D}$. We define the parameterized graph Hamiltonian:
$$\mathcal{H}(\boldsymbol{\theta}) = \mathbf{A} + \mathrm{diag}(\boldsymbol{\theta})$$
where $\boldsymbol{\theta} \in \mathbb{R}^N$ represents learnable on-site node potentials.

The unitary time evolution operator across duration $t$ is:
$$\mathbf{U}(t) = \exp\left(-i \mathcal{H}(\boldsymbol{\theta}) t\right) = \sum_{k=1}^N e^{-i \lambda_k t} \mathbf{v}_k \mathbf{v}_k^\dagger$$
where $\lambda_k, \mathbf{v}_k$ are the eigenvalues and eigenvectors of $\mathcal{H}$.

The quantum transition probability matrix $\mathbf{P}(t) \in \mathbb{R}^{N \times N}$ is:
$$\mathbf{P}_{jk}(t) = \left| \mathbf{U}_{jk}(t) \right|^2 = \left| \langle j | e^{-i \mathcal{H} t} | k \rangle \right|^2$$

### 2.3 Why Quantum Walks Break 1-WL
Unlike classical random walks whose transition matrix $\mathbf{M} = \mathbf{D}^{-1} \mathbf{A}$ converges to a stationary distribution that is purely local, quantum walks preserve quantum coherence and phase:
$$\mathbf{P}_{jk}(t) = \sum_{a, b} v_{ja} v_{ka} v_{jb} v_{kb} \cos\left( (\lambda_a - \lambda_b) t \right)$$
The off-diagonal beat frequencies $(\lambda_a - \lambda_b)$ encode global spectral invariants. For $C_{10}$ and $2 \times C_5$, the transition probabilities differ by $>0.53$ (demonstrated in `prototype_qw_isomorphism.py`), completely separating the graphs.

---

## 3. Architecture: QW-GNN (Quantum Walk Graph Neural Network)

```
   Input Graph G(A, X) 
           │
           ├──► [Parametric Quantum Walk Layer]
           │         U(t) = exp(-i (A + diag(θ)) t)
           │         P_t = |U(t)|^2 (Quantum Interference Matrix)
           │
           ├──► [Quantum Structural Positional Encoding (Q-SPE)]
           │         q_i = [diag(P(t_1)), diag(P(t_2)), ..., diag(P(t_K))]
           │
           └──► [Equivariant Message Passing with Q-SPE]
                     h_i^{(l+1)} = MLP(h_i^{(l)}, ∑_j P_ij(t) · W · h_j^{(l)})
```

1. **Multi-Scale Evolution Bank:** Evaluate $\mathbf{U}(t)$ across learnable time scales $t \in \{t_1, t_2, \dots, t_K\}$.
2. **Quantum Structural Positional Encoding (Q-SPE):** Autoconvolution return probabilities $\mathbf{P}_{ii}(t)$ provide provably 1-WL-transcending node identity features.
3. **Quantum-Gated Propagation:** Use transition probabilities $\mathbf{P}_{ij}(t)$ to modulate message-passing edge weights across long-range neighbors.

---

## 4. Benchmark Suite & Experimental Protocol

All experiments are designed to run cleanly on Apple Silicon (MacBook M5 Pro) using Pixi:

| Benchmark | Target Property | Why It Matters for npj QI |
| :--- | :--- | :--- |
| **Decagonal / Cycle Disjoint** | $C_{10}$ vs $2 \times C_5$ | Baseline sanity check: prove separation where 1-WL fails. |
| **Strongly Regular Graphs (SRG)** | SR16 (Shrikhande vs Rook $4 \times 4$), SR25 | Canonical graph isomorphism challenge; co-spectral test. |
| **Circular Skip Links (CSL)** | 10-class graph classification | Standard benchmark where standard GIN/GCN gets random chance (10%). |
| **BREC Benchmark** | 400 graph pairs (Basic, Regular, Extension, CFI) | The definitive NeurIPS/ICLR benchmark for testing expressive power beyond 1-WL and 3-WL. |
| **MUTAG / PROTEINS / QM9** | Real molecular graphs | Demonstrates practical molecular property prediction utility. |

---

## 5. Reviewer Strategy for *npj Quantum Information*

* **Journal Fit Reviewer:** Emphasize the bridge between quantum coherence / unitary dynamics and classical graph representation theory. Show that quantum interference acts as an inductive bias for non-local graph topology.
* **Methodology Reviewer:** Provide exact runtime complexity analysis ($\mathcal{O}(N^3)$ via eigendecomposition or Krylov subspace methods), ablation over time parameter $t$, and statistical error bounds ($p$-values, FDR correction).
* **Devil's Advocate Challenge:** Address classical simulability:
  * *Single-particle walks:* State honestly that single-particle CTQW is classically simulable in polynomial time, serving as an elite quantum-inspired classical architecture.
  * *Multi-particle walks:* Formulate the theoretical extension to 2-particle interacting bosons/fermions, where state space grows as $\mathcal{O}\binom{N}{k}$, establishing the formal link to exponential quantum advantage on quantum hardware.

---

## 6. Execution Roadmap (Milestones)

- [x] **Milestone 0:** Minimal proof-of-concept script (`prototype_qw_isomorphism.py`) validating $C_{10}$ vs $2 \times C_5$ separation ($p_{\text{diff}} > 0.53$).
- [ ] **Milestone 1:** Build the core PyTorch module `QWConv` (Quantum Walk Convolution layer) with learnable parameters $(t, \boldsymbol{\theta})$.
- [ ] **Milestone 2:** Implement the CSL (Circular Skip Links) benchmark; achieve >95% accuracy where classical MPNNs achieve 10%.
- [ ] **Milestone 3:** Run the full BREC 400-pair expressivity evaluation.
- [ ] **Milestone 4:** Multi-particle extension and theoretical theorem derivation for *npj Quantum Information*.
- [ ] **Milestone 5:** Manuscript drafting in LaTeX.
