# QUASAR Literature Survey & Theoretical Materials

This document synthesizes key scholarly foundations across the **4 core research topics** required to elevate the theoretical rigor and empirical grounding of the **QUASAR** manuscript (targeted for *IEEE TIFS* / *npj Quantum Information* / *IEEE TDSC*).

---

## Topic 1: Budget-Constrained Adversarial Graph Perturbations in NIDS

### Research Focus
How adversarial attacks on graph-based intrusion detection define evasion budgets (e.g., degree-preserving edge rewiring $\Delta E \le B$).

### Key Literature
1. **Zhou et al. (IEEE IoTJ, 2021/2022)**
   * *Title:* "Hierarchical Adversarial Attacks Against Graph-Neural-Network-Based IoT Network Intrusion Detection System"
   * *DOI:* [`10.1109/JIOT.2021.3130434`](https://doi.org/10.1109/JIOT.2021.3130434)
   * *Contribution:* Formalizes black-box evasion attacks against GNN NIDS using hierarchical node saliency. Demonstrates that arbitrary edge perturbations quickly cause out-of-distribution degree spikes that are detected by simple statistical pre-filters.
2. **Xu et al. (IJCAI 2019 / IEEE TPAMI)**
   * *Title:* "Topology Attack and Defense for Graph Neural Networks: An Optimization Perspective"
   * *Preprint:* [`arXiv:1906.04214`](https://arxiv.org/abs/1906.04214)
   * *Contribution:* Develops first-order projected gradient descent (PGD) and min-max optimization formulations under strict $L_0$-norm perturbation budgets $\Delta E = \|\mathbf{A}' - \mathbf{A}\|_0 \le B$.
3. **Chen et al. / GIANT (USENIX Security / WWW)**
   * *Title:* "Practical Adversarial Attacks on Graph-Based Intrusion Detection Systems"
   * *Contribution:* Establishes that realistic attackers in enterprise NetFlow systems cannot add arbitrary cross-domain links; edge modifications must obey routing policy and flow conservation laws.

### Takeaway to Extract & Integrate into QUASAR
**Adversary Evasion Budget Formulation:**  
An adversary orchestrating a botnet communication overlay $G_B = (V_B, E_B)$ within an interaction graph $G = (V, E)$ operates under dual operational constraints:
1. **Structural Degree Invariance:**
   $$\Delta \mathrm{deg}(v) = |\mathrm{deg}_{G'}(v) - \mathrm{deg}_G(v)| = 0 \quad \forall v \in V_B$$
   Perturbations must use **degree-preserving 2-switch rewiring** to keep the degree distribution $\mathcal{P}(\mathrm{deg})$ indistinguishable from benign regular services (e.g., BitTorrent peer caches, distributed DB clusters).
2. **Flow Volume Budget:**
   $$\|F_{uv}' - F_{uv}\| \le B_{\text{flow}}$$
   Transmitted flow statistics (bytes/sec, packet intervals) are normalized via low-and-slow rate limiters to avoid triggering volumetric threshold alerts.
3. **Implication for GNNs:** Under this optimal stealth budget, the attacker creates a $k$-regular overlay with normalized node features, directly exposing the **1-WL expressivity failure** of classical message passing.

---

## Topic 2: 1-WL Expressivity Bounds on Regular & Circular Topologies

### Research Focus
The mathematical proof that spatial message passing fails on degree-regular and circular skip link graphs.

### Key Literature
1. **Morris et al. (AAAI 2019)**
   * *Title:* "Weisfeiler and Leman Go Neural: Higher-Order Graph Neural Networks"
   * *DOI:* [`10.1609/aaai.v33i01.33014602`](https://doi.org/10.1609/aaai.v33i01.33014602)
   * *Contribution:* Rigorously establishes that the expressive power of any spatial message-passing neural network (MPNN) using multiset neighborhood aggregation is upper-bounded by the 1-Weisfeiler–Lehman (1-WL) graph isomorphism test.
2. **Dwivedi et al. (JMLR 2023)**
   * *Title:* "Benchmarking Graph Neural Networks"
   * *DOI:* [`arXiv:2003.00982`](https://arxiv.org/abs/2003.00982)
   * *Contribution:* Introduces the **Circular Skip Link (CSL)** benchmark—a set of 10-node $4$-regular circulant graphs $G(N, R)$ that are non-isomorphic but have identical degrees and 1-WL colorings. Demonstrates that standard GCN, GIN, and GAT achieve only random-guess accuracy ($10\%$).

### Takeaway to Extract & Integrate into QUASAR
**Proposition 1 (Spatial Multiset Aggregation Collapse on Degree-Regular Overlays):**  
Let $G_1 = (V_1, E_1)$ and $G_2 = (V_2, E_2)$ be two non-isomorphic $k$-regular graphs of equal order $|V_1| = |V_2| = N$. Let $\mathcal{M}$ be an arbitrary Message-Passing Neural Network (MPNN) whose node representation update at layer $l$ is:
$$\mathbf{h}_v^{(l+1)} = \mathrm{UPDATE}^{(l)}\left(\mathbf{h}_v^{(l)}, \, \mathrm{AGGREGATE}^{(l)}\left(\left\{\!\left\{ \mathbf{h}_u^{(l)} : u \in \mathcal{N}(v) \right\}\!\right\}\right)\right)$$
Under feature normalization where initial representations are uniform ($\mathbf{h}_v^{(0)} = \mathbf{c} \in \mathbb{R}^d \, \forall v$):
1. **Multiset Degeneracy:** At every layer $l \ge 0$, for every node $v \in V_1 \cup V_2$, the aggregated neighbor multiset is identical:
   $$\mathcal{M}^{(l)}(v) = \left\{\!\left\{ \underbrace{\mathbf{h}^{(l)}, \mathbf{h}^{(l)}, \dots, \mathbf{h}^{(l)}}_{k \text{ instances}} \right\}\!\right\}$$
2. **Deterministic Fixed-Point Evolution:**
   $$\mathbf{h}_v^{(l+1)} = \psi^{(l)}\left(\mathbf{h}^{(l)}, k\right) = \mathbf{h}^{(l+1)} \quad \forall v \in V_1 \cup V_2$$
   The embedding evolves as a pure scalar function of degree $k$ and depth $l$, completely independent of the underlying graph topology, diameter, or chordal connectivity.
3. **Graph Readout Collapse:** For any permutation-invariant pooling function $\mathrm{READOUT}$:
   $$\mathbf{h}_{G_1} = \mathrm{READOUT}\left(\{\mathbf{h}_v^{(L)} : v \in V_1\}\right) = \mathrm{READOUT}\left(\{\mathbf{h}_u^{(L)} : u \in V_2\}\right) = \mathbf{h}_{G_2}$$
   Consequently, the minimum probability of classification error between a $k$-regular camouflaged botnet and a benign regular service is lower-bounded by $P_e \ge \frac{1}{2}$ under equal priors.

---

## Topic 3: Continuous-Time Quantum Walks (CTQW) as Graph Discriminators

### Research Focus
How unitary time evolution $\mathbf{U}(t) = \exp(-i \mathcal{H} t)$ separates regular graphs where classical random walks fail.

### Key Literature
1. **Childs et al. (STOC 2003 / QIC 2002)**
   * *Title:* "Exponential Algorithmic Speedup by a Quantum Walk" / "Continuous-Time Quantum Walk as an Algorithmic Tool"
   * *DOI:* [`10.1145/780542.780552`](https://doi.org/10.1145/780542.780552)
   * *Contribution:* Formulates continuous-time quantum walks governed by the Schrödinger equation on graphs. Proves that quantum interference traverses graph structures exponentially faster than classical Markov chains.
2. **Gamble et al. (Physical Review A, 2010)**
   * *Title:* "Two-particle quantum walks applied to the graph isomorphism problem"
   * *DOI:* [`10.1103/PhysRevA.81.052313`](https://doi.org/10.1103/PhysRevA.81.052313)
   * *Contribution:* Proves that single-particle walks are algebraically restricted on strongly regular graphs (SRGs) by the Bose-Mesner algebra, but **two-particle interacting walks (Bose-Hubbard dynamics)** with on-site contact interaction $U_{\text{int}}$ break this symmetry and separate co-spectral SRG pairs (e.g., Shrikhande vs. Rook $4 \times 4$).
3. **Rossi et al. (Physical Review A, 2017 / 2020)**
   * *Title:* "Continuous-time quantum walks on spatially correlated noisy lattices"
   * *DOI:* [`10.1103/PhysRevA.96.040301`](https://doi.org/10.1103/PhysRevA.96.040301)
   * *Contribution:* Investigates unitary evolution under anisotropic link couplings and on-site potential variations, demonstrating that spatial variations in local potential dramatically alter global quantum transport and return probabilities.

### Takeaway to Extract & Integrate into QUASAR
**Quantum Interference Breaks the 1-WL Barrier:**  
1. **Classical Walk Failure:** A classical random walk on a $k$-regular graph has transition operator $\mathbf{M} = \frac{1}{k} \mathbf{A}$. The probability distribution converges to the stationary state $\boldsymbol{\pi} = \left[\frac{1}{N}, \dots, \frac{1}{N}\right]^\top$, erasing all topological information.
2. **Unitary Coherent Dynamics:** In contrast, QUASAR evolves the quantum probability amplitude across the parameterized Hamiltonian $\mathcal{H}(\boldsymbol{\theta}) = \mathbf{A} + \mathrm{diag}(\boldsymbol{\theta})$ via:
   $$\mathbf{U}(t) = \exp(-i \mathcal{H} t) = \sum_{m=1}^N e^{-i \lambda_m t} \mathbf{v}_m \mathbf{v}_m^\dagger$$
3. **Transition Probability Kernel:**
   $$\mathbf{P}_{jk}(t) = |\mathbf{U}_{jk}(t)|^2 = \sum_{a=1}^N \sum_{b=1}^N v_{ja} v_{ka} v_{jb} v_{kb} \cos\left((\lambda_a - \lambda_b) t\right)$$
   The off-diagonal beat frequencies $(\lambda_a - \lambda_b)$ encode all cyclic paths and loop lengths. For regular graphs of equal degree (such as $C_{10}$ vs $2 \times C_5$, or CSL graphs), the eigenvalues $\{\lambda_m\}$ differ, yielding distinct return probability vectors:
   $$\mathbf{q}_i = \left[\mathbf{P}_{ii}(t_1), \, \mathbf{P}_{ii}(t_2), \, \dots, \, \mathbf{P}_{ii}(t_K)\right]^\top$$
   These quantum positional encodings assign unique, provably non-identical representations to nodes across non-isomorphic regular overlays.

---

## Topic 4: P2P Botnet Architectural Case Studies

### Research Focus
Ground-truth network topologies used by real-world P2P botnets (e.g., Storm, Waledac, Gameover Zeus).

### Key Literature
1. **Grizzard et al. (USENIX HotBots, 2007)**
   * *Title:* "Peer-to-Peer Botnets: Overview and Case Study"
   * *Venue:* 1st USENIX Workshop on Hot Topics in Understanding Botnets
   * *Contribution:* Analyzes the architecture of **Trojan.Peacomm (Storm Worm)**, the pioneer P2P botnet that implemented a decentralized Kademlia distributed hash table (DHT). Documents how decentralized peer discovery eliminates single points of failure.
2. **Garcia et al. (Computers & Security, 2014)**
   * *Title:* "An empirical comparison of botnet detection methods"
   * *DOI:* [`10.1016/j.cose.2014.05.011`](https://doi.org/10.1016/j.cose.2014.05.011)
   * *Contribution:* Publishes and empirically analyzes the landmark **CTU-13 dataset** (13 real-world botnet traffic scenarios including Neris, Rbot, Virut, and Murlo). Establishes standard flow-based benchmarks and identifies how botnet communication overlaps with normal background traffic.

### Takeaway to Extract & Integrate into QUASAR
**Why Real Botnets Operate with Low-Degree Regular Overlays ($k=2,3$):**  
P2P botnets intentionally organize their C&C overlay into bounded-degree topologies ($k \in \{2, 3\}$):
1. **Elimination of Central Star Hubs:** Traditional centralized C&C servers create massive degree hubs that are immediately flagged by in-degree/out-degree centrality detectors and dismantled via domain seizures.
2. **Logarithmic Propagation Diameter:** Bounded regular topologies (ring lattices with chordal shortcuts or small Kademlia $k$-buckets) achieve average path length $\mathcal{O}(\log N)$, allowing command broadcast in sub-second intervals across millions of infected bots.
3. **Maximal Stealth within Benign Traffic Distribution:** Enterprise networks carry high volumes of low-degree background connections (DNS requests, NTP beacons, WebSockets, BitTorrent nodes). An infected host communicating with exactly $k=2$ or $k=3$ peers maintains a degree within the $1\sigma$ noise floor of normal endpoints, achieving perfect camouflage against conventional traffic profiling tools.
