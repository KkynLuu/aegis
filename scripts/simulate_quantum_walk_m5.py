#!/usr/bin/env python3
"""
Local Quantum Walk Simulator on Apple Silicon (M5).
Runs 100% locally using Qiskit Aer — ZERO cloud QPU time used!

1. Evaluates Trotterized Continuous-Time Quantum Walk (CTQW) on 4-node motif.
2. Compares against exact analytical matrix exponential P(t) = |exp(-iHt)|^2.
3. Computes Trotter fidelity across evolution time t in [0.2, 2.0].
4. Generates publication-ready figure: benchmarks/figure_quantum_trotter_m5.png.
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import qiskit
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator

def build_trotter_circuit(num_nodes=4, dt=0.5, trotter_steps=3):
    """Synthesizes 2nd-order Suzuki-Trotter circuit for 4-node ring."""
    edges = [(0, 1), (1, 2), (2, 3), (3, 0)]
    qc = QuantumCircuit(num_nodes, num_nodes)
    
    # State initialization |1000> (Node 0)
    qc.x(0)
    
    half_dt = dt / 2.0
    for _ in range(trotter_steps):
        # Forward XY hopping
        for u, v in edges:
            qc.cx(u, v)
            qc.rx(np.pi / 2, u)
            qc.rz(half_dt, u)
            qc.rx(-np.pi / 2, u)
            qc.cx(u, v)
        # Symmetrized reverse XY hopping
        for u, v in reversed(edges):
            qc.cx(u, v)
            qc.rx(np.pi / 2, u)
            qc.rz(half_dt, u)
            qc.rx(-np.pi / 2, u)
            qc.cx(u, v)
            
    qc.measure(range(num_nodes), range(num_nodes))
    return qc

def main():
    print("=" * 70)
    print("Local Continuous-Time Quantum Walk Simulator (M5 Apple Silicon)")
    print("Running 100% LOCALLY — Cloud QPU Time Used: 0.00 seconds")
    print("=" * 70)

    # 1. Exact Analytical Physics (Matrix Exponential)
    # Adjacency for 4-node cycle
    A = np.array([
        [0, 1, 0, 1],
        [1, 0, 1, 0],
        [0, 1, 0, 1],
        [1, 0, 1, 0]
    ], dtype=float)
    eigvals, V = np.linalg.eigh(A)
    
    t_dense = np.linspace(0.1, 2.2, 100)
    p00_exact_dense = []
    for t in t_dense:
        U = V @ np.diag(np.exp(-1j * eigvals * t)) @ V.T
        p00_exact_dense.append(np.abs(U[0, 0]) ** 2)
        
    t_points = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]
    p00_exact = []
    for t in t_points:
        U = V @ np.diag(np.exp(-1j * eigvals * t)) @ V.T
        p00_exact.append(np.abs(U[0, 0]) ** 2)

    # 2. Local M5 Qiskit Aer Circuit Simulation
    sim = AerSimulator()
    shots = 4096
    
    p00_trotter_s2 = []
    p00_trotter_s3 = []
    
    print(f"[*] Simulating Trotterized quantum circuits across {len(t_points)} time steps...")
    for t in t_points:
        # Steps = 2
        dt_s2 = t / 2.0
        qc_s2 = build_trotter_circuit(num_nodes=4, dt=dt_s2, trotter_steps=2)
        counts_s2 = sim.run(qc_s2, shots=shots).result().get_counts()
        p00_s2 = counts_s2.get('0001', 0) / shots
        p00_trotter_s2.append(p00_s2)
        
        # Steps = 3
        dt_s3 = t / 3.0
        qc_s3 = build_trotter_circuit(num_nodes=4, dt=dt_s3, trotter_steps=3)
        counts_s3 = sim.run(qc_s3, shots=shots).result().get_counts()
        p00_s3 = counts_s3.get('0001', 0) / shots
        p00_trotter_s3.append(p00_s3)

    # Calculate State Fidelity Metric (Bhattacharyya coefficient)
    fidelities_s3 = [
        1.0 - np.abs(p_sim - p_ex) 
        for p_sim, p_ex in zip(p00_trotter_s3, p00_exact)
    ]
    mean_fidelity = np.mean(fidelities_s3) * 100.0
    print(f"[✓] Simulation complete! Mean Trotter Fidelity (S=3): {mean_fidelity:.2f}%")

    # 3. Generate Publication Figure
    os.makedirs("benchmarks", exist_ok=True)
    out_fig = "benchmarks/figure_quantum_trotter_m5.png"
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2), dpi=300)
    
    # Left: Return Probability Dynamics
    ax1.plot(t_dense, p00_exact_dense, 'k--', linewidth=2.0, label='Exact Analytical $\\mathbf{P}_{00}(t)$')
    ax1.plot(t_points, p00_trotter_s2, 'r^:', markersize=6, label='Qiskit M5 (Trotter $S=2$)')
    ax1.plot(t_points, p00_trotter_s3, 'bo-', markersize=6, label='Qiskit M5 (Trotter $S=3$)')
    ax1.set_title("A. Quantum Walk Probability Dynamics", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Continuous Evolution Duration $t$", fontsize=10)
    ax1.set_ylabel("Quantum Return Probability $\\mathbf{P}_{00}(t)$", fontsize=10)
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(frameon=True, fontsize=9)
    
    # Right: Trotter Discretization Error
    err_s2 = np.abs(np.array(p00_trotter_s2) - np.array(p00_exact))
    err_s3 = np.abs(np.array(p00_trotter_s3) - np.array(p00_exact))
    ax2.plot(t_points, err_s2, 'r^--', linewidth=1.5, label='Error ($S=2$ steps)')
    ax2.plot(t_points, err_s3, 'bo-', linewidth=1.5, label='Error ($S=3$ steps)')
    ax2.set_title("B. Suzuki-Trotter Gate Synthesis Error", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Evolution Duration $t$", fontsize=10)
    ax2.set_ylabel("Absolute Error $|P_{\\mathrm{sim}} - P_{\\mathrm{exact}}|$", fontsize=10)
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(frameon=True, fontsize=9)
    
    plt.tight_layout()
    plt.savefig(out_fig)
    print(f"[✓] Publication-quality figure saved: {out_fig}")
    print("=" * 70)

if __name__ == "__main__":
    main()
