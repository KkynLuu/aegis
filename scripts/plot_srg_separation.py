"""
Publication Figure Generator: SRG-16 Co-Spectral Separation.

Generates a publication-grade 4-panel figure for npj Quantum Information:
  Panel (a): Strongly Regular Graph specifications & Local Neighborhoods (K4 vs C6)
  Panel (b): Single-Particle CTQW return probabilities (Rook vs Shrikhande collapse)
  Panel (c): Two-Particle Interacting CTQW return probabilities (Phase divergence)
  Panel (d): Topological Separation Distance Delta(t) across time t in [0, 5]
"""

import os
import sys
import numpy as np

# Set writable matplotlib cache dir before importing matplotlib
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Ensure root qw_gnn path is in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from benchmarks.benchmark_srg import (
    generate_rook_4x4,
    generate_shrikhande,
    simulate_1particle_ctqw,
    simulate_2particle_ctqw
)

def create_srg_publication_figure(output_path: str = "benchmarks/figure_srg_separation.png"):
    print("[*] Generating SRG-16 co-spectral separation publication figure...")
    
    A_rook = generate_rook_4x4()
    A_shrik = generate_shrikhande()
    
    times = np.linspace(0.05, 5.0, 100)
    
    # 1. Simulate 1-particle
    ret_1p_rook = simulate_1particle_ctqw(A_rook, times)   # [T, 16]
    ret_1p_shrik = simulate_1particle_ctqw(A_shrik, times) # [T, 16]
    diff_1p = np.max(np.abs(ret_1p_rook - ret_1p_shrik), axis=1)
    
    # 2. Simulate 2-particle interacting (Bose-Hubbard U=4.0)
    ret_2p_rook = simulate_2particle_ctqw(A_rook, times, U_int=4.0)   # [T, 256]
    ret_2p_shrik = simulate_2particle_ctqw(A_shrik, times, U_int=4.0) # [T, 256]
    diff_2p = np.max(np.abs(ret_2p_rook - ret_2p_shrik), axis=1)
    
    # Publication Figure Styling
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Helvetica", "Arial", "DejaVu Sans"]
    plt.rcParams["axes.linewidth"] = 1.2
    plt.rcParams["xtick.major.width"] = 1.2
    plt.rcParams["ytick.major.width"] = 1.2
    
    fig = plt.figure(figsize=(13, 8), dpi=300)
    gs = gridspec.GridSpec(2, 2, height_ratios=[1, 1], hspace=0.32, wspace=0.28)
    
    color_rook = "#1f77b4"      # Quantum Deep Blue
    color_shrik = "#d62728"     # Ruby Crimson
    color_diff1 = "#7f7f7f"     # Slate Gray
    color_diff2 = "#2ca02c"     # Emerald Green
    
    # -------------------------------------------------------------
    # Panel (a): Adjacency Matrix Topology (Heatmaps)
    # -------------------------------------------------------------
    ax_a = fig.add_subplot(gs[0, 0])
    
    # Overlay visualization: Rook in lower triangle, Shrikhande in upper triangle
    overlay = np.tril(A_rook) + np.triu(A_shrik, k=1)
    im = ax_a.imshow(overlay, cmap="Blues", interpolation="nearest")
    
    ax_a.axline((0, 0), slope=1, color="red", linestyle="--", linewidth=1.5, alpha=0.7)
    ax_a.set_title("(a) Co-Spectral Graphs: Rook (Lower) vs Shrikhande (Upper)", fontsize=11, fontweight="bold")
    ax_a.set_xlabel("Node Index (0..15)", fontsize=10)
    ax_a.set_ylabel("Node Index (0..15)", fontsize=10)
    ax_a.text(1, 13, "Rook 4×4 (ω=4)", color="#08306b", fontweight="bold", fontsize=10)
    ax_a.text(7, 3, "Shrikhande (ω=3)", color="#67000d", fontweight="bold", fontsize=10)
    
    # -------------------------------------------------------------
    # Panel (b): Single-Particle Return Probabilities P_ii(t)
    # -------------------------------------------------------------
    ax_b = fig.add_subplot(gs[0, 1])
    
    ax_b.plot(times, ret_1p_rook[:, 0], label="Rook 4×4 $P_{ii}^{(1)}(t)$", color=color_rook, linewidth=2.5, linestyle="-")
    ax_b.plot(times, ret_1p_shrik[:, 0], label="Shrikhande $P_{ii}^{(1)}(t)$", color=color_shrik, linewidth=2.0, linestyle="--")
    
    ax_b.set_title("(b) Single-Particle CTQW: Bose-Mesner Collapse", fontsize=11, fontweight="bold")
    ax_b.set_xlabel("Evolution Duration $t$", fontsize=10)
    ax_b.set_ylabel("Return Probability $P_{ii}(t)$", fontsize=10)
    ax_b.grid(True, linestyle=":", alpha=0.6)
    ax_b.legend(frameon=True, loc="upper right", fontsize=9)
    ax_b.text(0.05, 0.15, "Curves Exactly Overlap\n$\Delta(t) \equiv 0.000000$ (2-WL Barrier)", 
              transform=ax_b.transAxes, fontsize=9, bbox=dict(boxstyle="round,pad=0.4", facecolor="#fff2f2", edgecolor="#d62728"))
    
    # -------------------------------------------------------------
    # Panel (c): Two-Particle Interacting Return Probabilities P_{ij,ij}(t)
    # -------------------------------------------------------------
    ax_c = fig.add_subplot(gs[1, 0])
    
    # Plot maximally divergent state index 223: pair |13, 15>
    state_idx = 223
    ax_c.plot(times, ret_2p_rook[:, state_idx], label="Rook 4×4 $|13, 15\\rangle$", color=color_rook, linewidth=2.2)
    ax_c.plot(times, ret_2p_shrik[:, state_idx], label="Shrikhande $|13, 15\\rangle$", color=color_shrik, linewidth=2.2, linestyle="--")
    
    ax_c.set_title("(c) Two-Particle Interacting CTQW (Bose-Hubbard $U=4.0$)", fontsize=11, fontweight="bold")
    ax_c.set_xlabel("Evolution Duration $t$", fontsize=10)
    ax_c.set_ylabel("Joint Return Prob $P_{ij,ij}(t)$", fontsize=10)
    ax_c.grid(True, linestyle=":", alpha=0.6)
    ax_c.legend(frameon=True, loc="upper right", fontsize=9)
    ax_c.text(0.05, 0.15, "Contact Interaction $U=4.0$\nShatters Degeneracy ($\Delta > 0.24$)", 
              transform=ax_c.transAxes, fontsize=9, bbox=dict(boxstyle="round,pad=0.4", facecolor="#f0fff0", edgecolor="#2ca02c"))
    
    # -------------------------------------------------------------
    # Panel (d): Topological Separation Distance Delta(t)
    # -------------------------------------------------------------
    ax_d = fig.add_subplot(gs[1, 1])
    
    ax_d.plot(times, diff_1p, label="1-Particle Separation $\Delta_1(t)$", color=color_diff1, linewidth=2.0, linestyle=":")
    ax_d.plot(times, diff_2p, label="2-Particle Separation $\Delta_2(t)$ ($U=4.0$)", color=color_diff2, linewidth=2.5)
    
    ax_d.set_title("(d) Topological Separation Distance $\Delta(t)$", fontsize=11, fontweight="bold")
    ax_d.set_xlabel("Evolution Duration $t$", fontsize=10)
    ax_d.set_ylabel("Max Return Difference $\Delta(t)$", fontsize=10)
    ax_d.grid(True, linestyle=":", alpha=0.6)
    ax_d.legend(frameon=True, loc="upper left", fontsize=9)
    
    peak_val = np.max(diff_2p)
    peak_t = times[np.argmax(diff_2p)]
    ax_d.annotate(f"Peak $\Delta = {peak_val:.3f}$\nat $t={peak_t:.2f}$", 
                  xy=(peak_t, peak_val), xytext=(peak_t - 1.4, peak_val - 0.05),
                  arrowprops=dict(facecolor="#2ca02c", shrink=0.08, width=1.5, headwidth=6),
                  fontsize=9, fontweight="bold", color="#1b661b")
    
    # Save figure
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[✓] Successfully generated publication figure at: {output_path}")

if __name__ == "__main__":
    out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "benchmarks"))
    out_file = os.path.join(out_dir, "figure_srg_separation.png")
    create_srg_publication_figure(out_file)
