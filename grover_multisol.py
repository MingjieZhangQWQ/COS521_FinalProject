import math
import random
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

# ==========================================
# 1. Quantum Circuit Primitives
# ==========================================

def int_to_bits(x: int, n: int) -> str:
    return format(x, f"0{n}b")

def build_single_oracle(n: int, marked: int) -> QuantumCircuit:
    qc = QuantumCircuit(n)
    bits = int_to_bits(marked, n)
    for q, b in enumerate(reversed(bits)):
        if b == "0": qc.x(q)
    qc.h(n - 1)
    qc.mcx(list(range(n - 1)), n - 1)
    qc.h(n - 1)
    for q, b in enumerate(reversed(bits)):
        if b == "0": qc.x(q)
    return qc

def build_multi_oracle(n: int, marked_list: List[int]) -> QuantumCircuit:
    qc = QuantumCircuit(n)
    for m in marked_list:
        qc.compose(build_single_oracle(n, m), inplace=True)
    return qc

def build_diffusion(n: int) -> QuantumCircuit:
    qc = QuantumCircuit(n)
    qc.h(range(n))
    qc.x(range(n))
    qc.h(n - 1)
    qc.mcx(list(range(n - 1)), n - 1)
    qc.h(n - 1)
    qc.x(range(n))
    qc.h(range(n))
    return qc

# ==========================================
# 2. Experiment Helpers
# ==========================================

def get_trajectory(n: int, k: int, shots: int = 1000, seed: int = 42):
    """Simulates dynamics for Left Plot."""
    random.seed(seed)
    N = 2 ** n
    marked_list = random.sample(range(N), k)
    
    t_theory_opt = (math.pi / 4) * math.sqrt(N / k)
    T_max = int(1.8 * t_theory_opt) + 2
    
    oracle = build_multi_oracle(n, marked_list)
    diffusion = build_diffusion(n)
    backend = AerSimulator()
    
    trajectory = []
    
    for t in range(T_max):
        qc = QuantumCircuit(n)
        qc.h(range(n))
        for _ in range(t):
            qc.compose(oracle, inplace=True)
            qc.compose(diffusion, inplace=True)
        qc.measure_all()
        
        job = backend.run(transpile(qc, backend), shots=shots, seed_simulator=seed+t)
        counts = job.result().get_counts()
        
        hits = 0
        for m in marked_list:
            hits += counts.get(int_to_bits(m, n), 0)
        trajectory.append(hits / shots)
        
    return np.arange(T_max), np.array(trajectory), t_theory_opt

def run_distribution_experiment(n: int, k: int, shots: int = 2000, seed: int = 99):
    """
    Runs Grover at exactly t_opt and returns the distribution of marked items.
    Used for Right Plot.
    """
    random.seed(seed)
    N = 2 ** n
    marked_list = sorted(random.sample(range(N), k)) # Sort for better plotting
    
    # Calculate optimal step
    t_opt = int(round((math.pi / 4) * math.sqrt(N / k)))
    
    # Build circuit
    oracle = build_multi_oracle(n, marked_list)
    diffusion = build_diffusion(n)
    
    qc = QuantumCircuit(n)
    qc.h(range(n))
    for _ in range(t_opt):
        qc.compose(oracle, inplace=True)
        qc.compose(diffusion, inplace=True)
    qc.measure_all()
    
    # Run simulation
    backend = AerSimulator()
    job = backend.run(transpile(qc, backend), shots=shots, seed_simulator=seed)
    counts = job.result().get_counts()
    
    # Extract frequencies for marked items
    marked_freqs = []
    for m in marked_list:
        bitstr = int_to_bits(m, n)
        c = counts.get(bitstr, 0)
        marked_freqs.append(c / shots)
        
    # Calculate noise level (average probability of non-marked items)
    total_marked_hits = sum(marked_freqs) * shots
    total_noise_hits = shots - total_marked_hits
    num_noise_states = N - k
    avg_noise_freq = (total_noise_hits / num_noise_states) / shots if num_noise_states > 0 else 0
    
    return marked_list, marked_freqs, avg_noise_freq, t_opt

# ==========================================
# 3. Main Plotting Function
# ==========================================

def plot_combined_multi_solution():
    # Academic Style Configuration
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman"],
        "mathtext.fontset": "cm",
        "font.size": 12,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linestyle": "--"
    })
    
    # Setup Figure (1 Row, 2 Columns)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8))
    
    # --- LEFT PANEL: Dynamics Trace ---
    n = 10
    k_values = [1, 4, 16]
    colors = ["#003366", "#D95F02", "#1B9E77"] 
    
    print("Generating Left Panel (Dynamics)...")
    for i, k in enumerate(k_values):
        t_vals, probs, t_theory = get_trajectory(n, k, shots=1000)
        
        # Plot Trajectory
        ax1.plot(t_vals, probs, marker='o', markersize=4, linewidth=1.5, 
                color=colors[i], label=f"$k={k}$")
        
        # Plot Theoretical Line
        ax1.axvline(x=t_theory, color=colors[i], linestyle=':', alpha=0.8)
        # Add label for the first one only to avoid clutter
        if i == 0:
            ax1.text(t_theory + 0.5, 0.05, r"$t_{opt}$", color=colors[i], fontsize=10)

    ax1.set_xlabel("Iterations $t$")
    ax1.set_ylabel("Total Success Probability")
    ax1.set_title(f"Convergence Speedup ($N=2^{{{n}}}$)")
    ax1.set_ylim(-0.05, 1.05)
    ax1.set_xlim(left=0)
    ax1.legend(frameon=False, loc="lower right")
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # --- RIGHT PANEL: Distribution Histogram ---
    # Case study: n=10, k=4
    k_target = 4
    print(f"Generating Right Panel (Distribution for k={k_target})...")
    
    marked_indices, freqs, noise_freq, t_opt_used = run_distribution_experiment(n, k_target)
    
    # Prepare x-axis labels (State indices)
    x_pos = np.arange(len(marked_indices))
    labels = [f"|{m}\\rangle" for m in marked_indices]
    
    # Plot Bars for Marked States
    bars = ax2.bar(x_pos, freqs, color="#D95F02", alpha=0.8, width=0.6, label="Marked States")
    
    # Add a horizontal line for Noise Level (to show contrast)
    # Usually noise is very close to 0, but good to visualize scale
    ax2.axhline(y=noise_freq, color="gray", linestyle="--", linewidth=1, label="Avg Noise Level")
    
    # Add values on top of bars
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                 f'{height:.2f}', ha='center', va='bottom', fontsize=9)

    ax2.set_xlabel("Marked Solutions")
    ax2.set_ylabel("Measured Frequency")
    ax2.set_title(f"Solution Distribution ($k={k_target}, t={t_opt_used}$)")
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(labels)
    ax2.set_ylim(0, max(freqs)*1.2) # Give some headroom
    ax2.legend(frameon=False)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig("grover_multisol_combined.pdf", bbox_inches="tight")
    plt.show()

if __name__ == "__main__":
    plot_combined_multi_solution()