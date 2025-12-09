import math
import random
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

random.seed(0)
np.random.seed(0)

# ==========================================
# 1. Quantum Circuit Primitives (Oracle & Diffusion)
# ==========================================

def int_to_bits(x: int, n: int) -> str:
    """Helper: Convert integer to n-bit binary string."""
    return format(x, f"0{n}b")

def build_oracle(n: int, marked: int) -> QuantumCircuit:
    """
    Constructs the Grover Oracle.
    Flips the phase of the |marked> state.
    """
    qc = QuantumCircuit(n)
    bits = int_to_bits(marked, n)
    
    # Map |marked> to |11...1> using X gates
    for q, b in enumerate(reversed(bits)):
        if b == "0":
            qc.x(q)
            
    # Multi-controlled Z gate (implemented via H-MCX-H)
    qc.h(n - 1)
    qc.mcx(list(range(n - 1)), n - 1)
    qc.h(n - 1)
    
    # Undo X gates
    for q, b in enumerate(reversed(bits)):
        if b == "0":
            qc.x(q)
            
    return qc

def build_diffusion(n: int) -> QuantumCircuit:
    """
    Constructs the Grover Diffusion Operator.
    Reflects about the uniform superposition state |s>.
    """
    qc = QuantumCircuit(n)
    
    # H^n -> X^n -> MCZ -> X^n -> H^n
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

def get_probability_trajectory(n: int, max_iter: int, shots: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulates Grover's algorithm for a range of iterations [0, max_iter].
    Returns: (theoretical_curve, empirical_curve)
    """
    N = 2**n
    marked = random.randrange(N)
    
    oracle = build_oracle(n, marked)
    diffusion = build_diffusion(n)
    backend = AerSimulator()
    
    empirical_probs = []
    theory_probs = []
    
    # Theoretical angle theta where sin(theta) = 1/sqrt(N)
    theta = math.asin(1 / math.sqrt(N))
    
    # Loop through iterations
    for t in range(max_iter + 1):
        # 1. Theoretical Calculation: P = sin^2((2t + 1) * theta)
        p_th = math.sin((2 * t + 1) * theta) ** 2
        theory_probs.append(p_th)
        
        # 2. Empirical Simulation (Monte Carlo)
        qc = QuantumCircuit(n)
        qc.h(range(n))
        
        # Apply Grover operator t times
        for _ in range(t):
            qc.compose(oracle, inplace=True)
            qc.compose(diffusion, inplace=True)
            
        qc.measure_all()
        
        # Execute
        job = backend.run(transpile(qc, backend), shots=shots)
        counts = job.result().get_counts()
        
        target_str = int_to_bits(marked, n)
        hits = counts.get(target_str, 0)
        empirical_probs.append(hits / shots)
        
    return np.array(theory_probs), np.array(empirical_probs)

def run_scaling_experiment(n_values: List[int]) -> Tuple[List[float], List[int]]:
    """
    Calculates the optimal iteration count for various N to verify O(sqrt(N)) scaling.
    """
    print("Running Scaling Experiment...")
    sqrt_N_list = []
    t_opt_list = []
    
    for n in n_values:
        N = 2**n
        t_opt = int(round(math.pi / 4 * math.sqrt(N)))
        sqrt_N_list.append(math.sqrt(N))
        t_opt_list.append(t_opt)
        
    return sqrt_N_list, t_opt_list

# ==========================================
# 3. Main Plotting Function
# ==========================================

def plot_combined_figure():
    """
    Generates a two-panel publication-quality figure.
    Left: Dynamics for multiple n (4, 6, 8) with optimal cutoffs.
    Right: Scaling law O(sqrt(N)).
    """
    # Configure Matplotlib for academic style (Serif fonts, Times New Roman)
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman"],
        "mathtext.fontset": "cm",
        "font.size": 12,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linestyle": "--"
    })

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    # --- Panel 1: Multi-n Dynamics ---
    n_dynamics = [4, 6, 8]  # Compare small, medium, large search spaces
    colors = ["#1B9E77", "#D95F02", "#003366"]  # Green, Orange, Navy (Colorblind safe)
    markers = ['^', 's', 'o']
    
    # Determine x-axis limit based on the largest n
    max_n = max(n_dynamics)
    t_opt_max = int((math.pi/4) * math.sqrt(2**max_n))
    x_limit = int(1.8 * t_opt_max) # Show slightly past the peak of the largest n
    
    print(f"Running Dynamics Experiment for n={n_dynamics}...")
    
    iterations = np.arange(x_limit + 1)
    
    for i, n in enumerate(n_dynamics):
        # Run simulation
        # Note: We ignore p_th (theory curve) here as requested
        _, p_sim = get_probability_trajectory(n, x_limit, shots=1000)
        
        # Calculate Theoretical Optimal Iteration for this n
        N = 2**n
        t_theory_opt = (math.pi / 4) * math.sqrt(N)
        
        # Plot Simulation (Markers)
        mark_step = max(1, x_limit // 15) 
        label_text = f"$n={n}$ ($N={2**n}$)"
        ax1.plot(iterations, p_sim, marker=markers[i], color=colors[i], 
                 linewidth=1.5, markersize=5, markevery=mark_step, label=label_text)
        
        # [MODIFIED] Add Vertical Line for Optimal Iteration
        # Uses the same color as the trajectory to match them visually
        ax1.axvline(x=t_theory_opt, color=colors[i], linestyle=":", alpha=0.8)
        ax1.text(t_theory_opt + 0.5, 0.5, r"$t_{opt}$", color=colors[i], fontsize=10)

    ax1.set_xlabel("Number of Iterations $t$")
    ax1.set_ylabel("Success Probability")
    ax1.set_title("Grover Dynamics: Effect of Problem Size $n$")
    ax1.set_ylim(-0.05, 1.15) # Fixed to standard probability range
    ax1.set_xlim(0, 15)  # Dynamic limit to show full curves
    ax1.legend(frameon=False, loc="upper right")
    
    # Despine
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # --- Panel 2: Complexity Scaling ---
    n_vals_scaling = [3, 4, 5, 6, 7, 8, 9, 10]
    sqrt_N, t_opt = run_scaling_experiment(n_vals_scaling)
    
    # Plot Theoretical Line: y = (pi/4) * x
    x_theory = np.linspace(min(sqrt_N), max(sqrt_N), 100)
    y_theory = (np.pi / 4) * x_theory
    
    ax2.plot(x_theory, y_theory, "k--", alpha=0.6, label=r"Theory $\approx \frac{\pi}{4}\sqrt{N}$")
    ax2.scatter(sqrt_N, t_opt, marker="s", color="#003366", s=50, zorder=3, label="Optimal Steps")
    
    ax2.set_xlabel(r"$\sqrt{N}$")
    ax2.set_ylabel("Optimal Iterations $t_{opt}$")
    ax2.set_title(r"Complexity Scaling $O(\sqrt{N})$")
    ax2.legend(frameon=False)
    
    # Despine
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig("grover_unisol_comparison.pdf", bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    plot_combined_figure()