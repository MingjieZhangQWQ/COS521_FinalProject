import math
import random
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error

random.seed(0)
np.random.seed(0)

# ---------- Utilities (Same as before) ----------

def int_to_bits(x: int, n: int) -> str:
    return format(x, f"0{n}b")

def build_oracle(n: int, marked: int) -> QuantumCircuit:
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

# ---------- Experiment 3: Noise Analysis ----------

def run_noise_sensitivity(n: int, 
                          max_noise: float = 0.02, 
                          steps: int = 15, 
                          shots: int = 1000) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Simulates Grover's algorithm under varying levels of depolarizing noise.
    
    Args:
        n: Number of qubits.
        max_noise: Maximum gate error probability to simulate.
        steps: Number of data points.
    
    Returns:
        noise_levels: Array of error probabilities tested.
        success_rates: Array of measured success probabilities.
        random_guessing: Baseline probability (1/N).
    """
    # 1. Setup problem
    N = 2**n
    marked = random.randrange(N)
    t_opt = int(round(math.pi * math.sqrt(N) / 4.0))
    
    # 2. Build the ideal circuit once
    qc = QuantumCircuit(n)
    qc.h(range(n))
    oracle = build_oracle(n, marked)
    diffusion = build_diffusion(n)
    
    for _ in range(t_opt):
        qc.compose(oracle, inplace=True)
        qc.compose(diffusion, inplace=True)
    qc.measure_all()
    
    # 3. Noise Loop
    noise_levels = np.linspace(0, max_noise, steps)
    success_rates = []
    
    print(f"Running Noise Analysis (n={n}, t_opt={t_opt})...")
    
    for p_err in noise_levels:
        # Create a noise model: Depolarizing noise
        # We apply error to both single-qubit gates and CX gates
        noise_model = NoiseModel()
        
        # 1-qubit error
        error_1q = depolarizing_error(p_err, 1)
        noise_model.add_all_qubit_quantum_error(error_1q, ['u1', 'u2', 'u3', 'h', 'x'])
        
        # 2-qubit error (CX is usually noisier, but we assume uniform p for simplicity here)
        error_2q = depolarizing_error(p_err, 2)
        noise_model.add_all_qubit_quantum_error(error_2q, ['cx'])
        
    for p_err in noise_levels:
        # Create a noise model: Depolarizing noise
        # We apply error to both single-qubit gates and CX gates
        noise_model = NoiseModel()
        
        # 1-qubit error
        error_1q = depolarizing_error(p_err, 1)
        noise_model.add_all_qubit_quantum_error(error_1q, ['u1', 'u2', 'u3', 'h', 'x'])
        
        # 2-qubit error (CX is usually noisier, but we assume uniform p for simplicity here)
        error_2q = depolarizing_error(p_err, 2)
        noise_model.add_all_qubit_quantum_error(error_2q, ['cx'])
        
        # Run Simulation (with fixed seeds)
        backend = AerSimulator(noise_model=noise_model)

        compiled = transpile(qc, backend, seed_transpiler=0)
        job = backend.run(compiled, shots=shots, seed_simulator=0)

        counts = job.result().get_counts()
        
        # Calculate success rate
        target_bit = int_to_bits(marked, n)
        hits = counts.get(target_bit, 0)
        success_rates.append(hits / shots)

    return noise_levels, np.array(success_rates), 1.0/N

# ---------- Plotting ----------

def plot_noise_robustness():
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
    
    # Run Experiment
    n = 7  # 128 items
    noise_x, success_y, baseline = run_noise_sensitivity(n=n, max_noise=0.002, steps=20)
    
    fig, ax = plt.subplots(figsize=(7, 4.5))
    
    # Plot Experiment Data
    ax.plot(noise_x, success_y, 'o-', color="#8B0000", linewidth=1.5, 
            markersize=5, label="Grover Success Rate")
    
    # Plot Baseline (Random Guessing)
    ax.axhline(y=baseline, color="gray", linestyle="--", alpha=0.8, 
               label=r"Random Guessing ($1/N$)")
    
    # Labels and Titles
    ax.set_xlabel("Gate Error Probability $p$ (Depolarizing)")
    ax.set_ylabel("Success Probability at $t_{opt}$")
    
    # Limits
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlim(left=0)
    
    # Legend
    ax.legend(frameon=False)
    
    # Despine
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig("grover_noise_robustness.pdf", bbox_inches="tight")
    plt.show()

if __name__ == "__main__":
    plot_noise_robustness()