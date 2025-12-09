import numpy as np
import random
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit_aer import AerSimulator

random.seed(0)
np.random.seed(0)


# ==========================================
# 1. Component Construction
# ==========================================

def create_sudoku_oracle(var_qubits, clause_qubits, output_qubit):
    """
    Constructs the Oracle for a 2x2 Binary Sudoku.
    Variables: q0 (NW), q1 (NE), q2 (SW), q3 (SE)
    Constraints:
      1. Row 0: q0 != q1
      2. Row 1: q2 != q3
      3. Col 0: q0 != q2
      4. Col 1: q1 != q3
    """
    qc = QuantumCircuit(var_qubits, clause_qubits, output_qubit)
    
    # --- A. Compute Clauses (XOR) ---
    # We use CNOTs to compute parity. If q_i != q_j, then target becomes 1.
    
    # Clause 0 (Row 0): q0 != q1
    qc.cx(var_qubits[0], clause_qubits[0])
    qc.cx(var_qubits[1], clause_qubits[0])
    
    # Clause 1 (Row 1): q2 != q3
    qc.cx(var_qubits[2], clause_qubits[1])
    qc.cx(var_qubits[3], clause_qubits[1])
    
    # Clause 2 (Col 0): q0 != q2
    qc.cx(var_qubits[0], clause_qubits[2])
    qc.cx(var_qubits[2], clause_qubits[2])
    
    # Clause 3 (Col 1): q1 != q3
    qc.cx(var_qubits[1], clause_qubits[3])
    qc.cx(var_qubits[3], clause_qubits[3])
    
    # --- B. Check (Phase Kickback) ---
    # Flip the output qubit (state |->) ONLY if ALL clause qubits are 1
    qc.mcx(clause_qubits, output_qubit)
    
    # --- C. Uncompute (Clean up) ---
    # Reverse step A to reset clause_qubits to |0>. 
    # This is crucial to maintain interference!
    qc.cx(var_qubits[1], clause_qubits[3])
    qc.cx(var_qubits[3], clause_qubits[3])
    
    qc.cx(var_qubits[0], clause_qubits[2])
    qc.cx(var_qubits[2], clause_qubits[2])
    
    qc.cx(var_qubits[2], clause_qubits[1])
    qc.cx(var_qubits[3], clause_qubits[1])
    
    qc.cx(var_qubits[0], clause_qubits[0])
    qc.cx(var_qubits[1], clause_qubits[0])
    
    return qc

def create_diffuser(n):
    """Standard Grover Diffusion Operator D = 2|s><s| - I"""
    qc = QuantumCircuit(n)
    qc.h(range(n))
    qc.x(range(n))
    
    # Multi-controlled Z
    qc.h(n-1)
    qc.mcx(list(range(n-1)), n-1)
    qc.h(n-1)
    
    qc.x(range(n))
    qc.h(range(n))
    return qc

# ==========================================
# 2. Main Algorithm
# ==========================================

def solve_sudoku():
    # --- Setup Registers ---
    var_q = QuantumRegister(4, name='v')    # Variable qubits (The Grid)
    clause_q = QuantumRegister(4, name='c') # Clause checking qubits
    out_q = QuantumRegister(1, name='out')  # Phase kickback qubit
    c_bits = ClassicalRegister(4, name='measure')
    
    qc = QuantumCircuit(var_q, clause_q, out_q, c_bits)
    
    # --- Initialization ---
    # 1. Variables in uniform superposition
    qc.h(var_q)
    # 2. Output qubit in |-> state (for phase kickback)
    qc.x(out_q)
    qc.h(out_q)
    
    # --- Grover Iterations ---
    # N = 16, k = 2 (Solutions: 0110, 1001). 
    # Optimal Iterations ~ pi/4 * sqrt(16/2) = 2.22 -> 2 iterations
    oracle = create_sudoku_oracle(var_q, clause_q, out_q)
    diffuser = create_diffuser(4)
    
    for _ in range(2):
        qc.append(oracle, list(range(9))) # Append oracle to all qubits
        qc.append(diffuser, list(range(4))) # Append diffuser to variable qubits
        
    # --- Measurement ---
    qc.measure(var_q, c_bits)
    
    # --- Simulation ---
    backend = AerSimulator()

    compiled = transpile(qc, backend, seed_transpiler=0)
    job = backend.run(compiled, shots=1024, seed_simulator=0)

    result = job.result()
    counts = result.get_counts()
    return counts

# ==========================================
# 3. Visualization (The "Readable" Part)
# ==========================================

def plot_sudoku_solutions(counts):
    """
    Parses bitstrings and plots the top solutions as 2x2 grids.
    """
    # Sort counts to find the most probable states
    sorted_counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)
    
    # Extract top 2 candidates
    top_candidates = sorted_counts[:2]
    
    # Configuration for plotting
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    
    for i, (bitstring, count) in enumerate(top_candidates):
        ax = axes[i]
        
        # Qiskit bitstrings are little-endian (reversed), so we reverse it back
        # to map v0, v1, v2, v3 correctly
        bits = bitstring[::-1] 
        grid = np.array([
            [int(bits[0]), int(bits[1])],
            [int(bits[2]), int(bits[3])]
        ])
        
        # Draw the Grid
        ax.matshow(grid, cmap='Blues', vmin=0, vmax=1)
        
        # Add text (0 or 1) to the center of squares
        for (r, c), val in np.ndenumerate(grid):
            ax.text(c, r, str(val), va='center', ha='center', fontsize=25, color='black')
            
        # Styling
        ax.set_title(f"Solution {i+1}\n(Probability: {count/1024:.2%})", fontsize=12)
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Add grid lines manually for visual clarity
        ax.axhline(0.5, color='black', linewidth=2)
        ax.axvline(0.5, color='black', linewidth=2)

    plt.tight_layout()
    plt.savefig("sudoku_solutions.pdf", bbox_inches='tight')
    plt.show()

# ==========================================
# 4. Execution
# ==========================================

if __name__ == "__main__":
    print("Running 2x2 Binary Sudoku Solver...")
    counts = solve_sudoku()
    print("Raw Counts:", counts)
    plot_sudoku_solutions(counts)