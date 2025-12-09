# Grover’s Algorithm – Simulation and Applications (COS 521 Project Code)

This repository contains the code for **Sections 5 and 6** of my COS 521 final project report on Grover’s algorithm.  
It implements and reproduces all experiments on:

- **Sec. 5.1**: Single-solution Grover search and complexity scaling  
- **Sec. 5.2**: Multi-solution Grover search ($M > 1$)  
- **Sec. 5.3**: Robustness to depolarizing noise  
- **Sec. 6**: A constraint-satisfaction application to a 2×2 binary Sudoku puzzle  

All simulations are implemented in **Python** on noiseless and noisy simulators.

---

## Relation to the Report

- **Section 5.1 – Single-Solution Grover Search ($M = 1$)**  
  Implemented in `grover_unisol.py`.  
  The script:
  - Simulates Grover’s algorithm for different problem sizes $n \in \{4,6,8\}$;
  - Plots the success probability as a function of the number of iterations $t$;
  - Verifies that the optimal iteration scales as $t_{\mathrm{opt}} \approx \frac{\pi}{4}\sqrt{N}$ by plotting $t_{\mathrm{opt}}$ against $\sqrt{N}$.  
  Output figure: **`grover_unisol_comparison.pdf`**.

- **Section 5.2 – Multi-Solution Grover Search ($M > 1$)**  
  Implemented in `grover_multisol.py`.  
  The script has two parts:
  - **Left panel**: compares convergence for $M \in \{1,4,16\}$ at fixed $n = 10$  
    and shows that more marked items lead to faster convergence.
  - **Right panel**: runs Grover at $t_{\mathrm{opt}}$ for $M = 4$ and plots the  
    empirical distribution over all marked solutions, demonstrating almost uniform sampling.  
  Output figure: **`grover_multisol_combined.pdf`**.

- **Section 5.3 – Noise Robustness Analysis**  
  Implemented in `robustcheck.py`.  
  The script:
  - Fixes a single-solution instance with $n = 7$ (so $N = 128$);
  - Runs Grover’s algorithm at the theoretical optimal iteration $t_{\mathrm{opt}}$;
  - Adds depolarizing noise with varying gate error probability $p$ to both 1-qubit and 2-qubit gates;
  - Plots the success probability versus $p$, together with the classical random-guess baseline $1/N$.  
  Output figure: **`grover_noise_robustness.pdf`**.

- **Section 6 – Application to 2×2 Binary Sudoku**  
  Implemented in `sudoku.py`.  
  The script:
  - Encodes a 2×2 binary Sudoku with variables $v_0, v_1, v_2, v_3$ and four XOR constraints  
    (two rows and two columns require unequal bits);
  - Builds a **compute–check–uncompute** oracle: clause qubits store XOR results,  
    a multi-controlled phase-flip is applied when all constraints are satisfied, and then the  
    clause qubits are uncomputed back to $\lvert 0\rangle$;
  - Runs two Grover iterations (near the optimal value for $N=16, M=2$);
  - Measures the variable qubits and visualizes the two valid Sudoku grids as 2×2 images with their probabilities.  
  Output figure: **`sudoku_solutions.pdf`**.

---

## Repository Structure

```text
.
├── grover_unisol.py               # Sec 5.1: single-solution Grover + scaling plot
├── grover_unisol_comparison.pdf   # Figure for Sec 5.1
├── grover_multisol.py             # Sec 5.2: multi-solution Grover experiments
├── grover_multisol_combined.pdf   # Figure for Sec 5.2
├── grover_noise_robustness.py     # Sec 5.3: depolarizing-noise robustness
├── grover_noise_robustness.pdf    # Figure for Sec 5.3
├── sudoku.py                      # Sec 6: 2×2 binary Sudoku oracle + Grover search
├── sudoku_solutions.pdf           # Figure for Sec 6
└── README.md
