# Grover Algorithm Experiments — Simulation & Applications (COS 521)

This repository contains the full implementation of the experimental components (Sections 5 and 6) from our COS 521 final report *“Grover’s Algorithm and Quantum Search.”*  
It includes numerical simulations of Grover’s algorithm under different settings and a practical application to solving a binary Sudoku constraint satisfaction problem.

All experiments were implemented using Python and Qiskit.

---

## 📌 Contents Overview

This project contains four major experiment modules, directly corresponding to Sections **5.1–5.3** and **Section 6** of the report.

### **1. Single-Solution Grover Search (k = 1)**
Implements Grover’s algorithm for the case of a unique marked item (M = 1).  
Experiments validate:

- sinusoidal amplitude amplification  
- optimal iteration count \( t_{\text{opt}} \approx \frac{\pi}{4}\sqrt{N} \)  
- over-rotation effect when iterating past the optimal step  
- linear scaling of \( t_{\text{opt}} \) vs. \( \sqrt{N} \), confirming the \(O(\sqrt{N})\) query complexity  

**Related script:** `grover_unik.py`  
**Figures generated:** probability dynamics, complexity scaling plots.

---

### **2. Multi-Solution Grover Search (k > 1)**
Extends the experiments to multiple marked items.  
Simulations reproduce the theoretical scaling:

\[
t_{\text{opt}} \approx \frac{\pi}{4}\sqrt{N/M}.
\]

Observed behaviors include:

- faster convergence as M increases  
- shortening of oscillation period  
- uniform sampling across all marked states  

**Related scripts:** `grover_multik.py`, `grover_multisol_combined.py`  
**Figures generated:** multi-k probability curves, distribution histograms.

---

### **3. Noise Robustness Analysis**
Evaluates Grover’s sensitivity to depolarizing noise on all gates.

Key experimental findings include:

- exponential decay of success probability in noisy circuits  
- meaningful quantum advantage only when per-gate error rate \(p < 0.0015\)  
- demonstration of NISQ limitations due to circuit depth of oracle + diffusion operators  

**Related script:** `grover_noise_robustness.py`  
**Figure generated:** success probability vs. depolarizing noise level.

---

### **4. Application: Solving a 2×2 Binary Sudoku**
Implements Grover’s algorithm for a genuine constraint-satisfaction problem (CSP).  
A compute–check–uncompute oracle is built using:

- XOR checks for row/column constraints  
- auxiliary scratch qubits  
- multi-controlled Toffoli (MCX) gate for phase kickback  

The algorithm correctly identifies the two valid Sudoku grids:

- |1001⟩  
- |0110⟩  

Achieving a combined probability > 93% after only 2 Grover iterations.

**Related script:** `sudoku_solutions.py`  
**Figure generated:** measurement histogram of valid Sudoku solutions.

---

## 📁 Repository Structure
