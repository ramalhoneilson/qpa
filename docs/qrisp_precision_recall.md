# Precision and Recall Evaluation — Qrisp

## Setup

- **Target project**: `target_github_projects/Qrisp`
- **Ground truth**: `data/qrisp_ground_truth.csv` (145 files, 17 patterns)
- **Predictor**: `run_analysis.py --target-dir`
- **Resolution**: highest-scoring prediction per file; ties broken alphabetically by pattern.

---

## Overall Results

| Metric | Value |
|---|---|
| GT files | 145 |
| Files with at least one prediction | 29 (20 %) |
| Files with no prediction (missed) | 116 (80 %) |
| Correct predictions (TP) | 5 |
| Wrong predictions (FP) | 24 |
| **Micro Precision** | **0.172** |
| **Micro Recall** | **0.034** |
| **Micro F1** | **0.057** |
| Macro Precision | 0.393 |
| Macro Recall | 0.044 |
| Macro F1 | 0.181 |

---

## Per-Pattern Results

| Pattern | GT | Predicted | TP | FP | FN | Precision | Recall | F1 |
|---|---|---|---|---|---|---|---|---|
| Quantum Arithmetic | 34 | 1 | 1 | 0 | 33 | 1.000 | 0.029 | 0.057 |
| Quantum Approximate Optimization Algorithm (QAOA) | 29 | 1 | 1 | 0 | 28 | 1.000 | 0.034 | 0.067 |
| Quantum Logical Operators | 20 | 0 | 0 | 0 | 20 | — | 0.000 | — |
| Linear Combination of Unitaries | 12 | 0 | 0 | 0 | 12 | — | 0.000 | — |
| Hamiltonian Simulation | 9 | 0 | 0 | 0 | 9 | — | 0.000 | — |
| Circuit Construction Utility | 7 | 3 | 1 | 2 | 6 | 0.333 | 0.143 | 0.200 |
| Initialization | 5 | 1 | 1 | 0 | 4 | 1.000 | 0.200 | 0.333 |
| Quantum Amplitude Estimation | 5 | 0 | 0 | 0 | 5 | — | 0.000 | — |
| Variational Quantum Algorithm (VQA) | 5 | 0 | 0 | 0 | 5 | — | 0.000 | — |
| Quantum Phase Estimation (QPE) | 4 | 5 | 0 | 5 | 4 | 0.000 | 0.000 | — |
| Variational Quantum Eigensolver (VQE) | 4 | 0 | 0 | 0 | 4 | — | 0.000 | — |
| Domain Specific Application | 3 | 0 | 0 | 0 | 3 | — | 0.000 | — |
| Grover | 3 | 5 | 1 | 4 | 2 | 0.200 | 0.333 | 0.250 |
| Amplitude Amplification | 2 | 2 | 0 | 2 | 2 | 0.000 | 0.000 | — |
| Basis Change | 1 | 10 | 0 | 10 | 1 | 0.000 | 0.000 | — |
| Creating Entanglement | 1 | 0 | 0 | 0 | 1 | — | 0.000 | — |
| Data Encoding | 1 | 1 | 0 | 1 | 1 | 0.000 | 0.000 | — |

---

## Correct Predictions

| File | Pattern |
|---|---|
| `src/qrisp/algorithms/grover/grover_tools.py` | Grover |
| `src/qrisp/alg_primitives/arithmetic/modular_arithmetic/modular_qft_addition.py` | Quantum Arithmetic |
| `src/qrisp/alg_primitives/iterable_processing.py` | Circuit Construction Utility |
| `src/qrisp/alg_primitives/state_preparation/qiskit_state_preparation.py` | Initialization |
| `src/qrisp/examples/MaxCut_qaoa.py` | Quantum Approximate Optimization Algorithm (QAOA) |

---

## Wrong Predictions

| File | GT Pattern | Predicted Pattern |
|---|---|---|
| `src/qrisp/algorithms/cks.py` | Linear Combination of Unitaries | Grover |
| `src/qrisp/algorithms/gqsp/convolution.py` | Linear Combination of Unitaries | Basis Change |
| `src/qrisp/algorithms/gqsp/fourier_series_loader.py` | Data Encoding | Basis Change |
| `src/qrisp/algorithms/gqsp/gqet.py` | Linear Combination of Unitaries | Basis Change |
| `src/qrisp/algorithms/gqsp/hamiltonian_simulation.py` | Hamiltonian Simulation | Basis Change |
| `src/qrisp/algorithms/gqsp/qet.py` | Linear Combination of Unitaries | Grover |
| `src/qrisp/algorithms/hhl.py` | Linear Combination of Unitaries | Quantum Phase Estimation (QPE) |
| `src/qrisp/algorithms/lanczos.py` | Hamiltonian Simulation | Data Encoding |
| `src/qrisp/algorithms/qaoa/qaoa_benchmark_data.py` | Quantum Approximate Optimization Algorithm (QAOA) | Circuit Construction Utility |
| `src/qrisp/algorithms/qite.py` | Hamiltonian Simulation | Grover |
| `src/qrisp/algorithms/quantum_backtracking/backtracking_tree.py` | Grover | Basis Change |
| `src/qrisp/algorithms/quantum_counting.py` | Quantum Amplitude Estimation | Quantum Phase Estimation (QPE) |
| `src/qrisp/algorithms/shor/shors_algorithm.py` | Quantum Phase Estimation (QPE) | Basis Change |
| `src/qrisp/algorithms/vqe/vqe_benchmark_data.py` | Variational Quantum Eigensolver (VQE) | Circuit Construction Utility |
| `src/qrisp/alg_primitives/amplitude_amplification.py` | Amplitude Amplification | Grover |
| `src/qrisp/alg_primitives/arithmetic/modular_arithmetic/modular_qft_multiplication.py` | Quantum Arithmetic | Basis Change |
| `src/qrisp/alg_primitives/arithmetic/SBP_arithmetic.py` | Quantum Arithmetic | Basis Change |
| `src/qrisp/alg_primitives/iterative_qae.py` | Quantum Amplitude Estimation | Amplitude Amplification |
| `src/qrisp/alg_primitives/lcu.py` | Linear Combination of Unitaries | Amplitude Amplification |
| `src/qrisp/alg_primitives/qae.py` | Quantum Amplitude Estimation | Quantum Phase Estimation (QPE) |
| `src/qrisp/alg_primitives/qpe.py` | Quantum Phase Estimation (QPE) | Basis Change |
| `src/qrisp/examples/backtracking.py` | Domain Specific Application | Quantum Phase Estimation (QPE) |
| `src/qrisp/examples/diagonal_hamiltonian_application.py` | Hamiltonian Simulation | Basis Change |
| `src/qrisp/examples/tsp.py` | Domain Specific Application | Quantum Phase Estimation (QPE) |
