# Precision / Recall Evaluation — Amazon Braket Algorithm Library

_Evaluation mode: **single-label**_

## Summary

| Metric | Value |
|---|---|
| GT files | 19 |
| Files with prediction | 18 (95 %) |
| Files missed | 1 |
| Correct (TP) | 5 |
| Micro Precision | 0.278 |
| Micro Recall | 0.263 |
| Micro F1 | 0.270 |
| **Observed Macro Recall** | **0.375** |
| **Observed Macro Precision** | **0.660** |
| **Observed Macro F1** | **0.678** |

## Per-Pattern Results

_Single-label mode: only the highest-scoring prediction per file is evaluated. Observed recall = TP / (TP + FN) over patterns present in GT._

| Pattern | GT | Pred | TP | FP | FN | Precision | Recall | F1 |
|---|---|---|---|---|---|---|---|---|
| Domain Specific Application | 4 | 5 | 1 | 4 | 3 | 0.200 | 0.250 | 0.222 |
| Oracle | 3 | 0 | 0 | 0 | 3 |   —   | 0.000 |   —   |
| Variational Quantum Algorithm (VQA) | 3 | 0 | 0 | 0 | 3 |   —   | 0.000 |   —   |
| Creating Entanglement | 2 | 0 | 0 | 0 | 2 |   —   | 0.000 |   —   |
| Quantum Phase Estimation (QPE) | 2 | 10 | 1 | 9 | 1 | 0.100 | 0.500 | 0.167 |
| Basis Change | 1 | 1 | 1 | 0 | 0 | 1.000 | 1.000 | 1.000 |
| Circuit Construction Utility | 1 | 1 | 1 | 0 | 0 | 1.000 | 1.000 | 1.000 |
| Grover | 1 | 0 | 0 | 0 | 1 |   —   | 0.000 |   —   |
| Quantum Amplitude Estimation | 1 | 0 | 0 | 0 | 1 |   —   | 0.000 |   —   |
| Quantum Approximate Optimization Algorithm (QAOA) | 1 | 1 | 1 | 0 | 0 | 1.000 | 1.000 | 1.000 |

## File-Level Predictions

| File | GT Pattern | Predicted Pattern | Score | Match |
|---|---|---|---|---|
| `Grovers_Search.py` | Grover | Quantum Phase Estimation (QPE) | 1.0000 | 0 |
| `Quantum_Phase_Estimation_Algorithm.py` | Quantum Phase Estimation (QPE) | Quantum Phase Estimation (QPE) | 1.0000 | 1 |
| `Quantum_Fourier_Transform.py` | Basis Change | Basis Change | 1.0000 | 1 |
| `Quantum_Approximate_Optimization_Algorithm.py` | Quantum Approximate Optimization Algorithm (QAOA) | Quantum Approximate Optimization Algorithm (QAOA) | 1.0000 | 1 |
| `Shors_Algorithm.py` | Domain Specific Application | Quantum Phase Estimation (QPE) | 1.0000 | 0 |
| `Bernstein_Vazirani_Algorithm.py` | Oracle | Quantum Phase Estimation (QPE) | 1.0000 | 0 |
| `Deutsch_Jozsa_Algorithm.py` | Oracle | Quantum Phase Estimation (QPE) | 1.0000 | 0 |
| `Simons_Algorithm.py` | Oracle | Quantum Phase Estimation (QPE) | 1.0000 | 0 |
| `Quantum_Walk.py` | Domain Specific Application | Quantum Phase Estimation (QPE) | 1.0000 | 0 |
| `Quantum_Counting_Algorithm.py` | Quantum Amplitude Estimation | Domain Specific Application | 1.0000 | 0 |
| `Quantum_Circuit_Born_Machine.py` | Variational Quantum Algorithm (VQA) | Domain Specific Application | 1.0000 | 0 |
| `Bells_Inequality.py` | Creating Entanglement | Quantum Phase Estimation (QPE) | 1.0000 | 0 |
| `CHSH_Inequality.py` | Creating Entanglement | Domain Specific Application | 1.0000 | 0 |
| `HHL_Algorithm.py` | Domain Specific Application | Domain Specific Application | 1.0000 | 1 |
| `Quantum_Computing_Quantum_Monte_Carlo.py` | Domain Specific Application | Quantum Phase Estimation (QPE) | 1.0000 | 0 |
| `Quantum_Principal_Component_Analysis.py` | Quantum Phase Estimation (QPE) | Domain Specific Application | 1.0000 | 0 |
| `1_Shot_Allocation.py` | Variational Quantum Algorithm (VQA) | — | — |  |
| `2_Adaptive_Shot_Allocation.py` | Variational Quantum Algorithm (VQA) | Quantum Phase Estimation (QPE) | 1.0000 | 0 |
| `Random_Circuit.py` | Circuit Construction Utility | Circuit Construction Utility | 0.6616 | 1 |
