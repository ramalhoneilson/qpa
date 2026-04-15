# Precision and Recall Evaluation — Qualtran

## Setup

- **Target project**: `target_github_projects/Qualtran`
- **Ground truth**: `data/qualtran_ground_truth.csv` (168 files, 14 patterns)
- **Predictor**: `run_analysis.py --target-dir`
- **Resolution**: highest-scoring prediction per file; ties broken alphabetically by pattern.

---

## Overall Results

| Metric | Value |
|---|---|
| GT files | 168 |
| Files with at least one prediction | 31 (18 %) |
| Files with no prediction (missed) | 137 (82 %) |
| Correct predictions (TP) | 7 |
| Wrong predictions (FP) | 24 |
| **Micro Precision** | **0.226** |
| **Micro Recall** | **0.042** |
| **Micro F1** | **0.070** |
| Macro Precision | 0.342 |
| Macro Recall | 0.091 |
| Macro F1 | 0.193 |

---

## Per-Pattern Results

| Pattern | GT | Predicted | TP | FP | FN | Precision | Recall | F1 |
|---|---|---|---|---|---|---|---|---|
| Circuit Construction Utility | 40 | 1 | 0 | 1 | 40 | 0.000 | 0.000 | — |
| Quantum Arithmetic | 34 | 2 | 1 | 1 | 33 | 0.500 | 0.029 | 0.056 |
| Linear Combination of Unitaries | 30 | 0 | 0 | 0 | 30 | — | 0.000 | — |
| Initialization | 15 | 0 | 0 | 0 | 15 | — | 0.000 | — |
| Hamiltonian Simulation | 12 | 0 | 0 | 0 | 12 | — | 0.000 | — |
| Quantum Phase Estimation (QPE) | 9 | 1 | 1 | 0 | 8 | 1.000 | 0.111 | 0.200 |
| Quantum Logical Operators | 7 | 16 | 3 | 13 | 4 | 0.188 | 0.429 | 0.261 |
| Data Encoding | 5 | 5 | 1 | 4 | 4 | 0.200 | 0.200 | 0.200 |
| Basis Change | 4 | 0 | 0 | 0 | 4 | — | 0.000 | — |
| Domain Specific Application | 3 | 0 | 0 | 0 | 3 | — | 0.000 | — |
| SWAP Test | 3 | 0 | 0 | 0 | 3 | — | 0.000 | — |
| Amplitude Amplification | 2 | 6 | 1 | 5 | 1 | 0.167 | 0.500 | 0.250 |
| Quantum Amplitude Estimation | 2 | 0 | 0 | 0 | 2 | — | 0.000 | — |
| Uncompute | 2 | 0 | 0 | 0 | 2 | — | 0.000 | — |

---

## Correct Predictions

| File | Pattern |
|---|---|
| `qualtran/bloqs/cryptography/rsa/rsa_mod_exp.py` | Quantum Arithmetic |
| `qualtran/bloqs/data_loading/qrom.py` | Data Encoding |
| `qualtran/bloqs/hamiltonian_simulation/guided_hamiltonian.py` | Quantum Phase Estimation (QPE) |
| `qualtran/bloqs/mcmt/and_bloq.py` | Quantum Logical Operators |
| `qualtran/bloqs/mcmt/ctrl_spec_and.py` | Quantum Logical Operators |
| `qualtran/bloqs/mcmt/specialized_ctrl.py` | Quantum Logical Operators |
| `qualtran/bloqs/reflections/reflection_using_prepare.py` | Amplitude Amplification |

---

## Wrong Predictions

| File | GT Pattern | Predicted Pattern |
|---|---|---|
| `qualtran/bloqs/arithmetic/addition.py` | Quantum Arithmetic | Quantum Logical Operators |
| `qualtran/bloqs/arithmetic/comparison.py` | Quantum Arithmetic | Quantum Logical Operators |
| `qualtran/bloqs/arithmetic/controlled_addition.py` | Quantum Arithmetic | Quantum Logical Operators |
| `qualtran/bloqs/arithmetic/hamming_weight.py` | Quantum Arithmetic | Quantum Logical Operators |
| `qualtran/bloqs/arithmetic/permutation.py` | Quantum Arithmetic | Circuit Construction Utility |
| `qualtran/bloqs/basic_gates/rotation.py` | Circuit Construction Utility | Quantum Logical Operators |
| `qualtran/bloqs/block_encoding/chebyshev_polynomial.py` | Linear Combination of Unitaries | Amplitude Amplification |
| `qualtran/bloqs/bookkeeping/always.py` | Circuit Construction Utility | Quantum Logical Operators |
| `qualtran/bloqs/chemistry/df/double_factorization.py` | Hamiltonian Simulation | Amplitude Amplification |
| `qualtran/bloqs/chemistry/hubbard_model/qubitization/select_hubbard.py` | Linear Combination of Unitaries | Quantum Logical Operators |
| `qualtran/bloqs/chemistry/sf/single_factorization.py` | Hamiltonian Simulation | Amplitude Amplification |
| `qualtran/bloqs/chemistry/thc/prepare.py` | Linear Combination of Unitaries | Amplitude Amplification |
| `qualtran/bloqs/chemistry/trotter/grid_ham/potential.py` | Hamiltonian Simulation | Data Encoding |
| `qualtran/bloqs/mcmt/controlled_via_and.py` | Quantum Logical Operators | Quantum Arithmetic |
| `qualtran/bloqs/mod_arithmetic/mod_addition.py` | Quantum Arithmetic | Quantum Logical Operators |
| `qualtran/bloqs/mod_arithmetic/mod_division.py` | Quantum Arithmetic | Quantum Logical Operators |
| `qualtran/bloqs/mod_arithmetic/mod_subtraction.py` | Quantum Arithmetic | Quantum Logical Operators |
| `qualtran/bloqs/multiplexers/unary_iteration_bloq.py` | Circuit Construction Utility | Quantum Logical Operators |
| `qualtran/bloqs/optimization/k_xor_sat/kikuchi_adjacency_list.py` | Linear Combination of Unitaries | Quantum Logical Operators |
| `qualtran/bloqs/phase_estimation/lp_resource_state.py` | Quantum Phase Estimation (QPE) | Amplitude Amplification |
| `qualtran/bloqs/rotations/programmable_rotation_gate_array.py` | Circuit Construction Utility | Data Encoding |
| `qualtran/bloqs/state_preparation/prepare_uniform_superposition.py` | Initialization | Quantum Logical Operators |
| `qualtran/bloqs/state_preparation/state_preparation_alias_sampling.py` | Initialization | Data Encoding |
| `qualtran/bloqs/state_preparation/state_preparation_via_rotation.py` | Initialization | Data Encoding |
