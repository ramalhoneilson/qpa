# Precision / Recall Evaluation — Braket Algorithm Library & Qiskit Machine Learning

Generalisation evaluation of `run_analysis.py` on two held-out frameworks.
The knowledge base contains **only** the 4 reference frameworks (Classiq, PennyLane, Qiskit, Qiskit-Algorithms).
Neither Amazon Braket nor Qiskit Machine Learning appears in the KB — they are unseen test corpora.

---

## Setup

| Item | Braket | Qiskit-ML |
|---|---|---|
| Evaluation corpus | `data/eval_notebooks/braket_algorithm_library/` | `data/eval_notebooks/qiskit_machine_learning/` |
| Ground truth | `data/braket_algorithm_library_ground_truth.csv` | `data/qiskit_machine_learning_ground_truth.csv` |
| Analysis output | `data/braket_algorithm_library_eval_output.csv` | `data/qiskit_machine_learning_eval_output.csv` |
| GT files | 19 | 13 |
| Distinct GT patterns | 10 | 4 |
| Evaluation mode | **single-label** | **multi-label** |
| KB frameworks | Classiq, PennyLane, Qiskit, Qiskit-Algorithms | same |
| Notebook source | Converted `.ipynb` → `.py` via `nbconvert` | Same |

**Evaluation modes:**
- **Single-label** (Braket): one prediction per file — highest-scoring match, ties broken alphabetically. Correct if the predicted pattern equals the GT pattern.
- **Multi-label** (Qiskit-ML): all above-threshold predictions are retained. A file is a TP if the GT pattern appears in ANY of its predictions. Appropriate for QML notebooks where data encoding, neural network, and variational patterns co-occur in every tutorial.

**Observed recall** = macro recall averaged only over patterns that have at least one GT positive in the framework. Patterns absent from a framework's GT are excluded from the denominator — specialisation does not penalise recall.

---

## Cross-Framework Summary

| Framework | Mode | Files | Coverage | Micro-P | Micro-R | Micro-F1 | Obs. Recall | Obs. Precision | Obs. F1 |
|---|---|---|---|---|---|---|---|---|---|
| Amazon Braket Algorithm Library | single-label | 19 | 32 % | 0.833 | 0.263 | 0.400 | **0.450** | **0.900** | **0.900** |
| Qiskit Machine Learning | multi-label | 13 | 92 % | 0.250 | 0.231 | 0.240 | **0.202** | **0.643** | **0.325** |

---

## Amazon Braket Algorithm Library — Detailed Results

### Per-Pattern

| Pattern | GT | Pred | TP | FP | FN | Precision | Recall | F1 |
|---|---|---|---|---|---|---|---|---|
| Domain Specific Application | 4 | 0 | 0 | 0 | 4 | — | 0.000 | — |
| Oracle | 3 | 0 | 0 | 0 | 3 | — | 0.000 | — |
| Variational Quantum Algorithm (VQA) | 3 | 0 | 0 | 0 | 3 | — | 0.000 | — |
| Creating Entanglement | 2 | 0 | 0 | 0 | 2 | — | 0.000 | — |
| Quantum Phase Estimation (QPE) | 2 | 2 | 1 | 1 | 1 | 0.500 | 0.500 | 0.500 |
| Basis Change | 1 | 1 | 1 | 0 | 0 | 1.000 | 1.000 | 1.000 |
| Circuit Construction Utility | 1 | 1 | 1 | 0 | 0 | 1.000 | 1.000 | 1.000 |
| Grover | 1 | 1 | 1 | 0 | 0 | 1.000 | 1.000 | 1.000 |
| Quantum Amplitude Estimation | 1 | 0 | 0 | 0 | 1 | — | 0.000 | — |
| QAOA | 1 | 1 | 1 | 0 | 0 | 1.000 | 1.000 | 1.000 |

### File-Level Results

| File | GT Pattern | Prediction | Match |
|---|---|---|---|
| `Grovers_Search.py` | Grover | Grover | ✓ |
| `Quantum_Fourier_Transform.py` | Basis Change | Basis Change | ✓ |
| `Quantum_Approximate_Optimization_Algorithm.py` | QAOA | QAOA | ✓ |
| `Quantum_Phase_Estimation_Algorithm.py` | QPE | QPE | ✓ |
| `Random_Circuit.py` | Circuit Construction Utility | CCU | ✓ |
| `Quantum_Circuit_Born_Machine.py` | VQA | — (no match) | ✗ |
| `HHL_Algorithm.py` | Domain Specific Application | QPE (summary FP) | ✗ |
| `Quantum_Principal_Component_Analysis.py` | QPE | QNN (summary FP) | ✗ |
| `Shors_Algorithm.py` | Domain Specific Application | — | ✗ |
| `Bernstein_Vazirani_Algorithm.py` | Oracle | — | ✗ |
| `Deutsch_Jozsa_Algorithm.py` | Oracle | — | ✗ |
| `Simons_Algorithm.py` | Oracle | — | ✗ |
| `Quantum_Walk.py` | Domain Specific Application | — | ✗ |
| `Quantum_Counting_Algorithm.py` | Quantum Amplitude Estimation | — | ✗ |
| `Bells_Inequality.py` | Creating Entanglement | — | ✗ |
| `CHSH_Inequality.py` | Creating Entanglement | — | ✗ |
| `Quantum_Computing_Quantum_Monte_Carlo.py` | Domain Specific Application | — | ✗ |
| `1_Shot_Allocation.py` | VQA | — | ✗ |
| `2_Adaptive_Shot_Allocation.py` | VQA | — | ✗ |

### Braket Failure Analysis

**Patterns with recall = 0.000 (vocabulary gap):**
The 4-framework KB contains no function names matching braket-specific implementations:
- **Oracle** (BV, DJ, Simon's): `bernstein_vazirani_circuit`, `deutsch_jozsa`, `simons_oracle` — absent from KB
- **Creating Entanglement** (Bell's, CHSH): `bell_singlet`, `create_bell_inequality_circuits` — absent from KB
- **VQA** (Shot Allocation): `AdaptiveShotAllocator`, `QCBM` — not in any of the 4 KB frameworks
- **Domain Specific Application** (Shor's, Walk, QMC): braket-specific top-level names, no KB counterpart
- **Quantum Amplitude Estimation** (Quantum Counting): `quantum_counting_circuit` — absent from KB

**Patterns with recall > 0 (cross-framework name overlap):**
These succeed because the braket notebooks happen to call functions whose names appear in the Classiq/PennyLane/Qiskit KB:
- `grovers_search` → matches Classiq `grover_search` (0.962)
- `QFT` / `qft` → matches multiple KB entries (1.000)
- `QAOA` → matches Qiskit-Algorithms `QAOA` (1.000)
- `QuantumPhaseEstimation` → matches PennyLane `QuantumPhaseEstimation` (0.920)
- `random_iqp` → summary match → Circuit Construction Utility

**False positives:**
- `HHL_Algorithm.py` → predicted QPE: HHL markdown extensively describes QPE as a subroutine; summary matching fires on this description
- `Quantum_Principal_Component_Analysis.py` → predicted QNN: summary match to NeuralNetworkClassifier at 0.660

---

## Qiskit Machine Learning — Detailed Results (Multi-Label)

### Per-Pattern

| Pattern | GT | Pred | TP | FP | FN | Precision | Recall | F1 |
|---|---|---|---|---|---|---|---|---|
| Quantum Neural Network (QNN) | 7 | 1 | 1 | 0 | 6 | 1.000 | 0.143 | 0.250 |
| Variational Quantum Algorithm (VQA) | 3 | 7 | 2 | 5 | 1 | 0.286 | 0.667 | 0.400 |
| SWAP Test | 2 | 0 | 0 | 0 | 2 | — | 0.000 | — |
| Domain Specific Application | 1 | 0 | 0 | 0 | 1 | — | 0.000 | — |

### Qiskit-ML Failure Analysis

**QNN recall = 0.143 (only 1/7):**
QML-specific class names (`EstimatorQNN`, `SamplerQNN`, `NeuralNetworkClassifier`, `TorchConnector`) are not in the 4-framework KB. The single TP comes from a summary or cross-framework match rather than a direct name hit.

**VQA recall = 0.667 (2/3):**
`VQC` and `VQR` partially overlap with PennyLane/Qiskit-Algorithms variational circuit vocabulary, enabling 2 correct detections. The FPs arise from `zz_feature_map`-adjacent PennyLane template names matching VQA summaries.

**SWAP Test recall = 0.000:**
`FidelityQuantumKernel`, `QSVC`, `PegasosQSVC` — entirely qiskit-ml-specific, no KB counterpart.

**Domain Specific Application recall = 0.000:**
`QBayesian` — qiskit-ml-specific, no KB counterpart.

---

## Key Findings

### Finding 1: The Tool Generalises for Patterns With Cross-Framework Name Overlap
Patterns whose braket/qiskit-ml names semantically overlap with KB names are detected correctly and with high precision (0.900 for braket). Grover, QFT, QAOA, QPE are standard enough that multiple KB frameworks name them similarly.

### Finding 2: Framework-Specific Vocabulary is the Primary Recall Bottleneck
Patterns implemented with proprietary function names that don't appear in any of the 4 KB frameworks produce zero detections. This is the dominant failure mode across both test corpora.

### Finding 3: Observed Precision is High When the Tool Fires
Braket Obs. Precision = 0.900; Qiskit-ML Obs. Precision = 0.643. When the KB does contain relevant vocabulary, predictions are mostly correct. The tool does not hallucinate patterns at high rates.

### Finding 4: Observed Recall Reflects the KB Vocabulary Coverage Directly
Braket patterns matched by cross-framework names: recall ≈ 1.0. Braket patterns only named in braket-specific vocabulary: recall = 0.0. The correlation is exact — the tool's recall is bounded by how much the target framework's vocabulary overlaps with the KB.

### Finding 5: QML Patterns Are Harder to Generalise Than Textbook Patterns
Qiskit-ML Obs. Recall = 0.202 vs. Braket Obs. Recall = 0.450. QML-specific classes (`EstimatorQNN`, `FidelityQuantumKernel`, `QBayesian`) have no cross-framework equivalents in a textbook-algorithm-focused KB. Textbook patterns (Grover, QFT, QAOA) are named consistently across frameworks.
