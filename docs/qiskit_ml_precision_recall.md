# Precision / Recall Evaluation — Qiskit Machine Learning

_Evaluation mode: **multi-label**_

## Summary

| Metric | Value |
|---|---|
| GT files | 13 |
| Files with prediction | 13 (100 %) |
| Files missed | 0 |
| Correct (TP) | 4 |
| Micro Precision | 0.308 |
| Micro Recall | 0.308 |
| Micro F1 | 0.308 |
| **Observed Macro Recall** | **0.452** |
| **Observed Macro Precision** | **0.595** |
| **Observed Macro F1** | **0.439** |

## Per-Pattern Results

_Multi-label mode: TP = GT pattern detected in ANY prediction for the file. FP = pattern predicted for a file whose GT is different. Observed recall = TP / (TP + FN) over patterns present in GT._

| Pattern | GT | Pred | TP | FP | FN | Precision | Recall | F1 |
|---|---|---|---|---|---|---|---|---|
| Quantum Neural Network (QNN) | 7 | 1 | 1 | 0 | 6 | 1.000 | 0.143 | 0.250 |
| Variational Quantum Algorithm (VQA) | 3 | 7 | 2 | 5 | 1 | 0.286 | 0.667 | 0.400 |
| SWAP Test | 2 | 0 | 0 | 0 | 2 |   —   | 0.000 |   —   |
| Domain Specific Application | 1 | 2 | 1 | 1 | 0 | 0.500 | 1.000 | 0.667 |

## File-Level Predictions

| File | GT Pattern | All Detected Patterns | Top Prediction | Match |
|---|---|---|---|---|
| `01_neural_networks.py` | Quantum Neural Network (QNN) | Quantum Approximate Optimization Algorithm (QAOA), Quantum Neural Network (QNN) | Quantum Neural Network (QNN) | ✓ |
| `02a_training_a_quantum_model_on_a_real_dataset.py` | Variational Quantum Algorithm (VQA) | Data Encoding, Variational Quantum Algorithm (VQA) | Data Encoding | ✓ |
| `02_neural_network_classifier_and_regressor.py` | Quantum Neural Network (QNN) | Data Encoding, Quantum Approximate Optimization Algorithm (QAOA), Variational Quantum Algorithm (VQA) | Data Encoding | ✗ |
| `03_quantum_kernel.py` | SWAP Test | Data Encoding, Uncompute, Variational Quantum Eigensolver (VQE) | Data Encoding | ✗ |
| `04_torch_qgan.py` | Quantum Neural Network (QNN) | Variational Quantum Algorithm (VQA) | Variational Quantum Algorithm (VQA) | ✗ |
| `05_torch_connector.py` | Quantum Neural Network (QNN) | Data Encoding, Variational Quantum Algorithm (VQA) | Data Encoding | ✗ |
| `07_pegasos_qsvc.py` | SWAP Test | Data Encoding, Uncompute | Data Encoding | ✗ |
| `08_quantum_kernel_trainer.py` | Variational Quantum Algorithm (VQA) | Data Encoding, Quantum Phase Estimation (QPE), Uncompute | Data Encoding | ✗ |
| `09_saving_and_loading_models.py` | Variational Quantum Algorithm (VQA) | Variational Quantum Algorithm (VQA), Variational Quantum Eigensolver (VQE) | Variational Quantum Algorithm (VQA) | ✓ |
| `10_effective_dimension.py` | Quantum Neural Network (QNN) | Data Encoding, Quantum Phase Estimation (QPE), Variational Quantum Algorithm (VQA) | Data Encoding | ✗ |
| `11_quantum_convolutional_neural_networks.py` | Quantum Neural Network (QNN) | Data Encoding, Quantum Phase Estimation (QPE) | Data Encoding | ✗ |
| `12_quantum_autoencoder.py` | Quantum Neural Network (QNN) | Domain Specific Application, Quantum Phase Estimation (QPE), Variational Quantum Algorithm (VQA), Variational Quantum Eigensolver (VQE) | Domain Specific Application | ✗ |
| `13_quantum_bayesian_inference.py` | Domain Specific Application | Domain Specific Application | Domain Specific Application | ✓ |
