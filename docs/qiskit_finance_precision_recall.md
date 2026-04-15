# Precision and Recall Evaluation — qiskit-finance

## Setup

- **Target project**: `target_github_projects/qiskit-finance`
- **Ground truth**: `data/qiskit_finance_ground_truth.csv` (13 files, 4 patterns)
- **Predictor**: `run_analysis.py --target-dir`
- **Resolution**: highest-scoring prediction per file; ties broken alphabetically by pattern.

---

## Overall Results

| Metric | Value |
|---|---|
| GT files | 13 |
| Files with at least one prediction | 7 (54 %) |
| Files with no prediction (missed) | 6 (46 %) |
| Correct predictions (TP) | 4 |
| Wrong predictions (FP) | 3 |
| **Micro Precision** | **0.571** |
| **Micro Recall** | **0.308** |
| **Micro F1** | **0.400** |
| Macro Precision | 1.000 |
| Macro Recall | 0.250 |
| Macro F1 | 0.629 |

---

## Per-Pattern Results

| Pattern | GT | Predicted | TP | FP | FN | Precision | Recall | F1 |
|---|---|---|---|---|---|---|---|---|
| Data Encoding | 4 | 1 | 1 | 0 | 3 | 1.000 | 0.250 | 0.400 |
| Quantum Amplitude Estimation | 4 | 3 | 3 | 0 | 1 | 1.000 | 0.750 | 0.857 |
| Oracle | 3 | 0 | 0 | 0 | 3 | — | 0.000 | — |
| Domain Specific Application | 2 | 0 | 0 | 0 | 2 | — | 0.000 | — |

---

## Correct Predictions

| File | Pattern |
|---|---|
| `qiskit_finance/applications/estimation/european_call_delta.py` | Quantum Amplitude Estimation |
| `qiskit_finance/applications/estimation/european_call_pricing.py` | Quantum Amplitude Estimation |
| `qiskit_finance/applications/estimation/fixed_income_pricing.py` | Quantum Amplitude Estimation |
| `qiskit_finance/circuit/library/probability_distributions/gaussian_conditional_independence_model.py` | Data Encoding |

---

## Wrong Predictions

| File | GT Pattern | Predicted Pattern |
|---|---|---|
| `qiskit_finance/circuit/library/payoff_functions/european_call_delta_objective.py` | Oracle | Quantum Arithmetic |
| `qiskit_finance/circuit/library/probability_distributions/lognormal.py` | Data Encoding | Circuit Construction Utility |
| `qiskit_finance/circuit/library/probability_distributions/normal.py` | Data Encoding | Circuit Construction Utility |
