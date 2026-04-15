# Amazon Braket Algorithm Library — Ground Truth Coverage

Generated from: `target_github_projects/amazon-braket-algorithm-library`
Ground truth file: `data/knowledge_base/enriched_braket_algorithm_library_quantum_patterns.csv`

---

## Instance Count Summary

| Category | Count |
|---|---|
| Total named entries (functions + classes) | **79** |
| Distinct algorithm families | **16** |
| Distinct quantum patterns | **14** |
| Source notebooks | **19** |

Exceeds the 50-instance statistical threshold with comfortable margin.

---

## Algorithm Family Coverage

Each row names the algorithm family, the corresponding source module, and what quantum pattern(s) it exercises.

| # | Algorithm Family | Source Module | Patterns Exercised |
|---|---|---|---|
| 1 | **Grover's Search** | `grovers_search` | Grover, Oracle, Amplitude Amplification, Circuit Construction Utility |
| 2 | **Quantum Phase Estimation** | `quantum_phase_estimation` | Quantum Phase Estimation (QPE), Basis Change |
| 3 | **Quantum Fourier Transform** | `quantum_fourier_transform` | Basis Change |
| 4 | **QAOA** | `quantum_approximate_optimization` | Quantum Approximate Optimization Algorithm (QAOA) |
| 5 | **Shor's Algorithm** | `shors` | Domain Specific Application, Basis Change, Quantum Arithmetic |
| 6 | **Bernstein–Vazirani** | `bernstein_vazirani` | Oracle |
| 7 | **Deutsch–Jozsa** | `deutsch_jozsa` | Oracle |
| 8 | **HHL (Linear Systems)** | `hhl` | Domain Specific Application, Quantum Phase Estimation (QPE), Hamiltonian Simulation, Initialization |
| 9 | **Simon's Algorithm** | `simons` | Oracle |
| 10 | **Quantum Walk** | `quantum_walk` | Domain Specific Application, Quantum Arithmetic, Basis Change |
| 11 | **Quantum Counting** | `quantum_counting` | Quantum Amplitude Estimation, Amplitude Amplification, Oracle, Basis Change |
| 12 | **Quantum Circuit Born Machine (QCBM)** | `quantum_circuit_born_machine` | Variational Quantum Algorithm (VQA), Creating Entanglement |
| 13 | **Bell's Inequality** | `bells_inequality` | Creating Entanglement |
| 14 | **CHSH Inequality** | `chsh_inequality` | Creating Entanglement |
| 15 | **QC-QMC (Monte Carlo)** | `qc_qmc` | Domain Specific Application, Hamiltonian Simulation, Initialization, Quantum Amplitude Estimation |
| 16 | **Adaptive Shot Allocation** | `adaptive_shot_allocation` | Variational Quantum Algorithm (VQA) |

---

## Pattern Distribution

| Pattern | # Entries | Algorithm Families |
|---|---|---|
| Oracle | 15 | Grover, Bernstein–Vazirani, Deutsch–Jozsa, Simon's, Quantum Counting |
| Basis Change | 8 | QFT, Inverse QFT, Shor's, QPE, Quantum Counting, Quantum Walk |
| Domain Specific Application | 8 | Shor's, HHL, Quantum Walk, QC-QMC |
| Creating Entanglement | 8 | Bell's Inequality, CHSH, QCBM |
| Quantum Phase Estimation (QPE) | 7 | QPE, HHL |
| Quantum Approximate Optimization Algorithm (QAOA) | 6 | QAOA |
| Amplitude Amplification | 4 | Grover, Quantum Counting |
| Variational Quantum Algorithm (VQA) | 4 | QCBM, Adaptive Shot |
| Quantum Amplitude Estimation | 6 | Quantum Counting, QC-QMC |
| Hamiltonian Simulation | 3 | HHL, QC-QMC |
| Circuit Construction Utility | 3 | Grover helpers, Random Circuit |
| Quantum Arithmetic | 2 | Shor's, Quantum Walk |
| Initialization | 2 | HHL, QC-QMC |
| Grover | 1 | Grover's Search |

---

## Notebooks (Evaluation Corpus)

These are the actual files that will be analysed by the tool. The ground truth for each notebook is derived from the functions it calls.

| Notebook | Algorithm Family | Expected Patterns |
|---|---|---|
| `textbook/Grovers_Search.ipynb` | Grover's Search | Grover, Oracle, Amplitude Amplification |
| `textbook/Quantum_Phase_Estimation_Algorithm.ipynb` | QPE | Quantum Phase Estimation (QPE), Basis Change |
| `textbook/Quantum_Fourier_Transform.ipynb` | QFT | Basis Change |
| `textbook/Quantum_Approximate_Optimization_Algorithm.ipynb` | QAOA | Quantum Approximate Optimization Algorithm (QAOA) |
| `textbook/Shors_Algorithm.ipynb` | Shor's | Domain Specific Application, Basis Change, Quantum Arithmetic |
| `textbook/Bernstein_Vazirani_Algorithm.ipynb` | Bernstein–Vazirani | Oracle |
| `textbook/Deutsch_Jozsa_Algorithm.ipynb` | Deutsch–Jozsa | Oracle |
| `textbook/Simons_Algorithm.ipynb` | Simon's | Oracle |
| `textbook/Quantum_Walk.ipynb` | Quantum Walk | Domain Specific Application, Quantum Arithmetic |
| `textbook/Quantum_Counting_Algorithm.ipynb` | Quantum Counting | Quantum Amplitude Estimation, Amplitude Amplification |
| `textbook/Quantum_Circuit_Born_Machine.ipynb` | QCBM | Variational Quantum Algorithm (VQA), Creating Entanglement |
| `textbook/Bells_Inequality.ipynb` | Bell's Inequality | Creating Entanglement |
| `textbook/CHSH_Inequality.ipynb` | CHSH Inequality | Creating Entanglement |
| `advanced_algorithms/HHL_Algorithm.ipynb` | HHL | Domain Specific Application, Quantum Phase Estimation (QPE), Hamiltonian Simulation |
| `advanced_algorithms/Quantum_Computing_Quantum_Monte_Carlo.ipynb` | QC-QMC | Domain Specific Application, Hamiltonian Simulation |
| `advanced_algorithms/Quantum_Principal_Component_Analysis.ipynb` | QPCA | Quantum Phase Estimation (QPE), Domain Specific Application |
| `advanced_algorithms/adaptive_shot_allocation/1_Shot_Allocation.ipynb` | Adaptive Shot | Variational Quantum Algorithm (VQA) |
| `advanced_algorithms/adaptive_shot_allocation/2_Adaptive_Shot_Allocation.ipynb` | Adaptive Shot | Variational Quantum Algorithm (VQA) |
| `auxiliary_functions/Random_Circuit.ipynb` | Random Circuit | Circuit Construction Utility |

---

## Coverage Assessment

### Strengths (vs. PennyLane qml concern)

- **Textbook algorithms well-represented**: Grover, QPE, QFT, Shor's, BV, DJ, Simon's — 7 distinct oracle/transform-based algorithm families
- **No QML bias**: Zero quantum machine learning notebooks (no QNN, variational classifiers, transfer learning)
- **QAOA present but balanced**: 1 QAOA notebook out of 19 total (5%)
- **Advanced algorithms**: HHL, QMC, QPCA — rarely covered in other evaluation sets

### Limitations

- **No VQE**: Adaptive shot allocation is VQE-adjacent but no standalone VQE notebook
- **No Data Encoding**: No amplitude/angle embedding instances
- **Amazon Braket API specificity**: Notebooks use `braket.Circuit` rather than Qiskit/PennyLane. The tool must rely on semantic (comment-based) matching more than name matching
- **Library code risk (partial)**: The `src/` Python modules define algorithms from scratch — these will not name-match KB functions. The notebooks that *call* these functions are the valid evaluation targets.

### Recommendation

Evaluate on the **19 notebooks** (not on `src/` directly). The notebooks call the library functions and also contain explanatory comments. Both name-matching (braket circuit primitives like `Circuit().h()`, `Circuit().cnot()`) and summary-matching (comment blocks describing the algorithm) should fire against the KB.
