# QUANTUM CONCEPT ANALYSIS REPORT

## I. Overall Summary
- **Total Matches Found:** 751
- **Unique Files with Matches:** 313
- **Unique Concepts Matched:** 142
- **Total Patterns Defined:** 24
- **Total Patterns Found:** 23
- **Average Similarity Score:** 0.8839

## II. Match Type Breakdown

### Match Type Counts

| match_type   |   count |
|:-------------|--------:|
| name         |     475 |
| summary      |     276 |

### Average Score by Match Type

| match_type   |   similarity_score |
|:-------------|-------------------:|
| name         |             0.9948 |
| summary      |             0.6931 |

---

## III. Source Framework & Target Project Breakdown

### Matches by Source Framework

| framework         |   count |
|:------------------|--------:|
| classiq           |     331 |
| qiskit-algorithms |     194 |
| pennylane         |     119 |
| qiskit            |     107 |

### Matches by Target Project

| project                         |   count |
|:--------------------------------|--------:|
| classiq-library                 |     370 |
| Qualtran                        |      70 |
| qiskit-algorithms               |      43 |
| qiskit-finance                  |      38 |
| Qrisp                           |      33 |
| amazon-braket-algorithm-library |      22 |
| qiskit-machine-learning         |      22 |
| qiskit-optimization             |      17 |
| tensorcircuit-ng                |      17 |
| tensorcircuit                   |      16 |
| torchquantum                    |      11 |
| catalyst                        |      10 |
| ProjectQ                        |      10 |
| guppylang                       |       9 |
| covalent                        |       9 |
| qibo                            |       7 |
| qiskit-nature                   |       7 |
| qiskit-braket-provider          |       6 |
| Perceval                        |       6 |
| quimb                           |       5 |
| ReCirq                          |       4 |
| squlearn                        |       4 |
| mitiq                           |       3 |
| Cirq                            |       3 |
| piquasso                        |       2 |
| qiskit-addon-cutting            |       2 |
| thewalrus                       |       2 |
| client-superstaq                |       1 |
| OpenFermion                     |       1 |
| Pulser                          |       1 |

---

## IV. Cross-Framework Pattern Analysis

### Table 4.1: Source Pattern Analysis (Where patterns originate)

| pattern                                           |   Total Matches | Source Frameworks                             |
|:--------------------------------------------------|----------------:|:----------------------------------------------|
| Basis Change                                      |             142 | classiq, pennylane, qiskit                    |
| Quantum Phase Estimation (QPE)                    |              75 | classiq, pennylane, qiskit-algorithms         |
| Variational Quantum Eigensolver (VQE)             |              63 | pennylane, qiskit-algorithms                  |
| Domain Specific Application                       |              59 | classiq, pennylane                            |
| Quantum Approximate Optimization Algorithm (QAOA) |              51 | classiq, pennylane, qiskit, qiskit-algorithms |
| Variational Quantum Algorithm (VQA)               |              46 | classiq, pennylane, qiskit, qiskit-algorithms |
| Amplitude Amplification                           |              46 | classiq, qiskit-algorithms                    |
| Circuit Construction Utility                      |              41 | classiq, pennylane, qiskit                    |
| Data Encoding                                     |              38 | classiq, pennylane, qiskit                    |
| Quantum Arithmetic                                |              35 | classiq, pennylane, qiskit                    |
| Initialization                                    |              26 | classiq, pennylane, qiskit, qiskit-algorithms |
| Hamiltonian Simulation                            |              22 | classiq, pennylane, qiskit                    |
| Oracle                                            |              20 | classiq, qiskit, qiskit-algorithms            |
| Grover                                            |              19 | classiq, pennylane, qiskit, qiskit-algorithms |
| Quantum Neural Network (QNN)                      |              12 | pennylane                                     |
| Quantum Logical Operators                         |              12 | qiskit                                        |
| Uncompute                                         |              11 | qiskit-algorithms                             |
| Dynamic Circuit                                   |               7 | pennylane                                     |
| Linear Combination of Unitaries                   |               6 | classiq                                       |
| SWAP Test                                         |               6 | classiq                                       |
| Phase Shift                                       |               6 | classiq, pennylane                            |
| Creating Entanglement                             |               5 | classiq                                       |
| Quantum Amplitude Estimation                      |               1 | classiq                                       |

### Table 4.2: Adoption Pattern Analysis (Where patterns are used)

| pattern                                           |   Project Coverage | Found In Projects                                                                                                                                                                                                                                        |
|:--------------------------------------------------|-------------------:|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Quantum Approximate Optimization Algorithm (QAOA) |                 16 | ProjectQ, Qrisp, ReCirq, amazon-braket-algorithm-library, catalyst, classiq-library, covalent, qiskit-algorithms, qiskit-braket-provider, qiskit-finance, qiskit-machine-learning, qiskit-optimization, quimb, squlearn, tensorcircuit, tensorcircuit-ng |
| Variational Quantum Algorithm (VQA)               |                 14 | Perceval, ProjectQ, Pulser, classiq-library, covalent, qiskit-addon-cutting, qiskit-algorithms, qiskit-braket-provider, qiskit-finance, qiskit-nature, squlearn, tensorcircuit, tensorcircuit-ng, torchquantum                                           |
| Variational Quantum Eigensolver (VQE)             |                 14 | Perceval, ProjectQ, Qrisp, classiq-library, client-superstaq, mitiq, qiskit-algorithms, qiskit-braket-provider, qiskit-finance, qiskit-nature, qiskit-optimization, tensorcircuit, tensorcircuit-ng, torchquantum                                        |
| Circuit Construction Utility                      |                 10 | OpenFermion, Perceval, Qualtran, ReCirq, amazon-braket-algorithm-library, classiq-library, covalent, qiskit-braket-provider, qiskit-optimization, thewalrus                                                                                              |
| Amplitude Amplification                           |                  8 | Qrisp, Qualtran, amazon-braket-algorithm-library, classiq-library, guppylang, qibo, qiskit-algorithms, qiskit-finance                                                                                                                                    |
| Domain Specific Application                       |                  8 | Qrisp, Qualtran, amazon-braket-algorithm-library, classiq-library, guppylang, piquasso, qibo, qiskit-finance                                                                                                                                             |
| Data Encoding                                     |                  7 | Qrisp, Qualtran, catalyst, classiq-library, covalent, qiskit-machine-learning, squlearn                                                                                                                                                                  |
| Uncompute                                         |                  7 | ProjectQ, Qualtran, ReCirq, classiq-library, qiskit-addon-cutting, qiskit-algorithms, qiskit-machine-learning                                                                                                                                            |
| Quantum Phase Estimation (QPE)                    |                  7 | ProjectQ, Qrisp, Qualtran, amazon-braket-algorithm-library, classiq-library, guppylang, qiskit-finance                                                                                                                                                   |
| Grover                                            |                  7 | Perceval, Qrisp, amazon-braket-algorithm-library, catalyst, classiq-library, qiskit-algorithms, qiskit-optimization                                                                                                                                      |
| Basis Change                                      |                  6 | Qrisp, Qualtran, amazon-braket-algorithm-library, catalyst, classiq-library, qibo                                                                                                                                                                        |
| Quantum Neural Network (QNN)                      |                  5 | Cirq, classiq-library, qiskit-machine-learning, tensorcircuit, tensorcircuit-ng                                                                                                                                                                          |
| Quantum Arithmetic                                |                  5 | Qrisp, Qualtran, catalyst, classiq-library, qiskit-finance                                                                                                                                                                                               |
| Initialization                                    |                  5 | ProjectQ, Qualtran, classiq-library, covalent, qiskit-algorithms                                                                                                                                                                                         |
| Dynamic Circuit                                   |                  3 | Cirq, classiq-library, quimb                                                                                                                                                                                                                             |
| Oracle                                            |                  3 | classiq-library, qiskit-algorithms, qiskit-finance                                                                                                                                                                                                       |
| Quantum Logical Operators                         |                  2 | Qualtran, classiq-library                                                                                                                                                                                                                                |
| Hamiltonian Simulation                            |                  2 | classiq-library, qiskit-algorithms                                                                                                                                                                                                                       |
| Phase Shift                                       |                  2 | catalyst, classiq-library                                                                                                                                                                                                                                |
| Creating Entanglement                             |                  1 | classiq-library                                                                                                                                                                                                                                          |
| Linear Combination of Unitaries                   |                  1 | classiq-library                                                                                                                                                                                                                                          |
| Quantum Amplitude Estimation                      |                  1 | classiq-library                                                                                                                                                                                                                                          |
| SWAP Test                                         |                  1 | classiq-library                                                                                                                                                                                                                                          |

---

## V. Quantum Pattern Analysis

### Analysis of Newly Defined Patterns

Found **9** out of **9** newly defined patterns in the target projects.
| Pattern                         |   Matches |
|:--------------------------------|----------:|
| Basis Change                    |       142 |
| Domain Specific Application     |        59 |
| Circuit Construction Utility    |        41 |
| Data Encoding                   |        38 |
| Quantum Arithmetic              |        35 |
| Hamiltonian Simulation          |        22 |
| Quantum Logical Operators       |        12 |
| Linear Combination of Unitaries |         6 |
| Quantum Amplitude Estimation    |         1 |

### Patterns by Match Count (Overall)

| pattern                                           |   count |
|:--------------------------------------------------|--------:|
| Basis Change                                      |     142 |
| Quantum Phase Estimation (QPE)                    |      75 |
| Variational Quantum Eigensolver (VQE)             |      63 |
| Domain Specific Application                       |      59 |
| Quantum Approximate Optimization Algorithm (QAOA) |      51 |
| Amplitude Amplification                           |      46 |
| Variational Quantum Algorithm (VQA)               |      46 |
| Circuit Construction Utility                      |      41 |
| Data Encoding                                     |      38 |
| Quantum Arithmetic                                |      35 |
| Initialization                                    |      26 |
| Hamiltonian Simulation                            |      22 |
| Oracle                                            |      20 |
| Grover                                            |      19 |
| Quantum Neural Network (QNN)                      |      12 |
| Quantum Logical Operators                         |      12 |
| Uncompute                                         |      11 |
| Dynamic Circuit                                   |       7 |
| Phase Shift                                       |       6 |
| Linear Combination of Unitaries                   |       6 |
| SWAP Test                                         |       6 |
| Creating Entanglement                             |       5 |
| Quantum Amplitude Estimation                      |       1 |

### Average Score by Pattern

| pattern                                           |   similarity_score |
|:--------------------------------------------------|-------------------:|
| Dynamic Circuit                                   |             1      |
| Oracle                                            |             1      |
| Quantum Amplitude Estimation                      |             1      |
| SWAP Test                                         |             1      |
| Hamiltonian Simulation                            |             0.9842 |
| Basis Change                                      |             0.9711 |
| Creating Entanglement                             |             0.9648 |
| Initialization                                    |             0.9617 |
| Phase Shift                                       |             0.9539 |
| Linear Combination of Unitaries                   |             0.9472 |
| Circuit Construction Utility                      |             0.9452 |
| Quantum Logical Operators                         |             0.943  |
| Data Encoding                                     |             0.9065 |
| Amplitude Amplification                           |             0.8941 |
| Quantum Neural Network (QNN)                      |             0.8935 |
| Grover                                            |             0.8808 |
| Variational Quantum Algorithm (VQA)               |             0.8615 |
| Uncompute                                         |             0.8138 |
| Variational Quantum Eigensolver (VQE)             |             0.8056 |
| Quantum Arithmetic                                |             0.8019 |
| Domain Specific Application                       |             0.8008 |
| Quantum Approximate Optimization Algorithm (QAOA) |             0.7972 |
| Quantum Phase Estimation (QPE)                    |             0.7901 |

### All Patterns within each Source Framework (Sorted by Frequency)


#### Classiq

| pattern                                           |   count |
|:--------------------------------------------------|--------:|
| Basis Change                                      |      94 |
| Quantum Phase Estimation (QPE)                    |      47 |
| Domain Specific Application                       |      44 |
| Circuit Construction Utility                      |      24 |
| Amplitude Amplification                           |      22 |
| Initialization                                    |      20 |
| Hamiltonian Simulation                            |      17 |
| Quantum Arithmetic                                |      12 |
| Oracle                                            |      10 |
| Data Encoding                                     |       8 |
| Grover                                            |       7 |
| Linear Combination of Unitaries                   |       6 |
| SWAP Test                                         |       6 |
| Creating Entanglement                             |       5 |
| Phase Shift                                       |       5 |
| Variational Quantum Algorithm (VQA)               |       2 |
| Quantum Amplitude Estimation                      |       1 |
| Quantum Approximate Optimization Algorithm (QAOA) |       1 |

#### Pennylane

| pattern                                           |   count |
|:--------------------------------------------------|--------:|
| Basis Change                                      |      25 |
| Domain Specific Application                       |      15 |
| Data Encoding                                     |      14 |
| Quantum Neural Network (QNN)                      |      12 |
| Variational Quantum Algorithm (VQA)               |      12 |
| Quantum Approximate Optimization Algorithm (QAOA) |       9 |
| Dynamic Circuit                                   |       7 |
| Quantum Arithmetic                                |       6 |
| Grover                                            |       4 |
| Hamiltonian Simulation                            |       4 |
| Circuit Construction Utility                      |       3 |
| Variational Quantum Eigensolver (VQE)             |       3 |
| Initialization                                    |       1 |
| Phase Shift                                       |       1 |
| Quantum Phase Estimation (QPE)                    |       1 |

#### Qiskit

| pattern                                           |   count |
|:--------------------------------------------------|--------:|
| Basis Change                                      |      23 |
| Quantum Arithmetic                                |      17 |
| Data Encoding                                     |      16 |
| Circuit Construction Utility                      |      14 |
| Quantum Logical Operators                         |      12 |
| Variational Quantum Algorithm (VQA)               |       9 |
| Quantum Approximate Optimization Algorithm (QAOA) |       6 |
| Initialization                                    |       4 |
| Oracle                                            |       3 |
| Grover                                            |       2 |
| Hamiltonian Simulation                            |       1 |

#### Qiskit-algorithms

| pattern                                           |   count |
|:--------------------------------------------------|--------:|
| Variational Quantum Eigensolver (VQE)             |      60 |
| Quantum Approximate Optimization Algorithm (QAOA) |      35 |
| Quantum Phase Estimation (QPE)                    |      27 |
| Amplitude Amplification                           |      24 |
| Variational Quantum Algorithm (VQA)               |      23 |
| Uncompute                                         |      11 |
| Oracle                                            |       7 |
| Grover                                            |       6 |
| Initialization                                    |       1 |

---

## VI. Top Matched Concepts

### Top 20 Most Frequently Matched Concepts

| Framework         | Concept                         |   Matches |
|:------------------|:--------------------------------|----------:|
| Classiq           | ...hadamard_transform           |        59 |
| Qiskit-algorithms | ...QAOA                         |        35 |
| Qiskit-algorithms | ...VQE                          |        29 |
| Classiq           | ...qpe                          |        26 |
| Classiq           | ...apply_to_all                 |        22 |
| Classiq           | ...qft                          |        22 |
| Classiq           | ...qpe_flexible                 |        21 |
| Qiskit            | ...QFT                          |        21 |
| Pennylane         | ...QFT                          |        20 |
| Classiq           | ...qsvt                         |        19 |
| Qiskit-algorithms | ...SamplingVQE                  |        19 |
| Classiq           | ...suzuki_trotter               |        17 |
| Qiskit-algorithms | ...AdaptVQE                     |        12 |
| Qiskit-algorithms | ...AmplitudeEstimation          |        11 |
| Qiskit-algorithms | ...IterativePhaseEstimation     |        11 |
| Classiq           | ...phase_oracle                 |        10 |
| Qiskit-algorithms | ...PhaseEstimation              |        10 |
| Qiskit            | ...AND                          |        10 |
| Pennylane         | ...QuantumMonteCarlo            |        10 |
| Qiskit-algorithms | ...IterativeAmplitudeEstimation |         9 |

---

## VII. Unmatched Pattern Analysis

The following **2** patterns from the source files were **NOT found** in any project:

- Function Table
- Schmidt Decomposition
