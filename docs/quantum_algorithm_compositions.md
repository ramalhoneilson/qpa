# Quantum Algorithm Composition Hierarchies

How high-level quantum algorithms decompose into lower-level building blocks.

---

## 1. Core Primitives Dependency Overview

A single view showing which algorithms depend on which building blocks.

```mermaid
flowchart TD
    subgraph HIGH["High-Level Algorithms"]
        SHOR["Shor's Algorithm"]
        HHL["HHL (Linear Systems)"]
        QAOA["QAOA"]
        VQE["VQE"]
        VQD["VQD"]
        QMCI["Quantum Monte Carlo\n(QMCI)"]
        QBACK["Quantum Backtracking"]
        GQSP["GQSP"]
        IQPE["Iterative QPE"]
    end

    subgraph MID["Mid-Level Algorithms"]
        QPE["Quantum Phase\nEstimation (QPE)"]
        QAE["Quantum Amplitude\nEstimation (QAE)"]
        GROVER["Grover's Search"]
        LCU["Linear Combination\nof Unitaries (LCU)"]
        HAMILTONIAN["Hamiltonian Simulation"]
        VQA["Variational Quantum\nAlgorithm (VQA)"]
    end

    subgraph PRIM["Core Primitives"]
        AA["Amplitude Amplification"]
        BC["Basis Change\n(QFT / Hadamard)"]
        INIT["Initialization\n(State Prep)"]
        ORACLE["Oracle"]
        QA["Quantum Arithmetic"]
        QLO["Quantum Logical Operators"]
        DE["Data Encoding"]
        UNCOMP["Uncompute"]
        DSA["Domain Specific\nApplication"]
    end

    SHOR --> QPE
    SHOR --> QA
    HHL --> LCU
    HHL --> QPE
    HHL --> ORACLE
    QAOA --> VQA
    QAOA --> ORACLE
    QAOA --> DSA
    VQE --> VQA
    VQE --> HAMILTONIAN
    VQD --> VQE
    VQD --> UNCOMP
    QMCI --> QAE
    QMCI --> DE
    QBACK --> GROVER
    GQSP --> LCU
    GQSP --> QPE
    IQPE --> QPE

    QPE --> BC
    QPE --> INIT
    QPE --> QA
    QAE --> QPE
    QAE --> AA
    GROVER --> ORACLE
    GROVER --> AA
    LCU --> AA
    LCU --> DE
    LCU --> INIT
    HAMILTONIAN --> QLO
    HAMILTONIAN --> BC
    VQA --> INIT
    VQA --> BC

    AA --> ORACLE
    AA --> BC
    AA --> INIT
```

---

## 2. Amplitude-Based Algorithm Family

```mermaid
flowchart TD
    QMCI["Quantum Monte Carlo\nIntegration (QMCI)"]
    QAE["Quantum Amplitude\nEstimation (QAE)"]
    AA["Amplitude Amplification"]
    GROVER["Grover's Search"]

    QMCI --> QAE
    QMCI --> DE["Data Encoding\n(probability distribution)"]
    QMCI --> ORACLE["Oracle\n(payoff circuit)"]
    QMCI --> QA["Quantum Arithmetic"]

    QAE --> QPE["QPE"]
    QAE --> AA
    QAE --> ORACLE

    AA --> ORACLE
    AA --> INIT["Initialization"]
    AA --> BC["Basis Change\n(Hadamard diffuser)"]
    AA --> QA2["Quantum Arithmetic\n(inversion about mean)"]

    GROVER --> ORACLE
    GROVER --> AA

    QPE --> BC2["Basis Change (QFT)"]
    QPE --> QA3["Quantum Arithmetic\n(phase kickback)"]
    QPE --> INIT2["Initialization"]
```

---

## 3. Linear Systems & Simulation Family

```mermaid
flowchart TD
    HHL["HHL\n(Linear Systems)"]
    GQSP["GQSP\n(Generalised QSP)"]
    HAMILTONIAN["Hamiltonian Simulation\n(Trotterization)"]

    HHL --> LCU["Linear Combination\nof Unitaries (LCU)"]
    HHL --> QPE["Quantum Phase\nEstimation (QPE)"]
    HHL --> AA["Amplitude Amplification"]
    HHL --> ORACLE["Oracle\n(flag qubit rotation)"]
    HHL --> INIT["Initialization"]

    GQSP --> LCU
    GQSP --> QPE
    GQSP --> HAMILTONIAN
    GQSP --> QA["Quantum Arithmetic\n(polynomial approx)"]
    GQSP --> QLO["Quantum Logical Operators"]

    HAMILTONIAN --> QLO
    HAMILTONIAN --> BC["Basis Change\n(rotations)"]
    HAMILTONIAN --> QA2["Quantum Arithmetic\n(time step)"]
    HAMILTONIAN --> INIT

    LCU --> AA
    LCU --> DE["Data Encoding (PREPARE)"]
    LCU --> ORACLE2["Oracle (SELECT)"]
    LCU --> INIT

    QPE --> BC2["Basis Change (QFT)"]
    QPE --> INIT
```

---

## 4. Variational Algorithm Family

```mermaid
flowchart TD
    VQD["VQD\n(Variational Quantum Deflation)"]
    VQE["VQE\n(Variational Quantum Eigensolver)"]
    QAOA["QAOA"]

    VQD --> VQE
    VQD --> UNCOMP["Uncompute\n(compute-uncompute overlap)"]

    VQE --> VQA["Variational Quantum\nAlgorithm (VQA)"]
    VQE --> HAMILTONIAN["Hamiltonian Simulation"]
    VQE --> BC["Basis Change\n(measurement rotation)"]
    VQE --> QAE["QAE\n(gradient estimation)"]

    QAOA --> VQA
    QAOA --> INIT["Initialization"]
    QAOA --> ORACLE["Oracle\n(phase separator)"]
    QAOA --> BC2["Basis Change\n(mixer Hamiltonian)"]
    QAOA --> DSA["Domain Specific Application\n(cost Hamiltonian)"]
    QAOA --> QA["Quantum Arithmetic"]

    VQA --> INIT
    VQA --> BC3["Basis Change"]

    HAMILTONIAN --> QLO["Quantum Logical Operators"]
    HAMILTONIAN --> BC4["Basis Change"]
```

---

## 5. Phase Estimation Family

```mermaid
flowchart TD
    SHOR["Shor's Algorithm"]
    IQPE["Iterative QPE\n(Kitaev)"]
    QPE["Quantum Phase\nEstimation (QPE)"]

    SHOR --> QPE
    SHOR --> QA["Quantum Arithmetic\n(modular exponentiation)"]
    SHOR --> QLO["Quantum Logical Operators"]

    IQPE --> QPE
    IQPE --> BC_SINGLE["Basis Change\n(single-qubit QFT)"]
    IQPE --> QA2["Quantum Arithmetic\n(phase accumulation)"]
    IQPE --> INIT["Initialization"]
    IQPE --> ORACLE["Oracle\n(controlled unitary repetition)"]

    QPE --> BC["Basis Change (QFT)"]
    QPE --> CU["Controlled Unitary"]
    QPE --> INIT
    QPE --> QA3["Quantum Arithmetic\n(phase kickback)"]
```

---

## 6. Quantum Backtracking

```mermaid
flowchart TD
    QBACK["Quantum Backtracking"]
    QBACK --> GROVER["Grover's Search"]
    QBACK --> AA["Amplitude Amplification"]
    QBACK --> ORACLE["Oracle (predicate)"]
    QBACK --> QLO["Quantum Logical Operators"]
    QBACK --> INIT["Initialization"]
    QBACK --> BC["Basis Change"]

    GROVER --> AA
    GROVER --> ORACLE

    AA --> ORACLE
    AA --> BC
    AA --> INIT
```
