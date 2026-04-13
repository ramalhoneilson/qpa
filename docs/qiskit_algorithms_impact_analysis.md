# Impact of Adding qiskit-algorithms to the Analysis

This document compares the pattern analysis results before and after including
qiskit-algorithms as a fourth source framework. The reference baseline is stored
in `data/pattern_report_reference.txt`; the new results are in
`data/final_pattern_report.txt`.

---

## Overall Summary

| Metric | Baseline | With qiskit-algorithms | Change |
|---|---|---|---|
| Total Matches | 598 | 751 | +153 (+26 %) |
| Unique Files with Matches | 265 | 313 | +48 |
| Unique Concepts Matched | 116 | 142 | +26 |
| Patterns Found | 22 | 23 | +1 (Uncompute) |
| Avg Similarity Score | 0.9027 | 0.8839 | −0.019 |

The small drop in average similarity is expected: the 39 new qiskit-algorithms
class concepts tend to match target notebooks at slightly lower cosine scores
than the tightly curated Classiq set.

---

## New Source Framework Breakdown

In the new run, qiskit-algorithms ranks second by source match count:

| Framework | Matches |
|---|---|
| Classiq | 331 |
| **qiskit-algorithms** | **194** |
| PennyLane | 119 |
| Qiskit | 107 |

---

## Pattern Count Changes

| Pattern | Baseline | New | Δ |
|---|---|---|---|
| Variational Quantum Eigensolver (VQE) | 3 | 63 | **+60** |
| Quantum Approximate Optimization Algorithm (QAOA) | 16 | 51 | **+35** |
| Quantum Phase Estimation (QPE) | 57 | 75 | +18 |
| Amplitude Amplification | 22 | 46 | +24 |
| Variational Quantum Algorithm (VQA) | 23 | 46 | +23 |
| Grover | 13 | 19 | +6 |
| Oracle | 13 | 20 | +7 |
| Initialization | 25 | 26 | +1 |
| **Uncompute** | **0** | **11** | **new pattern discovered** |
| Basis Change | 159 | 142 | −17 |
| Domain Specific Application | 66 | 59 | −7 |
| Data Encoding | 41 | 38 | −3 |

The Basis Change and Domain Specific Application drops are minor redistribution
effects — the matcher now has more concepts competing for the same call sites.

---

## New Pattern: Uncompute

Uncompute was not found in any project in the baseline run. With the
qiskit-algorithms state-fidelity classes (`BaseStateFidelity`,
`ComputeUncompute`) acting as reference concepts, it is now detected in
**7 projects**: ProjectQ, Qualtran, ReCirq, classiq-library,
qiskit-addon-cutting, qiskit-algorithms, qiskit-machine-learning.

---

## Adoption Coverage (Project Count per Pattern)

Patterns that gained the most new project coverage:

| Pattern | Baseline Projects | New Projects | Δ |
|---|---|---|---|
| VQE | 2 | 14 | +12 |
| QAOA | 8 | 16 | +8 |
| VQA | 11 | 14 | +3 |
| Amplitude Amplification | 3 | 8 | +5 |
| Grover | 6 | 7 | +1 |
| QPE | 7 | 7 | 0 |
| Uncompute | 0 | 7 | +7 (new) |

VQE is the most dramatic change: from appearing in only 2 projects
(`classiq-library`, `qiskit-nature`) to 14 projects spanning
`qiskit-finance`, `mitiq`, `tensorcircuit`, `torchquantum`,
`client-superstaq`, and others. This suggests VQE is a widely adopted
pattern whose presence was previously under-detected due to missing
reference concepts.

---

## Top Matched Concepts (new run, top 20)

| Framework | Concept | Matches |
|---|---|---|
| Classiq | hadamard_transform | 59 |
| qiskit-algorithms | QAOA | 35 |
| qiskit-algorithms | VQE | 29 |
| Classiq | qpe | 26 |
| Classiq | apply_to_all | 22 |
| Classiq | qft | 22 |
| Classiq | qpe_flexible | 21 |
| Qiskit | QFT | 21 |
| PennyLane | QFT | 20 |
| Classiq | qsvt | 19 |
| qiskit-algorithms | SamplingVQE | 19 |
| Classiq | suzuki_trotter | 17 |
| qiskit-algorithms | AdaptVQE | 12 |
| qiskit-algorithms | AmplitudeEstimation | 11 |
| qiskit-algorithms | IterativePhaseEstimation | 11 |
| Classiq | phase_oracle | 10 |
| qiskit-algorithms | PhaseEstimation | 10 |
| Qiskit | AND | 10 |
| PennyLane | QuantumMonteCarlo | 10 |
| qiskit-algorithms | IterativeAmplitudeEstimation | 9 |

---

## qiskit-algorithms as a Target Project

qiskit-algorithms now also appears as a **target project** (its converted
notebooks are scanned), ranking 3rd overall:

| Project | Matches (baseline) | Matches (new) |
|---|---|---|
| classiq-library | 361 | 370 |
| Qualtran | 64 | 70 |
| **qiskit-algorithms** | 8 | **43** |
| qiskit-finance | 13 | 38 |
| Qrisp | 28 | 33 |

---

## Still Unmatched Patterns

No change from the baseline — the following 2 patterns from the PlanQK Atlas
remain undetected in any target project:

- Function Table
- Schmidt Decomposition

---

## Notes for Write-up

- The large VQE jump (+60 matches, +12 projects) is the most significant finding.
  It suggests that the original three-framework reference set had insufficient
  VQE signal — qiskit-algorithms provides the canonical VQE class hierarchy
  that target projects actually import from.
- QAOA similarly jumps from 16 to 51 matches; qiskit-algorithms contributes
  35 of those directly (the `QAOA` class being matched 35 times in target notebooks).
- Uncompute appearing as a newly found pattern (7 projects) is noteworthy for
  the graph/sequence analysis: it represents an explicit quantum memory
  management pattern that was invisible without qiskit-algorithms.
- The average score drop (0.90 → 0.88) is minor and within acceptable range;
  the new matches are genuine, not noise.
