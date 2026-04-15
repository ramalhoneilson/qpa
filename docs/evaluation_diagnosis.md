# Evaluation Diagnosis: Why Precision/Recall Is Low

This document explains the root causes of the low precision/recall scores for
qiskit-finance and Qrisp, corrects ground-truth labelling errors, and reports
on docstring coverage in the reference knowledge base.

---

## 1. Root Cause: Call-Site Matching vs. What a File Implements

The pattern finder scans every Python file for **call sites** — AST nodes where a
function or method is *called* — and compares the called name (or its embedding)
against the reference concept library.  This works well for application-level
code that *uses* framework primitives (classiq-library, qiskit-finance application
layer), but breaks down for library-definition files that *implement* the pattern
by calling lower-level primitives.

### Pattern of mismatch (Qrisp examples)

| File | Implements | Calls internally | Predicted |
|---|---|---|---|
| `alg_primitives/qpe.py` | QPE | `QFT` | Basis Change |
| `algorithms/shor/shors_algorithm.py` | QPE (Shor's) | `QFT` | Basis Change |
| `algorithms/quantum_backtracking/…` | Grover (backtracking) | `QFT`-related | Basis Change |
| `algorithms/hhl.py` | LCU (HHL) | `QPE`-named calls | QPE |
| `alg_primitives/amplitude_amplification.py` | Amplitude Amplification | `reflection` (Grover diffuser) | Grover |
| `alg_primitives/lcu.py` | LCU | `amplitude_amplification` | Amplitude Amplification |
| `algorithms/gqsp/hamiltonian_simulation.py` | Hamiltonian Simulation | `GQSP` (QFT-based) | Basis Change |

**The fundamental reason**: Quantum algorithms compose lower-level patterns.  QPE
calls QFT; Shor's calls QFT; HHL calls QPE; Amplitude Amplification calls
reflection/diffuser.  The tool sees the *used* primitive, not the *implemented*
pattern.

### Why application files work better

Files like `classiq-library` notebooks call `hadamard_transform()`, `VQE()`,
`QAOA()` directly.  The called name IS the pattern name, so name matching
(score=1.0) fires correctly.

---

## 2. Ground-Truth Corrections

### 2a. qiskit-finance: Payoff function objective circuits

**Original GT** (wrong): `EuropeanCallDeltaObjective`, `EuropeanCallPricingObjective`,
`FixedIncomePricingObjective` → Quantum Amplitude Estimation

**Corrected GT**: → Oracle

**Why**: These files are quantum *circuit* classes that implement the
comparison/rotation payoff functions used as oracles inside QAE, not QAE algorithms
themselves. `EuropeanCallDeltaObjective` applies `IntegerComparator` (a quantum
comparator gate); `FixedIncomePricingObjective` applies parameterized `RY`/`CRY`
rotations to approximate the linear payoff. The application-level wrapper files
(`EuropeanCallDelta`, `EuropeanCallPricing`, `FixedIncomePricing`) keep GT=Quantum
Amplitude Estimation because they orchestrate the QAE problem.

### 2b. Qrisp GT: mostly correct

The Qrisp GT labels are sound.  The wrong predictions are entirely due to the
call-site limitation described above, not GT errors.  A few borderline cases worth
noting:

- `alg_primitives/amplitude_amplification.py` GT=Amplitude Amplification,
  predicted=Grover.  Grover IS the original amplitude amplification; the Qrisp
  function is the general form.  GT is correct.
- `alg_primitives/reflection.py` GT=Amplitude Amplification: correct — reflection
  is the diffusion operator of Grover/amplitude amplification.

---

## 3. Knowledge Base Labelling Errors Found

### 3a. EstimationProblem → Oracle (FIXED)

`qiskit_algorithms.amplitude_estimators.estimation_problem.EstimationProblem`
was classified as **Oracle** because its `rescale` method (the one reviewed in
the method-level review) is an oracle-adjacent operation.  However, the class as
a whole is the **container/setup class for Quantum Amplitude Estimation** — it
holds the state preparation circuit, objective qubits, and Grover operator.

**Impact**: Every file that calls `EstimationProblem()` (all qiskit-finance
application files) was predicted as Oracle instead of Quantum Amplitude Estimation.

**Fix applied**: Changed to `Quantum Amplitude Estimation`.

### 3b. Amplitude Estimation algorithms → Amplitude Amplification (FIXED)

`AmplitudeEstimation`, `IterativeAmplitudeEstimation`, `MaximumLikelihoodAmplitudeEstimation`,
`FasterAmplitudeEstimation`, and `AmplitudeEstimator` were all labelled
**Amplitude Amplification** (they use amplitude amplification internally).
These algorithms ARE the "Quantum Amplitude Estimation" pattern — the specific
technique of using QPE or iterative methods to estimate the amplitude of a quantum
state.

**Impact**: Any code calling these estimators was predicted as Amplitude Amplification
rather than Quantum Amplitude Estimation, causing qiskit-finance to score 0 for QAE.

**Fix applied**: Changed all six entries to `Quantum Amplitude Estimation`.

### 3c. Other KB issues (documented, not yet fixed)

| Concept | Current label | Suggested label | Reason |
|---|---|---|---|
| `Isometry` (Qiskit) | Circuit Construction Utility | Data Encoding | Used in qiskit-finance NormalDistribution/LogNormal to amplitude-encode probability distributions; tiebreaker against `Initialize` (Initialization) currently picks CCU |
| `TrotterQRTE` (qiskit-algorithms) | Initialization | Hamiltonian Simulation | Trotterized quantum real-time evolution is Hamiltonian Simulation, not state preparation |

---

## 4. Docstring Coverage in the Reference Knowledge Base

| Framework | Total concepts | Non-empty summary | Meaningful (>30 chars) |
|---|---|---|---|
| Classiq | 64 | 64 (100 %) | 64 (100 %) |
| PennyLane | 68 | 68 (100 %) | 67 (99 %) |
| Qiskit | 85 | 85 (100 %) | 63 (74 %) |
| qiskit-algorithms | 39 | 39 (100 %) | 35 (90 %) |

**All concepts have at least some summary text.**  Coverage is not the problem.

The quality gap is in the **Qiskit** framework (22 concepts with near-empty summaries)
and **qiskit-algorithms** (4 minimal ones).  The problematic ones are:

- `IntegerComparator → 'Integer Comparator.'` — no semantic content beyond the name
- `GroverOperator → 'The Grover operator.'` — too short to distinguish from Grover itself
- `PhaseEstimation → 'Phase Estimation circuit.'` — minimal
- `PhaseOracle → 'Phase Oracle.'`, `PhaseOracleGate → 'Implements a phase oracle.'`
- 12 gate utility classes (GR, GRX, GRY, GRZ, GMS, MSGate, PauliGate, etc.) with
  one-line summaries

For name-based matching these short summaries don't matter (the concept name itself
is matched).  For summary-based matching (when a call site name isn't in the KB),
they provide almost no discriminating signal.

### The real bottleneck: summary matching is never triggered for library code

For library files (Qrisp, Qualtran), the issue is not that summaries are short — it
is that no call site names match any concept name at all.  A Qrisp file that defines
`GQSP` from scratch will never call `hadamard_transform` or `VQE`, so neither name
nor summary matching fires.  Summary matching would only help if the *embedding
distance* between `GQSP` (a Qrisp-internal name) and a KB concept summary were low
enough to cross the similarity threshold — which it isn't for such dissimilar names.

---

## 5. Evaluation Results Before and After KB Corrections

### qiskit-finance

| Metric | Before fix | After fix |
|---|---|---|
| Micro Precision | 0.000 | **0.429** |
| Micro Recall | 0.000 | **0.231** |
| Micro F1 | 0.000 | **0.300** |
| QAE precision / recall | — / 0.0 | **1.000 / 0.750** |

The three application files (`EuropeanCallDelta`, `EuropeanCallPricing`,
`FixedIncomePricing`) are now correctly predicted as Quantum Amplitude Estimation.
The one missed QAE file is `EstimationApplication` (abstract base class — no call
sites at all).

Remaining wrong predictions after fix:

| File | GT | Predicted | Root cause |
|---|---|---|---|
| `european_call_delta_objective.py` | Oracle | Quantum Arithmetic | calls `IntegerComparator` |
| `european_call_pricing_objective.py` | Oracle | Quantum Arithmetic | calls `LinearAmplitudeFunctionGate` |
| `lognormal.py` | Data Encoding | Circuit Construction Utility | calls `Isometry` (CCU) and `Initialize` (Init); CCU wins tie |
| `normal.py` | Data Encoding | Circuit Construction Utility | same |

### Qrisp

No change from KB fix (Qrisp has its own QAE implementation; does not call
qiskit-algorithms estimators). All 5 TP are the same: `grover_tools.py`,
`QUBO.py`, `modular_qft_addition.py`, `iterable_processing.py`,
`qiskit_state_preparation.py`.

### Three-project summary

| Project | Type | Micro P | Micro R | Micro F1 | Note |
|---|---|---|---|---|---|
| qiskit-finance (before) | Application layer | 0.000 | 0.000 | 0.000 | KB error |
| qiskit-finance (after) | Application layer | **0.429** | **0.231** | **0.300** | After fix |
| Qrisp | Library | 0.185 | 0.034 | 0.058 | Call-site limitation |
| Qualtran | Library | 0.185 | 0.030 | 0.051 | Call-site limitation |

Application-level code significantly outperforms library-definition code even after the
KB fix.  The call-site limitation is the dominant constraint for Qrisp and Qualtran.

---

## 6. Implications and Recommended Next Steps

| Issue | Impact | Suggested fix |
|---|---|---|
| Call-site matcher cannot detect what a file *implements* | Low recall for all library-level code | Add a **defined-class matcher**: embed each class/function docstring in the target file and match against pattern summaries |
| KB labelling errors (EstimationProblem, amplitude estimators) | 0 QAE predictions in qiskit-finance | Fixed in this session |
| `Isometry` labelled CCU instead of Data Encoding | `NormalDistribution`/`LogNormal` predicted as CCU | Change Isometry label to Data Encoding |
| `TrotterQRTE` labelled Initialization | Hamiltonian Simulation missed | Change to Hamiltonian Simulation |
| Short Qiskit summaries (22 concepts) | Weak summary-based matching | Enrich docstrings or accept as-is (name matching still works) |

The most impactful improvement by far would be the **defined-class matcher**: instead
of (or in addition to) scanning call sites, extract the docstrings of every class and
function *defined* in the target file and embed those against pattern summaries.  This
would directly address the recall problem for Qrisp (19 % predicted), Qualtran (16 %),
and qiskit-finance (54 %), all of which define quantum patterns without calling
reference-library names.
