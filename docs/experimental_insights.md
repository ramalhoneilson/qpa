# Experimental Insights

Design decisions, failure modes, and lessons learned during the QPA evaluation.
Written to support the experiment description and discussion sections of any publication or report.

---

## 1. Fundamental Tool Limitation: Call-Site Matching vs. Pattern Implementation

The semantic analyser scans source files for **call sites** — function/method names that are *called* — and compares them against the reference knowledge base. This works well for **application-level code** that calls named framework primitives (e.g., `VQE()`, `grover_search()`, `QFT()`), but breaks for **library-definition code** that implements patterns by composing lower-level primitives.

The failure mode is systematic: quantum algorithms compose lower-level patterns.
- QPE calls QFT → QPE files are predicted as Basis Change
- Shor's calls QFT → Shor's files are predicted as Basis Change
- HHL calls QPE → HHL files are predicted as QPE (correct by accident)
- Amplitude Amplification calls a reflection/diffuser → predicted as Grover

**Consequence**: Qrisp (F1=0.058) and Qualtran (F1=0.051) scored near-zero, not because the tool is wrong about what they call, but because the tool is designed for a different level of abstraction.

**Implication for evaluation**: ground truth must be drawn from application-level code, not from library source code. The distinction is: "does this file *use* a quantum algorithm by name, or does it *implement* one from scratch?"

**Potential improvement**: add a **defined-class matcher** that embeds the docstrings of every class and function *defined* in the target file and matches those against pattern summaries. This would complement the call-site matcher and cover library-definition files.

---

## 2. Ground Truth Source Selection

### Why Classiq-library is circular (unusable as GT)

The knowledge base was built by extracting concepts from the Classiq Python SDK. Classiq-library notebooks directly call those same SDK functions (`hadamard_transform()`, `qpe()`, `qft()`). Name matching fires at cosine similarity ≈ 1.0 trivially because the evaluation vocabulary is identical to the training vocabulary. The tool would look artificially perfect — it tells you nothing about generalization to unseen code.

### Why PennyLane qml is distribution-skewed

The `pennylane/qml` repository is PennyLane's machine learning tutorial site. Its content is dominated by variational classifiers, QNNs, QAOA as an optimization/ML tool, and near-term device demonstrations. Coverage of textbook quantum algorithms (Grover, standalone QPE, QAE, Shor's) is thin — typically 1–3 notebooks each. Evaluating on qml would give high recall for `VariationalCircuit`/`QAOA`/`VQE` and near-zero recall for `QuantumPhaseEstimation`/`GroverOperator`/`AmplitudeEstimation` as standalone algorithms. The distribution bias would invalidate cross-pattern comparison.

### Why Amazon Braket Algorithm Library is the best available GT source

- Explicitly a cross-framework algorithm showcase — breadth is the stated goal
- Covers 16 algorithm families: Grover, QPE, QFT, QAOA, Shor's, BV, Deutsch-Jozsa, Simon's, HHL, Quantum Walk, Quantum Counting, QCBM, Bell's inequality, CHSH, QMC, Adaptive Shot
- Zero QML content (no QNNs, no variational classifiers, no transfer learning)
- Application-level code: notebooks call named library functions, not implement from scratch
- Notebooks contain rich explanatory comments → summary matching can trigger in addition to name matching

---

## 3. Evaluation Attempts and Why They Failed

| Target | Type | Files | Micro F1 | Root Cause |
|---|---|---|---|---|
| qiskit-finance (before KB fix) | Application layer | 13 | 0.000 | KB labelling errors (EstimationProblem, amplitude estimators) |
| qiskit-finance (after KB fix) | Application layer | 13 | 0.300 | Too few files for stable metrics |
| Qrisp | Library source | 58 | 0.058 | Call-site limitation; library defines patterns from scratch |
| Qualtran | Library source | 168 | 0.051 | Same; fault-tolerant bloq library, entirely different abstraction layer |

**Statistical reliability**: With fewer than ~50 instances, precision/recall estimates are highly sensitive to individual misclassifications. qiskit-finance had only 13 annotated files; a single wrong prediction moves F1 by 0.08.

---

## 4. Knowledge Base Labelling Errors Found

Two systematic errors were found in `enriched_qiskit_algorithms_quantum_patterns.csv`:

**Error 1 — EstimationProblem misclassified as Oracle**
`qiskit_algorithms.amplitude_estimators.estimation_problem.EstimationProblem` was labelled Oracle because one of its methods is oracle-adjacent. The class as a whole is the container/setup object for Quantum Amplitude Estimation. Every qiskit-finance file that calls `EstimationProblem()` was therefore predicted as Oracle instead of QAE.
→ Fixed to `Quantum Amplitude Estimation`.

**Error 2 — Amplitude Estimation algorithms mislabelled as Amplitude Amplification**
`AmplitudeEstimation`, `IterativeAmplitudeEstimation`, `MaximumLikelihoodAmplitudeEstimation`, `FasterAmplitudeEstimation`, `AmplitudeEstimator` were all labelled Amplitude Amplification because they *use* it internally. These algorithms ARE the Quantum Amplitude Estimation pattern.
→ All fixed to `Quantum Amplitude Estimation`.

**Impact**: these two fixes alone raised qiskit-finance from Micro F1=0.000 to F1=0.300.

**Other labelling issues (not yet fixed)**:
- `Isometry` (Qiskit): labelled Circuit Construction Utility; should be Data Encoding — it is used in `NormalDistribution`/`LogNormalDistribution` to amplitude-encode probability distributions
- `TrotterQRTE` (qiskit-algorithms): labelled Initialization; should be Hamiltonian Simulation — Trotterized real-time evolution is simulation, not state preparation

---

## 5. Identifier Normalization

Adding `normalize_identifier()` (splits `snake_case` and `CamelCase` into space-separated lowercase tokens before embedding) improved name-based matching:

- Total matches: 751 → 805 (+7.2%)
- Name-based matches: 475 → 531 (+11.8%)
- Average score: 0.8839 → 0.8920
- Largest per-pattern gains: Grover (+0.046), VQA (+0.048), Amplitude Amplification (+0.044)

The 56 additional name-based matches are high-confidence (mean score held at 0.9948) — they were genuine identifiers that fell just below the 0.90 threshold before normalization, not noise.

Multi-word patterns benefit most from normalization because the concept name maps to multiple tokens (`grover_operator` → `grover operator`), which the sentence transformer encodes as semantically richer input.

---

## 6. Name Matching vs. Summary Matching — Threshold Behavior

The tool uses two matching modes:
- **Name matching** (threshold 0.90): compares extracted call names against KB concept short names
- **Summary matching** (threshold 0.65): compares concatenated comment blocks against KB concept summaries

Key observations:
- Name matching has near-perfect precision (avg score 0.9948) but zero recall for library code (no call names match KB names)
- Summary matching fires on comment descriptions independently of what functions are called; it is the only path to detecting patterns in files with non-standard naming conventions
- For braket notebooks, both modes should fire: braket primitives (`Circuit().h()`, `Circuit().cnot()`) will name-match KB circuit utilities; descriptive markdown cells will summary-match pattern descriptions
- Summary matching is more susceptible to false positives — a notebook that describes QPE in passing will match the QPE pattern even if it doesn't implement it

---

## 7. C++ Multilanguage Analysis Findings

Running the same semantic analysis on C++ quantum framework examples (cuda-quantum, qpp, intel-qs, qrack):

- **cuda-quantum**: uses single-character gate primitives (`h`, `x`, `mz`, `cnot`) — these are too short to match concept names even after normalization. Low match count is expected and correct; would require a cuda-quantum-specific KB.
- **qpp and intel-qs**: used named functions (`qft`, `Initialize`, `QPE`) that match KB concepts well. These produced meaningful matches.
- **General insight**: C++ quantum frameworks tend to use low-level gate APIs more than high-level named algorithm calls. Summary matching (comment blocks) is relatively more important for C++ than for Python.

---

## 8. Evaluation Corpus Selection: Notebooks vs. Source Library Files

**Use notebooks (or application scripts), not library source files**, as the evaluation corpus.

Library source files define algorithms from scratch using low-level gates and primitives that don't appear in the reference knowledge base. The tool's recall collapses to near-zero on them regardless of GT quality.

For braket-algorithm-library specifically:
- `src/braket/experimental/algorithms/*/` — implements algorithms from scratch. Call sites are braket primitives (`Circuit().h()`, `Circuit().cnot()`), not framework-level algorithm names. These will not name-match the KB.
- `notebooks/textbook/*.ipynb` and `notebooks/advanced_algorithms/*.ipynb` — call the library functions by name (`grovers_search()`, `quantum_phase_estimation()`, `qft()`), and contain markdown cells explaining the algorithm. Both name and summary matching should fire.

This distinction also explains the earlier qiskit-finance partial success: the application files (`EuropeanCallDelta.py`, `EuropeanCallPricing.py`) call `EstimationProblem()` and `AmplitudeEstimation()` by name, while the lower-level objective files (`*Objective.py`) implement payoff circuits from scratch.

---

## 9. Pattern Coverage Gaps in the Knowledge Base

The reference KB covers patterns derived from Classiq, PennyLane, Qiskit, and qiskit-algorithms. Patterns present in the KB:

- Well covered: VQA, QAOA, QFT/Basis Change, QPE, Grover, Amplitude Amplification, QAE, Data Encoding, Hamiltonian Simulation, Oracle, Initialization
- Present but sparse: SWAP Test, Linear Combination of Unitaries, Dynamic Circuit, Schmidt Decomposition, Quantum Arithmetic, Creating Entanglement

Patterns **not** in the KB that appear in evaluated projects:
- **Quantum Walk** (braket, Qrisp) — no KB concept covers it directly
- **Quantum Error Correction** (Qualtran, PECOS) — absent from all four KB frameworks
- **Fault-tolerant bloq primitives** (Qualtran T-gate, Toffoli, QROM) — outside KB scope entirely
- **Born Machine / Generative QML** (braket QCBM) — partially covered by VQA but no specific pattern

---

## 10. Statistical Sufficiency

A minimum of ~50 independent ground truth instances is needed for precision/recall estimates with error bars below ±0.15. With fewer instances, a single misclassification produces metric swings larger than the measured effect size.

The braket-algorithm-library ground truth has 79 entries across 19 notebooks, satisfying this threshold with margin. For further robustness, the target would be 100+ instances with balanced pattern distribution (no single pattern exceeding ~30% of instances).

---

## 11. Supplementing the GT with QML Coverage

Amazon Braket Algorithm Library has zero quantum machine learning content. To avoid evaluating only textbook-algorithm recall, the GT must be supplemented with at least one QML-heavy corpus. Candidate sources ranked by suitability:

**1. qiskit-machine-learning (best fit)**
Calls functions directly present in the existing KB: `SamplerQNN`, `EstimatorQNN`, `QNNCircuit`, `VQC`, `QSVC`, `PegasosQSVC`, `FidelityQuantumKernel`. Name matching fires reliably because the vocabulary is Qiskit-family. Patterns added that braket lacks: Quantum Neural Network (QNN), Variational Quantum Algorithm (VQA) as classifier, Data Encoding (ZZFeatureMap, PauliFeatureMap), quantum kernel methods.

**2. Filtered PennyLane qml demos**
Rather than the full qml repo (which saturates on VQA/QNN), select only demos covering Data Encoding and quantum kernels. Avoids QML-saturation bias while contributing missing pattern coverage.

**3. qiskit-tutorials repo**
Application-level notebooks for VQE, VQC, QAOA. Redundant with qiskit-machine-learning for most patterns. Secondary option.

**Patterns still absent after braket + qiskit-ml:**
- Quantum Error Correction (not in the KB at all)
- Hamiltonian Simulation as a standalone use case (only appears as a subroutine in HHL/QMC)

---

## 12. The Right Metric for Specialised Frameworks: Per-Framework Observed Recall

Standard micro-averaged F1 conflates framework specialisation with tool failure. When a framework has no QML content, the tool accumulates false negatives for QNN/Data Encoding patterns that simply do not exist in that corpus — the metric reports low recall as if the tool failed, when the framework never had those patterns.

**The correct metric is per-framework observed recall:**

> For each (framework, pattern) pair where the framework's GT contains at least one positive, compute TP / (TP + FN). Average only over patterns that are present in that framework's GT ("observed" patterns).

This is sometimes called **conditional recall** or **sensitivity over present classes**. Concretely:
- For braket: compute recall over its 14 present patterns (Grover, QPE, QFT, QAOA, Oracle, …). QNN is undefined — braket has no QNN, so there is no denominator.
- For qiskit-machine-learning: compute recall over its present patterns (QNN, VQA, Data Encoding, …). QPE is undefined.
- Average across (framework, pattern) pairs where the denominator ≥ 1.

**Precision remains useful** to answer the complementary question: "when the tool claims a pattern is present, is it right?" False positives are still meaningful — a QML-heavy KB should not hallucinate QML patterns in textbook notebooks.

**Recommended reporting structure:**

| Metric | Question answered | Sensitivity to specialisation |
|---|---|---|
| Per-framework observed recall | Does the tool find what's there? | None — absent patterns are excluded |
| Per-framework precision | Is what it finds correct? | Low — FP on absent patterns penalised |
| Macro F1 over present patterns | Combined summary per framework | Low — excludes absent patterns |

Do **not** report a single pooled F1 across all frameworks as the primary result. Report per-framework observed recall and precision, then discuss whether patterns with low recall are due to vocabulary mismatch (fixable via KB extension) or abstraction-level mismatch (fundamental call-site limitation).
