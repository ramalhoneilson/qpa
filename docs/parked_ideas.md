# Parked Ideas

Ideas to revisit later. Not currently being worked on.

---

## Idea: Self-Expanding Knowledge Base via High-Confidence Matches

**Source:** `idea.txt`

**Description:**
Run `run_analysis.py` on the reference frameworks and collect all concepts that have a name match score ≥ 0.99 AND a concept match score ≥ 0.90. Automatically promote these high-confidence matches into the knowledge base as new entries. Re-run the analysis with the extended KB and measure the delta (new concepts found vs. previous run). Repeat until the delta is zero (convergence).

**Why it's interesting:**
This is a bootstrapping / iterative KB expansion loop. It would let the KB grow organically from the frameworks being analysed rather than requiring manual curation. The convergence criterion (delta = 0) makes it self-terminating.

**Open questions before implementing:**
- What is the correct "concept match" score — is this the name similarity, the summary similarity, or both?
- Risk of circular reinforcement: high-confidence matches from framework X expanding the KB may then over-fit to framework X vocabulary, hurting generalization to other frameworks.
- Need a held-out validation set to check that each KB expansion round doesn't degrade recall on unseen frameworks.
- Should the threshold (99% name / 90% summary) be tuned first, or is it a reasonable starting point?

---

## Idea: Quantum Zoo as Extended Knowledge Base

**Status:** Design discussed, mapping table created, not yet implemented.

**Description:**
Parse the Quantum Algorithm Zoo (https://github.com/stephenjordan/stephenjordan.github.io/blob/master/index.html) and add each algorithm's description as a wildcard KB entry mapped to the appropriate Pattern Atlas pattern.

**Key design decisions already made:**
- Map by "what the algorithm IS" not "what subroutines it USES"
- Algorithms that are canonical examples of two patterns → two separate KB entries with different concept names, same summary
- ~30 unambiguous mappings; ~6 edge cases for quick colleague discussion
- The 4 Zoo categories map to: Oracle/Grover (Oracular), QPE/QFT (Algebraic), empty (Simulation — no matching pattern), QAOA/VQA (Optimization)
- No kappa study needed; the Zoo's own category structure serves as the grouping rationale

**Expected impact:**
- Oracle pattern recall: 0.000 → ~0.6+ (BV, DJ, Simon's descriptions match Oracle language)
- Domain Specific Application recall: improvement for Shor's, Walk, QMC
- No impact on QML patterns (Zoo has limited ML coverage)

---

## Idea: Local Comment Matching (Cell-Level Context)

**Status:** Implemented, tested, reverted.

**Description:**
Instead of embedding the whole-file comment blob, extract the Jupyter cell markdown description (lines between `# In[N]:` boundaries) and associate it with the call sites within that code cell. Each cell gets a targeted description rather than a diluted whole-file average.

**Why it was reverted:**
For the generalization test (4-KB vs unseen frameworks), this approach added FPs without improving recall. The bottleneck is KB vocabulary gap, not comment granularity — a cell comment saying "Bernstein-Vazirani finds a hidden string" still can't match because the 4-KB has no Oracle concept description using that vocabulary.

**When it WOULD help:**
Once the KB has wildcard/Zoo entries covering the missing patterns, cell-level matching would be strictly better than whole-file matching — each cell gets a precise, targeted context instead of a noisy average across all cells in the notebook.

**Implementation:** The cell-boundary parser (`# In[N]:` markers) was written and validated. Can be re-enabled by swapping `extract_comments_from_script` back to the cell-boundary version in `run_analysis.py`.
