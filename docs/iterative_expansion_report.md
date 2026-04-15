# Iterative KB Expansion Report

**Strategy:** summary-seeded name harvesting.
Files are confirmed for a concept when `summary_score ≥ 0.75`.
Callable names from confirmed files are harvested when `name_score ≥ 0.45`
against the confirmed concept, and are not already in the KB.

| Iter | Total matches | Confirmed files | New KB entries | Delta |
|---|---|---|---|---|
| 1 | 805 | 39 | 27 | 805 |
| 2 | 1268 | 122 | 11 | 463 |
| 3 | 1922 | 159 | 3 | 654 |
| 4 | 2041 | 170 | 2 | 119 |
| 5 | 2151 | 178 | 0 | 110 |
| 6 | 2151 | 178 | 0 | 0 |

## Convergence

Converged after **6** iteration(s): delta = 0 and no new KB entries.

## Expanded KB Summary

- Total new KB entries added: **43**
- Entries in `data/expanded_quantum_concepts.json`: **43**
