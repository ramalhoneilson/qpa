# Embedding Model Comparison Report

Models compared: all-mpnet-base-v2, jina-embeddings-v2-base-code
KB concepts: 327 (classiq only — see script for full count)

## Load Time

| Model | Load (s) |
|---|---|
| all-mpnet-base-v2 | 2.1 |
| jina-embeddings-v2-base-code | 3.2 |

## Name Channel — Best-match score distribution (code element → KB concept)

**all-mpnet-base-v2**  p50=1.000  p75=1.000  p90=1.000  max=1.000  enc=16.44s

**jina-embeddings-v2-base-code**  p50=1.000  p75=1.000  p90=1.000  max=1.000  enc=4.26s

## Summary Channel — Best-match score distribution (comment block → KB summary)

**all-mpnet-base-v2**  p50=0.611  p75=0.690  p90=0.747  max=0.823  enc=16.06s

**jina-embeddings-v2-base-code**  p50=0.635  p75=0.708  p90=0.780  max=0.914  enc=16.52s

## Spot-check Pairs

| Pair | all-mpnet-base-v2 | jina-embeddings-v2-base-code |
|---|---|---|
| exact match (QFT) | 1.0000 | 1.0000 |
| near match (Grover) | 0.9620 | 0.9846 |
| exact match (QAOA) | 1.0000 | 1.0000 |
| exact match (VQE) | 1.0000 | 1.0000 |
| gap case (BV→Oracle) | 0.1952 | 0.5195 |
| gap case (Bell→Entanglement) | 0.3887 | 0.6156 |
| utility exact | 1.0000 | 1.0000 |
| noise (fit→VQE) | 0.1630 | 0.4724 |
| noise (append→QFT) | 0.0657 | 0.2188 |
