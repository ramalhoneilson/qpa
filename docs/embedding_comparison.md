# Embedding Model Comparison — qiskit-finance

| Model | Micro P | Micro R | Micro F1 | Macro P | Macro R | Macro F1 | Matches | Index (s) | Query (s) |
|---|---|---|---|---|---|---|---|---|---|
| all-mpnet-base-v2 | 0.500 | 0.308 | 0.381 | 1.000 | 0.250 | 0.629 | 22 | 2.4 | 28.6 |
| all-MiniLM-L6-v2 | 0.571 | 0.308 | 0.400 | 1.000 | 0.250 | 0.629 | 21 | 2.0 | 6.3 |
| all-MiniLM-L12-v2 | 0.571 | 0.308 | 0.400 | 1.000 | 0.250 | 0.629 | 18 | 1.9 | 9.2 |
| BAAI/bge-small-en-v1.5 | 0.444 | 0.308 | 0.364 | 1.000 | 0.250 | 0.629 | 959 | 2.0 | 12.0 |
| multi-qa-mpnet-base-dot-v1 | 0.500 | 0.308 | 0.381 | 1.000 | 0.250 | 0.629 | 22 | 2.5 | 68.3 |
