# Benchmark 365 giorni, griglia 10 s (12 punti = 120 s di storia)

Test set: 36 giorni (2022-11-26 -> 2022-12-31), 60000 finestre.

Budget comune: 25 epoche x 400,000 campioni/epoca = 10,000,000 campioni per modello.

| modello | MDE medio (m) | ADE 3D (m) | FDE 3D (m) | 10 s (m) | 30 s (m) | 60 s (m) | 90 s (m) | 180 s (m) | 300 s (m) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| AeroTraj-Ultra (ours) | 1256.0 | 1256.0 | 4418.0 | 65.0 | 123.8 | 326.2 | 631.5 | 1971.7 | 4418.0 |
| CV least-squares (no learning) | 2624.0 | 2624.0 | 8562.7 | 154.4 | 422.9 | 942.2 | 1577.8 | 4084.0 | 8562.7 |
| Kalman filter (CV, learnable noise) | 2069.4 | 2069.4 | 7232.8 | 92.5 | 239.8 | 590.5 | 1073.9 | 3187.2 | 7232.8 |
| IMM (CV/CA/CT, learnable noise) | 2416.2 | 2416.2 | 7913.0 | 147.6 | 391.8 | 861.7 | 1438.8 | 3744.2 | 7913.0 |
| LSTM regression | 2358.3 | 2358.3 | 7304.1 | 160.7 | 442.1 | 953.5 | 1551.4 | 3738.1 | 7304.1 |
| FlightBERT (IEEE T-ITS 2023), BCE-dominant loss | 23376.9 | 23376.9 | 60537.6 | 3664.4 | 6977.6 | 12834.5 | 19162.3 | 37085.4 | 60537.6 |
| FlightBERT++ V2 (AAAI 2024), reference | 1463.2 | 1463.2 | 5074.1 | 76.5 | 162.6 | 406.2 | 759.5 | 2300.2 | 5074.1 |
| FlightBERT++ V2 + ACoAtt (2023 block) | 1437.1 | 1437.1 | 4964.6 | 76.6 | 162.5 | 405.4 | 752.7 | 2260.8 | 4964.6 |
| BiLSTM regression | 1851.0 | 1851.0 | 6110.6 | 107.4 | 271.2 | 621.8 | 1078.2 | 2917.0 | 6110.6 |
| FlightBERT++ V2 + ACoAtt, MSE-only loss | 15161.8 | 15161.8 | 40934.2 | 1367.1 | 4091.7 | 8029.1 | 12113.5 | 24435.0 | 40934.2 |
