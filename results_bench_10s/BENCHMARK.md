# Benchmark 365 giorni, griglia 10 s (12 punti = 120 s di storia)

Test set: 36 giorni (2022-11-26 -> 2022-12-31), 60000 finestre.

Budget comune: 25 epoche x 400,000 campioni/epoca = 10,000,000 campioni per modello.

| modello | MDE medio (m) | ADE 3D (m) | FDE 3D (m) | 10 s (m) | 30 s (m) | 60 s (m) | 90 s (m) | 180 s (m) | 300 s (m) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| AeroTraj-Ultra (ours) | 1256.0 | 1256.0 | 4418.0 | 65.0 | 123.8 | 326.2 | 631.5 | 1971.7 | 4418.0 |
| CV least-squares (no learning) | 2624.0 | 2624.0 | 8562.7 | 154.4 | 422.9 | 942.2 | 1577.8 | 4084.0 | 8562.7 |
| Kalman filter (CV, learnable noise) | 2215.6 | 2215.6 | 7568.3 | 111.8 | 292.9 | 688.8 | 1210.8 | 3421.3 | 7568.3 |
| IMM (CV/CA/CT, learnable noise) | 2416.2 | 2416.2 | 7913.0 | 147.6 | 391.8 | 861.7 | 1438.8 | 3744.2 | 7913.0 |
| LSTM regression | 2358.3 | 2358.3 | 7304.1 | 160.7 | 442.1 | 953.5 | 1551.4 | 3738.1 | 7304.1 |
| FlightBERT++ V2 (AAAI 2024), faithful, MSE-only loss | 15383.2 | 15383.2 | 41460.3 | 1370.8 | 4106.4 | 8223.1 | 12350.3 | 24788.2 | 41460.3 |
| FlightBERT++ V2 (AAAI 2024), faithful | 1330.8 | 1330.8 | 4562.9 | 73.7 | 151.7 | 377.4 | 703.9 | 2115.5 | 4562.9 |
| BiLSTM regression | 1801.3 | 1801.3 | 6002.8 | 101.5 | 251.9 | 584.9 | 1028.1 | 2838.4 | 6002.8 |
| FlightBERT++ V2 (AAAI 2024), no ACoAtt | 1465.1 | 1465.1 | 5082.0 | 76.7 | 162.3 | 406.0 | 758.0 | 2305.3 | 5082.0 |
| FlightBERT (IEEE T-ITS 2023), BCE-dominant loss | 23376.9 | 23376.9 | 60537.6 | 3664.4 | 6977.6 | 12834.5 | 19162.3 | 37085.4 | 60537.6 |
