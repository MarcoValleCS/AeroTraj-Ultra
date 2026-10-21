# Benchmark unificato - OpenSky 90 giorni, orizzonte 5 s -> 300 s

Test set: 9 giorni (2022-03-23 -> 2022-03-31), stessi split e stessa metrica
(MDE 3-D ECEF, metri) per tutti i modelli.

| model                          |   mde_mean_m |   mde_5s |   mde_10s |   mde_15s |   mde_20s |   mde_30s |   mde_45s |   mde_60s |   mde_90s |   mde_120s |   mde_150s |   mde_180s |   mde_240s |   mde_300s |
|:-------------------------------|-------------:|---------:|----------:|----------:|----------:|----------:|----------:|----------:|----------:|-----------:|-----------:|-----------:|-----------:|-----------:|
| **AeroTraj-Ultra + retrieval** |        957.2 |     61.1 |      66.7 |      74   |      85.6 |     119.6 |     194.1 |     297.6 |     572.7 |      918.6 |     1326.1 |     1786.3 |     2848.8 |     4092.4 |
| ours                           |        991.6 |     61.2 |      67   |      74.6 |      86.7 |     122.3 |     200.2 |     308   |     593.9 |      952.2 |     1374.7 |     1852.9 |     2953.2 |     4243.8 |
| fbpp_v2                        |       1219.9 |     68   |      84   |     103.2 |     127   |     179.1 |     283.3 |     420.9 |     773.5 |     1210   |     1718.9 |     2287.6 |     3577.4 |     5025.6 |
| lstm                           |       1296.4 |     77.7 |      96.9 |     117.8 |     142.6 |     206.6 |     327.6 |     478.4 |     846.4 |     1296.9 |     1825.3 |     2417.2 |     3755.4 |     5264.8 |
| FlightPREV (legacy, re-impl.)  |       1495.8 |    102.8 |     123.8 |     125.5 |     155.8 |     188.4 |     305.9 |     470.9 |     871.9 |     1395.2 |     2024.6 |     2753.1 |     4461.1 |     6466.5 |
| cv_doppler                     |       1507.7 |     64.2 |      76.1 |      91.7 |     113.3 |     171.5 |     291.7 |     450.7 |     868.8 |     1406.3 |     2054.8 |     2802.7 |     4565   |     6643   |
| cfm                            |       1704.3 |     87.2 |      99   |     114.6 |     137.6 |     208.7 |     354.3 |     552.7 |    1060.5 |     1708.2 |     2421.3 |     3228.3 |     5057.7 |     7125.8 |
| cv_lsq                         |       1802.5 |     81.5 |     114.9 |     151.7 |     193.9 |     291.5 |     466.4 |     676.4 |    1185.5 |     1804   |     2524.5 |     3337.4 |     5213.8 |     7391.8 |
| imm_res                        |       2019.2 |     90.9 |     145.9 |     206.4 |     271.8 |     414   |     650.4 |     913.7 |    1506.4 |     2174.7 |     2909.6 |     3710.3 |     5534.8 |     7721.2 |
| ascent                         |       2348.9 |    105.7 |     179.3 |     258   |     342.5 |     523.9 |     820.2 |    1144.9 |    1861.1 |     2646.8 |     3496   |     4398   |     6333.1 |     8426   |
| attn_lstm                      |       3170   |    100.4 |     168.8 |     245   |     330.9 |     508.8 |     797.3 |    1105   |    1904.7 |     2967.1 |     4208.8 |     5551.3 |     8627.7 |    14694.6 |
| ddpm                           |       4300.5 |    146.4 |     223.9 |     300.7 |     399.8 |     631.2 |    1050.8 |    1600.7 |    2786.6 |     4233.4 |     5954.7 |     8128.6 |    12577.5 |    17871.6 |

## Note

- `FlightPREV (legacy, re-impl.)`: architettura TrajTransformer del repo
  MarcoValleCS/flight ri-implementata su OpenSky, orizzonte esteso a 5-300 s.
  Gli embedding categorici originali (ac_type/operator/dep/arr) non esistono nei
  dati OpenSky e sono sostituiti da due embedding sull'area geografica di
  origine/destinazione del segmento: e' un handicap per il modello legacy.

- `AeroTraj-Ultra + retrieval`: modello con prior d'intenzione recuperata dalle
  traiettorie passate piu' simili + testa multi-iperspettrale (K=8).

## Banda di incertezza calibrata (conformal, Mondrian per fase di volo)

| Livello nominale | Copertura (ensemble) | Raggio medio |
|---:|---:|---:|
| 80% | 0.807 | 642 m |
| 90% | 0.904 | 1,190 m |
| 95% | 0.953 | 2,116 m |