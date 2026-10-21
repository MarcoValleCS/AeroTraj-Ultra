# AeroTraj-Ultra (FlightBERT++ V3) - report comparativo delle run

## 1. Dataset

| Run | Giorni | Train/Val/Test | Punti 5 s | Segmenti | Voli unici | Finestre (stride 1) | Righe 1 s lette |
|---|---:|---|---:|---:|---:|---:|---:|
| 40d CPU (original) | 40 | 32/4/4 | 2,110,871 | 4,765 | 3,118 | 1,486,656 | 41,943,040 |
| 40d GPU (control) | 40 | 32/4/4 | 2,110,871 | 4,765 | 3,118 | 1,486,656 | 41,943,040 |
| 90d GPU (nuovo) | 90 | 72/9/9 | 65,753,931 | 149,894 | 97,509 | 46,117,817 | 1,050,128,833 |

## 2. Addestramento

| Run | Epoche | Step/epoch | Parametri | Test 5 s [m] | Test 1 min [m] | Test 5 min [m] | Test 10 min [m] | MDE media [m] |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 40d CPU (original) | 2 | 36 | 1,277,415 | 61.5 | 355.8 | 4938.3 | 15486.5 | 3625.9 |
| 40d GPU (control) | 30 | 36 | 1,277,415 | 61.0 | 336.0 | 4337.6 | 12213.5 | 3019.5 |
| 90d GPU (nuovo) | 30 | 1110 | 1,277,415 | 61.2 | 304.4 | 3955.9 | 11107.6 | 2751.0 |

## 3. MDE 3D ECEF [m] per orizzonte

### 40d CPU (original)

| Modello | 5 s | 30 s | 1 min | 5 min | 10 min | Media |
|---|---:|---:|---:|---:|---:|---:|
| AeroTraj-Ultra (ours) | 61.5 | 139.1 | 355.8 | 4,938.4 | 15,486.5 | 3,625.9 |
| AeroTraj-Ultra w/o MAF gate | 61.5 | 139.2 | 355.8 | 4,952.5 | 15,545.7 | 3,638.5 |
| Physics kinematic prior | 61.3 | 142.0 | 371.0 | 5,456.1 | 17,546.8 | 4,054.5 |
| FlightBERT++ V2 | 93.8 | 394.1 | 874.9 | 7,371.6 | 20,299.5 | 5,183.0 |
| Deep LSTM | 253.6 | 756.1 | 1,411.9 | 8,494.1 | 19,496.7 | 5,609.9 |
| CV (raw Doppler) | 61.9 | 151.0 | 388.1 | 5,590.4 | 17,840.7 | 4,139.7 |
| CV (least-squares) | n/a | n/a | n/a | n/a | n/a | n/a |

### 40d GPU (control)

| Modello | 5 s | 30 s | 1 min | 5 min | 10 min | Media |
|---|---:|---:|---:|---:|---:|---:|
| AeroTraj-Ultra (ours) | 61.0 | 133.7 | 336.0 | 4,337.6 | 12,213.5 | 3,019.5 |
| AeroTraj-Ultra w/o MAF gate | 61.0 | 133.8 | 336.1 | 4,339.6 | 12,223.6 | 3,021.3 |
| Physics kinematic prior | 61.3 | 142.0 | 369.2 | 5,450.7 | 17,546.8 | 4,052.2 |
| FlightBERT++ V2 | 79.2 | 259.6 | 596.6 | 5,999.5 | 16,621.4 | 4,190.7 |
| Deep LSTM | 130.1 | 476.9 | 996.2 | 6,632.3 | 16,519.8 | 4,523.6 |
| CV (raw Doppler) | 61.9 | 151.0 | 388.1 | 5,590.4 | 17,840.7 | 4,139.7 |
| CV (least-squares) | 78.1 | 264.7 | 604.4 | 6,309.1 | 18,929.4 | 4,580.5 |

### 90d GPU (nuovo)

| Modello | 5 s | 30 s | 1 min | 5 min | 10 min | Media |
|---|---:|---:|---:|---:|---:|---:|
| AeroTraj-Ultra (ours) | 61.2 | 121.7 | 304.4 | 3,955.9 | 11,107.6 | 2,751.0 |
| AeroTraj-Ultra w/o MAF gate | 61.2 | 121.8 | 304.8 | 3,957.2 | 11,117.2 | 2,752.7 |
| Physics kinematic prior | 62.3 | 146.6 | 383.3 | 5,654.4 | 18,182.8 | 4,202.2 |
| FlightBERT++ V2 | 75.3 | 223.4 | 521.6 | 5,667.9 | 15,588.5 | 3,935.6 |
| Deep LSTM | 112.9 | 379.4 | 790.3 | 6,288.3 | 15,963.1 | 4,276.0 |
| CV (raw Doppler) | 63.0 | 155.5 | 401.2 | 5,769.7 | 18,448.5 | 4,278.2 |
| CV (least-squares) | 79.6 | 271.4 | 621.4 | 6,537.1 | 19,661.5 | 4,751.8 |

## 4. Effetto dei dati (stessa ricetta di training)

- 40 giorni (GPU): **3,019.5 m**
- 90 giorni (GPU): **2,751.0 m**
- Variazione: **-8.9 %**

> Attenzione: i due modelli hanno test set *diversi* (finestra cronologica diversa).
> Per un confronto apples-to-apples usare `eval_ckpt.py` che valuta entrambi
> sullo stesso split mai visto in training.
