# AeroTraj-Ultra — 10-second grid: comparison, dataset, verification, training.
#
# Runs unattended, in this order:
#   1. wait for the GPU (the 1.2M-sample 5 s run is still training);
#   2. evaluate 90d / 365d-400k / 365d-1.2M on BOTH test splits, so the two
#      training budgets are compared apples-to-apples;
#   3. derive data365_10s from data365 by decimating in time (no re-download);
#   4. VERIFY the new grid — in particular that the Gray-Code delta lattices do
#      not saturate, since a coarser grid doubles every per-step displacement;
#      if the verification fails the chain STOPS instead of wasting the GPU;
#   5. train config_365d_10s.json: dt=10 s, 12-point window = 120 s of history,
#      horizons 10/30/60/90/180/300 s, 150 epochs at 1.2M samples/epoch.
#
# Usage:  powershell -ExecutionPolicy Bypass -File run_10s_chain.ps1

$ErrorActionPreference = "Continue"
$PY   = "C:\Users\marco\anaconda3\python.exe"
$root = "C:\Users\marco\Desktop\FlightPREV2"
$log  = "$env:TEMP"
Set-Location $root

function Say($m) { Write-Output ("[{0}] {1}" -f (Get-Date).ToString("HH:mm:ss"), $m) }

Say "1) attesa che la GPU sia libera ..."
$deadline = (Get-Date).AddHours(3)
while ((Get-Date) -lt $deadline) {
    if (@(Get-Process python -ErrorAction SilentlyContinue).Count -eq 0) { break }
    Start-Sleep -Seconds 30
}
if (@(Get-Process python -ErrorAction SilentlyContinue).Count -ne 0) {
    Say "TIMEOUT in attesa della GPU, esco."; exit 1
}
Say "   GPU libera."

Say "2) confronto 90d / 365d-400k / 365d-1.2M sui due test set ..."
& $PY -u eval_cross.py --model "90d=config_90d_300s.json=checkpoints_bench/ours.pt" --model "365d-400k=config_365d.json=checkpoints_365/aerotraj_ultra_best.pt" --model "365d-1.2M=config_365d_1200k.json=checkpoints_365_1200k/aerotraj_ultra_best.pt" --out results_365/cross_budgets.json *> "$log\cross_budgets.log"
Say "   confronto salvato in results_365/cross_budgets.json"
Get-Content "$log\cross_budgets.log" -Tail 12 | ForEach-Object { Say "   | $_" }

Say "3) decimazione temporale data365 -> data365_10s ..."
& $PY -u build_10s_dataset.py --src data365 --dst data365_10s --factor 2 --inp-seq-len 12 --max-horizon-step 30 *> "$log\build_10s.log"
if ($LASTEXITCODE -ne 0) { Say "   ERRORE nella decimazione, esco."; exit 1 }
Get-Content "$log\build_10s.log" | ForEach-Object { Say "   | $_" }

Say "4) verifica della griglia a 10 s (bit dei delta, round-trip, forward) ..."
& $PY -u verify_10s.py --config config_365d_10s.json *> "$log\verify_10s.log"
$vok = $LASTEXITCODE
Get-Content "$log\verify_10s.log" | ForEach-Object { Say "   | $_" }
if ($vok -ne 0) { Say "   VERIFICA FALLITA (exit $vok): mi fermo prima di sprecare GPU."; exit 1 }

Say "5) training: dt=10 s, 120 s di storia, 6 orizzonti, 150 epoche @ 1.2M ..."
& $PY -u run.py --config config_365d_10s.json --fresh --skip-baselines --compile reduce-overhead *> "$log\train_10s.log"
Say "   training terminato: $log\train_10s.log"
Get-Content "$log\train_10s.log" -Tail 26 | ForEach-Object { Say "   | $_" }
Say "fatto."
