# AeroTraj-Ultra — 365-day run with 1.2M samples/epoch (150 epochs).
#
# The first 365-day run used 400k samples/epoch, i.e. 60M samples over 150 epochs.
# It never triggered early stopping and its validation loss was still decreasing at
# epoch 150, so the optimisation budget and not the data was the binding constraint.
# This run triples the samples per epoch: 180M samples, 3x the gradient steps, same
# data, same architecture, same recipe. Outputs go to separate directories so the
# two budgets can be compared directly.
#
# It waits for any running GPU job (the latency benchmark) to finish first, so the
# two do not contend for the device.
#
# Usage:  powershell -ExecutionPolicy Bypass -File run_365d_1200k_chain.ps1

$ErrorActionPreference = "Continue"
$PY   = "C:\Users\marco\anaconda3\python.exe"
$root = "C:\Users\marco\Desktop\FlightPREV2"
$log  = "$env:TEMP"
Set-Location $root

function Say($m) { Write-Output ("[{0}] {1}" -f (Get-Date).ToString("HH:mm:ss"), $m) }

Say "attesa che la GPU sia libera ..."
$deadline = (Get-Date).AddHours(1)
while ((Get-Date) -lt $deadline) {
    if (@(Get-Process python -ErrorAction SilentlyContinue).Count -eq 0) { break }
    Start-Sleep -Seconds 30
}
if (@(Get-Process python -ErrorAction SilentlyContinue).Count -ne 0) {
    Say "TIMEOUT: un processo python e' ancora attivo, esco."
    exit 1
}
Say "GPU libera."

Say "training 365 giorni, 150 epoche, 1.2M campioni/epoca, torch.compile ..."
& $PY -u run.py --config config_365d_1200k.json --fresh --skip-baselines --compile reduce-overhead *> "$log\train365_1200k.log"
Say "training terminato: $log\train365_1200k.log"

Say "ultime righe del log:"
Get-Content "$log\train365_1200k.log" -Tail 26
Say "fatto."
