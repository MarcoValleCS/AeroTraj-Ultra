# AeroTraj-Ultra — chained runner for the 365-day study.
#
#   1. wait for the 365-day dataset build to finish (it is resumable: the
#      per-day cache in data365/_cache_days means a restart never re-downloads);
#   2. re-measure the inference latency on an IDLE machine (the earlier
#      retrieval numbers were taken while the build was saturating the CPU, and
#      the retrieval path is launch-bound, so it is the most load-sensitive part);
#   3. train 150 epochs on the 365-day split with torch.compile enabled.
#
# Usage:  powershell -ExecutionPolicy Bypass -File run_365d_chain.ps1

$ErrorActionPreference = "Continue"
$PY   = "C:\Users\marco\anaconda3\python.exe"
$root = "C:\Users\marco\Desktop\FlightPREV2"
$log  = "$env:TEMP"
Set-Location $root

function Say($m) { Write-Output ("[{0}] {1}" -f (Get-Date).ToString("HH:mm:ss"), $m) }

Say "attesa della build del dataset a 365 giorni ..."
$deadline = (Get-Date).AddHours(10)
$needed = @("train_points.npy", "val_points.npy", "test_points.npy", "dataset_stats.json")
while ((Get-Date) -lt $deadline) {
    $built   = ($needed | Where-Object { -not (Test-Path "$root\data365\$_") }).Count -eq 0
    $running = @(Get-Process python -ErrorAction SilentlyContinue).Count -gt 0
    if ($built -and -not $running) { break }
    Start-Sleep -Seconds 120
}
if (($needed | Where-Object { -not (Test-Path "$root\data365\$_") }).Count -ne 0) {
    Say "TIMEOUT: la build non e' terminata entro 10 ore, esco."
    exit 1
}
Start-Sleep -Seconds 15
Say "dataset pronto. file:"
Get-ChildItem "$root\data365" -File | ForEach-Object {
    Say ("   {0,-22} {1,8:N2} GB" -f $_.Name, ($_.Length / 1GB))
}

Say "misura pulita della latenza (macchina libera) ..."
& $PY -u bench_latency_opt.py *> "$log\lat_opt_clean.log"
Say "latenza: $log\lat_opt_clean.log"

Say "training 365 giorni, 150 epoche, torch.compile(reduce-overhead) ..."
& $PY -u run.py --config config_365d.json --fresh --skip-baselines --compile reduce-overhead *> "$log\train365.log"
Say "training terminato: $log\train365.log"

Say "ultime righe del log di training:"
Get-Content "$log\train365.log" -Tail 25
Say "fatto."
