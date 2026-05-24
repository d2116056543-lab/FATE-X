$ErrorActionPreference = "Stop"
Set-Location "E:\sbw\ADAPT_repro\ADAPT"
Write-Host "---RUN_ADAPT FILE---"
Get-Item "src\tasks\run_adapt.py" | Select-Object FullName,Length,LastWriteTime | Format-List
Write-Host "---LATEST/BEST PATCH MARKERS---"
Select-String -Path "src\tasks\run_adapt.py" -Pattern "save_repro_checkpoint|checkpoint_latest|checkpoint_best|best_repro_score" | Format-Table -AutoSize
Write-Host "---PY COMPILE---"
wsl -d ADAPT-Ubuntu -- bash -lc "source /opt/conda/etc/profile.d/conda.sh; conda activate adapt; cd /mnt/e/sbw/ADAPT_repro/ADAPT; python -m py_compile src/tasks/run_adapt.py"
Write-Host "py_compile_ok"
