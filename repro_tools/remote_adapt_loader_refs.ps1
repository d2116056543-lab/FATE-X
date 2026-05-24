$ErrorActionPreference = "Continue"
$repo = "E:\sbw\ADAPT_repro\ADAPT"
Set-Location $repo
Write-Host "---RUN_ADAPT DATA ARGS---"
Select-String -Path "src\tasks\run_adapt.py","src\configs\config.py","src\datasets\*.py" -Pattern "data_dir|train_yaml|val_yaml|test_yaml|caption_linelist|lineidx|img:" -CaseSensitive:$false | Select-Object Path,LineNumber,Line | Format-Table -AutoSize
Write-Host "---ALL CONFIGS NAMES---"
Get-ChildItem "src\configs\VidSwinBert" -File | Select-Object Name,Length | Format-Table -AutoSize
