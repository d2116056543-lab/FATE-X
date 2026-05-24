$ErrorActionPreference = "Continue"
$pattern = "Baidu|baidu|Pan|Netdisk|aria2|IDM|Thunder|Download"
Get-Process |
  Where-Object { $_.ProcessName -match $pattern } |
  Select-Object ProcessName, Id, CPU, StartTime |
  Format-Table -AutoSize
