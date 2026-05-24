$ErrorActionPreference = "Continue"
Set-Location "E:\sbw\ADAPT_repro\ADAPT"
if (Get-Command rg -ErrorAction SilentlyContinue) {
  rg -n "max_iter|limited|samples|debug|num_train_epochs|logging_steps" src\configs src\tasks src\datasets
} else {
  Select-String -Path "src\configs\*.py","src\tasks\*.py","src\datasets\*.py" -Pattern "max_iter|limited|samples|debug|num_train_epochs|logging_steps"
}
