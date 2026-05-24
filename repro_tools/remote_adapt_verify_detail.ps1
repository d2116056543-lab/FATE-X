$ErrorActionPreference = "Stop"
$repo = "E:\sbw\ADAPT_repro\ADAPT"
Set-Location $repo

Write-Host "---SCRIPT LIST---"
Get-ChildItem scripts -File | Sort-Object Name | Select-Object Name,Length,LastWriteTime | Format-Table -AutoSize

Write-Host "---OFFICIAL TEST SCRIPT CANDIDATES---"
Get-ChildItem scripts -File | Where-Object { $_.Name -match "test|eval|BDD" } | Select-Object Name,Length | Format-Table -AutoSize

Write-Host "---DATASET COUNT SUMMARY---"
$jsonPath = "repro_logs\adapt_preprocessed_dataset_validation.json"
if (Test-Path $jsonPath) {
  $j = Get-Content $jsonPath -Raw | ConvertFrom-Json
  foreach ($root in @("datasets/BDDX","datasets_part/BDDX","datasets/BDDX_des","datasets/BDDX_exp","datasets_part/BDDX_des","datasets_part/BDDX_exp")) {
    if ($j.dirs.PSObject.Properties.Name -contains $root) {
      $d = $j.dirs.$root
      Write-Host ("[{0}]" -f $root)
      foreach ($split in @("training","validation","testing")) {
        if ($d.splits.PSObject.Properties.Name -contains $split) {
          $s = $d.splits.$split
          $label = $s.PSObject.Properties["$split.label.tsv"].Value
          $cap = $s.PSObject.Properties["$split.caption.tsv"].Value
          $coco = $s.PSObject.Properties["${split}_32frames_caption_coco_format.json"].Value
          $frame = $null
          foreach ($p in $s.PSObject.Properties.Name) {
            if ($p -match "frame_tsv.*img.tsv|frame_tsv_part.*img.tsv") { $frame = $s.$p; break }
          }
          Write-Host ("  {0}: labels={1} captions={2} coco_annotations={3} frame_tsv_bytes={4}" -f $split, $label.lines, $cap.lines, $coco.annotations, $frame.size)
        }
      }
    }
  }
}

Write-Host "---MODELS---"
Get-ChildItem models -Recurse -File -ErrorAction SilentlyContinue |
  Where-Object { $_.FullName -match "bert-base-uncased|video_swin|swin|pytorch_model|config.json|vocab.txt" } |
  Select-Object FullName,Length | Format-Table -AutoSize

Write-Host "---CHECKPOINTS---"
Get-ChildItem checkpoints -Recurse -File -ErrorAction SilentlyContinue |
  Select-Object FullName,Length | Format-Table -AutoSize

Write-Host "---WSL ENV CHECK---"
wsl -d ADAPT-Ubuntu -- bash -lc "source /opt/conda/etc/profile.d/conda.sh && conda activate adapt && cd /mnt/e/sbw/ADAPT_repro/ADAPT && python - <<'PY'
import torch, apex, sklearn, os, shutil, subprocess
print('python_ok')
print('torch', torch.__version__, 'cuda', torch.version.cuda, 'cuda_available', torch.cuda.is_available())
print('gpu', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')
print('apex', getattr(apex, '__file__', 'no_file'))
print('sklearn', sklearn.__version__)
print('java', shutil.which('java'))
PY"

Write-Host "---DISK---"
Get-PSDrive -PSProvider FileSystem | Select-Object Name,Free,Used,Root | Format-Table -AutoSize
