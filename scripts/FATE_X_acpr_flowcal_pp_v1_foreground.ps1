param(
  [string]$Config = "configs/acpr_flowcal_pp_v1_bddx_32f_224.yaml",
  [string]$OutputDir = ".background_runs/acpr_flowcal_pp_v1_smoke",
  [string]$Device = "cuda"
)
$ErrorActionPreference = "Stop"
Write-Host "ACPR foreground runner attached to this console."
python -m fate_x.engine.audit_acpr_flowcal_pp --config $Config --output_dir "$OutputDir/preflight" --device $Device
python -m fate_x.engine.train_acpr_flowcal_pp --config $Config --output_dir "$OutputDir/train" --device $Device --epochs 1 --max_steps 8 --batch_size 1 --beam_size 1
