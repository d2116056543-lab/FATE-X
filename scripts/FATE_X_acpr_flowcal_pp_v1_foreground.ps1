param(
  [string]$Config = "configs/acpr_flowcal_pp_v1_bddx_32f_224.yaml",
  [string]$OutputDir = ".background_runs/acpr_flowcal_pp_v1_smoke",
  [string]$Device = "cuda",
  [switch]$RequireReviewPass
)
$ErrorActionPreference = "Stop"
Write-Host "ACPR foreground runner attached to this console."

if ($RequireReviewPass) {
  $passFile = ".background_runs/acpr_flowcal_pp_v1_preflight/REVIEW_PASS_ACPR_FLOWCAL_PP_V1.txt"
  if (-not (Test-Path $passFile)) {
    throw "Formal ACPR-FlowCal++ training is blocked: missing $passFile. Run the full audit gates and write the review pass for the exact clean pushed HEAD first."
  }
}

python -m fate_x.engine.probe_acpr_flowcal_memory --config $Config --output_dir "$OutputDir/preflight" --device $Device
python -m fate_x.engine.run_acpr_flowcal_preflight_gates --config $Config --output_dir "$OutputDir/preflight" --device $Device
python -m fate_x.engine.audit_acpr_flowcal_pp --config $Config --output_dir "$OutputDir/preflight" --device $Device
python -m fate_x.engine.supervise_acpr_flowcal_foreground `
  --output_dir "$OutputDir/preflight" `
  --heartbeat_seconds 60 `
  --command python -m fate_x.engine.train_acpr_flowcal_pp --config $Config --output_dir "$OutputDir/train" --device $Device --epochs 1 --max_steps 8 --batch_size 1 --beam_size 1
