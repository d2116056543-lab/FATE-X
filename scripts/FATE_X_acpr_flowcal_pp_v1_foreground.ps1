param(
  [string]$Config = "configs/acpr_flowcal_pp_v1_bddx_32f_224.yaml",
  [string]$OutputDir = ".background_runs/acpr_flowcal_pp_v1_formal_foreground",
  [string]$Device = "cuda",
  [int]$Epochs = 21,
  [int]$MaxSteps = 0,
  [int]$BatchSize = 4,
  [int]$GradientAccumulationSteps = 16,
  [int]$BeamSize = 3,
  [int]$CheckpointEverySteps = 500,
  [string]$ReviewPassDir = "",
  [switch]$RunPreflight,
  [switch]$RequireReviewPass
)
$ErrorActionPreference = "Stop"
Write-Host "ACPR foreground runner attached to this console."

if ($RequireReviewPass) {
  if ([string]::IsNullOrWhiteSpace($ReviewPassDir)) {
    throw "Formal ACPR-FlowCal++ training is blocked: -RequireReviewPass needs -ReviewPassDir pointing to the review artifacts for the current clean pushed HEAD."
  }
  $passFile = Join-Path $ReviewPassDir "REVIEW_PASS_ACPR_FLOWCAL_PP_V1.txt"
  if (-not (Test-Path $passFile)) {
    throw "Formal ACPR-FlowCal++ training is blocked: missing $passFile. Run the full audit gates and write the review pass for the exact clean pushed HEAD first."
  }
  $head = (git rev-parse HEAD).Trim()
  $passText = Get-Content $passFile -Raw
  if ($passText -notmatch [regex]::Escape($head)) {
    throw "Formal ACPR-FlowCal++ training is blocked: $passFile is not bound to current HEAD $head."
  }
}

if ($RunPreflight) {
  python -m fate_x.engine.probe_acpr_flowcal_memory --config $Config --output_dir "$OutputDir/preflight" --device $Device
  python -m fate_x.engine.run_acpr_flowcal_preflight_gates --config $Config --output_dir "$OutputDir/preflight" --device $Device
  python -m fate_x.engine.audit_acpr_flowcal_pp --config $Config --output_dir "$OutputDir/preflight" --device $Device
} else {
  New-Item -ItemType Directory -Force -Path "$OutputDir/preflight" | Out-Null
}

python -m fate_x.engine.supervise_acpr_flowcal_foreground `
  --output_dir "$OutputDir/preflight" `
  --heartbeat_seconds 60 `
  --command python -m fate_x.engine.train_acpr_flowcal_pp --config $Config --output_dir "$OutputDir/train" --device $Device --epochs $Epochs --max_steps $MaxSteps --batch_size $BatchSize --beam_size $BeamSize --gradient_accumulation_steps $GradientAccumulationSteps --checkpoint_every_steps $CheckpointEverySteps
