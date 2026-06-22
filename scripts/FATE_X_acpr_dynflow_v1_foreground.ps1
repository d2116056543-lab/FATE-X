param(
  [string]$Config = "configs\acpr_dynflow_v1_bddx_32f_224.yaml",
  [string]$OutputDir = "E:\sbw\FATE_Drive\active_runs\acpr_dynflow_v1_formal",
  [string]$Device = "cuda",
  [switch]$RequireReviewPass
)
$ErrorActionPreference = "Stop"
if ($RequireReviewPass -and -not (Test-Path ".background_runs\acpr_dynflow_v1_preflight\REVIEW_PASS_ACPR_DYNFLOW_V1.txt")) {
  throw "Missing REVIEW_PASS_ACPR_DYNFLOW_V1.txt for current SHA"
}
python -m fate_x.engine.supervise_acpr_dynflow_foreground --config $Config --output_dir $OutputDir --device $Device

