param(
  [string]$Config = "configs/acpr_dynflow_swin_v1_bddx_32f_224.yaml",
  [switch]$RequireReviewPass
)

$ErrorActionPreference = "Stop"
$argsList = @(
  "-m",
  "fate_x.engine.supervise_acpr_dynflow_swin_foreground",
  "--config",
  $Config
)
if ($RequireReviewPass) {
  $pass = ".background_runs/acpr_dynflow_swin_v1_preflight/REVIEW_PASS_ACPR_DYNFLOW_SWIN_V1.txt"
  if (!(Test-Path $pass)) { throw "review pass missing: $pass" }
  $argsList += "--require_review_pass"
}
python @argsList
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
