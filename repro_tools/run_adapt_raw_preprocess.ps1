param(
  [string]$RepoRoot = "E:\sbw\ADAPT_repro\ADAPT",
  [string]$Python = "E:\Anaconda\envs\sbw39\python.exe",
  [string]$RawVideoDir = "E:\sbw\ADAPT_repro\ADAPT\datasets\BDDX_raw\videos",
  [string]$CsvPath = "E:\sbw\ADAPT_repro\ADAPT\datasets\BDDX_raw\annotations\BDD-X-Dataset\BDD-X-Annotations_v1.csv",
  [string]$CaptionsJson = "E:\sbw\ADAPT_repro\ADAPT\datasets\BDDX\captions_BDDX.json",
  [string]$FrameDir = "E:\sbw\ADAPT_repro\ADAPT\datasets\BDDX_raw\frames\32frames",
  [string]$FrameTsvDir = "E:\sbw\ADAPT_repro\ADAPT\datasets\BDDX\frame_tsv",
  [string]$DatasetDir = "E:\sbw\ADAPT_repro\ADAPT\datasets\BDDX",
  [int]$NumWorkers = 16,
  [switch]$SkipFrameExtraction,
  [switch]$SkipFrameTsv,
  [switch]$SkipDatasetTsv
)

$ErrorActionPreference = "Stop"
Set-Location $RepoRoot

$LogDir = Join-Path $RepoRoot "repro_logs"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
New-Item -ItemType Directory -Force -Path $RawVideoDir | Out-Null
New-Item -ItemType Directory -Force -Path $FrameDir | Out-Null
New-Item -ItemType Directory -Force -Path $FrameTsvDir | Out-Null
New-Item -ItemType Directory -Force -Path $DatasetDir | Out-Null

Write-Host "ADAPT raw preprocessing paths:"
Write-Host "  RepoRoot      = $RepoRoot"
Write-Host "  RawVideoDir   = $RawVideoDir"
Write-Host "  CsvPath       = $CsvPath"
Write-Host "  CaptionsJson  = $CaptionsJson"
Write-Host "  FrameDir      = $FrameDir"
Write-Host "  FrameTsvDir   = $FrameTsvDir"
Write-Host "  DatasetDir    = $DatasetDir"

if (!(Test-Path $CaptionsJson)) {
  Write-Host "captions_BDDX.json not found; building from public BDD-X CSV."
  & $Python repro_tools\build_captions_bddx_from_csv.py `
    --csv $CsvPath `
    --output $CaptionsJson `
    2>&1 | Tee-Object -FilePath (Join-Path $LogDir "build_captions_bddx_from_csv.log")
}

$env:ADAPT_BDDX_RAW_VIDEO_DIR = $RawVideoDir
$env:ADAPT_BDDX_FRAME_DIR = $FrameDir
$env:ADAPT_BDDX_CAPTIONS_JSON = $CaptionsJson
$env:ADAPT_BDDX_FRAME_TSV_DIR = $FrameTsvDir
$env:ADAPT_BDDX_DATASET_DIR = $DatasetDir
$env:ADAPT_BDDX_NUM_WORKERS = [string]$NumWorkers
$env:ADAPT_BDDX_NUM_FRAMES = "32"
$env:ADAPT_BDDX_IMAGE_SIZE = "256"

& $Python repro_tools\check_adapt_raw_inputs.py `
  --csv $CsvPath `
  --captions_json $CaptionsJson `
  --raw_video_dir $RawVideoDir `
  --frame_dir $FrameDir `
  --dataset_dir $DatasetDir `
  --summary (Join-Path $LogDir "raw_preprocess_preflight.json") `
  2>&1 | Tee-Object -FilePath (Join-Path $LogDir "raw_preprocess_preflight.log")

if (!$SkipFrameExtraction) {
  Write-Host "Running author src/prepro/extract_frames.py"
  & $Python src\prepro\extract_frames.py `
    2>&1 | Tee-Object -FilePath (Join-Path $LogDir "raw_extract_frames.log")
}

if (!$SkipFrameTsv) {
  Write-Host "Running author src/prepro/create_image_frame_tsv.py"
  & $Python src\prepro\create_image_frame_tsv.py `
    2>&1 | Tee-Object -FilePath (Join-Path $LogDir "raw_create_image_frame_tsv.log")
}

if (!$SkipDatasetTsv) {
  Write-Host "Running author src/prepro/tsv_preproc_BDDX.py"
  & $Python src\prepro\tsv_preproc_BDDX.py `
    2>&1 | Tee-Object -FilePath (Join-Path $LogDir "raw_tsv_preproc_BDDX.log")
}

Write-Host "Organizing frame TSVs and YAML files to *_32frames names used by scripts/BDDX_*.sh"
& $Python repro_tools\organize_bddx_prepro_outputs.py `
  --dataset_dir $DatasetDir `
  --frame_tsv_dir $FrameTsvDir `
  --num_frames 32 `
  --image_size 256 `
  --summary (Join-Path $LogDir "raw_preprocess_organized_summary.json") `
  2>&1 | Tee-Object -FilePath (Join-Path $LogDir "raw_preprocess_organized_summary.log")

Write-Host "ADAPT raw preprocessing completed."
