$ErrorActionPreference = "Stop"
$repo = "E:\sbw\ADAPT_repro\ADAPT"
Set-Location $repo

$cmd = @'
source /opt/conda/etc/profile.d/conda.sh
conda activate adapt
cd /mnt/e/sbw/ADAPT_repro/ADAPT
mkdir -p repro_logs
CUDA_VISIBLE_DEVICES=0 timeout 300s python src/tasks/run_adapt.py \
  --val_yaml BDDX/testing_32frames.yaml \
  --do_eval true \
  --do_train false \
  --eval_model_dir checkpoints/basemodel/checkpoints/ \
  --limited_samples 8 \
  --per_gpu_eval_batch_size 1 2>&1 | tee repro_logs/single_gpu_pretrained_eval_smoke.log
status=${PIPESTATUS[0]}
if [ "$status" = "124" ]; then
  echo "eval smoke timeout reached after 300s" | tee -a repro_logs/single_gpu_pretrained_eval_smoke.log
  exit 0
fi
exit "$status"
'@

wsl -d ADAPT-Ubuntu -- bash -lc $cmd
