# ADAPT Official Runbook For Later Linux Reproduction

Last update: 2026-05-21 14:35:57

This runbook records the official commands and local paths for the `jxbbb/ADAPT` reproduction. It intentionally does not patch official code.

## 1. Official environment route

The ADAPT README recommends Docker and states Linux with NVIDIA GPUs is required. After Linux/Docker is available, run from the repository root:

```bash
cd /path/to/ADAPT
export CUDA_VISIBLE_DEVICES=0,1,2,3
sh launch_container.sh
```

The official `launch_container.sh` binds:

```text
repo root       -> /videocap
datasets        -> /videocap/datasets
datasets_part   -> /videocap/datasets_part
models          -> /videocap/models
output          -> /videocap/output
```

On the current remote Windows workspace, the equivalent repo root is:

```text
E:\sbw\ADAPT_repro\ADAPT
```

## 2. Required data before any official run

The official ADAPT scripts expect BDD-X under:

```text
E:\sbw\ADAPT_repro\ADAPT\datasets\BDDX
```

At minimum, these currently missing files must exist:

```text
datasets\BDDX\training_32frames.yaml
datasets\BDDX\testing_32frames.yaml
datasets\BDDX\frame_tsv\bddx_train_32frame.img.tsv
datasets\BDDX\frame_tsv\bddx_train_32frame.img.lineidx
datasets\BDDX\frame_tsv\bddx_test_32frame.img.tsv
datasets\BDDX\frame_tsv\bddx_test_32frame.img.lineidx
```

Use the Baidu Pan link from the README:

```text
https://pan.baidu.com/s/1_eW-kLVBhf8lWGObGa4i9A?pwd=7zrz
password: 7zrz
```

## 3. Required checkpoint before official evaluation

The official test script uses:

```bash
EVAL_DIR='checkpoints/basemodel/checkpoints/'
CUDA_VISIBLE_DEVICES=5 python src/tasks/run_adapt.py \
  --val_yaml BDDX/testing_32frames.yaml \
  --do_eval true \
  --do_train false \
  --eval_model_dir $EVAL_DIR
```

So after manual Google Drive download, preserve the checkpoint layout under:

```text
E:\sbw\ADAPT_repro\ADAPT\checkpoints\basemodel\checkpoints\
```

Official checkpoint folder from README:

```text
https://drive.google.com/drive/folders/1GYO9MVgrDMBUXdULgs5mzmcpDstAGCn4?usp=share_link
```

## 4. Official multitask training command

The repository script `scripts/BDDX_multitask.sh` runs:

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 \
NCCL_P2P_DISABLE=1 \
OMPI_COMM_WORLD_SIZE="4" \
python -m torch.distributed.launch --nproc_per_node=4 --nnodes=1 --node_rank=0 --master_port=45978 src/tasks/run_adapt.py \
  --config src/configs/VidSwinBert/BDDX_multi_default.json \
  --train_yaml BDDX/training_32frames.yaml \
  --val_yaml BDDX/testing_32frames.yaml \
  --per_gpu_train_batch_size 4 \
  --per_gpu_eval_batch_size 16 \
  --num_train_epochs 40 \
  --learning_rate 0.0002 \
  --max_num_frames 32 \
  --pretrained_2d 0 \
  --backbone_coef_lr 0.05 \
  --mask_prob 0.5 \
  --max_masked_token 45 \
  --zero_opt_stage 1 \
  --mixed_precision_method deepspeed \
  --deepspeed_fp16 \
  --gradient_accumulation_steps 4 \
  --learn_mask_enabled \
  --loss_sparse_w 0.1 \
  --use_sep_cap \
  --multitask \
  --signal_types course speed \
  --loss_sensor_w 0.05 \
  --max_grad_norm 1 \
  --output_dir ./output/multitask/sensor_course_speed
```

## 5. Official caption-only and signal-only scripts

Caption-only:

```bash
sh scripts/BDDX_only_caption.sh
```

Signal-only:

```bash
sh scripts/BDDX_only_signal.sh
```

## 6. Current Windows sbw39 status

This is the status from `repro_logs/preflight_status.json`:

```text
CUDA available: yes, NVIDIA GeForce RTX 4090
Official BDDX data ready: no
Official Windows runtime ready: no
deepspeed import: missing
apex import: missing
```

Therefore the correct next step is not to run a patched Windows version and call it official. The exact reproduction path is:

```text
1. Put official BDDX preprocessed files into datasets\BDDX.
2. Put official checkpoints into checkpoints.
3. Run inside Linux/Docker with Apex/DeepSpeed support.
4. Execute scripts/BDDX_test.sh and scripts/BDDX_multitask.sh as needed.
```


