# FlowTrace PMT V1 Progress

Generated: 2026-06-18 23:55 CST

## Session Summary

User reported that the smoke was taking around six hours. Investigation showed the smoke was not actually bounded by training steps. A hard train-step limiter was implemented and verified. The smoke now stops at `8/8`, but the full strict gate remains failed due to missing PMT/reason gradients and missing intervention evidence.

## Commands and Verification

### Failing Test Before Fix

```text
python -m pytest tests/test_flowtrace_config_contract.py::test_flowtrace_bridge_adds_hard_train_step_limit_for_real_smoke -q
FAILED: ValueError: '--flowtrace_max_train_steps' is not in list
```

### Tests After Fix

```text
python -m pytest tests/test_flowtrace_config_contract.py tests/test_flowtrace_strict_audit_contract.py tests/test_flowtrace_losses.py tests/test_flowtrace_real_smoke_evidence.py tests/test_flowtrace_e2e_smoke.py tests/test_flowtrace_sinkhorn_transport.py tests/test_flowtrace_token_pmt.py -q
24 passed, 10 warnings in 36.78s
```

### Bounded Smoke Command

Command manifest:

```json
{
  "config_path": "configs/flowtrace_pmt_v1_bddx_32f_224.yaml",
  "output_dir": ".background_runs/flowtrace_pmt_v1_real_smoke_wsl_bounded8_hardlimit",
  "direct_image_training": true,
  "feature_cache_enabled": false,
  "token_cache_enabled": false,
  "train_yaml": "datasets_part/BDDX/training_32frames.yaml",
  "test_yaml_as_val_yaml": "datasets/BDDX/testing_32frames.yaml",
  "adapt_checkpoint": "checkpoints/basemodel/checkpoints/model.bin",
  "bert_dir": "models/captioning/bert-base-uncased",
  "epochs": 1,
  "micro_batch": 1,
  "gradient_accumulation_steps": 1,
  "effective_batch": 1,
  "flowtrace_max_train_steps": 8,
  "max_eval_samples": 8,
  "flowtrace_smoke_evidence": ".background_runs/flowtrace_pmt_v1_real_smoke_wsl_bounded8_hardlimit/flowtrace_real_smoke_summary.json",
  "command": [
    "/opt/conda/envs/adapt/bin/python",
    "-u",
    "src/tasks/run_adapt.py",
    "--config",
    "src/configs/VidSwinBert/BDDX_multi_default.json",
    "--do_train",
    "--evaluate_during_training",
    "--data_dir",
    ".",
    "--train_yaml",
    "datasets_part/BDDX/training_32frames.yaml",
    "--val_yaml",
    "datasets/BDDX/testing_32frames.yaml",
    "--model_name_or_path",
    "models/captioning/bert-base-uncased",
    "--pretrained_checkpoint",
    "checkpoints/basemodel/checkpoints/",
    "--output_dir",
    ".background_runs/flowtrace_pmt_v1_real_smoke_wsl_bounded8_hardlimit",
    "--num_train_epochs",
    "1",
    "--per_gpu_train_batch_size",
    "1",
    "--per_gpu_eval_batch_size",
    "1",
    "--gradient_accumulation_steps",
    "1",
    "--learning_rate",
    "0.0002",
    "--backbone_coef_lr",
    "0.05",
    "--max_num_frames",
    "32",
    "--img_res",
    "224",
    "--pretrained_2d",
    "false",
    "--kinetics",
    "600",
    "--vidswin_size",
    "base",
    "--grid_feat",
    "true",
    "--mask_prob",
    "0.5",
    "--max_masked_tokens",
    "45",
    "--max_gen_length",
    "15",
    "--num_beams",
    "1",
    "--use_sep_cap",
    "true",
    "--multitask",
    "true",
    "--signal_types",
    "course",
    "speed",
    "--loss_sensor_w",
    "0.05",
    "--max_grad_norm",
    "1.0",
    "--num_workers",
    "4",
    "--fate_x_enabled",
    "true",
    "--video_token_reducer",
    "none",
    "--temporal_evidence_memory",
    "none",
    "--fate_x_reduce_control",
    "false",
    "--fate_x_text_reduce_only",
    "true",
    "--flowtrace_enabled",
    "true",
    "--flowtrace_state_dim",
    "256",
    "--flowtrace_pmt_rank",
    "32",
    "--learn_mask_enabled",
    "--attn_mask_type",
    "learn_vid_att",
    "--loss_sparse_w",
    "0.1",
    "--mixed_precision_method",
    "deepspeed",
    "--deepspeed_fp16",
    "true",
    "--deepspeed_bf16",
    "false",
    "--zero_opt_stage",
    "1",
    "--limited_samples",
    "8",
    "--flowtrace_max_train_steps",
    "8",
    "--limited_eval_samples",
    "8",
    "--flowtrace_smoke_evidence",
    ".background_runs/flowtrace_pmt_v1_real_smoke_wsl_bounded8_hardlimit/flowtrace_real_smoke_summary.json"
  ]
}
```

Run manifest:

```json
{
  "engine": "flowtrace_pmt_v1_real_adapt_direct_image",
  "formal_trainer": true,
  "feature_cache_enabled": false,
  "token_cache_enabled": false,
  "command": [
    "/opt/conda/envs/adapt/bin/python",
    "-u",
    "src/tasks/run_adapt.py",
    "--config",
    "src/configs/VidSwinBert/BDDX_multi_default.json",
    "--do_train",
    "--evaluate_during_training",
    "--data_dir",
    ".",
    "--train_yaml",
    "datasets_part/BDDX/training_32frames.yaml",
    "--val_yaml",
    "datasets/BDDX/testing_32frames.yaml",
    "--model_name_or_path",
    "models/captioning/bert-base-uncased",
    "--pretrained_checkpoint",
    "checkpoints/basemodel/checkpoints/",
    "--output_dir",
    ".background_runs/flowtrace_pmt_v1_real_smoke_wsl_bounded8_hardlimit",
    "--num_train_epochs",
    "1",
    "--per_gpu_train_batch_size",
    "1",
    "--per_gpu_eval_batch_size",
    "1",
    "--gradient_accumulation_steps",
    "1",
    "--learning_rate",
    "0.0002",
    "--backbone_coef_lr",
    "0.05",
    "--max_num_frames",
    "32",
    "--img_res",
    "224",
    "--pretrained_2d",
    "false",
    "--kinetics",
    "600",
    "--vidswin_size",
    "base",
    "--grid_feat",
    "true",
    "--mask_prob",
    "0.5",
    "--max_masked_tokens",
    "45",
    "--max_gen_length",
    "15",
    "--num_beams",
    "1",
    "--use_sep_cap",
    "true",
    "--multitask",
    "true",
    "--signal_types",
    "course",
    "speed",
    "--loss_sensor_w",
    "0.05",
    "--max_grad_norm",
    "1.0",
    "--num_workers",
    "4",
    "--fate_x_enabled",
    "true",
    "--video_token_reducer",
    "none",
    "--temporal_evidence_memory",
    "none",
    "--fate_x_reduce_control",
    "false",
    "--fate_x_text_reduce_only",
    "true",
    "--flowtrace_enabled",
    "true",
    "--flowtrace_state_dim",
    "256",
    "--flowtrace_pmt_rank",
    "32",
    "--learn_mask_enabled",
    "--attn_mask_type",
    "learn_vid_att",
    "--loss_sparse_w",
    "0.1",
    "--mixed_precision_method",
    "deepspeed",
    "--deepspeed_fp16",
    "true",
    "--deepspeed_bf16",
    "false",
    "--zero_opt_stage",
    "1",
    "--limited_samples",
    "8",
    "--flowtrace_max_train_steps",
    "8",
    "--limited_eval_samples",
    "8",
    "--flowtrace_smoke_evidence",
    ".background_runs/flowtrace_pmt_v1_real_smoke_wsl_bounded8_hardlimit/flowtrace_real_smoke_summary.json"
  ]
}
```

## Important Smoke Log Lines

```text
06/18/2026 23:34:20 - INFO - __main__ -   FlowTrace hard smoke train-step limit enabled: 8; capped max_global_step=8, max_iter=8
06/18/2026 23:34:24 - INFO - __main__ -   input_ids = torch.Size([1, 30])
06/18/2026 23:34:24 - INFO - __main__ -   img_feats = torch.Size([1, 32, 3, 224, 224])
06/18/2026 23:34:24 - INFO - __main__ -   car_info = torch.Size([1, 2, 32])
06/18/2026 23:34:40 - WARNING - tensorboardX.x2num -   NaN or Inf found in input tensor.
06/18/2026 23:35:08 - INFO - __main__ -   Save checkpoint to .background_runs/flowtrace_pmt_v1_real_smoke_wsl_bounded8_hardlimit/checkpoint_latest
06/18/2026 23:35:08 - INFO - __main__ -   Perform evaluation at iteration 1, global_step 1
06/18/2026 23:35:48 - INFO - __main__ -   evaluation result saved to .background_runs/flowtrace_pmt_v1_real_smoke_wsl_bounded8_hardlimit/checkpoint-0-1/pred.BDDX.testing_32frames.beam1.max15.eval.json
06/18/2026 23:35:55 - INFO - __main__ -   signal evaluation skipped: no valid control rows; wrote .background_runs/flowtrace_pmt_v1_real_smoke_wsl_bounded8_hardlimit/checkpoint-0-1/pred.BDDX.testing_32frames.beam1.max15.signal_unavailable.json
06/18/2026 23:36:24 - INFO - __main__ -   Save checkpoint to .background_runs/flowtrace_pmt_v1_real_smoke_wsl_bounded8_hardlimit/checkpoint_latest
06/18/2026 23:36:51 - INFO - __main__ -   Save checkpoint to .background_runs/flowtrace_pmt_v1_real_smoke_wsl_bounded8_hardlimit/checkpoint_best
06/18/2026 23:36:52 - WARNING - tensorboardX.x2num -   NaN or Inf found in input tensor.
06/18/2026 23:36:53 - WARNING - tensorboardX.x2num -   NaN or Inf found in input tensor.
06/18/2026 23:36:55 - WARNING - tensorboardX.x2num -   NaN or Inf found in input tensor.
06/18/2026 23:36:56 - WARNING - tensorboardX.x2num -   NaN or Inf found in input tensor.
[2026-06-18 23:36:56,643] [INFO] [unfused_optimizer.py:281:_update_scale] Grad overflow on iteration: 4
06/18/2026 23:36:57 - WARNING - tensorboardX.x2num -   NaN or Inf found in input tensor.
06/18/2026 23:36:59 - WARNING - tensorboardX.x2num -   NaN or Inf found in input tensor.
06/18/2026 23:37:00 - WARNING - tensorboardX.x2num -   NaN or Inf found in input tensor.
06/18/2026 23:37:00 - INFO - __main__ -   eta: 0:00:00  iter: 8  global_step: 8  speed: 0.1 images/sec  loss: 10.3598 (10.4238)  loss_sparsity: 0.3712 (0.3712)  acc: 0.0000 (0.0000)  loss_sensor: 0.1289 (0.1280)  batch_time: 1.4393 (1.3855)  data_time: 0.0003 (0.0003)  lr (Visual Encoder): 2.50e-06  lr (LM): 5.00e-05  max mem: 7738
06/18/2026 23:37:29 - INFO - __main__ -   Save checkpoint to .background_runs/flowtrace_pmt_v1_real_smoke_wsl_bounded8_hardlimit/checkpoint_latest
06/18/2026 23:37:29 - INFO - __main__ -   Perform evaluation at iteration 8, global_step 8
06/18/2026 23:38:06 - INFO - __main__ -   evaluation result saved to .background_runs/flowtrace_pmt_v1_real_smoke_wsl_bounded8_hardlimit/checkpoint-1-8/pred.BDDX.testing_32frames.beam1.max15.eval.json
06/18/2026 23:38:13 - INFO - __main__ -   signal evaluation skipped: no valid control rows; wrote .background_runs/flowtrace_pmt_v1_real_smoke_wsl_bounded8_hardlimit/checkpoint-1-8/pred.BDDX.testing_32frames.beam1.max15.signal_unavailable.json
06/18/2026 23:38:43 - INFO - __main__ -   Save checkpoint to .background_runs/flowtrace_pmt_v1_real_smoke_wsl_bounded8_hardlimit/checkpoint_latest
06/18/2026 23:38:43 - INFO - __main__ -   FlowTrace hard smoke train-step limit reached: 8/8
06/18/2026 23:38:43 - INFO - __main__ -   Total training time: 0:04:19.789397 (32.4737 s / iter)
```

## Full Smoke Log

```text
/mnt/e/sbw/FATE_Drive/fate_x_flowtrace_pmt_v1_worktree
[2026-06-18 23:33:25,228] [INFO] [real_accelerator.py:191:get_accelerator] Setting ds_accelerator to cuda (auto detect)
06/18/2026 23:33:53 - WARNING - azureml.core -   Failure while loading azureml_run_type_providers. Failed to load entrypoint azureml.scriptrun = azureml.core.script_run:ScriptRun._from_run_dto with exception (packaging 26.2 (/opt/conda/envs/adapt/lib/python3.8/site-packages), Requirement.parse('packaging<22.0,>=20.0')).
06/18/2026 23:33:53 - INFO - azureml.core.run -   Could not load the run context. Logging offline
no distributed training ...
06/18/2026 23:33:53 - INFO - __main__ -   Setup CUDA, GPU & distributed training
06/18/2026 23:33:53 - INFO - __main__ -   Fairscale is not enabled. We will disable the relevant args --fairscale_fp16.
06/18/2026 23:33:53 - INFO - __main__ -   Disable restorer for deepspeed or fairscale
06/18/2026 23:33:53 - INFO - __main__ -   Disable --mask_tag_prob
06/18/2026 23:33:53 - INFO - __main__ -   Check arguments
06/18/2026 23:33:53 - INFO - __main__ -   creating output_dir at: .background_runs/flowtrace_pmt_v1_real_smoke_wsl_bounded8_hardlimit
06/18/2026 23:33:53 - INFO - __main__ -   device: cuda, n_gpu: 1, rank: 0, 16-bits training: deepspeed, fp16, -1
06/18/2026 23:33:53 - INFO - __main__ -   Pytorch version is: 1.13.1+cu117
06/18/2026 23:33:53 - INFO - __main__ -   Cuda version is: 11.7
06/18/2026 23:33:53 - INFO - __main__ -   cuDNN version is : 8500
06/18/2026 23:33:53 - INFO - __main__ -   video swin (config path): src/modeling/video_swin/swin_base_patch244_window877_kinetics600_22k.py
/opt/conda/envs/adapt/lib/python3.8/site-packages/torch/functional.py:504: UserWarning: torch.meshgrid: in an upcoming release, it will be required to pass the indexing argument. (Triggered internally at ../aten/src/ATen/native/TensorShape.cpp:3190.)
  return _VF.meshgrid(tensors, **kwargs)  # type: ignore[attr-defined]
06/18/2026 23:33:57 - INFO - src.layers.bert.modeling_utils -   loading configuration file models/captioning/bert-base-uncased/config.json
06/18/2026 23:33:57 - INFO - src.layers.bert.modeling_utils -   Model config {
  "architectures": [
    "BertForMaskedLM"
  ],
  "attention_probs_dropout_prob": 0.1,
  "finetuning_task": "image_captioning",
  "gradient_checkpointing": false,
  "hidden_act": "gelu",
  "hidden_dropout_prob": 0.1,
  "hidden_size": 768,
  "initializer_range": 0.02,
  "intermediate_size": 3072,
  "layer_norm_eps": 1e-12,
  "max_position_embeddings": 512,
  "model_type": "bert",
  "num_attention_heads": 12,
  "num_hidden_layers": 12,
  "num_labels": 2,
  "output_attentions": false,
  "output_hidden_states": false,
  "pad_token_id": 0,
  "position_embedding_type": "absolute",
  "torchscript": false,
  "transformers_version": "4.6.0.dev0",
  "type_vocab_size": 2,
  "use_cache": true,
  "vocab_size": 30522
}

06/18/2026 23:33:57 - INFO - src.layers.bert.tokenization_utils -   Model name 'models/captioning/bert-base-uncased' not found in model shortcut name list (bert-base-uncased, bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, bert-base-multilingual-cased, bert-base-chinese, bert-base-german-cased, bert-large-uncased-whole-word-masking, bert-large-cased-whole-word-masking, bert-large-uncased-whole-word-masking-finetuned-squad, bert-large-cased-whole-word-masking-finetuned-squad, bert-base-cased-finetuned-mrpc). Assuming 'models/captioning/bert-base-uncased' is a path or url to a directory containing tokenizer files.
06/18/2026 23:33:57 - INFO - src.layers.bert.tokenization_utils -   Didn't find file models/captioning/bert-base-uncased/added_tokens.json. We won't load it.
06/18/2026 23:33:57 - INFO - src.layers.bert.tokenization_utils -   Didn't find file models/captioning/bert-base-uncased/special_tokens_map.json. We won't load it.
06/18/2026 23:33:57 - INFO - src.layers.bert.tokenization_utils -   loading file None
06/18/2026 23:33:57 - INFO - src.layers.bert.tokenization_utils -   loading file None
06/18/2026 23:33:57 - INFO - src.layers.bert.tokenization_utils -   loading file models/captioning/bert-base-uncased/vocab.txt
06/18/2026 23:33:58 - INFO - __main__ -   Update config parameter img_feature_dim: -1 -> 512
06/18/2026 23:33:58 - INFO - src.layers.bert.modeling_bert -   BertImgModel Image Dimension: 512
06/18/2026 23:34:00 - INFO - __main__ -   Init model from scratch.
06/18/2026 23:34:00 - INFO - __main__ -   Model total parameters: 136106810
06/18/2026 23:34:00 - INFO - src.layers.bert.modeling_utils -   loading configuration file models/captioning/bert-base-uncased/config.json
06/18/2026 23:34:00 - INFO - src.layers.bert.modeling_utils -   Model config {
  "architectures": [
    "BertForMaskedLM"
  ],
  "attention_probs_dropout_prob": 0.1,
  "finetuning_task": "image_captioning",
  "gradient_checkpointing": false,
  "hidden_act": "gelu",
  "hidden_dropout_prob": 0.1,
  "hidden_size": 768,
  "initializer_range": 0.02,
  "intermediate_size": 3072,
  "layer_norm_eps": 1e-12,
  "max_position_embeddings": 512,
  "model_type": "bert",
  "num_attention_heads": 12,
  "num_hidden_layers": 12,
  "num_labels": 2,
  "output_attentions": false,
  "output_hidden_states": false,
  "pad_token_id": 0,
  "position_embedding_type": "absolute",
  "torchscript": false,
  "transformers_version": "4.6.0.dev0",
  "type_vocab_size": 2,
  "use_cache": true,
  "vocab_size": 30522
}

06/18/2026 23:34:01 - INFO - __main__ -   Loading state dict from checkpoint checkpoints/basemodel/checkpoints/model.bin
06/18/2026 23:34:13 - INFO - __main__ -   yaml_file:datasets_part/BDDX/training_32frames.yaml
06/18/2026 23:34:13 - INFO - src.utils.tsv_file -   loading lineidx: datasets_part/BDDX/training.caption.lineidx
06/18/2026 23:34:19 - INFO - __main__ -   Use_asr: False
06/18/2026 23:34:19 - INFO - __main__ -   isTrainData: True
[PyAV video parameters] Num of Frame: 32, FPS: 3, Sampling: uniform
06/18/2026 23:34:19 - INFO - src.utils.tsv_file -   loading lineidx: datasets_part/BDDX/frame_tsv_part/training_32frames_img_size256.img.lineidx
06/18/2026 23:34:19 - INFO - __main__ -   Train with 1 images per GPU.
06/18/2026 23:34:19 - INFO - __main__ -   Total batch size 1
06/18/2026 23:34:19 - INFO - __main__ -   Total training steps 16392
06/18/2026 23:34:19 - INFO - __main__ -   yaml_file:datasets/BDDX/testing_32frames.yaml
06/18/2026 23:34:19 - INFO - src.utils.tsv_file -   loading lineidx: datasets/BDDX/../../datasets/BDDX/testing.caption.lineidx
06/18/2026 23:34:20 - INFO - __main__ -   Use_asr: False
06/18/2026 23:34:20 - INFO - __main__ -   isTrainData: False
[PyAV video parameters] Num of Frame: 32, FPS: 3, Sampling: uniform
06/18/2026 23:34:20 - INFO - src.utils.tsv_file -   loading lineidx: datasets/BDDX/../../datasets/BDDX/frame_tsv/testing_32frames_img_size256.img.lineidx
06/18/2026 23:34:20 - INFO - __main__ -   FlowTrace hard smoke train-step limit enabled: 8; capped max_global_step=8, max_iter=8
06/18/2026 23:34:20 - INFO - __main__ -   {'flops_profiler': {'detailed': True,
                    'enabled': False,
                    'module_depth': -1,
                    'profile_step': 1,
                    'top_modules': 3},
 'fp16': {'enabled': True},
 'gradient_clipping': 1.0,
 'logging': {'steps_per_print': 200},
 'train_batch_size': 1}
[2026-06-18 23:34:20,749] [INFO] [logging.py:96:log_dist] [Rank -1] DeepSpeed info: version=0.14.0, git-hash=unknown, git-branch=unknown
[2026-06-18 23:34:20,749] [INFO] [comm.py:637:init_distributed] cdb=None
[2026-06-18 23:34:20,749] [INFO] [comm.py:652:init_distributed] Not using the DeepSpeed or dist launchers, attempting to detect MPI environment...
[2026-06-18 23:34:22,316] [INFO] [comm.py:702:mpi_discovery] Discovered MPI settings of world_rank=0, local_rank=0, world_size=1, master_addr=172.19.37.47, master_port=29500
[2026-06-18 23:34:22,316] [INFO] [comm.py:668:init_distributed] Initializing TorchBackend in DeepSpeed with backend nccl
06/18/2026 23:34:22 - INFO - torch.distributed.distributed_c10d -   Added key: store_based_barrier_key:1 to store for rank: 0
06/18/2026 23:34:22 - INFO - torch.distributed.distributed_c10d -   Rank 0: Completed store-based barrier for key:store_based_barrier_key:1 with 1 nodes.
06/18/2026 23:34:22 - INFO - torch.distributed.distributed_c10d -   Added key: store_based_barrier_key:2 to store for rank: 0
06/18/2026 23:34:22 - INFO - torch.distributed.distributed_c10d -   Rank 0: Completed store-based barrier for key:store_based_barrier_key:2 with 1 nodes.
[2026-06-18 23:34:22,752] [INFO] [logging.py:96:log_dist] [Rank 0] DeepSpeed Flops Profiler Enabled: False
[2026-06-18 23:34:22,756] [INFO] [logging.py:96:log_dist] [Rank 0] Using client Optimizer as basic optimizer
[2026-06-18 23:34:22,756] [INFO] [logging.py:96:log_dist] [Rank 0] Removing param_group that has no 'params' in the basic Optimizer
[2026-06-18 23:34:22,820] [INFO] [logging.py:96:log_dist] [Rank 0] DeepSpeed Basic Optimizer = AdamW
[2026-06-18 23:34:22,821] [INFO] [logging.py:96:log_dist] [Rank 0] Creating fp16 unfused optimizer with dynamic loss scale
[2026-06-18 23:34:22,821] [INFO] [unfused_optimizer.py:45:__init__] Fused Lamb Legacy : False
[2026-06-18 23:34:23,351] [INFO] [logging.py:96:log_dist] [Rank 0] DeepSpeed Final Optimizer = AdamW
[2026-06-18 23:34:23,352] [INFO] [logging.py:96:log_dist] [Rank 0] DeepSpeed using client LR scheduler
[2026-06-18 23:34:23,352] [INFO] [logging.py:96:log_dist] [Rank 0] DeepSpeed LR Scheduler = <src.solver.lr_scheduler.WarmupLinearLR object at 0x7764f911d310>
[2026-06-18 23:34:23,352] [INFO] [logging.py:96:log_dist] [Rank 0] step=0, skipped=0, lr=[1e-05, 0.0002, 1e-05, 0.0002], mom=[(0.9, 0.999), (0.9, 0.999), (0.9, 0.999), (0.9, 0.999)]
[2026-06-18 23:34:23,354] [INFO] [config.py:996:print] DeepSpeedEngine configuration:
[2026-06-18 23:34:23,354] [INFO] [config.py:1000:print]   activation_checkpointing_config  {
    "partition_activations": false,
    "contiguous_memory_optimization": false,
    "cpu_checkpointing": false,
    "number_checkpoints": null,
    "synchronize_checkpoint_boundary": false,
    "profile": false
}
[2026-06-18 23:34:23,355] [INFO] [config.py:1000:print]   aio_config ................... {'block_size': 1048576, 'queue_depth': 8, 'thread_count': 1, 'single_submit': False, 'overlap_events': True}
[2026-06-18 23:34:23,355] [INFO] [config.py:1000:print]   amp_enabled .................. False
[2026-06-18 23:34:23,355] [INFO] [config.py:1000:print]   amp_params ................... False
[2026-06-18 23:34:23,355] [INFO] [config.py:1000:print]   autotuning_config ............ {
    "enabled": false,
    "start_step": null,
    "end_step": null,
    "metric_path": null,
    "arg_mappings": null,
    "metric": "throughput",
    "model_info": null,
    "results_dir": "autotuning_results",
    "exps_dir": "autotuning_exps",
    "overwrite": true,
    "fast": true,
    "start_profile_step": 3,
    "end_profile_step": 5,
    "tuner_type": "gridsearch",
    "tuner_early_stopping": 5,
    "tuner_num_trials": 50,
    "model_info_path": null,
    "mp_size": 1,
    "max_train_batch_size": null,
    "min_train_batch_size": 1,
    "max_train_micro_batch_size_per_gpu": 1.024000e+03,
    "min_train_micro_batch_size_per_gpu": 1,
    "num_tuning_micro_batch_sizes": 3
}
[2026-06-18 23:34:23,355] [INFO] [config.py:1000:print]   bfloat16_enabled ............. False
[2026-06-18 23:34:23,355] [INFO] [config.py:1000:print]   bfloat16_immediate_grad_update  False
[2026-06-18 23:34:23,355] [INFO] [config.py:1000:print]   checkpoint_parallel_write_pipeline  False
[2026-06-18 23:34:23,355] [INFO] [config.py:1000:print]   checkpoint_tag_validation_enabled  True
[2026-06-18 23:34:23,355] [INFO] [config.py:1000:print]   checkpoint_tag_validation_fail  False
[2026-06-18 23:34:23,355] [INFO] [config.py:1000:print]   comms_config ................. <deepspeed.comm.config.DeepSpeedCommsConfig object at 0x7764f91716a0>
[2026-06-18 23:34:23,355] [INFO] [config.py:1000:print]   communication_data_type ...... None
[2026-06-18 23:34:23,355] [INFO] [config.py:1000:print]   compile_config ............... enabled=False backend='inductor' kwargs={}
[2026-06-18 23:34:23,355] [INFO] [config.py:1000:print]   compression_config ........... {'weight_quantization': {'shared_parameters': {'enabled': False, 'quantizer_kernel': False, 'schedule_offset': 0, 'quantize_groups': 1, 'quantize_verbose': False, 'quantization_type': 'symmetric', 'quantize_weight_in_forward': False, 'rounding': 'nearest', 'fp16_mixed_quantize': False, 'quantize_change_ratio': 0.001}, 'different_groups': {}}, 'activation_quantization': {'shared_parameters': {'enabled': False, 'quantization_type': 'symmetric', 'range_calibration': 'dynamic', 'schedule_offset': 1000}, 'different_groups': {}}, 'sparse_pruning': {'shared_parameters': {'enabled': False, 'method': 'l1', 'schedule_offset': 1000}, 'different_groups': {}}, 'row_pruning': {'shared_parameters': {'enabled': False, 'method': 'l1', 'schedule_offset': 1000}, 'different_groups': {}}, 'head_pruning': {'shared_parameters': {'enabled': False, 'method': 'topk', 'schedule_offset': 1000}, 'different_groups': {}}, 'channel_pruning': {'shared_parameters': {'enabled': False, 'method': 'l1', 'schedule_offset': 1000}, 'different_groups': {}}, 'layer_reduction': {'enabled': False}}
[2026-06-18 23:34:23,355] [INFO] [config.py:1000:print]   curriculum_enabled_legacy .... False
[2026-06-18 23:34:23,355] [INFO] [config.py:1000:print]   curriculum_params_legacy ..... False
[2026-06-18 23:34:23,355] [INFO] [config.py:1000:print]   data_efficiency_config ....... {'enabled': False, 'seed': 1234, 'data_sampling': {'enabled': False, 'num_epochs': 1000, 'num_workers': 0, 'curriculum_learning': {'enabled': False}}, 'data_routing': {'enabled': False, 'random_ltd': {'enabled': False, 'layer_token_lr_schedule': {'enabled': False}}}}
[2026-06-18 23:34:23,355] [INFO] [config.py:1000:print]   data_efficiency_enabled ...... False
[2026-06-18 23:34:23,355] [INFO] [config.py:1000:print]   dataloader_drop_last ......... False
[2026-06-18 23:34:23,355] [INFO] [config.py:1000:print]   disable_allgather ............ False
[2026-06-18 23:34:23,355] [INFO] [config.py:1000:print]   dump_state ................... False
[2026-06-18 23:34:23,355] [INFO] [config.py:1000:print]   dynamic_loss_scale_args ...... None
[2026-06-18 23:34:23,355] [INFO] [config.py:1000:print]   eigenvalue_enabled ........... False
[2026-06-18 23:34:23,355] [INFO] [config.py:1000:print]   eigenvalue_gas_boundary_resolution  1
[2026-06-18 23:34:23,355] [INFO] [config.py:1000:print]   eigenvalue_layer_name ........ bert.encoder.layer
[2026-06-18 23:34:23,355] [INFO] [config.py:1000:print]   eigenvalue_layer_num ......... 0
[2026-06-18 23:34:23,355] [INFO] [config.py:1000:print]   eigenvalue_max_iter .......... 100
[2026-06-18 23:34:23,355] [INFO] [config.py:1000:print]   eigenvalue_stability ......... 1e-06
[2026-06-18 23:34:23,355] [INFO] [config.py:1000:print]   eigenvalue_tol ............... 0.01
[2026-06-18 23:34:23,355] [INFO] [config.py:1000:print]   eigenvalue_verbose ........... False
[2026-06-18 23:34:23,355] [INFO] [config.py:1000:print]   elasticity_enabled ........... False
[2026-06-18 23:34:23,355] [INFO] [config.py:1000:print]   flops_profiler_config ........ {
    "enabled": false,
    "recompute_fwd_factor": 0.0,
    "profile_step": 1,
    "module_depth": -1,
    "top_modules": 3,
    "detailed": true,
    "output_file": null
}
[2026-06-18 23:34:23,355] [INFO] [config.py:1000:print]   fp16_auto_cast ............... False
[2026-06-18 23:34:23,355] [INFO] [config.py:1000:print]   fp16_enabled ................. True
[2026-06-18 23:34:23,355] [INFO] [config.py:1000:print]   fp16_master_weights_and_gradients  False
[2026-06-18 23:34:23,355] [INFO] [config.py:1000:print]   global_rank .................. 0
[2026-06-18 23:34:23,355] [INFO] [config.py:1000:print]   grad_accum_dtype ............. None
[2026-06-18 23:34:23,356] [INFO] [config.py:1000:print]   gradient_accumulation_steps .. 1
[2026-06-18 23:34:23,356] [INFO] [config.py:1000:print]   gradient_clipping ............ 1.0
[2026-06-18 23:34:23,356] [INFO] [config.py:1000:print]   gradient_predivide_factor .... 1.0
[2026-06-18 23:34:23,356] [INFO] [config.py:1000:print]   graph_harvesting ............. False
[2026-06-18 23:34:23,356] [INFO] [config.py:1000:print]   hybrid_engine ................ enabled=False max_out_tokens=512 inference_tp_size=1 release_inference_cache=False pin_parameters=True tp_gather_partition_size=8
[2026-06-18 23:34:23,356] [INFO] [config.py:1000:print]   initial_dynamic_scale ........ 65536
[2026-06-18 23:34:23,356] [INFO] [config.py:1000:print]   load_universal_checkpoint .... False
[2026-06-18 23:34:23,356] [INFO] [config.py:1000:print]   loss_scale ................... 0
[2026-06-18 23:34:23,356] [INFO] [config.py:1000:print]   memory_breakdown ............. False
[2026-06-18 23:34:23,356] [INFO] [config.py:1000:print]   mics_hierarchial_params_gather  False
[2026-06-18 23:34:23,356] [INFO] [config.py:1000:print]   mics_shard_size .............. -1
[2026-06-18 23:34:23,356] [INFO] [config.py:1000:print]   monitor_config ............... tensorboard=TensorBoardConfig(enabled=False, output_path='', job_name='DeepSpeedJobName') wandb=WandbConfig(enabled=False, group=None, team=None, project='deepspeed') csv_monitor=CSVConfig(enabled=False, output_path='', job_name='DeepSpeedJobName') enabled=False
[2026-06-18 23:34:23,356] [INFO] [config.py:1000:print]   nebula_config ................ {
    "enabled": false,
    "persistent_storage_path": null,
    "persistent_time_interval": 100,
    "num_of_version_in_retention": 2,
    "enable_nebula_load": true,
    "load_path": null
}
[2026-06-18 23:34:23,356] [INFO] [config.py:1000:print]   optimizer_legacy_fusion ...... False
[2026-06-18 23:34:23,356] [INFO] [config.py:1000:print]   optimizer_name ............... None
[2026-06-18 23:34:23,356] [INFO] [config.py:1000:print]   optimizer_params ............. None
[2026-06-18 23:34:23,356] [INFO] [config.py:1000:print]   pipeline ..................... {'stages': 'auto', 'partition': 'best', 'seed_layers': False, 'activation_checkpoint_interval': 0, 'pipe_partitioned': True, 'grad_partitioned': True}
[2026-06-18 23:34:23,356] [INFO] [config.py:1000:print]   pld_enabled .................. False
[2026-06-18 23:34:23,356] [INFO] [config.py:1000:print]   pld_params ................... False
[2026-06-18 23:34:23,356] [INFO] [config.py:1000:print]   prescale_gradients ........... False
[2026-06-18 23:34:23,356] [INFO] [config.py:1000:print]   scheduler_name ............... None
[2026-06-18 23:34:23,356] [INFO] [config.py:1000:print]   scheduler_params ............. None
[2026-06-18 23:34:23,356] [INFO] [config.py:1000:print]   seq_parallel_communication_data_type  torch.float32
[2026-06-18 23:34:23,356] [INFO] [config.py:1000:print]   sparse_attention ............. None
[2026-06-18 23:34:23,356] [INFO] [config.py:1000:print]   sparse_gradients_enabled ..... False
[2026-06-18 23:34:23,356] [INFO] [config.py:1000:print]   steps_per_print .............. 10
[2026-06-18 23:34:23,356] [INFO] [config.py:1000:print]   train_batch_size ............. 1
[2026-06-18 23:34:23,356] [INFO] [config.py:1000:print]   train_micro_batch_size_per_gpu  1
[2026-06-18 23:34:23,356] [INFO] [config.py:1000:print]   use_data_before_expert_parallel_  False
[2026-06-18 23:34:23,356] [INFO] [config.py:1000:print]   use_node_local_storage ....... False
[2026-06-18 23:34:23,356] [INFO] [config.py:1000:print]   wall_clock_breakdown ......... False
[2026-06-18 23:34:23,356] [INFO] [config.py:1000:print]   weight_quantization_config ... None
[2026-06-18 23:34:23,356] [INFO] [config.py:1000:print]   world_size ................... 1
[2026-06-18 23:34:23,356] [INFO] [config.py:1000:print]   zero_allow_untested_optimizer  False
[2026-06-18 23:34:23,356] [INFO] [config.py:1000:print]   zero_config .................. stage=0 contiguous_gradients=True reduce_scatter=True reduce_bucket_size=500,000,000 use_multi_rank_bucket_allreduce=True allgather_partitions=True allgather_bucket_size=500,000,000 overlap_comm=False load_from_fp32_weights=True elastic_checkpoint=False offload_param=None offload_optimizer=None sub_group_size=1,000,000,000 cpu_offload_param=None cpu_offload_use_pin_memory=None cpu_offload=None prefetch_bucket_size=50,000,000 param_persistence_threshold=100,000 model_persistence_threshold=sys.maxsize max_live_parameters=1,000,000,000 max_reuse_distance=1,000,000,000 gather_16bit_weights_on_model_save=False stage3_gather_fp16_weights_on_model_save=False ignore_unused_parameters=True legacy_stage1=False round_robin_gradients=False zero_hpz_partition_size=1 zero_quantized_weights=False zero_quantized_nontrainable_weights=False zero_quantized_gradients=False mics_shard_size=-1 mics_hierarchical_params_gather=False memory_efficient_linear=True pipeline_loading_checkpoint=False override_module_apply=True
[2026-06-18 23:34:23,356] [INFO] [config.py:1000:print]   zero_enabled ................. False
[2026-06-18 23:34:23,356] [INFO] [config.py:1000:print]   zero_force_ds_cpu_optimizer .. True
[2026-06-18 23:34:23,356] [INFO] [config.py:1000:print]   zero_optimization_stage ...... 0
[2026-06-18 23:34:23,356] [INFO] [config.py:986:print_user_config]   json = {
    "train_batch_size": 1,
    "fp16": {
        "enabled": true
    },
    "gradient_clipping": 1.0,
    "flops_profiler": {
        "enabled": false,
        "profile_step": 1,
        "module_depth": -1,
        "top_modules": 3,
        "detailed": true
    },
    "logging": {
        "steps_per_print": 200
    }
}
06/18/2026 23:34:23 - INFO - __main__ -   Training/evaluation parameters: {'data_dir': '.', 'output_dir': '.background_runs/flowtrace_pmt_v1_real_smoke_wsl_bounded8_hardlimit', 'train_yaml': 'datasets_part/BDDX/training_32frames.yaml', 'model_name_or_path': 'checkpoints/basemodel/checkpoints/', 'config_name': '', 'tokenizer_name': '', 'num_hidden_layers': -1, 'hidden_size': -1, 'num_attention_heads': -1, 'intermediate_size': -1, 'img_feature_dim': 512, 'load_partial_weights': False, 'freeze_embedding': False, 'drop_out': 0.1, 'max_seq_length': 30, 'max_seq_a_length': 15, 'max_img_seq_length': 784, 'do_lower_case': True, 'add_od_labels': False, 'od_label_conf': 0.0, 'use_asr': False, 'use_sep_cap': True, 'use_swap_cap': False, 'use_car_sensor': False, 'multitask': True, 'only_signal': False, 'signal_types': ['course', 'speed'], 'unique_labels_on': False, 'no_sort_by_conf': False, 'mask_prob': 0.5, 'max_masked_tokens': 45, 'attn_mask_type': 'learn_vid_att', 'text_mask_type': 'random', 'tag_to_mask': ['noun', 'verb'], 'mask_tag_prob': -1, 'tagger_model_path': 'models/flair/en-pos-ontonotes-fast-v0.5.pt', 'random_mask_prob': 0, 'on_memory': False, 'effective_batch_size': 1, 'per_gpu_train_batch_size': 1, 'num_workers': 4, 'limited_samples': 8, 'limited_eval_samples': 8, 'learning_rate': 0.0002, 'weight_decay': 0.05, 'adam_epsilon': 1e-08, 'max_grad_norm': 1.0, 'warmup_ratio': 0.1, 'scheduler': 'warmup_linear', 'gradient_accumulation_steps': 1, 'num_train_epochs': 1, 'logging_steps': 20, 'save_steps': 8, 'restore_ratio': -1, 'device': device(type='cuda'), 'seed': 88, 'local_rank': 0, 'mixed_precision_method': 'deepspeed', 'zero_opt_stage': -1, 'amp_opt_level': 0, 'deepspeed_fp16': True, 'deepspeed_bf16': False, 'fairscale_fp16': False, 'pretrained_checkpoint': 'checkpoints/basemodel/checkpoints/', 'debug': False, 'debug_speed': False, 'config': 'src/configs/VidSwinBert/BDDX_multi_default.json', 'eval_model_dir': 'checkpoints/basemodel/checkpoints/', 'val_yaml': 'datasets/BDDX/testing_32frames.yaml', 'test_yaml': 'coco_caption/test.yaml', 'do_train': True, 'do_test': False, 'do_eval': False, 'do_signal_eval': False, 'evaluate_during_training': True, 'per_gpu_eval_batch_size': 1, 'mask_img_feat': False, 'max_masked_img_tokens': 10, 'tie_weights': False, 'label_smoothing': 0, 'drop_worst_ratio': 0, 'drop_worst_after': 0, 'max_gen_length': 15, 'output_hidden_states': False, 'num_return_sequences': 1, 'num_beams': 1, 'num_keep_best': 1, 'temperature': 1, 'top_k': 0, 'top_p': 1, 'repetition_penalty': 1, 'length_penalty': 1, 'use_cbs': False, 'min_constraints_to_satisfy': 2, 'use_hypo': False, 'decoding_constraint': False, 'remove_bad_endings': False, 'scst': False, 'sc_train_sample_n': 5, 'sc_baseline_type': 'greedy', 'cider_cached_tokens': 'coco_caption/gt/coco-train-words.p', 'max_num_frames': 32, 'img_res': 224, 'patch_size': 32, 'grid_feat': True, 'kinetics': '600', 'pretrained_2d': False, 'vidswin_size': 'base', 'freeze_backbone': False, 'use_checkpoint': True, 'backbone_coef_lr': 0.05, 'reload_pretrained_swin': False, 'learn_mask_enabled': True, 'loss_sparse_w': 0.1, 'loss_sensor_w': 0.05, 'sparse_mask_soft2hard': False, 'transfer_method': -1, 'att_mask_expansion': -1, 'resume_checkpoint': 'None', 'resume_repro_checkpoint_dir': '', 'test_video_fname': 'None', 'fate_x_enabled': True, 'video_token_reducer': 'none', 'fate_x_keep_ratio': 0.5, 'fate_x_num_summary_tokens': 1, 'fate_x_min_tokens': 128, 'fate_x_min_tokens_per_frame': 1, 'fate_x_temporal_tokens': 0, 'fate_x_spatial_tokens_per_frame': 0, 'fate_x_summary_mode': 'cluster', 'fate_x_text_reduce_only': True, 'fate_x_reduce_control': False, 'fate_x_control_reducer': 'none', 'temporal_evidence_memory': 'none', 'reference_effective_batch': 64, 'base_learning_rate_at_reference_batch': 0.0002, 'auto_scale_lr': False, 'num_gpus_for_lr': 1, 'phrase_faithfulness_enabled': False, 'visualize_phrase_attention': False, 'flowtrace_enabled': True, 'flowtrace_state_dim': 256, 'flowtrace_pmt_rank': 32, 'flowtrace_smoke_evidence': '.background_runs/flowtrace_pmt_v1_real_smoke_wsl_bounded8_hardlimit/flowtrace_real_smoke_summary.json', 'flowtrace_max_train_steps': 8, 'use_clip_model': True, 'num_gpus': 1, 'distributed': False, 'lr_actual': 0.0002, 'backbone_lr': 1e-05, 'loss_divided_by_accumulation': False, 'accumulation_handled_by': 'deepspeed', 'max_iter': 8, 'max_global_step': 8, 'global_iters_per_epoch': 8, 'resume_repro_meta': None}
06/18/2026 23:34:23 - INFO - __main__ -   saving args to .background_runs/flowtrace_pmt_v1_real_smoke_wsl_bounded8_hardlimit/log/args.json
06/18/2026 23:34:23 - INFO - src.utils.tsv_file -   re-open datasets_part/BDDX/training.caption.tsv because the process id changed
06/18/2026 23:34:23 - INFO - src.utils.tsv_file -   re-open datasets_part/BDDX/training.caption.tsv because the process id changed
06/18/2026 23:34:23 - INFO - src.utils.tsv_file -   re-open datasets_part/BDDX/training.caption.tsv because the process id changed
06/18/2026 23:34:23 - INFO - src.utils.tsv_file -   re-open datasets_part/BDDX/training.caption.tsv because the process id changed
06/18/2026 23:34:23 - INFO - src.utils.tsv_file -   re-open datasets_part/BDDX/frame_tsv_part/training_32frames_img_size256.img.tsv because the process id changed
06/18/2026 23:34:23 - INFO - src.utils.tsv_file -   re-open datasets_part/BDDX/frame_tsv_part/training_32frames_img_size256.img.tsv because the process id changed
06/18/2026 23:34:23 - INFO - src.utils.tsv_file -   re-open datasets_part/BDDX/frame_tsv_part/training_32frames_img_size256.img.tsv because the process id changed
06/18/2026 23:34:23 - INFO - src.utils.tsv_file -   re-open datasets_part/BDDX/frame_tsv_part/training_32frames_img_size256.img.tsv because the process id changed
06/18/2026 23:34:24 - INFO - __main__ -   input_ids = torch.Size([1, 30])
06/18/2026 23:34:24 - INFO - __main__ -   attention_mask = torch.Size([1, 814, 814])
06/18/2026 23:34:24 - INFO - __main__ -   token_type_ids = torch.Size([1, 30])
06/18/2026 23:34:24 - INFO - __main__ -   img_feats = torch.Size([1, 32, 3, 224, 224])
06/18/2026 23:34:24 - INFO - __main__ -   masked_pos = torch.Size([1, 30])
06/18/2026 23:34:24 - INFO - __main__ -   masked_ids = torch.Size([1, 45])
06/18/2026 23:34:24 - INFO - __main__ -   car_info = torch.Size([1, 2, 32])
/opt/conda/envs/adapt/lib/python3.8/site-packages/torch/utils/checkpoint.py:31: UserWarning: None of the inputs have requires_grad=True. Gradients will be None
  warnings.warn("None of the inputs have requires_grad=True. Gradients will be None")
06/18/2026 23:34:40 - WARNING - tensorboardX.x2num -   NaN or Inf found in input tensor.
06/18/2026 23:34:41 - INFO - __main__ -   ModelSaver save trial NO. 0
06/18/2026 23:35:08 - INFO - __main__ -   Save checkpoint to .background_runs/flowtrace_pmt_v1_real_smoke_wsl_bounded8_hardlimit/checkpoint_latest
06/18/2026 23:35:08 - INFO - __main__ -   Perform evaluation at iteration 1, global_step 1
06/18/2026 23:35:09 - INFO - src.utils.tsv_file -   re-open datasets/BDDX/../../datasets/BDDX/testing.caption.tsv because the process id changed
06/18/2026 23:35:09 - INFO - src.utils.tsv_file -   re-open datasets/BDDX/../../datasets/BDDX/testing.caption.tsv because the process id changed
06/18/2026 23:35:09 - INFO - src.utils.tsv_file -   re-open datasets/BDDX/../../datasets/BDDX/testing.caption.tsv because the process id changed
06/18/2026 23:35:09 - INFO - src.utils.tsv_file -   re-open datasets/BDDX/../../datasets/BDDX/frame_tsv/testing_32frames_img_size256.img.tsv because the process id changed

0it [00:00, ?it/s]06/18/2026 23:35:09 - INFO - src.utils.tsv_file -   re-open datasets/BDDX/../../datasets/BDDX/frame_tsv/testing_32frames_img_size256.img.tsv because the process id changed
06/18/2026 23:35:09 - INFO - src.utils.tsv_file -   re-open datasets/BDDX/../../datasets/BDDX/frame_tsv/testing_32frames_img_size256.img.tsv because the process id changed
06/18/2026 23:35:09 - INFO - src.utils.tsv_file -   re-open datasets/BDDX/../../datasets/BDDX/testing.caption.tsv because the process id changed
06/18/2026 23:35:09 - INFO - src.utils.tsv_file -   re-open datasets/BDDX/../../datasets/BDDX/frame_tsv/testing_32frames_img_size256.img.tsv because the process id changed

1it [00:01,  1.79s/it]
2it [00:02,  1.23s/it]
3it [00:03,  1.10s/it]
4it [00:04,  1.00s/it]
5it [00:05,  1.04it/s]
6it [00:06,  1.09it/s]
7it [00:06,  1.12it/s]
8it [00:07,  1.14it/s]
8it [00:07,  1.01it/s]
06/18/2026 23:35:17 - INFO - __main__ -   Inference model computing time: 0.9041170477867126 seconds per batch
loading annotations into memory...
0:00:00.020920
creating index...
index created!
Loading and preparing results...
DONE (t=0.00s)
creating index...
index created!
tokenization...
PTBTokenizer tokenized 52 tokens at 292.06 tokens per second.
PTBTokenizer tokenized 95 tokens at 1007.64 tokens per second.
setting up scorers...
computing Bleu score...
{'testlen': 88, 'reflen': 45, 'guess': [88, 80, 72, 64], 'correct': [0, 0, 0, 0]}
ratio: 1.955555555512099
Bleu_1: 0.000
Bleu_2: 0.000
Bleu_3: 0.000
Bleu_4: 0.000
computing METEOR score...
METEOR: 0.000
computing Rouge score...
ROUGE_L: 0.000
computing CIDEr score...
CIDEr: 0.000
computing SPICE score...
WARNING: An illegal reflective access operation has occurred
WARNING: Illegal reflective access by org.nustaq.serialization.FSTClazzInfo (file:/mnt/e/sbw/FATE_Drive/fate_x_flowtrace_pmt_v1_worktree/src/evalcap/coco_caption/pycocoevalcap/spice/lib/fst-2.47.jar) to field java.lang.String.value
WARNING: Please consider reporting this to the maintainers of org.nustaq.serialization.FSTClazzInfo
WARNING: Use --illegal-access=warn to enable warnings of further illegal reflective access operations
WARNING: All illegal access operations will be denied in a future release
Parsing reference captions
Parsing test captions
Warning: Nashorn engine is planned to be removed from a future JDK release
SPICE evaluation took: 1.404 s
SPICE: 0.000
loading annotations into memory...
0:00:00.019865
creating index...
index created!
Loading and preparing results...
DONE (t=0.00s)
creating index...
index created!
tokenization...
PTBTokenizer tokenized 103 tokens at 797.20 tokens per second.
Jun 18, 2026 11:35:34 PM edu.stanford.nlp.process.PTBLexer next
WARNING: Untokenizable: ಾ (U+CBE, decimal: 3262)
PTBTokenizer tokenized 69 tokens at 518.91 tokens per second.
setting up scorers...
computing Bleu score...
{'testlen': 58, 'reflen': 88, 'guess': [58, 50, 42, 34], 'correct': [0, 0, 0, 0]}
ratio: 0.6590909090834194
Bleu_1: 0.000
Bleu_2: 0.000
Bleu_3: 0.000
Bleu_4: 0.000
computing METEOR score...
METEOR: 0.000
computing Rouge score...
ROUGE_L: 0.000
computing CIDEr score...
CIDEr: 0.000
computing SPICE score...
WARNING: An illegal reflective access operation has occurred
WARNING: Illegal reflective access by org.nustaq.serialization.FSTClazzInfo (file:/mnt/e/sbw/FATE_Drive/fate_x_flowtrace_pmt_v1_worktree/src/evalcap/coco_caption/pycocoevalcap/spice/lib/fst-2.47.jar) to field java.lang.String.value
WARNING: Please consider reporting this to the maintainers of org.nustaq.serialization.FSTClazzInfo
WARNING: Use --illegal-access=warn to enable warnings of further illegal reflective access operations
WARNING: All illegal access operations will be denied in a future release
Parsing reference captions
Parsing test captions
Warning: Nashorn engine is planned to be removed from a future JDK release
SPICE evaluation took: 1.026 s
SPICE: 0.000
06/18/2026 23:35:48 - INFO - __main__ -   evaluation result: [{'Bleu_1': 1.1363636363507233e-17, 'Bleu_2': 1.1918282365427698e-17, 'Bleu_3': 1.2541946622942019e-17, 'Bleu_4': 1.3250391724213923e-17, 'METEOR': 0.0, 'ROUGE_L': 0.0, 'CIDEr': 0.0, 'SPICE': 0.0}, {'Bleu_1': 1.0278670152800239e-17, 'Bleu_2': 1.1070466554185467e-17, 'Bleu_3': 1.202677428603392e-17, 'Bleu_4': 1.3215493085391265e-17, 'METEOR': 0.0, 'ROUGE_L': 0.0, 'CIDEr': 0.0, 'SPICE': 0.0}]
06/18/2026 23:35:48 - INFO - __main__ -   evaluation result saved to .background_runs/flowtrace_pmt_v1_real_smoke_wsl_bounded8_hardlimit/checkpoint-0-1/pred.BDDX.testing_32frames.beam1.max15.eval.json
06/18/2026 23:35:48 - INFO - src.utils.tsv_file -   re-open datasets/BDDX/../../datasets/BDDX/testing.caption.tsv because the process id changed
06/18/2026 23:35:48 - INFO - src.utils.tsv_file -   re-open datasets/BDDX/../../datasets/BDDX/testing.caption.tsv because the process id changed
06/18/2026 23:35:48 - INFO - src.utils.tsv_file -   re-open datasets/BDDX/../../datasets/BDDX/testing.caption.tsv because the process id changed

0it [00:00, ?it/s]06/18/2026 23:35:48 - INFO - src.utils.tsv_file -   re-open datasets/BDDX/../../datasets/BDDX/frame_tsv/testing_32frames_img_size256.img.tsv because the process id changed
06/18/2026 23:35:48 - INFO - src.utils.tsv_file -   re-open datasets/BDDX/../../datasets/BDDX/frame_tsv/testing_32frames_img_size256.img.tsv because the process id changed
06/18/2026 23:35:48 - INFO - src.utils.tsv_file -   re-open datasets/BDDX/../../datasets/BDDX/frame_tsv/testing_32frames_img_size256.img.tsv because the process id changed
06/18/2026 23:35:48 - INFO - src.utils.tsv_file -   re-open datasets/BDDX/../../datasets/BDDX/testing.caption.tsv because the process id changed
06/18/2026 23:35:48 - INFO - src.utils.tsv_file -   re-open datasets/BDDX/../../datasets/BDDX/frame_tsv/testing_32frames_img_size256.img.tsv because the process id changed

1it [00:01,  1.43s/it]
2it [00:02,  1.09s/it]
3it [00:03,  1.02it/s]
4it [00:03,  1.08it/s]
5it [00:04,  1.13it/s]
6it [00:05,  1.14it/s]
7it [00:06,  1.14it/s]
8it [00:07,  1.16it/s]
8it [00:07,  1.08it/s]
06/18/2026 23:35:55 - INFO - __main__ -   signal evaluation skipped: no valid control rows; wrote .background_runs/flowtrace_pmt_v1_real_smoke_wsl_bounded8_hardlimit/checkpoint-0-1/pred.BDDX.testing_32frames.beam1.max15.signal_unavailable.json
Attempted to log scalar metric CIDEr:
0.0
Attempted to log scalar metric CIDEr:
0.0
06/18/2026 23:35:56 - INFO - __main__ -   ModelSaver save trial NO. 0
06/18/2026 23:36:24 - INFO - __main__ -   Save checkpoint to .background_runs/flowtrace_pmt_v1_real_smoke_wsl_bounded8_hardlimit/checkpoint_latest
06/18/2026 23:36:24 - INFO - __main__ -   ModelSaver save trial NO. 0
06/18/2026 23:36:51 - INFO - __main__ -   Save checkpoint to .background_runs/flowtrace_pmt_v1_real_smoke_wsl_bounded8_hardlimit/checkpoint_best
/opt/conda/envs/adapt/lib/python3.8/site-packages/torch/utils/checkpoint.py:31: UserWarning: None of the inputs have requires_grad=True. Gradients will be None
  warnings.warn("None of the inputs have requires_grad=True. Gradients will be None")
06/18/2026 23:36:52 - WARNING - tensorboardX.x2num -   NaN or Inf found in input tensor.
06/18/2026 23:36:53 - WARNING - tensorboardX.x2num -   NaN or Inf found in input tensor.
06/18/2026 23:36:55 - WARNING - tensorboardX.x2num -   NaN or Inf found in input tensor.
06/18/2026 23:36:56 - WARNING - tensorboardX.x2num -   NaN or Inf found in input tensor.
[2026-06-18 23:36:56,643] [INFO] [unfused_optimizer.py:281:_update_scale] Grad overflow on iteration: 4
[2026-06-18 23:36:56,643] [INFO] [unfused_optimizer.py:282:_update_scale] Reducing dynamic loss scale from 65536.0 to 32768.0
[2026-06-18 23:36:56,643] [INFO] [unfused_optimizer.py:207:step] [deepspeed] fp16 dynamic loss scale overflow! Skipping step. Attempted loss scale: 65536.0, reducing to 32768.0
06/18/2026 23:36:57 - WARNING - tensorboardX.x2num -   NaN or Inf found in input tensor.
06/18/2026 23:36:59 - WARNING - tensorboardX.x2num -   NaN or Inf found in input tensor.
06/18/2026 23:37:00 - WARNING - tensorboardX.x2num -   NaN or Inf found in input tensor.
06/18/2026 23:37:00 - INFO - __main__ -   eta: 0:00:00  iter: 8  global_step: 8  speed: 0.1 images/sec  loss: 10.3598 (10.4238)  loss_sparsity: 0.3712 (0.3712)  acc: 0.0000 (0.0000)  loss_sensor: 0.1289 (0.1280)  batch_time: 1.4393 (1.3855)  data_time: 0.0003 (0.0003)  lr (Visual Encoder): 2.50e-06  lr (LM): 5.00e-05  max mem: 7738
06/18/2026 23:37:01 - INFO - __main__ -   ModelSaver save trial NO. 0
06/18/2026 23:37:29 - INFO - __main__ -   Save checkpoint to .background_runs/flowtrace_pmt_v1_real_smoke_wsl_bounded8_hardlimit/checkpoint_latest
06/18/2026 23:37:29 - INFO - __main__ -   Perform evaluation at iteration 8, global_step 8
06/18/2026 23:37:29 - INFO - src.utils.tsv_file -   re-open datasets/BDDX/../../datasets/BDDX/testing.caption.tsv because the process id changed
06/18/2026 23:37:29 - INFO - src.utils.tsv_file -   re-open datasets/BDDX/../../datasets/BDDX/testing.caption.tsv because the process id changed
06/18/2026 23:37:29 - INFO - src.utils.tsv_file -   re-open datasets/BDDX/../../datasets/BDDX/testing.caption.tsv because the process id changed

0it [00:00, ?it/s]06/18/2026 23:37:29 - INFO - src.utils.tsv_file -   re-open datasets/BDDX/../../datasets/BDDX/frame_tsv/testing_32frames_img_size256.img.tsv because the process id changed
06/18/2026 23:37:29 - INFO - src.utils.tsv_file -   re-open datasets/BDDX/../../datasets/BDDX/frame_tsv/testing_32frames_img_size256.img.tsv because the process id changed
06/18/2026 23:37:29 - INFO - src.utils.tsv_file -   re-open datasets/BDDX/../../datasets/BDDX/frame_tsv/testing_32frames_img_size256.img.tsv because the process id changed
06/18/2026 23:37:29 - INFO - src.utils.tsv_file -   re-open datasets/BDDX/../../datasets/BDDX/testing.caption.tsv because the process id changed
06/18/2026 23:37:29 - INFO - src.utils.tsv_file -   re-open datasets/BDDX/../../datasets/BDDX/frame_tsv/testing_32frames_img_size256.img.tsv because the process id changed

1it [00:01,  1.43s/it]
2it [00:02,  1.08s/it]
3it [00:03,  1.04it/s]
4it [00:03,  1.08it/s]
5it [00:04,  1.10it/s]
6it [00:05,  1.11it/s]
7it [00:06,  1.14it/s]
8it [00:07,  1.15it/s]
8it [00:07,  1.07it/s]
06/18/2026 23:37:37 - INFO - __main__ -   Inference model computing time: 0.8583926856517792 seconds per batch
loading annotations into memory...
0:00:00.014104
creating index...
index created!
Loading and preparing results...
DONE (t=0.00s)
creating index...
index created!
tokenization...
PTBTokenizer tokenized 52 tokens at 558.62 tokens per second.
PTBTokenizer tokenized 95 tokens at 1012.13 tokens per second.
setting up scorers...
computing Bleu score...
{'testlen': 88, 'reflen': 45, 'guess': [88, 80, 72, 64], 'correct': [0, 0, 0, 0]}
ratio: 1.955555555512099
Bleu_1: 0.000
Bleu_2: 0.000
Bleu_3: 0.000
Bleu_4: 0.000
computing METEOR score...
METEOR: 0.000
computing Rouge score...
ROUGE_L: 0.000
computing CIDEr score...
CIDEr: 0.000
computing SPICE score...
WARNING: An illegal reflective access operation has occurred
WARNING: Illegal reflective access by org.nustaq.serialization.FSTClazzInfo (file:/mnt/e/sbw/FATE_Drive/fate_x_flowtrace_pmt_v1_worktree/src/evalcap/coco_caption/pycocoevalcap/spice/lib/fst-2.47.jar) to field java.lang.String.value
WARNING: Please consider reporting this to the maintainers of org.nustaq.serialization.FSTClazzInfo
WARNING: Use --illegal-access=warn to enable warnings of further illegal reflective access operations
WARNING: All illegal access operations will be denied in a future release
Parsing reference captions
Parsing test captions
Warning: Nashorn engine is planned to be removed from a future JDK release
SPICE evaluation took: 1.039 s
SPICE: 0.000
loading annotations into memory...
0:00:00.020210
creating index...
index created!
Loading and preparing results...
DONE (t=0.00s)
creating index...
index created!
tokenization...
PTBTokenizer tokenized 103 tokens at 1115.27 tokens per second.
Jun 18, 2026 11:37:52 PM edu.stanford.nlp.process.PTBLexer next
WARNING: Untokenizable: ಾ (U+CBE, decimal: 3262)
PTBTokenizer tokenized 69 tokens at 509.47 tokens per second.
setting up scorers...
computing Bleu score...
{'testlen': 58, 'reflen': 88, 'guess': [58, 50, 42, 34], 'correct': [0, 0, 0, 0]}
ratio: 0.6590909090834194
Bleu_1: 0.000
Bleu_2: 0.000
Bleu_3: 0.000
Bleu_4: 0.000
computing METEOR score...
METEOR: 0.000
computing Rouge score...
ROUGE_L: 0.000
computing CIDEr score...
CIDEr: 0.000
computing SPICE score...
WARNING: An illegal reflective access operation has occurred
WARNING: Illegal reflective access by org.nustaq.serialization.FSTClazzInfo (file:/mnt/e/sbw/FATE_Drive/fate_x_flowtrace_pmt_v1_worktree/src/evalcap/coco_caption/pycocoevalcap/spice/lib/fst-2.47.jar) to field java.lang.String.value
WARNING: Please consider reporting this to the maintainers of org.nustaq.serialization.FSTClazzInfo
WARNING: Use --illegal-access=warn to enable warnings of further illegal reflective access operations
WARNING: All illegal access operations will be denied in a future release
Parsing reference captions
Parsing test captions
Warning: Nashorn engine is planned to be removed from a future JDK release
SPICE evaluation took: 1.042 s
SPICE: 0.000
06/18/2026 23:38:06 - INFO - __main__ -   evaluation result: [{'Bleu_1': 1.1363636363507233e-17, 'Bleu_2': 1.1918282365427698e-17, 'Bleu_3': 1.2541946622942019e-17, 'Bleu_4': 1.3250391724213923e-17, 'METEOR': 0.0, 'ROUGE_L': 0.0, 'CIDEr': 0.0, 'SPICE': 0.0}, {'Bleu_1': 1.0278670152800239e-17, 'Bleu_2': 1.1070466554185467e-17, 'Bleu_3': 1.202677428603392e-17, 'Bleu_4': 1.3215493085391265e-17, 'METEOR': 0.0, 'ROUGE_L': 0.0, 'CIDEr': 0.0, 'SPICE': 0.0}]
06/18/2026 23:38:06 - INFO - __main__ -   evaluation result saved to .background_runs/flowtrace_pmt_v1_real_smoke_wsl_bounded8_hardlimit/checkpoint-1-8/pred.BDDX.testing_32frames.beam1.max15.eval.json
06/18/2026 23:38:06 - INFO - src.utils.tsv_file -   re-open datasets/BDDX/../../datasets/BDDX/testing.caption.tsv because the process id changed
06/18/2026 23:38:06 - INFO - src.utils.tsv_file -   re-open datasets/BDDX/../../datasets/BDDX/testing.caption.tsv because the process id changed
06/18/2026 23:38:06 - INFO - src.utils.tsv_file -   re-open datasets/BDDX/../../datasets/BDDX/testing.caption.tsv because the process id changed

0it [00:00, ?it/s]06/18/2026 23:38:06 - INFO - src.utils.tsv_file -   re-open datasets/BDDX/../../datasets/BDDX/frame_tsv/testing_32frames_img_size256.img.tsv because the process id changed
06/18/2026 23:38:06 - INFO - src.utils.tsv_file -   re-open datasets/BDDX/../../datasets/BDDX/frame_tsv/testing_32frames_img_size256.img.tsv because the process id changed
06/18/2026 23:38:06 - INFO - src.utils.tsv_file -   re-open datasets/BDDX/../../datasets/BDDX/frame_tsv/testing_32frames_img_size256.img.tsv because the process id changed
06/18/2026 23:38:06 - INFO - src.utils.tsv_file -   re-open datasets/BDDX/../../datasets/BDDX/testing.caption.tsv because the process id changed
06/18/2026 23:38:06 - INFO - src.utils.tsv_file -   re-open datasets/BDDX/../../datasets/BDDX/frame_tsv/testing_32frames_img_size256.img.tsv because the process id changed

1it [00:01,  1.48s/it]
2it [00:02,  1.11s/it]
3it [00:03,  1.02it/s]
4it [00:04,  1.07it/s]
5it [00:04,  1.11it/s]
6it [00:05,  1.14it/s]
7it [00:06,  1.15it/s]
8it [00:07,  1.17it/s]
8it [00:07,  1.08it/s]
06/18/2026 23:38:13 - INFO - __main__ -   signal evaluation skipped: no valid control rows; wrote .background_runs/flowtrace_pmt_v1_real_smoke_wsl_bounded8_hardlimit/checkpoint-1-8/pred.BDDX.testing_32frames.beam1.max15.signal_unavailable.json
Attempted to log scalar metric CIDEr:
0.0
Attempted to log scalar metric CIDEr:
0.0
06/18/2026 23:38:14 - INFO - __main__ -   ModelSaver save trial NO. 0
06/18/2026 23:38:43 - INFO - __main__ -   Save checkpoint to .background_runs/flowtrace_pmt_v1_real_smoke_wsl_bounded8_hardlimit/checkpoint_latest
06/18/2026 23:38:43 - INFO - __main__ -   FlowTrace hard smoke train-step limit reached: 8/8
06/18/2026 23:38:43 - INFO - __main__ -   Total training time: 0:04:19.789397 (32.4737 s / iter)

```
