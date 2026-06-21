from __future__ import absolute_import, division, print_function

import os
import sys
pythonpath = os.path.abspath(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
print(pythonpath)
sys.path.insert(0, pythonpath)
import os.path as op
import json
import time
import datetime
import shutil
import glob
import torch
import torch.distributed as dist
import gc
import numpy as np
try:
    import deepspeed
except ImportError:  # Windows/single-GPU reproduction can run without DeepSpeed.
    deepspeed = None
try:
    from apex import amp
    from apex.parallel import DistributedDataParallel as ApexDDP
except ImportError:  # Apex is optional; fall back to native PyTorch training.
    amp = None
    ApexDDP = None
from torch.nn.parallel import DistributedDataParallel as TorchDDP
from tqdm import tqdm
from src.configs.config import (basic_check_arguments, shared_configs, restore_training_settings)
from src.datasets.vl_dataloader import make_data_loader
from src.evalcap.utils_caption_evaluate import evaluate_on_coco_caption, two_cap_evaluate_on_coco_caption
from src.utils.logger import LOGGER as logger
from src.utils.logger import (TB_LOGGER, RunningMeter, add_log_to_file)
from src.utils.load_save import TrainingRestorer, TrainingSaver
from src.utils.comm import (is_main_process,
                            get_rank, get_world_size, dist_init)
from src.utils.miscellaneous import (NoOp, mkdir, set_seed, str_to_bool,
                                    delete_tsv_files, concat_tsv_files)
from src.utils.metric_logger import MetricLogger
from src.utils.tsv_file_ops import tsv_writer, double_tsv_writer, reorder_tsv_keys
from src.utils.deepspeed import get_deepspeed_config, fp32_to_fp16, fp32_to_bf16
from src.modeling.video_captioning_e2e_vid_swin_bert import VideoTransformer
from src.modeling.multitask_e2e_vid_swin_bert import MultitaskVideoTransformer
from src.modeling.load_swin import get_swin_model, reload_pretrained_swin
from src.modeling.load_bert import get_bert_model
from src.solver import AdamW, WarmupLinearLR
from fate_x.engine.checkpoint_utils import filter_compatible_state_dict
from fate_x.engine.fate_x_compat import validate_fate_x_mask_compatibility
from fate_x.engine.lr_scaling import apply_lr_scaling_to_args

from azureml.core.run import Run
aml_run = Run.get_context()

def compute_score_with_logits(logits, labels):
    logits = torch.max(logits, -1)[1].data # argmax
    return logits == labels


def save_repro_checkpoint(training_saver, args, tag, step, model, optimizer, metadata):
    """Save stable latest/best checkpoints without changing ADAPT's official checkpoint dirs."""
    if not is_main_process():
        return
    checkpoint_dir = op.join(args.output_dir, tag)
    if op.isdir(checkpoint_dir):
        shutil.rmtree(checkpoint_dir)
    training_saver.save_model(checkpoint_dir, step, model, optimizer)
    with open(op.join(checkpoint_dir, 'repro_checkpoint_meta.json'), 'w') as f:
        json.dump(metadata, f, indent=2)


def load_compatible_state_dict(model, checkpoint, *, context: str):
    """Load checkpoint tensors that match model keys and shapes.

    This preserves official ADAPT initialization while allowing the official
    1-signal basemodel to initialize a 2-signal course+speed training run.
    """
    filtered, skipped = filter_compatible_state_dict(model, checkpoint)
    incompatible = {
        k: v for k, v in skipped.items()
        if v.get("reason") == "shape_mismatch"
    }
    if skipped:
        logger.info(
            f"{context}: loading {len(filtered)} compatible tensors; "
            f"skipping {len(skipped)} incompatible/missing tensors."
        )
        for key, info in list(skipped.items())[:20]:
            logger.info(f"{context}: skipped checkpoint key {key}: {info}")
        if len(skipped) > 20:
            logger.info(f"{context}: skipped checkpoint keys truncated at 20 of {len(skipped)}.")
    if incompatible:
        logger.info(f"{context}: shape mismatches were intentionally left randomly initialized.")
    return model.load_state_dict(filtered, strict=False)


def _unwrap_model(model):
    return getattr(model, "module", model)


def _collect_flowtrace_grad_norms(model):
    """Collect gradient evidence for required FlowTrace trainable paths."""
    root = _unwrap_model(model)
    groups = {
        "transport": ("flowtrace_encoder.tracks.transport", "flowtrace_encoder.transport"),
        "track_queries": ("flowtrace_encoder.tracks.track_queries", "tracks.track_queries"),
        "state_composer": ("flowtrace_encoder.composer",),
        "reason_state_head": ("flowtrace_encoder.reason",),
        "pmt": ("token_pmt_adapter",),
    }
    norms = {name: 0.0 for name in groups}
    for param_name, param in root.named_parameters():
        if param.grad is None:
            continue
        grad_norm = float(param.grad.detach().float().norm().cpu())
        if not np.isfinite(grad_norm):
            grad_norm = 0.0
        for group_name, patterns in groups.items():
            if any(pattern in param_name for pattern in patterns):
                norms[group_name] += grad_norm
    return norms


def _update_flowtrace_smoke_evidence(path, payload):
    """Merge evidence into the real-data smoke summary."""
    if not path:
        return
    path = op.abspath(str(path))
    os.makedirs(op.dirname(path), exist_ok=True)
    data = {}
    if op.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                existing = json.load(f)
            if isinstance(existing, dict):
                data.update(existing)
        except Exception:
            data["previous_summary_unreadable"] = True
    data.update(payload)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def _safe_float(value, default=0.0):
    try:
        if torch.is_tensor(value):
            value = value.detach().float().cpu()
            if value.numel() == 0:
                return default
            value = value.mean().item()
        out = float(value)
        return out if np.isfinite(out) else default
    except Exception:
        return default


def _all_finite(*values):
    for value in values:
        try:
            tensor = value.detach().float() if torch.is_tensor(value) else torch.as_tensor(value).float()
            if not bool(torch.isfinite(tensor).all().item()):
                return False
        except Exception:
            return False
    return True


def _write_flowtrace_smoke_train_evidence(
    args,
    *,
    img_keys,
    inputs,
    logits,
    masked_ids,
    loss,
    model,
    flowtrace_loss_components,
):
    evidence_path = getattr(args, "flowtrace_smoke_evidence", "")
    if not evidence_path or not is_main_process():
        return

    evidence = {
        "real_data_smoke": True,
        "direct_image_training": True,
        "feature_cache_enabled": False,
        "token_cache_enabled": False,
        "forward_backward": True,
        "train_samples": int(getattr(args, "limited_samples", 0) or 0),
        "batch_shapes": {
            key: list(value.shape) for key, value in inputs.items() if torch.is_tensor(value)
        },
    }
    evidence["grad_norms"] = _collect_flowtrace_grad_norms(model)
    evidence["no_nan_inf"] = _all_finite(loss, logits)

    token_logprobs = []
    token_ids = []
    if masked_ids.numel() > 0 and logits.numel() > 0:
        target = masked_ids.to(logits.device).long()
        log_probs = torch.log_softmax(logits.detach().float(), dim=-1)
        gathered = log_probs.gather(-1, target.unsqueeze(-1)).squeeze(-1)
        token_logprobs = [float(x) for x in gathered[:32].detach().cpu().tolist()]
        token_ids = [str(int(x)) for x in target[:32].detach().cpu().tolist()]
    evidence["decoder_logprobs"] = bool(token_logprobs)
    evidence["decoder_token_logprobs_head"] = token_logprobs

    components = flowtrace_loss_components or {}
    state_off_delta = _safe_float(components.get("intervention_state_off_delta"))
    equal_mass_delta = _safe_float(components.get("intervention_equal_mass_delta"))
    intervention_available = _safe_float(components.get("intervention_available")) > 0.5
    evidence["state_off_intervention"] = intervention_available and state_off_delta > 0.0
    evidence["random_equal_mass_intervention"] = intervention_available and equal_mass_delta >= 0.0
    evidence["intervention_state_off_delta"] = state_off_delta
    evidence["intervention_equal_mass_delta"] = equal_mass_delta

    sample_id = str(img_keys[0] if img_keys else "real_smoke_0")
    try:
        from fate_x.engine.write_eval_artifacts import write_fate_x_eval_artifacts

        epoch_dir = write_fate_x_eval_artifacts(
            args.output_dir,
            0,
            [
                {
                    "sample_id": sample_id,
                    "split": "train_smoke",
                    "prediction": "teacher_forced_masked_tokens",
                    "tokens": token_ids,
                    "token_logprobs": token_logprobs,
                    "token_stats": {"masked_token_count": int(masked_ids.numel())},
                    "phrase_scores": [
                        {
                            "phrase": "flowtrace_smoke_real_batch",
                            "deletion_score": state_off_delta,
                            "sufficiency_score": equal_mass_delta,
                        }
                    ],
                }
            ],
            run_manifest={
                "source": "real_bddx_train_batch",
                "feature_cache_enabled": False,
                "token_cache_enabled": False,
            },
        )
        evidence["artifact_schema"] = op.exists(op.join(str(epoch_dir), "predictions.jsonl"))
        evidence["artifact_schema_dir"] = str(epoch_dir)
    except Exception as exc:
        evidence["artifact_schema"] = False
        evidence["artifact_schema_error"] = repr(exc)

    try:
        from fate_x.explain.flowtrace_renderer import FlowTraceRenderer

        bundle = getattr(_unwrap_model(model), "fate_x_last_flowtrace_bundle", None)
        if bundle is None:
            evidence["flowtrace_canvas"] = False
            evidence["flowtrace_canvas_error"] = "missing_flowtrace_bundle"
        else:
            canvas = FlowTraceRenderer().render_canvas(
                bundle,
                op.join(args.output_dir, "flowtrace_smoke_visuals"),
                sample_id.replace("/", "_").replace("\\", "_"),
            )
            evidence["flowtrace_canvas"] = bool(canvas.get("png"))
            evidence["flowtrace_canvas_path"] = canvas.get("png")
    except Exception as exc:
        evidence["flowtrace_canvas"] = False
        evidence["flowtrace_canvas_error"] = repr(exc)

    _update_flowtrace_smoke_evidence(evidence_path, evidence)


def _finalize_flowtrace_smoke_evidence(args):
    evidence_path = getattr(args, "flowtrace_smoke_evidence", "")
    if not evidence_path or not is_main_process():
        return
    eval_files = glob.glob(op.join(args.output_dir, "checkpoint-*", "*.eval.json"))
    eval_files += glob.glob(op.join(args.output_dir, "*.eval.json"))
    checkpoint_latest = op.exists(op.join(args.output_dir, "checkpoint_latest", "model.bin"))
    _update_flowtrace_smoke_evidence(
        evidence_path,
        {
            "eval_completed": bool(eval_files),
            "eval_files": [op.normpath(path) for path in eval_files[:8]],
            "eval_samples": int(getattr(args, "limited_eval_samples", 0) or 0),
            "checkpoint_latest": bool(checkpoint_latest),
        },
    )


def _flowtrace_train_step_limit(args):
    try:
        return max(0, int(getattr(args, "flowtrace_max_train_steps", 0) or 0))
    except (TypeError, ValueError):
        return 0


def _reached_flowtrace_train_step_limit(args, global_step):
    limit = _flowtrace_train_step_limit(args)
    return limit > 0 and int(global_step) >= limit


def _stack_signal_rows(gt_signals, pred_signals):
    if not gt_signals or not pred_signals:
        return None
    return torch.stack(gt_signals, dim=0), torch.stack(pred_signals, dim=0)


def maybe_load_repro_optimizer(args, optimizer):
    checkpoint_dir = getattr(args, 'resume_repro_checkpoint_dir', '')
    if not checkpoint_dir or checkpoint_dir == 'None':
        return None
    optimizer_path = op.join(checkpoint_dir, 'optmizer_state.bin')
    meta_path = op.join(checkpoint_dir, 'repro_checkpoint_meta.json')
    meta = {}
    if op.isfile(meta_path):
        with open(meta_path, 'r') as f:
            meta = json.load(f)
    if op.isfile(optimizer_path):
        try:
            dump = torch.load(optimizer_path, map_location='cpu')
            optimizer.load_state_dict(dump.get('optimizer', dump))
            logger.info(f"Loaded repro optimizer state from {optimizer_path}")
        except Exception as e:
            logger.info(f"Could not load repro optimizer state from {optimizer_path}: {e}")
    return meta


def maybe_fast_forward_repro_scheduler(args, scheduler):
    meta = getattr(args, 'resume_repro_meta', None)
    if not meta:
        return
    global_step = int(meta.get('global_step', 0))
    if global_step <= 0:
        return
    try:
        scheduler.step(global_step)
    except TypeError:
        scheduler.last_epoch = global_step
        if hasattr(scheduler, '_step_count'):
            scheduler._step_count = global_step
    logger.info(f"Fast-forwarded repro scheduler to global_step={global_step}")


def mixed_precision_init(args, model):
    max_iter = args.max_iter
    max_global_step = args.max_global_step
    global_iters_per_epoch = args.global_iters_per_epoch

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']

    decay_param_tp = [(n, p) for n, p in param_optimizer if not any(nd in n for nd in no_decay)]
    no_decay_param_tp = [(n, p) for n, p in param_optimizer if any(nd in n for nd in no_decay)]

    decay_swin_param_tp = [(n, p) for n, p in decay_param_tp if "swin." in n]
    decay_bert_param_tp = [(n, p) for n, p in decay_param_tp if "swin." not in n]

    no_decay_swin_param_tp = [(n, p) for n, p in no_decay_param_tp if "swin." in n]
    no_decay_bert_param_tp = [(n, p) for n, p in no_decay_param_tp if "swin." not in n]

    weight_decay = 0.2
    coef_lr = args.backbone_coef_lr
    optimizer_grouped_parameters = [
        {'params': [p for n, p in decay_swin_param_tp],
            'weight_decay': weight_decay,
            'lr': args.learning_rate * coef_lr},
        {'params': [p for n, p in decay_bert_param_tp],
            'weight_decay': weight_decay},
        {'params': [p for n, p in no_decay_swin_param_tp],
            'weight_decay': 0.0,
            'lr': args.learning_rate * coef_lr},
        {'params': [p for n, p in no_decay_bert_param_tp],
            'weight_decay': 0.0}
    ]

    if args.mixed_precision_method == "fairscale":
        from fairscale.optim.oss import OSS
        optimizer = OSS(
            params=optimizer_grouped_parameters, optim=AdamW, lr=args.learning_rate,
            eps=args.adam_epsilon)
    else:
        optimizer = AdamW(
            optimizer_grouped_parameters, lr=args.learning_rate,
            eps=args.adam_epsilon)
    if args.scheduler == "warmup_linear":
        scheduler = WarmupLinearLR(
            optimizer, max_global_step, warmup_ratio=args.warmup_ratio)
    else:
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=int(max_iter/2.0), gamma=0.1)

    if args.mixed_precision_method == "deepspeed":
        if deepspeed is None:
            raise RuntimeError(
                "mixed_precision_method=deepspeed requires the deepspeed package. "
                "Install deepspeed or use --mixed_precision_method apex for native single-GPU fallback."
            )
        config = get_deepspeed_config(args)
        model, optimizer, _, _ = deepspeed.initialize(
            config_params=config,
            model=model,
            optimizer=optimizer,
            lr_scheduler=scheduler)
    elif args.mixed_precision_method == "fairscale":
        from fairscale.optim.grad_scaler import ShardedGradScaler
        scaler = ShardedGradScaler()
        # this is equivalent to deepspeed zero_opt_stage = 2
        from fairscale.nn.data_parallel import ShardedDataParallel as ShardedDDP
        model = ShardedDDP(
            model, optimizer,
            reduce_buffer_size= 0 if args.fairscale_fp16 else 2 ** 23, # 2 ** 23 is the default value
            reduce_fp16=args.fairscale_fp16)
    else:
        # opt_level O0 is fp32. If Apex is unavailable, run native PyTorch fp32.
        if amp is not None:
            model, optimizer = amp.initialize(
                model, optimizer,
                enabled=True,
                opt_level=f'O{args.amp_opt_level}')
        if args.distributed: #
            if ApexDDP is not None:
                model = ApexDDP(model)
            else:
                device_ids = [args.local_rank] if torch.cuda.is_available() else None
                model = TorchDDP(model, device_ids=device_ids)
    return args, model, optimizer, scheduler

def train(args, train_dataloader, val_dataloader, model, tokenizer, training_saver, optimizer, scheduler):
    meters = MetricLogger(delimiter='  ')
    max_iter = args.max_iter
    max_global_step = args.max_global_step
    global_iters_per_epoch = args.global_iters_per_epoch

    eval_log = []
    best_score = 0

    best_score_exp = 0
    best_B4_exp = 0
    best_score_des_add_exp = 0
    best_repro_score = None
    best_repro_meta = op.join(args.output_dir, 'checkpoint_best', 'repro_checkpoint_meta.json')
    if op.isfile(best_repro_meta):
        try:
            with open(best_repro_meta, 'r') as f:
                best_repro_score = float(json.load(f).get('best_metric_value'))
            logger.info(f"Loaded existing checkpoint_best score: {best_repro_score}")
        except Exception as e:
            logger.info(f"Could not load existing checkpoint_best metadata: {e}")

    start_training_time = time.time()
    end = time.time()
    log_start = time.time()
    running_loss = RunningMeter('train_loss')
    running_batch_acc = RunningMeter('train_batch_acc')

    if getattr(args, 'resume_repro_meta', None):
        global_step = int(args.resume_repro_meta.get('global_step', 0))
        logger.info(f"Resume repro training metadata global_step={global_step}")
    elif args.restore_ratio > 0:
        restorer = TrainingRestorer(args, model, optimizer)
        global_step = restorer.global_step
    else:
        global_step = 0

    TB_LOGGER.global_step = global_step
    if not is_main_process() or args.restore_ratio <= 0:
        restorer = NoOp()

    training_saver.save_args(args)
    training_saver.save_tokenizer(tokenizer)

    for iteration, (img_keys, batch, meta_data) in enumerate(train_dataloader):
        iteration += 1
        data_time = time.time() - end
        batch = tuple(t.to(args.device) for t in batch)
        model.train()
        # img_feats (B, #F, C, W, H)
        inputs = {
            'input_ids': batch[0], 'attention_mask': batch[1],
            'token_type_ids': batch[2], 'img_feats': batch[3],
            'masked_pos': batch[4], 'masked_ids': batch[5],
            'car_info': batch[6],
        }

        if iteration == 1:
            for k, v in inputs.items():
                logger.info(f'{k} = {v.shape}')

        if getattr(args, 'deepspeed_bf16', False):
            # DeepSpeed does not autocast inputs.
            inputs = fp32_to_bf16(inputs)
        elif args.deepspeed_fp16:
            # deepspeed does not autocast inputs
            inputs = fp32_to_fp16(inputs)

        if args.mixed_precision_method == "fairscale":
            with torch.cuda.amp.autocast(enabled=True):
                outputs = model(**inputs)
        else:
            outputs = model(**inputs)
        flowtrace_loss = None
        flowtrace_loss_components = None
        if (
            isinstance(outputs, tuple)
            and len(outputs) >= 2
            and isinstance(outputs[-1], dict)
            and "flowtrace_loss_components" in outputs[-1]
        ):
            flowtrace_loss_components = outputs[-1]["flowtrace_loss_components"]
            flowtrace_loss = outputs[-2]
            outputs = outputs[:-2]
        loss, logits = outputs[:2]

        if args.multitask:
            logits_sensor = outputs[-2]
            loss_sensor = outputs[-3]
            loss = loss + (loss_sensor * args.loss_sensor_w)
        if flowtrace_loss is not None:
            loss = loss + flowtrace_loss
        if args.learn_mask_enabled:
            loss_sparsity = outputs[-1]
            loss = loss + (loss_sparsity * args.loss_sparse_w)
        masked_ids = inputs['masked_ids']
        masked_ids = masked_ids[masked_ids != -1]
        batch_score = compute_score_with_logits(logits, masked_ids)
        batch_acc = torch.sum(batch_score.float()) / torch.sum(inputs['masked_pos'])

        if args.learn_mask_enabled:
            loss_dict = {'loss': loss, 'loss_sparsity': loss_sparsity.item(), 'acc': batch_acc}
        else:
            loss_dict = {'loss': loss, 'acc': batch_acc}

        if args.multitask:
            loss_dict['loss_sensor'] = loss_sensor
        if flowtrace_loss is not None:
            loss_dict['loss_flowtrace'] = flowtrace_loss.detach()
            if flowtrace_loss_components:
                for key, value in flowtrace_loss_components.items():
                    if isinstance(value, (int, float)):
                        loss_dict[f'flowtrace_{key}'] = float(value)
        meters.update(**loss_dict)

        running_loss(loss.item())
        running_batch_acc(batch_acc.item())

        # backward pass
        backward_now = iteration % args.gradient_accumulation_steps == 0
        loss_for_backward = loss if args.mixed_precision_method == "deepspeed" else loss / float(args.gradient_accumulation_steps)
        if args.mixed_precision_method == "deepspeed":
            model.backward(loss)
        elif args.mixed_precision_method == "fairscale":
            scaler.scale(loss_for_backward).backward()
        else:
            if amp is not None:
                with amp.scale_loss(loss_for_backward, optimizer, delay_unscale=not backward_now) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss_for_backward.backward()
        if backward_now:
            _write_flowtrace_smoke_train_evidence(
                args,
                img_keys=img_keys,
                inputs=inputs,
                logits=logits,
                masked_ids=masked_ids,
                loss=loss,
                model=model,
                flowtrace_loss_components=flowtrace_loss_components,
            )
            global_step += 1
            TB_LOGGER.add_scalar('train/loss', running_loss.val, global_step)
            if args.multitask:
                TB_LOGGER.add_scalar('train/loss_sensor', loss_sensor.cpu(), global_step)
            if args.learn_mask_enabled:
                TB_LOGGER.add_scalar('train/loss_sparsity', loss_sparsity.cpu(), global_step)

            lr_VisBone = optimizer.param_groups[0]["lr"]
            lr_LM = optimizer.param_groups[1]["lr"]

            TB_LOGGER.add_scalar(
                "train/lr_lm", lr_LM, global_step)
            TB_LOGGER.add_scalar(
                "train/ls_visBone", lr_VisBone, global_step)

            if args.max_grad_norm != -1:
                grad_params = (
                    amp.master_params(optimizer)
                    if amp is not None and args.mixed_precision_method == "apex"
                    else model.parameters()
                )
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    grad_params, args.max_grad_norm)
                TB_LOGGER.add_scalar("train/grad_norm", grad_norm, global_step)
            TB_LOGGER.step()
            if args.mixed_precision_method == "deepspeed":
                model.step()
            elif args.mixed_precision_method == "fairscale":
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                model.zero_grad()
            else:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            restorer.step()

        batch_time = time.time() - end

        if backward_now:
            if global_step % args.logging_steps == 0 or global_step == max_global_step:
                if 'time_info' in meters.meters:
                    avg_time = meters.meters['time_info']['compute'].global_avg
                    eta_seconds = avg_time * (max_iter - iteration)
                    eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                else:
                    eta_string = 'Unknown'
                eta_seconds = batch_time * (max_iter - iteration)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                speed = args.num_gpus * args.logging_steps * len(batch[0]) / (time.time() - log_start)
                memory = torch.cuda.max_memory_allocated() / 1024.0 / 1024.0
                logger.info(
                    meters.delimiter.join(
                        [
                            f"eta: {eta_string}",
                            f"iter: {iteration}",
                            f"global_step: {global_step}",
                            f'speed: {speed:.1f} images/sec',
                            f"{meters}",
                            f"lr (Visual Encoder): {lr_VisBone:.2e}",
                            f"lr (LM): {lr_LM:.2e}",
                            f"max mem: {memory:.0f}",
                        ]
                    )
                )
                TB_LOGGER.add_scalar("train/speed", speed, global_step)
                TB_LOGGER.add_scalar("train/memory", memory, global_step)
                TB_LOGGER.add_scalar("train/batch_time", batch_time, global_step)
                TB_LOGGER.add_scalar("train/data_time", data_time, global_step)
                log_start = time.time()


            if (args.save_steps > 0 and global_step % args.save_steps == 0) or global_step == max_global_step or global_step == 1:
                epoch = global_step // global_iters_per_epoch

                checkpoint_dir = op.join(args.output_dir, 'checkpoint-{}-{}'.format(
                    epoch, global_step))
                if get_world_size() > 1:
                    dist.barrier()

                if get_world_size() > 1:
                    dist.barrier()
                if is_main_process():
                    pre_eval_meta = {
                        'tag': 'checkpoint_latest',
                        'epoch': int(epoch),
                        'iteration': int(iteration),
                        'global_step': int(global_step),
                        'metric_name': None,
                        'metric_value': None,
                        'saved_at': datetime.datetime.now().isoformat(),
                        'note': 'pre_eval_latest_for_resume',
                    }
                    save_repro_checkpoint(
                        training_saver, args, 'checkpoint_latest',
                        global_step, model, optimizer, pre_eval_meta)
                if args.evaluate_during_training:
                    logger.info(f"Perform evaluation at iteration {iteration}, global_step {global_step}")
                    evaluate_file = evaluate(args, val_dataloader, model, tokenizer, checkpoint_dir)
                    if args.multitask:
                        signal_evaluate(args, val_dataloader, model, tokenizer, checkpoint_dir)
                    if get_world_size() > 1:
                        dist.barrier()

                    # this is dull and foolish but effective and efficient
                    if is_main_process():
                        repro_metric_name = None
                        repro_metric_value = None
                        if args.use_sep_cap:
                            evaluate_files = [evaluate_file.replace('BDDX', 'BDDX_des'), evaluate_file.replace('BDDX', 'BDDX_exp')]
                            caps_name = ['des', 'exp']
                            score_des_add_exp = 0
                            for cap_ord, eval_file in enumerate(evaluate_files):
                                with open(eval_file, 'r') as f:
                                    res = json.load(f)
                                val_log = {f'valid/{caps_name[cap_ord]}_{k}': v for k,v in res.items()}
                                TB_LOGGER.log_scalar_dict(val_log)
                                aml_run.log(name='CIDEr', value=float(res['CIDEr']))

                                score_des_add_exp += res['CIDEr']

                                if cap_ord == 0 and res['CIDEr'] > 2.4:
                                    print(f"best B4:{best_B4_exp}\tbest exp cider:{best_score_exp}\tbest cider sum:{score_des_add_exp}")
                                    training_saver.save_model(
                                        checkpoint_dir, global_step, model, optimizer)
                                elif cap_ord == 1 and res['CIDEr'] > 1.0:
                                    print(f"best B4:{best_B4_exp}\tbest exp cider:{best_score_exp}\tbest cider sum:{score_des_add_exp}")
                                    training_saver.save_model(
                                        checkpoint_dir, global_step, model, optimizer)

                                if cap_ord == 1:
                                    best_score_exp = max(best_score_exp, res['CIDEr'])
                                    best_B4_exp = max(best_B4_exp, res['Bleu_4'])
                                    best_score_des_add_exp = max(best_score_des_add_exp, score_des_add_exp)
                                    res['epoch'] = epoch
                                    res['iteration'] = iteration
                                    res['best_B4_exp'] = best_B4_exp
                                    res['best_CIDEr_exp'] = best_score_exp
                                    res['best_CIDEr_sum'] = score_des_add_exp
                                    eval_log.append(res)
                                    with open(op.join(args.output_dir, args.val_yaml.replace('/','_')+'eval_logs.json'), 'w') as f:
                                        json.dump(eval_log, f)
                                    repro_metric_name = 'CIDEr_des_plus_exp'
                                    repro_metric_value = float(score_des_add_exp)
                        else:
                            with open(evaluate_file, 'r') as f:
                                res = json.load(f)
                            val_log = {f'valid/{k}': v for k,v in res.items()}
                            TB_LOGGER.log_scalar_dict(val_log)
                            aml_run.log(name='CIDEr', value=float(res['CIDEr']))

                            best_score = max(best_score, res['CIDEr'])
                            res['epoch'] = epoch
                            res['iteration'] = iteration
                            res['best_CIDEr'] = best_score
                            eval_log.append(res)
                            with open(op.join(args.output_dir, args.val_yaml.replace('/','_')+'eval_logs.json'), 'w') as f:
                                json.dump(eval_log, f)
                            repro_metric_name = 'CIDEr'
                            repro_metric_value = float(res['CIDEr'])
                        if repro_metric_value is not None:
                            repro_meta = {
                                'tag': 'checkpoint_latest',
                                'epoch': int(epoch),
                                'iteration': int(iteration),
                                'global_step': int(global_step),
                                'metric_name': repro_metric_name,
                                'metric_value': repro_metric_value,
                                'saved_at': datetime.datetime.now().isoformat(),
                            }
                            save_repro_checkpoint(
                                training_saver, args, 'checkpoint_latest',
                                global_step, model, optimizer, repro_meta)
                            if best_repro_score is None or repro_metric_value > best_repro_score:
                                best_repro_score = repro_metric_value
                                best_meta = dict(repro_meta)
                                best_meta['tag'] = 'checkpoint_best'
                                best_meta['best_metric_value'] = best_repro_score
                                save_repro_checkpoint(
                                    training_saver, args, 'checkpoint_best',
                                    global_step, model, optimizer, best_meta)
                    if get_world_size() > 1:
                        dist.barrier()

        if iteration > 2:
            meters.update(
                batch_time=batch_time,
                data_time=data_time,
            )
        end = time.time()

        if _reached_flowtrace_train_step_limit(args, global_step):
            logger.info(
                f"FlowTrace hard smoke train-step limit reached: "
                f"{global_step}/{_flowtrace_train_step_limit(args)}"
            )
            break

        if global_step >= max_global_step and (max_iter - iteration):
            logger.info(f'Missing {max_iter - iteration} iterations, early break')
            break

    total_training_time = time.time() - start_training_time
    total_time_str = str(datetime.timedelta(seconds=total_training_time))
    logger.info(f'Total training time: {total_time_str} ({(total_training_time / max_iter):.4f} s / iter)')
    return checkpoint_dir

def get_predict_file(output_dir, args, data_yaml_file):
    cc = ['pred']
    # example data_yaml_file: datasets/coco_caption/test.yaml
    data = data_yaml_file.split('/')[-2]
    if data != 'coco_caption':
        cc.append(data)
    cc.append(op.splitext(op.basename(data_yaml_file))[0])
    cc.append('beam{}'.format(args.num_beams))
    cc.append('max{}'.format(args.max_gen_length))
    if args.num_keep_best != 1:
        cc.append('best{}'.format(args.num_keep_best))
    if args.output_hidden_states:
        cc.append('hidden')
    return op.join(output_dir, '{}.tsv'.format('.'.join(cc)))

def get_evaluate_file(predict_file):
    assert predict_file.endswith('.tsv')
    return op.splitext(predict_file)[0] + '.eval.json'

def evaluate(args, val_dataloader, model, tokenizer, output_dir):
    predict_file = get_predict_file(output_dir, args,
            val_dataloader.dataset.yaml_file)
    test(args, val_dataloader, model, tokenizer, predict_file)

    if get_world_size() > 1:
        dist.barrier()
    evaluate_file = get_evaluate_file(predict_file)
    if is_main_process():
        caption_file = val_dataloader.dataset.get_caption_file_in_coco_format()
        data = val_dataloader.dataset.yaml_file.split('/')[-2]
        if args.use_sep_cap:
            result = two_cap_evaluate_on_coco_caption(predict_file, caption_file, outfile=evaluate_file)
        else:
            result = evaluate_on_coco_caption(predict_file, caption_file, outfile=evaluate_file)
        logger.info(f'evaluation result: {str(result)}')
        logger.info(f'evaluation result saved to {evaluate_file}')
    if get_world_size() > 1:
        dist.barrier()
    return evaluate_file

def test(args, test_dataloader, model, tokenizer, predict_file):

    cls_token_id, sep_token_id, pad_token_id, mask_token_id, period_token_id = \
        tokenizer.convert_tokens_to_ids([tokenizer.cls_token, tokenizer.sep_token,
        tokenizer.pad_token, tokenizer.mask_token, '.'])
    world_size = get_world_size()
    if world_size == 1:
        cache_file = predict_file
    else:
        # local_rank would not work for cross-node distributed training
        cache_file = op.splitext(predict_file)[0] + '_{}_{}'.format(get_rank(),
                world_size) + op.splitext(predict_file)[1]

    model.eval()
    def gen_rows():
        time_meter = 0
        # restore existing results for long running inference tasks
        exist_key2pred = {}
        tmp_file = cache_file + '.tmp.copy'
        if op.isfile(tmp_file):
            with open(tmp_file, 'r') as fp:
                for line in fp:
                    parts = line.strip().split('\t')
                    if len(parts) == 2:
                        exist_key2pred[parts[0]] = parts[1]

        with torch.no_grad():
            for step, (img_keys, batch, meta_data) in tqdm(enumerate(test_dataloader)):
                # torch.cuda.empty_cache()
                # is_exist = True
                # for k in img_keys:
                #     if k not in exist_key2pred:
                #         is_exist = False
                #         break
                # if is_exist:
                #     for k in img_keys:
                #         yield k, exist_key2pred[k]
                #         # return k, exist_key2pred[k]
                #     continue
                # if step > 4:
                #     break
                batch = tuple(t.to(args.device) for t in batch)
                inputs = {'is_decode': True,
                    'input_ids': batch[0], 'attention_mask': batch[1],
                    'token_type_ids': batch[2], 'img_feats': batch[3],
                    'masked_pos': batch[4],
                    'car_info': batch[5],
                    'do_sample': False,
                    'bos_token_id': cls_token_id,
                    'pad_token_id': pad_token_id,
                    'eos_token_ids': [sep_token_id],
                    'mask_token_id': mask_token_id,
                    # for adding od labels
                    'add_od_labels': args.add_od_labels, 'od_labels_start_posid': args.max_seq_a_length,
                    # hyperparameters of beam search
                    'max_length': args.max_gen_length if not args.use_sep_cap else args.max_gen_length*2,
                    'use_sep_cap': args.use_sep_cap,
                    'num_beams': args.num_beams,
                    "temperature": args.temperature,
                    "top_k": args.top_k,
                    "top_p": args.top_p,
                    "repetition_penalty": args.repetition_penalty,
                    "length_penalty": args.length_penalty,
                    "num_return_sequences": args.num_return_sequences,
                    "num_keep_best": args.num_keep_best,
                }

                tic = time.time()
                # captions, logprobs

                if getattr(args, 'deepspeed_bf16', False):
                    # deepspeed does not auto cast inputs.
                    inputs = fp32_to_bf16(inputs)
                elif args.deepspeed_fp16:
                    # deepspeed does not auto cast inputs.
                    inputs = fp32_to_fp16(inputs)

                if args.mixed_precision_method == "fairscale":
                    with torch.cuda.amp.autocast(enabled=True):
                        outputs = model(**inputs)
                else:
                    outputs = model(**inputs)
                time_meter += time.time() - tic
                all_caps = outputs[0]  # batch_size * num_keep_best * max_len
                all_confs = torch.exp(outputs[1])

                if not args.use_sep_cap:
                    for img_key, caps, confs in zip(img_keys, all_caps, all_confs):
                        res = []
                        for cap, conf in zip(caps, confs):
                            cap = tokenizer.decode(cap.tolist(), skip_special_tokens=True)
                            res.append({'caption': cap, 'conf': conf.item()})
                        if isinstance(img_key, torch.Tensor):
                            img_key = img_key.item()
                        yield img_key, json.dumps(res)
                        # return img_key, json.dumps(res)
                else:
                    for img_key, caps, confs in zip(img_keys, all_caps, all_confs):
                        all_cap_a = []
                        all_cap_b = []
                        sep_place = args.max_gen_length
                        for cap, conf in zip(caps, confs):
                            cap_1 = tokenizer.decode(cap.tolist()[:sep_place], skip_special_tokens=True)
                            cap_2 = tokenizer.decode(cap.tolist()[sep_place:], skip_special_tokens=True)
                            all_cap_a.append({'caption': cap_1, 'conf': conf.item()})
                            all_cap_b.append({'caption': cap_2, 'conf': conf.item()})
                        if isinstance(img_key, torch.Tensor):
                            img_key = img_key.item()
                        if args.use_swap_cap:
                            yield img_key, json.dumps(all_cap_b), json.dumps(all_cap_a)
                        else:
                            yield img_key, json.dumps(all_cap_a), json.dumps(all_cap_b)
                        # return img_key, json.dumps(all_cap_a), json.dumps(all_cap_b)

        logger.info(f"Inference model computing time: {(time_meter / (step+1))} seconds per batch")

    # a = gen_rows()
    if args.use_sep_cap:
        double_tsv_writer(gen_rows(), cache_file)
    else:
        tsv_writer(gen_rows(), cache_file)
    if world_size > 1:
        dist.barrier()
    if world_size > 1 and is_main_process():
        cache_files = [op.splitext(predict_file)[0] + '_{}_{}'.format(i, world_size) + \
            op.splitext(predict_file)[1] for i in range(world_size)]
        concat_tsv_files(cache_files, predict_file)
        delete_tsv_files(cache_files)
        reorder_tsv_keys(predict_file, test_dataloader.dataset.image_keys, predict_file)
    if world_size > 1:
        dist.barrier()

def signal_evaluate(args, val_dataloader, model, tokenizer, output_dir):
    predict_file = get_predict_file(output_dir, args,
            val_dataloader.dataset.yaml_file)

    cls_token_id, sep_token_id, pad_token_id, mask_token_id, period_token_id = \
        tokenizer.convert_tokens_to_ids([tokenizer.cls_token, tokenizer.sep_token,
        tokenizer.pad_token, tokenizer.mask_token, '.'])
    world_size = get_world_size()

    cache_file = predict_file

    model.eval()
    def gen_rows():
        time_meter = 0
        # restore existing results for long running inference tasks
        exist_key2pred = {}
        tmp_file = cache_file + '.tmp.copy'
        if op.isfile(tmp_file):
            with open(tmp_file, 'r') as fp:
                for line in fp:
                    parts = line.strip().split('\t')
                    if len(parts) == 2:
                        exist_key2pred[parts[0]] = parts[1]

        gt_signals = []
        pred_signals = []

        with torch.no_grad():
            for step, (img_keys, batch, meta_data) in tqdm(enumerate(val_dataloader)):

                # if step > 4:
                #     break

                batch = tuple(t.to(args.device) for t in batch)
                inputs = {'is_decode': True,
                    'input_ids': batch[0], 'attention_mask': batch[1],
                    'token_type_ids': batch[2], 'img_feats': batch[3],
                    'masked_pos': batch[4],
                    'car_info': batch[5],
                    'do_sample': False,
                    'bos_token_id': cls_token_id,
                    'pad_token_id': pad_token_id,
                    'eos_token_ids': [sep_token_id],
                    'mask_token_id': mask_token_id,
                    # for adding od labels
                    'add_od_labels': args.add_od_labels, 'od_labels_start_posid': args.max_seq_a_length,
                    # hyperparameters of beam search
                    'max_length': args.max_gen_length if not args.use_sep_cap else args.max_gen_length*2,
                    'use_sep_cap': args.use_sep_cap,
                    'num_beams': args.num_beams,
                    "temperature": args.temperature,
                    "top_k": args.top_k,
                    "top_p": args.top_p,
                    "repetition_penalty": args.repetition_penalty,
                    "length_penalty": args.length_penalty,
                    "num_return_sequences": args.num_return_sequences,
                    "num_keep_best": args.num_keep_best,
                }


                if getattr(args, 'deepspeed_bf16', False):
                    # deepspeed does not auto cast inputs.
                    inputs = fp32_to_bf16(inputs)
                elif args.deepspeed_fp16:
                    # deepspeed does not auto cast inputs.
                    inputs = fp32_to_fp16(inputs)

                if args.mixed_precision_method == "fairscale":
                    with torch.cuda.amp.autocast(enabled=True):
                        outputs = model(**inputs)
                else:
                    outputs = model(**inputs)
                outputs = outputs
                for b in range(len(batch[0])):
                    # if all of the control signal is -1, then we know the info file is missed
                    # however, this missed info doesn't affect the results of captions due to our multi-task architecture
                    if not (batch[5][b]==-1).all():
                        gt_signals.append(batch[5][b])
                        pred_signals.append(outputs[-2][b])

        return _stack_signal_rows(gt_signals, pred_signals)

    signal_rows = gen_rows()
    if signal_rows is None:
        if is_main_process():
            unavailable_file = op.splitext(predict_file)[0] + ".signal_unavailable.json"
            with open(unavailable_file, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "available": False,
                        "reason": "no_valid_control_signal_rows",
                        "limited_eval_samples": int(getattr(args, "limited_eval_samples", 0) or 0),
                    },
                    f,
                    indent=2,
                )
            logger.info(f"signal evaluation skipped: no valid control rows; wrote {unavailable_file}")
        return None

    gt_signals, pred_signals = signal_rows

    if world_size > 1:
        dist.barrier()

    if is_main_process():
        print("computing signal prediction score")
        sigma_1 = 0.1
        sigma_2 = 0.5
        sigma_3 = 1
        sigma_4 = 5
        sigma_5 = 10

        sig1_acc = 0
        sig2_acc = 0
        sig3_acc = 0
        sig4_acc = 0
        sig5_acc = 0
        assert len(gt_signals) == len(pred_signals)

        for signal_order in range(len(args.signal_types)):
            signal_name = args.signal_types[signal_order]
            gt_signal = gt_signals[:, signal_order, :].cpu()
            pred_signal = pred_signals[:, :, signal_order].cpu()
            import numpy as np
            from sklearn.metrics import mean_squared_error
            rmse_signal = np.sqrt(mean_squared_error(gt_signal, pred_signal))

            print(f"{signal_name} \t rmse:{rmse_signal}")
            all_num = gt_signal.shape[0] * gt_signal.shape[1]   # B*frame_num
            sig1_acc = (np.count_nonzero(abs(gt_signal-pred_signal)<sigma_1)/all_num,)
            print(f"sig1_acc \t {sig1_acc}")
            sig2_acc = (np.count_nonzero(abs(gt_signal-pred_signal)<sigma_2)/all_num,)
            print(f"sig1_acc \t {sig2_acc}")
            sig3_acc = (np.count_nonzero(abs(gt_signal-pred_signal)<sigma_3)/all_num,)
            print(f"sig1_acc \t {sig3_acc}")
            sig4_acc = (np.count_nonzero(abs(gt_signal-pred_signal)<sigma_4)/all_num,)
            print(f"sig1_acc \t {sig4_acc}")
            sig5_acc = (np.count_nonzero(abs(gt_signal-pred_signal)<sigma_5)/all_num,)
            print(f"sig1_acc \t {sig5_acc}")
            print(all_num)
            if not os.path.exists(op.dirname(predict_file)):
                os.makedirs(op.dirname(predict_file))
            with open(op.dirname(predict_file) +f'/{signal_name}_test_data.json', 'w') as json_file:
                json_file.write(str({f"rmse_{signal_name}":rmse_signal,
                        # "rmse_speed":rmse_speed,
                        "sig1_acc":sig1_acc,
                        "sig2_acc":sig2_acc,
                        "sig3_acc":sig3_acc,
                        "sig4_acc":sig4_acc,
                        "sig5_acc":sig5_acc,
                        }))
    if world_size > 1:
        dist.barrier()
    if get_world_size() > 1:
        dist.barrier()
    return

def check_arguments(args):
    # shared basic checks
    basic_check_arguments(args)
    # additional sanity check:
    args.max_img_seq_length = int((args.max_num_frames/2)*(int(args.img_res)/32)*(int(args.img_res)/32))

    if args.freeze_backbone or args.backbone_coef_lr == 0:
        args.backbone_coef_lr = 0
        args.freeze_backbone = True

    if 'reload_pretrained_swin' not in args.keys():
        args.reload_pretrained_swin = False

    if not len(args.pretrained_checkpoint) and args.reload_pretrained_swin:
        logger.info("No pretrained_checkpoint to be loaded, disable --reload_pretrained_swin")
        args.reload_pretrained_swin = False

    validate_fate_x_mask_compatibility(args)
    apply_lr_scaling_to_args(args)

    if args.learn_mask_enabled==True and args.attn_mask_type != 'learn_without_crossattn' and args.attn_mask_type != 'learn_with_swap_crossattn':
        args.attn_mask_type = 'learn_vid_att'

def update_existing_config_for_inference(args):
    ''' load adapt args for evaluation and inference
    '''
    assert args.do_test or args.do_eval
    checkpoint = args.eval_model_dir
    try:
        json_path = op.join(checkpoint, os.pardir, 'log', 'args.json')
        f = open(json_path,'r')
        json_data = json.load(f)

        from easydict import EasyDict
        train_args = EasyDict(json_data)
    except Exception as e:
        train_args = torch.load(op.join(checkpoint, 'training_args.bin'))

    train_args.eval_model_dir = args.eval_model_dir
    train_args.resume_checkpoint = args.eval_model_dir + 'model.bin'
    train_args.model_name_or_path = 'models/captioning/bert-base-uncased/'
    train_args.do_train = False
    train_args.do_eval = True
    train_args.do_signal_eval = True if hasattr(args, 'do_signal_eval') and args.do_signal_eval else False
    train_args.do_test = True
    train_args.val_yaml = args.val_yaml
    train_args.test_video_fname = args.test_video_fname
    train_args.signal_types = args.signal_types
    return train_args

def get_custom_args(base_config):
    parser = base_config.parser
    parser.add_argument('--max_num_frames', type=int, default=32)
    parser.add_argument('--img_res', type=int, default=224)
    parser.add_argument('--patch_size', type=int, default=32)
    parser.add_argument("--grid_feat", type=str_to_bool, nargs='?', const=True, default=True)
    parser.add_argument("--kinetics", type=str, default='400', help="400 or 600")
    parser.add_argument("--pretrained_2d", type=str_to_bool, nargs='?', const=True, default=False)
    parser.add_argument("--vidswin_size", type=str, default='base')
    parser.add_argument('--freeze_backbone', type=str_to_bool, nargs='?', const=True, default=False)
    parser.add_argument('--use_checkpoint', type=str_to_bool, nargs='?', const=True, default=False)
    parser.add_argument('--backbone_coef_lr', type=float, default=0.001)
    parser.add_argument("--reload_pretrained_swin", type=str_to_bool, nargs='?', const=True, default=False)
    parser.add_argument('--learn_mask_enabled', type=str_to_bool, nargs='?', const=True, default=False)
    parser.add_argument('--loss_sparse_w', type=float, default=0)
    parser.add_argument('--loss_sensor_w', type=float, default=0)
    parser.add_argument('--sparse_mask_soft2hard', type=str_to_bool, nargs='?', const=True, default=False)
    parser.add_argument('--transfer_method', type=int, default=-1,
                        help="0: load all ADAPT pre-trained weights, 1: load only pre-trained sparse mask")
    parser.add_argument('--att_mask_expansion', type=int, default=-1,
                        help="-1: random init, 0: random init and then diag-based copy, 1: interpolation")
    parser.add_argument('--resume_checkpoint', type=str, default='None')
    parser.add_argument('--resume_repro_checkpoint_dir', type=str, default='',
                        help="Resume from a repro checkpoint directory containing model.bin, optmizer_state.bin, and repro_checkpoint_meta.json.")
    parser.add_argument('--test_video_fname', type=str, default='None')
    # FATE-X flags are default-off so original ADAPT behavior is preserved.
    parser.add_argument('--fate_x_enabled', type=str_to_bool, nargs='?', const=True, default=False)
    parser.add_argument('--video_token_reducer', type=str, default='none', choices=['none', 'merge', 'topk_merge', 'per_frame_topk_merge'])
    parser.add_argument('--fate_x_keep_ratio', type=float, default=0.5)
    parser.add_argument('--fate_x_num_summary_tokens', type=int, default=1)
    parser.add_argument('--fate_x_min_tokens', type=int, default=128)
    parser.add_argument('--fate_x_min_tokens_per_frame', type=int, default=1)
    parser.add_argument('--fate_x_temporal_tokens', type=int, default=0)
    parser.add_argument('--fate_x_spatial_tokens_per_frame', type=int, default=0)
    parser.add_argument('--fate_x_summary_mode', type=str, default='cluster', choices=['global_mean', 'cluster', 'per_frame_cluster'])
    parser.add_argument('--fate_x_text_reduce_only', type=str_to_bool, nargs='?', const=True, default=True)
    parser.add_argument('--fate_x_reduce_control', type=str_to_bool, nargs='?', const=True, default=False)
    parser.add_argument('--fate_x_control_reducer', type=str, default='none', choices=['none', 'per_frame_topk_merge', 'temporal_ordered_topk'])
    parser.add_argument('--temporal_evidence_memory', type=str, default='none', choices=['none', 'queries'])
    parser.add_argument('--reference_effective_batch', type=int, default=64)
    parser.add_argument('--base_learning_rate_at_reference_batch', type=float, default=0.0002)
    parser.add_argument('--auto_scale_lr', type=str_to_bool, nargs='?', const=True, default=False)
    parser.add_argument('--num_gpus_for_lr', type=int, default=1)
    parser.add_argument('--phrase_faithfulness_enabled', type=str_to_bool, nargs='?', const=True, default=False)
    parser.add_argument('--visualize_phrase_attention', type=str_to_bool, nargs='?', const=True, default=False)
    parser.add_argument('--flowtrace_enabled', type=str_to_bool, nargs='?', const=True, default=False)
    parser.add_argument('--flowtrace_state_dim', type=int, default=256)
    parser.add_argument('--flowtrace_pmt_rank', type=int, default=32)
    parser.add_argument('--flowtrace_smoke_evidence', type=str, default='')
    parser.add_argument('--flowtrace_max_train_steps', type=int, default=0)
    args = base_config.parse_args()
    return args

def main(args):
    if args.do_train==False or args.do_eval==True:
        args = update_existing_config_for_inference(args)

    args.device = torch.device(args.device)

    dist_init(args)
    logger.info("Setup CUDA, GPU & distributed training")

    check_arguments(args)
    logger.info("Check arguments")

    resume_repro_checkpoint_dir = getattr(args, 'resume_repro_checkpoint_dir', '')
    if resume_repro_checkpoint_dir and resume_repro_checkpoint_dir != 'None':
        args.resume_checkpoint = op.join(resume_repro_checkpoint_dir, 'model.bin')
        logger.info(f"Resume repro checkpoint dir: {resume_repro_checkpoint_dir}")
        logger.info(f"Resume repro model checkpoint: {args.resume_checkpoint}")

    mkdir(args.output_dir)
    logger.info(f"creating output_dir at: {args.output_dir}")

    set_seed(args.seed, args.num_gpus)

    if args.mixed_precision_method == "apex":
        fp16_trainning = f"apex O{args.amp_opt_level}"
    elif args.mixed_precision_method == "deepspeed":
        if getattr(args, 'deepspeed_bf16', False):
            precision_info = f'bf16, {args.zero_opt_stage}'
        elif args.deepspeed_fp16:
            precision_info = f'fp16, {args.zero_opt_stage}'
        else:
            precision_info = f'amp, {args.amp_opt_level}'
        fp16_trainning = f"deepspeed, {precision_info}"
    elif args.mixed_precision_method == "fairscale":
        assert args.distributed, "fairscale can only be used for distributed training"
        fp16_trainning = f"fairscale, fp16: {args.fairscale_fp16}, default zero_opt 2"
    else:
        fp16_trainning = None

    logger.info(
        "device: {}, n_gpu: {}, rank: {}, "
        "16-bits training: {}".format(
            args.device, args.num_gpus, get_rank(), fp16_trainning))

    if not is_main_process():
        logger.disabled = True
        training_saver = NoOp()
    else:
        training_saver = TrainingSaver(args.output_dir)
        TB_LOGGER.create(op.join(args.output_dir, 'log'))
        add_log_to_file(op.join(args.output_dir, 'log', "log.txt"))

    logger.info(f"Pytorch version is: {torch.__version__}")
    logger.info(f"Cuda version is: {torch.version.cuda}")
    logger.info(f"cuDNN version is : {torch.backends.cudnn.version()}" )

    # Get Video Swin backbone
    swin_model = get_swin_model(args)

    # Get BERT and tokenizer for DCG (Driving Caption Generation)
    bert_model, config, tokenizer = get_bert_model(args)

    # build ADAPT based on training configs
    if args.multitask:
        vl_transformer = MultitaskVideoTransformer(args, config, swin_model, bert_model)
    else:
        vl_transformer = VideoTransformer(args, config, swin_model, bert_model)
    vl_transformer.freeze_backbone(freeze=args.freeze_backbone)

    if args.do_eval:
        # load weights for eval/inference
        logger.info(f"Loading state dict from checkpoint {args.resume_checkpoint}")
        cpu_device = torch.device('cpu')
        pretrained_model = torch.load(args.resume_checkpoint, map_location=cpu_device)

        load_compatible_state_dict(vl_transformer, pretrained_model, context="resume_checkpoint")

    elif args.do_train and args.pretrained_checkpoint != '':
        ckpt_path = args.pretrained_checkpoint+'model.bin'
        assert op.exists(ckpt_path), f"{ckpt_path} does not exist"
        logger.info(f"Loading state dict from checkpoint {ckpt_path}")
        cpu_device = torch.device('cpu')
        pretrained_model = torch.load(ckpt_path, map_location=cpu_device)

        if args.learn_mask_enabled == False:
            load_compatible_state_dict(vl_transformer, pretrained_model, context="pretrained_checkpoint")

        elif args.learn_mask_enabled == True:
            pretrained_mask_shape = pretrained_model['learn_vid_att.weight'].shape
            init_mask_shape = vl_transformer.learn_vid_att.weight.shape

            #-------------------------------------------------------------
            # transfer at the same frame rate
            if pretrained_mask_shape==init_mask_shape:
                # init using entire pre-trained ADAPT weights
                if args.transfer_method==0:
                    load_compatible_state_dict(vl_transformer, pretrained_model, context="pretrained_checkpoint_mask")
                # init using only pre-trained sparse att mask weights
                else:
                    vl_transformer.reload_attn_mask(pretrained_model['learn_vid_att.weight'])
            #-------------------------------------------------------------
            # transfer across different frame rates
            else:
                # init using entire pre-trained ADAPT weights, except sparse attn mask
                if args.transfer_method==0:
                    if isinstance(pretrained_model, dict):
                        new_state_dict={}
                        for k,v in zip(pretrained_model.keys(), pretrained_model.values()):
                            if k!='learn_vid_att.weight' or k=='learn_vid_att.weight' and pretrained_mask_shape==init_mask_shape:
                                new_state_dict[k]=v
                        load_compatible_state_dict(vl_transformer, new_state_dict, context="pretrained_checkpoint_mask_transfer")
                        del new_state_dict
                    else:
                        pretrained_model_state_dict = pretrained_model.state_dict()
                        new_state_dict={}
                        for k,v in zip(pretrained_model_state_dict.keys(), pretrained_model_state_dict.values()):
                            if k!='learn_vid_att.weight' or k=='learn_vid_att.weight' and pretrained_mask_shape==init_mask_shape:
                                new_state_dict[k]=v
                        load_compatible_state_dict(vl_transformer, new_state_dict, context="pretrained_checkpoint_mask_transfer")
                        del new_state_dict

                # expand pre-trained sparse att mask to the desired size
                if args.att_mask_expansion==0:
                    vl_transformer.diag_based_init_attn_mask(pretrained_model['learn_vid_att.weight'])
                elif args.att_mask_expansion==1:
                    vl_transformer.bilinear_init_attn_mask(pretrained_model['learn_vid_att.weight'])
                else:
                    vl_transformer.random_init_attn_mask()

        del pretrained_model
        gc.collect()
        torch.cuda.empty_cache()

        args.eval_model_dir = args.pretrained_checkpoint
        checkpoint = args.eval_model_dir
        assert op.isdir(checkpoint)
        vl_transformer.max_img_seq_length = int(args.max_img_seq_length)
        vl_transformer.config.num_visual_tokens = int(args.max_img_seq_length)
        args.model_name_or_path = args.pretrained_checkpoint
        if args.reload_pretrained_swin:
            vl_transformer.swin = reload_pretrained_swin(vl_transformer.swin, args)

    vl_transformer.to(args.device)

    if args.do_train:
        args = restore_training_settings(args)
        train_dataloader = make_data_loader(args, args.train_yaml, tokenizer, args.distributed, is_train=True)
        val_dataloader = make_data_loader(args, args.val_yaml, tokenizer, args.distributed, is_train=False)

        args.max_iter = len(train_dataloader)
        args.max_global_step =  args.max_iter// args.gradient_accumulation_steps
        flowtrace_step_limit = _flowtrace_train_step_limit(args)
        if flowtrace_step_limit > 0:
            args.max_global_step = min(args.max_global_step, flowtrace_step_limit)
            args.max_iter = min(args.max_iter, args.max_global_step * args.gradient_accumulation_steps)
            logger.info(
                f"FlowTrace hard smoke train-step limit enabled: "
                f"{flowtrace_step_limit}; capped max_global_step={args.max_global_step}, max_iter={args.max_iter}"
            )
        args.global_iters_per_epoch = max(1, args.max_global_step // max(1, args.num_train_epochs))
        args.save_steps = args.max_global_step if flowtrace_step_limit > 0 else args.global_iters_per_epoch
        # args.save_steps = 10

        args, vl_transformer, optimizer, scheduler = mixed_precision_init(args, vl_transformer)
        args.resume_repro_meta = maybe_load_repro_optimizer(args, optimizer)
        maybe_fast_forward_repro_scheduler(args, scheduler)
        train(args, train_dataloader, val_dataloader, vl_transformer, tokenizer, training_saver, optimizer, scheduler)
        _finalize_flowtrace_smoke_evidence(args)

    elif args.do_eval:
        val_dataloader = make_data_loader(args, args.val_yaml, tokenizer, args.distributed, is_train=False)
        args, vl_transformer, _, _ = mixed_precision_init(args, vl_transformer)
        if args.do_signal_eval:
            signal_evaluate(args, val_dataloader, vl_transformer, tokenizer, args.eval_model_dir)
        else:
            evaluate_file = evaluate(args, val_dataloader, vl_transformer, tokenizer, args.eval_model_dir)

    if args.distributed:
        dist.destroy_process_group()

if __name__ == "__main__":
    shared_configs.shared_video_captioning_config(cbs=True, scst=True)
    args = get_custom_args(shared_configs)
    main(args)
