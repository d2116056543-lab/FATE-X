from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .adapt_video_backbone import ADAPTVideoBackboneV2
from .adapt_motion_backbone import ADAPTMotionBackbone
from .axis_aware_control_adapter import AxisAwareReasonControlAdapter
from .axis_aware_flow_composer import AxisAwareFlowComposer
from .config import FlowCalV2Config
from .lane_flow_field import PredicateConditionedLaneFlowField
from .local_partial_transport import LocalPartialTransportV2
from .semantic_reason_memory import SemanticReasonMemoryBuilder
from .temporal_predicate_tracker import TransportedNamedPredicateTracker
from .temporal_seca import TemporalSECAV2
from .types import FlowCalV2Batch, FlowCalV2Bundle, FlowCalV2TrainOutput, GeneratedSequence, InterventionSpecV2
from fate_x.losses.acpr_flowcal_v2_losses import control_rmse_loss, masked_language_model_loss


class ACPRFlowCalV2Model(nn.Module):
    def __init__(self, config: Optional[FlowCalV2Config] = None, captioning_model: Optional[nn.Module] = None):
        super().__init__()
        self.config = config or FlowCalV2Config()
        self.captioning_model = captioning_model
        d = self.config.hidden_dim
        self.video = ADAPTVideoBackboneV2(
            output_dim=d,
            adapt_feature_dim=self.config.adapt_feature_dim,
            adapt_checkpoint=self.config.adapt_checkpoint,
            use_real_video_swin=self.config.use_real_video_swin,
            image_resolution=self.config.image_resolution,
        )
        self.transport = LocalPartialTransportV2(dim=d)
        self.predicates = TransportedNamedPredicateTracker(dim=d)
        self.lane_flow = PredicateConditionedLaneFlowField(dim=d)
        self.flow = AxisAwareFlowComposer(dim=d)
        self.memory = SemanticReasonMemoryBuilder(input_dim=d, hidden_dim=self.config.text_hidden_dim)
        self.motion = ADAPTMotionBackbone.from_adapt_checkpoint(
            self.config.adapt_checkpoint,
            input_dim=self.config.adapt_feature_dim,
            hidden_dim=self.config.text_hidden_dim,
        )
        self.control_adapter = AxisAwareReasonControlAdapter(hidden_dim=self.config.text_hidden_dim)
        # Fallback path for unit tests only. Formal V2 must use ADAPT's released
        # BertForImageCaptioning via ``captioning_model`` so that zero-gate V2
        # preserves the previous best checkpoint instead of reinitializing text.
        self.token_embed = nn.Embedding(self.config.text_vocab_size, self.config.text_hidden_dim)
        self.seca = TemporalSECAV2(hidden_dim=self.config.text_hidden_dim)
        self.lm_head = nn.Linear(self.config.text_hidden_dim, self.config.text_vocab_size)

    def build_visual_state(self, batch: FlowCalV2Batch) -> FlowCalV2Bundle:
        video = self.video(batch.frames)
        transport = self.transport(video.fused_grid, video.coarse_aligned)
        predicates = self.predicates(video.fused_grid, transport)
        lane_flow = self.lane_flow(predicates, video.fused_grid)
        flow_state = self.flow(predicates, lane_flow)
        return FlowCalV2Bundle(video=video, local_transport=transport, predicates=predicates, lane_flow=lane_flow, flow_state=flow_state, diagnostics={})

    def _scale_dataclass_tensors(self, obj: Any, field_names: Sequence[str], scale: float) -> Any:
        updates: Dict[str, Tensor] = {}
        for name in field_names:
            value = getattr(obj, name, None)
            if isinstance(value, torch.Tensor):
                updates[name] = value * scale
        return replace(obj, **updates) if updates else obj

    def apply_intervention_to_visual_state(
        self,
        bundle: FlowCalV2Bundle,
        intervention: Optional[InterventionSpecV2],
    ) -> FlowCalV2Bundle:
        if intervention is None:
            return bundle
        kind = intervention.kind
        keep = max(0.0, 1.0 - float(intervention.strength))
        diagnostics = dict(bundle.diagnostics)
        diagnostics["intervention_kind"] = kind
        diagnostics["intervention_keep"] = keep

        if bundle.predicates is not None and (kind in {"all_flow_off", "predicate_off"} or "predicate_off" in kind):
            predicates = self._scale_dataclass_tensors(
                bundle.predicates,
                ("attention", "tokens", "presence_logits", "presence_probs", "confidence", "descriptor"),
                keep,
            )
            bundle = replace(bundle, predicates=predicates)

        flow_like = kind in {"all_flow_off", "flow_off", "longitudinal_flow_off", "lateral_flow_off"} or "flow_off" in kind
        if flow_like and bundle.lane_flow is not None:
            lane_flow = self._scale_dataclass_tensors(
                bundle.lane_flow,
                ("occupancy", "relative_motion", "motion_coherence", "stopped_tendency", "queue_pressure", "temporal_tokens", "descriptor"),
                keep,
            )
            bundle = replace(bundle, lane_flow=lane_flow)
        if flow_like and bundle.flow_state is not None:
            flow_state = self._scale_dataclass_tensors(
                bundle.flow_state,
                (
                    "semantic_tokens",
                    "semantic_logits",
                    "semantic_probs",
                    "semantic_evidence",
                    "lane_tokens",
                    "axis_tokens",
                    "axis_logits",
                    "axis_probs",
                    "direction_tokens",
                    "direction_logits",
                    "direction_probs",
                    "flow_to_predicate_attention",
                ),
                keep,
            )
            bundle = replace(bundle, flow_state=flow_state)
        return replace(bundle, diagnostics=diagnostics)

    def apply_intervention_to_reason_state(
        self,
        bundle: FlowCalV2Bundle,
        intervention: Optional[InterventionSpecV2],
    ) -> FlowCalV2Bundle:
        if intervention is None or bundle.reason_memory is None:
            return bundle
        kind = intervention.kind
        keep = max(0.0, 1.0 - float(intervention.strength))
        if kind in {"memory_off", "all_memory_off"} or "memory_off" in kind:
            memory = replace(
                bundle.reason_memory,
                values=bundle.reason_memory.values * keep,
                confidence=bundle.reason_memory.confidence * keep,
                semantic_state=bundle.reason_memory.semantic_state * keep,
            )
            return replace(bundle, reason_memory=memory)
        return bundle

    def build_reason_state(self, bundle: FlowCalV2Bundle) -> FlowCalV2Bundle:
        bundle.reason_memory = self.memory(bundle.predicates, bundle.lane_flow, bundle.flow_state)
        density_per_sample = bundle.lane_flow.occupancy.mean(dim=(1, 2)).detach()
        density = density_per_sample.mean().detach()
        queue_per_sample = bundle.lane_flow.queue_pressure.mean(dim=(1, 2)).detach()
        motion_per_sample = bundle.lane_flow.relative_motion.norm(dim=-1).mean(dim=(1, 2)).detach()
        stopped_per_sample = bundle.lane_flow.stopped_tendency.mean(dim=(1, 2)).detach()
        coherence_per_sample = bundle.lane_flow.motion_coherence.mean(dim=(1, 2)).detach()
        transport_shift_per_sample = bundle.local_transport.common_shift.norm(dim=-1).mean(dim=1).detach()
        transport_dustbin_per_sample = bundle.local_transport.dustbin_prob.mean(dim=(1, 2, 3)).detach()
        bundle.diagnostics.update(
            {
                "traffic_density": density,
                "traffic_density_per_sample": density_per_sample,
                "traffic_queue": queue_per_sample.mean().detach(),
                "traffic_queue_per_sample": queue_per_sample,
                "traffic_motion": motion_per_sample.mean().detach(),
                "traffic_motion_per_sample": motion_per_sample,
                "traffic_stopped": stopped_per_sample.mean().detach(),
                "traffic_stopped_per_sample": stopped_per_sample,
                "traffic_coherence": coherence_per_sample.mean().detach(),
                "traffic_coherence_per_sample": coherence_per_sample,
                "traffic_transport_shift": transport_shift_per_sample.mean().detach(),
                "traffic_transport_shift_per_sample": transport_shift_per_sample,
                "traffic_transport_dustbin": transport_dustbin_per_sample.mean().detach(),
                "traffic_transport_dustbin_per_sample": transport_dustbin_per_sample,
                "transport_dustbin": transport_dustbin_per_sample.mean().detach(),
            }
        )
        return bundle

    @staticmethod
    def _attention_for_captioning(input_ids: Tensor, img_feats: Tensor, attention_mask: Optional[Tensor]) -> Tensor:
        total = int(input_ids.shape[1] + img_feats.shape[1])
        if attention_mask is None or attention_mask.shape[-1] != total:
            return torch.ones(input_ids.shape[0], total, total, dtype=torch.long, device=input_ids.device)
        return attention_mask.to(input_ids.device)

    @staticmethod
    def _extract_caption_loss_and_logits(outputs: Any) -> Tuple[Optional[Tensor], Tensor]:
        if isinstance(outputs, (tuple, list)):
            loss = outputs[0] if len(outputs) >= 1 and torch.is_tensor(outputs[0]) and outputs[0].ndim == 0 else None
            if len(outputs) >= 2 and torch.is_tensor(outputs[1]):
                return loss, outputs[1]
            if len(outputs) >= 1 and torch.is_tensor(outputs[0]):
                return None, outputs[0]
        if torch.is_tensor(outputs):
            return None, outputs
        raise TypeError(f"captioning_model returned unsupported output type: {type(outputs)!r}")

    def _caption_image_features(self, bundle: FlowCalV2Bundle) -> Tensor:
        return bundle.video.dense_tokens_projected

    def forward_text(self, batch: FlowCalV2Batch, bundle: FlowCalV2Bundle) -> Tuple[Tensor, Tensor]:
        if (
            self.captioning_model is not None
            and batch.input_ids is not None
            and batch.masked_pos is not None
            and batch.masked_ids is not None
            and batch.token_type_ids is not None
        ):
            device = batch.frames.device
            input_ids = batch.input_ids.to(device)
            img_feats = self._caption_image_features(bundle).to(device)
            attention_mask = self._attention_for_captioning(input_ids, img_feats, batch.attention_mask)
            common = dict(
                input_ids=input_ids,
                img_feats=img_feats,
                attention_mask=attention_mask,
                masked_pos=batch.masked_pos.to(device),
                masked_ids=batch.masked_ids.to(device),
                token_type_ids=batch.token_type_ids.to(device),
                is_training=True,
            )
            baseline_loss, baseline = self._extract_caption_loss_and_logits(self.captioning_model(**common))
            enhanced_loss, enhanced = self._extract_caption_loss_and_logits(
                self.captioning_model(
                    **common,
                    acpr_flow_bundle=bundle,
                    acpr_temporal_seca=self.seca,
                    acpr_text_len=input_ids.shape[-1],
                )
            )
            self._last_adapt_text_loss = enhanced_loss if enhanced_loss is not None else baseline_loss
            return baseline, enhanced
        self._last_adapt_text_loss = None
        if batch.input_ids is None:
            b = batch.frames.shape[0]
            input_ids = torch.zeros(b, 30, dtype=torch.long, device=batch.frames.device)
        else:
            input_ids = batch.input_ids.to(batch.frames.device)
        hidden = self.token_embed(input_ids.clamp_min(0).clamp_max(self.config.text_vocab_size - 1))
        enhanced, _ = self.seca(hidden, bundle.reason_memory, batch.token_type_ids, hidden.shape[1])
        return self.lm_head(hidden), self.lm_head(enhanced)

    def forward_control(self, batch: FlowCalV2Batch, bundle: FlowCalV2Bundle) -> Tuple[Tensor, Tensor, Tensor]:
        dense = bundle.video.dense_tokens_projected
        base, hidden = self.motion.predict(dense, steps=self.config.num_frames)
        adapted = self.control_adapter(base, hidden, bundle.reason_memory, None)
        return base, adapted.final_prediction, hidden

    def _control_target_btc(self, car_info: Tensor, steps: int, device: torch.device) -> Tensor:
        target = car_info.to(device).transpose(1, 2)
        if target.shape[1] != steps:
            target = F.interpolate(target.transpose(1, 2), size=steps, mode="linear", align_corners=False).transpose(1, 2)
        return target

    def forward(self, batch: FlowCalV2Batch, stage: str = "R", intervention: Optional[InterventionSpecV2] = None) -> FlowCalV2TrainOutput:
        bundle = self.build_visual_state(batch)
        bundle = self.apply_intervention_to_visual_state(bundle, intervention)
        bundle = self.build_reason_state(bundle)
        bundle = self.apply_intervention_to_reason_state(bundle, intervention)
        base_logits, enhanced_logits = self.forward_text(batch, bundle)
        base_control, final_control, control_hidden = self.forward_control(batch, bundle)
        device = batch.frames.device
        if batch.masked_ids is not None and batch.masked_pos is not None:
            adapt_text_loss = getattr(self, "_last_adapt_text_loss", None)
            if isinstance(adapt_text_loss, torch.Tensor):
                text_loss = adapt_text_loss
            else:
                text_loss = masked_language_model_loss(enhanced_logits, batch.masked_ids.to(device), batch.masked_pos.to(device))
        else:
            text_loss = enhanced_logits.mean() * 0.0
        if batch.car_info is not None:
            ctrl_target = self._control_target_btc(batch.car_info, final_control.shape[1], device)
            speed_loss = control_rmse_loss(final_control[..., 1:2], ctrl_target[..., 1:2])
            course_loss = control_rmse_loss(final_control[..., 0:1], ctrl_target[..., 0:1])
        else:
            speed_loss = final_control.mean() * 0.0
            course_loss = final_control.mean() * 0.0
        auxiliary = bundle.lane_flow.queue_pressure.mean() * 0.01
        total = text_loss + speed_loss + course_loss + auxiliary
        return FlowCalV2TrainOutput(
            action_text_loss=text_loss,
            explanation_text_loss=text_loss,
            speed_loss=speed_loss,
            course_loss=course_loss,
            auxiliary_loss=auxiliary,
            total_loss=total,
            baseline_masked_logits=base_logits,
            enhanced_masked_logits=enhanced_logits,
            control_base_prediction=base_control,
            control_final_prediction=final_control,
            control_hidden=control_hidden,
            loss_components={"text": text_loss, "speed": speed_loss, "course": course_loss, "auxiliary": auxiliary},
            gradient_diagnostics={},
            bundle=bundle,
        )

    def encode_text(self, input_ids: Tensor) -> Tensor:
        if self.captioning_model is not None:
            word_embeddings = getattr(getattr(getattr(self.captioning_model, "bert", None), "embeddings", None), "word_embeddings", None)
            if word_embeddings is not None:
                return word_embeddings(input_ids)
        return self.token_embed(input_ids)

    def decode_adapt_compatible(self, token_ids: Tensor, tokenizer: Any = None) -> List[str]:
        texts: List[str] = []
        for row in token_ids.detach().cpu().tolist():
            if tokenizer is not None:
                texts.append(tokenizer.decode(row, skip_special_tokens=True).strip())
            else:
                texts.append(" ".join(str(int(x)) for x in row if int(x) > 0).strip())
        return texts

    def _generate_with_adapt_captioner(self, batch: FlowCalV2Batch, max_seq_length: int, tokenizer: Any = None) -> Tuple[Tensor, Tensor]:
        if self.captioning_model is None:
            raise RuntimeError("formal V2 ADAPT-compatible generation requires captioning_model")
        if batch.input_ids is None or batch.masked_pos is None or batch.token_type_ids is None:
            raise RuntimeError("formal V2 generation requires ADAPT input_ids/masked_pos/token_type_ids")
        bundle = self.build_visual_state(batch)
        bundle = self.build_reason_state(bundle)
        device = batch.frames.device
        input_ids = batch.input_ids.to(device)
        img_feats = self._caption_image_features(bundle).to(device)
        attention_mask = self._attention_for_captioning(input_ids, img_feats, batch.attention_mask)
        cls_token_id = int(getattr(tokenizer, "cls_token_id", 101))
        pad_token_id = int(getattr(tokenizer, "pad_token_id", 0))
        sep_token_id = int(getattr(tokenizer, "sep_token_id", 102))
        mask_token_id = int(getattr(tokenizer, "mask_token_id", 103))
        return self.captioning_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=batch.token_type_ids.to(device),
            img_feats=img_feats,
            masked_pos=batch.masked_pos.to(device),
            car_info=batch.car_info.to(device) if batch.car_info is not None else None,
            is_decode=True,
            do_sample=False,
            bos_token_id=cls_token_id,
            pad_token_id=pad_token_id,
            eos_token_ids=[sep_token_id],
            mask_token_id=mask_token_id,
            add_od_labels=False,
            od_labels_start_posid=15,
            max_length=max_seq_length,
            use_sep_cap=True,
            num_beams=1,
            temperature=1.0,
            top_k=0,
            top_p=1.0,
            repetition_penalty=1.0,
            length_penalty=1.0,
            num_return_sequences=1,
            num_keep_best=1,
            acpr_flow_bundle=bundle,
            acpr_temporal_seca=self.seca,
            acpr_text_len=max_seq_length,
        )

    def generate_adapt_caption_pairs(self, batch: FlowCalV2Batch, max_length: int = 15, tokenizer: Any = None) -> List[Dict[str, Any]]:
        if self.captioning_model is not None:
            total_length = max_length * 2
            ids, confs = self._generate_with_adapt_captioner(batch, total_length, tokenizer=tokenizer)
            ids = ids[:, 0, :].detach()
            confs = torch.exp(confs[:, 0]).detach() if torch.is_tensor(confs) and confs.ndim >= 2 else torch.ones(ids.shape[0], device=ids.device)
            descriptions = self.decode_adapt_compatible(ids[:, :max_length], tokenizer=tokenizer)
            explanations = self.decode_adapt_compatible(ids[:, max_length:total_length], tokenizer=tokenizer)
        else:
            out = self.forward(batch)
            logits = out.enhanced_masked_logits[:, : max_length * 2]
            ids = logits.argmax(dim=-1)
            descriptions = self.decode_adapt_compatible(ids[:, :max_length], tokenizer=tokenizer)
            explanations = self.decode_adapt_compatible(ids[:, max_length : max_length * 2], tokenizer=tokenizer)
            confs = torch.ones(ids.shape[0], device=ids.device)
        rows: List[Dict[str, Any]] = []
        sample_ids = batch.sample_ids or [str(i) for i in range(ids.shape[0])]
        for idx, (sample_id, des, exp) in enumerate(zip(sample_ids, descriptions, explanations)):
            conf = float(confs[idx].detach().cpu()) if torch.is_tensor(confs) else 1.0
            rows.append(
                {
                    "img_key": sample_id,
                    "description": des,
                    "explanation": exp,
                    "description_conf": conf,
                    "explanation_conf": conf,
                }
            )
        return rows

    def generate_explanation_with_logprobs(self, batch: FlowCalV2Batch, max_length: int = 15) -> GeneratedSequence:
        out = self.forward(batch)
        logits = out.enhanced_masked_logits[:, :max_length]
        dist = torch.distributions.Categorical(logits=logits)
        ids = dist.sample()
        logp = dist.log_prob(ids)
        return GeneratedSequence(token_ids=ids, logprobs=logp, texts=self.decode_adapt_compatible(ids))
