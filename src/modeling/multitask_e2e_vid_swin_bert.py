import torch
from fairscale.nn.misc import checkpoint_wrapper
import random
from fate_x.models.video_token_reducer import VideoTokenReducer
from fate_x.models.temporal_evidence_memory import TemporalEvidenceMemory
from fate_x.models.flowtrace_pmt_model import FlowTracePMTModel
from fate_x.models.token_pmt_adapter import TokenPMTAdapter
from src.modeling.load_sensor_pred_head import get_sensor_pred_model

class MultitaskVideoTransformer(torch.nn.Module):
    """ This is the multi-task module that performs Driving Caption Generation and Control Signal Prediction. """
    def __init__(self, args, config, swin, transformer_encoder):
        """ Initializes the model.
        Parameters:
            args: basic args of ADAPT, mostly defined in `src/configs/VidSwinBert/BDDX_multi_default.json` and input args
            config: config of transformer_encoder, mostly defined in `models/captioning/bert-base-uncased/config.json`
            swin: torch module of the backbone to be used. See `src/modeling/load_swin.py`
            transformer_encoder: torch module of the transformer architecture. See `src/modeling/load_bert.py`
        """
        super(MultitaskVideoTransformer, self).__init__()
        self.config = config
        self.use_checkpoint = args.use_checkpoint and not args.freeze_backbone
        if self.use_checkpoint:
            self.swin = checkpoint_wrapper(swin, offload_to_cpu=True)
        else:
            self.swin = swin
        self.trans_encoder = transformer_encoder
        self.img_feature_dim = int(args.img_feature_dim)
        self.use_grid_feat = args.grid_feat
        self.latent_feat_size = self.swin.backbone.norm.normalized_shape[0]
        self.fc = torch.nn.Linear(self.latent_feat_size, self.img_feature_dim)
        self.fate_x_enabled = getattr(args, 'fate_x_enabled', False)
        self.video_token_reducer = getattr(args, 'video_token_reducer', 'none')
        self.temporal_evidence_memory = getattr(args, 'temporal_evidence_memory', 'none')
        self.fate_x_text_reduce_only = getattr(args, 'fate_x_text_reduce_only', True)
        self.fate_x_reduce_control = getattr(args, 'fate_x_reduce_control', False)
        self.fate_x_control_reducer_mode = getattr(args, 'fate_x_control_reducer', 'none')
        if self.fate_x_control_reducer_mode == 'temporal_ordered_topk':
            self.fate_x_control_reducer_mode = 'per_frame_topk_merge'
        self.fate_x_last_stats = {}
        self.fate_x_last_provenance = None
        if self.fate_x_enabled and self.video_token_reducer != 'none':
            self.fate_x_reducer = VideoTokenReducer(
                self.img_feature_dim,
                keep_ratio=getattr(args, 'fate_x_keep_ratio', 0.5),
                num_summary_tokens=getattr(args, 'fate_x_num_summary_tokens', 1),
                min_tokens=getattr(args, 'fate_x_min_tokens', 128),
                mode=self.video_token_reducer,
                temporal_tokens=getattr(args, 'fate_x_temporal_tokens', None) or None,
                spatial_tokens_per_frame=getattr(args, 'fate_x_spatial_tokens_per_frame', None) or None,
                summary_mode=getattr(args, 'fate_x_summary_mode', 'cluster'),
            )
        else:
            self.fate_x_reducer = None
        if self.fate_x_enabled and self.fate_x_reduce_control and self.fate_x_control_reducer_mode != 'none':
            self.fate_x_control_reducer = VideoTokenReducer(
                self.img_feature_dim,
                keep_ratio=getattr(args, 'fate_x_keep_ratio', 0.5),
                num_summary_tokens=getattr(args, 'fate_x_num_summary_tokens', 1),
                min_tokens=getattr(args, 'fate_x_min_tokens', 128),
                mode=self.fate_x_control_reducer_mode,
                temporal_tokens=getattr(args, 'fate_x_temporal_tokens', None) or None,
                spatial_tokens_per_frame=getattr(args, 'fate_x_spatial_tokens_per_frame', None) or None,
                summary_mode=getattr(args, 'fate_x_summary_mode', 'cluster'),
            )
        else:
            self.fate_x_control_reducer = None
        if self.fate_x_enabled and self.temporal_evidence_memory == 'queries':
            self.fate_x_memory = TemporalEvidenceMemory(self.img_feature_dim)
        else:
            self.fate_x_memory = None
        self.flowtrace_enabled = bool(getattr(args, 'flowtrace_enabled', False))
        self.flowtrace_encoder = None
        self.token_pmt_adapter = None
        if self.flowtrace_enabled:
            state_dim = int(getattr(args, 'flowtrace_state_dim', 256))
            self.flowtrace_encoder = None  # lazy-init after first multiscale forward reveals dims
            self.token_pmt_adapter = TokenPMTAdapter(
                hidden_dim=int(getattr(config, 'hidden_size', self.img_feature_dim)),
                state_dim=state_dim,
                rank=int(getattr(args, 'flowtrace_pmt_rank', 32)),
            )
            self.flowtrace_state_dim = state_dim
        self.compute_mask_on_the_fly = False # deprecated
        self.mask_prob = args.mask_prob
        self.mask_token_id = -1
        self.max_img_seq_length = args.max_img_seq_length

        # get Control Signal Prediction Head
        self.sensor_pred_head = get_sensor_pred_model(args)

        # if only_signal is True, it means we 
        # remove Driving Caption Generation head and only use Control Signal Prediction head 
        self.only_signal = getattr(args, 'only_signal', False)

        # sparse attention mask defined in SwinBert
        self.learn_mask_enabled = getattr(args, 'learn_mask_enabled', False)
        self.sparse_mask_soft2hard = getattr(args, 'sparse_mask_soft2hard', False)
        if self.learn_mask_enabled==True:
            self.learn_vid_att = torch.nn.Embedding(args.max_img_seq_length*args.max_img_seq_length,1)
            self.sigmoid = torch.nn.Sigmoid()

    def forward(self, *args, **kwargs):
        """ The forward process of ADAPT, 
        Parameters:
            input_ids: word tokens of input sentences tokenized by tokenizer
            attention_mask: multimodal attention mask in Vision-Language transformer
            token_type_ids: typen tokens of input sentences, 
                            0 means it is a narration sentence and 1 means a reasoning sentence, same size with input_ids
            img_feats: preprocessed frames of the video
            masked_pos: [MASK] position when performing MLM, used to locate the masked words
            masked_ids: groung truth of [MASK] when performing MLM
            car_info: control signals of ego car in the video
        """

        # grad cam can only input a tuple (args, kwargs)
        if isinstance(args, tuple) and len(args) != 0:
            kwargs = args[0]
            args= ()

        # video swin to extract video features
        images = kwargs['img_feats']
        B, S, C, H, W = images.shape  # batch, segment, chanel, hight, width
        # (B x S x C x H x W) --> (B x C x S x H x W)
        images = images.permute(0, 2, 1, 3, 4)
        backbone_out = self.swin(images, return_stages=self.flowtrace_enabled)
        flowtrace_bundle = None
        if self.flowtrace_enabled and isinstance(backbone_out, dict):
            vid_feats = backbone_out.get('final_tokens', backbone_out.get('final'))
            stages = backbone_out.get('stages', [])
        else:
            vid_feats = backbone_out
            stages = []

        # tokenize video features to video tokens
        if self.use_grid_feat==True:
            vid_feats = vid_feats.permute(0, 2, 3, 4, 1)
        vid_feats = vid_feats.view(B, -1, self.latent_feat_size)

        # use an mlp to transform video token dimension
        vid_feats_dense = self.fc(vid_feats)
        vid_feats_text = vid_feats_dense
        vid_feats_control = vid_feats_dense
        if self.flowtrace_enabled and len(stages) >= 2:
            fine_stage, coarse_stage = stages[-2], stages[-1]
            fine_dim = int(fine_stage.shape[1])
            coarse_dim = int(coarse_stage.shape[1])
            if self.flowtrace_encoder is None:
                self.flowtrace_encoder = FlowTracePMTModel(
                    fine_dim=fine_dim,
                    coarse_dim=coarse_dim,
                    dense_dim=self.img_feature_dim,
                    state_dim=self.flowtrace_state_dim,
                ).to(vid_feats_dense.device)
            flowtrace_bundle = self.flowtrace_encoder(vid_feats_dense, fine_stage, coarse_stage)

        text_kwargs = dict(kwargs)
        control_kwargs = dict(kwargs)

        # Optional FATE-X token reducer/event memory. Default-off preserves ADAPT.
        # Caption/text sees reduced/event evidence tokens; control/CSP keeps dense
        # tokens unless an explicitly temporal-order-preserving control reducer is enabled.
        if self.fate_x_enabled:
            vid_feats_text = self._apply_fate_x_tokens(vid_feats_text, text_kwargs)
            if self.fate_x_reduce_control and self.fate_x_control_reducer is not None:
                control_reduced = self.fate_x_control_reducer(vid_feats_control)
                vid_feats_control = control_reduced['tokens']
                self.fate_x_last_stats.update({
                    'control_reduced_tokens': int(vid_feats_control.shape[1]),
                    'control_branch_dense': False,
                })
            else:
                self.fate_x_last_stats.update({
                    'control_reduced_tokens': int(vid_feats_control.shape[1]),
                    'control_branch_dense': True,
                })
        else:
            self.fate_x_last_stats = {}

        self.fate_x_last_stats.update({
            'dense_visual_tokens': int(vid_feats_dense.shape[1]),
            'text_visual_tokens': int(vid_feats_text.shape[1]),
            'control_visual_tokens': int(vid_feats_control.shape[1]),
            'fate_x_text_reduce_only': bool(self.fate_x_text_reduce_only),
            'fate_x_reduce_control': bool(self.fate_x_reduce_control),
            'flowtrace_enabled': bool(self.flowtrace_enabled),
            'flowtrace_state_tokens': 0 if flowtrace_bundle is None else int(flowtrace_bundle.state_memory.shape[1]),
        })

        # prepare branch-specific transformer inputs
        text_kwargs['img_feats'] = vid_feats_text
        control_kwargs['img_feats'] = vid_feats_control
        if flowtrace_bundle is not None:
            text_kwargs['flowtrace_bundle'] = flowtrace_bundle
            text_kwargs['flowtrace_pmt_adapter'] = self.token_pmt_adapter
            text_kwargs['flowtrace_pmt_scale'] = getattr(self, 'flowtrace_pmt_scale', 1.0)

        # disable bert attention outputs to avoid some bugs
        if self.trans_encoder.bert.encoder.output_attentions:
            self.trans_encoder.bert.encoder.set_output_attentions(False)
        
        if self.only_signal:
            # only Control Signal Prediction head 
            sensor_outputs = self.sensor_pred_head(*args, **control_kwargs)
            return sensor_outputs
        
        else:
            # learn soft attention mask
            if self.learn_mask_enabled:
                text_kwargs['attention_mask'] = text_kwargs['attention_mask'].float()
                vid_att_len = self.max_img_seq_length
                learn_att = self.learn_vid_att.weight.reshape(vid_att_len,vid_att_len)
                learn_att = self.sigmoid(learn_att)
                diag_mask = torch.diag(torch.ones(vid_att_len)).cuda()
                video_attention = (1. - diag_mask)*learn_att
                learn_att = diag_mask + video_attention
                if self.sparse_mask_soft2hard:
                    learn_att = (learn_att>=0.5)*1.0
                    learn_att = learn_att.cuda()
                    learn_att.requires_grad = False
                text_kwargs['attention_mask'][:, -vid_att_len::, -vid_att_len::] = learn_att

            # Driving Caption Generation head, output is ()
            outputs = self.trans_encoder(*args, **text_kwargs)

            # Control Signal Prediction head, output is ()
            sensor_outputs = self.sensor_pred_head(*args, **control_kwargs)

            outputs = outputs + sensor_outputs

            # sparse attention mask loss
            if self.learn_mask_enabled:
                loss_sparsity = self.get_loss_sparsity(video_attention)  
                outputs = outputs + (loss_sparsity, )

            return outputs
    


    def _resize_fate_x_attention_mask(self, kwargs, new_vid_len):
        """Resize multimodal attention mask after FATE-X token compression.

        ADAPT's video tokens occupy the final max_img_seq_length positions. When
        token count changes we preserve the text-text block and create fully
        visible text/video and video/video blocks for the new visual tokens.
        """
        if 'attention_mask' not in kwargs:
            return
        attention_mask = kwargs['attention_mask']
        if attention_mask is None or attention_mask.dim() != 3:
            return
        old_total = attention_mask.shape[-1]
        old_vid_len = int(self.max_img_seq_length)
        text_len = max(old_total - old_vid_len, 0)
        new_total = text_len + int(new_vid_len)
        if new_total == old_total:
            return
        new_mask = attention_mask.new_ones((attention_mask.shape[0], new_total, new_total))
        if text_len > 0:
            new_mask[:, :text_len, :text_len] = attention_mask[:, :text_len, :text_len]
        kwargs['attention_mask'] = new_mask

    def _apply_fate_x_tokens(self, vid_feats, kwargs):
        provenance = None
        if self.fate_x_reducer is not None:
            reduced = self.fate_x_reducer(vid_feats)
            vid_feats = reduced['tokens']
            provenance = reduced['provenance']
            self.fate_x_last_stats = dict(reduced['stats'])
        if self.fate_x_memory is not None:
            mem = self.fate_x_memory(vid_feats)
            vid_feats = torch.cat([mem['event_tokens'], vid_feats], dim=1)
            self.fate_x_last_stats.update({'event_tokens': mem['event_tokens'].shape[1]})
        self.fate_x_last_provenance = provenance
        self._resize_fate_x_attention_mask(kwargs, vid_feats.shape[1])
        return vid_feats

    def get_loss_sparsity(self, video_attention):
        sparsity_loss = 0
        sparsity_loss += (torch.mean(torch.abs(video_attention)))
        return sparsity_loss

    def reload_attn_mask(self, pretrain_attn_mask): 
        import numpy
        pretrained_num_tokens = int(numpy.sqrt(pretrain_attn_mask.shape[0]))

        pretrained_learn_att = pretrain_attn_mask.reshape(
                                pretrained_num_tokens,pretrained_num_tokens)
        scale_factor = 1
        vid_att_len = self.max_img_seq_length
        learn_att = self.learn_vid_att.weight.reshape(vid_att_len,vid_att_len)
        with torch.no_grad():
            for i in range(int(scale_factor)):
                learn_att[pretrained_num_tokens*i:pretrained_num_tokens*(i+1), 
                            pretrained_num_tokens*i:pretrained_num_tokens*(i+1)] = pretrained_learn_att 

    def freeze_backbone(self, freeze=True):
        for _, p in self.swin.named_parameters():
            p.requires_grad =  not freeze

