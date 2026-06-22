from pathlib import Path
import subprocess

ROOT = Path(r"E:\sbw\FATE_Drive\fate_x_flowtrace_pmt_v1_worktree")

sensor = subprocess.check_output(["git", "show", "HEAD:src/modeling/load_sensor_pred_head.py"], cwd=ROOT).decode("utf-8")
needle = """        self.config = BertConfig.from_pretrained(args.config_name if args.config_name else \\
            args.model_name_or_path, num_labels=2, finetuning_task='image_captioning')"""
sensor = sensor.replace(
    needle,
    needle
    + """
        if self.config is None:
            # FlowCal V2 fallback config for local contract tests or missing BERT config paths.
            self.config = BertConfig()
            self.config.num_hidden_layers = min(int(getattr(self.config, 'num_hidden_layers', 2)), 2)""",
)
start = sensor.index("    def forward(self, *args, **kwargs):")
end = sensor.index("    def get_attn_mask", start)
replacement = '''    def encode(self, img_feats, attention_mask=None):
        """Encode video features without using control targets."""
        img_embedding_output = self.img_embedding(img_feats)
        img_embedding_output = self.img_dropout(img_embedding_output)
        extended_attention_mask = self.get_attn_mask(img_embedding_output) if attention_mask is None else attention_mask
        encoder_outputs = self.encoder(img_embedding_output, extended_attention_mask)
        return encoder_outputs[0]

    def predict(self, img_feats, frame_num=None):
        """Predict control signals from visual features only."""
        hidden = self.encode(img_feats)
        if frame_num is None:
            frame_num = hidden.shape[1]
        sequence_output = hidden[:, :int(frame_num), :]
        return self.decoder(sequence_output)

    def forward(self, *args, **kwargs):
        """Backward-compatible ADAPT control prediction."""
        vid_feats = kwargs['img_feats']
        car_info = kwargs.get('car_info')
        return_hidden = kwargs.get('return_hidden', False)

        if car_info is None:
            pred_tensor = self.predict(vid_feats)
            if return_hidden:
                return pred_tensor, self.encode(vid_feats)
            return pred_tensor

        car_info = car_info.permute(0, 2, 1)
        B, S, C = car_info.shape
        assert C == self.sensor_dim, f"{C}, {self.sensor_dim}"
        sequence_output = self.encode(vid_feats)[:, :S, :]
        pred_tensor = self.decoder(sequence_output)
        loss = self.get_l2_loss(pred_tensor, car_info)
        if return_hidden:
            return loss, pred_tensor, sequence_output
        return loss, pred_tensor

'''
sensor = sensor[:start] + replacement + sensor[end:]
(ROOT / "src/modeling/load_sensor_pred_head.py").open("w", encoding="utf-8", newline="\n").write(sensor)

bert = subprocess.check_output(["git", "show", "HEAD:src/layers/bert/modeling_bert.py"], cwd=ROOT).decode("utf-8")
append = '''

class FlowCalV2TypedLMHook(nn.Module):
    """Opt-in typed residual hook used by ACPR FlowCal V2 before an LM head."""

    def __init__(self, hidden_size, num_types=8):
        super().__init__()
        self.type_embedding = nn.Embedding(num_types, hidden_size)
        self.gate = nn.Linear(hidden_size * 2, hidden_size)

    def forward(self, hidden_states, type_ids):
        type_vec = self.type_embedding(type_ids.clamp_min(0))
        if type_vec.ndim == 2:
            type_vec = type_vec.unsqueeze(1).expand_as(hidden_states)
        gate = torch.sigmoid(self.gate(torch.cat([hidden_states, type_vec], dim=-1)))
        return hidden_states + gate * type_vec


def flowcal_v2_generation_logprobs(prediction_scores, token_ids):
    """Gather generation log-probabilities for SCST-style V2 training."""
    log_probs = F.log_softmax(prediction_scores, dim=-1)
    return log_probs.gather(-1, token_ids.unsqueeze(-1)).squeeze(-1)
'''
(ROOT / "src/layers/bert/modeling_bert.py").open("w", encoding="utf-8", newline="\n").write(bert.rstrip() + append)
print("restored minimal diffs")
