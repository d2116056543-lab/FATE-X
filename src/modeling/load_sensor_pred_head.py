import torch
from src.utils.logger import LOGGER as logger
from torch import nn
from src.layers.bert.modeling_bert import BertEncoder
from src.layers.bert import BertConfig, BertEncoder

def get_sensor_pred_model(args):
    return Sensor_Pred_Head(args)


class Sensor_Pred_Head(torch.nn.Module):
    """ This is the Control Signal Prediction head that performs sensor regression """
    def __init__(self, args):
        """ Initializes the prediction head.
        A simple transformer that performs sensor regression.
        We simply use a transformer to regress the whole signals of a video, which is superficial and could be optimized to a large extent.
        """
        super(Sensor_Pred_Head, self).__init__()

        self.img_feature_dim = int(args.img_feature_dim)
        self.use_grid_feat = args.grid_feat

        # Motion Transformer implemented by bert
        self.config = BertConfig.from_pretrained(args.config_name if args.config_name else \
            args.model_name_or_path, num_labels=2, finetuning_task='image_captioning')
        if self.config is None:
            # FlowCal V2 fallback config for local contract tests or missing BERT config paths.
            self.config = BertConfig()
            self.config.num_hidden_layers = min(int(getattr(self.config, 'num_hidden_layers', 2)), 2)
        self.encoder = BertEncoder(self.config)

        # type number of control signals to be used
        # TODO: Set this variable as an argument, corresponging to the control signal in dataloader

        self.sensor_dim = len(args.signal_types)
        self.sensor_embedding = torch.nn.Linear(self.sensor_dim, self.config.hidden_size)
        self.sensor_dropout = nn.Dropout(self.config.hidden_dropout_prob)

        # a mlp to transform the dimension of video feature
        self.img_dim = self.img_feature_dim
        self.img_embedding = nn.Linear(self.img_dim, self.config.hidden_size, bias=True)
        self.img_dropout = nn.Dropout(self.config.hidden_dropout_prob)

        # a sample regression decoder
        self.decoder = nn.Linear(self.config.hidden_size, self.sensor_dim)


    def encode(self, img_feats, attention_mask=None):
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

    def get_attn_mask(self, img_embedding_output):
        """Get attention mask that should be passed to motion transformer."""
        device = img_embedding_output.device
        bsz = img_embedding_output.shape[0]
        img_len = img_embedding_output.shape[1]


        attention_mask = torch.ones((bsz, img_len, img_len), dtype=torch.long)


        if attention_mask.dim() == 2:
            extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        elif attention_mask.dim() == 3:
            extended_attention_mask = attention_mask.unsqueeze(1)
        else:
            raise NotImplementedError

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype) # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        return extended_attention_mask.to(device)

    def get_l2_loss(self, pred, targ):
        loss_func = nn.MSELoss()
        return loss_func(pred, targ)
