import torch
import torch.nn as nn
from einops import rearrange
from motion_inbetween_space.model import transformer
class STTransformer(nn.Module):
    def __init__(self, config):
        super(STTransformer, self).__init__()
        self.config = config

        self.d_mask = config["d_mask"]
        self.constrained_slices = [
            slice(*i) for i in config["constrained_slices"]
        ]
        self.add_kf = config["add_kf"]
        self.dropout = config["dropout"]
        self.pre_lnorm = config["pre_lnorm"]
        self.n_layer = config["n_layer"]
        self.n_splayer = config["n_splayer"]
        # decoder - >temporal:24*28->512->280
        self.encoder = nn.Linear(self.config["d_encoder_in"], self.config["d_model"])
        self.decoder = nn.Sequential(
            nn.Linear(self.config["d_model"], self.config["d_decoder_h"]),
            nn.PReLU(),
            nn.Linear(self.config["d_decoder_h"], self.config["d_out"])
        )
        self.keyframe_pos_layer = nn.Sequential(
            nn.Linear(2, self.config["d_model"]),
            nn.PReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.config["d_model"], self.config["d_model"]),
            nn.Dropout(self.dropout)
        )
        self.layer_norm = nn.LayerNorm(self.config["d_model"])
        self.Spatial_norm=nn.LayerNorm(self.config["d_model_sp"])   # spatial layer norm

        self.att_layers = nn.ModuleList()
        self.pff_layers = nn.ModuleList()
        self.sp_att_layers = nn.ModuleList()
        self.sp_pff_layers = nn.ModuleList()
        for i in range(self.n_splayer):        
            self.sp_att_layers.append(
                transformer.SPMultiHeadedAttention(
                    self.config["n_head_sp"], self.config["d_model_sp"],
                    self.config["d_head_sp"], dropout=self.config["dropout"],
                    pre_lnorm=self.config["pre_lnorm"],
                    bias=self.config["atten_bias"]
                )
            )

            self.sp_pff_layers.append(
                transformer.PositionwiseFeedForward(
                    self.config["d_model_sp"], self.config["d_pff_inner_sp"],
                    dropout=self.config["dropout"],
                    pre_lnorm=self.config["pre_lnorm"]
                )
            )
            
        for i in range(self.n_layer):
            self.att_layers.append(
                transformer.AlibiMultiHeadedAttention(
                    self.config["n_head"], self.config["d_model"],
                    self.config["d_head"], dropout=self.config["dropout"],
                    pre_lnorm=self.config["pre_lnorm"],
                    bias=self.config["atten_bias"]
                )
            )

            self.pff_layers.append(
                transformer.PositionwiseFeedForward(
                    self.config["d_model"], self.config["d_pff_inner"],
                    dropout=self.config["dropout"],
                    pre_lnorm=self.config["pre_lnorm"]
                )
            )



    def Spatial_forward_features(self, x,mask=None):
        b, _, f, p = x.shape  ##### b is batch size, f is number of frames, p is number of joints
        x = rearrange(x, 'b c f p  -> (b f) p  c', )

        #x = self.encoder(x) # embedding
        # x += self.spatial_pos_embed
        # x = self.pos_drop(x)
        # rel_pos_emb = self.get_rel_pos_emb_sp(x.shape[-2], x.dtype, x.device)
        for i in range(self.n_splayer):
            x = self.sp_att_layers[i](x,mask=mask)
            x = self.sp_pff_layers[i](x)

        x = self.Spatial_norm(x)
        x = rearrange(x, '(b f) w c -> b f (w c)', f=f)
        return x

    def forward(self, x, keyframe_pos, mask=None,sp_mask=None):
        x = x.permute(0, 3, 1, 2)
        b, _, _, p = x.shape
        ### now x is [batch_size, 2 channels, receptive frames, joint_num], following image data
        x = self.Spatial_forward_features(x,sp_mask)    # (b,seq,280)
        x = self.encoder(x)
        if self.add_kf:
            x = x + self.keyframe_pos_layer(keyframe_pos)
        for i in range(self.n_layer):
            x = self.att_layers[i](x, mask=mask)
            x = self.pff_layers[i](x)

        x = self.layer_norm(x)
        x = self.decoder(x)

        return x

# add Alibi version
class ContextTransformer(nn.Module):
    def __init__(self, config):
        super(ContextTransformer, self).__init__()
        self.config = config

        self.d_mask = config["d_mask"]
        self.constrained_slices = [
            slice(*i) for i in config["constrained_slices"]
        ]
        self.add_kf = config["add_kf"]
        self.dropout = config["dropout"]
        self.pre_lnorm = config["pre_lnorm"]
        self.n_layer = config["n_layer"]
        
        self.encoder = nn.Sequential(
            nn.Linear(self.config["d_encoder_in"], self.config["d_encoder_h"]),
            nn.PReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.config["d_encoder_h"], self.config["d_model"]),
            nn.PReLU(),
            nn.Dropout(self.dropout)
        )

        self.decoder = nn.Sequential(
            nn.Linear(self.config["d_model"], self.config["d_decoder_h"]),
            nn.PReLU(),
            nn.Linear(self.config["d_decoder_h"], self.config["d_out"])
        )

        # self.rel_pos_layer = nn.Sequential(
        #     nn.Linear(1, self.config["d_head"]),
        #     nn.PReLU(),
        #     nn.Dropout(self.dropout),
        #     nn.Linear(self.config["d_head"], self.config["d_head"]),
        #     nn.Dropout(self.dropout)
        # )

        self.keyframe_pos_layer = nn.Sequential(
            nn.Linear(2, self.config["d_model"]),
            nn.PReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.config["d_model"], self.config["d_model"]),
            nn.Dropout(self.dropout)
        )

        self.layer_norm = nn.LayerNorm(self.config["d_model"])
        self.att_layers = nn.ModuleList()
        self.pff_layers = nn.ModuleList()

        for i in range(self.n_layer):
            self.att_layers.append(
                transformer.AlibiMultiHeadedAttention(
                    self.config["n_head"], self.config["d_model"],
                    self.config["d_head"], dropout=self.config["dropout"],
                    pre_lnorm=self.config["pre_lnorm"],
                    bias=self.config["atten_bias"]
                )
            )

            self.pff_layers.append(
                transformer.PositionwiseFeedForward(
                    self.config["d_model"], self.config["d_pff_inner"],
                    dropout=self.config["dropout"],
                    pre_lnorm=self.config["pre_lnorm"]
                )
            )

    # def get_rel_pos_emb(self, window_len:int, dtype:torch.dtype, device:torch.device):
    #     pos_idx = torch.arange(-window_len + 1, window_len,
    #                            dtype=dtype, device=device)
    #     pos_idx = pos_idx[None, :, None]        # (1, seq, 1)
    #     rel_pos_emb = self.rel_pos_layer(pos_idx)
    #     return rel_pos_emb

    def forward(self, x, keyframe_pos, mask=None):
        x = self.encoder(x)
        if self.add_kf:
            x = x + self.keyframe_pos_layer(keyframe_pos)

        # rel_pos_emb = self.get_rel_pos_emb(x.shape[-2], x.dtype, x.device)

        for i in range(self.n_layer):
            x = self.att_layers[i](x, mask=mask)
            x = self.pff_layers[i](x)
        #if self.pre_lnorm:
        #    x = self.layer_norm(x)
        x = self.layer_norm(x)
        x = self.decoder(x)

        return x


class DetailTransformer(nn.Module):
    def __init__(self, config):
        super(DetailTransformer, self).__init__()
        self.config = config

        self.d_mask = config["d_mask"]
        self.constrained_slices = [
            slice(*i) for i in config["constrained_slices"]
        ]

        self.dropout = config["dropout"]
        self.pre_lnorm = config["pre_lnorm"]
        self.n_layer = config["n_layer"]

        self.encoder = nn.Sequential(
            nn.Linear(self.config["d_encoder_in"], self.config["d_encoder_h"]),
            nn.PReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.config["d_encoder_h"], self.config["d_model"]),
            nn.PReLU(),
            nn.Dropout(self.dropout)
        )

        self.decoder = nn.Sequential(
            nn.Linear(self.config["d_model"], self.config["d_decoder_h"]),
            nn.PReLU(),
            nn.Linear(self.config["d_decoder_h"], self.config["d_out"])
        )

        self.rel_pos_layer = nn.Sequential(
            nn.Linear(1, self.config["d_head"]),
            nn.PReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.config["d_head"], self.config["d_head"]),
            nn.Dropout(self.dropout)
        )

        self.layer_norm = nn.LayerNorm(self.config["d_model"])
        self.att_layers = nn.ModuleList()
        self.pff_layers = nn.ModuleList()

        for i in range(self.n_layer):
            self.att_layers.append(
                transformer.RelMultiHeadedAttention(
                    self.config["n_head"], self.config["d_model"],
                    self.config["d_head"], dropout=self.config["dropout"],
                    pre_lnorm=self.config["pre_lnorm"],
                    bias=self.config["atten_bias"]
                )
            )

            self.pff_layers.append(
                transformer.PositionwiseFeedForward(
                    self.config["d_model"], self.config["d_pff_inner"],
                    dropout=self.config["dropout"],
                    pre_lnorm=self.config["pre_lnorm"]
                )
            )

    def get_rel_pos_emb(self, window_len, dtype, device):
        pos_idx = torch.arange(-window_len + 1, window_len,
                               dtype=dtype, device=device)
        pos_idx = pos_idx[None, :, None]        # (1, seq, 1)
        rel_pos_emb = self.rel_pos_layer(pos_idx)
        return rel_pos_emb

    def forward(self, x, mask=None):
        x = self.encoder(x)
        rel_pos_emb = self.get_rel_pos_emb(x.shape[-2], x.dtype, x.device)

        for i in range(self.n_layer):
            x = self.att_layers[i](x, rel_pos_emb, mask=mask)
            x = self.pff_layers[i](x)
        if self.pre_lnorm:
            x = self.layer_norm(x)
        x = self.decoder(x)

        return x


class Discriminator(nn.Module):
    def __init__(self, config):
        super(Discriminator, self).__init__()
        self.config = config

        self.layers = nn.Sequential(
            nn.Conv1d(config["d_in"], config["d_conv1"],
                      kernel_size=config["kernel_size"],
                      stride=1, padding=0),
            nn.ReLU(),
            nn.Conv1d(config["d_conv1"], config["d_conv2"],
                      kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv1d(config["d_conv2"], 1, kernel_size=1,
                      stride=1, padding=0),
        )

    def forward(self, data):
        data = data.transpose(-1, -2)     # (batch, dim, seq)
        return self.layers(data)          # (batch, 1, seq)
