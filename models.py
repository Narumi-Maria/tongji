from typing_extensions import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()

        self.rnn = torch.nn.RNN(15, 15, batch_first=True)

        self.layers = [ 
            ("fc1", nn.Linear(1500, 512)),
            ("dr1", nn.Dropout()),
            ("bn1", nn.BatchNorm1d(512)),
            ("act1", nn.ReLU()),
            ("fc2", nn.Linear(512, 128)),
            # ("dr2", nn.Dropout(p=0.2)),
            ("bn2", nn.BatchNorm1d(128)),
            ("act2", nn.ReLU()),
            ("fc3", nn.Linear(128, 20)),
        ]
        self.layers = nn.Sequential(OrderedDict(self.layers))

        # self.conv = nn.Conv1d(in_channels = 15, out_channels = 15, kernel_size = 11)
    
    def forward(self, feature):
        
        # # feature: (bz, 100, 15)
        # out = self.layers(feature.flatten(1, 2))

        feature, _ = self.rnn(feature)
        assert feature.shape[-1] == 15 and feature.shape[-2] == 100
        # print(feature.shape)
        out = self.layers(feature.flatten(1, 2))


        return out


class MLP_per_frame(nn.Module):
    def __init__(self):
        super(MLP_per_frame, self).__init__()

        self.layers = [
            ("fc1", nn.Linear(15, 32)),
            ("act1", nn.ReLU()),
            ("fc2", nn.Linear(32, 20)),
        ]
        self.layers = nn.Sequential(OrderedDict(self.layers))
    
    def forward(self, feature):
        
        # feature: (bz, 100, 15)
        out = self.layers(feature)

        return out


class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()

        self.rnn = torch.nn.RNN(15, 20, batch_first=True)
    
    def forward(self, feature):
        
        # feature: (bz, 100, 15)
        out, hn = self.rnn(feature)

        return hn.squeeze()


class Conv(nn.Module):
    def __init__(self):
        super(Conv, self).__init__()

        self.layers = [
            ("fc1", nn.Linear(64, 64)),
            ("act1", nn.ReLU()),
            ("fc2", nn.Linear(64, 20)),
        ]
        self.layers = nn.Sequential(OrderedDict(self.layers))

        self.conv1 = nn.Conv1d(in_channels = 15, out_channels = 15, kernel_size = 11)
        self.pool1 = torch.nn.MaxPool1d(3, stride=2)
        self.conv2 = nn.Conv1d(in_channels = 64, out_channels = 64, kernel_size = 4)
    
    def forward(self, feature):
        
        # feature: (bz, 100, 15)
        feature = feature.transpose(1, 2) # (bz, 15, 100)
        out = self.conv1(feature) # (bz, 15, 90)
        out = self.pool1(out) # (bz, 64, 5)
        out = self.conv2(out) # (bz, 64, 2)
        out = torch.mean(out, dim=-1) # (bz, 64)
        out = self.layers(out) # (bz, 20)

        return out


# (X)
class transformer(nn.Module):
    def __init__(self):
        super(transformer, self).__init__()

        width = 15
        layers = 2
        heads = 1

        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width)) # (100)
        self.positional_embedding = nn.Parameter(scale * torch.randn((16, width)))
        self.ln_pre = LayerNorm(width)
        self.transformer = Transformer(width, layers, heads)
        self.ln_post = LayerNorm(width)

        self.layers = [
            ("fc1", nn.Linear(15, 20)),
            # ("act1", nn.ReLU()),
            # ("fc2", nn.Linear(64, 20)),
        ]
        self.layers = nn.Sequential(OrderedDict(self.layers))

    def forward(self, feature):
        
        # (bz, 100, 15)
        # feature = feature.transpose(1, 2) # (bz, 15, 100)
        x = torch.cat(
            [self.class_embedding.unsqueeze(0).unsqueeze(0).expand(feature.shape[0], 1, -1),
             feature], dim=1)  # (bz, 101, 15)
        x = x + self.positional_embedding
        x = self.ln_pre(x)

        x = x.transpose(0, 1) # (16, bz, 100)
        x = self.transformer(x)
        x = self.ln_post(x[0, :, :]) # (bz, 100)
        
        out = self.layers(x)

        return out

class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)

class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)

class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])
        self.init()

    def init(self):
        proj_std = (self.width ** -0.5) * ((2 * self.layers) ** -0.5)
        attn_std = self.width ** -0.5
        fc_std = (2 * self.width) ** -0.5
        for block in self.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)
