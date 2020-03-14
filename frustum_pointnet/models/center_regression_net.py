import torch
import torch.nn as nn

from models.point_dan.point_dan import grad_reverse, adapt_layer_off, CALayer
from models.utils import create_mlp_components

__all__ = ['CenterRegressionNet', 'CenterRegressionPointDan', 'CenterRegressionSimpleDanNet']


class CenterRegressionNet(nn.Module):
    blocks = (128, 128, 256)

    def __init__(self, num_classes=3, width_multiplier=1):
        super().__init__()
        self.in_channels = 3
        self.num_classes = num_classes

        layers, channels = create_mlp_components(in_channels=self.in_channels, out_channels=self.blocks,
                                                 classifier=False, dim=2, width_multiplier=width_multiplier)
        self.features = nn.Sequential(*layers)

        layers, _ = create_mlp_components(in_channels=(channels + num_classes), out_channels=[256, 128, 3],
                                          classifier=True, dim=1, width_multiplier=width_multiplier)
        self.regression = nn.Sequential(*layers)

    def forward(self, inputs):
        coords = inputs['coords']
        one_hot_vectors = inputs['one_hot_vectors']
        assert one_hot_vectors.dim() == 2  # [B, C]

        features = self.features(coords)
        features = features.max(dim=-1, keepdim=False).values
        return self.regression(torch.cat([features, one_hot_vectors], dim=1))


class CenterRegressionPointDanGenerator(nn.Module):
    blocks1 = (64, 64)
    blocks2 = (128, 256)

    def __init__(self, num_classes, width_multiplier):
        super().__init__()
        self.in_channels = 3
        self.num_classes = num_classes

        layers, channels = create_mlp_components(in_channels=self.in_channels, out_channels=self.blocks1,
                                                 classifier=False, dim=2, width_multiplier=width_multiplier)
        self.pre = nn.Sequential(*layers)

        self.node = adapt_layer_off()

        layers, channels = create_mlp_components(in_channels=2*channels, out_channels=self.blocks2,
                                                 classifier=False, dim=2, width_multiplier=width_multiplier)
        self.features = nn.Sequential(*layers)
        self.channels = channels

    def forward(self, x, node=False):
        x_loc = x.squeeze(-1)
        x = self.pre(x)
        
        x = x.unsqueeze(3)
        x, node_feat, node_off = self.node(x, x_loc)
        
        x = x.squeeze(-1)
        x = self.features(x)
        x = x.max(dim=-1, keepdim=False).values

        if node:
            return x, node_feat, node_off
        else:
            return x, node_feat


class CenterRegressionPointDan(nn.Module):

    def __init__(self, num_classes=3, width_multiplier=1):
        super().__init__()
        self.g = CenterRegressionPointDanGenerator(num_classes, width_multiplier)
        channels = self.g.channels

        self.attention_s = CALayer(64 * 64)
        self.attention_t = CALayer(64 * 64)

        layers, _ = create_mlp_components(in_channels=(channels + num_classes), out_channels=[256, 128, 3],
                                          classifier=True, dim=1, width_multiplier=width_multiplier)
        self.r1 = nn.Sequential(*layers)

        layers, _ = create_mlp_components(in_channels=(channels + num_classes), out_channels=[256, 128, 3],
                                          classifier=True, dim=1, width_multiplier=width_multiplier)
        self.r2 = nn.Sequential(*layers)

    def forward(self, inputs, constant=1, adaptation=False,
                adaptation_s=False, adaptation_t=False):
        coords = inputs['coords']
        one_hot_vectors = inputs['one_hot_vectors']
        assert one_hot_vectors.dim() == 2  # [B, C]

        features, feat_ori, node_idx = self.g(coords, node=True)

        batch_size = feat_ori.size(0)
        node_features = None

        if adaptation_s:
            # source domain sa node feat
            feat_node = feat_ori.view(batch_size, -1)
            feat_node_s = self.attention_s(feat_node.unsqueeze(2).unsqueeze(3))
            node_features = feat_node_s

        elif adaptation_t:
            # target domain sa node feat
            feat_node = feat_ori.view(batch_size, -1)
            feat_node_t = self.attention_t(feat_node.unsqueeze(2).unsqueeze(3))
            node_features =  feat_node_t

        if adaptation:
            features = grad_reverse(features, constant)

        reg_input = torch.cat([features, one_hot_vectors], dim=1)

        y1 = self.r1(reg_input)
        y2 = self.r2(reg_input)

        if adaptation_s or adaptation_t:
            return y1, y2, node_features
        else:
            return y1, y2


class CenterRegressionSimpleDanNet(nn.Module):
    blocks = (128, 128, 256)

    def __init__(self, num_classes=3, width_multiplier=1):
        super().__init__()
        self.in_channels = 3
        self.num_classes = num_classes

        layers, channels = create_mlp_components(in_channels=self.in_channels, out_channels=self.blocks,
                                                 classifier=False, dim=2, width_multiplier=width_multiplier)
        self.g = nn.Sequential(*layers)

        layers, _ = create_mlp_components(in_channels=(channels + num_classes), out_channels=[256, 128, 3],
                                          classifier=True, dim=1, width_multiplier=width_multiplier)
        self.r1 = nn.Sequential(*layers)

        layers, _ = create_mlp_components(in_channels=(channels + num_classes), out_channels=[256, 128, 3],
                                          classifier=True, dim=1, width_multiplier=width_multiplier)
        self.r2 = nn.Sequential(*layers)

    def forward(self, inputs, adaptation=False, constant=1.0):
        coords = inputs['coords']
        one_hot_vectors = inputs['one_hot_vectors']
        assert one_hot_vectors.dim() == 2  # [B, C]

        features = self.features(coords)
        features = features.max(dim=-1, keepdim=False).values

        if adaptation:
            features = grad_reverse(features, constant)

        reg_input = torch.cat([features, one_hot_vectors], dim=1)

        y1 = self.r1(reg_input)
        y2 = self.r2(reg_input)

        return y1, y2
