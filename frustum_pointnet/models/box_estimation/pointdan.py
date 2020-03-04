import torch
import torch.nn as nn

from models.point_dan.point_dan import transform_net, grad_reverse
from models.utils import create_pointnet_components, create_mlp_components

__all__ = ['BoxEstimationPointDan']


class PointnetG(nn.Module):
    def __init__(self, num_classes, width_multiplier=1, voxel_resolution_multiplier=1):
        super(PointnetG, self).__init__()
        self.trans_net1 = transform_net(3, 3)

        blocks = ((128, 2, None), (256, 1, None), (512, 1, None))
        self.in_channels = 3
        self.num_classes = num_classes

        layers, channels_point, _ = create_pointnet_components(
            blocks=blocks, in_channels=self.in_channels, with_se=False, normalize=True, eps=1e-15,
            width_multiplier=width_multiplier, voxel_resolution_multiplier=voxel_resolution_multiplier
        )
        self.channels_point = channels_point
        self.features = nn.Sequential(*layers)

    def forward(self, x, node=False):

        transform = self.trans_net1(x)
        x = x.transpose(2, 1)
        x = x.squeeze(-1)
        x = torch.bmm(x, transform)

        x = x.transpose(2, 1)
        x = self.features(x)

        return x


class BoxEstimationPointDan(nn.Module):

    def __init__(self, num_classes=3, extra_feature_channels=1,
                 num_heading_angle_bins=12, num_size_templates=8,
                 width_multiplier=1):
        super(BoxEstimationPointDan, self).__init__()

        self.num_classes = num_classes

        self.g = PointnetG(num_classes)

        channels_point = self.g.channels_point

        layers, _ = create_mlp_components(
            in_channels=channels_point + num_classes,
            out_channels=[512, 256, (3 + num_heading_angle_bins * 2 + num_size_templates * 4)],
            classifier=True, dim=1, width_multiplier=width_multiplier
        )
        self.c1 = nn.Sequential(*layers)

        layers, _ = create_mlp_components(
            in_channels=channels_point + num_classes,
            out_channels=[512, 256, (3 + num_heading_angle_bins * 2 + num_size_templates * 4)],
            classifier=True, dim=1, width_multiplier=width_multiplier
        )
        self.c2 = nn.Sequential(*layers)

    def forward(self, inputs, constant=1, adaptation=False):

        coords = inputs['coords']
        one_hot_vectors = inputs['one_hot_vectors']
        assert one_hot_vectors.dim() == 2  # [B, C]

        coords = coords.unsqueeze(-1)
        point_feat = self.g(coords, node=True)

        if adaptation:
            point_feat = grad_reverse(point_feat, constant)

        features = point_feat.max(dim=-1, keepdim=False).values
        cls_input = torch.cat([features, one_hot_vectors], dim=1)

        y1 = self.c1(cls_input)
        y2 = self.c2(cls_input)

        return y1, y2
