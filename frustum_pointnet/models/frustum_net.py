import functools

import numpy as np
import torch.nn as nn

import modules.functional as F
from models.box_estimation import *
from models.point_dan.point_dan import InstanceSegmentationPointDAN, \
    InstanceSegmentationPointDanSimple
from models.segmentation import *
from models.center_regression_net import CenterRegressionNet, \
    CenterRegressionPointDan

__all__ = ['FrustumPointNet', 'FrustumPointNet2', "FrustumPointDanParallel",
           'FrustumPVCNNE', 'FrustumPointDAN', "FrustumPointDAN2",
           "FrustumPointDanSimple"]


class FrustumNet(nn.Module):
    def __init__(self, num_classes, instance_segmentation_net, box_estimation_net,
                 num_heading_angle_bins, num_size_templates, num_points_per_object,
                 size_templates, extra_feature_channels=1, width_multiplier=1):
        super().__init__()
        if not isinstance(width_multiplier, (list, tuple)):
            width_multiplier = [width_multiplier] * 3
        self.in_channels = 3 + extra_feature_channels
        self.num_classes = num_classes
        self.num_heading_angle_bins = num_heading_angle_bins
        self.num_size_templates = num_size_templates
        self.num_points_per_object = num_points_per_object

        self.inst_seg_net = instance_segmentation_net(num_classes=num_classes,
                                                      extra_feature_channels=extra_feature_channels,
                                                      width_multiplier=width_multiplier[0])
        self.center_reg_net = CenterRegressionNet(num_classes=num_classes, width_multiplier=width_multiplier[1])
        self.box_est_net = box_estimation_net(num_classes=num_classes, num_heading_angle_bins=num_heading_angle_bins,
                                              num_size_templates=num_size_templates,
                                              width_multiplier=width_multiplier[2])
        self.register_buffer('size_templates', size_templates.view(1, self.num_size_templates, 3))

    def forward(self, inputs):
        features = inputs['features']
        one_hot_vectors = inputs['one_hot_vectors']
        assert one_hot_vectors.dim() == 2

        # foreground/background segmentation
        mask_logits = self.inst_seg_net({'features': features, 'one_hot_vectors': one_hot_vectors})
        # mask out Background points
        foreground_coords, foreground_coords_mean, _ = F.logits_mask(
            coords=features[:, :3, :], logits=mask_logits, num_points_per_object=self.num_points_per_object
        )
        # center regression
        delta_coords = self.center_reg_net({'coords': foreground_coords, 'one_hot_vectors': one_hot_vectors})
        foreground_coords = foreground_coords - delta_coords.unsqueeze(-1)
        # box estimation
        estimation = self.box_est_net({'coords': foreground_coords, 'one_hot_vectors': one_hot_vectors})
        estimations = estimation.split([3, self.num_heading_angle_bins, self.num_heading_angle_bins,
                                        self.num_size_templates, self.num_size_templates * 3], dim=-1)

        # parse results
        outputs = dict()
        outputs['mask_logits'] = mask_logits
        outputs['center_reg'] = foreground_coords_mean + delta_coords
        outputs['center'] = estimations[0] + outputs['center_reg']
        outputs['heading_scores'] = estimations[1]
        outputs['heading_residuals_normalized'] = estimations[2]
        outputs['heading_residuals'] = estimations[2] * (np.pi / self.num_heading_angle_bins)
        outputs['size_scores'] = estimations[3]
        size_residuals_normalized = estimations[4].view(-1, self.num_size_templates, 3)
        outputs['size_residuals_normalized'] = size_residuals_normalized
        outputs['size_residuals'] = size_residuals_normalized * self.size_templates

        return outputs


class FrustumPointNet(FrustumNet):
    def __init__(self, num_classes, num_heading_angle_bins, num_size_templates, num_points_per_object,
                 size_templates, extra_feature_channels=1, width_multiplier=1):
        super().__init__(num_classes=num_classes, instance_segmentation_net=InstanceSegmentationPointNet,
                         box_estimation_net=BoxEstimationPointNet, num_heading_angle_bins=num_heading_angle_bins,
                         num_size_templates=num_size_templates, num_points_per_object=num_points_per_object,
                         size_templates=size_templates, extra_feature_channels=extra_feature_channels,
                         width_multiplier=width_multiplier)


class FrustumPointNet2(FrustumNet):
    def __init__(self, num_classes, num_heading_angle_bins, num_size_templates, num_points_per_object,
                 size_templates, extra_feature_channels=1, width_multiplier=1):
        super().__init__(num_classes=num_classes, instance_segmentation_net=InstanceSegmentationPointNet2,
                         box_estimation_net=BoxEstimationPointNet2, num_heading_angle_bins=num_heading_angle_bins,
                         num_size_templates=num_size_templates, num_points_per_object=num_points_per_object,
                         size_templates=size_templates, extra_feature_channels=extra_feature_channels,
                         width_multiplier=width_multiplier)


class FrustumPVCNNE(FrustumNet):
    def __init__(self, num_classes, num_heading_angle_bins, num_size_templates, num_points_per_object,
                 size_templates, extra_feature_channels=1, width_multiplier=1, voxel_resolution_multiplier=1):
        instance_segmentation_net = functools.partial(InstanceSegmentationPVCNN,
                                                      voxel_resolution_multiplier=voxel_resolution_multiplier)
        super().__init__(num_classes=num_classes, instance_segmentation_net=instance_segmentation_net,
                         box_estimation_net=BoxEstimationPointNet, num_heading_angle_bins=num_heading_angle_bins,
                         num_size_templates=num_size_templates, num_points_per_object=num_points_per_object,
                         size_templates=size_templates, extra_feature_channels=extra_feature_channels,
                         width_multiplier=width_multiplier)


class FrustumPointDAN(FrustumNet):
    def __init__(self, num_classes, num_heading_angle_bins, num_size_templates,
                 num_points_per_object, size_templates, extra_feature_channels=1,
                 width_multiplier=1,
                 instance_segmentation_net=InstanceSegmentationPointDAN):
        super().__init__(num_classes=num_classes, instance_segmentation_net=instance_segmentation_net,
                         box_estimation_net=BoxEstimationPointNet, num_heading_angle_bins=num_heading_angle_bins,
                         num_size_templates=num_size_templates, num_points_per_object=num_points_per_object,
                         size_templates=size_templates, extra_feature_channels=extra_feature_channels,
                         width_multiplier=width_multiplier)

    def forward(self, inputs):
        features = inputs['features']
        one_hot_vectors = inputs['one_hot_vectors']
        assert one_hot_vectors.dim() == 2

        # foreground/background segmentation
        mask_logits1, mask_logits2 = self.inst_seg_net(
            {'features': features, 'one_hot_vectors': one_hot_vectors})
        mask_logits = (mask_logits1 + mask_logits2) / 2.0

        # mask out Background points
        foreground_coords, foreground_coords_mean, _ = F.logits_mask(
            coords=features[:, :3, :], logits=mask_logits,
            num_points_per_object=self.num_points_per_object
        )
        # center regression
        delta_coords = self.center_reg_net({'coords': foreground_coords,
                                            'one_hot_vectors': one_hot_vectors})
        foreground_coords = foreground_coords - delta_coords.unsqueeze(-1)
        # box estimation
        estimation = self.box_est_net({'coords': foreground_coords,
                                       'one_hot_vectors': one_hot_vectors})
        estimations = estimation.split([3, self.num_heading_angle_bins,
                                        self.num_heading_angle_bins,
                                        self.num_size_templates,
                                        self.num_size_templates * 3], dim=-1)

        # parse results
        outputs = dict()
        outputs['mask_logits'] = mask_logits
        outputs['mask_logits1'] = mask_logits1
        outputs['mask_logits2'] = mask_logits2
        outputs['center_reg'] = foreground_coords_mean + delta_coords
        outputs['center'] = estimations[0] + outputs['center_reg']
        outputs['heading_scores'] = estimations[1]
        outputs['heading_residuals_normalized'] = estimations[2]
        outputs['heading_residuals'] = estimations[2] * (np.pi / self.num_heading_angle_bins)
        outputs['size_scores'] = estimations[3]
        size_residuals_normalized = estimations[4].view(-1, self.num_size_templates, 3)
        outputs['size_residuals_normalized'] = size_residuals_normalized
        outputs['size_residuals'] = size_residuals_normalized * self.size_templates

        return outputs


class FrustumPointDanSimple(FrustumPointDAN):
    def __init__(self, num_classes, num_heading_angle_bins, num_size_templates,
                 num_points_per_object, size_templates, extra_feature_channels=1,
                 width_multiplier=1):
        super().__init__(num_classes=num_classes,
                         instance_segmentation_net=InstanceSegmentationPointDanSimple,
                         num_heading_angle_bins=num_heading_angle_bins,
                         num_size_templates=num_size_templates,
                         num_points_per_object=num_points_per_object,
                         size_templates=size_templates,
                         extra_feature_channels=extra_feature_channels,
                         width_multiplier=width_multiplier)


class FrustumPointDAN2(nn.Module):
    def __init__(self, num_classes, num_heading_angle_bins, num_size_templates, num_points_per_object,
                 size_templates, extra_feature_channels=1, width_multiplier=1):
        super().__init__()
        if not isinstance(width_multiplier, (list, tuple)):
            width_multiplier = [width_multiplier] * 3
        self.in_channels = 3 + extra_feature_channels
        self.num_classes = num_classes
        self.num_heading_angle_bins = num_heading_angle_bins
        self.num_size_templates = num_size_templates
        self.num_points_per_object = num_points_per_object

        self.inst_seg_net = InstanceSegmentationPointDAN(num_classes=num_classes,
                                                      extra_feature_channels=extra_feature_channels,
                                                      width_multiplier=width_multiplier[0])
        self.center_reg_net = CenterRegressionPointDan(num_classes=num_classes, width_multiplier=width_multiplier[1])
        self.box_est_net = BoxEstimationPointDan(num_classes=num_classes, num_heading_angle_bins=num_heading_angle_bins,
                                              num_size_templates=num_size_templates,
                                              width_multiplier=width_multiplier[2])
        self.register_buffer('size_templates', size_templates.view(1, self.num_size_templates, 3))

    def forward(self, inputs, cons=1, adaptation=False,
                adaptation_s=False, adaptation_t=False):

        features = inputs['features']
        one_hot_vectors = inputs['one_hot_vectors']
        assert one_hot_vectors.dim() == 2

        # foreground/background segmentation
        if not adaptation_s and not adaptation_t:
            mask_logits1, mask_logits2 = self.inst_seg_net(
                {'features': features, 'one_hot_vectors': one_hot_vectors}, cons, adaptation)
            mask_logits = (mask_logits1 + mask_logits2) / 2.0

            # mask out Background points
            foreground_coords, foreground_coords_mean, _ = F.logits_mask(
                coords=features[:, :3, :], logits=mask_logits,
                num_points_per_object=self.num_points_per_object
            )
            # center regression
            delta_coords1, delta_coords2 = self.center_reg_net({'coords': foreground_coords,
                                                'one_hot_vectors': one_hot_vectors}, cons, adaptation)
            delta_coords = (delta_coords1 + delta_coords2) / 2.0
            foreground_coords = foreground_coords - delta_coords.unsqueeze(-1)
            # box estimation
            estimation1, estimation2 = self.box_est_net({'coords': foreground_coords,
                                           'one_hot_vectors': one_hot_vectors}, cons, adaptation)

            estimation = (estimation1 + estimation2) / 2.0

            estimations = estimation.split([3, self.num_heading_angle_bins,
                                            self.num_heading_angle_bins,
                                            self.num_size_templates,
                                            self.num_size_templates * 3], dim=-1)

            estimations1 = estimation1.split([3, self.num_heading_angle_bins,
                                            self.num_heading_angle_bins,
                                            self.num_size_templates,
                                            self.num_size_templates * 3], dim=-1)

            estimations2 = estimation2.split([3, self.num_heading_angle_bins,
                                            self.num_heading_angle_bins,
                                            self.num_size_templates,
                                            self.num_size_templates * 3], dim=-1)


            # parse results
            outputs = dict()
            outputs['mask_logits'] = mask_logits
            outputs['mask_logits1'] = mask_logits1
            outputs['mask_logits2'] = mask_logits2

            outputs['center_reg'] = foreground_coords_mean + delta_coords
            outputs['center_reg1'] = foreground_coords_mean + delta_coords1
            outputs['center_reg2'] = foreground_coords_mean + delta_coords2

            outputs['center'] = estimations[0] + outputs['center_reg']
            outputs['center1'] = estimations1[0] + outputs['center_reg1']
            outputs['center2'] = estimations2[0] + outputs['center_reg2']

            outputs['heading_scores'] = estimations[1]
            outputs['heading_residuals_normalized'] = estimations[2]
            outputs['heading_residuals'] = estimations[2] * (np.pi / self.num_heading_angle_bins)
            outputs['size_scores'] = estimations[3]

            outputs['heading_scores1'] = estimations1[1]
            outputs['heading_residuals_normalized1'] = estimations1[2]
            outputs['heading_residuals1'] = estimations1[2] * (np.pi / self.num_heading_angle_bins)
            outputs['size_scores1'] = estimations1[3]

            outputs['heading_scores2'] = estimations2[1]
            outputs['heading_residuals_normalized2'] = estimations2[2]
            outputs['heading_residuals2'] = estimations2[2] * (np.pi / self.num_heading_angle_bins)
            outputs['size_scores2'] = estimations2[3]

            size_residuals_normalized = estimations[4].view(-1, self.num_size_templates, 3)
            outputs['size_residuals_normalized'] = size_residuals_normalized
            outputs['size_residuals'] = size_residuals_normalized * self.size_templates

            size_residuals_normalized1 = estimations1[4].view(-1, self.num_size_templates, 3)
            outputs['size_residuals_normalized1'] = size_residuals_normalized1
            outputs['size_residuals1'] = size_residuals_normalized1 * self.size_templates

            size_residuals_normalized2 = estimations2[4].view(-1, self.num_size_templates, 3)
            outputs['size_residuals_normalized2'] = size_residuals_normalized2
            outputs['size_residuals2'] = size_residuals_normalized2 * self.size_templates
        else:
            mask_logits1, mask_logits2, seg_mmd_feat = self.inst_seg_net(
                {'features': features, 'one_hot_vectors': one_hot_vectors},
                cons, adaptation, adaptation_s, adaptation_t)
            mask_logits = (mask_logits1 + mask_logits2) / 2.0

            # mask out Background points
            foreground_coords, foreground_coords_mean, _ = F.logits_mask(
                coords=features[:, :3, :], logits=mask_logits,
                num_points_per_object=self.num_points_per_object
            )
            # center regression
            delta_coords1, delta_coords2, centre_mmd_feat = self.center_reg_net({'coords': foreground_coords,
                                                'one_hot_vectors': one_hot_vectors},
                                                               cons, adaptation,
                                                               adaptation_s, adaptation_t)
            delta_coords = (delta_coords1 + delta_coords2) / 2.0
            foreground_coords = foreground_coords - delta_coords.unsqueeze(-1)
            # box estimation
            box_mmd_feat = self.box_est_net({'coords': foreground_coords,
                                           'one_hot_vectors': one_hot_vectors},
                                                        cons, adaptation,
                                                        adaptation_s, adaptation_t)
            outputs = {"seg_mmd_feat": seg_mmd_feat,
                       "cen_mmd_feat": centre_mmd_feat,
                       "box_mmd_feat": box_mmd_feat}

        return outputs


class FrustumPointDanParallel(nn.Module):
    def __init__(self, num_classes, num_heading_angle_bins, num_size_templates, num_points_per_object,
                 size_templates, extra_feature_channels=1, width_multiplier=1):
        super().__init__()
        if not isinstance(width_multiplier, (list, tuple)):
            width_multiplier = [width_multiplier] * 3
        self.in_channels = 3 + extra_feature_channels
        self.num_classes = num_classes
        self.num_heading_angle_bins = num_heading_angle_bins
        self.num_size_templates = num_size_templates
        self.num_points_per_object = num_points_per_object

        self.inst_seg_net = InstanceSegmentationPointDAN(num_classes=num_classes,
                                                      extra_feature_channels=extra_feature_channels,
                                                      width_multiplier=width_multiplier[0])
        self.center_reg_nets = nn.ModuleList([CenterRegressionNet(num_classes=num_classes,
                                                    width_multiplier=width_multiplier[1])
                                for _ in range(2)])

        self.box_est_nets = nn.ModuleList([BoxEstimationPointNet(num_classes=num_classes, num_heading_angle_bins=num_heading_angle_bins,
                                              num_size_templates=num_size_templates,
                                              width_multiplier=width_multiplier[2])
                             for _ in range(2)])
        self.register_buffer('size_templates', size_templates.view(1, self.num_size_templates, 3))

    def forward(self, inputs):
        features = inputs['features']
        one_hot_vectors = inputs['one_hot_vectors']
        assert one_hot_vectors.dim() == 2

        # foreground/background segmentation
        masks = self.inst_seg_net({'features': features,
                                   'one_hot_vectors': one_hot_vectors})

        outputs_list = []

        for i, mask_logits in enumerate(masks):
            # mask out Background points
            foreground_coords, foreground_coords_mean, _ = F.logits_mask(
                coords=features[:, :3, :], logits=mask_logits,
                num_points_per_object=self.num_points_per_object
            )
            # center regression
            delta_coords = self.center_reg_nets[i]({'coords': foreground_coords,
                                                'one_hot_vectors': one_hot_vectors})
            foreground_coords = foreground_coords - delta_coords.unsqueeze(-1)
            # box estimation
            estimation = self.box_est_nets[i]({'coords': foreground_coords,
                                           'one_hot_vectors': one_hot_vectors})
            estimations = estimation.split([3, self.num_heading_angle_bins,
                                            self.num_heading_angle_bins,
                                            self.num_size_templates,
                                            self.num_size_templates * 3], dim=-1)

            # parse results
            outputs = dict()
            outputs['mask_logits'] = mask_logits
            outputs['center_reg'] = foreground_coords_mean + delta_coords
            outputs['center'] = estimations[0] + outputs['center_reg']
            outputs['heading_scores'] = estimations[1]
            outputs['heading_residuals_normalized'] = estimations[2]
            outputs['heading_residuals'] = estimations[2] * (np.pi / self.num_heading_angle_bins)
            outputs['size_scores'] = estimations[3]
            size_residuals_normalized = estimations[4].view(-1, self.num_size_templates, 3)
            outputs['size_residuals_normalized'] = size_residuals_normalized
            outputs['size_residuals'] = size_residuals_normalized * self.size_templates

            outputs_list.append(outputs)

        if self.training:
            return outputs_list
        else:
            outputs = outputs_list[0]
            for k, v in outputs_list[1].items():
                outputs[k] = (outputs[k] + v) / 2.0

            return outputs

