import torch
import torch.nn as nn
from mmcv.cnn import ConvModule, normal_init
from mmcv.ops import DeformConv2d

from mmdet.core import multi_apply, multiclass_nms
from ..builder import HEADS
from .anchor_free_head import AnchorFreeHead

INF = 1e8


class FeatureAlign(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 deform_groups=4):
        super(FeatureAlign, self).__init__()
        offset_channels = kernel_size * kernel_size * 2
        self.conv_offset = nn.Conv2d(
            4, deform_groups * offset_channels, 1, bias=False)
        self.conv_adaption = DeformConv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=(kernel_size - 1) // 2,
            deform_groups=deform_groups)
        self.relu = nn.ReLU(inplace=True)

    def init_weights(self):
        normal_init(self.conv_offset, std=0.1)
        normal_init(self.conv_adaption, std=0.01)

    def forward(self, x, shape):
        offset = self.conv_offset(shape)
        x = self.relu(self.conv_adaption(x, offset))
        return x


@HEADS.register_module()
class FoveaHead(AnchorFreeHead):
    """FoveaBox: Beyond Anchor-based Object Detector
    https://arxiv.org/abs/1904.03797
    """

    def __init__(self,
                 num_classes,
                 in_channels,
                 base_edge_list=(16, 32, 64, 128, 256),
                 scale_ranges=((8, 32), (16, 64), (32, 128), (64, 256), (128,
                                                                         512)),
                 sigma=0.4,
                 with_deform=False,
                 deform_groups=4,
                 **kwargs):
        self.base_edge_list = base_edge_list
        self.scale_ranges = scale_ranges
        self.sigma = sigma
        self.with_deform = with_deform
        self.deform_groups = deform_groups
        super().__init__(num_classes, in_channels, **kwargs)

    def _init_layers(self):
        # box branch
        super()._init_reg_convs()
        self.conv_reg = nn.Conv2d(self.feat_channels, 4, 3, padding=1)

        # cls branch
        if not self.with_deform:
            super()._init_cls_convs()
            self.conv_cls = nn.Conv2d(
                self.feat_channels, self.cls_out_channels, 3, padding=1)
        else:
            self.cls_convs = nn.ModuleList()
            self.cls_convs.append(
                ConvModule(
                    self.feat_channels, (self.feat_channels * 4),
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    bias=self.norm_cfg is None))
            self.cls_convs.append(
                ConvModule((self.feat_channels * 4), (self.feat_channels * 4),
                           1,
                           stride=1,
                           padding=0,
                           conv_cfg=self.conv_cfg,
                           norm_cfg=self.norm_cfg,
                           bias=self.norm_cfg is None))
            self.feature_adaption = FeatureAlign(
                self.feat_channels,
                self.feat_channels,
                kernel_size=3,
                deform_groups=self.deform_groups)
            self.conv_cls = nn.Conv2d(
                int(self.feat_channels * 4),
                self.cls_out_channels,
                3,
                padding=1)

    def init_weights(self):
        super().init_weights()
        if self.with_deform:
            self.feature_adaption.init_weights()

    def forward_single(self, x):
        cls_feat = x
        reg_feat = x
        for reg_layer in self.reg_convs:
            reg_feat = reg_layer(reg_feat)
        bbox_pred = self.conv_reg(reg_feat)
        if self.with_deform:
            cls_feat = self.feature_adaption(cls_feat, bbox_pred.exp())
        for cls_layer in self.cls_convs:
            cls_feat = cls_layer(cls_feat)
        cls_score = self.conv_cls(cls_feat)
        return cls_score, bbox_pred

    def _get_points_single(self, *args, **kwargs):
        y, x = super()._get_points_single(*args, **kwargs)
        return y + 0.5, x + 0.5

    def loss(self,
             cls_scores,
             bbox_preds,
             gt_bbox_list,
             gt_label_list,
             img_metas,
             gt_bboxes_ignore=None):
        assert len(cls_scores) == len(bbox_preds)

        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        points = self.get_points(featmap_sizes, bbox_preds[0].dtype,
                                 bbox_preds[0].device)
        num_imgs = cls_scores[0].size(0)
        flatten_cls_scores = [
            cls_score.permute(0, 2, 3, 1).reshape(-1, self.cls_out_channels)
            for cls_score in cls_scores
        ]
        flatten_bbox_preds = [
            bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4)
            for bbox_pred in bbox_preds
        ]
        flatten_cls_scores = torch.cat(flatten_cls_scores)
        flatten_bbox_preds = torch.cat(flatten_bbox_preds)
        flatten_labels, flatten_bbox_targets = self.get_targets(
            gt_bbox_list, gt_label_list, featmap_sizes, points)

        # FG cat_id: [0, num_classes -1], BG cat_id: num_classes
        pos_inds = ((flatten_labels >= 0)
                    & (flatten_labels < self.num_classes)).nonzero().view(-1)
        num_pos = len(pos_inds)

        loss_cls = self.loss_cls(
            flatten_cls_scores, flatten_labels, avg_factor=num_pos + num_imgs)
        if num_pos > 0:
            pos_bbox_preds = flatten_bbox_preds[pos_inds]
            pos_bbox_targets = flatten_bbox_targets[pos_inds]
            pos_weights = pos_bbox_targets.new_zeros(
                pos_bbox_targets.size()) + 1.0
            loss_bbox = self.loss_bbox(
                pos_bbox_preds,
                pos_bbox_targets,
                pos_weights,
                avg_factor=num_pos)
        else:
            loss_bbox = torch.tensor(
                0,
                dtype=flatten_bbox_preds.dtype,
                device=flatten_bbox_preds.device)
        return dict(loss_cls=loss_cls, loss_bbox=loss_bbox)

    def get_targets(self, gt_bbox_list, gt_label_list, featmap_sizes, points):
        label_list, bbox_target_list = multi_apply(
            self._get_target_single,
            gt_bbox_list,
            gt_label_list,
            featmap_size_list=featmap_sizes,
            point_list=points)
        flatten_labels = [
            torch.cat([
                labels_level_img.flatten() for labels_level_img in labels_level
            ]) for labels_level in zip(*label_list)
        ]
        flatten_bbox_targets = [
            torch.cat([
                bbox_targets_level_img.reshape(-1, 4)
                for bbox_targets_level_img in bbox_targets_level
            ]) for bbox_targets_level in zip(*bbox_target_list)
        ]
        flatten_labels = torch.cat(flatten_labels)
        flatten_bbox_targets = torch.cat(flatten_bbox_targets)
        return flatten_labels, flatten_bbox_targets

    def _get_target_single(self,
                           gt_bboxes_raw,
                           gt_labels_raw,
                           featmap_size_list=None,
                           point_list=None):

        gt_areas = torch.sqrt((gt_bboxes_raw[:, 2] - gt_bboxes_raw[:, 0]) *
                              (gt_bboxes_raw[:, 3] - gt_bboxes_raw[:, 1]))
        label_list = []
        bbox_target_list = []
        # for each pyramid, find the cls and box target
        for base_len, (lower_bound, upper_bound), stride, featmap_size, \
            (y, x) in zip(self.base_edge_list, self.scale_ranges,
                          self.strides, featmap_size_list, point_list):
            # FG cat_id: [0, num_classes -1], BG cat_id: num_classes
            labels = gt_labels_raw.new_zeros(featmap_size) + self.num_classes
            bbox_targets = gt_bboxes_raw.new(featmap_size[0], featmap_size[1],
                                             4) + 1
            # scale assignment
            hit_indices = ((gt_areas >= lower_bound) &
                           (gt_areas <= upper_bound)).nonzero().flatten()
            if len(hit_indices) == 0:
                label_list.append(labels)
                bbox_target_list.append(torch.log(bbox_targets))
                continue
            _, hit_index_order = torch.sort(-gt_areas[hit_indices])
            hit_indices = hit_indices[hit_index_order]
            gt_bboxes = gt_bboxes_raw[hit_indices, :] / stride
            gt_labels = gt_labels_raw[hit_indices]
            half_w = 0.5 * (gt_bboxes[:, 2] - gt_bboxes[:, 0])
            half_h = 0.5 * (gt_bboxes[:, 3] - gt_bboxes[:, 1])
            # valid fovea area: left, right, top, down
            pos_left = torch.ceil(
                gt_bboxes[:, 0] + (1 - self.sigma) * half_w - 0.5).long().\
                clamp(0, featmap_size[1] - 1)
            pos_right = torch.floor(
                gt_bboxes[:, 0] + (1 + self.sigma) * half_w - 0.5).long().\
                clamp(0, featmap_size[1] - 1)
            pos_top = torch.ceil(
                gt_bboxes[:, 1] + (1 - self.sigma) * half_h - 0.5).long().\
                clamp(0, featmap_size[0] - 1)
            pos_down = torch.floor(
                gt_bboxes[:, 1] + (1 + self.sigma) * half_h - 0.5).long().\
                clamp(0, featmap_size[0] - 1)
            for px1, py1, px2, py2, label, (gt_x1, gt_y1, gt_x2, gt_y2) in \
                    zip(pos_left, pos_top, pos_right, pos_down, gt_labels,
                        gt_bboxes_raw[hit_indices, :]):
                labels[py1:py2 + 1, px1:px2 + 1] = label
                bbox_targets[py1:py2 + 1, px1:px2 + 1, 0] = \
                    (stride * x[py1:py2 + 1, px1:px2 + 1] - gt_x1) / base_len
                bbox_targets[py1:py2 + 1, px1:px2 + 1, 1] = \
                    (stride * y[py1:py2 + 1, px1:px2 + 1] - gt_y1) / base_len
                bbox_targets[py1:py2 + 1, px1:px2 + 1, 2] = \
                    (gt_x2 - stride * x[py1:py2 + 1, px1:px2 + 1]) / base_len
                bbox_targets[py1:py2 + 1, px1:px2 + 1, 3] = \
                    (gt_y2 - stride * y[py1:py2 + 1, px1:px2 + 1]) / base_len
            bbox_targets = bbox_targets.clamp(min=1. / 16, max=16.)
            label_list.append(labels)
            bbox_target_list.append(torch.log(bbox_targets))
        return label_list, bbox_target_list

    def get_bboxes(self,
                   cls_scores,
                   bbox_preds,
                   img_metas,
                   cfg=None,
                   rescale=None):
        assert len(cls_scores) == len(bbox_preds)
        num_levels = len(cls_scores)
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        points = self.get_points(
            featmap_sizes,
            bbox_preds[0].dtype,
            bbox_preds[0].device,
            flatten=True)
        result_list = []
        for img_id in range(len(img_metas)):
            cls_score_list = [
                cls_scores[i][img_id].detach() for i in range(num_levels)
            ]
            bbox_pred_list = [
                bbox_preds[i][img_id].detach() for i in range(num_levels)
            ]
            img_shape = img_metas[img_id]['img_shape']
            scale_factor = img_metas[img_id]['scale_factor']
            det_bboxes = self._get_bboxes_single(cls_score_list,
                                                 bbox_pred_list, featmap_sizes,
                                                 points, img_shape,
                                                 scale_factor, cfg, rescale)
            result_list.append(det_bboxes)
        return result_list

    def _get_bboxes_single(self,
                           cls_scores,
                           bbox_preds,
                           featmap_sizes,
                           point_list,
                           img_shape,
                           scale_factor,
                           cfg,
                           rescale=False):
        cfg = self.test_cfg if cfg is None else cfg
        assert len(cls_scores) == len(bbox_preds) == len(point_list)
        det_bboxes = []
        det_scores = []
        for cls_score, bbox_pred, featmap_size, stride, base_len, (y, x) \
                in zip(cls_scores, bbox_preds, featmap_sizes, self.strides,
                       self.base_edge_list, point_list):
            assert cls_score.size()[-2:] == bbox_pred.size()[-2:]
            scores = cls_score.permute(1, 2, 0).reshape(
                -1, self.cls_out_channels).sigmoid()
            bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, 4).exp()
            nms_pre = cfg.get('nms_pre', -1)
            if (nms_pre > 0) and (scores.shape[0] > nms_pre):
                max_scores, _ = scores.max(dim=1)
                _, topk_inds = max_scores.topk(nms_pre)
                bbox_pred = bbox_pred[topk_inds, :]
                scores = scores[topk_inds, :]
                y = y[topk_inds]
                x = x[topk_inds]
            x1 = (stride * x - base_len * bbox_pred[:, 0]).\
                clamp(min=0, max=img_shape[1] - 1)
            y1 = (stride * y - base_len * bbox_pred[:, 1]).\
                clamp(min=0, max=img_shape[0] - 1)
            x2 = (stride * x + base_len * bbox_pred[:, 2]).\
                clamp(min=0, max=img_shape[1] - 1)
            y2 = (stride * y + base_len * bbox_pred[:, 3]).\
                clamp(min=0, max=img_shape[0] - 1)
            bboxes = torch.stack([x1, y1, x2, y2], -1)
            det_bboxes.append(bboxes)
            det_scores.append(scores)
        det_bboxes = torch.cat(det_bboxes)
        if rescale:
            det_bboxes /= det_bboxes.new_tensor(scale_factor)
        det_scores = torch.cat(det_scores)
        padding = det_scores.new_zeros(det_scores.shape[0], 1)
        # remind that we set FG labels to [0, num_class-1] since mmdet v2.0
        # BG cat_id: num_class
        det_scores = torch.cat([det_scores, padding], dim=1)
        det_bboxes, det_labels = multiclass_nms(det_bboxes, det_scores,
                                                cfg.score_thr, cfg.nms,
                                                cfg.max_per_img)
        return det_bboxes, det_labels
