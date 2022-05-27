import numpy as np
import torch
from mmcv.cnn import (build_conv_layer, build_norm_layer, build_upsample_layer,
                      constant_init, is_norm, kaiming_init)
from mmcv.runner import auto_fp16
from torch import nn as nn
import torch.nn.functional as F
from mmdet.models import NECKS
from mmdet.models.necks import FPN
from mmcv.cnn import ConvModule, xavier_init

class gapcontext(nn.Module):
    def __init__(self, in_channels, out_channels, conv_cfg=None, norm_cfg=None, act_cfg=None):
        super().__init__()
        self.gap = nn.Sequential(
                    ConvModule(
                        in_channels,
                        in_channels,
                        1,
                        padding=0,
                        stride=1,
                        conv_cfg=conv_cfg,
                        norm_cfg=norm_cfg,
                        act_cfg=act_cfg,
                        inplace=False),
                    nn.AdaptiveAvgPool2d(1))
        self.conv = ConvModule(
                        in_channels,
                        out_channels,
                        1,
                        padding=0,
                        stride=1,
                        conv_cfg=conv_cfg,
                        norm_cfg=norm_cfg,
                        act_cfg=act_cfg,
                        inplace=False)

    def forward(self, x):
        gap = self.gap(x)
        out = x + F.interpolate(gap, x.shape[2:], mode='bilinear', align_corners=True)
        out = self.conv(out)
        return out

@NECKS.register_module()
class FPNC(FPN):

    def __init__(self,
            conv_cfg=None,
            norm_cfg=None,
            act_cfg=None,
            final_dim=(900, 1600), 
            downsample=4, 
            use_adp=False,
            fuse_conv_cfg=None,
            outC=256,
            **kwargs):
        super(FPNC, self).__init__(
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            **kwargs)
        self.target_size = (final_dim[0] // downsample, final_dim[1] // downsample)
        self.use_adp = use_adp
        if use_adp:
            adp_list = []
            for i in range(self.num_outs):
                if i==0:
                    resize = nn.AdaptiveAvgPool2d(self.target_size)
                else:
                    resize = nn.Upsample(size = self.target_size, mode='bilinear', align_corners=True)
                adp = nn.Sequential(
                    resize,
                    ConvModule(
                        self.out_channels,
                        self.out_channels,
                        1,
                        padding=0,
                        conv_cfg=fuse_conv_cfg,
                        norm_cfg=norm_cfg,
                        act_cfg=act_cfg,
                        inplace=False),
                )
                adp_list.append(adp)
            self.adp = nn.ModuleList(adp_list)

        self.reduc_conv = ConvModule(
                self.out_channels * self.num_outs,
                outC,
                3,
                padding=1,
                conv_cfg=fuse_conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                inplace=False)

    @auto_fp16()
    def forward(self, x):
        outs = super().forward(x)
        if len(outs) > 1:
            resize_outs = []
            if self.use_adp:
                for i in range(len(outs)):
                    feature = self.adp[i](outs[i])
                    resize_outs.append(feature)
            else:
                target_size = self.target_size
                for i in range(len(outs)):
                    feature = outs[i]
                    if feature.shape[2:] != target_size:
                        feature = F.interpolate(feature, target_size,  mode='bilinear', align_corners=True)
                    resize_outs.append(feature)
            out = torch.cat(resize_outs, dim=1)
            out = self.reduc_conv(out)
                

        else:
            out = outs[0]
        return [out]
