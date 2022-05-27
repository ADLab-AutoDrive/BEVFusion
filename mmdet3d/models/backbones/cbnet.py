import torch
import torch.nn as nn
import torch.nn.functional as F

from mmcv.cnn import constant_init
from mmdet.utils import get_root_logger
from mmdet.models.builder import BACKBONES
from .swin_transformer import SwinTransformer
from mmcv.runner import BaseModule
from mmdet.models.backbones.resnet import ResNet, build_norm_layer, _BatchNorm


'''
For Swin Transformer
'''
class _SwinTransformer(SwinTransformer):
    def _freeze_stages(self):
        if self.frozen_stages >= 0 and hasattr(self, 'patch_embed'):
            self.patch_embed.eval()
            for param in self.patch_embed.parameters():
                param.requires_grad = False

        if self.frozen_stages >= 1 and self.ape:
            self.absolute_pos_embed.requires_grad = False

        if self.frozen_stages >= 2:
            self.pos_drop.eval()
            for i in range(0, self.frozen_stages - 1):
                m = self.layers[i]
                if m is None:
                    continue
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False

    def del_layers(self, del_stages):
        self.del_stages = del_stages
        if self.del_stages>=0:
            del self.patch_embed
        
        if self.del_stages >=1 and self.ape:
            del self.absolute_pos_embed
        
        for i in range(0, self.del_stages - 1):
            self.layers[i] = None

    def forward(self, x, cb_feats=None, pre_tmps=None):
        """Forward function."""
        outs = []
        tmps = []
        if hasattr(self, 'patch_embed'):
            x = self.patch_embed(x)

            Wh, Ww = x.size(2), x.size(3)
            if self.ape:
                # interpolate the position embedding to the corresponding size
                absolute_pos_embed = F.interpolate(
                    self.absolute_pos_embed, size=(Wh, Ww), mode='bicubic')
                x = (x + absolute_pos_embed).flatten(2).transpose(1, 2)  # B Wh*Ww C
            else:
                x = x.flatten(2).transpose(1, 2)
            x = self.pos_drop(x)

            tmps.append((x, Wh, Ww))
        else:
            x, Wh, Ww = pre_tmps[0]

        for i in range(self.num_layers):
            layer = self.layers[i]
            if layer is None:
                x_out, H, W, x, Wh, Ww = pre_tmps[i+1]
            else:
                if cb_feats is not None:
                    x = x + cb_feats[i]
                x_out, H, W, x, Wh, Ww = layer(x, Wh, Ww)
            tmps.append((x_out, H, W, x, Wh, Ww))

            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                x_out = norm_layer(x_out)

                out = x_out.view(-1, H, W,
                                 self.num_features[i]).permute(0, 3, 1, 2).contiguous()
                outs.append(out)

        return tuple(outs), tmps

    def train(self, mode=True):
        """Convert the model into training mode while keep layers freezed."""
        super(_SwinTransformer, self).train(mode)
        self._freeze_stages()


@BACKBONES.register_module()
class CBSwinTransformer(BaseModule):
    def __init__(self, embed_dim=96, cb_zero_init=True, cb_del_stages=1, **kwargs):
        super(CBSwinTransformer, self).__init__()
        self.cb_zero_init = cb_zero_init
        self.cb_del_stages = cb_del_stages
        self.cb_modules = nn.ModuleList()
        for cb_idx in range(2):
            cb_module = _SwinTransformer(embed_dim=embed_dim, **kwargs)
            if cb_idx > 0:
                cb_module.del_layers(cb_del_stages)
            self.cb_modules.append(cb_module)

        self.num_layers = self.cb_modules[0].num_layers

        cb_inplanes = [embed_dim * 2 ** i for i in range(self.num_layers)]

        self.cb_linears = nn.ModuleList()
        for i in range(self.num_layers):
            linears = nn.ModuleList()
            if i >= self.cb_del_stages-1:
                jrange = 4 - i
                for j in range(jrange):
                    if cb_inplanes[i + j] != cb_inplanes[i]:
                        layer = nn.Conv2d(cb_inplanes[i + j], cb_inplanes[i], 1)
                    else:
                        layer = nn.Identity()
                    linears.append(layer)
            self.cb_linears.append(linears)

    def _freeze_stages(self):
        for m in self.cb_modules:
            m._freeze_stages()

    def init_weights(self, pretrained=None):
        """Initialize the weights in backbone.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """
        # constant_init(self.cb_linears, 0)
        if self.cb_zero_init:
            for ls in self.cb_linears:
                for m in ls:
                    constant_init(m, 0)
                        
        for m in self.cb_modules:
            m.init_weights(pretrained=pretrained)

    def spatial_interpolate(self, x, H, W):
        B, C = x.shape[:2]
        if H != x.shape[2] or W != x.shape[3]:
            # B, C, size[0], size[1]
            x = F.interpolate(x, size=(H, W), mode='nearest')
        x = x.view(B, C, -1).permute(0, 2, 1).contiguous()  # B, T, C
        return x

    def _get_cb_feats(self, feats, tmps):
        cb_feats = []
        Wh, Ww = tmps[0][-2:]
        for i in range(self.num_layers):
            feed = 0
            if i >= self.cb_del_stages-1:
                jrange = 4 - i
                for j in range(jrange):
                    tmp = self.cb_linears[i][j](feats[j + i])
                    tmp = self.spatial_interpolate(tmp, Wh, Ww)
                    feed += tmp
            cb_feats.append(feed)
            Wh, Ww = tmps[i+1][-2:]

        return cb_feats

    def forward(self, x):
        outs = []
        for i, module in enumerate(self.cb_modules):
            if i == 0:
                feats, tmps = module(x)
            else:
                feats, tmps = module(x, cb_feats, tmps)

            outs.append(feats)
            
            if i < len(self.cb_modules)-1:
                cb_feats = self._get_cb_feats(outs[-1], tmps)  
        if len(outs) >1:
            outs = outs[-1]
        return tuple(outs)

    def train(self, mode=True):
        """Convert the model into training mode while keep layers freezed."""
        super(CBSwinTransformer, self).train(mode)
        for m in self.cb_modules:
            m.train(mode=mode)
        self._freeze_stages()
        for m in self.cb_linears.modules():
            # trick: eval have effect on BatchNorm only
            if isinstance(m, _BatchNorm):
                m.eval()

