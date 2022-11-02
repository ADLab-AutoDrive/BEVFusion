import torch
from mmcv.runner import force_fp32
from torch.nn import functional as F
from torch.distributions import Normal
import numpy as np

from mmdet.models import DETECTORS
from mmdet3d.models.detectors import MVXFasterRCNN
from .cam_stream_lss import LiftSplatShoot
from mmcv.cnn import (build_conv_layer, build_norm_layer, build_upsample_layer,
                      constant_init, is_norm, kaiming_init)
from torchvision.utils import save_image
from mmcv.cnn import ConvModule, xavier_init
import torch.nn as nn
class SE_Block(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c, c, kernel_size=1, stride=1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return x * self.att(x)

@DETECTORS.register_module()
class BEVF_FasterRCNN_Aug(MVXFasterRCNN):
    """Multi-modality BEVFusion using Faster R-CNN."""

    def __init__(self, lss=False, lc_fusion=False, camera_stream=False,
                camera_depth_range=[4.0, 45.0, 1.0], img_depth_loss_weight=1.0,  img_depth_loss_method='kld',
                grid=0.6, num_views=6, se=False,
                final_dim=(900, 1600), pc_range=[-50, -50, -5, 50, 50, 3], downsample=4, imc=256, lic=384, **kwargs):
        """
        Args:
            lss (bool): using default downsampled r18 BEV encoder in LSS.
            lc_fusion (bool): fusing multi-modalities of camera and LiDAR in BEVFusion.
            camera_stream (bool): using camera stream.
            camera_depth_range, img_depth_loss_weight, img_depth_loss_method: for depth supervision wich is not mentioned in paper.
            grid, num_views, final_dim, pc_range, downsample: args for LSS, see cam_stream_lss.py.
            imc (int): channel dimension of camera BEV feature.
            lic (int): channel dimension of LiDAR BEV feature.

        """
        super(BEVF_FasterRCNN_Aug, self).__init__(**kwargs)
        self.num_views = num_views
        self.lc_fusion = lc_fusion
        self.img_depth_loss_weight = img_depth_loss_weight
        self.img_depth_loss_method = img_depth_loss_method
        self.camera_depth_range = camera_depth_range
        self.lift = camera_stream
        self.se = se
        if camera_stream:
            self.lift_splat_shot_vis = LiftSplatShoot(lss=lss, grid=grid, inputC=imc, camC=64, 
            pc_range=pc_range, final_dim=final_dim, downsample=downsample)
        if lc_fusion:
            if se:
                self.seblock = SE_Block(lic)
            self.reduc_conv = ConvModule(
                lic + imc,
                lic,
                3,
                padding=1,
                conv_cfg=None,
                norm_cfg=dict(type='BN', eps=1e-3, momentum=0.01),
                act_cfg=dict(type='ReLU'),
                inplace=False)
            
        self.freeze_img = kwargs.get('freeze_img', False)
        self.init_weights(pretrained=kwargs.get('pretrained', None))
        self.freeze()

    def freeze(self):
        if self.freeze_img:
            if self.with_img_backbone:
                for param in self.img_backbone.parameters():
                    param.requires_grad = False
            if self.with_img_neck:
                for param in self.img_neck.parameters():
                    param.requires_grad = False
            if self.lift:
                for param in self.lift_splat_shot_vis.parameters():
                    param.requires_grad = False


    def extract_pts_feat(self, pts, img_feats, img_metas, gt_bboxes_3d=None):
        """Extract features of points."""
        if not self.with_pts_backbone:
            return None
        voxels, num_points, coors = self.voxelize(pts)
        voxel_features = self.pts_voxel_encoder(voxels, num_points, coors,
                                                )
        # print(voxel_features.shape) [23884, 64]
        batch_size = coors[-1, 0] + 1
        x = self.pts_middle_encoder(voxel_features, coors, batch_size)
        # print(x.shape)# [1, 64, 400, 400])
        # print(type(gt_bboxes_3d),len(gt_bboxes_3d),gt_bboxes_3d[0])
        # if gt_bboxes_3d:
        #     # from mmdet3d.core.bbox import Box3DMode
        #     # bboxs = gt_bboxes_3d[0].convert_to(Box3DMode.DEPTH).tensor
        #     bboxs = gt_bboxes_3d[0].tensor
        #     # print(bboxs.shape) # ([23, 9]) [3, 9])
        #     bboxs = bboxs[:, [3,4,5,0,1,2,6]]
        #     self.lift_splat_shot_vis.draw_encode_voxels_bev(x, bboxs)
        #     exit(0)

        x = self.pts_backbone(x)
        # print(x[0].shape) ([1, 64, 200, 200])
        # print(len(x))    3
        if self.with_pts_neck:
            x = self.pts_neck(x)
            # print(x[0].shape) [1, 384, 200, 200])

        # bboxs = gt_bboxes_3d[0].tensor
        # bboxs = bboxs[:, [3,4,5,0,1,2,6]]
        # self.lift_splat_shot_vis.draw_encode_voxels_bev(x[0], bboxs, filename='vis/lidar_bev_box',scale=0.5)
        # exit(0)
        return x
    
    
    def extract_feat(self, points, img, img_metas, img_aug_matrix=None, lidar_aug_matrix=None, gt_bboxes_3d=None):
        """Extract features from images and points."""
        # print(img_aug_matrix)
        # print(img_metas)
        img_feats = self.extract_img_feat(img, img_metas)
        pts_feats = self.extract_pts_feat(points, img_feats, img_metas)

        if self.lift:
            BN, C, H, W = img_feats[0].shape
            batch_size = BN//self.num_views
            # print(img_feats[0].shape) # [6, 256, 112, 200]
            # print(pts_feats[0].shape) # [1, 384, 200, 200]    tf ([1, 512, 180, 180])
            img_feats_view = img_feats[0].view(batch_size, self.num_views, C, H, W)
            # print(img_feats_view.shape) # [1, 6, 256, 112, 200]     torch.Size([1, 6, 256, 112, 200])
            rots = []
            trans = []
            for sample_idx in range(batch_size):
                rot_list = []
                trans_list = []
                # print(len(img_metas[sample_idx]['lidar2img'])) # 6个cam
                for mat in img_metas[sample_idx]['lidar2img']:  
                    mat = torch.Tensor(mat).to(img_feats_view.device)
                    rot_list.append(mat.inverse()[:3, :3])
                    # print(mat.inverse()[:3, 3], mat[:3, 3].inverse()) must have at least 2 dimensions. 
                    trans_list.append(mat.inverse()[:3, 3].view(-1))
                    # print(mat.inverse()[:3, 3].view(-1), mat.inverse()[:3, 3])
                rot_list = torch.stack(rot_list, dim=0)
                trans_list = torch.stack(trans_list, dim=0)
                rots.append(rot_list)
                trans.append(trans_list)
            rots = torch.stack(rots)
            trans = torch.stack(trans)
            lidar2img_rt = img_metas[sample_idx]['lidar2img']  #### extrinsic parameters for multi-view images

            # print(len(img_aug_matrix),img_aug_matrix[0].shape) 6 torch.Size([2, 4, 4])
            post_rots=None
            post_trans=None
            if img_aug_matrix is not None:
                img_aug_matrix = torch.stack(img_aug_matrix).permute(1, 0, 2, 3)
                 # print(img_aug_matrix.shape) torch.Size([2, 6, 4, 4])
                post_rots = img_aug_matrix[..., :3, :3]
                post_trans = img_aug_matrix[..., :3, 3]
            
            extra_rots=None
            extra_trans=None
            if lidar_aug_matrix is not None:
                # print(lidar_aug_matrix.shape) torch.Size([2, 4, 4])
                # print(lidar_aug_matrix)
                lidar_aug_matrix = lidar_aug_matrix.unsqueeze(1).repeat(1, self.num_views, 1, 1)
                # print(lidar_aug_matrix[:,0,:,:])
                # print(lidar_aug_matrix.shape)
                extra_rots = lidar_aug_matrix[..., :3, :3]
                extra_trans = lidar_aug_matrix[..., :3, 3]
            # bboxs = gt_bboxes_3d[0].tensor[:, [3,4,5,0,1,2,6]]
            # img_bev_feat, depth_dist = self.lift_splat_shot_vis(img_feats_view, rots, trans, lidar2img_rt=lidar2img_rt, img_metas=img_metas,
            #                                                     post_rots=post_rots, post_trans=post_trans, bboxs=bboxs)
            img_bev_feat, depth_dist = self.lift_splat_shot_vis(img_feats_view, rots, trans, lidar2img_rt=lidar2img_rt, img_metas=img_metas,
                                                                post_rots=post_rots, post_trans=post_trans, extra_rots=extra_rots,extra_trans=extra_trans)
            

            # print(img_bev_feat.shape) # [1, 256, 200, 200]
            if pts_feats is None:
                pts_feats = [img_bev_feat] ####cam stream only
            else:
                if self.lc_fusion:
                    pts_feats = [self.reduc_conv(torch.cat([img_bev_feat, pts_feats[0]], dim=1))]
                    # print(pts_feats[0].shape) [1, 384, 200, 200]
                    if self.se:
                        pts_feats = [self.seblock(pts_feats[0])]
        return dict(
            img_feats = img_feats,
            pts_feats = pts_feats,
            depth_dist = depth_dist
        )
        # return (img_feats, pts_feats, depth_dist)
    
    def simple_test(self, points, img_metas, img=None, img_aug_matrix=None, rescale=False):
        """Test function without augmentaiton."""
        # print(type(img_metas))
        # from torchvision import utils as vutils
        # img_show = img[0].cpu()
        # for i,sing_img in enumerate(img_show):
        #     filename='vis/cam_'+str(i)+'.png'
        #     vutils.save_image(sing_img, 'vis/test_camimage_'+str(i)+'.png',normalize=True)
        # # print(img_aug_matrix[0])
        # exit(0)
        if img_aug_matrix is not None:
            img_aug_matrix = img_aug_matrix[0]
        feature_dict = self.extract_feat(
            points, img=img, img_metas=img_metas, img_aug_matrix=img_aug_matrix)
        img_feats = feature_dict['img_feats']
        pts_feats = feature_dict['pts_feats'] 
        depth_dist = feature_dict['depth_dist']

        bbox_list = [dict() for i in range(len(img_metas))]
        if pts_feats and self.with_pts_bbox:
            bbox_pts = self.simple_test_pts(
                # pts_feats, img_feats, img_metas, rescale=rescale)
                pts_feats, img_metas, rescale=rescale)
            for result_dict, pts_bbox in zip(bbox_list, bbox_pts):
                result_dict['pts_bbox'] = pts_bbox
        if img_feats and self.with_img_bbox:
            bbox_img = self.simple_test_img(
                img_feats, img_metas, rescale=rescale)
            for result_dict, img_bbox in zip(bbox_list, bbox_img):
                result_dict['img_bbox'] = img_bbox
        return bbox_list

    def forward_train(self,
                      points=None,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      gt_labels=None,
                      gt_bboxes=None,
                      img=None,
                      img_depth=None,
                      img_aug_matrix=None,
                      lidar_aug_matrix=None,
                      proposals=None,
                      gt_bboxes_ignore=None):
        # print(img_aug_matrix)
        # print(points[0].shape, img[0].shape) torch.Size([360012, 4]) torch.Size([6, 3, 448, 800])

        # points_numpy = points[0].cpu()
        # import matplotlib.pyplot as plt
        # from mpl_toolkits.mplot3d import Axes3D
        # fig = plt.figure(figsize=(10, 7.5))
        # ax = Axes3D(fig)
        # ax.view_init(30, 150)
        # ax.scatter(xs=[-54, 54], ys=[-54, 54], zs=[-10, 10], c='white')
        # ax.scatter(xs=points_numpy[:, 1], ys=-points_numpy[:, 0], zs=points_numpy[:, 2], c='blue', s=1)
        # ax.set_xlabel('x')
        # ax.set_ylabel('y')
        # ax.set_zlabel('z')
        # plt.gca().set_box_aspect((54, 54, 10))
        # plt.savefig('vis/lidar.png')

        # from torchvision import utils as vutils
        # img_show = img[0].cpu()
        # for i,sing_img in enumerate(img_show):
        #     filename='vis/cam_'+str(i)+'.png'
        #     vutils.save_image(sing_img, 'vis/test_camimage_'+str(i)+'.png',normalize=True)
        # exit(0)

        # # print(type(gt_bboxes_3d),gt_bboxes_3d[0])
        # visualize_lidar('vis/lidar_bbox.png', points_numpy, gt_bboxes_3d[0])

        feature_dict = self.extract_feat(
            points, img=img, img_metas=img_metas, gt_bboxes_3d=gt_bboxes_3d, img_aug_matrix=img_aug_matrix, lidar_aug_matrix=lidar_aug_matrix)
        img_feats = feature_dict['img_feats']
        pts_feats = feature_dict['pts_feats'] 
        depth_dist = feature_dict['depth_dist']

        losses = dict()
        if pts_feats:
            losses_pts = self.forward_pts_train(pts_feats, gt_bboxes_3d,
                                                gt_labels_3d, img_metas,
                                                gt_bboxes_ignore)
            losses.update(losses_pts)
        if img_feats:
            losses_img = self.forward_img_train(
                img_feats,
                img_metas=img_metas,
                gt_bboxes=gt_bboxes,
                gt_labels=gt_labels,
                gt_bboxes_ignore=gt_bboxes_ignore,
                proposals=proposals)
            if img_depth is not None:
                loss_depth = self.depth_dist_loss(depth_dist, img_depth, loss_method=self.img_depth_loss_method, img=img) * self.img_depth_loss_weight
                losses.update(img_depth_loss=loss_depth)
            losses.update(losses_img)
        return losses
    
    def depth_dist_loss(self, predict_depth_dist, gt_depth, loss_method='kld', img=None):
        # predict_depth_dist: B, N, D, H, W
        # gt_depth: B, N, H', W'
        B, N, D, H, W = predict_depth_dist.shape
        guassian_depth, min_depth = gt_depth[..., 1:], gt_depth[..., 0]
        mask = (min_depth>=self.camera_depth_range[0]) & (min_depth<=self.camera_depth_range[1])
        mask = mask.view(-1)
        guassian_depth = guassian_depth.view(-1, D)[mask]
        predict_depth_dist = predict_depth_dist.permute(0, 1, 3, 4, 2).reshape(-1, D)[mask]
        if loss_method=='kld':
            loss = F.kl_div(torch.log(predict_depth_dist), guassian_depth, reduction='mean', log_target=False)
        elif loss_method=='mse':
            loss = F.mse_loss(predict_depth_dist, guassian_depth)
        else:
            raise NotImplementedError
        return loss




from typing import List, Optional, Tuple
import cv2
import mmcv
import numpy as np
from matplotlib import pyplot as plt
import os
from mmdet3d.core.bbox import LiDARInstance3DBoxes
def visualize_lidar(
    fpath,
    lidar,
    bboxes,
    labels: Optional[np.ndarray] = None,
    classes: Optional[List[str]] = None,
    xlim: Tuple[float, float] = (-50, 50),
    ylim: Tuple[float, float] = (-50, 50),
    color: Optional[Tuple[int, int, int]] = None,
    radius: float = 15,
    thickness: float = 25,
) -> None:
    fig = plt.figure(figsize=(xlim[1] - xlim[0], ylim[1] - ylim[0]))

    ax = plt.gca()
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect(1)
    ax.set_axis_off()

    if lidar is not None:
        plt.scatter(
            lidar[:, 0],
            lidar[:, 1],
            s=radius,
            c="white",
        )

    if bboxes is not None and len(bboxes) > 0:
        coords = bboxes.corners[:, [0, 3, 7, 4, 0], :2]
        for index in range(coords.shape[0]):
            # name = classes[labels[index]]
            plt.plot(
                coords[index, :, 0],
                coords[index, :, 1],
                linewidth=thickness,
                # color=np.array(color or OBJECT_PALETTE[name]) / 255,
            )

    mmcv.mkdir_or_exist(os.path.dirname(fpath))
    fig.savefig(
        fpath,
        dpi=10,
        facecolor="black",
        format="png",
        bbox_inches="tight",
        pad_inches=0,
    )
    plt.close()

