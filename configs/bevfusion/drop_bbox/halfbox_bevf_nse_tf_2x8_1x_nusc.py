_base_ = [
    '../../_base_/datasets/nusc_halfbox_tf.py',
    '../../_base_/schedules/schedule_1x.py',

]
point_cloud_range = [-54.0, -54.0, -5.0, 54.0, 54.0, 3.0]
class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]
voxel_size = [0.075, 0.075, 0.2]
out_size_factor = 8
final_dim=(900, 1600) # HxW
downsample=8
imc = 256

input_modality = dict(
    use_lidar=True,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=False)
num_views = 6
model = dict(
    type='BEVF_TransFusion',
    se=False,
    camera_stream=True, 
    grid=0.6, 
    num_views=6,
    final_dim=final_dim,
    downsample=downsample, 
    imc=imc, 
    lic=256 * 2,
    lc_fusion=True,
    pc_range = point_cloud_range,
    img_backbone=dict(
        type='CBSwinTransformer',
                embed_dim=96,
                depths=[2, 2, 6, 2],
                num_heads=[3, 6, 12, 24],
                window_size=7,
                mlp_ratio=4.,
                qkv_bias=True,
                qk_scale=None,
                drop_rate=0.,
                attn_drop_rate=0.,
                drop_path_rate=0.2,
                ape=False,
                patch_norm=True,
                out_indices=(0, 1, 2, 3),
                use_checkpoint=False),
    img_neck=dict(
        type='FPNC',
        final_dim=final_dim,
        downsample=downsample, 
        in_channels=[96, 192, 384, 768],
        out_channels=256,
        outC=imc,
        use_adp=True,
        num_outs=5),
    pts_voxel_layer=dict(
        max_num_points=10,
        voxel_size=voxel_size,
        max_voxels=(120000, 160000),
        point_cloud_range=point_cloud_range),
    pts_voxel_encoder=dict(
        type='HardSimpleVFE',
        num_features=5,
    ),
    pts_middle_encoder=dict(
        type='SparseEncoder',
        in_channels=5,
        sparse_shape=[41, 1440, 1440],
        output_channels=128,
        order=('conv', 'norm', 'act'),
        encoder_channels=((16, 16, 32), (32, 32, 64), (64, 64, 128), (128, 128)),
        encoder_paddings=((0, 0, 1), (0, 0, 1), (0, 0, [0, 1, 1]), (0, 0)),
        block_type='basicblock'),
    pts_backbone=dict(
        type='SECOND',
        in_channels=256,
        out_channels=[128, 256],
        layer_nums=[5, 5],
        layer_strides=[1, 2],
        norm_cfg=dict(type='BN', eps=0.001, momentum=0.01),
        conv_cfg=dict(type='Conv2d', bias=False)),
    pts_neck=dict(
        type='SECONDFPN',
        in_channels=[128, 256],
        out_channels=[256, 256],
        upsample_strides=[1, 2],
        norm_cfg=dict(type='BN', eps=0.001, momentum=0.01),
        upsample_cfg=dict(type='deconv', bias=False),
        use_conv_for_no_stride=True),
    pts_bbox_head=dict(
        type='TransFusionHead',
        fuse_img=False,
        num_views=num_views,
        in_channels_img=256,
        out_size_factor_img=4,
        num_proposals=200,
        auxiliary=True,
        in_channels=256 * 2,
        hidden_channel=128,
        num_classes=len(class_names),
        num_decoder_layers=1,
        num_heads=8,
        learnable_query_pos=False,
        initialize_by_heatmap=True,
        nms_kernel_size=3,
        ffn_channel=256,
        dropout=0.1,
        bn_momentum=0.1,
        activation='relu',
        common_heads=dict(center=(2, 2), height=(1, 2), dim=(3, 2), rot=(2, 2), vel=(2, 2)),
        bbox_coder=dict(
            type='TransFusionBBoxCoder',
            pc_range=point_cloud_range[:2],
            voxel_size=voxel_size[:2],
            out_size_factor=out_size_factor,
            post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            score_threshold=0.0,
            code_size=10,
        ),
        loss_cls=dict(type='FocalLoss', use_sigmoid=True, gamma=2, alpha=0.25, reduction='mean', loss_weight=1.0),
        # loss_iou=dict(type='CrossEntropyLoss', use_sigmoid=True, reduction='mean', loss_weight=0.0),
        loss_bbox=dict(type='L1Loss', reduction='mean', loss_weight=0.25),
        loss_heatmap=dict(type='GaussianFocalLoss', reduction='mean', loss_weight=1.0),
    ),
    train_cfg=dict(
        pts=dict(
            dataset='nuScenes',
            assigner=dict(
                type='HungarianAssigner3D',
                iou_calculator=dict(type='BboxOverlaps3D', coordinate='lidar'),
                cls_cost=dict(type='FocalLossCost', gamma=2, alpha=0.25, weight=0.15),
                reg_cost=dict(type='BBoxBEVL1Cost', weight=0.25),
                iou_cost=dict(type='IoU3DCost', weight=0.25)
            ),
            pos_weight=-1,
            gaussian_overlap=0.1,
            min_radius=2,
            grid_size=[1440, 1440, 40],  # [x_len, y_len, 1]
            voxel_size=voxel_size,
            out_size_factor=out_size_factor,
            code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2],
            point_cloud_range=point_cloud_range)),
    test_cfg=dict(
        pts=dict(
            dataset='nuScenes',
            grid_size=[1440, 1440, 40],
            out_size_factor=out_size_factor,
            pc_range=point_cloud_range[0:2],
            voxel_size=voxel_size[:2],
            nms_type=None,
        )))
optimizer = dict(type='AdamW', lr=0.0001, betas=(0.9, 0.999), weight_decay=0.05,
                 paramwise_cfg=dict(custom_keys={'absolute_pos_embed': dict(decay_mult=0.),
                                                 'relative_position_bias_table': dict(decay_mult=0.),
                                                 'norm': dict(decay_mult=0.)}))
optimizer_config = dict(grad_clip=dict(max_norm=10, norm_type=2))

# total_epochs = 9
checkpoint_config = dict(interval=1)
log_config = dict(
    interval=50,
    hooks=[dict(type='TextLoggerHook'),
           dict(type='TensorboardLoggerHook')])
dist_params = dict(backend='nccl')
log_level = 'INFO'

load_from = 'work_dirs/fusion_tf_wose.pth'
resume_from = None
workflow = [('train', 1)]
gpu_ids = range(0, 8)

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=6,)

# fp16 = dict(loss_scale=32.0)
