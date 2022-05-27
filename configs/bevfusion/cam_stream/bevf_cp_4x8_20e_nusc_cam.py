_base_ = [
    '../../_base_/datasets/nusc_cam_cp.py',
    '../../_base_/models/centerpoint_dcn_nus.py',
    '../../_base_/schedules/cyclic_20e.py', 
    '../../_base_/default_runtime.py'
]
voxel_size = [0.075, 0.075, 0.2]
point_cloud_range = [-54, -54, -5.0, 54, 54, 3.0]
# For nuScenes we usually do 10-class detection
class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]
optimizer = dict(_delete_=True, type='AdamW', lr=0.0001, betas=(0.9, 0.999), weight_decay=0.05,
                 paramwise_cfg=dict(custom_keys={'absolute_pos_embed': dict(decay_mult=0.),
                                                 'relative_position_bias_table': dict(decay_mult=0.),
                                                 'norm': dict(decay_mult=0.)}))

final_dim=(900, 1600) # HxW
downsample=8
imc = 256
model = dict(
    type='BEVF_CenterPoint',
    camera_stream=True, 
    grid=0.6, 
    num_views=6,
    final_dim=final_dim,
    downsample=downsample, 
    imc=imc, 
    lic=256 * 2,
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
        use_adp=True,
        outC=imc,
        num_outs=5),
    pts_bbox_head=dict(
        type='CenterHead',
        in_channels=imc,
        separate_head=dict(
            type='DCNSeparateHead',
            dcn_config=dict(
                type='DCN',
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                padding=1,
                groups=4),
            init_bias=-2.19,
            final_kernel=3),
        bbox_coder=dict(
            voxel_size=voxel_size[:2], pc_range=point_cloud_range[:2])),
    train_cfg=dict(
        pts=dict(
            grid_size=[1440, 1440, 40],
            voxel_size=voxel_size,
            point_cloud_range=point_cloud_range)),
    test_cfg=dict(
        pts=dict(voxel_size=voxel_size[:2], pc_range=point_cloud_range[:2], nms_type='circle')))


data = dict(
    samples_per_gpu=4,
    workers_per_gpu=6,)

load_img_from = 'work_dirs/mask_rcnn_dbswin-t_fpn_3x_nuim_cocopre/epoch_36.pth'
