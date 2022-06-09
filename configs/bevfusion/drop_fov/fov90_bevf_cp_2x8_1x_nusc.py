_base_ = [
    '../../_base_/datasets/nusc_fov90_cp.py',
    '../../_base_/models/centerpoint_01voxel_second_secfpn_nus.py',
    '../../_base_/schedules/schedule_1x.py',
    '../../_base_/default_runtime.py'
]
voxel_size = [0.075, 0.075, 0.2]
point_cloud_range = [-54, -54, -5.0, 54, 54, 3.0]
# For nuScenes we usually do 10-class detection
class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]
final_dim=(900, 1600) # HxW
downsample=8
imc = 256

model = dict(
    type='BEVF_CenterPoint',
    se=True,
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
    pts_voxel_layer=dict(voxel_size=voxel_size, point_cloud_range=point_cloud_range),
    pts_middle_encoder=dict(sparse_shape=[41, 1440, 1440]),
    pts_bbox_head=dict(
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
    samples_per_gpu=2,
    workers_per_gpu=6,)


optimizer = dict(type='AdamW', lr=0.0001, betas=(0.9, 0.999), weight_decay=0.05,
                paramwise_cfg=dict(custom_keys={'absolute_pos_embed': dict(decay_mult=0.),
                                                 'relative_position_bias_table': dict(decay_mult=0.),
                                                 'norm': dict(decay_mult=0.)}))

load_from = 'work_dirs/bevf_cp_4x8_6e_nusc/epoch_6.pth'
