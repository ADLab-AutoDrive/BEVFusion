point_cloud_range = [-54.0, -54.0, -5.0, 54.0, 54.0, 3.0]
class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]
voxel_size = [0.075, 0.075, 0.2]

dataset_type = 'NuScenesDataset'
data_root = 'data/nuscenes/'
input_modality = dict(
    use_lidar=True,
    use_camera=False,
    use_radar=False,
    use_map=False,
    use_external=False)
train_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=[0, 1, 2, 3, 4],
    ),
    dict(
        type='LoadPointsFromMultiSweeps',
        sweeps_num=10,
        use_dim=[0, 1, 2, 3, 4],
    ),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
    # dict(
        # type='ObjectSample',
        # db_sampler=dict(
        #     data_root=data_root,
        #     info_path=data_root + 'nuscenes_dbinfos_train.pkl',
        #     rate=1.0,
        #     prepare=dict(
        #         filter_by_difficulty=[-1],
        #         filter_by_min_points=dict(
        #             car=5,
        #             truck=5,
        #             bus=5,
        #             trailer=5,
        #             construction_vehicle=5,
        #             traffic_cone=5,
        #             barrier=5,
        #             motorcycle=5,
        #             bicycle=5,
        #             pedestrian=5)),
        #     classes=class_names,
        #     sample_groups=dict(
        #         car=2,
        #         truck=3,
        #         construction_vehicle=7,
        #         bus=4,
        #         trailer=6,
        #         barrier=2,
        #         motorcycle=6,
        #         bicycle=6,
        #         pedestrian=2,
        #         traffic_cone=2),
        #     points_loader=dict(
        #         type='LoadPointsFromFile',
        #         coord_type='LIDAR',
        #         load_dim=5,
        #         use_dim=[0, 1, 2, 3, 4],
        #     ))),
    dict(
        type='GlobalRotScaleTrans',
        rot_range=[-0.3925 * 2, 0.3925 * 2],
        scale_ratio_range=[0.9, 1.1],
        translation_std=[0.5, 0.5, 0.5]),
    dict(
        type='RandomFlip3D',
        sync_2d=False,
        flip_ratio_bev_horizontal=0.5,
        flip_ratio_bev_vertical=0.5),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectNameFilter', classes=class_names),
    dict(type='PointShuffle'),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(type='Collect3D', keys=['points', 'gt_bboxes_3d', 'gt_labels_3d'])
]
test_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=[0, 1, 2, 3, 4],
    ),
    dict(
        type='LoadPointsFromMultiSweeps',
        sweeps_num=10,
        use_dim=[0, 1, 2, 3, 4],
    ),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type='GlobalRotScaleTrans',
                rot_range=[0, 0],
                scale_ratio_range=[1.0, 1.0],
                translation_std=[0, 0, 0]),
            dict(type='RandomFlip3D'),
            dict(
                type='DefaultFormatBundle3D',
                class_names=class_names,
                with_label=False),
            dict(type='Collect3D', keys=['points'])
        ])
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=6,
    train=dict(
        type='CBGSDataset',
            dataset=dict(
                type=dataset_type,
                data_root=data_root,
                ann_file=data_root + '/nuscenes_infos_train.pkl',
                load_interval=1,
                pipeline=train_pipeline,
                classes=class_names,
                modality=input_modality,
                test_mode=False,
                box_type_3d='LiDAR'
            )
        ),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + '/nuscenes_infos_val.pkl',
        load_interval=1,
        pipeline=test_pipeline,
        classes=class_names,
        modality=input_modality,
        test_mode=True,
        box_type_3d='LiDAR'),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + '/nuscenes_infos_val.pkl',
        load_interval=1,
        pipeline=test_pipeline,
        classes=class_names,
        modality=input_modality,
        test_mode=True,
        box_type_3d='LiDAR'))