from __future__ import division

import argparse
import copy
import logging
import mmcv
import os
import json
import time
import torch
import warnings
from mmcv import Config, DictAction
from mmcv.runner import get_dist_info, init_dist
from os import path as osp
from collections import OrderedDict

from mmdet3d import __version__
from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_detector
from mmdet3d.utils import collect_env, get_root_logger
from mmdet.apis import set_random_seed, train_detector


def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument(
        '--resume-from', help='the checkpoint file to resume from')
    parser.add_argument(
        '--no-validate',
        action='store_true',
        help='whether not to evaluate the checkpoint during training')
    group_gpus = parser.add_mutually_exclusive_group()
    group_gpus.add_argument(
        '--gpus',
        type=int,
        help='number of gpus to use '
        '(only applicable to non-distributed training)')
    group_gpus.add_argument(
        '--gpu-ids',
        type=int,
        nargs='+',
        help='ids of gpus to use '
        '(only applicable to non-distributed training)')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument(
        '--options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file (deprecate), '
        'change to --cfg-options instead.')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument(
        '--autoscale-lr',
        action='store_true',
        help='automatically scale lr with the number of gpus')
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    if args.options and args.cfg_options:
        raise ValueError(
            '--options and --cfg-options cannot be both specified, '
            '--options is deprecated in favor of --cfg-options')
    if args.options:
        warnings.warn('--options is deprecated in favor of --cfg-options')
        args.cfg_options = args.options

    return args


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    
    autodrive_hyper_params = os.environ.get("HYPER_PARAMS")
    if autodrive_hyper_params is not None:
        autodrive_hyper_params = json.loads(autodrive_hyper_params)
        print(autodrive_hyper_params)
        cfg.merge_from_dict(autodrive_hyper_params)

    # import modules from string list.
    if cfg.get('custom_imports', None):
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(**cfg['custom_imports'])

    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])
    if args.resume_from is not None:
        cfg.resume_from = args.resume_from
    if args.gpu_ids is not None:
        cfg.gpu_ids = args.gpu_ids
    else:
        cfg.gpu_ids = range(1) if args.gpus is None else range(args.gpus)

    if args.autoscale_lr:
        # apply the linear scaling rule (https://arxiv.org/abs/1706.02677)
        cfg.optimizer['lr'] = cfg.optimizer['lr'] * len(cfg.gpu_ids) / 8

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)
        # re-set gpu_ids with distributed training mode
        _, world_size = get_dist_info()
        cfg.gpu_ids = range(world_size)

    # create work_dir
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
    # dump config
    cfg.dump(osp.join(cfg.work_dir, osp.basename(args.config)))
    # init the logger before other steps
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(cfg.work_dir, f'{timestamp}.log')
    logger = get_root_logger(log_file=log_file, log_level=cfg.log_level)

    # add a logging filter
    logging_filter = logging.Filter('mmdet')
    logging_filter.filter = lambda record: record.find('mmdet') != -1

    # init the meta dict to record some important information such as
    # environment info and seed, which will be logged
    meta = dict()
    # log env info
    env_info_dict = collect_env()
    env_info = '\n'.join([(f'{k}: {v}') for k, v in env_info_dict.items()])
    dash_line = '-' * 60 + '\n'
    logger.info('Environment info:\n' + dash_line + env_info + '\n' +
                dash_line)
    meta['env_info'] = env_info
    meta['config'] = cfg.pretty_text

    # log some basic info
    logger.info(f'Distributed training: {distributed}')
    logger.info(f'Config:\n{cfg.pretty_text}')

    # set random seeds
    if args.seed is not None:
        logger.info(f'Set random seed to {args.seed}, '
                    f'deterministic: {args.deterministic}')
        set_random_seed(args.seed, deterministic=args.deterministic)
    cfg.seed = args.seed
    meta['seed'] = args.seed
    meta['exp_name'] = osp.basename(args.config)

    model = build_detector(
        cfg.model,
        train_cfg=cfg.get('train_cfg'),
        test_cfg=cfg.get('test_cfg'))
    # sync_bn = cfg.get('sync_bn', True)
    # if distributed and sync_bn:
    #     model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    #     print('Convert to SyncBatchNorm')

    if 'freeze_lidar_components' in cfg and cfg['freeze_lidar_components'] is True:
        logger.info(f"param need to update:")
        param_grad = []
        param_nograd = []

        for name, param in model.named_parameters():
            if 'pts' in name and 'pts_bbox_head' not in name:
                param.requires_grad = False
            if 'pts_bbox_head.decoder.0' in name:
                param.requires_grad = False
            if 'pts_bbox_head.shared_conv' in name and 'pts_bbox_head.shared_conv_img' not in name:
                param.requires_grad = False
            if 'pts_bbox_head.heatmap_head' in name and 'pts_bbox_head.heatmap_head_img' not in name:
                param.requires_grad = False
            if 'pts_bbox_head.prediction_heads.0' in name:
                param.requires_grad = False
            if 'pts_bbox_head.class_encoding' in name:
                param.requires_grad = False

        from torch import nn

        def fix_bn(m):
            if isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.BatchNorm2d):
                m.track_running_stats = False

        model.pts_voxel_layer.apply(fix_bn)
        model.pts_voxel_encoder.apply(fix_bn)
        model.pts_middle_encoder.apply(fix_bn)
        model.pts_backbone.apply(fix_bn)
        model.pts_neck.apply(fix_bn)
        print('freeze lidar backbone')
        if 'TransFusion' in cfg.model.type and ('no_freeze_head' not in cfg):
            print('freez head')
            model.pts_bbox_head.heatmap_head.apply(fix_bn)
            model.pts_bbox_head.shared_conv.apply(fix_bn)
            model.pts_bbox_head.class_encoding.apply(fix_bn)
            model.pts_bbox_head.decoder[0].apply(fix_bn)
            model.pts_bbox_head.prediction_heads[0].apply(fix_bn)
        for name, param in model.named_parameters():
            if param.requires_grad is True:
                logger.info(name)
                param_grad.append(name)
            else:
                param_nograd.append(name)

    logger.info(f'Model:\n{model}')
    datasets = [build_dataset(cfg.data.train)]
    if len(cfg.workflow) == 2:
        val_dataset = copy.deepcopy(cfg.data.val)
        # in case we use a dataset wrapper
        if 'dataset' in cfg.data.train:
            val_dataset.pipeline = cfg.data.train.dataset.pipeline
        else:
            val_dataset.pipeline = cfg.data.train.pipeline
        # set test_mode=False here in deep copied config
        # which do not affect AP/AR calculation later
        # refer to https://mmdetection3d.readthedocs.io/en/latest/tutorials/customize_runtime.html#customize-workflow  # noqa
        val_dataset.test_mode = False
        datasets.append(build_dataset(val_dataset))
    if cfg.checkpoint_config is not None:
        # save mmdet version, config file content and class names in
        # checkpoints as meta data
        cfg.checkpoint_config.meta = dict(
            mmdet_version=__version__,
            config=cfg.pretty_text,
            CLASSES=datasets[0].CLASSES)
    # add an attribute for visualization convenience
    model.CLASSES = datasets[0].CLASSES
    
    if 'load_img_from' in cfg:
        print(cfg.load_img_from)
        checkpoint= torch.load(cfg.load_img_from, map_location='cpu')
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint
        if list(state_dict.keys())[0].startswith('module.'):
            state_dict = {k[7:]: v for k, v in state_dict.items()}
        ckpt = state_dict
        new_ckpt = OrderedDict()
        for k, v in ckpt.items():
            if k.startswith('backbone'):
                new_v = v
                new_k = k.replace('backbone.', 'img_backbone.')
            elif k.startswith('neck'):
                new_v = v
                new_k = k.replace('neck.', 'img_neck.')
            else:
                continue
            new_ckpt[new_k] = new_v
        model.load_state_dict(new_ckpt, strict=False)
    
    if 'load_lift_from' in cfg:
        print(cfg.load_lift_from)
        checkpoint= torch.load(cfg.load_lift_from, map_location='cpu')
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint
        ckpt = state_dict
        new_ckpt = OrderedDict()
        for k, v in ckpt.items():
            if k.startswith('pts_bbox_head'):
                continue
            else:
                new_v = v
                new_k = k
            new_ckpt[new_k] = new_v
        model.load_state_dict(new_ckpt, strict=False)

    train_detector(
        model,
        datasets,
        cfg,
        distributed=distributed,
        validate=(not args.no_validate),
        timestamp=timestamp,
        meta=meta)


if __name__ == '__main__':
    main()
