# Copyright (c) Open-MMLab. All rights reserved.
import os.path as osp
import platform
import shutil
import time
import warnings

import torch

import mmcv
from mmcv.runner import EpochBasedRunner, save_checkpoint, get_host_info, RUNNERS
import numpy as np
from random import choice
import random
from .epoch_based_runneramp import EpochBasedRunnerAmp
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
np.random.seed(0)
random.seed(0)

@RUNNERS.register_module()
class EpochBasedRunnerSuperAmp(EpochBasedRunnerAmp):
    """Epoch-based Runner.

    This runner train models epoch by epoch.
    """
    def __init__(self,
                 panas_type=0,
                 panas_c_range=[128, 256],
                 panas_d_range=[3, 12],
                 head_d_range = [1, 3],
                 cb_type = 3,
                 cb_step = 4,
                 c_interval = 16,
                 search_backbone=True,
                 search_neck=False,
                 search_head=False,
                 sandwich=False,
                 **kwargs):
        self.panas_type = panas_type
        self.step = 0
        self.panas_c_range = panas_c_range
        self.panas_d_range = panas_d_range
        self.head_d_range = head_d_range
        self.cb_type = cb_type
        self.cb_step = cb_step
        self.c_interval = c_interval
        self.sandwich = sandwich

        self.search_backbone = search_backbone
        self.search_neck = search_neck
        self.search_head =  search_head

        self.arch = None

        super(EpochBasedRunnerSuperAmp, self).__init__(**kwargs)

        
    def get_cand_arch(self, max_arch=False, min_arch=False):
        arch = {}
        if self.search_neck:
            arch['panas_arch'] = [np.random.randint(self.panas_type) for i in range(self.panas_d_range[1])]
            arch['panas_d'] = np.random.randint(self.panas_d_range[0], self.panas_d_range[1] + 1)
            # arch['panas_d'] = self.panas_d_range[1]
            arch['panas_c'] = np.random.randint(self.panas_c_range[0], self.panas_c_range[
                1] + self.c_interval) // self.c_interval * self.c_interval
            if max_arch:
                arch['panas_d'] = self.panas_d_range[1]
                arch['panas_c'] = self.panas_c_range[1]
            if min_arch:
                arch['panas_d'] = self.panas_d_range[0]
                arch['panas_c'] = self.panas_c_range[0]
        if self.search_head:
            if self.head_d_range:
                # arch['head_step'] = self.head_d_range[1]
                arch['head_step'] = np.random.randint(self.head_d_range[0], self.head_d_range[1] + 1)
                if max_arch:
                    arch['head_step'] = self.head_d_range[1]
                if min_arch:
                    arch['head_step'] = self.head_d_range[0]
            else:
                arch['head_step'] = 1

        if self.search_backbone:
            arch['cb_type'] = np.random.randint(self.cb_type)
            arch['cb_step'] = np.random.randint(1, self.cb_step + 1)
            if max_arch:
                arch['cb_type'] = self.cb_type-1
                arch['cb_step'] = self.cb_step
            if min_arch:
                arch['cb_type'] = 0
                arch['cb_step'] = 1
        return arch

    def set_grad_none(self, **kwargs):
        self.model.module.set_grad_none(**kwargs)

    def train(self, data_loader, **kwargs):
        self.model.train()
        self.mode = 'train'
        self.data_loader = data_loader
        self._max_iters = self._max_epochs * len(self.data_loader)
        self.call_hook('before_train_epoch')
        time.sleep(2)  # Prevent possible deadlock during epoch transition

        for i, data_batch in enumerate(self.data_loader):
            self._inner_iter = i
            self.call_hook('before_train_iter')
            self.arch = self.get_cand_arch()
            self.model.module.set_arch(self.arch, **kwargs)

            if self.sandwich:
                self.archs = []
                self.archs.append(self.get_cand_arch(max_arch=True))
                self.archs.append(self.get_cand_arch(min_arch=True))
                self.archs.append(self.get_cand_arch())
                self.archs.append(self.arch)
                self.model.module.set_archs(self.archs, **kwargs)
                # for idx, arch in enumerate(self.archs):
                #     self.logger.info(f'arch {idx}: {arch}')
            # self.logger.info(f'arch: {self.arch}')
            self.run_iter(data_batch, train_mode=True, **kwargs)
            self.call_hook('after_train_iter')
            self._iter += 1

        self.call_hook('after_train_epoch')
        self._epoch += 1


