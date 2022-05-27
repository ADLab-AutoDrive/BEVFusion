# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import warnings

from mmcv.runner import CheckpointHook, HOOKS, allreduce_params
from mmcv.fileio import FileClient


@HOOKS.register_module()
class CheckpointHook_Autodrive(CheckpointHook):

    def before_run(self, runner):
        if not self.out_dir:
            self.out_dir = runner.work_dir

        self.file_client = FileClient.infer_client(self.file_client_args,
                                                   self.out_dir)

        runner.logger.info((f'Checkpoints will be saved to {self.out_dir} by '
                            f'{self.file_client.name}.'))

        # disable the create_symlink option because some file backends do not
        # allow to create a symlink
        if 'create_symlink' in self.args:
            if self.args[
                    'create_symlink'] and not self.file_client.allow_symlink:
                self.args['create_symlink'] = False
                warnings.warn(
                    ('create_symlink is set as True by the user but is changed'
                     'to be False because creating symbolic link is not '
                     f'allowed in {self.file_client.name}'))
        else:
            self.args['create_symlink'] = self.file_client.allow_symlink

    def after_epoch(self, runner):
        if runner.mode != 'train':
            return
        cur_lr = runner.current_lr()
        if isinstance(cur_lr, list):
            cur_lr = cur_lr[0]
        loss = runner.outputs['loss']
        epoch = runner.epoch + 1
        checkpoint_name = 'epoch_{}.pth'.format(epoch)
        r ={
            'step': epoch,
            'weights': checkpoint_name,
            'lr': cur_lr,
            'loss': loss
        }
        rank, _ = get_dist_info()
        if rank == 0:
            with open("/result/iterations", "a") as result_file:
                result_file.write("{}\n".format(";".join(["{}:{}".format(k, v) for k, v in r.items()])))

