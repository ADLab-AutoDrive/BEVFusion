from mmcv.runner import HOOKS, Hook
from mmcv.runner import get_dist_info

@HOOKS.register_module()
class AutoDriveHook(Hook):

    def __init__(self, *args, **kwargs):
        pass

    def before_run(self, runner):
        pass

    def after_run(self, runner):
        pass

    def before_epoch(self, runner):
        pass

    def after_epoch(self, runner):
        if runner.mode != 'train':
            return
        cur_lr = runner.current_lr()
        if isinstance(cur_lr, list):
            cur_lr = cur_lr[0]
        # print(runner.log_buffer.output)
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

    def before_iter(self, runner):
        pass

    def after_iter(self, runner):
        pass

    def get_epoch(self, runner):
        if runner.mode == 'train':
            epoch = runner.epoch + 1
        elif runner.mode == 'val':
            # normal val mode
            # runner.epoch += 1 has been done before val workflow
            epoch = runner.epoch
        else:
            raise ValueError(f"runner mode should be 'train' or 'val', "
                             f'but got {runner.mode}')
        return epoch