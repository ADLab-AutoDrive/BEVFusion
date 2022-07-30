# Copyright (c) Open-MMLab. All rights reserved.
import copy

from torch.nn.utils import clip_grad


from mmcv.runner import OptimizerHook, HOOKS, Fp16OptimizerHook

@HOOKS.register_module()
class OptimizerHookSuper(OptimizerHook):

    def after_train_iter(self, runner):
        # runner.optimizer.zero_grad(set_to_none=True)
        runner.optimizer.zero_grad()
        runner.outputs['loss'].backward()
        for p in runner.model.parameters():
            if p.grad is not None and p.grad.sum() == 0:
                p.grad = None

        if self.grad_clip is not None:
            grad_norm = self.clip_grads(runner.model.parameters())
            if grad_norm is not None:
                # Add grad norm to the logger
                runner.log_buffer.update({'grad_norm': float(grad_norm)},
                                         runner.outputs['num_samples'])

        runner.optimizer.step()

@HOOKS.register_module()
class Fp16OptimizerHookSuper(Fp16OptimizerHook):

    def after_train_iter(self, runner):

        # clear grads of last iteration
        runner.model.zero_grad(set_to_none=True)
        runner.optimizer.zero_grad(set_to_none=True)

        self.loss_scaler.scale(runner.outputs['loss']).backward()
        self.loss_scaler.unscale_(runner.optimizer)



        # grad clip
        if self.grad_clip is not None:
            grad_norm = self.clip_grads(runner.model.parameters())
            if grad_norm is not None:
                # Add grad norm to the logger
                runner.log_buffer.update({'grad_norm': float(grad_norm)},
                                         runner.outputs['num_samples'])
        # backward and update scaler
        self.loss_scaler.step(runner.optimizer)
        self.loss_scaler.update(self._scale_update_param)

        # save state_dict of loss_scaler
        runner.meta.setdefault(
            'fp16', {})['loss_scaler'] = self.loss_scaler.state_dict()






