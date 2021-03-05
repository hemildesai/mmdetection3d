from collections import OrderedDict

import torch
import torch.distributed as dist

from deepspeed.runtime.engine import DeepSpeedEngine
from mmcv.parallel.data_container import DataContainer
from mmcv.parallel.scatter_gather import scatter_kwargs

class DSEngine(DeepSpeedEngine):
    def _parse_losses(self, losses):
        log_vars = OrderedDict()
        for loss_name, loss_value in losses.items():
            if isinstance(loss_value, torch.Tensor):
                log_vars[loss_name] = loss_value.mean()
            elif isinstance(loss_value, list):
                log_vars[loss_name] = sum(_loss.mean() for _loss in loss_value)
            else:
                raise TypeError(
                    f'{loss_name} is not a tensor or list of tensors')

        loss = sum(_value for _key, _value in log_vars.items()
                   if 'loss' in _key)

        log_vars['loss'] = loss
        for loss_name, loss_value in log_vars.items():
            # reduce loss when distributed training
            if dist.is_available() and dist.is_initialized():
                loss_value = loss_value.data.clone()
                dist.all_reduce(loss_value.div_(dist.get_world_size()))
            log_vars[loss_name] = loss_value.item()

        return loss, log_vars

    def scatter_datacontainer(self, inputs, kwargs, device_ids):
        return scatter_kwargs(inputs, kwargs, device_ids, dim=0)

    def train_step(self, *inputs, **kwargs):
        inputs, kwargs = self.scatter_datacontainer(inputs, kwargs, device_ids=[torch.cuda.current_device()])
        data = inputs[0][0]

        losses = self(**data)
        loss, log_vars = self._parse_losses(losses)
        outputs = dict(
            loss=loss, log_vars=log_vars, num_samples=len(data['img_metas']))

        return outputs

    def val_step(self, *inputs, **kwargs):
        inputs, kwargs = self.scatter_datacontainer(inputs, kwargs, device_ids=[torch.cuda.current_device()])
        data = inputs[0][0]

        losses = self(**data)
        loss, log_vars = self._parse_losses(losses)

        outputs = dict(
            loss=loss, log_vars=log_vars, num_samples=len(data['img_metas']))

        return outputs