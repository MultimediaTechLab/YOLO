import os
from copy import deepcopy
from math import exp
from pathlib import Path
from typing import List, Optional, Type, Union

import torch
import torch.distributed as dist
from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks import Callback
from omegaconf import ListConfig
from torch import Tensor, no_grad
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR, SequentialLR, _LRScheduler

from yolo.config.config import IDX_TO_ID, NMSConfig, OptimizerConfig, SchedulerConfig
from yolo.model.yolo import YOLO
from yolo.utils.bounding_box_utils import Anc2Box, Vec2Box, bbox_nms, transform_bbox
from yolo.utils.logger import logger


def lerp(start: float, end: float, step: Union[int, float], total: int = 1):
    """
    Linearly interpolates between start and end values.

    start * (1 - step) + end * step

    Parameters:
        start (float): The starting value.
        end (float): The ending value.
        step (int): The current step in the interpolation process.
        total (int): The total number of steps.

    Returns:
        float: The interpolated value.

    Example:
        >>> # from big to small
        >>> lerp(1, 0, 10, 100)
        0.9
        >>> # from small to big
        >>> lerp(0, 1, 10, 100)
        0.1
    """
    return start + (end - start) * step / total


class EMA(Callback):
    def __init__(self, decay: float = 0.9999, tau: float = 2000):
        super().__init__()
        logger.info(":chart_with_upwards_trend: Enable Model EMA")
        self.decay = decay
        self.tau = tau
        self.step = 0
        self.ema_state_dict = None

    def setup(self, trainer, pl_module, stage):
        pl_module.ema = deepcopy(pl_module.model)
        self.tau /= trainer.world_size
        for param in pl_module.ema.parameters():
            param.requires_grad = False

    def on_validation_start(self, trainer: "Trainer", pl_module: "LightningModule"):
        if self.ema_state_dict is None:
            self.ema_state_dict = deepcopy(pl_module.model.state_dict())
        pl_module.ema.load_state_dict(self.ema_state_dict)

    @no_grad()
    def on_train_batch_start(self, trainer: "Trainer", pl_module: "LightningModule", *args, **kwargs) -> None:
        if self.ema_state_dict is None:
            # If validation sanity checks are disabled, then we need to
            # initialize the ema state before training starts.
            self.ema_state_dict = deepcopy(pl_module.model.state_dict())
            pl_module.ema.load_state_dict(self.ema_state_dict)

    @no_grad()
    def on_train_batch_end(self, trainer: "Trainer", pl_module: "LightningModule", *args, **kwargs) -> None:
        self.step += 1
        decay_factor = self.decay * (1 - exp(-self.step / self.tau))
        for key, param in pl_module.model.state_dict().items():
            self.ema_state_dict[key] = lerp(param.detach(), self.ema_state_dict[key], decay_factor)


def create_optimizer(model: YOLO, optim_cfg: OptimizerConfig) -> Optimizer:
    """Create an optimizer for the given model parameters based on the configuration.

    Returns:
        An instance of the optimizer configured according to the provided settings.
    """
    optimizer_class: Type[Optimizer] = getattr(torch.optim, optim_cfg.type)
    optim_args = optim_cfg.args

    # Note the arguments that the optimizer class actually accepts
    valid_optim_args = set(optim_args.keys())

    named_params = dict(model.named_parameters())

    named_groups = {}

    # Define groups that will have their optimizer params overwritten
    # (NOTE: these params are valid for SGD, but may not be for other
    # optimizers)
    named_groups['bias'] = {name for name in named_params if 'bias' in name}
    named_groups['norm'] = {name for name in named_params if "weight" in name and "bn" in name}
    named_groups['conv'] = {name for name in named_params if "weight" in name and "bn" not in name}

    if __debug__:
        import itertools as it
        # Check that all groups are disjoint
        for g1, g2 in it.combinations(named_groups.values(), 2):
            assert len(g1 & g2) == 0
        # Check that all parmeters are in a group
        used = set.union(*named_groups.values())
        all_param_name = set(named_params.keys())
        unused = all_param_name - used
        assert len(unused) == 0

    named_group_overrides = {
        'bias': {"momentum": 0.937, "weight_decay": 0},
        'conv': {"momentum": 0.937},
        'norm': {"momentum": 0.937, "weight_decay": 0},
    }

    # Remove any of the overrides that are valid arguments to the optimizer.
    named_group_overrides = {
        name: {k: v for k, v in overrides.items() if k in valid_optim_args}
        for name, overrides in named_group_overrides.items()
    }

    # Map the group names to the parameter objects
    named_groups = {
        group_name: [named_params[name] for name in param_names]
        for group_name, param_names in named_groups.items()
    }

    # Setup the input to standard torch optimizers
    param_groups = [
        {"name": name, "params": params, **named_group_overrides[name]}
        for name, params in named_groups.items()
    ]

    # TODO: load momentum from config instead a fix number
    warmup_schedule = {
        'momentum': {
            'start': 0.8,     # Start Momemtum
            'normal': 0.937,  # Normal Momemtum
            'peak_epoch': 3   # The warm up epoch num
        }
    }

    def next_epoch(self, batch_num, epoch_idx):
        """
        Args:
            batch_num (int): the number of batches in the epoch
            epoch_id (int): the epoch index
        """
        self.min_lr = self.max_lr
        self.max_lr = {
            group['name']: group["lr"]
            for group in self.param_groups
        }
        if 'momentum' in valid_optim_args:
            mom0 = warmup_schedule['momentum']['start']
            mom1 = warmup_schedule['momentum']['normal']
            peak_epoch = warmup_schedule['momentum']['peak_epoch']
            self.min_mom = lerp(mom0, mom1, min(epoch_idx, peak_epoch), peak_epoch)
            self.max_mom = lerp(mom0, mom1, min(epoch_idx + 1, peak_epoch), peak_epoch)
        self.batch_num = batch_num
        self.batch_idx = 0

    def next_batch(self):
        self.batch_idx += 1
        lr_dict = dict()
        for param_group in self.param_groups:
            group_name = param_group['name']
            min_lr, max_lr = self.min_lr[group_name], self.max_lr[group_name]
            param_group["lr"] = lerp(min_lr, max_lr, self.batch_idx, self.batch_num)
            lr_dict[f"LR/{group_name}"] = param_group["lr"]
            if "momentum" in valid_optim_args:
                param_group["momentum"] = lerp(self.min_mom, self.max_mom, self.batch_idx, self.batch_num)
                lr_dict[f"momentum/{group_name}"] = param_group["momentum"]
        return lr_dict

    # Monkey patch in methods/attributes for more control over the schedule.
    optimizer_class.next_batch = next_batch
    optimizer_class.next_epoch = next_epoch
    optimizer = optimizer_class(param_groups, **optim_args)
    optimizer.max_lr = {
        'bias': 0.1,
        'conv': 0,
        'norm': 0,
    }

    if 0:
        # Test the schedule.
        import ubelt as ub
        batch_num = 3
        for epoch_idx in range(3):
            optimizer.next_epoch(batch_num, epoch_idx)
            ignore_names = {'defaults', 'param_groups', 'state'}
            ignore_names |= {k for k in optimizer.__dict__.keys() if k.startswith('_')}
            optim_dict = {k: v for k, v in optimizer.__dict__.items() if k not in ignore_names}
            print(f'optim_dict = {ub.urepr(optim_dict, nl=1)}')

            for _ in range(batch_num):
                lr_dict = optimizer.next_batch()
                print(f'lr_dict = {ub.urepr(lr_dict, nl=0)}')

    return optimizer


def create_scheduler(optimizer: Optimizer, schedule_cfg: SchedulerConfig) -> _LRScheduler:
    """Create a learning rate scheduler for the given optimizer based on the configuration.

    Returns:
        An instance of the scheduler configured according to the provided settings.
    """
    scheduler_class: Type[_LRScheduler] = getattr(torch.optim.lr_scheduler, schedule_cfg.type)
    schedule = scheduler_class(optimizer, **schedule_cfg.args)
    if hasattr(schedule_cfg, "warmup"):
        wepoch = schedule_cfg.warmup.epochs
        lambda1 = lambda epoch: (epoch + 1) / wepoch if epoch < wepoch else 1
        lambda2 = lambda epoch: 10 - 9 * ((epoch + 1) / wepoch) if epoch < wepoch else 1
        warmup_schedule = LambdaLR(optimizer, lr_lambda=[lambda2, lambda1, lambda1])
        schedule = SequentialLR(optimizer, schedulers=[warmup_schedule, schedule], milestones=[wepoch - 1])
    return schedule


def initialize_distributed() -> None:
    rank = int(os.getenv("RANK", "0"))
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    world_size = int(os.getenv("WORLD_SIZE", "1"))

    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    logger.info(f"🔢 Initialized process group; rank: {rank}, size: {world_size}")
    return local_rank


def get_device(device_spec: Union[str, int, List[int]]) -> torch.device:
    ddp_flag = False
    if isinstance(device_spec, (list, ListConfig)):
        ddp_flag = True
        device_spec = initialize_distributed()
    if torch.cuda.is_available() and "cuda" in str(device_spec):
        return torch.device(device_spec), ddp_flag
    if not torch.cuda.is_available():
        if device_spec != "cpu":
            logger.warning(f"❎ Device spec: {device_spec} not support, Choosing CPU instead")
        return torch.device("cpu"), False

    device = torch.device(device_spec)
    return device, ddp_flag


class PostProcess:
    """
    TODO: function document
    scale back the prediction and do nms for pred_bbox
    """

    def __init__(self, converter: Union[Vec2Box, Anc2Box], nms_cfg: NMSConfig) -> None:
        self.converter = converter
        self.nms = nms_cfg

    def __call__(
        self, predict, rev_tensor: Optional[Tensor] = None, image_size: Optional[List[int]] = None
    ) -> List[Tensor]:
        if image_size is not None:
            self.converter.update(image_size)
        prediction = self.converter(predict["Main"])
        pred_class, _, pred_bbox = prediction[:3]
        pred_conf = prediction[3] if len(prediction) == 4 else None
        if rev_tensor is not None:
            pred_bbox = (pred_bbox - rev_tensor[:, None, 1:]) / rev_tensor[:, 0:1, None]
        pred_bbox = bbox_nms(pred_class, pred_bbox, self.nms, pred_conf)
        return pred_bbox


def collect_prediction(predict_json: List, local_rank: int) -> List:
    """
    Collects predictions from all distributed processes and gathers them on the main process (rank 0).

    Args:
        predict_json (List): The prediction data (can be of any type) generated by the current process.
        local_rank (int): The rank of the current process. Typically, rank 0 is the main process.

    Returns:
        List: The combined list of predictions from all processes if on rank 0, otherwise predict_json.
    """
    if dist.is_initialized() and local_rank == 0:
        all_predictions = [None for _ in range(dist.get_world_size())]
        dist.gather_object(predict_json, all_predictions, dst=0)
        predict_json = [item for sublist in all_predictions for item in sublist]
    elif dist.is_initialized():
        dist.gather_object(predict_json, None, dst=0)
    return predict_json


def predicts_to_json(img_paths, predicts, rev_tensor):
    """
    TODO: function document
    turn a batch of imagepath and predicts(n x 6 for each image) to a List of diction(Detection output)
    """
    batch_json = []
    for img_path, bboxes, box_reverse in zip(img_paths, predicts, rev_tensor):
        scale, shift = box_reverse.split([1, 4])
        bboxes = bboxes.clone()
        bboxes[:, 1:5] = (bboxes[:, 1:5] - shift[None]) / scale[None]
        bboxes[:, 1:5] = transform_bbox(bboxes[:, 1:5], "xyxy -> xywh")
        for cls, *pos, conf in bboxes:
            bbox = {
                "image_id": int(Path(img_path).stem),
                "category_id": IDX_TO_ID[int(cls)],
                "bbox": [float(p) for p in pos],
                "score": float(conf),
            }
            batch_json.append(bbox)
    return batch_json
