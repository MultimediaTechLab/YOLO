"""
Module for initializing logging tools used in machine learning and data processing.
Supports integration with Weights & Biases (wandb), Loguru, TensorBoard, and other
logging frameworks as needed.

This setup ensures consistent logging across various platforms, facilitating
effective monitoring and debugging.

Example:
    from tools.logger import custom_logger
    custom_logger()
"""

import logging
from collections import deque
from logging import FileHandler
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import wandb
from lightning import LightningModule, Trainer, seed_everything
from lightning.pytorch.callbacks import Callback, RichModelSummary, RichProgressBar
from lightning.pytorch.callbacks.progress.rich_progress import CustomProgress
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger
from lightning.pytorch.utilities import rank_zero_only
from omegaconf import ListConfig
from rich import get_console, reconfigure
from rich.console import Console, Group
from rich.logging import RichHandler
from rich.table import Table
from rich.text import Text
from torch import Tensor
from torch.nn import ModuleList
from typing_extensions import override

from yolo.config.config import Config, YOLOLayer
from yolo.model.yolo import YOLO
from yolo.utils.logger import logger
from yolo.utils.model_utils import EMA
from yolo.utils.solver_utils import make_ap_table


# TODO: should be moved to correct position
def set_seed(seed):
    seed_everything(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    # torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class YOLOCustomProgress(CustomProgress):
    def get_renderable(self):
        renderable = Group(*self.get_renderables())
        if hasattr(self, "table"):
            renderable = Group(*self.get_renderables(), self.table)
        return renderable


class YOLORichProgressBar(RichProgressBar):
    @override
    @rank_zero_only
    def _init_progress(self, trainer: "Trainer") -> None:
        if self.is_enabled and (self.progress is None or self._progress_stopped):
            self._reset_progress_bar_ids()
            reconfigure(**self._console_kwargs)
            self._console = Console()
            self._console.clear_live()
            self.progress = YOLOCustomProgress(
                *self.configure_columns(trainer),
                auto_refresh=False,
                disable=self.is_disabled,
                console=self._console,
            )
            self.progress.start()

            self._progress_stopped = False

            self.max_result = 0
            self.past_results = deque(maxlen=5)
            self.progress.table = Table()

    @override
    def _get_train_description(self, current_epoch: int) -> str:
        return Text("[cyan]Train [white]|")

    @override
    @rank_zero_only
    def on_train_start(self, trainer, pl_module):
        self._init_progress(trainer)
        num_epochs = trainer.max_epochs - 1
        self.task_epoch = self._add_task(
            total_batches=num_epochs,
            description=f"[cyan]Start Training {num_epochs} epochs",
        )
        self.max_result = 0
        self.past_results.clear()

    @override
    @rank_zero_only
    def on_train_batch_end(self, trainer, pl_module, outputs, batch: Any, batch_idx: int):
        self._update(self.train_progress_bar_id, batch_idx + 1)
        self._update_metrics(trainer, pl_module)
        epoch_descript = "[cyan]Train [white]|"
        batch_descript = "[green]Train [white]|"
        metrics = self.get_metrics(trainer, pl_module)
        metrics.pop("v_num")
        for metrics_name, metrics_val in metrics.items():
            if "Loss_step" in metrics_name:
                epoch_descript += f"{metrics_name.removesuffix('_step').split('/')[1]: ^9}|"
                batch_descript += f"   {metrics_val:2.2f}  |"

        self.progress.update(self.task_epoch, advance=1 / self.total_train_batches, description=epoch_descript)
        self.progress.update(self.train_progress_bar_id, description=batch_descript)
        self.refresh()

    @override
    @rank_zero_only
    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx) -> None:
        if self.is_disabled:
            return
        if trainer.sanity_checking:
            self._update(self.val_sanity_progress_bar_id, batch_idx + 1)
        elif self.val_progress_bar_id is not None:
            self._update(self.val_progress_bar_id, batch_idx + 1)
            _, mAP = outputs
            mAP_desc = f" mAP :{mAP['map']*100:6.2f} | mAP50 :{mAP['map_50']*100:6.2f} |"
            self.progress.update(self.val_progress_bar_id, description=f"[green]Valid [white]|{mAP_desc}")
        self.refresh()

    @override
    @rank_zero_only
    def on_train_end(self, trainer: "Trainer", pl_module: "LightningModule") -> None:
        self._update_metrics(trainer, pl_module)
        self.progress.remove_task(self.train_progress_bar_id)
        self.train_progress_bar_id = None

    @override
    @rank_zero_only
    def on_validation_end(self, trainer: "Trainer", pl_module: "LightningModule") -> None:
        if trainer.state.fn == "fit":
            self._update_metrics(trainer, pl_module)
        self.reset_dataloader_idx_tracker()
        all_metrics = self.get_metrics(trainer, pl_module)

        ap_ar_list = [
            key
            for key in all_metrics.keys()
            if key.startswith(("map", "mar")) and not key.endswith(("_step", "_epoch"))
        ]
        score = np.array([all_metrics[key] for key in ap_ar_list]) * 100

        self.progress.table, ap_main = make_ap_table(score, self.past_results, self.max_result, trainer.current_epoch)
        self.max_result = np.maximum(score, self.max_result)
        self.past_results.append((trainer.current_epoch, ap_main))

    @override
    def refresh(self) -> None:
        if self.progress:
            self.progress.refresh()

    @property
    def validation_description(self) -> str:
        return "[green]Validation"


class YOLORichModelSummary(RichModelSummary):
    @staticmethod
    @override
    def summarize(
        summary_data: List[Tuple[str, List[str]]],
        total_parameters: int,
        trainable_parameters: int,
        model_size: float,
        total_training_modes: Dict[str, int],
        **summarize_kwargs: Any,
    ) -> None:
        from lightning.pytorch.utilities.model_summary import get_human_readable_count

        console = get_console()

        header_style: str = summarize_kwargs.get("header_style", "bold magenta")
        table = Table(header_style=header_style)
        table.add_column(" ", style="dim")
        table.add_column("Name", justify="left", no_wrap=True)
        table.add_column("Type")
        table.add_column("Params", justify="right")
        table.add_column("Mode")

        column_names = list(zip(*summary_data))[0]

        for column_name in ["In sizes", "Out sizes"]:
            if column_name in column_names:
                table.add_column(column_name, justify="right", style="white")

        rows = list(zip(*(arr[1] for arr in summary_data)))
        for row in rows:
            table.add_row(*row)

        console.print(table)

        parameters = []
        for param in [trainable_parameters, total_parameters - trainable_parameters, total_parameters, model_size]:
            parameters.append("{:<{}}".format(get_human_readable_count(int(param)), 10))

        grid = Table(header_style=header_style)
        table.add_column(" ", style="dim")
        grid.add_column("[bold]Attributes[/]")
        grid.add_column("Value")

        grid.add_row("[bold]Trainable params[/]", f"{parameters[0]}")
        grid.add_row("[bold]Non-trainable params[/]", f"{parameters[1]}")
        grid.add_row("[bold]Total params[/]", f"{parameters[2]}")
        grid.add_row("[bold]Total estimated model params size (MB)[/]", f"{parameters[3]}")
        grid.add_row("[bold]Modules in train mode[/]", f"{total_training_modes['train']}")
        grid.add_row("[bold]Modules in eval mode[/]", f"{total_training_modes['eval']}")

        console.print(grid)


class ImageLogger(Callback):
    def on_validation_batch_end(self, trainer: Trainer, pl_module, outputs, batch, batch_idx) -> None:
        if batch_idx != 0:
            return
        batch_size, images, targets, rev_tensor, img_paths = batch
        predicts, _ = outputs
        gt_boxes = targets[0] if targets.ndim == 3 else targets
        pred_boxes = predicts[0] if isinstance(predicts, list) else predicts
        images = [images[0]]
        step = trainer.current_epoch
        for logger in trainer.loggers:
            if isinstance(logger, WandbLogger):
                logger.log_image("Input Image", images, step=step)
                logger.log_image("Ground Truth", images, step=step, boxes=[log_bbox(gt_boxes)])
                logger.log_image("Prediction", images, step=step, boxes=[log_bbox(pred_boxes)])


def setup_logger(logger_name, quite=False):
    class EmojiFormatter(logging.Formatter):
        def format(self, record, emoji=":high_voltage:"):
            return f"{emoji} {super().format(record)}"

    rich_handler = RichHandler(markup=True)
    rich_handler.setFormatter(EmojiFormatter("%(message)s"))
    rich_logger = logging.getLogger(logger_name)
    if rich_logger:
        rich_logger.handlers.clear()
        rich_logger.addHandler(rich_handler)
        if quite:
            rich_logger.setLevel(logging.ERROR)

    coco_logger = logging.getLogger("faster_coco_eval.core.cocoeval")
    coco_logger.setLevel(logging.ERROR)


def setup(cfg: Config):
    quite = hasattr(cfg, "quite")
    setup_logger("lightning.fabric", quite=quite)
    setup_logger("lightning.pytorch", quite=quite)

    def custom_wandb_log(string="", level=int, newline=True, repeat=True, prefix=True, silent=False):
        if silent:
            return
        for line in string.split("\n"):
            logger.info(Text.from_ansi(":globe_with_meridians: " + line))

    wandb.errors.term._log = custom_wandb_log

    save_path = validate_log_directory(cfg, cfg.name)

    progress, loggers = [], []

    if hasattr(cfg.task, "ema") and cfg.task.ema.enable:
        progress.append(EMA(cfg.task.ema.decay))
    if quite:
        logger.setLevel(logging.ERROR)
        return progress, loggers, save_path

    progress.append(YOLORichProgressBar())
    progress.append(YOLORichModelSummary())
    progress.append(ImageLogger())
    if cfg.use_tensorboard:
        loggers.append(TensorBoardLogger(log_graph="all", save_dir=save_path))
    if cfg.use_wandb:
        loggers.append(WandbLogger(project="YOLO", name=cfg.name, save_dir=save_path, id=None))

    return progress, loggers, save_path


def log_model_structure(model: Union[ModuleList, YOLOLayer, YOLO]):
    if isinstance(model, YOLO):
        model = model.model
    console = Console()
    table = Table(title="Model Layers")

    table.add_column("Index", justify="center")
    table.add_column("Layer Type", justify="center")
    table.add_column("Tags", justify="center")
    table.add_column("Params", justify="right")
    table.add_column("Channels (IN->OUT)", justify="center")

    for idx, layer in enumerate(model, start=1):
        layer_param = sum(x.numel() for x in layer.parameters())  # number parameters
        in_channels, out_channels = getattr(layer, "in_c", None), getattr(layer, "out_c", None)
        if in_channels and out_channels:
            if isinstance(in_channels, (list, ListConfig)):
                in_channels = "M"
            if isinstance(out_channels, (list, ListConfig)):
                out_channels = "M"
            channels = f"{str(in_channels): >4} -> {str(out_channels): >4}"
        else:
            channels = "-"
        table.add_row(str(idx), layer.layer_type, layer.tags, f"{layer_param:,}", channels)
    console.print(table)


@rank_zero_only
def validate_log_directory(cfg: Config, exp_name: str) -> Path:
    base_path = Path(cfg.out_path, cfg.task.task)
    save_path = base_path / exp_name

    if not cfg.exist_ok:
        index = 1
        old_exp_name = exp_name
        while save_path.is_dir():
            exp_name = f"{old_exp_name}{index}"
            save_path = base_path / exp_name
            index += 1
        if index > 1:
            logger.opt(colors=True).warning(
                f"🔀 Experiment directory exists! Changed <red>{old_exp_name}</> to <green>{exp_name}</>"
            )

    save_path.mkdir(parents=True, exist_ok=True)
    if not getattr(cfg, "quite", False):
        logger.info(f"📄 Created log folder: [blue b u]{save_path}[/]")
    logger.addHandler(FileHandler(save_path / "output.log"))
    return save_path


def log_bbox(
    bboxes: Tensor, class_list: Optional[List[str]] = None, image_size: Tuple[int, int] = (640, 640)
) -> List[dict]:
    """
    Convert bounding boxes tensor to a list of dictionaries for logging, normalized by the image size.

    Args:
        bboxes (Tensor): Bounding boxes with shape (N, 5) or (N, 6), where each box is [class_id, x_min, y_min, x_max, y_max, (confidence)].
        class_list (Optional[List[str]]): List of class names. Defaults to None.
        image_size (Tuple[int, int]): The size of the image, used for normalization. Defaults to (640, 640).

    Returns:
        List[dict]: List of dictionaries containing normalized bounding box information.
    """
    bbox_list = []
    scale_tensor = torch.Tensor([1, *image_size, *image_size]).to(bboxes.device)
    normalized_bboxes = bboxes[:, :5] / scale_tensor
    for bbox in normalized_bboxes:
        class_id, x_min, y_min, x_max, y_max, *conf = [float(val) for val in bbox]
        if class_id == -1:
            break
        bbox_entry = {
            "position": {"minX": x_min, "maxX": x_max, "minY": y_min, "maxY": y_max},
            "class_id": int(class_id),
        }
        if class_list:
            bbox_entry["box_caption"] = class_list[int(class_id)]
        if conf:
            bbox_entry["scores"] = {"confidence": conf[0]}
        bbox_list.append(bbox_entry)

    return {"predictions": {"box_data": bbox_list}}
