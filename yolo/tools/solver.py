from math import ceil
from pathlib import Path

from lightning import LightningModule
from torchmetrics.detection import MeanAveragePrecision

from yolo.config.config import Config
from yolo.model.yolo import create_model
from yolo.tools.data_loader import create_dataloader
from yolo.tools.drawer import draw_bboxes
from yolo.tools.loss_functions import create_loss_function
from yolo.utils.bounding_box_utils import create_converter, to_metrics_format
from yolo.utils.deploy_utils import FastModelLoader
from yolo.utils.export_utils import ModelExporter
from yolo.utils.logger import logger
from yolo.utils.model_utils import PostProcess, create_optimizer, create_scheduler


class BaseModel(LightningModule):
    def __init__(self, cfg: Config, export_mode: bool = False):
        super().__init__()
        self.model = create_model(cfg.model, class_num=cfg.dataset.class_num, weight_path=cfg.weight, export_mode=export_mode)

    def forward(self, x):
        return self.model(x)


class ValidateModel(BaseModel):
    def __init__(self, cfg: Config):
        super().__init__(cfg)
        self.cfg = cfg
        if self.cfg.task.task == "validation":
            self.validation_cfg = self.cfg.task
        else:
            self.validation_cfg = self.cfg.task.validation
        self.metric = MeanAveragePrecision(iou_type="bbox", box_format="xyxy", backend="faster_coco_eval")
        self.metric.warn_on_many_detections = False
        self.val_loader = create_dataloader(self.validation_cfg.data, self.cfg.dataset, self.validation_cfg.task)
        self.ema = self.model

    def setup(self, stage):
        self.vec2box = create_converter(
            self.cfg.model.name, self.model, self.cfg.model.anchor, self.cfg.image_size, self.device
        )
        self.post_process = PostProcess(self.vec2box, self.validation_cfg.nms)

    def val_dataloader(self):
        return self.val_loader

    def validation_step(self, batch, batch_idx):
        batch_size, images, targets, rev_tensor, img_paths = batch
        H, W = images.shape[2:]
        predicts = self.post_process(self.ema(images), image_size=[W, H])
        mAP = self.metric(
            [to_metrics_format(predict) for predict in predicts], [to_metrics_format(target) for target in targets]
        )
        return predicts, mAP

    def on_validation_epoch_end(self):
        epoch_metrics = self.metric.compute()
        del epoch_metrics["classes"]
        self.log_dict(epoch_metrics, prog_bar=True, sync_dist=True, rank_zero_only=True)
        self.log_dict(
            {"PyCOCO/AP @ .5:.95": epoch_metrics["map"], "PyCOCO/AP @ .5": epoch_metrics["map_50"]},
            sync_dist=True,
            rank_zero_only=True,
        )
        self.metric.reset()


class TrainModel(ValidateModel):
    def __init__(self, cfg: Config):
        super().__init__(cfg)
        self.cfg = cfg
        self.train_loader = create_dataloader(self.cfg.task.data, self.cfg.dataset, self.cfg.task.task)

    def setup(self, stage):
        super().setup(stage)
        self.loss_fn = create_loss_function(self.cfg, self.vec2box)

    def train_dataloader(self):
        return self.train_loader

    def on_train_epoch_start(self):
        self.trainer.optimizers[0].next_epoch(
            ceil(len(self.train_loader) / self.trainer.world_size), self.current_epoch
        )
        self.vec2box.update(self.cfg.image_size)

    def training_step(self, batch, batch_idx):
        lr_dict = self.trainer.optimizers[0].next_batch()
        batch_size, images, targets, *_ = batch
        predicts = self(images)
        aux_predicts = self.vec2box(predicts["AUX"])
        main_predicts = self.vec2box(predicts["Main"])
        loss, loss_item = self.loss_fn(aux_predicts, main_predicts, targets)
        self.log_dict(
            loss_item,
            prog_bar=True,
            on_epoch=True,
            batch_size=batch_size,
            rank_zero_only=True,
        )
        self.log_dict(lr_dict, prog_bar=False, logger=True, on_epoch=False, rank_zero_only=True)
        return loss * batch_size

    def configure_optimizers(self):
        optimizer = create_optimizer(self.model, self.cfg.task.optimizer)
        scheduler = create_scheduler(optimizer, self.cfg.task.scheduler)
        return [optimizer], [scheduler]


class InferenceModel(BaseModel):
    def __init__(self, cfg: Config):
        if hasattr(cfg.model.model, "auxiliary"):
            cfg.model.model.auxiliary = {}

        export_mode = False
        fast_inference = cfg.task.fast_inference
        # TODO check if we can use export mode for all formats
        if fast_inference == "coreml":
            export_mode = True

        super().__init__(cfg, export_mode=export_mode)
        self.cfg = cfg
        self.predict_loader = create_dataloader(cfg.task.data, cfg.dataset, cfg.task.task)

    def setup(self, stage):
        self.vec2box = create_converter(
            self.cfg.model.name, self.model, self.cfg.model.anchor, self.cfg.image_size, self.device
        )

        if self.cfg.task.fast_inference:
            self.fast_model = FastModelLoader(self.cfg, self.model).load_model(self.device)

        self.post_process = PostProcess(self.vec2box, self.cfg.task.nms)

    def predict_dataloader(self):
        return self.predict_loader

    def predict_step(self, batch, batch_idx):
        images, rev_tensor, origin_frame = batch
        if hasattr(self, "fast_model") and self.fast_model:
            predictions = self.fast_model(images)
        else:
            predictions = self(images)
        predicts = self.post_process(predictions, rev_tensor=rev_tensor)
        img = draw_bboxes(origin_frame, predicts, idx2label=self.cfg.dataset.class_list)
        if getattr(self.predict_loader, "is_stream", None):
            fps = self._display_stream(img)
        else:
            fps = None
        if getattr(self.cfg.task, "save_predict", None):
            self._save_image(img, batch_idx)
        return img, fps

    def _save_image(self, img, batch_idx):
        save_image_path = Path(self.trainer.default_root_dir) / f"frame{batch_idx:03d}.png"
        img.save(save_image_path)
        print(f"💾 Saved visualize image at {save_image_path}")


class ExportModel(BaseModel):
    def __init__(self, cfg: Config):
        if hasattr(cfg.model.model, "auxiliary"):
            cfg.model.model.auxiliary = {}

        export_mode = False
        format = cfg.task.format
        # TODO check if we can use export mode for all formats
        if self.format == "coreml":
            export_mode = True

        super().__init__(cfg, export_mode=export_mode)
        self.cfg = cfg
        self.format = format
        self.model_exporter = ModelExporter(self.cfg, self.model, format=self.format)

    def export(self):
        if self.format == "onnx":
            self.model_exporter.export_onnx()
        if self.format == "tflite":
            self.model_exporter.export_tflite()
        if self.format == "coreml":
            self.model_exporter.export_coreml()
