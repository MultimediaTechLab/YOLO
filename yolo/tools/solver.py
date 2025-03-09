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
from yolo.utils.model_utils import PostProcess, create_optimizer, create_scheduler


class BaseModel(LightningModule):
    def __init__(self, cfg: Config):
        super().__init__()
        # TODO: refactor to know what the class labels actually are instead of
        # just the number.
        self.model = create_model(cfg.model, class_num=cfg.dataset.class_num, weight_path=cfg.weight)

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
        raw_predicts = self.ema(images)
        predicts = self.post_process(raw_predicts, image_size=[W, H])
        mAP = self.metric(
            [to_metrics_format(predict) for predict in predicts], [to_metrics_format(target) for target in targets]
        )
        outputs = {
            'predicts': predicts,
            'mAP': mAP,
        }
        return outputs

    def on_validation_epoch_end(self):
        epoch_metrics = self.metric.compute()
        epoch_metrics.pop("classes", None)
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

        # Flag that lets plugins communicate with the model
        self.request_draw = False

        # TODO: if we defer creating the model until the dataset is loaded, we
        # can introspect the number of categories and other things to make user
        # configuration have less interdependencies and thus be more robust.
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
        raw_predicts = self(images)
        aux_predicts = self.vec2box(raw_predicts["AUX"])
        main_predicts = self.vec2box(raw_predicts["Main"])
        loss, loss_item = self.loss_fn(aux_predicts, main_predicts, targets)
        self.log_dict(
            loss_item,
            prog_bar=True,
            on_epoch=True,
            batch_size=batch_size,
            rank_zero_only=True,
        )
        self.log_dict(lr_dict, prog_bar=False, logger=True, on_epoch=False, rank_zero_only=True)
        total_loss = loss * batch_size
        stage = self.trainer.state.stage.value
        self.log(f'{stage}_loss', total_loss, prog_bar=True, batch_size=batch_size)
        output = {}
        output['loss'] = total_loss
        if self.request_draw:
            H, W = images.shape[2:]
            predicts = self.post_process(raw_predicts, image_size=[W, H])
            output['predicts'] = predicts
        return output

    def configure_optimizers(self):
        optimizer = create_optimizer(self.model, self.cfg.task.optimizer)
        scheduler = create_scheduler(optimizer, self.cfg.task.scheduler)
        return [optimizer], [scheduler]


class InferenceModel(BaseModel):
    def __init__(self, cfg: Config):
        super().__init__(cfg)
        self.cfg = cfg
        # TODO: Add FastModel
        self.predict_loader = create_dataloader(cfg.task.data, cfg.dataset, cfg.task.task)

        print(f'self.predict_loader._is_coco={self.predict_loader._is_coco}')
        if self.predict_loader._is_coco:
            # Setup a kwcoco file to write to if the user requests it.
            self.pred_dset = self.predict_loader.coco_dset.copy()
            self.pred_dset.reroot(absolute=True)

    def on_predict_start(self, *args, **kwargs):
        import rich
        import ubelt as ub
        out_dpath = ub.Path(self.trainer.default_root_dir).absolute()
        rich.print(f'Predict in: [link={out_dpath}]{out_dpath}[/link]')
        if self.predict_loader._is_coco:
            out_coco_fpath = out_dpath / 'pred.kwcoco.zip'
            self.pred_dset.fpath = out_coco_fpath
            rich.print(f'Coco prediction is enabled in: {self.pred_dset.fpath}')

    def on_predict_end(self, *args, **kwargs):
        print('[InferenceModel] on_predict_end')
        dset = self.pred_dset
        print(f'dset.fpath={dset.fpath}')
        dset.dump()
        print('Finished prediction')

    def setup(self, stage):
        self.vec2box = create_converter(
            self.cfg.model.name, self.model, self.cfg.model.anchor, self.cfg.image_size, self.device
        )
        self.post_process = PostProcess(self.vec2box, self.cfg.task.nms)

    def predict_dataloader(self):
        return self.predict_loader

    def predict_step(self, batch, batch_idx):

        images, rev_tensor, origin_frame, metadata = batch

        assert metadata is not None
        img = metadata['img']
        classes = metadata['classes']
        image_id = img['id']
        predicts = self.post_process(self(images), rev_tensor=rev_tensor)

        WRITE_TO_COCO = 1
        if WRITE_TO_COCO:
            from yolo.utils.kwcoco_utils import tensor_to_kwimage
            dset = self.pred_dset
            for yolo_annot_tensor in predicts:
                pred_dets = tensor_to_kwimage(yolo_annot_tensor, classes=classes).numpy()
                anns = list(pred_dets.to_coco(dset=dset))
                print(f"Detected {len(anns)} boxes")
                for ann in anns:
                    ann['image_id'] = image_id
                    dset.add_annotation(**ann)

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
