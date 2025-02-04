
from yolo.config.config import Config
from yolo.model.yolo import YOLO
from yolo.utils.logger import logger
from pathlib import Path

class ModelExporter():
    def __init__(self, cfg: Config, model: YOLO):
        self.model = model
        self.cfg = cfg
        self.class_num = cfg.dataset.class_num
        self.format = self.cfg.task.format
        if cfg.weight == True:
            cfg.weight = Path("weights") / f"{cfg.model.name}.pt"
        self.model_path = f"{Path(self.cfg.weight).stem}.{self.format}"

    def export_onnx(self):
        logger.info(f":package: Exporting model to onnx format")
        import torch
        dummy_input = torch.ones((1, 3, *self.cfg.image_size))

        # TODO move duplicated export code also used in fast inference to a separate file
        torch.onnx.export(
            self.model,
            dummy_input,
            self.model_path,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        )

    def export_flite(self):
        logger.info(f":package: Exporting model to tflite format")
        logger.info(f":construction: Not implemented yet")

    def export_coreml(self):
        logger.info(f":package: Exporting model to coreml format")
        logger.info(f":construction: Not implemented yet")
        import torch
        dummy_input = torch.ones((1, 3, *self.cfg.image_size))
        traced_model = torch.jit.trace(self.model, dummy_input)
        out = traced_model(dummy_input)
        