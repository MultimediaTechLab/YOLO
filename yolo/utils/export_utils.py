
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

        onnx_model_path = f"{Path(self.cfg.weight).stem}.onnx"

        # TODO move duplicated export code also used in fast inference to a separate file
        torch.onnx.export(
            self.model,
            dummy_input,
            onnx_model_path,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes=None #{"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        )

        return onnx_model_path
    def export_flite(self):
        logger.info(f":package: Exporting model to tflite format")
        logger.info(f":construction: Not implemented yet")

    def export_coreml(self):
        logger.info(f":package: Exporting model to coreml format")

        import torch
        dummy_input = torch.ones((1, 3, *self.cfg.image_size))
        
        self.model.eval()
        traced_model = torch.jit.trace(self.model, dummy_input)
        
        import coremltools as ct
        import logging
        logging.getLogger("coremltools").disabled = True
        model_from_trace = ct.convert(
            traced_model,
            inputs=[ct.TensorType(shape=dummy_input.shape)],
            convert_to="neuralnetwork",
        )
        model_from_trace.save(f"{Path(self.cfg.weight).stem}.mlmodel")
        logger.info(f":white_check_mark: Model exported to coreml format")
