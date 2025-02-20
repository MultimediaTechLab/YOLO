
from typing import Dict, List, Optional
from yolo.config.config import Config
from yolo.model.yolo import YOLO
from yolo.utils.logger import logger
from pathlib import Path

class ModelExporter():
    def __init__(self, cfg: Config, model: YOLO, format: str, model_path: Optional[str] = None):
        self.model = model
        self.cfg = cfg
        self.class_num = cfg.dataset.class_num
        self.format = format
        if cfg.weight == True:
            cfg.weight = Path("weights") / f"{cfg.model.name}.pt"
        if model_path:
            self.model_path = model_path
        else:
            self.model_path = f"{Path(self.cfg.weight).stem}.{self.format}"

    def export_onnx(self, dynamic_axes : Optional[Dict[str, Dict[int, str]]] = None, model_path: Optional[str] = None):
        logger.info(f":package: Exporting model to onnx format")
        import torch
        dummy_input = torch.ones((1, 3, *self.cfg.image_size))

        if model_path:
            onnx_model_path = model_path
        else:
            onnx_model_path = self.model_path

        # onnx_model_path = f"{Path(self.cfg.weight).stem}.onnx"
        output_names : List[str] = [
            "1_class_scores_small", "2_box_features_small", "3_bbox_deltas_small",
            "4_class_scores_medium", "5_box_features_medium", "6_bbox_deltas_medium",
            "7_class_scores_large", "8_box_features_large", "9_bbox_deltas_large"
        ]

        torch.onnx.export(
            self.model,
            dummy_input,
            onnx_model_path,
            input_names=["input"],
            output_names=output_names,
            dynamic_axes=dynamic_axes,
        )



        logger.info(f":inbox_tray: ONNX model saved to {onnx_model_path}")

        return onnx_model_path
    
    def export_tflite(self):
        logger.info(f":package: Exporting model to tflite format")

        import torch
        self.model.eval()
        example_inputs = (torch.rand(1, 3, *self.cfg.image_size),)

        import ai_edge_torch
        edge_model = ai_edge_torch.convert(self.model, example_inputs)
        edge_model.export(self.model_path)

        logger.info(f":white_check_mark: Model exported to tflite format")

    def export_coreml(self):
        logger.info(f":package: Exporting model to coreml format")

        import torch
        
        self.model.eval()
        example_inputs = (torch.rand(1, 3, *self.cfg.image_size),)
        exported_program = torch.export.export(self.model, example_inputs)

        import logging
        import coremltools as ct

        # Convert to Core ML program using the Unified Conversion API.
        logging.getLogger("coremltools").disabled = True
        
        output_names : List[str] = [
            "1_class_scores_small", "2_box_features_small", "3_bbox_deltas_small",
            "4_class_scores_medium", "5_box_features_medium", "6_bbox_deltas_medium",
            "7_class_scores_large", "8_box_features_large", "9_bbox_deltas_large"
        ]
        
        model_from_export = ct.convert(exported_program, 
                                       outputs=[ct.TensorType(name=name) for name in output_names])
        
        model_from_export.save(f"{Path(self.cfg.weight).stem}.mlpackage")
        logger.info(f":white_check_mark: Model exported to coreml format")