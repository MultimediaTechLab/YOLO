from pathlib import Path

import torch
from torch import Tensor

from yolo.config.config import Config
from yolo.model.yolo import YOLO, create_model
from yolo.utils.export_utils import ModelExporter
from yolo.utils.logger import logger


class FastModelLoader:
    def __init__(self, cfg: Config, model: YOLO):
        self.cfg = cfg
        self.model = model
        self.compiler: str = cfg.task.fast_inference
        self.class_num = cfg.dataset.class_num

        self._validate_compiler()
        if cfg.weight == True:
            cfg.weight = Path("weights") / f"{cfg.model.name}.pt"

        extention = self.compiler
        if self.compiler == "coreml":
            extention = "mlpackage"

        self.model_path = f"{Path(cfg.weight).stem}.{extention}"

    def _validate_compiler(self):
        if self.compiler not in ["onnx", "trt", "deploy", "coreml", "tflite"]:
            logger.warning(f":warning: Compiler '{self.compiler}' is not supported. Using original model.")
            self.compiler = None
        if self.cfg.device == "mps" and self.compiler == "trt":
            logger.warning(":red_apple: TensorRT does not support MPS devices. Using original model.")
            self.compiler = None

    def load_model(self, device):
        if self.compiler == "onnx":
            return self._load_onnx_model(device)
        if self.compiler == "tflite":
            return self._load_tflite_model(device)
        elif self.compiler == "coreml":
            return self._load_coreml_model(device)
        elif self.compiler == "trt":
            return self._load_trt_model().to(device)
        elif self.compiler == "deploy":
            self.cfg.model.model.auxiliary = {}
        return create_model(self.cfg.model, class_num=self.class_num, weight_path=self.cfg.weight).to(device)

    def _load_tflite_model(self, device):

        if not Path(self.model_path).exists():
            self._create_tflite_model()

        from ai_edge_litert.interpreter import Interpreter

        try:
            interpreter = Interpreter(model_path=self.model_path)
            interpreter.allocate_tensors()
            logger.info(":rocket: Using TensorFlow Lite as MODEL framework!")
        except Exception as e:
            logger.warning(f"🈳 Error loading TFLite model: {e}")
            interpreter = self._create_tflite_model()

        def tflite_forward(self: Interpreter, x: Tensor):

            # Get input & output tensor details
            input_details = self.get_input_details()
            output_details = sorted(self.get_output_details(), key=lambda d: d["name"])  # Sort by 'name'

            # Convert input tensor to NumPy and assign it to the model
            x_numpy = x.cpu().numpy()
            self.set_tensor(input_details[0]["index"], x_numpy)

            model_outputs, layer_output = [], []
            x_numpy = x.cpu().numpy()
            self.set_tensor(input_details[0]["index"], x_numpy)
            self.invoke()
            for idx, output_detail in enumerate(output_details):
                predict = self.get_tensor(output_detail["index"])
                layer_output.append(torch.from_numpy(predict).to(device))
                if idx % 3 == 2:
                    model_outputs.append(layer_output)
                    layer_output = []
            if len(model_outputs) == 6:
                model_outputs = model_outputs[:3]
            return {"Main": model_outputs}

        Interpreter.__call__ = tflite_forward

        return interpreter

    def _load_onnx_model(self, device):

        # TODO install onnxruntime or onnxruntime-gpu if needed

        from onnxruntime import InferenceSession

        def onnx_forward(self: InferenceSession, x: Tensor):
            x = {self.get_inputs()[0].name: x.cpu().numpy()}
            model_outputs, layer_output = [], []
            for idx, predict in enumerate(self.run(None, x)):
                layer_output.append(torch.from_numpy(predict).to(device))
                if idx % 3 == 2:
                    model_outputs.append(layer_output)
                    layer_output = []
            if len(model_outputs) == 6:
                model_outputs = model_outputs[:3]
            return {"Main": model_outputs}

        InferenceSession.__call__ = onnx_forward

        if device == "cpu":
            providers = ["CPUExecutionProvider"]
        elif device == "coreml":
            providers = ["CoreMLExecutionProvider"]
        else:
            providers = ["CUDAExecutionProvider"]
        try:
            ort_session = InferenceSession(self.model_path, providers=providers)
            logger.info(":rocket: Using ONNX as MODEL frameworks!")
        except Exception as e:
            logger.warning(f"🈳 Error loading ONNX model: {e}")
            ort_session = self._create_onnx_model(providers)
        return ort_session

    def _create_onnx_model(self, providers):
        from onnxruntime import InferenceSession

        model_exporter = ModelExporter(self.cfg, self.model, format="onnx", model_path=self.model_path)
        model_exporter.export_onnx(dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}})
        return InferenceSession(self.model_path, providers=providers)

    def _load_coreml_model(self, device):
        from coremltools import models

        def coreml_forward(self, x: Tensor):
            x = x.cpu().numpy()
            model_outputs = []
            predictions = self.predict({"x": x})

            output_keys = ["preds_cls", "preds_anc", "preds_box"]
            for key in output_keys:
                model_outputs.append(torch.from_numpy(predictions[key]).to(device))

            return model_outputs

        models.MLModel.__call__ = coreml_forward

        if not Path(self.model_path).exists():
            self._create_coreml_model()

        try:
            model_coreml = models.MLModel(self.model_path)
            logger.info(":rocket: Using CoreML as MODEL frameworks!")
        except FileNotFoundError:
            logger.warning(f"🈳 No found model weight at {self.model_path}")
            return None
        
        return model_coreml

    def _create_tflite_model(self):
        model_exporter = ModelExporter(self.cfg, self.model, format="tflite", model_path=self.model_path)
        model_exporter.export_tflite()

    def _create_coreml_model(self):
        model_exporter = ModelExporter(self.cfg, self.model, format="coreml", model_path=self.model_path)
        model_exporter.export_coreml()

    def _load_trt_model(self):
        from torch2trt import TRTModule

        try:
            model_trt = TRTModule()
            model_trt.load_state_dict(torch.load(self.model_path))
            logger.info(":rocket: Using TensorRT as MODEL frameworks!")
        except FileNotFoundError:
            logger.warning(f"🈳 No found model weight at {self.model_path}")
            model_trt = self._create_trt_model()
        return model_trt

    def _create_trt_model(self):
        from torch2trt import torch2trt

        model = create_model(self.cfg.model, class_num=self.class_num, weight_path=self.cfg.weight).eval()
        dummy_input = torch.ones((1, 3, *self.cfg.image_size)).cuda()
        logger.info(f"♻️ Creating TensorRT model")
        model_trt = torch2trt(model.cuda(), [dummy_input])
        torch.save(model_trt.state_dict(), self.model_path)
        logger.info(f":inbox_tray: TensorRT model saved to {self.model_path}")
        return model_trt
