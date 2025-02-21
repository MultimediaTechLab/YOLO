from yolo.utils.model_utils import prediction_to_sv
import torch
import numpy as np


def test_prediction_to_sv():

    predictions = []
    detections = prediction_to_sv(predictions)
    assert len(detections) == 0

    xyxy = torch.tensor([[60, 60, 160, 160], [40, 40, 120, 120]], dtype=torch.float32)
    confidence = torch.tensor([0.5, 0.5], dtype=torch.float32).unsqueeze(1)
    class_id = torch.tensor([0, 0], dtype=torch.int64).unsqueeze(1)
    predictions = [torch.cat([class_id, xyxy, confidence], dim=1)]

    detections = prediction_to_sv(predictions)
    assert len(detections) == 2
    assert np.allclose(detections.xyxy, xyxy.numpy())
    assert np.allclose(detections.confidence, confidence.numpy().flatten())
    assert np.allclose(detections.class_id, class_id.numpy().flatten())
