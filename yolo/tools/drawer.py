from typing import  Optional, Union

import numpy as np
import torch
from PIL import Image
import supervision as sv
from torchvision.transforms.functional import to_pil_image

from yolo.config.config import ModelConfig
from yolo.model.yolo import YOLO
from yolo.utils.logger import logger


def draw_bboxes(
    img: Union[Image.Image, torch.Tensor],
    detections: sv.Detections,
    *,
    idx2label: Optional[list] = None,
):
    """
    Draw bounding boxes on an image.

    Args:
    - img (PIL Image or torch.Tensor): Image on which to draw the bounding boxes.
    - bboxes (List of Lists/Tensors): Bounding boxes with [class_id, x_min, y_min, x_max, y_max],
      where coordinates are normalized [0, 1].
    """
    # Convert tensor image to PIL Image if necessary
    if isinstance(img, torch.Tensor):
        if img.dim() > 3:
            logger.warning("🔍 >3 dimension tensor detected, using the 0-idx image.")
            img = img[0]
        img = to_pil_image(img)

    box_annotator = sv.ColorAnnotator(color_lookup=sv.ColorLookup.CLASS)
    label_annotator = sv.LabelAnnotator(color_lookup=sv.ColorLookup.CLASS)
    img = img.copy()
    img = box_annotator.annotate(img, detections=detections)
    if idx2label:
        labels = [
            f"{str(idx2label[int(class_id)])} {confidence:.2f}"
            for class_id, confidence
            in zip(detections.class_id, detections.confidence)
        ]
        img = label_annotator.annotate(img, labels=labels, detections=detections)
    return img


def draw_model(*, model_cfg: ModelConfig = None, model: YOLO = None, v7_base=False):
    from graphviz import Digraph

    if model_cfg:
        from yolo.model.yolo import create_model

        model = create_model(model_cfg)
    elif model is None:
        raise ValueError("Drawing Object is None")

    model_size = len(model.model) + 1
    model_mat = np.zeros((model_size, model_size), dtype=bool)

    layer_name = ["INPUT"]
    for idx, layer in enumerate(model.model, start=1):
        layer_name.append(str(type(layer)).split(".")[-1][:-2])
        if layer.tags is not None:
            layer_name[-1] = f"{layer.tags}-{layer_name[-1]}"
        if isinstance(layer.source, int):
            source = layer.source + (layer.source < 0) * idx
            model_mat[source, idx] = True
        else:
            for source in layer.source:
                source = source + (source < 0) * idx
                model_mat[source, idx] = True

    pattern_mat = []
    if v7_base:
        pattern_list = [("ELAN", 8, 3), ("ELAN", 8, 55), ("MP", 5, 11)]
        for name, size, position in pattern_list:
            pattern_mat.append(
                (name, size, model_mat[position : position + size, position + 1 : position + 1 + size].copy())
            )

    dot = Digraph(comment="Model Flow Chart")
    node_idx = 0

    for idx in range(model_size):
        for jdx in range(idx, model_size - 7):
            for name, size, pattern in pattern_mat:
                if (model_mat[idx : idx + size, jdx : jdx + size] == pattern).all():
                    layer_name[idx] = name
                    model_mat[idx : idx + size, jdx : jdx + size] = False
                    model_mat[idx, idx + size] = True
        dot.node(str(idx), f"{layer_name[idx]}")
        node_idx += 1
        for jdx in range(idx, model_size):
            if model_mat[idx, jdx]:
                dot.edge(str(idx), str(jdx))
    try:
        dot.render("Model-arch", format="png", cleanup=True)
        logger.info(":artist_palette: Drawing Model Architecture at Model-arch.png")
    except:
        logger.warning(":warning: Could not find graphviz backend, continue without drawing the model architecture")
