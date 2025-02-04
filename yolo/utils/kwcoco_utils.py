"""
Helpers for COCO / KWCoco integration
"""


def tensor_to_kwimage(yolo_annot_tensor, classes=None):
    """
    Convert a raw output tensor to a kwimage Detections object

    Args:
        yolo_annot_tensor (Tensor):
            Each row corresponds to an annotation.
            yolo_annot_tensor[:, 0] is the class index
            yolo_annot_tensor[:, 1:5] is the ltrb bounding box
            yolo_annot_tensor[:, 5] is the objectness confidence
            Other columns are the per-class confidence

        classes (kwcoco.CategoryTree):
            ...

    Example:
        yolo_annot_tensor = torch.rand(1, 6)
    """
    import kwimage
    class_idxs = yolo_annot_tensor[:, 0].int()
    boxes = kwimage.Boxes(yolo_annot_tensor[:, 1:5], format='xyxy')
    dets = kwimage.Detections(
        boxes=boxes,
        class_idxs=class_idxs,
        classes=classes,
    )

    if yolo_annot_tensor.shape[1] > 5:
        scores = yolo_annot_tensor[:, 5]
        dets.data['scores'] = scores

    if classes is not None:
        if hasattr(classes, 'idx_to_id'):
            # Add class-id information if that is available
            import torch
            idx_to_id = torch.Tensor(classes.idx_to_id).int().to(class_idxs.device)
            class_ids = idx_to_id[class_idxs]
            dets.data['class_ids'] = class_ids
    return dets
