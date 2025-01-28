"""
Helpers for COCO / KWCoco integration
"""


def tensor_to_kwimage(yolo_annot_tensor):
    """
    Convert a raw output tensor to a kwimage Detections object
    """
    import kwimage
    class_idxs = yolo_annot_tensor[:, 0].int()
    boxes = kwimage.Boxes(yolo_annot_tensor[:, 1:5], format='xyxy')
    dets = kwimage.Detections(
        boxes=boxes,
        class_idxs=class_idxs
    )

    if yolo_annot_tensor.shape[1] > 5:
        scores = yolo_annot_tensor[:, 5]
        dets.data['scores'] = scores
    return dets
