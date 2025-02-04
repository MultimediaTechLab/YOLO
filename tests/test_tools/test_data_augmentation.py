import sys
from pathlib import Path

import torch
from PIL import Image
from torchvision.transforms import functional as TF

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

from yolo.tools.data_augmentation import (
    AugmentationComposer,
    HorizontalFlip,
    Mosaic,
)


def test_horizontal_flip():
    # Create a mock image and bounding boxes
    img = Image.new("RGB", (100, 100), color="red")
    boxes = torch.tensor([[1, 0.05, 0.1, 0.7, 0.9]])  # class, xmin, ymin, xmax, ymax

    flip_transform = HorizontalFlip(prob=1)  # Set probability to 1 to ensure flip
    flipped_img, flipped_boxes = flip_transform(img, boxes)

    # Assert image is flipped by comparing it to a manually flipped image
    assert TF.hflip(img) == flipped_img

    # Assert bounding boxes are flipped correctly
    expected_boxes = torch.tensor([[1, 0.3, 0.1, 0.95, 0.9]])
    assert torch.allclose(flipped_boxes, expected_boxes), "Bounding boxes were not flipped correctly"


def test_compose():
    # Define two mock transforms that simply return the inputs
    def mock_transform(image, boxes):
        return image, boxes

    compose = AugmentationComposer([mock_transform, mock_transform])
    img = Image.new("RGB", (640, 640), color="blue")
    boxes = torch.tensor([[0, 0.2, 0.2, 0.8, 0.8]])

    transformed_img, transformed_boxes, rev_tensor = compose(img, boxes)
    tensor_img = TF.pil_to_tensor(img).to(torch.float32) / 255

    assert (transformed_img == tensor_img).all(), "Image should not be altered"
    assert torch.equal(transformed_boxes, boxes), "Boxes should not be altered"


def test_mosaic():
    img = Image.new("RGB", (100, 100), color="green")
    boxes = torch.tensor([[0, 0.25, 0.25, 0.75, 0.75]])

    # Mock parent with image_size and get_more_data method
    class MockParent:
        base_size = 100

        def get_more_data(self, num_images):
            return [(img, boxes) for _ in range(num_images)]

    mosaic = Mosaic(prob=1)  # Ensure mosaic is applied
    mosaic.set_parent(MockParent())

    mosaic_img, mosaic_boxes = mosaic(img, boxes)

    # Checks here would depend on the exact expected behavior of the mosaic function,
    # such as dimensions and content of the output image and boxes.

    assert mosaic_img.size == (100, 100), "Mosaic image size should be same"
    assert len(mosaic_boxes) > 0, "Should have some bounding boxes"
