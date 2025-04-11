# How To modified YOLO

To facilitate easy customization of the YOLO model, we've structured the codebase to allow for changes through configuration files and minimal code adjustments. This guide will walk you through the steps to customize various components of the model including the architecture, blocks, data loaders, and loss functions.

## Examples

```shell
# Train
python yolo/lazy.py task=train dataset=dev

# Validate
python yolo/lazy.py task=validation
python yolo/lazy.py task=validation model=v9-s
python yolo/lazy.py task=validation dataset=toy
python yolo/lazy.py task=validation dataset=toy name=validation

# Inference
python yolo/lazy.py task=inference
python yolo/lazy.py task=inference device=cpu
python yolo/lazy.py task=inference +quite=True
python yolo/lazy.py task=inference name=AnyNameYouWant
python yolo/lazy.py task=inference image_size=\[480,640]
python yolo/lazy.py task=inference task.nms.min_confidence=0.1
python yolo/lazy.py task=inference task.fast_inference=deploy
python yolo/lazy.py task=inference task.fast_inference=onnx device=cpu
python yolo/lazy.py task=inference task.data.source=data/toy/images/train
```

## Custom Model Architecture

You can change the model architecture simply by modifying the YAML configuration file. Here's how:

1. **Modify Architecture in Config:**

   Navigate to your model's configuration file (typically formate like `yolo/config/model/v9-c.yaml`).
   - Adjust the architecture settings under the `architecture` section. Ensure that every module you reference exists in `module.py`, or refer to the next section on how to add new modules.

    ```yaml
    model:
      foo:
        - ADown:
            args: {out_channels: 256}
        - RepNCSPELAN:
            source: -2
            args: {out_channels: 512, part_channels: 256}
            tags: B4
      bar:
        - Concat:
            source: [-2, B4]
    ```

   `tags`: Use this to labels any module you want, and could be the module source.

   `source`: Set this to the index of the module output you wish to use as input; default is `-1` which refers to the last module's output. Capable tags, relative position, absolute position

   `args`: A dictionary used to initialize parameters for convolutional or bottleneck layers.

   `output`: Whether to serve as the output of the model.

## Custom Block

To add or modify a block in the model:

1. **Create a New Module:**

   Define a new class in `module.py` that inherits from `nn.Module`.

   The constructor should accept `in_channels` as a parameter. Make sure to calculate `out_channels` based on your model's requirements or configure it through the YAML file using `args`.

    ```python
    class CustomBlock(nn.Module):
        def __init__(self, in_channels, out_channels, **kwargs):
            super().__init__()
            self.module = # conv, bool, ...
        def forward(self, x):
            return self.module(x)
    ```

2. **Reference in Config:**
   ```yaml
    ...
    - CustomBlock:
        args: {out_channels: int, etc: ...}
        ...
    ...
   ```


## Custom Data Augmentation

Custom transformations should be designed to accept an image and its bounding boxes, and return them after applying the desired changes. Here’s how you can define such a transformation:


1. **Define Dataset:**

    Your class must have a `__call__` method that takes a PIL image and its corresponding bounding boxes as input, and returns them after processing.


   ```python
    class CustomTransform:
        def __init__(self, prob=0.5):
            self.prob = prob

        def __call__(self, image, boxes):
            return image, boxes
   ```
2. **Update CustomTransform in Config:**

    Specify your custom transformation in a YAML config `yolo/config/data/augment.yaml`. For examples:
    ```yaml
    Mosaic: 1
    # ... (Other Transform)
    CustomTransform: 0.5
    ```


- **Utils**
    - **bbox_utils**
        - `class` Anchor2Box: transform predicted anchor to bounding box
        - `class` Matcher: given prediction and groudtruth, find the groundtruth for each prediction
        - `func` calculate_iou: calculate iou for given two list of bbox
        - `func` transform_bbox: transform bbox from {xywh, xyxy, xcycwh} to {xywh, xyxy, xcycwh}
        - `func` generate_anchors: given image size, make the anchor point for the given size
    - **dataset_utils**
        - `func` locate_label_paths:
        - `func` create_image_metadata:
        - `func` organize_annotations_by_image:
        - `func` scale_segmentation:
    - **logging_utils**
        - `class` ProgressTracker: A class to handle output for each batch, epoch
        - `func` log_model_structure: give a torch model, print it as a table
        - `func` validate_log_directory: for given experiemnt, check if the log folder already existed
    - **model_utils**
        - `class` ExponentialMovingAverage: a mirror of model, do ema on model
        - `func` create_optimizer: return a optimzer, for example SDG, ADAM
        - `func` create_scheduler: return a scheduler, for example Step, Lambda
    - **module_utils**
        - `func` get_layer_map:
        - `func` auto_pad: given a convolution block, return how many pixel should conv padding
        - `func` create_activation_function: given a `func` name, return a activation `func`tion
        - `func` round_up: given number and divider, return a number is mutliplcation of divider
        - `func` divide_into_chunks: for a given list and n, seperate list to n sub list
    - **trainer**
        - `class` Trainer: a class can automatic train the model
- **Tools**
    - **converter_json2txt**
        - `func` discretize_categories: given the dictionary class, turn id from 1: class
        - `func` process_annotations: handle the whole dataset annotations
        - `func` process_annotation: handle a annotation(a list of bounding box)
        - `func` normalize_segmentation: normalize segmentation position to 0~1
        - `func` convert_annotations: convert json annotations to txt file structure
    - **data_augment**
        - `class` AugmentationComposer: Compose a list of data augmentation strategy
        - `class` VerticalFlip: a custom data augmentation, Random Vertical Flip
        - `class` Mosaic: a data augmentation strategy, follow YOLOv5
    - **dataloader**
        - `class` YoloDataset: a custom dataset for training yolo's model
        - `class` YoloDataLoader: a dataloader base on torch's dataloader, with custom allocate function
        - `func` create_dataloader: given a config file, return a YOLO dataloader
    - **drawer**
        - `func` draw_bboxes: given a image and list of bbox, draw bbox on the image
        - `func` draw_model: visualize the given model
    - **get_dataset**
        - `func` download_file: for a given link, download the file
        - `func` unzip_file: unzip the downloaded zip to data/
        - `func` check_files: check if the dataset file numbers is correct
        - `func` prepare_dataset: automatic download the dataset and check if it is correct
    - **loss**
        - `class` BoxLoss: a Custom Loss for bounding box
        - `class` YOLOLoss: a implementation of yolov9 loss
        - `class` DualLoss: a implementation of yolov9 loss with auxiliary detection head
