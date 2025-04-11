All In 1
========

:file:`yolo.lazy` is a packaged file that includes :guilabel:`training`, :guilabel:`validation`, and :guilabel:`inference` tasks.
For detailed function documentation, thercheck out the IPython notebooks to learn how to import and use these function
the following section will break down operation inside of lazy, also supporting directly import/call the function.

[TOC], setup, build, dataset, train, validation, inference
To train the model, you can run:

Train Model
----------


- batch size check / cuda
- training time / check
- build model / check
- dataset / check

.. code-block:: bash

    python yolo/lazy.py task=train

You can customize the training process by overriding the following common arguments:

- ``name``: :guilabel:`str`
  The experiment name.

- ``model``: :guilabel:`str`
  Model backbone, options include [model_zoo] v9-c, v7, v9-e, etc.

- ``cpu_num``: :guilabel:`int`
  Number of CPU workers (num_workers).

- ``out_path``: :guilabel:`Path`
  The output path for saving models and logs.

- ``weight``: :guilabel:`Path | bool | None`
  The path to pre-trained weights, False for training from scratch, None for default weights.

- ``use_TensorBoard``: :guilabel:`bool`
  Whether to use TensorBoard for logging.

- ``image_size``: :guilabel:`int | [int, int]`
  The input image size.

- ``+quiet``: :guilabel:`bool`
  Optional, disable all output.

- ``task.epoch``: :guilabel:`int`
  Total number of training epochs.

- ``task.data.batch_size``: :guilabel:`int`
  The size of each batch (auto-batch sizing [WIP]).

Examples
~~~~~~~~

To train a model with a specific batch size and image size, you can run:

.. code-block:: bash

    python yolo/lazy.py task=train task.data.batch_size=12 image_size=1280

Multi-GPU Training with DDP
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For multi-GPU training, we use Distributed Data Parallel (DDP) for efficient and scalable training.
DDP enable training model with mutliple GPU, even the GPUs aren't on the same machine. For more details, you can refer to the `DDP tutorial <https://pytorch.org/tutorials/intermediate/ddp_tutorial.html>`_.

To train on multiple GPUs, replace the ``python`` command with ``torchrun --nproc_per_node=[GPU_NUM]``. The ``nproc_per_node`` argument specifies the number of GPUs to use.


.. tabs::

   .. tab:: bash
    .. code-block:: bash

        torchrun --nproc_per_node=2 yolo/lazy.py task=train device=[0,1]

   .. tab:: zsh
    .. code-block:: bash

        torchrun --nproc_per_node=2 yolo/lazy.py task=train device=\[0,1\]


Training on a Custom Dataset
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To use the auto-download module, we suggest users construct the dataset config in the following format.
If the config files include `auto_download`, the model will automatically download the dataset when creating the dataloader.

Here is an example dataset config file:

.. literalinclude:: ../../yolo/config/dataset/dev.yaml
  :language: YAML

Both of the following formats are acceptable:

- ``path``: :guilabel:`str`
  The path to the dataset.

- ``train, validation``: :guilabel:`str`
  The training and validation directory names under `/images`. If using txt as ground truth, these should also be the names under `/labels/`.

- ``class_num``: :guilabel:`int`
  The number of dataset classes.

- ``class_list``: :guilabel:`List[str]`
  Optional, the list of class names, used only for visualizing the bounding box classes.

- ``auto_download``: :guilabel:`dict`
  Optional, whether to auto-download the dataset.

The dataset should include labels or annotations, preferably in JSON format for compatibility with pycocotools during inference:

.. code-block:: text

    DataSetName/
    ├── annotations
    │   ├── train_json_name.json
    │   └── val_json_name.json
    ├── labels/
    │   ├── train/
    │   │   ├── AnyLabelName.txt
    │   │   └── ...
    │   └── validation/
    │       └── ...
    └── images/
        ├── train/
        │   ├── AnyImageNameN.{png,jpg,jpeg}
        │   └── ...
        └── validation/
            └── ...


Validation Model
----------------

During training, this block will be auto-executed. You may also run this task manually to generate a JSON file representing the predictions for a given validation dataset. If the validation set includes JSON annotations, it will run pycocotools for evaluation.

We recommend setting ``task.data.shuffle`` to False and turning off ``task.data.data_augment``.

You can customize the validation process by overriding the following arguments:

- ``task.nms.min_confidence``: :guilabel:`str`
  The minimum confidence of model prediction.

- ``task.nms.min_iou``: :guilabel:`str`
  The minimum IoU threshold for NMS (Non-Maximum Suppression).

Examples
~~~~~~~~

.. tabs::

   .. tab:: git-cloned
      .. code-block:: bash

         python yolo/lazy.py task=validation task.nms.min_iou=0.9

   .. tab:: PyPI
      .. code-block:: bash

         yolo task=validation task.nms.min_iou=0.9


Model Inference
---------------

.. note::
   The ``dataset`` parameter shouldn't be overridden because the model requires the ``class_num`` of the dataset. If the classes have names, please provide the ``class_list``.

You can customize the inference process by overriding the following arguments:

- ``task.fast_inference``: :guilabel:`str`
  Optional. Values can be `onnx`, `trt`, `deploy`, or `None`. `deploy` will detach the model auxiliary head.

- ``task.data.source``: :guilabel:`str | Path | int`
  This argument will be auto-resolved and could be a webcam ID, image folder path, video/image path.

- ``task.nms.min_confidence``: :guilabel:`str`
  The minimum confidence of model prediction.

- ``task.nms.min_iou``: :guilabel:`str`
  The minimum IoU threshold for NMS (Non-Maximum Suppression).

Examples
~~~~~~~~

.. tabs::

   .. tab:: git-cloned
      .. code-block:: bash

         python yolo/lazy.py model=v9-m task.nms.min_confidence=0.1 task.data.source=0 task.fast_inference=onnx

   .. tab:: PyPI
      .. code-block:: bash

         yolo model=v9-m task.nms.min_confidence=0.1 task.data.source=0 task.fast_inference=onnx
