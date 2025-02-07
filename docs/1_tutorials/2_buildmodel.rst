Build Model
===========

In YOLOv7, the prediction will be ``Anchor``, and in YOLOv9, it will predict ``Vector``. The converter will turn the bounding box to the vector.

The overall model flowchart is as follows:

.. mermaid::

    flowchart LR
    Input-->Model;
    Model--Class-->NMS;
    Model--Anc/Vec-->Converter;
    Converter--Box-->NMS;
    NMS-->Output;

Load Model
~~~~~~~~~~

Using `create_model`, it will automatically create the :class:`~yolo.model.yolo.YOLO` model and load the provided weights.

Arguments:

- **model**: :class:`~yolo.config.config.ModelConfig`
  The model configuration.
- **class_num**: :guilabel:`int`
  The number of classes in the dataset, used for the YOLO's prediction head.
- **weight_path**: :guilabel:`Path | bool`
  The path to the model weights.
    - If `False`, weights are not loaded.
    - If :guilabel:`True | None`, default weights are loaded.
    - If a `Path`, the model weights are loaded from the specified path.

.. code-block:: python

    model = create_model(cfg.model, cfg=cfg, class_num=cfg.dataset.class_num, weight_path=cfg.weight)
    model = model.to(device)

Deploy Model
~~~~~~~~~~~~

In the deployment version, we will remove the auxiliary branch of the model for fast inference. If the config includes ONNX and TensorRT, it will load/compile the model to ONNX or TensorRT format after removing the auxiliary branch.

.. code-block:: python

    model = FastModelLoader(cfg).load_model(device)

Autoload Converter
~~~~~~~~~~~~~~~~~~

Autoload the converter based on the model type (v7 or v9).

Arguments:

- **Model Name**: :guilabel:`str`
  Used for choosing ``Vec2Box`` or ``Anc2Box``.
- **Anchor Config**: The anchor configuration, used to generate the anchor grid.
- **model**, **image_size**: Used for auto-detecting the anchor grid.

.. code-block:: python

    converter = create_converter(cfg.model.name, model, cfg.model.anchor, cfg.image_size, device)
