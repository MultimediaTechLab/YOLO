Config
======



.. autoclass:: yolo.config.config.Config
    :members:
    :undoc-members:

.. automodule:: yolo.config.config
    :members:
    :undoc-members:



.. mermaid::

    classDiagram
    class AnchorConfig {
        List~int~ strides
        Optional~int~ reg_max
        Optional~int~ anchor_num
        List~List~int~~ anchor
    }

    class LayerConfig {
        Dict args
        Union~List~int~~ source
        str tags
    }

    class BlockConfig {
        List~Dict~LayerConfig~~ block
    }

    class ModelConfig {
        Optional~str~ name
        AnchorConfig anchor
        Dict~BlockConfig~ model
    }

    AnchorConfig --> ModelConfig
    LayerConfig --> BlockConfig
    BlockConfig --> ModelConfig

.. mermaid::

    classDiagram
    class DownloadDetail {
        str url
        int file_size
    }

    class DownloadOptions {
        Dict~DownloadDetail~ details
    }

    class DatasetConfig {
        str path
        int class_num
        List~str~ class_list
        Optional~DownloadOptions~ auto_download
    }

    class DataConfig {
        bool shuffle
        int batch_size
        bool pin_memory
        int cpu_num
        List~int~ image_size
        Dict~int~ data_augment
        Optional~Union~str~~ source
    }

    DownloadDetail --> DownloadOptions
    DownloadOptions --> DatasetConfig

.. mermaid::

    classDiagram
    class OptimizerArgs {
        float lr
        float weight_decay
    }

    class OptimizerConfig {
        str type
        OptimizerArgs args
    }

    class MatcherConfig {
        str iou
        int topk
        Dict~str~ factor
    }

    class LossConfig {
        Dict~str~ objective
        Union~bool~ aux
        MatcherConfig matcher
    }

    class SchedulerConfig {
        str type
        Dict~str~ warmup
        Dict~str~ args
    }

    class EMAConfig {
        bool enabled
        float decay
    }

    class TrainConfig {
        str task
        int epoch
        DataConfig data
        OptimizerConfig optimizer
        LossConfig loss
        SchedulerConfig scheduler
        EMAConfig ema
        ValidationConfig validation
    }

    class NMSConfig {
        int min_confidence
        int min_iou
    }

    class InferenceConfig {
        str task
        NMSConfig nms
        DataConfig data
        Optional~None~ fast_inference
        bool save_predict
    }

    class ValidationConfig {
        str task
        NMSConfig nms
        DataConfig data
    }

    OptimizerArgs --> OptimizerConfig
    OptimizerConfig --> TrainConfig
    MatcherConfig --> LossConfig
    LossConfig --> TrainConfig
    SchedulerConfig --> TrainConfig
    EMAConfig --> TrainConfig
    NMSConfig --> InferenceConfig
    NMSConfig --> ValidationConfig


.. mermaid::

    classDiagram
    class GeneralConfig {
        str name
        Union~str~ device
        int cpu_num
        List~int~ class_idx_id
        List~int~ image_size
        str out_path
        bool exist_ok
        int lucky_number
        bool use_TensorBoard
        Optional~str~ weight
    }

.. mermaid::

    classDiagram
    class Config {
        Union~ValidationConfig~ task
        DatasetConfig dataset
        ModelConfig model
        GeneralConfig model
    }

    DatasetConfig --> Config
    DataConfig --> TrainConfig
    DataConfig --> InferenceConfig
    DataConfig --> ValidationConfig
    InferenceConfig --> Config
    ValidationConfig --> Config
    TrainConfig --> Config
    GeneralConfig --> Config
