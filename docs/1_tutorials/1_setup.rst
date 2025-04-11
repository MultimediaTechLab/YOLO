Setup Config
============

To set up your configuration, you will need to generate a configuration class based on :class:`~yolo.config.config.Config`, which can be achieved using `hydra <https://hydra.cc/>`_.
The configuration will include all the necessary settings for your ``task``, including general configuration, ``dataset`` information, and task-specific information (``train``, ``inference``, ``validation``).

Next, create the progress logger to handle the output and progress bar. This class is based on `rich <https://github.com/Textualize/rich>`_'s progress bar.

.. tabs::

   .. tab:: decorator
      .. code-block:: python

         import hydra
         from yolo import ProgressLogger
         from yolo.config.config import Config

         @hydra.main(config_path="config", config_name="config", version_base=None)
         def main(cfg: Config):
             progress = ProgressLogger(cfg, exp_name=cfg.name)
             pass

   .. tab:: initialize & compose
      .. code-block:: python

         from hydra import compose, initialize
         from yolo import ProgressLogger
         from yolo.config.config import Config

         with initialize(config_path="config", version_base=None):
             cfg = compose(config_name="config", overrides=["task=train", "model=v9-c"])

         progress = ProgressLogger(cfg, exp_name=cfg.name)

TODO: add a config over view
