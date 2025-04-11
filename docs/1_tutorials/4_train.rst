Train & Validation
==================

Training Model
----------------

To train a model, the :class:`~yolo.tools.solver.ModelTrainer` can help manage the training process. Initialize the :class:`~yolo.tools.solver.ModelTrainer` and use the :func:`~yolo.tools.solver.ModelTrainer.solve` function to start the training.

Before starting the training, don't forget to start the progress logger to enable logging the process status. This will also enable TensorBoard if configured.

.. code-block:: python

  from yolo import ModelTrainer
  solver = ModelTrainer(cfg, model, converter, progress, device, use_ddp)
  progress.start()
  solver.solve(dataloader)

Training Diagram
~~~~~~~~~~~~~~~~

The following diagram illustrates the training process:

.. mermaid::

  flowchart LR
    subgraph TS["trainer.solve"]
      subgraph TE["train one epoch"]
        subgraph "train one batch"
          backpropagation-->TF[forward]
          TF-->backpropagation
        end
      end
      subgraph validator.solve
          VC["calculate mAP"]-->VF[forward]
          VF[forward]-->VC
      end
    end
    TE-->validator.solve
    validator.solve-->TE

Validation Model
----------------

To validate the model performance, we follow a similar approach as the training process using :class:`~yolo.tools.solver.ModelValidator`.

.. code-block:: python

   from yolo import ModelValidator
   solver = ModelValidator(cfg, model, converter, progress, device, use_ddp)
   progress.start()
   solver.solve(dataloader)

The :class:`~yolo.tools.solver.ModelValidator` class helps manage the validation process, ensuring that the model's performance is evaluated accurately.

.. note:: The original training process already includes the validation phase. Call this separately if you want to run the validation again after the training is completed.
