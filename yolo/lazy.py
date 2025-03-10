import sys
from pathlib import Path

import hydra
from omegaconf.dictconfig import DictConfig

# FIXME: messing with sys.path is a bad idea. Factor this out.
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))


@hydra.main(config_path="config", config_name="config", version_base=None)
def main(cfg: DictConfig):
    from yolo.utils.logging_utils import setup
    callbacks, loggers, save_path = setup(cfg)

    from yolo.utils.trainer import YoloTrainer as Trainer
    from yolo.tools.solver import InferenceModel, TrainModel, ValidateModel

    trainer_kwargs = dict(
        ###
        # Not Allowed to be overwritten (FIXME: can we fix this)
        callbacks=callbacks,
        logger=loggers,
        ###
        # Uses a non-standard configuration location (Should we refactor this?)
        default_root_dir=save_path,
        max_epochs=getattr(cfg.task, "epoch", None),
        enable_progress_bar=not getattr(cfg, "quite", False),
    )
    if len(cfg.trainer.keys() & trainer_kwargs.keys()) > 0:
        unsupported = set(cfg.trainer.keys() & trainer_kwargs.keys())
        raise AssertionError(
            f'Cannot specify unsupported trainer args: {unsupported!r} '
            'in the trainer config'
        )
    trainer_kwargs.update(cfg.trainer)
    trainer = Trainer(**trainer_kwargs)

    if cfg.task.task == "train":
        model = TrainModel(cfg)
        trainer.fit(model)
    if cfg.task.task == "validation":
        model = ValidateModel(cfg)
        trainer.validate(model)
    if cfg.task.task == "inference":
        model = InferenceModel(cfg)
        trainer.predict(model)


if __name__ == "__main__":
    main()
