import sys
from pathlib import Path

import hydra
import torch
from lightning import Trainer

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from yolo.config.config import Config
from yolo.tools.solver import BaseModel, InferenceModel, TrainModel, ValidateModel
from yolo.utils.logging_utils import setup


@hydra.main(config_path="config", config_name="config", version_base=None)
def main(cfg: Config):
    callbacks, loggers, save_path = setup(cfg)

    trainer = Trainer(
        accelerator="auto",
        max_epochs=getattr(cfg.task, "epoch", None),
        precision="16-mixed",
        callbacks=callbacks,
        logger=loggers,
        log_every_n_steps=1,
        gradient_clip_val=10,
        gradient_clip_algorithm="value",
        deterministic=True,
        enable_progress_bar=not getattr(cfg, "quite", False),
        default_root_dir=save_path,
    )

    if cfg.task.task == "train":
        model = TrainModel(cfg)
        trainer.fit(model)

        export_onnx(model, cfg)
    if cfg.task.task == "validation":
        model = ValidateModel(cfg)
        trainer.validate(model)
    if cfg.task.task == "inference":
        model = InferenceModel(cfg)
        trainer.predict(model)


def export_onnx(model: TrainModel, cfg: Config):
    model_state_dict = {}
    for model_key, model_weight in model.state_dict().items():
        if 'ema' not in model_key:
            model_state_dict[model_key] = model_weight

    cfg.model.auxiliary = {}
    cfg.model.is_exporting = True
    export_model = BaseModel(cfg).to('cpu')
    export_model.load_state_dict(model_state_dict)

    dummy_input = torch.ones((1, *cfg.image_size, 3))
    torch.onnx.export(
        export_model,
        dummy_input,
        '/home/nik/work/model.onnx',
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    )


if __name__ == "__main__":
    main()
