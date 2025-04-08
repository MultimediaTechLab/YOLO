import glob
import os
import sys
from pathlib import Path

import hydra
import torch
from lightning import Trainer

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from yolo.config.config import Config
from yolo.model.yolo import create_model
from yolo.tools.solver import BaseModel, InferenceModel, TrainModel, ValidateModel
from yolo.utils.logging_utils import setup


@hydra.main(config_path='config', config_name='config', version_base=None)
def main(cfg: Config):
    callbacks, loggers, save_path = setup(cfg)

    trainer = Trainer(
        accelerator='auto',
        max_epochs=getattr(cfg.task, 'epoch', None),
        precision='16-mixed',
        callbacks=callbacks,
        logger=loggers,
        log_every_n_steps=1,
        gradient_clip_val=10,
        gradient_clip_algorithm='value',
        deterministic=True,
        enable_progress_bar=not getattr(cfg, 'quite', False),
        default_root_dir=save_path,
    )

    if cfg.task.task == 'train':
        model = TrainModel(cfg)
        trainer.fit(model)

        export_onnx(cfg)
    if cfg.task.task == 'validation':
        model = ValidateModel(cfg)
        trainer.validate(model)
    if cfg.task.task == 'inference':
        model = InferenceModel(cfg)
        trainer.predict(model)


def _get_latest_checkpoint(cfg: Config) -> str:
    checkpoints_directory = os.path.join(cfg.out_path, cfg.task.task, cfg.name, 'checkpoints')
    checkpoint_files      = glob.glob('%s/*.ckpt' % checkpoints_directory)
    if len(checkpoint_files) == 0:
        raise RuntimeError('No checkpoint found after training completed')

    latest_checkpoint_file = max(checkpoint_files, key=os.path.getctime)
    return latest_checkpoint_file


def export_onnx(cfg: Config):
    print('Exporting model to ONNX...')
    checkpoint_file = _get_latest_checkpoint(cfg)

    checkpoint_weights = torch.load(checkpoint_file, map_location=torch.device('cpu'), weights_only=False)
    if 'state_dict' in checkpoint_weights:
        checkpoint_weights = checkpoint_weights['state_dict']
    else:
        raise RuntimeError('Could not find state_dict in checkpoint after training completed')

    export_model_state_dict = {}
    for model_key, model_weight in checkpoint_weights.items():
        model_key = model_key.replace('model.model.', 'model.')
        export_model_state_dict[model_key] = model_weight

    cfg.model.is_exporting    = True
    cfg.model.model.auxiliary = {}
    export_model = create_model(cfg.model, cfg=cfg, class_num=cfg.dataset.class_num, weight_path=False).to('cpu')
    export_model.load_state_dict(export_model_state_dict, strict=False)

    dummy_input = torch.zeros((1, *cfg.image_size, 3))
    torch.onnx.export(
        export_model,
        (dummy_input,),
        cfg.export_path,
        export_params=True,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}},
    )

    print('Successfully exported model to %s' % cfg.export_path)


if __name__ == '__main__':
    main()
