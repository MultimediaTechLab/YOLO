import sys
from pathlib import Path

import hydra
import torch

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from yolo.config.config import Config
from yolo.model.yolo import create_model
from yolo.tools.data_loader import create_dataloader
from yolo.tools.solver import ModelTester, ModelTrainer
from yolo.utils.bounding_box_utils import Vec2Box
from yolo.utils.deploy_utils import FastModelLoader
from yolo.utils.logging_utils import custom_logger, validate_log_directory


@hydra.main(config_path="config", config_name="config", version_base=None)
def main(cfg: Config):
    custom_logger()
    save_path = validate_log_directory(cfg, exp_name=cfg.name)
    dataloader = create_dataloader(cfg.task.data, cfg.dataset, cfg.task.task)
    device = torch.device(cfg.device)
    if getattr(cfg.task, "fast_inference", False):
        model = FastModelLoader(cfg).load_model()
        device = torch.device(cfg.device)
    else:
        model = create_model(cfg.model, class_num=cfg.class_num, weight_path=cfg.weight, device=device)

    vec2box = Vec2Box(model, cfg.image_size, device)

    if cfg.task.task == "train":
        trainer = ModelTrainer(cfg, model, vec2box, save_path, device)
        trainer.solve(dataloader)

    if cfg.task.task == "inference":
        tester = ModelTester(cfg, model, vec2box, save_path, device)
        tester.solve(dataloader)


if __name__ == "__main__":
    main()
