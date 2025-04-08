import sys
from pathlib import Path

import hydra

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))


from yolo import (
    Config,
    FastModelLoader,
    ModelTester,
    ProgressLogger,
    create_converter,
    create_dataloader,
    create_model,
)
from yolo.utils.model_utils import get_device


@hydra.main(config_path="config", config_name="config", version_base=None)
def main(cfg: Config):
    progress = ProgressLogger(cfg, exp_name=cfg.name)
    device, use_ddp = get_device(cfg.device)
    dataloader = create_dataloader(cfg.task.data, cfg.dataset, cfg.task.task, use_ddp)
    if getattr(cfg.task, "fast_inference", False):
        model = FastModelLoader(cfg).load_model(device)
    else:
        model = create_model(cfg.model, cfg=cfg, class_num=cfg.dataset.class_num, weight_path=cfg.weight)
        model = model.to(device)

    converter = create_converter(cfg.model.name, model, cfg.model.anchor, cfg.image_size, device)

    solver = ModelTester(cfg, model, converter, progress, device)
    progress.start()
    solver.solve(dataloader)


if __name__ == "__main__":
    main()
