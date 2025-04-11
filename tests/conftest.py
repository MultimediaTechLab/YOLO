import sys
from pathlib import Path

import pytest
import torch
from hydra import compose, initialize
from lightning import Trainer

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from yolo import Anc2Box, Config, Vec2Box, create_converter, create_model
from yolo.model.yolo import YOLO
from yolo.tools.data_loader import StreamDataLoader, create_dataloader
from yolo.tools.dataset_preparation import prepare_dataset
from yolo.utils.logging_utils import set_seed, setup


def pytest_configure(config):
    config.addinivalue_line("markers", "requires_cuda: mark test to run only if CUDA is available")


def get_cfg(overrides=[]) -> Config:
    config_path = "../yolo/config"
    with initialize(config_path=config_path, version_base=None):
        cfg: Config = compose(config_name="config", overrides=overrides)
        set_seed(cfg.lucky_number)
        return cfg


@pytest.fixture(scope="session")
def train_cfg() -> Config:
    return get_cfg(overrides=["task=train", "dataset=mock"])


@pytest.fixture(scope="session")
def validation_cfg():
    return get_cfg(overrides=["task=validation", "dataset=mock"])


@pytest.fixture(scope="session")
def inference_cfg():
    return get_cfg(overrides=["task=inference"])


@pytest.fixture(scope="session")
def inference_v7_cfg():
    return get_cfg(overrides=["task=inference", "model=v7"])


@pytest.fixture(scope="session")
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture(scope="session")
def model(train_cfg: Config, device) -> YOLO:
    model = create_model(train_cfg.model, train_cfg)
    return model.to(device)


@pytest.fixture(scope="session")
def model_v7(inference_v7_cfg: Config, device) -> YOLO:
    model = create_model(inference_v7_cfg.model, inference_v7_cfg)
    return model.to(device)


@pytest.fixture(scope="session")
def solver(train_cfg: Config) -> Trainer:
    callbacks, loggers, save_path = setup(train_cfg)
    trainer = Trainer(
        accelerator="auto",
        max_epochs=getattr(train_cfg.task, "epoch", None),
        precision="16-mixed",
        callbacks=callbacks,
        logger=loggers,
        log_every_n_steps=1,
        gradient_clip_val=10,
        deterministic=True,
        default_root_dir=save_path,
    )
    return trainer


@pytest.fixture(scope="session")
def vec2box(train_cfg: Config, model: YOLO, device) -> Vec2Box:
    vec2box = create_converter(train_cfg.model.name, model, train_cfg.model.anchor, train_cfg.image_size, device)
    return vec2box


@pytest.fixture(scope="session")
def anc2box(inference_v7_cfg: Config, model: YOLO, device) -> Anc2Box:
    anc2box = create_converter(
        inference_v7_cfg.model.name, model, inference_v7_cfg.model.anchor, inference_v7_cfg.image_size, device
    )
    return anc2box


@pytest.fixture(scope="session")
def train_dataloader(train_cfg: Config):
    prepare_dataset(train_cfg.dataset, task="train")
    return create_dataloader(train_cfg.task.data, train_cfg.dataset, train_cfg.task.task)


@pytest.fixture(scope="session")
def validation_dataloader(validation_cfg: Config):
    prepare_dataset(validation_cfg.dataset, task="val")
    return create_dataloader(validation_cfg.task.data, validation_cfg.dataset, validation_cfg.task.task)


@pytest.fixture(scope="session")
def file_stream_data_loader(inference_cfg: Config):
    return StreamDataLoader(inference_cfg.task.data)


@pytest.fixture(scope="session")
def file_stream_data_loader_v7(inference_v7_cfg: Config):
    return StreamDataLoader(inference_v7_cfg.task.data)


@pytest.fixture(scope="session")
def directory_stream_data_loader(inference_cfg: Config):
    inference_cfg.task.data.source = "tests/data/images/train"
    return StreamDataLoader(inference_cfg.task.data)
