import sys
from pathlib import Path

import pytest
import torch
from hydra import compose, initialize
from omegaconf import OmegaConf

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

from yolo.config.config import Config
from yolo.model.yolo import YOLO, create_model

config_path = "../../yolo/config"
config_name = "config"


def test_build_model_v9c():
    with initialize(config_path=config_path, version_base=None):
        cfg: Config = compose(config_name=config_name)

        OmegaConf.set_struct(cfg.model, False)
        cfg.weight = None
        model = YOLO(cfg.model)
        assert len(model.model) == 39


def test_build_model_v9m():
    with initialize(config_path=config_path, version_base=None):
        cfg: Config = compose(config_name=config_name, overrides=[f"model=v9-m"])

        OmegaConf.set_struct(cfg.model, False)
        cfg.weight = None
        model = YOLO(cfg.model)
        assert len(model.model) == 39


def test_build_model_v7():
    with initialize(config_path=config_path, version_base=None):
        cfg: Config = compose(config_name=config_name, overrides=[f"model=v7"])

        OmegaConf.set_struct(cfg.model, False)
        cfg.weight = None
        model = YOLO(cfg.model)
        assert len(model.model) == 106


@pytest.fixture
def cfg() -> Config:
    with initialize(config_path="../../yolo/config", version_base=None):
        cfg: Config = compose(config_name="config")
        cfg.weight = None
    return cfg


@pytest.fixture
def model(cfg: Config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_model(cfg.model, cfg=cfg, weight_path=None)
    return model.to(device)


def test_model_basic_status(model):
    assert isinstance(model, YOLO)
    assert len(model.model) == 39


def test_yolo_forward_output_shape(model):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 2 - batch size, 3 - number of channels, 640x640 - image dimensions
    dummy_input = torch.rand(2, 3, 640, 640, device=device)

    # Forward pass through the model
    output = model(dummy_input)
    output_shape = [(cls.shape, anc.shape, box.shape) for cls, anc, box in output["Main"]]
    assert output_shape == [
        (torch.Size([2, 80, 80, 80]), torch.Size([2, 16, 4, 80, 80]), torch.Size([2, 4, 80, 80])),
        (torch.Size([2, 80, 40, 40]), torch.Size([2, 16, 4, 40, 40]), torch.Size([2, 4, 40, 40])),
        (torch.Size([2, 80, 20, 20]), torch.Size([2, 16, 4, 20, 20]), torch.Size([2, 4, 20, 20])),
    ]
