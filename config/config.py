from dataclasses import dataclass
from typing import Dict, List, Union


@dataclass
class Model:
    anchor: List[List[int]]
    model: Dict[str, List[Dict[str, Union[Dict, List, int]]]]


@dataclass
class Download:
    auto: bool
    path: str


@dataclass
class DataLoaderConfig:
    batch_size: int
    shuffle: bool
    num_workers: int
    pin_memory: bool


@dataclass
class OptimizerArgs:
    lr: float
    weight_decay: float


@dataclass
class OptimizerConfig:
    type: str
    args: OptimizerArgs


@dataclass
class SchedulerArgs:
    step_size: int
    gamma: float


@dataclass
class SchedulerConfig:
    type: str
    args: SchedulerArgs


@dataclass
class EMAConfig:
    enabled: bool
    decay: float


@dataclass
class TrainConfig:
    optimizer: OptimizerConfig
    scheduler: SchedulerConfig
    ema: EMAConfig


@dataclass
class HyperConfig:
    data: DataLoaderConfig
    train: TrainConfig


@dataclass
class Dataset:
    file_name: str
    num_files: int


@dataclass
class Datasets:
    base_url: str
    images: Dict[str, Dataset]


@dataclass
class Download:
    auto: bool
    save_path: str
    datasets: Datasets


@dataclass
class Config:
    model: Model
    download: Download
    hyper: HyperConfig
