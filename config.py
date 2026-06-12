import yaml
from os import path
from dataclasses import dataclass
from typing import Optional, Tuple, Iterator, Dict, List
from importlib import import_module

import torch
from torch.nn import Parameter, Module
from torch.utils.data import DataLoader, ConcatDataset

from dataset import VIFDataset, read_and_split, ImageLabels

# -----------------------------
# Configuration objects
# -----------------------------
class ModelConfig:
    def __init__(self, target: str, params: dict, pretrained: Optional[str]=None) -> None:
        self.target: str = target
        self.params = {}
        for k, v in params.items():
            if self.check(v):
                self.params[k] = ModelConfig(**v)
            else:
                self.params[k] = v
        self.pretrained: Optional[str] = pretrained

    @staticmethod
    def check(param: dict) -> bool:
        return isinstance(param, dict) and {'target', 'params'}.issubset(param.keys())

    def __repr__(self) -> str:
        cls = self.__class__.__name__
        return f'{cls}(target={self.target}, params={self.params}, pretrained={self.pretrained})'

class TrainArgs:
    def __init__(self,
        epochs: int,
        accumulation_steps: int,
        optimizer: dict,
    ) -> None:
        self.epochs = epochs
        self.accumulation_steps = accumulation_steps
        self.optimizer = ModelConfig(**optimizer)
    def __repr__(self) -> str:
        return '\n'.join([
            'TrainArgs(',
            f'epochs={self.epochs}',
            f'accumulation_steps={self.accumulation_steps}',
            f'optimizer={self.optimizer}'
        ])

@dataclass
class DatasetArgs:
    """
    url: { "MSRS": "...", "Road": "...", "TNO": "..." }
    ratio: (train, valid) or (train, valid, test)
    """
    url: Dict[str, str]
    batch_size: int
    shuffle: bool
    ratio: Optional[Tuple[float, float] | Tuple[float, float, float]] = None
    num_workers: int = 4
    pin_memory: bool = True
    drop_last: bool = False

@dataclass
class TrainConfig:
    model: ModelConfig
    train: TrainArgs
    dataset: DatasetArgs


# -----------------------------
# Load configuration
# -----------------------------
def load_config(file: str='/home/raytrack/Fusion/CMfused11/config/model.yaml') -> TrainConfig:
    config_path = path.abspath(file)
    print(f"使用的配置文件路径: {config_path}")
    assert path.isfile(file), f"配置文件不存在: {file}"
    with open(file) as f:
        config = yaml.full_load(f)
    model = ModelConfig(**config['model'])
    train = TrainArgs(**config['train'])
    dataset = DatasetArgs(**config['dataset'])
    return TrainConfig(model=model, train=train, dataset=dataset)


# -----------------------------
# Build the model dynamically
# -----------------------------
def load_model(config: ModelConfig, device: torch.device):
    kwargs = {}
    module_name, cls_name = config.target.rsplit('.', 1)
    print(f"[Model] {module_name}.{cls_name}")
    module = import_module(module_name)
    cls = getattr(module, cls_name)

    # Recursively construct submodules
    for k, v in config.params.items():
        kwargs[k] = load_model(v, device) if isinstance(v, ModelConfig) else v

    model: Module = cls(**kwargs)
    if config.pretrained and path.isfile(config.pretrained) and hasattr(model, 'load_state_dict'):
        print(f'Loading pretrained weights from {config.pretrained}')
        state_dict = torch.load(config.pretrained, map_location='cpu')
        model.load_state_dict(state_dict, strict=False)
    return model.to(device)


# -----------------------------
# Load datasets (supports MSRS / Road / TNO)
# -----------------------------
def load_dataset(config: DatasetArgs):
    """
    Returns:
      - [train_loader] when only the training split is available
      - [train_loader, valid_loader] when a validation split is available
    """
    # Allowed keys: any combination can be defined in YAML, and missing ones are ignored
    supported_keys = ['CTMRI', 'SPECTMRI', 'PETMRI']

    # Unified split ratios: default to (0.8, 0.2) if not provided
    ratios = config.ratio if config.ratio is not None else (0.8, 0.2)

    train_dsets: List[torch.utils.data.Dataset] = []
    valid_dsets: List[torch.utils.data.Dataset] = []

    for key in supported_keys:
        root = config.url.get(key, None)
        if not root:
            continue
        print(f"[Dataset] Loading {key} from {root} with ratios={ratios}")

        # read_and_split should return ImageLabels(train, valid, test?) based on ratios
        labels: ImageLabels = read_and_split(root, ratios=ratios)

        # Training split
        if labels.train:
            train_dsets.append(VIFDataset(labels.train))

        # Validation split (read_and_split may return None if valid is absent)
        if getattr(labels, 'valid', None):
            valid_dsets.append(VIFDataset(labels.valid))

        # You can also handle the test split if needed:
        # if getattr(labels, 'test', None):
        #     test_dsets.append(VIFDataset(labels.test))

    assert len(train_dsets) > 0, "没有可用的训练数据集，请检查 dataset.url 配置路径是否正确。"

    # Build the DataLoaders
    train_loader = DataLoader(
        ConcatDataset(train_dsets) if len(train_dsets) > 1 else train_dsets[0],
        batch_size=config.batch_size,
        shuffle=config.shuffle,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        drop_last=config.drop_last,
    )

    if len(valid_dsets) > 0:
        valid_loader = DataLoader(
            ConcatDataset(valid_dsets) if len(valid_dsets) > 1 else valid_dsets[0],
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=max(1, config.num_workers // 2),
            pin_memory=config.pin_memory,
            drop_last=False,
        )
        return [train_loader, valid_loader]

    return [train_loader]


# -----------------------------
# Optimizer
# -----------------------------
def load_optimizer(config: ModelConfig, params: Iterator[Parameter]):
    module_name, cls_name = config.target.rsplit('.', 1)
    module = import_module(module_name)
    optim = getattr(module, cls_name)
    return optim(params, **config.params)
