from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import yaml


PRESET_DIR = Path(__file__).parent / "presets"

BACKBONE_SIZES = ("vits", "vitb", "vitl", "vitg")


@dataclass
class BackboneConfig:
    name: str = "vits"          # vits | vitb | vitl | vitg
    patch_size: int = 14
    embed_dim: int = 384
    num_heads: int = 6
    depth: int = 12
    img_size: int = 518         # default inference resolution
    pretrained: bool = True
    freeze: bool = True


@dataclass
class DepthHeadConfig:
    type: str = "dpt"
    # HuggingFace repo for the pretrained DA3 DPT head weights.
    # Filled in per-preset; None means random init (for training from scratch).
    hf_repo: Optional[str] = None


@dataclass
class TrainingConfig:
    lr: float = 1e-4
    batch_size: int = 16
    num_epochs: int = 100
    num_workers: int = 8
    # Relative weights for the three DA3 losses (depth, ray, point-cloud)
    loss_weights: dict = field(default_factory=lambda: {
        "depth": 1.0,
        "ray": 0.1,
        "pointcloud": 0.05,
    })


@dataclass
class Config:
    backbone: BackboneConfig = field(default_factory=BackboneConfig)
    depth_head: DepthHeadConfig = field(default_factory=DepthHeadConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)


def _apply_dict(obj: object, d: dict) -> None:
    """Recursively apply a nested dict onto a dataclass instance."""
    for key, value in d.items():
        if not hasattr(obj, key):
            raise ValueError(f"Unknown config field: '{key}'")
        if isinstance(value, dict):
            _apply_dict(getattr(obj, key), value)
        else:
            setattr(obj, key, value)


def load_config(preset: str) -> Config:
    """Load a named preset (e.g. 'vits') and return a Config instance."""
    if preset not in BACKBONE_SIZES:
        raise ValueError(f"Unknown preset '{preset}'. Choose from {BACKBONE_SIZES}.")
    path = PRESET_DIR / f"{preset}.yaml"
    with open(path) as f:
        overrides = yaml.safe_load(f) or {}
    cfg = Config()
    _apply_dict(cfg, overrides)
    return cfg
