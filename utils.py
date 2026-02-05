import os
import json
import time
import random
from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple

import numpy as np
import torch
from torchvision.utils import make_grid, save_image


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def timestamp() -> str:
    return time.strftime("%Y%m%d-%H%M%S")


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def save_config(path: str, cfg: Dict[str, Any]):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)


@torch.no_grad()
def save_sample_grid(
    images: torch.Tensor,
    out_path: str,
    nrow: int = 8
):
    # images assumed in [-1, 1]
    grid = make_grid(images, nrow=nrow, normalize=True, value_range=(-1, 1))
    save_image(grid, out_path)


class EMA:
    """
    Optional: Exponential Moving Average for Generator weights
    (Simple, improves sample quality for demos).
    """
    def __init__(self, model: torch.nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {}
        self._register(model)

    def _register(self, model: torch.nn.Module):
        for name, p in model.named_parameters():
            if p.requires_grad:
                self.shadow[name] = p.detach().clone()

    @torch.no_grad()
    def update(self, model: torch.nn.Module):
        for name, p in model.named_parameters():
            if p.requires_grad:
                self.shadow[name].mul_(self.decay).add_(p.detach(), alpha=1 - self.decay)

    @torch.no_grad()
    def apply_to(self, model: torch.nn.Module):
        for name, p in model.named_parameters():
            if p.requires_grad:
                p.copy_(self.shadow[name])
