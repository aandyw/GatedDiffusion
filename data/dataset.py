from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torchvision
from einops import rearrange
from PIL import Image
from torch.utils.data import Dataset, DataLoader

from datasets import load_dataset


class MagicBrushDataset(Dataset):
    def __init__(
        self,
        path: str,
        cache_dir: str,
        split: str = "train",
        min_resize_res: int = 256,
        max_resize_res: int = 256,
        crop_res: int = 256,
        flip_prob: float = 0.0,
    ):
        self.dataset = load_dataset(path, cache_dir=cache_dir, split=split)
        self.min_resize_res = min_resize_res
        self.max_resize_res = max_resize_res
        self.crop_res = crop_res
        self.flip_prob = flip_prob

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, i: int) -> dict[str, Any]:
        sample = self.dataset[i]
        image_0 = sample["source_img"]
        image_1 = sample["target_img"]
        prompt = sample["instruction"]

        reize_res = torch.randint(self.min_resize_res, self.max_resize_res + 1, ()).item()
        image_0 = image_0.resize((reize_res, reize_res), Image.Resampling.LANCZOS)
        image_1 = image_1.resize((reize_res, reize_res), Image.Resampling.LANCZOS)

        image_0 = rearrange(2 * torch.tensor(np.array(image_0)).float() / 255 - 1, "h w c -> c h w")
        image_1 = rearrange(2 * torch.tensor(np.array(image_1)).float() / 255 - 1, "h w c -> c h w")

        crop = torchvision.transforms.RandomCrop(self.crop_res)
        flip = torchvision.transforms.RandomHorizontalFlip(float(self.flip_prob))
        image_0, image_1 = flip(crop(torch.cat((image_0, image_1)))).chunk(2)

        # TODO: remove improper duplicate samples
        if image_0.shape != (3, self.crop_res, self.crop_res):
            return self.__getitem__(i + 1)

        return dict(edited=image_1, edit=dict(source=image_0, prompt=prompt))
