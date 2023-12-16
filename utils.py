import sys
from PIL import Image
from typing import List

import torch
import torch.nn.functional as F


def scale_tensors(tensors: torch.FloatTensor, image_size: int) -> torch.FloatTensor:
    scaled_tensors = F.interpolate(
        tensors,
        size=(image_size, image_size),
        mode="bilinear",
        align_corners=False,
    )
    return scaled_tensors
