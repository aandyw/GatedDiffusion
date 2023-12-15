import sys
from PIL import Image
from typing import List

import torch
import torch.nn.functional as F


def scale_images(images: torch.FloatTensor, image_size: int) -> torch.FloatTensor:
    scaled_images = F.interpolate(
        images.to(torch.float32),
        size=(image_size, image_size),
        mode="bilinear",
        align_corners=False,
    )
    return scaled_images
