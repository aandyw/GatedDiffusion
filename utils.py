import sys
from PIL import Image
from typing import List

import torch
import torch.nn.functional as F

from diffusers import AutoencoderKL, DDIMScheduler, DDPMScheduler


def scale_images(images: torch.FloatTensor, image_size: int) -> torch.FloatTensor:
    scaled_images = F.interpolate(
        images.to(torch.float32),
        size=(image_size, image_size),
        mode="bilinear",
        align_corners=False,
    )
    return scaled_images


def visualize_all_masks(masks: List[torch.FloatTensor]):
    widths, heights = zip(*(i.size for i in masks))
    total_width = sum(widths)
    max_height = max(heights)

    new_im = Image.new("RGB", (total_width, max_height))

    x_offset = 0
    for im in masks:
        new_im.paste(im, (x_offset, 0))
        x_offset += im.size[0]
    return new_im
