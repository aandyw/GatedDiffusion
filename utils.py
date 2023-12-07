import sys
from PIL import Image
from typing import List

import torch
import torch.nn.functional as F

from diffusers import AutoencoderKL, DDIMScheduler, DDPMScheduler


def scale_images(images: torch.FloatTensor, image_size: int = 256) -> torch.FloatTensor:
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


def extract_noise(
    noise_scheduler: DDPMScheduler,
    x_noisy: torch.FloatTensor,
    source_encoded: torch.FloatTensor,
    timesteps: torch.IntTensor,
) -> torch.FloatTensor:
    """
    Inverse process of noise_scheduler.add_noise
    """
    alphas_cumprod = noise_scheduler.alphas_cumprod.to(device=x_noisy.device, dtype=x_noisy.dtype)

    timesteps = timesteps.long()
    sqrt_alpha_prod = alphas_cumprod[timesteps] ** 0.5
    sqrt_alpha_prod = sqrt_alpha_prod.flatten()
    while len(sqrt_alpha_prod.shape) < len(source_encoded.shape):
        sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)

    sqrt_one_minus_alpha_prod = (1 - alphas_cumprod[timesteps]) ** 0.5
    sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
    while len(sqrt_one_minus_alpha_prod.shape) < len(source_encoded.shape):
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)

    noise_tilde = (x_noisy - sqrt_alpha_prod * source_encoded) / sqrt_one_minus_alpha_prod
    return noise_tilde
