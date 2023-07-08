#!/usr/bin/env python
# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Script to fine-tune InstructPix2Pix."""

import argparse
import logging
import math
import os
from pathlib import Path
from typing import Optional

import datasets
import diffusers
import numpy as np
import PIL
import requests
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from diffusers import (AutoencoderKL, DDPMScheduler, UNet2DConditionModel)
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from diffusers.utils import check_min_version, deprecate, is_wandb_available
from diffusers.utils.import_utils import is_xformers_available
from packaging import version
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

from pipeline import StableDiffusionInstructPix2PixPipeline


DATASET_NAME_MAPPING = {
    "osunlp/MagicBrush": (
        "source_img",
        "mask_img",
        "instruction",
        "target_img",
    ),
}
WANDB_TABLE_COL_NAMES = ["source_img", "instruction", "target_img"]

pretrained_model_id = "timbrooks/instruct-pix2pix"
dataset_name = "osunlp/MagicBrush"
weight_dtype = torch.float16 # mixed precision
device = "cuda"


dataset = datasets.load_dataset(dataset_name, cache_dir="data")

# Load scheduler, tokenizer and models.
noise_scheduler = DDPMScheduler.from_pretrained(
    pretrained_model_id, subfolder="scheduler"
)
tokenizer = CLIPTokenizer.from_pretrained(
    pretrained_model_id,
    subfolder="tokenizer"
)
text_encoder = CLIPTextModel.from_pretrained(
    pretrained_model_id,
    subfolder="text_encoder"
)
vae = AutoencoderKL.from_pretrained(
    pretrained_model_id, subfolder="vae"
)
unet = UNet2DConditionModel.from_pretrained(
    pretrained_model_id,
    subfolder="unet"
)

# Freeze vae and text_encoder
vae.requires_grad_(False)
text_encoder.requires_grad_(False)


pipeline = StableDiffusionInstructPix2PixPipeline.from_pretrained(
    pretrained_model_id,
    unet=unet,
    torch_dtype=weight_dtype,
)
pipeline = pipeline.to(device)
pipeline.set_progress_bar_config(disable=True)
