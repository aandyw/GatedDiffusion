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

import logging
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
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    UNet2DConditionModel,
    StableDiffusionInstructPix2PixPipeline,
)
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from diffusers.utils import check_min_version, deprecate, is_wandb_available
from diffusers.utils.import_utils import is_xformers_available
from packaging import version
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

from modules.mask import Mask
from modules.model import Model

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# DATASET_NAME_MAPPING = {
#     "osunlp/MagicBrush": (
#         "source_img",
#         "mask_img",
#         "instruction",
#         "target_img",
#     ),
# }
WANDB_TABLE_COL_NAMES = ["source_img", "instruction", "target_img"]

pretrained_model_id = "timbrooks/instruct-pix2pix"
dataset_name = "osunlp/MagicBrush"
original_image_column = "source_img"
edit_prompt_column = "instruction"
edited_image_column = "target_img"
weight_dtype = torch.float32  # mixed precision with 16-bit
resolution = 512
batch_size = 4
device = "cuda"


def main():
    # Load data
    dataset = datasets.load_dataset(dataset_name, cache_dir="data")

    ### Preprocessing data ###
    def tokenize_captions(captions):
        inputs = tokenizer(
            captions,
            max_length=tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return inputs.input_ids

    train_transforms = transforms.Compose(
        [
            transforms.CenterCrop(resolution),
            transforms.Lambda(lambda x: x),
        ]
    )

    def convert_to_np(image, resolution):
        image = image.convert("RGB").resize((resolution, resolution))
        return np.array(image).transpose(2, 0, 1)

    def preprocess_images(samples):
        original_images = np.concatenate(
            [
                convert_to_np(image, resolution)
                for image in samples[original_image_column]
            ]
        )
        edited_images = np.concatenate(
            [convert_to_np(image, resolution) for image in samples[edited_image_column]]
        )
        # We need to ensure that the original and the edited images undergo the same
        # augmentation transforms.
        images = np.concatenate([original_images, edited_images])
        images = torch.tensor(images)
        images = 2 * (images / 255) - 1
        return train_transforms(images)

    def preprocess_train(samples):
        # Preprocess images.
        preprocessed_images = preprocess_images(samples)
        # Since the original and edited images were concatenated before
        # applying the transformations, we need to separate them and reshape
        # them accordingly.
        original_images, edited_images = preprocessed_images.chunk(2)
        original_images = original_images.reshape(-1, 3, resolution, resolution)
        edited_images = edited_images.reshape(-1, 3, resolution, resolution)

        # Collate the preprocessed images into the `examples`.
        samples["original_pixel_values"] = original_images
        samples["edited_pixel_values"] = edited_images

        # Preprocess the captions.
        captions = [caption for caption in samples[edit_prompt_column]]
        samples["input_ids"] = tokenize_captions(captions)
        return samples

    def collate_fn(examples):
        original_pixel_values = torch.stack(
            [example["original_pixel_values"] for example in examples]
        )
        original_pixel_values = original_pixel_values.to(
            memory_format=torch.contiguous_format
        ).float()
        edited_pixel_values = torch.stack(
            [example["edited_pixel_values"] for example in examples]
        )
        edited_pixel_values = edited_pixel_values.to(
            memory_format=torch.contiguous_format
        ).float()
        input_ids = torch.stack([example["input_ids"] for example in examples])
        return {
            "original_pixel_values": original_pixel_values,
            "edited_pixel_values": edited_pixel_values,
            "input_ids": input_ids,
        }

    train_dataset = dataset["train"].with_transform(preprocess_train)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, shuffle=True, collate_fn=collate_fn, batch_size=batch_size
    )

    # Load scheduler, tokenizer and models.
    noise_scheduler = DDPMScheduler.from_pretrained(
        pretrained_model_id, subfolder="scheduler"
    )
    tokenizer = CLIPTokenizer.from_pretrained(
        pretrained_model_id, subfolder="tokenizer"
    )
    text_encoder = CLIPTextModel.from_pretrained(
        pretrained_model_id, subfolder="text_encoder"
    )
    vae = AutoencoderKL.from_pretrained(pretrained_model_id, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(pretrained_model_id, subfolder="unet")

    # Freeze vae and text_encoder
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)

    # pipeline = StableDiffusionInstructPix2PixPipeline.from_pretrained(
    #     pretrained_model_id,
    #     torch_dtype=weight_dtype,
    # )
    # pipeline = pipeline.to(device)
    vae = vae.to(device)
    text_encoder = text_encoder.to(device)
    unet = unet.to(device)

    # Initialize learnable mask
    gated_unet_mask = UNet2DConditionModel.from_pretrained(
        pretrained_model_id, subfolder="unet"
    )
    mask = Mask(gated_unet_mask)
    mask = mask.to(device)

    # Create model
    model = Model(mask, unet).to(device)

    ### Training ###
    num_epochs = 100
    learning_rate = 1e-6
    gradient_accumulation_steps = 4

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num epochs = {num_epochs}")
    logger.info(f"  Batch size = {batch_size}")
    global_step = 0

    progress_bar = tqdm(
        range(
            global_step,
            num_epochs * len(train_dataset),
        )
    )
    progress_bar.set_description("Steps")

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for step, batch in enumerate(train_dataloader):
            batch["original_pixel_values"] = batch["original_pixel_values"].to(device)
            batch["edited_pixel_values"] = batch["edited_pixel_values"].to(device)
            batch["input_ids"] = batch["input_ids"].to(device)

            optimizer.zero_grad()

            latents = vae.encode(
                batch["edited_pixel_values"].to(weight_dtype)
            ).latent_dist.sample()
            latents = latents * vae.config.scaling_factor

            noise = torch.randn_like(latents)
            bsz = latents.shape[0]

            # Sample random timestep for each image
            timesteps = torch.randint(
                0, noise_scheduler.num_train_timesteps, (bsz,), device=latents.device
            )
            timesteps = timesteps.long()

            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            encoder_hidden_states = text_encoder(batch["input_ids"])[0]

            original_image_embeds = vae.encode(
                batch["original_pixel_values"].to(weight_dtype)
            ).latent_dist.mode()

            concatenated_noisy_latents = torch.cat(
                [noisy_latents, original_image_embeds], dim=1
            )

            if noise_scheduler.config.prediction_type == "epsilon":
                target = noise
            elif noise_scheduler.config.prediction_type == "v_prediction":
                target = noise_scheduler.get_velocity(latents, noise, timesteps)
            else:
                raise ValueError(
                    f"Unknown prediction type {noise_scheduler.config.prediction_type}"
                )

            # Predict noise residual and compute loss
            model_pred = model(
                original_image_embeds,
                concatenated_noisy_latents,
                timesteps,
                encoder_hidden_states,
            )
            loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
            # Backpropagate
            loss = loss / gradient_accumulation_steps
            loss.backward()

            if step % gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            # log losses
            train_loss += loss.item()
            progress_bar.update(1)

        avg_loss = train_loss / len(train_dataloader)
        logger.info(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_loss}")


if __name__ == "__main__":
    main()
