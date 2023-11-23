import os
import math
import wandb
import logging
import argparse

from tqdm.auto import tqdm
from omegaconf import OmegaConf
from omegaconf import DictConfig
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration, set_seed
from accelerate.logging import get_logger

import torch
import torch.nn as nn
import torch.nn.functional as F

import diffusers
from diffusers import AutoencoderKL, DDIMScheduler, DDPMScheduler
from diffusers.models import UNet2DConditionModel
from diffusers.pipelines import StableDiffusionPipeline
from diffusers.optimization import get_scheduler

import transformers
from transformers import CLIPTextModel

from data.dataset import MagicBrushDataset
from models.mask_model import MaskModel

WANDB_TABLE_COL_NAMES = ["original_image", "edited_image", "edit_prompt"]

logger = get_logger(__name__, log_level="INFO")


def main(
    run_name: str,
    config: DictConfig,
):
    model_args = config.model
    train_args = config.train

    wandb.init(project=config.logging.wandb_project, name=run_name, dir=config.logging.dir)

    logging_dir = os.path.join(model_args.output_dir, config.logging.dir)
    accelerator_project_config = ProjectConfiguration(project_dir=model_args.output_dir, logging_dir=logging_dir)
    accelerator = Accelerator(
        gradient_accumulation_steps=train_args.gradient_accumulation_steps,
        mixed_precision=train_args.mixed_precision,
        project_config=accelerator_project_config,
    )

    generator = torch.Generator(device=accelerator.device).manual_seed(train_args.seed)

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)

    if train_args.seed is not None:
        set_seed(train_args.seed)

    if accelerator.is_main_process:
        if model_args.output_dir is not None:
            os.makedirs(model_args.output_dir, exist_ok=True)

    def collate_fn(batch):
        source_pixel_values = torch.stack([sample["source"] for sample in batch])
        source_pixel_values = source_pixel_values.to(memory_format=torch.contiguous_format).float()
        edited_pixel_values = torch.stack([sample["edited"] for sample in batch])
        edited_pixel_values = edited_pixel_values.to(memory_format=torch.contiguous_format).float()
        prompts = torch.stack([sample["prompt"] for sample in batch])
        return {
            "source_pixel_values": source_pixel_values,
            "edited_pixel_values": edited_pixel_values,
            "prompts": prompts,
        }

    train_dataset = MagicBrushDataset(
        config.data.path, cache_dir=model_args.output_dir, model_path=model_args.model_path
    )
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, shuffle=True, collate_fn=collate_fn, batch_size=train_args.batch_size
    )

    # Load scheduler, tokenizer and models.
    noise_scheduler = DDPMScheduler.from_pretrained(model_args.model_path, subfolder="scheduler")
    text_encoder = CLIPTextModel.from_pretrained(model_args.model_path, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(model_args.model_path, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(model_args.model_path, subfolder="unet")
    mask_unet = MaskModel(model_args.model_path, train_args.gradient_checkpointing)

    logging.info("Initializing the InstructPix2Pix UNet from the pretrained UNet.")
    in_channels = 12
    out_channels = unet.conv_in.out_channels
    unet.register_to_config(in_channels=in_channels)

    with torch.no_grad():
        new_conv_in = nn.Conv2d(
            in_channels, out_channels, unet.conv_in.kernel_size, unet.conv_in.stride, unet.conv_in.padding
        )
        new_conv_in.weight.zero_()
        new_conv_in.weight[:, :8, :, :].copy_(unet.conv_in.weight)
        unet.conv_in = new_conv_in

    # Freeze vae and text_encoder
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)

    if train_args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    # create hooks for saving and loading model
    def save_model_hook(models, weights, output_dir):
        if accelerator.is_main_process:
            for i, model in enumerate(models):
                model.save_pretrained(os.path.join(output_dir, "unet"))

                # make sure to pop weight so that corresponding model is not saved again
                weights.pop()

    def load_model_hook(models, input_dir):
        for i in range(len(models)):
            # pop models so that they are not loaded again
            model = models.pop()

            # load diffusers style into model
            load_model = UNet2DConditionModel.from_pretrained(input_dir, subfolder="unet")
            model.register_to_config(**load_model.config)

            model.load_state_dict(load_model.state_dict())
            del load_model

    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    params = list(unet.parameters()) + list(mask_unet.parameters())
    optimizer = torch.optim.AdamW(
        params,
        lr=train_args.learning_rate,
        betas=(train_args.adam_beta1, train_args.adam_beta2),
        weight_decay=train_args.adam_weight_decay,
        eps=train_args.adam_epsilon,
    )

    lr_scheduler = get_scheduler(
        train_args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=train_args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=train_args.max_train_steps * accelerator.num_processes,
    )

    unet, mask_unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet, mask_unet, optimizer, train_dataloader, lr_scheduler
    )

    torch.cuda.empty_cache()

    text_encoder.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)

    if accelerator.is_main_process:
        accelerator.init_trackers("instruct-pix2pix", config=vars(args))

    total_batch_size = train_args.batch_size * accelerator.num_processes * train_args.gradient_accumulation_steps

    logging.info("***** Running training *****")
    logging.info(f"  Num examples = {len(train_dataset)}")
    logging.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logging.info(f"  Num Epochs = {train_args.num_epochs}")
    logging.info(f"  Instantaneous batch size per device = {train_args.batch_size}")
    logging.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logging.info(f"  Gradient Accumulation steps = {train_args.gradient_accumulation_steps}")
    logging.info(f"  Total optimization steps = {train_args.max_train_steps}")

    global_step = 0
    first_epoch = 0

    # TODO: finish training loop
    progress_bar = tqdm(range(global_step, train_args.max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")

    for epoch in range(first_epoch, train_args.num_epochs):
        unet.train()
        mask_unet.train()
        train_loss = 0.0
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(unet):
                latents = vae.encode(batch["edited_pixel_values"].to(weight_dtype)).latent_dist.sample()
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
                timesteps = timesteps.long()

                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                encoder_hidden_states = text_encoder(batch["prompts"])[0]
                original_image_embeds = vae.encode(batch["source_pixel_values"].to(weight_dtype)).latent_dist.mode()

                # Concatenate the `original_image_embeds` with the `noisy_latents`.
                concatenated_noisy_latents = torch.cat([noisy_latents, original_image_embeds], dim=1)

                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                mask = mask_unet(concatenated_noisy_latents, timesteps, encoder_hidden_states).mask

                concatenated_noisy_latents = torch.cat([noisy_latents, original_image_embeds, mask], dim=1)

                noise_hat = unet(concatenated_noisy_latents, timesteps, encoder_hidden_states).sample

                noise_tilde = noisy_latents - original_image_embeds  # x_noisy - src_encoded
                noise_hat = mask * noise_hat + (1.0 - mask) * noise_tilde

                loss = F.mse_loss(noise_hat, target.float(), reduction="mean")

                avg_loss = accelerator.gather(loss.repeat(train_args.batch_size)).mean()
                train_loss += avg_loss.item() / train_args.gradient_accumulation_steps

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(unet.parameters(), train_args.max_grad_norm)
                    # NOTE: clip grads of mask unet?
                    # accelerator.clip_grad_norm_(mask_unet.parameters(), train_args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": train_loss}, step=global_step)
                train_loss = 0.0

                if global_step % train_args.checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        save_path = os.path.join(model_args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

            logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)

        # TODO: inference pipeline

        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            # TODO: save
            pass

    accelerator.end_training()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    config = OmegaConf.load(args.config)
    main(run_name=args.name, config=config)
