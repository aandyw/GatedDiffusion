import os
import math
import wandb
import logging
import argparse
import json

from PIL import Image
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
from diffusers.optimization import get_scheduler

import transformers
from transformers import CLIPTextModel, CLIPTokenizer

from data.dataset import MagicBrushDataset
from models.mask_unet_model import MaskUNetModel
from pipelines.pipeline_gated_diffusion import GatedDiffusionPipeline
from utils import scale_images

WANDB_TABLE_COL_NAMES = ["original_image", "edited_image", "masks", "final_mask", "edit_prompt"]

logger = get_logger(__name__, log_level="INFO")


def main(
    run_name: str,
    config: DictConfig,
):
    model_args = config.model
    train_args = config.train

    logging_dir = os.path.join(model_args.output_dir, config.logging.dir)
    logging_dir = os.path.join(logging_dir, run_name)

    if model_args.output_dir is not None:
        os.makedirs(logging_dir, exist_ok=True)
        os.makedirs(os.path.join(logging_dir, "checkpoints"), exist_ok=True)
        os.makedirs(os.path.join(logging_dir, "models"), exist_ok=True)

    accelerator_project_config = ProjectConfiguration(project_dir=logging_dir, logging_dir=logging_dir)
    accelerator = Accelerator(
        log_with="wandb",
        gradient_accumulation_steps=train_args.gradient_accumulation_steps,
        mixed_precision=train_args.mixed_precision,
        project_config=accelerator_project_config,
    )
    accelerator.init_trackers(
        project_name=config.logging.wandb_project,
        config=OmegaConf.to_container(config),
        init_kwargs={"wandb": {"name": run_name, "dir": logging_dir}},
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

    with open(os.path.join(config.data.val_path, "prompts.json"), "r") as json_file:
        validation_dataset = json.load(json_file)

    if validation_dataset is None:
        raise ValueError(f"Problem with val_path {config.data.val_path}")

    validation_images = [
        Image.open(os.path.join(config.data.val_path, dir, "source.jpg")).convert("RGB").resize((256, 256))
        for dir in validation_dataset.keys()
    ]
    validation_prompts = validation_dataset.values()

    # Load scheduler, tokenizer and models.
    noise_scheduler = DDPMScheduler.from_pretrained(model_args.model_path, subfolder="scheduler")
    text_encoder = CLIPTextModel.from_pretrained(model_args.model_path, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(model_args.model_path, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(model_args.model_path, subfolder="unet")
    mask_unet = MaskUNetModel.from_pretrained(model_args.model_path, subfolder="unet")

    # Freeze vae and text_encoder
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)

    if train_args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
        mask_unet.enable_gradient_checkpointing()

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

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / train_args.gradient_accumulation_steps)
    max_train_steps = train_args.num_epochs * num_update_steps_per_epoch

    lr_scheduler = get_scheduler(
        train_args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=train_args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=max_train_steps * accelerator.num_processes,
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
    logging.info(f"  Total optimization steps = {max_train_steps}")

    global_step = 0
    first_epoch = 0

    progress_bar = tqdm(range(global_step, max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")

    for epoch in range(first_epoch, train_args.num_epochs):
        unet.train()
        mask_unet.train()
        train_loss = 0.0
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate([unet, mask_unet]):
                latents = vae.encode(batch["edited_pixel_values"].to(weight_dtype)).latent_dist.sample()
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
                timesteps = timesteps.long()

                x_noisy = noise_scheduler.add_noise(latents, noise, timesteps)  # noisy latents
                encoder_hidden_states = text_encoder(batch["prompts"])[0]

                # Get the additional image embedding for conditioning.
                # Instead of getting a diagonal Gaussian here, we simply take the mode.
                source_encoded = vae.encode(
                    batch["source_pixel_values"].to(weight_dtype)
                ).latent_dist.mode()  # original_image_embeds
                source_noisy = noise_scheduler.add_noise(source_encoded, noise, timesteps)

                # Concatenate the `source_encoded` with the `x_noisy`.
                concatenated_noisy_latents = torch.cat([x_noisy, source_encoded], dim=1)

                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                mask = mask_unet(concatenated_noisy_latents, timesteps, encoder_hidden_states).mask

                noise_hat = unet(concatenated_noisy_latents, timesteps, encoder_hidden_states).sample

                noise_tilde = x_noisy - source_encoded
                # TODO: scale noise_tilde
                noise_hat = mask * noise_hat + (1.0 - mask) * noise_tilde

                loss = F.mse_loss(noise_hat, target.float(), reduction="mean")

                avg_loss = accelerator.gather(loss.repeat(train_args.batch_size)).mean()
                train_loss += avg_loss.item() / train_args.gradient_accumulation_steps

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    # TODO: clip both models
                    accelerator.clip_grad_norm_(mask_unet.parameters(), train_args.max_grad_norm)
                    # accelerator.clip_grad_norm_(unet.parameters(), train_args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                # log images in diffusion process
                log_images = {
                    "edited_images": batch["edited_pixel_values"],
                    "source_images": batch["source_pixel_values"],
                    "edited_noisy": scale_images(x_noisy),
                    "source_noisy": scale_images(source_noisy),
                    "source_encoded": scale_images(source_encoded),
                }

                wandb_images = []
                for k, tensor in log_images.items():
                    wandb_img = wandb.Image(tensor, caption=k)
                    wandb_images.append(wandb_img)

                accelerator.log({"train_loss": train_loss, "training_images": wandb_images}, step=global_step)
                train_loss = 0.0

                if global_step % train_args.checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        save_path = os.path.join(os.path.join(logging_dir, "checkpoints"), f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

            logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)

        if accelerator.is_main_process and epoch % train_args.validation_epochs == 0:
            logger.info("Running validation...")

            pipeline = GatedDiffusionPipeline.from_pretrained(
                model_args.model_path,
                unet=accelerator.unwrap_model(unet),
                mask_unet=accelerator.unwrap_model(mask_unet),
                text_encoder=accelerator.unwrap_model(text_encoder),
                vae=accelerator.unwrap_model(vae),
                safety_checker=None,
                torch_dtype=weight_dtype,
            )
            pipeline = pipeline.to(accelerator.device)
            pipeline.set_progress_bar_config(disable=True)

            # run inference on single iamge
            edited_images = []
            with torch.autocast(
                str(accelerator.device).replace(":0", ""), enabled=accelerator.mixed_precision == "fp16"
            ):
                for validation_prompt, validation_image in zip(validation_prompts, validation_images):
                    result = pipeline(
                        noise_scheduler=noise_scheduler,
                        prompt=validation_prompt,
                        image=validation_image,
                        num_inference_steps=20,
                        image_guidance_scale=1.5,
                        guidance_scale=7,
                        generator=generator,
                    )
                    edited_image = result.images[0]
                    edited_images.append(
                        (
                            validation_image,
                            edited_image,
                            validation_prompt,
                            result.masks[len(result.masks) // 2 - 1],
                            result.final_mask[0],
                        )
                    )

            for tracker in accelerator.trackers:
                if tracker.name == "wandb":
                    wandb_table = wandb.Table(columns=WANDB_TABLE_COL_NAMES)
                    for (
                        validation_image,
                        edited_image,
                        validation_prompt,
                        masks,
                        final_mask,
                    ) in edited_images:
                        wandb_table.add_data(
                            wandb.Image(validation_image),
                            wandb.Image(edited_image),
                            wandb.Image(masks),
                            wandb.Image(final_mask),
                            validation_prompt,
                        )
                    tracker.log({"validation": wandb_table})

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        pipeline = GatedDiffusionPipeline.from_pretrained(
            model_args.model_path,
            unet=accelerator.unwrap_model(unet),
            mask_unet=accelerator.unwrap_model(mask_unet),
            text_encoder=accelerator.unwrap_model(text_encoder),
            vae=accelerator.unwrap_model(vae),
            safety_checker=None,
            torch_dtype=weight_dtype,
        )
        pipeline.save_pretrained(os.path.join(logging_dir, "models"))

        edited_images = []
        pipeline = pipeline.to(accelerator.device)
        with torch.autocast(str(accelerator.device).replace(":0", "")):
            for validation_prompt, validation_image in zip(validation_prompts, validation_images):
                result = pipeline(
                    noise_scheduler=noise_scheduler,
                    prompt=validation_prompt,
                    image=validation_image,
                    num_inference_steps=20,
                    image_guidance_scale=1.5,
                    guidance_scale=7,
                    generator=generator,
                )
                edited_image = result.images[0]
                edited_images.append(
                    (
                        validation_image,
                        edited_image,
                        validation_prompt,
                        result.masks[len(result.masks) // 2 - 1],
                        result.final_mask[0],
                    )
                )

        for tracker in accelerator.trackers:
            if tracker.name == "wandb":
                wandb_table = wandb.Table(columns=WANDB_TABLE_COL_NAMES)
                for (
                    validation_image,
                    edited_image,
                    validation_prompt,
                    masks,
                    final_mask,
                ) in edited_images:
                    wandb_table.add_data(
                        wandb.Image(validation_image),
                        wandb.Image(edited_image),
                        validation_prompt,
                        wandb.Image(masks),
                        wandb.Image(final_mask),
                    )
                tracker.log({"test": wandb_table})

    accelerator.end_training()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    config = OmegaConf.load(args.config)
    main(run_name=args.name, config=config)
