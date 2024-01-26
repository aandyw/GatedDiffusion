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

from diffusers import StableDiffusionDiffEditPipeline
from diffusers import AutoencoderKL, DDIMScheduler, DDPMScheduler, DDIMInverseScheduler
from diffusers.models import UNet2DConditionModel
from diffusers.optimization import get_scheduler

from transformers import CLIPTextModel

from data.dataset import MagicBrushDataset
from models.mask_unet_model import MaskUNetModel
from pipelines.pipeline_gated_diffusion import GatedDiffusionPipeline
from utils import scale_tensors

WANDB_TABLE_COL_NAMES = [
    "original_image",
    "edited_image",
    "edited_image_without_mask",
    "masks",
    "final_mask",
    "edit_prompt",
]


logger = get_logger(__name__, log_level="INFO")

torch.autograd.set_detect_anomaly(True)


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

    if config.inference.method not in ["last", "all", "both", "none"]:
        raise ValueError(f"Problem with inference method {config.inference.method}")

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
    mask_unet = MaskUNetModel.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="unet")

    # Freeze vae and text_encoder
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)

    if not train_args.joint_training:
        unet.requires_grad_(False)

    if train_args.gradient_checkpointing:
        if train_args.joint_training:
            unet.enable_gradient_checkpointing()
        mask_unet.enable_gradient_checkpointing()

    # create hooks for saving and loading model
    def save_model_hook(models, weights, output_dir):
        if accelerator.is_main_process:
            for i, model in enumerate(models):
                if isinstance(model, MaskUNetModel):
                    model.save_pretrained(os.path.join(output_dir, "mask_unet"))
                else:
                    model.save_pretrained(os.path.join(output_dir, "unet"))

                # make sure to pop weight so that corresponding model is not saved again
                weights.pop()

    def load_model_hook(models, input_dir):
        for i in range(len(models)):
            # pop models so that they are not loaded again
            model = models.pop()

            if isinstance(model, MaskUNetModel):
                load_model = MaskUNetModel.from_pretrained(input_dir, subfolder="mask_unet")
            else:
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

    unet_optimizer = torch.optim.AdamW(
        unet.parameters(),
        lr=train_args.unet_learning_rate,
        betas=(train_args.adam_beta1, train_args.adam_beta2),
        weight_decay=train_args.adam_weight_decay,
        eps=train_args.adam_epsilon,
    )

    mask_optimizer = torch.optim.AdamW(
        mask_unet.parameters(),
        lr=train_args.mask_learning_rate,
        betas=(train_args.adam_beta1, train_args.adam_beta2),
        weight_decay=train_args.adam_weight_decay,
        eps=train_args.adam_epsilon,
    )

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / train_args.gradient_accumulation_steps)
    max_train_steps = train_args.num_epochs * num_update_steps_per_epoch

    unet_lr_scheduler = get_scheduler(
        train_args.lr_scheduler,
        optimizer=unet_optimizer,
        num_warmup_steps=train_args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=max_train_steps * accelerator.num_processes,
    )

    mask_lr_scheduler = get_scheduler(
        train_args.lr_scheduler,
        optimizer=mask_optimizer,
        num_warmup_steps=train_args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=max_train_steps * accelerator.num_processes,
    )

    (
        unet,
        mask_unet,
        unet_optimizer,
        mask_optimizer,
        train_dataloader,
        unet_lr_scheduler,
        mask_lr_scheduler,
    ) = accelerator.prepare(
        unet, mask_unet, unet_optimizer, mask_optimizer, train_dataloader, unet_lr_scheduler, mask_lr_scheduler
    )

    torch.cuda.empty_cache()

    text_encoder.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / train_args.gradient_accumulation_steps)
    max_train_steps = train_args.num_epochs * num_update_steps_per_epoch

    total_batch_size = train_args.batch_size * accelerator.num_processes * train_args.gradient_accumulation_steps

    inversed_latents_pipeline = StableDiffusionDiffEditPipeline.from_pretrained(
        "stabilityai/stable-diffusion-2-1", torch_dtype=weight_dtype
    )
    inversed_latents_pipeline = inversed_latents_pipeline.to(accelerator.device)
    inversed_latents_pipeline.inverse_scheduler = DDIMInverseScheduler.from_config(inversed_latents_pipeline.scheduler.config)

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

    if train_args.resume_from_checkpoint:
        path = os.path.join(config.logging.dir, train_args.resume_from_checkpoint)

        if path is None:
            accelerator.print(
                f"Checkpoint '{train_args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            train_args.resume_from_checkpoint = None
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(model_args.output_dir, path))
            global_step = int(path.split("-")[-1])

            resume_global_step = global_step * train_args.gradient_accumulation_steps
            first_epoch = global_step // num_update_steps_per_epoch
            resume_step = resume_global_step % (num_update_steps_per_epoch * train_args.gradient_accumulation_steps)

    progress_bar = tqdm(range(global_step, max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")

    for epoch in range(first_epoch, train_args.num_epochs):
        models = [mask_unet]
        mask_unet.train()

        if train_args.joint_training:
            unet.train()
            models.append(unet)

        train_loss_ip2p = 0.0
        train_loss = 0.0
        for step, batch in enumerate(train_dataloader):
            # Skip steps until we reach the resumed step
            if train_args.resume_from_checkpoint and epoch == first_epoch and step < resume_step:
                if step % train_args.gradient_accumulation_steps == 0:
                    progress_bar.update(1)
                continue

            with accelerator.accumulate(models):
                latents = vae.encode(batch["edited_pixel_values"].to(weight_dtype)).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
                timesteps = timesteps.long()

                x_noisy = noise_scheduler.add_noise(latents, noise, timesteps)  # noisy latents
                encoder_hidden_states = text_encoder(batch["prompts"])[0]

                # Get the additional image embedding for conditioning.
                # Instead of getting a diagonal Gaussian here, we simply take the mode.

                # original_image_embed
                source_encoded = vae.encode(batch["source_pixel_values"].to(weight_dtype)).latent_dist.mode()
                source_noisy = noise_scheduler.add_noise(source_encoded, noise, timesteps)

                # Concatenate the `source_encoded` with the `x_noisy`.
                concatenated_noisy_latents = torch.cat([x_noisy, source_encoded], dim=1)

                # We only want to use epsilon parameterization
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                mask = mask_unet(source_noisy, timesteps, encoder_hidden_states).mask

                model_pred = unet(concatenated_noisy_latents, timesteps, encoder_hidden_states).sample

                # Compute the absolute difference for each pair in the batch
                # TODO: get differences using the provide mask_img in magicbrush dataset? masks provided are loose
                differences = torch.abs(torch.subtract(batch["edited_pixel_values"], batch["source_pixel_values"]))
                differences = scale_tensors(differences, 32)

                # Calculate the mean difference across the color channels
                mean_differences = differences.mean(dim=1)

                # Thresholding to create binary masks for each pair in the batch
                threshold_value = 0.1  # Adjust as needed
                ground_truth_mask = torch.where(mean_differences > threshold_value, 1.0, 0.0)
                ground_truth_mask = ground_truth_mask.unsqueeze(1)
                ground_truth_mask = ground_truth_mask.to(weight_dtype)

                noise_tilde = None
                noise_hat = None
                if train_args.joint_training:
                    noise_tilde = (1.0 - ground_truth_mask) * noise
                    noise_hat = mask * model_pred + (1.0 - mask) * noise_tilde
                    loss_ip2p = F.mse_loss(model_pred, target.float(), reduction="mean")
                    loss = F.mse_loss(noise_hat, target.float(), reduction="mean") + loss_ip2p

                    avg_loss_ip2p = accelerator.gather(loss_ip2p.repeat(train_args.batch_size)).mean()
                    train_loss_ip2p = avg_loss_ip2p.item() / train_args.gradient_accumulation_steps
                else:
                    criterion = nn.BCELoss()
                    loss_ip2p = F.mse_loss(model_pred, target.float(), reduction="mean")
                    loss = criterion(mask, ground_truth_mask.float()) + loss_ip2p

                    avg_loss_ip2p = accelerator.gather(loss_ip2p.repeat(train_args.batch_size)).mean()
                    train_loss_ip2p = avg_loss_ip2p.item() / train_args.gradient_accumulation_steps

                avg_loss = accelerator.gather(loss.repeat(train_args.batch_size)).mean()
                train_loss += avg_loss.item() / train_args.gradient_accumulation_steps

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    params = list(mask_unet.parameters())
                    if train_args.joint_training:
                        params += list(unet.parameters())
                    accelerator.clip_grad_norm_(params, train_args.max_grad_norm)

                mask_optimizer.step()
                mask_lr_scheduler.step()
                mask_optimizer.zero_grad()

                if train_args.joint_training:
                    unet_optimizer.step()
                    unet_lr_scheduler.step()
                    unet_optimizer.zero_grad()

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                # log images in diffusion process
                log_images = {
                    "edited_images": batch["edited_pixel_values"],
                    "source_images": batch["source_pixel_values"],
                    "edited_noisy": scale_tensors(x_noisy, 256),
                    "edited_encoded": scale_tensors(latents, 256),
                    "source_noisy": scale_tensors(source_noisy, 256),
                    "source_encoded": scale_tensors(source_encoded, 256),
                    "noise": scale_tensors(noise, 256) if noise is not None else noise,
                    "noise_tilde": noise_tilde,
                    "noise_hat": scale_tensors(noise_hat, 256) if noise_hat is not None else noise_hat,
                    "mask": scale_tensors(mask, 256),
                    "ground_truth_mask": ground_truth_mask,
                }

                for k, tensor in log_images.items():
                    if tensor is None:
                        continue
                    log_images[k] = [wandb.Image(tensor[i, :, :, :]) for i in range(tensor.shape[0])]

                accelerator.log(
                    {"train_loss": train_loss, "train_loss_ip2p": train_loss_ip2p, **log_images},
                    step=global_step,
                )
                train_loss_ip2p = 0.0
                train_loss = 0.0

                if global_step % train_args.checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        save_path = os.path.join(os.path.join(logging_dir, "checkpoints"), f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

            logs = {
                "step_loss": loss.detach().item(),
                "unet_lr": unet_lr_scheduler.get_last_lr()[0],
                "mask_lr": mask_lr_scheduler.get_last_lr()[0],
            }
            progress_bar.set_postfix(**logs)

        if accelerator.is_main_process and epoch % train_args.validation_epochs == 0:
            logger.info("Running validation...")

            pipeline = GatedDiffusionPipeline.from_pretrained(
                model_args.model_path,
                unet=accelerator.unwrap_model(unet),
                mask_unet=accelerator.unwrap_model(mask_unet),
                text_encoder=accelerator.unwrap_model(text_encoder),
                inverse_scheduler=inversed_latents_pipeline,
                vae=accelerator.unwrap_model(vae),
                safety_checker=None,
                torch_dtype=weight_dtype,
            )
            pipeline = pipeline.to(accelerator.device)
            pipeline.set_progress_bar_config(disable=True)

            # run inference on single iamge
            val_images = {
                "source_image": [],
                "edited_image_without_mask": [],
                "edited_image_mask_all_timestep": [],
                "edited_image_mask_last_timestep": [],
                "masks_all_timestep": [],
                "masks_last_timestep": [],
            }
            with torch.autocast(
                str(accelerator.device).replace(":0", ""), enabled=accelerator.mixed_precision == "fp16"
            ):
                for validation_prompt, validation_image in zip(validation_prompts, validation_images):
                    val_images["source_image"].append(wandb.Image(validation_image, caption=validation_prompt))

                    if config.inference.method == "both" or config.inference.method == "all":
                        result = pipeline(
                            prompt=validation_prompt,
                            image=validation_image,
                            num_inference_steps=20,
                            image_guidance_scale=1.5,
                            guidance_scale=7,
                            generator=generator,
                            method="all",
                            hard_mask=config.inference.hard_mask,
                        )
                        edited_image = wandb.Image(result.images[0], caption=validation_prompt)
                        masks = wandb.Image(result.masks[0], caption=validation_prompt)
                        val_images["edited_image_mask_all_timestep"].append(edited_image)
                        val_images["masks_all_timestep"].append(masks)

                    if config.inference.method == "both" or config.inference.method == "last":
                        result = pipeline(
                            prompt=validation_prompt,
                            image=validation_image,
                            num_inference_steps=20,
                            image_guidance_scale=1.5,
                            guidance_scale=7,
                            generator=generator,
                            method="last",
                            hard_mask=config.inference.hard_mask,
                        )
                        edited_image = wandb.Image(result.images[0], caption=validation_prompt)
                        mask = wandb.Image(result.masks[0], caption=validation_prompt)
                        val_images["edited_image_mask_last_timestep"].append(edited_image)
                        val_images["masks_last_timestep"].append(mask)

                    result = pipeline(
                        prompt=validation_prompt,
                        image=validation_image,
                        num_inference_steps=20,
                        image_guidance_scale=1.5,
                        guidance_scale=7,
                        generator=generator,
                        method="none",
                        hard_mask=config.inference.hard_mask,
                    )
                    edited_image = wandb.Image(result.images[0], caption=validation_prompt)
                    val_images["edited_image_without_mask"].append(edited_image)

            for tracker in accelerator.trackers:
                if tracker.name == "wandb":
                    tracker.log(val_images)

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        pipeline = GatedDiffusionPipeline.from_pretrained(
            model_args.model_path,
            unet=accelerator.unwrap_model(unet),
            mask_unet=accelerator.unwrap_model(mask_unet),
            text_encoder=accelerator.unwrap_model(text_encoder),
            inverse_scheduler=inversed_latents_pipeline,
            vae=accelerator.unwrap_model(vae),
            safety_checker=None,
            torch_dtype=weight_dtype,
        )
        pipeline.save_pretrained(os.path.join(logging_dir, "models"))

    accelerator.end_training()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    config = OmegaConf.load(args.config)
    main(run_name=args.name, config=config)
