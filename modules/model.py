import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, mask, unet):
        super().__init__()
        self.mask = mask
        self.unet = unet

    def forward(
        self,
        original_image_embeds,
        concatenated_noisy_latents,
        timesteps,
        encoder_hidden_states,
    ):
        generated_img = self.unet(
            concatenated_noisy_latents, timesteps, encoder_hidden_states
        ).sample
        gated_unet_mask = self.mask(
            concatenated_noisy_latents, timesteps, encoder_hidden_states
        ).sample

        out = torch.mul(gated_unet_mask, original_image_embeds) + torch.mul(
            1 - gated_unet_mask, generated_img
        )

        return out
