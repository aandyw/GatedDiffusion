import torch
import torch.nn as nn

from diffusers import UNet2DConditionModel


class Model(UNet2DConditionModel):
    def __init__(self, mask, unet):
        super().__init__()
        self.mask = mask
        self.unet = unet
        self.conv_act = nn.Sigmoid()

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
        unet_mask = self.mask(
            concatenated_noisy_latents, timesteps, encoder_hidden_states
        ).sample
        act_mask = self.conv_act(unet_mask)

        out = torch.mul(act_mask, original_image_embeds) + torch.mul(
            1 - act_mask, generated_img
        )
        return out
