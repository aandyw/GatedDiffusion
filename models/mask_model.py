from dataclasses import dataclass
from typing import Union, Tuple

import torch
import torch.nn as nn
from diffusers.models import UNet2DConditionModel
from diffusers.utils import BaseOutput


@dataclass
class MaskModelOutput(BaseOutput):
    """
    The output of [`MaskModelOutput`].

    Args:
        sample (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            The hidden states output conditioned on `encoder_hidden_states` input. Output of last layer of model.
    """

    mask: torch.FloatTensor = None


class MaskModel(nn.Module):
    def __init__(self, model_path, gradient_checkpointing):
        super(MaskModel, self).__init__()
        self.unet = UNet2DConditionModel.from_pretrained(model_path, subfolder="unet")
        self.act_fn = nn.ReLU()

        if gradient_checkpointing:
            self.unet.enable_gradient_checkpointing()

    def forward(
        self,
        noisy_latents: torch.FloatTensor,
        timesteps: Union[torch.Tensor, float, int],
        encoder_hidden_states: torch.Tensor,
    ) -> Union[MaskModelOutput, Tuple]:
        unet_out = self.unet(noisy_latents, timesteps, encoder_hidden_states).sample
        out = self.act_fn(unet_out.logits)
        return MaskModelOutput(mask=out)
