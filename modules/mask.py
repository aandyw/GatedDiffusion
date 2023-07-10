""" Learnable Mask with UNet """
from typing import Union, Tuple

import torch
import torch.nn as nn

from diffusers.models import UNet2DConditionModel


class Mask(nn.Module):
    def __init__(self, act_fn: str = "sigmoid"):
        super().__init__()

        self.unet = UNet2DConditionModel()
        self.conv_act = self.get_activation(act_fn)

    def get_activation(self, act_fn: str) -> nn.Module:
        if act_fn == "sigmoid":
            return nn.Sigmoid()
        elif act_fn == "relu":
            return nn.ReLU()
        elif act_fn == "leakyrelu":
            return nn.LeakyReLU()
        elif act_fn in ["swish", "silu"]:
            return nn.SiLU()
        elif act_fn == "mish":
            return nn.Mish()
        elif act_fn == "gelu":
            return nn.GELU()
        else:
            raise ValueError(f"Unsupported activation function: {act_fn}")

    def forward(
        self,
        sample: torch.FloatTensor,
        timestep: Union[torch.Tensor, float, int],
        encoder_hidden_states: torch.Tensor,
    ) -> Union[torch.FloatTensor, Tuple]:
        sample = self.unet(sample, timestep, encoder_hidden_states)
        out = self.conv_act(sample)
        return out
