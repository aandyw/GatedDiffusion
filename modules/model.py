import torch.nn as nn


class Model(nn.Module):
    def __init__(self, pipeline, mask):
        super().__init__()
        self.pipeline = pipeline
        self.mask = mask

    def forward(self, concatenated_noisy_latents, timesteps, encoder_hidden_states):
        out = self.pipeline(
            latents = concatenated_noisy_latents,
            prompt_embeds = encoder_hidden_states
        )
        return
