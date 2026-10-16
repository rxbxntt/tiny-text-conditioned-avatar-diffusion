import math

import torch
import torch.nn as nn


class TimestepEmbedding(nn.Module):
    """
    Sinusoidal embedding of discrete diffusion timesteps.

    Input:
        timesteps: Tensor of shape [B], dtype torch.long.

    Output:
        time_embedding: Tensor of shape [B, embedding_dim], dtype torch.float32.

    The embedding is deterministic and does not contain trainable parameters.
    It is intended to provide the U-Net with information about the current
    position in the diffusion chain.
    """

    def __init__(self, num_timesteps, embedding_dim, max_period=10000.0):
        super().__init__()

        if not isinstance(num_timesteps, int) or num_timesteps <= 0:
            raise ValueError("num_timesteps must be a positive integer.")

        if not isinstance(embedding_dim, int) or embedding_dim <= 0:
            raise ValueError("embedding_dim must be a positive integer.")

        if embedding_dim % 2 != 0:
            raise ValueError("embedding_dim must be even.")

        if max_period <= 0:
            raise ValueError("max_period must be positive.")

        self.num_timesteps = num_timesteps
        self.embedding_dim = embedding_dim
        self.max_period = float(max_period)

        half_dim = embedding_dim // 2

        # Frequencies used by the sinusoidal basis. They are fixed and are
        # stored as a buffer so they follow the module across devices.
        frequencies = torch.exp(
            -math.log(self.max_period)
            * torch.arange(half_dim, dtype=torch.float32)
            / half_dim
        )

        self.register_buffer("frequencies", frequencies)

    def forward(self, timesteps):
        """
        Encode a batch of diffusion timesteps.

        Args:
            timesteps: Tensor with shape [B], dtype torch.long, containing
                values in {0, ..., num_timesteps - 1}.

        Returns:
            Tensor with shape [B, embedding_dim].
        """
        if not isinstance(timesteps, torch.Tensor):
            raise TypeError("timesteps must be a torch.Tensor.")

        if timesteps.dim() != 1:
            raise ValueError("timesteps must have shape [B].")

        if timesteps.dtype != torch.long:
            raise ValueError("timesteps must have dtype torch.long.")

        if timesteps.numel() == 0:
            raise ValueError("timesteps must contain at least one element.")

        if torch.any(timesteps < 0) or torch.any(
            timesteps >= self.num_timesteps
        ):
            raise ValueError("timesteps contain an invalid value.")

        # [B] -> [B, 1]. The multiplication broadcasts the timestep values
        # over all sinusoidal frequencies.
        angles = timesteps.float().unsqueeze(1) * self.frequencies.unsqueeze(0)

        # [B, D/2] + [B, D/2] -> [B, D]
        embedding = torch.cat(
            [torch.sin(angles), torch.cos(angles)],
            dim=1,
        )

        return embedding
