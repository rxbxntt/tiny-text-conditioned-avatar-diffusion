import torch
import torch.nn as nn


class TextConditioningProjection(nn.Module):
    """
    Proietta le rappresentazioni prodotte dal Text Encoder
    nello spazio dimensionale utilizzato dalla Cross-Attention
    della U-Net.

    Input:
        text_features: [B, L, input_dim]

    Output:
        context: [B, L, output_dim]
    """

    def __init__(self, input_dim=64, output_dim=128):
        super().__init__()

        if not isinstance(input_dim, int) or input_dim <= 0:
            raise ValueError("input_dim must be a positive integer.")

        if not isinstance(output_dim, int) or output_dim <= 0:
            raise ValueError("output_dim must be a positive integer.")

        self.input_dim = input_dim
        self.output_dim = output_dim

        self.projection = nn.Linear(
            input_dim,
            output_dim
        )

    def forward(self, text_features):
        if not isinstance(text_features, torch.Tensor):
            raise TypeError("text_features must be a torch.Tensor.")

        if text_features.dim() != 3:
            raise ValueError(
                "text_features must have shape [B, L, D]."
            )

        if text_features.size(-1) != self.input_dim:
            raise ValueError(
                f"Expected last dimension {self.input_dim}, "
                f"got {text_features.size(-1)}."
            )

        return self.projection(text_features)