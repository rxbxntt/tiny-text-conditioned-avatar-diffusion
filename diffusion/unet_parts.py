import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    """
    Residual block della diffusion U-Net.

    Il blocco riceve:
        x:
            feature map [B, C_in, H, W]

        time_embedding:
            embedding del timestep [B, time_dim]

    e restituisce:
        feature map [B, C_out, H, W]
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_dim: int,
        num_groups: int = 32,
    ):
        super().__init__()

        if in_channels <= 0 or out_channels <= 0:
            raise ValueError(
                "in_channels and out_channels must be positive."
            )

        if time_dim <= 0:
            raise ValueError(
                "time_dim must be positive."
            )

        if in_channels % num_groups != 0:
            raise ValueError(
                "in_channels must be divisible by num_groups."
            )

        if out_channels % num_groups != 0:
            raise ValueError(
                "out_channels must be divisible by num_groups."
            )

        # Primo blocco convoluzionale
        self.norm1 = nn.GroupNorm(
            num_groups,
            in_channels
        )

        self.activation = nn.SiLU()

        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1
        )

        # Trasforma l'embedding del timestep
        # nel numero di canali necessario
        self.time_projection = nn.Linear(
            time_dim,
            out_channels
        )

        # Secondo blocco convoluzionale
        self.norm2 = nn.GroupNorm(
            num_groups,
            out_channels
        )

        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1
        )

        # Connessione residuale
        if in_channels == out_channels:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1
            )

    def forward(
        self,
        x: torch.Tensor,
        time_embedding: torch.Tensor
    ) -> torch.Tensor:

        if x.dim() != 4:
            raise ValueError(
                "x must have shape [B, C, H, W]."
            )

        if time_embedding.dim() != 2:
            raise ValueError(
                "time_embedding must have shape [B, time_dim]."
            )

        if x.size(0) != time_embedding.size(0):
            raise ValueError(
                "x and time_embedding must have the same batch size."
            )

        # Percorso residuale
        residual = self.shortcut(x)

        # Primo blocco
        h = self.norm1(x)
        h = self.activation(h)
        h = self.conv1(h)

        # Conditioning sul timestep
        time_bias = self.time_projection(
            time_embedding
        )

        # [B, C] -> [B, C, 1, 1]
        time_bias = time_bias.unsqueeze(-1).unsqueeze(-1)

        h = h + time_bias

        # Secondo blocco
        h = self.norm2(h)
        h = self.activation(h)
        h = self.conv2(h)

        # Residual connection
        return h + residual

class Downsample(nn.Module):
    """
    Riduce la risoluzione spaziale di una feature map di un fattore 2.

    Input:
        [B, in_channels, H, W]

    Output:
        [B, out_channels, H/2, W/2]
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
    ):
        super().__init__()

        if in_channels <= 0 or out_channels <= 0:
            raise ValueError(
                "in_channels and out_channels must be positive."
            )

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=2,
            padding=1,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        if x.dim() != 4:
            raise ValueError(
                "x must have shape [B, C, H, W]."
            )

        return self.conv(x)

class Upsample(nn.Module):
    """
    Aumenta la risoluzione spaziale di una feature map
    di un fattore 2.

    Input:
        [B, in_channels, H, W]

    Output:
        [B, out_channels, 2H, 2W]
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
    ):
        super().__init__()

        if in_channels <= 0 or out_channels <= 0:
            raise ValueError(
                "in_channels and out_channels must be positive."
            )

        self.conv = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=4,
            stride=2,
            padding=1,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        if x.dim() != 4:
            raise ValueError(
                "x must have shape [B, C, H, W]."
            )

        return self.conv(x)

class CrossAttention(nn.Module):
    """
    Cross-attention tra le feature della U-Net e le feature testuali.

    Input:
        x:
            feature map della U-Net
            [B, query_dim, H, W]

        context:
            embedding testuali
            [B, N_tokens, context_dim]

    Output:
        feature map condizionata
        [B, query_dim, H, W]
    """

    def __init__(
        self,
        query_dim: int,
        context_dim: int,
        num_heads: int = 4,
    ):
        super().__init__()

        if query_dim <= 0:
            raise ValueError(
                "query_dim must be positive."
            )

        if context_dim <= 0:
            raise ValueError(
                "context_dim must be positive."
            )

        if num_heads <= 0:
            raise ValueError(
                "num_heads must be positive."
            )

        if query_dim % num_heads != 0:
            raise ValueError(
                "query_dim must be divisible by num_heads."
            )

        self.query_dim = query_dim
        self.context_dim = context_dim
        self.num_heads = num_heads

        self.attention = nn.MultiheadAttention(
            embed_dim=query_dim,
            num_heads=num_heads,
            kdim=context_dim,
            vdim=context_dim,
            batch_first=True,
        )

        self.output_projection = nn.Linear(
            query_dim,
            query_dim,
        )

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
    ) -> torch.Tensor:

        if x.dim() != 4:
            raise ValueError(
                "x must have shape [B, C, H, W]."
            )

        if context.dim() != 3:
            raise ValueError(
                "context must have shape [B, N_tokens, context_dim]."
            )

        batch_size, channels, height, width = x.shape

        if channels != self.query_dim:
            raise ValueError(
                "x channel dimension must match query_dim."
            )

        if context.size(0) != batch_size:
            raise ValueError(
                "x and context must have the same batch size."
            )

        if context.size(2) != self.context_dim:
            raise ValueError(
                "context dimension must match context_dim."
            )

        # [B, C, H, W]
        #       ↓
        # [B, H*W, C]
        queries = x.flatten(2).transpose(1, 2)

        # Cross-attention:
        # queries  ← U-Net features
        # keys     ← text embeddings
        # values   ← text embeddings
        attended, _ = self.attention(
            query=queries,
            key=context,
            value=context,
        )

        attended = self.output_projection(attended)

        # Residual connection
        attended = attended + queries

        # [B, H*W, C]
        #       ↓
        # [B, C, H, W]
        output = attended.transpose(1, 2).reshape(
            batch_size,
            channels,
            height,
            width,
        )

        return output