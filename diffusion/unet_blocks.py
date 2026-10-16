import torch
import torch.nn as nn

from diffusion.unet_parts import ResidualBlock, Downsample, CrossAttention, Upsample


class EncoderBlock(nn.Module):
    """
    Blocco encoder della diffusion U-Net.

    Il blocco:
        1. applica un ResidualBlock;
        2. conserva le feature come skip connection;
        3. riduce la risoluzione spaziale tramite Downsample.

    Input:
        x: [B, in_channels, H, W]
        time_embedding: [B, time_dim]

    Output:
        features: [B, out_channels, H/2, W/2]
        skip:     [B, out_channels, H, W]
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_dim: int,
    ):
        super().__init__()

        self.residual = ResidualBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            time_dim=time_dim,
        )

        self.downsample = Downsample(
            in_channels=out_channels,
            out_channels=out_channels,
        )

    def forward(
        self,
        x: torch.Tensor,
        time_embedding: torch.Tensor,
    ):
        """
        Forward pass del blocco encoder.
        """

        x = self.residual(
            x,
            time_embedding,
        )

        skip = x

        x = self.downsample(x)

        return x, skip

class ConditionedEncoderBlock(nn.Module):
    """
    Blocco encoder condizionato dal testo.

    Il blocco:
        1. applica un ResidualBlock;
        2. applica cross-attention con il contesto testuale;
        3. conserva le feature come skip connection;
        4. riduce la risoluzione tramite Downsample.

    Input:
        x:
            [B, in_channels, H, W]

        time_embedding:
            [B, time_dim]

        context:
            [B, N_tokens, context_dim]

    Output:
        features:
            [B, out_channels, H/2, W/2]

        skip:
            [B, out_channels, H, W]
    """



    def __init__(
                self,
                in_channels: int,
                out_channels: int,
                time_dim: int,
                context_dim=None,
                num_heads: int = 4,
        ):
            super().__init__()

            self.residual = ResidualBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                time_dim=time_dim,
            )

            if context_dim is not None:
                self.attention = CrossAttention(
                    query_dim=out_channels,
                    context_dim=context_dim,
                    num_heads=num_heads,
                )
            else:
                self.attention = None

            self.downsample = Downsample(
                in_channels=out_channels,
                out_channels=out_channels,
            )

    def forward(
            self,
            x: torch.Tensor,
            time_embedding: torch.Tensor,
            context=None,
    ):
        x = self.residual(
            x,
            time_embedding,
        )

        if self.attention is not None:
            if context is None:
                raise ValueError(
                    "context is required when cross-attention is enabled."
                )

            x = self.attention(
                x,
                context,
            )

        skip = x

        x = self.downsample(x)

        return x, skip

class BottleneckBlock(nn.Module):
    """
    Blocco bottleneck della diffusion U-Net.

    Il blocco:
        1. applica un ResidualBlock;
        2. applica cross-attention con il contesto testuale.

    Non modifica:
        - la risoluzione spaziale;
        - il numero di canali.

    Input:
        x:
            [B, channels, H, W]

        time_embedding:
            [B, time_dim]

        context:
            [B, N_tokens, context_dim]

    Output:
        [B, channels, H, W]
    """

    def __init__(
                self,
                channels: int,
                time_dim: int,
                context_dim=None,
                num_heads: int = 4,
        ):
            super().__init__()

            self.residual = ResidualBlock(
                in_channels=channels,
                out_channels=channels,
                time_dim=time_dim,
            )

            if context_dim is not None:
                self.attention = CrossAttention(
                    query_dim=channels,
                    context_dim=context_dim,
                    num_heads=num_heads,
                )
            else:
                self.attention = None

    def forward(
            self,
            x: torch.Tensor,
            time_embedding: torch.Tensor,
            context=None,
    ) -> torch.Tensor:

        x = self.residual(
            x,
            time_embedding,
        )

        if self.attention is not None:
            if context is None:
                raise ValueError(
                    "context is required when cross-attention is enabled."
                )

            x = self.attention(
                x,
                context,
            )

        return x

class DecoderUpsampleBlock(nn.Module):
    """
    Primo blocco del decoder della diffusion U-Net.

    Aumenta la risoluzione spaziale di un fattore 2
    e modifica il numero di canali.

    Input:
        x:
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

        self.upsample = Upsample(
            in_channels=in_channels,
            out_channels=out_channels,
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass del blocco di upsampling.
        """

        return self.upsample(x)

class SkipConnection(nn.Module):
    """
    Concatena le feature del decoder con una skip connection
    proveniente dall'encoder.

    Input:
        decoder_features:
            [B, decoder_channels, H, W]

        skip:
            [B, skip_channels, H, W]

    Output:
        [B, decoder_channels + skip_channels, H, W]
    """

    def forward(
        self,
        decoder_features: torch.Tensor,
        skip: torch.Tensor,
    ) -> torch.Tensor:
        """
        Concatena decoder features e skip lungo la dimensione
        dei canali.
        """

        if decoder_features.dim() != 4:
            raise ValueError(
                "decoder_features must have shape [B, C, H, W]."
            )

        if skip.dim() != 4:
            raise ValueError(
                "skip must have shape [B, C, H, W]."
            )

        if decoder_features.size(0) != skip.size(0):
            raise ValueError(
                "decoder_features and skip must have the same batch size."
            )

        if decoder_features.size(2) != skip.size(2):
            raise ValueError(
                "decoder_features and skip must have the same height."
            )

        if decoder_features.size(3) != skip.size(3):
            raise ValueError(
                "decoder_features and skip must have the same width."
            )

        return torch.cat(
            [decoder_features, skip],
            dim=1,
        )

class DecoderBlock(nn.Module):
    """
    Blocco decoder della diffusion U-Net.

    Il blocco:
        1. concatena le feature del decoder con la skip connection;
        2. applica un ResidualBlock condizionato dal timestep.

    Input:
        decoder_features:
            [B, decoder_channels, H, W]

        skip:
            [B, skip_channels, H, W]

        time_embedding:
            [B, time_dim]

    Output:
        [B, out_channels, H, W]
    """

    def __init__(
        self,
        decoder_channels: int,
        skip_channels: int,
        out_channels: int,
        time_dim: int,
    ):
        super().__init__()

        self.skip_connection = SkipConnection()

        self.residual = ResidualBlock(
            in_channels=(
                decoder_channels + skip_channels
            ),
            out_channels=out_channels,
            time_dim=time_dim,
        )

    def forward(
        self,
        decoder_features: torch.Tensor,
        skip: torch.Tensor,
        time_embedding: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass del blocco decoder.
        """

        x = self.skip_connection(
            decoder_features,
            skip,
        )

        x = self.residual(
            x,
            time_embedding,
        )

        return x

class InputProjection(nn.Module):
    """
    Proietta l'immagine RGB nello spazio dei canali interni della U-Net.

    Input:
        [B, 3, H, W]

    Output:
        [B, out_channels, H, W]
    """

    def __init__(self, in_channels=3, out_channels=64):
        super().__init__()

        if in_channels <= 0:
            raise ValueError("in_channels must be positive.")

        if out_channels <= 0:
            raise ValueError("out_channels must be positive.")

        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding=1,
        )

    def forward(self, x):
        if x.dim() != 4:
            raise ValueError(
                "x must have shape [B, C, H, W]."
            )

        if x.size(1) != self.conv.in_channels:
            raise ValueError(
                "Input channel dimension does not match "
                "in_channels."
            )

        return self.conv(x)

class OutputProjection(nn.Module):
    """
    Proietta le feature interne della U-Net nello spazio RGB.

    Input:
        [B, in_channels, H, W]

    Output:
        [B, 3, H, W]
    """

    def __init__(self, in_channels=64, out_channels=3):
        super().__init__()

        if in_channels <= 0:
            raise ValueError("in_channels must be positive.")

        if out_channels <= 0:
            raise ValueError("out_channels must be positive.")

        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding=1,
        )

    def forward(self, x):
        if x.dim() != 4:
            raise ValueError(
                "x must have shape [B, C, H, W]."
            )

        if x.size(1) != self.conv.in_channels:
            raise ValueError(
                "Input channel dimension does not match "
                "in_channels."
            )

        return self.conv(x)