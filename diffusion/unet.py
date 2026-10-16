import torch
import torch.nn as nn

from diffusion.unet_blocks import (
    InputProjection,
    OutputProjection,
    EncoderBlock,
    ConditionedEncoderBlock,
    BottleneckBlock,
    DecoderUpsampleBlock,
    DecoderBlock,
)


class DiffusionUNet(nn.Module):
    """
    U-Net per la predizione del processo di diffusion.

    Input:
        x:
            [B, 3, 32, 32]

        time_embedding:
            [B, time_dim]

        context:
            [B, N_tokens, context_dim]

    Output:
        [B, 3, 32, 32]
    """

    def __init__(
        self,
        time_dim,
        context_dim=None,
        num_heads=4,
    ):
        super().__init__()

        self.input_projection = InputProjection(
            in_channels=3,
            out_channels=64,
        )

        self.encoder1 = EncoderBlock(
            in_channels=64,
            out_channels=128,
            time_dim=time_dim,
        )

        self.encoder2 = ConditionedEncoderBlock(
            in_channels=128,
            out_channels=256,
            time_dim=time_dim,
            context_dim=context_dim,
            num_heads=num_heads,
        )

        self.bottleneck = BottleneckBlock(
            channels=256,
            time_dim=time_dim,
            context_dim=context_dim,
            num_heads=num_heads,
        )

        self.decoder_upsample1 = DecoderUpsampleBlock(
            in_channels=256,
            out_channels=128,
        )

        self.decoder1 = DecoderBlock(
            decoder_channels=128,
            skip_channels=256,
            out_channels=128,
            time_dim=time_dim,
        )

        self.decoder_upsample2 = DecoderUpsampleBlock(
            in_channels=128,
            out_channels=64,
        )

        self.decoder2 = DecoderBlock(
            decoder_channels=64,
            skip_channels=128,
            out_channels=64,
            time_dim=time_dim,
        )

        self.output_projection = OutputProjection(
            in_channels=64,
            out_channels=3,
        )

    def forward(
        self,
        x,
        time_embedding,
        context=None,
    ):
        x = self.input_projection(x)

        x, skip32 = self.encoder1(
            x,
            time_embedding,
        )

        x, skip16 = self.encoder2(
            x,
            time_embedding,
            context,
        )

        x = self.bottleneck(
            x,
            time_embedding,
            context,
        )

        x = self.decoder_upsample1(x)

        x = self.decoder1(
            x,
            skip16,
            time_embedding,
        )

        x = self.decoder_upsample2(x)

        x = self.decoder2(
            x,
            skip32,
            time_embedding,
        )

        x = self.output_projection(x)

        return x