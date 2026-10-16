import torch

from pathlib import Path
from text.tokenizer import CaptionTokenizer
from text.text_encoder import TextEncoder
from text.text_conditioning import TextConditioningProjection
from diffusion.time_step_embedding import TimestepEmbedding
from diffusion.unet import DiffusionUNet


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

VOCAB_SIZE = 219
HIDDEN_SIZE = 64
NUM_LAYERS = 2
NUM_HEADS = 4
MAX_LENGTH = 18

CONDITIONING_INPUT_DIM = 64
CONDITIONING_OUTPUT_DIM = 128

TIME_EMBEDDING_DIM = 64
NUM_TIMESTEPS = 1000


PROJECT_ROOT = Path(__file__).resolve().parent.parent

CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "training"
    / "checkpoint_conditioned_ep50"
    / "checkpoint_epoch_50.pt"
)


# ---------------------------------------------------------
# Utility
# ---------------------------------------------------------

def count_parameters(model):
    """Return total and trainable parameter counts."""
    total = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    trainable = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    return total, trainable


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():

    device = torch.device("cpu")

    print("===== PARAMETER COUNT =====")
    print(f"Device: {device}")
    print()

    # -----------------------------------------------------
    # 1. Text Encoder
    # -----------------------------------------------------

    text_encoder = TextEncoder(
        vocab_size=VOCAB_SIZE,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        num_heads=NUM_HEADS,
        max_length=MAX_LENGTH,
        dropout=0.0,
    ).to(device)

    # -----------------------------------------------------
    # 2. Text Conditioning Projection
    # -----------------------------------------------------

    text_conditioning = TextConditioningProjection(
        input_dim=CONDITIONING_INPUT_DIM,
        output_dim=CONDITIONING_OUTPUT_DIM,
    ).to(device)

    # -----------------------------------------------------
    # 3. Timestep Embedding
    # -----------------------------------------------------

    timestep_embedding = TimestepEmbedding(
        num_timesteps=NUM_TIMESTEPS,
        embedding_dim=TIME_EMBEDDING_DIM,
    ).to(device)

    # -----------------------------------------------------
    # 4. U-Net
    # -----------------------------------------------------

    unet = DiffusionUNet(
        time_dim=TIME_EMBEDDING_DIM,
        context_dim=CONDITIONING_OUTPUT_DIM,
        num_heads=NUM_HEADS,
    ).to(device)

    # -----------------------------------------------------
    # 5. Load checkpoint
    # -----------------------------------------------------

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device,
    )

    text_encoder.load_state_dict(
        checkpoint["text_encoder"]
    )

    text_conditioning.load_state_dict(
        checkpoint["text_conditioning"]
    )

    unet.load_state_dict(
        checkpoint["unet"]
    )

    # -----------------------------------------------------
    # 6. Count parameters
    # -----------------------------------------------------

    text_total, text_trainable = count_parameters(
        text_encoder
    )

    conditioning_total, conditioning_trainable = (
        count_parameters(text_conditioning)
    )

    timestep_total, timestep_trainable = (
        count_parameters(timestep_embedding)
    )

    unet_total, unet_trainable = count_parameters(
        unet
    )

    total_parameters = (
        text_total
        + conditioning_total
        + timestep_total
        + unet_total
    )

    total_trainable = (
        text_trainable
        + conditioning_trainable
        + timestep_trainable
        + unet_trainable
    )

    # -----------------------------------------------------
    # 7. Print results
    # -----------------------------------------------------

    print(f"Text Encoder:          {text_total:,}")
    print(f"Conditioning Projection:{conditioning_total:,}")
    print(f"Timestep Embedding:     {timestep_total:,}")
    print(f"U-Net:                 {unet_total:,}")
    print("--------------------------------")
    print(f"Total parameters:      {total_parameters:,}")
    print(f"Trainable parameters:  {total_trainable:,}")

    # -----------------------------------------------------
    # 8. Sanity checks
    # -----------------------------------------------------

    assert unet_total == 5_108_867, (
        "Unexpected U-Net parameter count."
    )

    assert timestep_total == 0, (
        "Timestep embedding should have no trainable parameters."
    )

    assert total_parameters == 5_231_171, (
        "Unexpected total parameter count."
    )

    assert total_trainable == 5_231_171, (
        "Unexpected trainable parameter count."
    )

    print()
    print("Sanity checks: PASSED")


if __name__ == "__main__":
    main()