from pathlib import Path

import torch
from torchvision.utils import save_image

from diffusion.sampler import sample

from data.cartoon_dataset import CartoonSetDataset
from data.dataset_split import create_dataset_splits

from text.caption_generator import generate_caption
from text.tokenizer import CaptionTokenizer
from text.text_encoder import TextEncoder
from text.text_conditioning import TextConditioningProjection

from diffusion.time_step_embedding import TimestepEmbedding
from diffusion.unet import DiffusionUNet
from diffusion.diffuser import DiffusionScheduler

from prompt.prompt_to_caption import prompt_to_caption

# ============================================================
# Project paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

DATASET_ROOT = (
    r"C:\Users\utente\.cache\kagglehub\datasets"
    r"\paraglondhe\cartoonset100k\versions\2"
    r"\cartoonset100k\0"
)

CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "training"
    / "checkpoint_conditioned_ep50"
    / "checkpoint_epoch_50.pt"
)

UNCONDITIONAL_CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "training"
    / "unconditional_checkpoint.pt"
)

GENERATED_DIR = PROJECT_ROOT / "generated"


# ============================================================
# Model configuration
# ============================================================

MAX_LENGTH = 18

TEXT_HIDDEN_SIZE = 64
TEXT_NUM_LAYERS = 2
TEXT_NUM_HEADS = 4

CONDITIONING_INPUT_DIM = 64
CONDITIONING_OUTPUT_DIM = 128

TIME_EMBEDDING_DIM = 64
NUM_TIMESTEPS = 1000

NUM_HEADS = 4


# ============================================================
# Reproducibility
# ============================================================

def set_seed(seed):
    """Set the random seed used for generation."""
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def fit_tokenizer_on_train(dataset, train_indices, tokenizer):
    """Build the tokenizer vocabulary from training captions only."""
    captions = []

    for index in train_indices:
        sample = dataset.samples[index]

        metadata = dataset._load_metadata(
            sample["metadata"]
        )

        caption = generate_caption(metadata)
        captions.append(caption)

    tokenizer.fit(captions)

    return tokenizer

def load_components(device):
    """Create the model components and load the trained checkpoint."""

    # ---------------------------------------------------------
    # 1. Dataset and training split
    # ---------------------------------------------------------

    dataset = CartoonSetDataset(
        root_dir=DATASET_ROOT,
        expected_samples=10_000,
    )

    train_indices, _, _, _ = create_dataset_splits(
        dataset=dataset,
        ood_hair_color=4,
        ood_eye_color=1,
        seed=42,
    )

    # ---------------------------------------------------------
    # 2. Tokenizer
    # ---------------------------------------------------------

    tokenizer = CaptionTokenizer()

    fit_tokenizer_on_train(
        dataset,
        train_indices,
        tokenizer,
    )

    # ---------------------------------------------------------
    # 3. Text encoder
    # ---------------------------------------------------------

    text_encoder = TextEncoder(
        vocab_size=tokenizer.vocab_size,
        hidden_size=TEXT_HIDDEN_SIZE,
        num_layers=TEXT_NUM_LAYERS,
        num_heads=TEXT_NUM_HEADS,
        max_length=MAX_LENGTH,
        dropout=0.0,
    ).to(device)

    # ---------------------------------------------------------
    # 4. Text conditioning
    # ---------------------------------------------------------

    conditioning_projection = TextConditioningProjection(
        input_dim=CONDITIONING_INPUT_DIM,
        output_dim=CONDITIONING_OUTPUT_DIM,
    ).to(device)

    # ---------------------------------------------------------
    # 5. Timestep embedding
    # ---------------------------------------------------------

    timestep_embedding = TimestepEmbedding(
        num_timesteps=NUM_TIMESTEPS,
        embedding_dim=TIME_EMBEDDING_DIM,
    ).to(device)

    # ---------------------------------------------------------
    # 6. U-Net
    # ---------------------------------------------------------

    unet = DiffusionUNet(
        time_dim=TIME_EMBEDDING_DIM,
        context_dim=CONDITIONING_OUTPUT_DIM,
        num_heads=NUM_HEADS,
    ).to(device)

    unconditional_unet = DiffusionUNet(
        time_dim=TIME_EMBEDDING_DIM,
        context_dim=None,
        num_heads=NUM_HEADS,
    ).to(device)

    # ---------------------------------------------------------
    # 7. Diffusion scheduler
    # ---------------------------------------------------------

    scheduler = DiffusionScheduler(
        num_timesteps=NUM_TIMESTEPS,
        schedule_type="linear",
    ).to(device)

    # ---------------------------------------------------------
    # 8. Load checkpoint
    # ---------------------------------------------------------

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device,
    )

    text_encoder.load_state_dict(
        checkpoint["text_encoder"]
    )

    conditioning_projection.load_state_dict(
        checkpoint["text_conditioning"]
    )

    unet.load_state_dict(
        checkpoint["unet"]
    )

    unconditional_checkpoint = torch.load(
        UNCONDITIONAL_CHECKPOINT_PATH,
        map_location=device,
    )

    unconditional_unet.load_state_dict(
        unconditional_checkpoint["unet"]
    )

    # ---------------------------------------------------------
    # 9. Evaluation mode
    # ---------------------------------------------------------

    text_encoder.eval()
    conditioning_projection.eval()
    unet.eval()
    unconditional_unet.eval()

    return (
        tokenizer,
        text_encoder,
        conditioning_projection,
        timestep_embedding,
        unet,
        unconditional_unet,
        scheduler,
    )


@torch.no_grad()
def generate_image(
    prompt,
    seed,
    device,
    tokenizer,
    text_encoder,
    conditioning_projection,
    timestep_embedding,
    unet,
    unconditional_unet,
    scheduler,
    guidance_scale,
):
    """Generate one image from a natural-language prompt."""

    set_seed(seed)

    caption = prompt_to_caption(prompt)

    print("User prompt:", prompt)
    print("Training-style caption:", caption)

    token_ids = tokenizer.encode(
        caption,
        max_length=MAX_LENGTH,
        padding=True,
        truncation=False,
    )

    print("Token IDs:", token_ids)
    print("Decoded:", tokenizer.decode(token_ids))

    image = sample(
        captions=[caption],
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        text_conditioning=conditioning_projection,
        timestep_embedding=timestep_embedding,
        unet=unet,
        unconditional_unet=unconditional_unet,
        scheduler=scheduler,
        device=device,
        max_length=MAX_LENGTH,
        guidance_scale=guidance_scale,
    )

    # Keep the output in the valid image range.
    image = image.clamp(0.0, 1.0)

    return image

def get_next_output_path():
    GENERATED_DIR.mkdir(exist_ok=True)

    counter = 1

    while True:
        output_path = GENERATED_DIR / f"generation_{counter:03d}.png"

        if not output_path.exists():
            return output_path

        counter += 1


if __name__ == "__main__":
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Device: {device}")

    (
        tokenizer,
        text_encoder,
        conditioning_projection,
        timestep_embedding,
        unet,
        unconditional_unet,
        scheduler,
    ) = load_components(device)

    prompt = input("Prompt: ")
    seed = int(input("Seed: "))

    print("\nGenerating...")

    image = generate_image(
        prompt=prompt,
        seed=seed,
        device=device,
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        conditioning_projection=conditioning_projection,
        timestep_embedding=timestep_embedding,
        unet=unet,
        scheduler=scheduler,
        unconditional_unet=None,
        guidance_scale=1.0,
    )

    print("\nGeneration completed.")
    print(f"Shape: {image.shape}")
    print(f"Min: {image.min().item():.4f}")
    print(f"Max: {image.max().item():.4f}")
    print(f"Mean: {image.mean().item():.4f}")
    print(f"Finite: {torch.isfinite(image).all().item()}")

    output_path = get_next_output_path()
    save_image(image, output_path)

    print(f"Image saved to: {output_path}")