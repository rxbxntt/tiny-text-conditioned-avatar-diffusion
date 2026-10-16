import csv
import time
from pathlib import Path

import torch
from torchvision.utils import save_image

from data.cartoon_dataset import CartoonSetDataset
from data.dataset_split import create_dataset_splits

from text.caption_generator import generate_caption
from text.tokenizer import CaptionTokenizer
from text.text_encoder import TextEncoder
from text.text_conditioning import TextConditioningProjection

from diffusion.time_step_embedding import TimestepEmbedding
from diffusion.unet import DiffusionUNet
from diffusion.diffuser import DiffusionScheduler


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

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

OUTPUT_DIR = PROJECT_ROOT / "evaluation" / "ood"

BATCH_SIZE = 16
SEED = 42

IMAGE_SIZE = 32
MAX_LENGTH = 18

NUM_TIMESTEPS = 1000
SCHEDULE_TYPE = "linear"

OOD_HAIR_COLOR = 4
OOD_EYE_COLOR = 1


# ============================================================
# TOKENIZER
# ============================================================

def fit_tokenizer_on_indices(dataset, indices, tokenizer):
    """
    Fit the tokenizer only on captions generated from the
    training split.
    """

    captions = []

    for index in indices:
        sample = dataset.samples[index]

        metadata = dataset._load_metadata(
            sample["metadata"]
        )

        caption = generate_caption(metadata)
        captions.append(caption)

    tokenizer.fit(captions)

    return tokenizer


# ============================================================
# SAMPLING
# ============================================================

@torch.no_grad()
def sample_ood(
    captions,
    tokenizer,
    text_encoder,
    text_conditioning,
    timestep_embedding,
    unet,
    scheduler,
    device,
):
    """
    Generate images from OOD captions using the same
    reverse DDPM sampling procedure used by the project sampler.

    Input:
        captions: list[str]

    Output:
        Tensor [B, 3, 32, 32]
    """

    batch_size = len(captions)

    # --------------------------------------------------------
    # 1. Tokenize captions
    # --------------------------------------------------------

    token_ids = []
    attention_masks = []

    for caption in captions:

        ids = tokenizer.encode(
            caption,
            max_length=MAX_LENGTH,
            padding=True,
            truncation=True,
        )

        mask = tokenizer.attention_mask(ids)

        token_ids.append(ids)
        attention_masks.append(mask)

    input_ids = torch.tensor(
        token_ids,
        dtype=torch.long,
        device=device,
    )

    attention_mask = torch.tensor(
        attention_masks,
        dtype=torch.long,
        device=device,
    )

    # --------------------------------------------------------
    # 2. Encode text
    # --------------------------------------------------------

    text_features = text_encoder(
        input_ids=input_ids,
        attention_mask=attention_mask,
    )

    context = text_conditioning(text_features)

    # --------------------------------------------------------
    # 3. Start from Gaussian noise
    # --------------------------------------------------------

    x_t = torch.randn(
        batch_size,
        3,
        IMAGE_SIZE,
        IMAGE_SIZE,
        device=device,
    )

    # --------------------------------------------------------
    # 4. Reverse diffusion
    # --------------------------------------------------------

    for timestep in reversed(
        range(scheduler.num_timesteps)
    ):

        timesteps = torch.full(
            (batch_size,),
            timestep,
            dtype=torch.long,
            device=device,
        )

        time_embedding = timestep_embedding(
            timesteps
        )

        predicted_noise = unet(
            x_t,
            time_embedding,
            context,
        )

        x_t = scheduler.reverse_step(
            x_t=x_t,
            predicted_noise=predicted_noise,
            timesteps=timesteps,
        )

    return x_t


# ============================================================
# MAIN
# ============================================================

def main():

    start_time = time.time()

    print("===== OOD EVALUATION =====")

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Device: {device}")

    # --------------------------------------------------------
    # Reproducibility
    # --------------------------------------------------------

    torch.manual_seed(SEED)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    dataset = CartoonSetDataset(
        root_dir=DATASET_ROOT,
        expected_samples=10_000,
    )

    print(f"Dataset samples: {len(dataset.samples)}")

    # --------------------------------------------------------
    # Dataset split
    # --------------------------------------------------------

    (
        train_indices,
        val_indices,
        test_indices,
        ood_indices,
    ) = create_dataset_splits(
        dataset=dataset,
        ood_hair_color=OOD_HAIR_COLOR,
        ood_eye_color=OOD_EYE_COLOR,
        seed=42,
    )

    print()
    print("===== DATASET SPLIT =====")
    print(f"Train: {len(train_indices)}")
    print(f"Validation: {len(val_indices)}")
    print(f"Test: {len(test_indices)}")
    print(f"OOD: {len(ood_indices)}")

    if len(train_indices) != 8227:
        raise RuntimeError("Unexpected train split size.")

    if len(val_indices) != 1028:
        raise RuntimeError("Unexpected validation split size.")

    if len(test_indices) != 515:
        raise RuntimeError("Unexpected test split size.")

    if len(ood_indices) != 230:
        raise RuntimeError("Unexpected OOD split size.")

    # --------------------------------------------------------
    # Tokenizer
    # --------------------------------------------------------

    tokenizer = CaptionTokenizer()

    fit_tokenizer_on_indices(
        dataset=dataset,
        indices=train_indices,
        tokenizer=tokenizer,
    )

    print()
    print("===== TOKENIZER =====")
    print(f"Vocabulary size: {tokenizer.vocab_size}")

    # --------------------------------------------------------
    # Models
    # --------------------------------------------------------

    text_encoder = TextEncoder(
        vocab_size=tokenizer.vocab_size,
        hidden_size=64,
        num_layers=2,
        num_heads=4,
        max_length=MAX_LENGTH,
        dropout=0.0,
    ).to(device)

    text_conditioning = TextConditioningProjection(
        input_dim=64,
        output_dim=128,
    ).to(device)

    timestep_embedding = TimestepEmbedding(
        num_timesteps=NUM_TIMESTEPS,
        embedding_dim=64,
    ).to(device)

    unet = DiffusionUNet(
        time_dim=64,
        context_dim=128,
        num_heads=4,
    ).to(device)

    # --------------------------------------------------------
    # Diffusion scheduler
    # --------------------------------------------------------

    scheduler = DiffusionScheduler(
        num_timesteps=NUM_TIMESTEPS,
        schedule_type=SCHEDULE_TYPE,
    )

    # --------------------------------------------------------
    # Load checkpoint
    # --------------------------------------------------------

    print()
    print("===== CHECKPOINT =====")
    print(f"Loading: {CHECKPOINT_PATH}")

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

    print(
        f"Epoch: {checkpoint.get('epoch', 'unknown')}"
    )

    if "train_loss" in checkpoint:
        print(
            f"Train loss: {checkpoint['train_loss']:.4f}"
        )

    if "val_loss" in checkpoint:
        print(
            f"Val loss: {checkpoint['val_loss']:.4f}"
        )

    # --------------------------------------------------------
    # Evaluation mode
    # --------------------------------------------------------

    text_encoder.eval()
    text_conditioning.eval()
    unet.eval()

    # --------------------------------------------------------
    # Output directory
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # OOD manifest
    # --------------------------------------------------------

    manifest_path = OUTPUT_DIR / "ood_manifest.csv"

    manifest_file = open(
        manifest_path,
        "w",
        newline="",
        encoding="utf-8",
    )

    writer = csv.writer(manifest_file)

    writer.writerow(
        [
            "output_index",
            "dataset_index",
            "hair_color",
            "eye_color",
            "caption",
            "path",
        ]
    )

    # --------------------------------------------------------
    # Generate OOD samples
    # --------------------------------------------------------

    generated = 0
    total_ood = len(ood_indices)

    print()
    print("===== OOD SAMPLING =====")
    print(f"Total OOD samples: {total_ood}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Reverse diffusion steps: {NUM_TIMESTEPS}")

    generation_start = time.time()

    for start in range(
        0,
        total_ood,
        BATCH_SIZE,
    ):

        batch_indices = ood_indices[
            start:start + BATCH_SIZE
        ]

        captions = []

        # ----------------------------------------------------
        # Build OOD captions
        # ----------------------------------------------------

        for dataset_index in batch_indices:

            sample = dataset.samples[
                dataset_index
            ]

            metadata = dataset._load_metadata(
                sample["metadata"]
            )

            # Sanity check:
            if metadata["hair_color"] != OOD_HAIR_COLOR:
                raise RuntimeError(
                    "OOD sample has unexpected hair_color."
                )

            if metadata["eye_color"] != OOD_EYE_COLOR:
                raise RuntimeError(
                    "OOD sample has unexpected eye_color."
                )

            caption = generate_caption(metadata)

            captions.append(caption)

        # ----------------------------------------------------
        # Sampling
        # ----------------------------------------------------

        images = sample_ood(
            captions=captions,
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            text_conditioning=text_conditioning,
            timestep_embedding=timestep_embedding,
            unet=unet,
            scheduler=scheduler,
            device=device,
        )

        # ----------------------------------------------------
        # Sanity checks
        # ----------------------------------------------------

        expected_batch_size = len(batch_indices)

        if images.shape != (
            expected_batch_size,
            3,
            IMAGE_SIZE,
            IMAGE_SIZE,
        ):
            raise RuntimeError(
                f"Unexpected output shape: {images.shape}"
            )

        if not torch.isfinite(images).all():
            raise RuntimeError(
                "Generated images contain NaN or Inf."
            )

        # ----------------------------------------------------
        # Save images
        # ----------------------------------------------------

        for local_index, dataset_index in enumerate(
            batch_indices
        ):

            image = images[local_index]

            # Keep the same output convention as the sampler.
            image = image.clamp(0.0, 1.0)

            output_index = generated

            image_path = (
                OUTPUT_DIR
                / f"ood_{output_index:04d}.png"
            )

            save_image(
                image,
                image_path,
            )

            # Recover metadata for manifest.
            sample = dataset.samples[
                dataset_index
            ]

            metadata = dataset._load_metadata(
                sample["metadata"]
            )

            writer.writerow(
                [
                    output_index,
                    dataset_index,
                    metadata["hair_color"],
                    metadata["eye_color"],
                    captions[local_index],
                    str(image_path),
                ]
            )

            generated += 1

        # ----------------------------------------------------
        # Progress
        # ----------------------------------------------------

        elapsed = time.time() - generation_start

        average_time = elapsed / generated

        remaining = total_ood - generated

        estimated_remaining = (
            remaining * average_time
        )

        print(
            f"Generated "
            f"{generated}/{total_ood} "
            f"| elapsed: {elapsed / 60:.1f} min "
            f"| estimated remaining: "
            f"{estimated_remaining / 60:.1f} min",
            flush=True,
        )

    manifest_file.close()

    # --------------------------------------------------------
    # Final checks
    # --------------------------------------------------------

    image_files = list(
        OUTPUT_DIR.glob("ood_*.png")
    )

    elapsed_total = time.time() - start_time

    print()
    print("===== FINAL CHECK =====")
    print(f"Generated samples: {generated}")
    print(f"PNG files: {len(image_files)}")
    print(f"Manifest: {manifest_path}")
    print(f"Total elapsed: {elapsed_total / 60:.2f} min")

    if generated != 230:
        raise RuntimeError(
            "The number of generated OOD samples is not 230."
        )

    if len(image_files) != 230:
        raise RuntimeError(
            "The number of generated PNG files is not 230."
        )

    print()
    print("===== OOD EVALUATION COMPLETED =====")


if __name__ == "__main__":
    main()