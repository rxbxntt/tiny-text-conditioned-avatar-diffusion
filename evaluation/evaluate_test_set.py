import csv
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


BATCH_SIZE = 2
SEED = 42

OOD_HAIR_COLOR = 4
OOD_EYE_COLOR = 1

DATASET_ROOT = (
    r"C:\Users\utente\.cache\kagglehub\datasets"
    r"\paraglondhe\cartoonset100k\versions\2"
    r"\cartoonset100k\0"
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "training"
    / "checkpoint_conditioned_ep50"
    / "checkpoint_epoch_50.pt"
)

OUTPUT_DIR = Path("evaluation/ordinary_test")
MANIFEST_PATH = OUTPUT_DIR / "test_manifest.csv"


def fit_tokenizer_on_indices(dataset, indices, tokenizer):
    captions = []

    for index in indices:
        sample = dataset.samples[index]
        metadata = dataset._load_metadata(sample["metadata"])
        captions.append(generate_caption(metadata))

    tokenizer.fit(captions)
    return tokenizer


@torch.no_grad()
def sample_conditional_batch(
    captions,
    tokenizer,
    text_encoder,
    conditioning_projection,
    timestep_embedding,
    unet,
    scheduler,
    device,
    generator,
):
    input_ids = []
    attention_masks = []

    for caption in captions:
        token_ids = tokenizer.encode(
            caption,
            max_length=18,
            padding=True,
            truncation=False,
        )

        input_ids.append(token_ids)
        attention_masks.append(
            tokenizer.attention_mask(token_ids)
        )

    input_ids = torch.tensor(
        input_ids,
        dtype=torch.long,
        device=device,
    )

    attention_mask = torch.tensor(
        attention_masks,
        dtype=torch.bool,
        device=device,
    )

    text_features = text_encoder(
        input_ids=input_ids,
        attention_mask=attention_mask,
    )

    context = conditioning_projection(text_features)

    batch_size = len(captions)

    x = torch.randn(
        batch_size,
        3,
        32,
        32,
        device=device,
        generator=generator,
    )

    for timestep in reversed(range(1000)):

        timesteps = torch.full(
            (batch_size,),
            timestep,
            dtype=torch.long,
            device=device,
        )

        time_embedding = timestep_embedding(timesteps)

        predicted_noise = unet(
            x=x,
            time_embedding=time_embedding,
            context=context,
        )

        x = scheduler.reverse_step(
            x_t=x,
            predicted_noise=predicted_noise,
            timesteps=timesteps,
        )

    return x.clamp(0.0, 1.0)


def main():

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Device: {device}")

    # --------------------------------------------------
    # Dataset
    # --------------------------------------------------

    dataset = CartoonSetDataset(
        root_dir=DATASET_ROOT,
        expected_samples=10_000,
    )

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

    print("\n===== DATASET SPLIT =====")
    print(f"Train: {len(train_indices)}")
    print(f"Validation: {len(val_indices)}")
    print(f"Test: {len(test_indices)}")
    print(f"OOD: {len(ood_indices)}")

    if len(test_indices) != 515:
        raise RuntimeError(
            f"Expected 515 test samples, got {len(test_indices)}."
        )

    # --------------------------------------------------
    # Tokenizer
    # --------------------------------------------------

    tokenizer = CaptionTokenizer()

    fit_tokenizer_on_indices(
        dataset,
        train_indices,
        tokenizer,
    )

    print("\n===== TOKENIZER =====")
    print(f"Vocabulary size: {tokenizer.vocab_size}")

    # --------------------------------------------------
    # Model
    # --------------------------------------------------

    text_encoder = TextEncoder(
        vocab_size=tokenizer.vocab_size,
        hidden_size=64,
        num_layers=2,
        num_heads=4,
        max_length=18,
        dropout=0.0,
    ).to(device)

    conditioning_projection = TextConditioningProjection(
        input_dim=64,
        output_dim=128,
    ).to(device)

    timestep_embedding = TimestepEmbedding(
        num_timesteps=1000,
        embedding_dim=64,
    ).to(device)

    unet = DiffusionUNet(
        time_dim=64,
        context_dim=128,
        num_heads=4,
    ).to(device)

    scheduler = DiffusionScheduler(
        num_timesteps=1000,
        schedule_type="linear",
    )

    # --------------------------------------------------
    # Checkpoint
    # --------------------------------------------------

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

    text_encoder.eval()
    conditioning_projection.eval()
    unet.eval()

    print("\n===== CHECKPOINT =====")
    print(f"Epoch: {checkpoint['epoch']}")
    print(f"Train loss: {checkpoint['train_loss']:.4f}")
    print(
        f"Validation loss: "
        f"{checkpoint['validation_loss']:.4f}"
    )

    # --------------------------------------------------
    # Output
    # --------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    generator = torch.Generator(
        device=device
    )
    generator.manual_seed(SEED)

    manifest_rows = []

    print("\n===== ORDINARY TEST SET SAMPLING =====")
    print(f"Test samples: {len(test_indices)}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Sampling seed: {SEED}")

    for start in range(
        0,
        len(test_indices),
        BATCH_SIZE,
    ):

        batch_indices = test_indices[
            start:start + BATCH_SIZE
        ]

        captions = []

        for dataset_index in batch_indices:

            sample = dataset.samples[dataset_index]

            metadata = dataset._load_metadata(
                sample["metadata"]
            )

            captions.append(
                generate_caption(metadata)
            )

        images = sample_conditional_batch(
            captions=captions,
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            conditioning_projection=conditioning_projection,
            timestep_embedding=timestep_embedding,
            unet=unet,
            scheduler=scheduler,
            device=device,
            generator=generator,
        )

        # Sanity checks
        expected_shape = (
            len(batch_indices),
            3,
            32,
            32,
        )

        if images.shape != expected_shape:
            raise RuntimeError(
                f"Unexpected batch shape: {images.shape}"
            )

        if not torch.isfinite(images).all():
            raise RuntimeError(
                "Generated images contain non-finite values."
            )

        if images.min().item() < 0.0:
            raise RuntimeError(
                "Generated images contain values < 0."
            )

        if images.max().item() > 1.0:
            raise RuntimeError(
                "Generated images contain values > 1."
            )

        # Save images
        for local_index, (
            dataset_index,
            image,
        ) in enumerate(
            zip(batch_indices, images)
        ):

            output_index = (
                start + local_index
            )

            output_path = (
                OUTPUT_DIR
                / f"test_{output_index:04d}.png"
            )

            save_image(
                image,
                output_path,
            )

            manifest_rows.append(
                {
                    "output_index": output_index,
                    "dataset_index": dataset_index,
                    "caption": captions[local_index],
                    "path": str(output_path),
                }
            )

        completed = min(
            start + len(batch_indices),
            len(test_indices),
        )

        print(
            f"Generated "
            f"{completed}/{len(test_indices)}"
        )

    # --------------------------------------------------
    # Manifest
    # --------------------------------------------------

    with MANIFEST_PATH.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=[
                "output_index",
                "dataset_index",
                "caption",
                "path",
            ],
        )

        writer.writeheader()
        writer.writerows(manifest_rows)

    # --------------------------------------------------
    # Final checks
    # --------------------------------------------------

    generated_files = sorted(
        OUTPUT_DIR.glob("test_*.png")
    )

    print("\n===== SANITY CHECK =====")
    print(
        f"Expected images: "
        f"{len(test_indices)}"
    )
    print(
        f"Generated images: "
        f"{len(generated_files)}"
    )
    print(
        f"Manifest rows: "
        f"{len(manifest_rows)}"
    )

    if len(generated_files) != len(test_indices):
        raise RuntimeError(
            "Generated image count does not match "
            "test set size."
        )

    if len(manifest_rows) != len(test_indices):
        raise RuntimeError(
            "Manifest size does not match "
            "test set size."
        )

    print(
        "Ordinary test-set sampling "
        "completed successfully."
    )


if __name__ == "__main__":
    main()