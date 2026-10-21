"""
Generate images from the final text-conditioned diffusion model.

This script:
1. Loads the dataset and reproduces the fixed train/validation/test/OOD split.
2. Fits the tokenizer using only the training captions.
3. Reconstructs the text encoder, conditioning module, timestep embedding,
   U-Net, and diffusion scheduler.
4. Loads the final checkpoint.
5. Generates the ordinary test set and the compositional OOD set using
   classifier-free guidance (CFG).
6. Saves generated images together with CSV manifests.

Evaluation metrics such as KID, diversity, parameter count, and sampling
benchmark are handled by separate scripts.
"""



import sys
import json

PROJECT_ROOT = "/kaggle/input/datasets/rubinett/dl-avatar/tiny-text-conditioned-avatar-diffusion-development"
sys.path.insert(0, PROJECT_ROOT)

from __future__ import annotations

import sys
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

# Change only this path if the Kaggle dataset slug is different.


DATASET_ROOT = Path(
    "/kaggle/input/datasets/rubinett/full-10k-set/0"
)

# Upload final_cfg_checkpoint.pt as a Kaggle dataset or copy it to /kaggle/working.
CHECKPOINT_PATH = Path(
    "/kaggle/input/datasets/rubinett/final-cfg-checkpoint/final_cfg_checkpoint.pt"
)

OUTPUT_ROOT = Path("/kaggle/working/final_evaluation")

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

IMAGE_SIZE = 32
NUM_TIMESTEPS = 1000
SCHEDULE_TYPE = "linear"

BATCH_SIZE = 16
SEED = 42
GUIDANCE_SCALE = 2.0


OOD_HAIR_COLOR = 4
OOD_EYE_COLOR = 1




# ============================================================
# DATA / TOKENIZER
# ============================================================

def fit_tokenizer_on_indices(dataset, indices, tokenizer):
    captions = []

    for index in indices:
        sample = dataset.samples[index]
        metadata = dataset._load_metadata(sample["metadata"])
        captions.append(generate_caption(metadata))

    tokenizer.fit(captions)
    return tokenizer


def get_sorted_dataset_images(dataset_root: Path):
    paths = sorted(dataset_root.glob("*.png"))

    if len(paths) != 10_000:
        raise RuntimeError(
            f"Expected 10000 dataset PNGs, found {len(paths)}."
        )

    return paths


# ============================================================
# MODEL
# ============================================================

def build_models(tokenizer, device):
    text_encoder = TextEncoder(
        vocab_size=tokenizer.vocab_size,
        hidden_size=64,
        num_layers=2,
        num_heads=4,
        max_length=128,
        dropout=0.0,
    ).to(device)

    conditioning_projection = TextConditioningProjection(
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

    scheduler = DiffusionScheduler(
        num_timesteps=NUM_TIMESTEPS,
        schedule_type=SCHEDULE_TYPE,
    )

    scheduler.betas = scheduler.betas.to(device)
    scheduler.alphas = scheduler.alphas.to(device)
    scheduler.alpha_bars = scheduler.alpha_bars.to(device)

    return (
        text_encoder,
        conditioning_projection,
        timestep_embedding,
        unet,
        scheduler,
    )


def load_checkpoint(
    checkpoint_path,
    text_encoder,
    conditioning_projection,
    timestep_embedding,
    unet,
    device,
):
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )

    required_keys = {
        "epoch",
        "model_state_dict",
        "train_loss",
        "val_loss",
        "config",
    }

    missing = required_keys - set(checkpoint.keys())

    if missing:
        raise RuntimeError(
            f"Checkpoint is missing required keys: {sorted(missing)}"
        )

    model_state = checkpoint["model_state_dict"]

    expected_model_keys = {
        "text_encoder",
        "conditioning_projection",
        "timestep_embedding",
        "unet",
    }

    missing_model = expected_model_keys - set(model_state.keys())

    if missing_model:
        raise RuntimeError(
            "Checkpoint model_state_dict is missing: "
            f"{sorted(missing_model)}"
        )

    text_encoder.load_state_dict(model_state["text_encoder"])
    conditioning_projection.load_state_dict(
        model_state["conditioning_projection"]
    )
    timestep_embedding.load_state_dict(
        model_state["timestep_embedding"]
    )
    unet.load_state_dict(model_state["unet"])

    text_encoder.eval()
    conditioning_projection.eval()
    timestep_embedding.eval()
    unet.eval()

    return checkpoint


# ============================================================
# SAMPLING
# ============================================================

def encode_captions(captions, tokenizer, device):
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

    return input_ids, attention_mask


@torch.no_grad()
def sample_cfg_batch(
    captions,
    tokenizer,
    text_encoder,
    conditioning_projection,
    timestep_embedding,
    unet,
    scheduler,
    device,
    generator,
    guidance_scale=2.0,
):
    # --------------------------------------------------
    # Tokenization
    # --------------------------------------------------
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

    # --------------------------------------------------
    # Text conditioning
    # --------------------------------------------------
    text_features = text_encoder(
        input_ids=input_ids,
        attention_mask=attention_mask,
    )

    conditional_context = conditioning_projection(
        text_features
    )

    # CFG unconditional context:
    # same dimensionality, but zero conditioning
    unconditional_context = torch.zeros_like(
        conditional_context
    )

    # --------------------------------------------------
    # Initial Gaussian noise
    # --------------------------------------------------
    batch_size = len(captions)

    x = torch.randn(
        batch_size,
        3,
        32,
        32,
        device=device,
        generator=generator,
    )

    # --------------------------------------------------
    # Reverse diffusion + CFG
    # --------------------------------------------------
    for timestep in reversed(range(1000)):

        timesteps = torch.full(
            (batch_size,),
            timestep,
            dtype=torch.long,
            device=device,
        )

        time_embedding = timestep_embedding(
            timesteps
        )

        # Conditional prediction
        eps_cond = unet(
            x=x,
            time_embedding=time_embedding,
            context=conditional_context,
        )

        # Unconditional prediction
        eps_uncond = unet(
            x=x,
            time_embedding=time_embedding,
            context=unconditional_context,
        )

        # Classifier-Free Guidance
        predicted_noise = (
            eps_uncond
            + guidance_scale
            * (eps_cond - eps_uncond)
        )

        x = scheduler.reverse_step(
            x_t=x,
            predicted_noise=predicted_noise,
            timesteps=timesteps,
        )

    return x.clamp(0.0, 1.0)


def generate_split(
    split_name,
    indices,
    dataset,
    tokenizer,
    models,
    device,
    output_dir,
    seed,
):
    (
        text_encoder,
        conditioning_projection,
        timestep_embedding,
        unet,
        scheduler,
    ) = models

    output_dir.mkdir(parents=True, exist_ok=True)

    generator = torch.Generator(device=device)
    generator.manual_seed(seed)

    manifest_rows = []

    start_time = time.perf_counter()

    print("\n" + "=" * 60)
    print(f"GENERATING {split_name.upper()}")
    print("=" * 60)
    print(f"Samples: {len(indices)}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Seed: {seed}")
    print(f"Reverse steps: {NUM_TIMESTEPS}")

    for start in range(0, len(indices), BATCH_SIZE):
        batch_indices = indices[start:start + BATCH_SIZE]

        captions = []

        for dataset_index in batch_indices:
            sample = dataset.samples[dataset_index]

            metadata = dataset._load_metadata(
                sample["metadata"]
            )

            captions.append(
                generate_caption(metadata)
            )

        images = sample_cfg_batch(
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

        expected_shape = (
            len(batch_indices),
            3,
            IMAGE_SIZE,
            IMAGE_SIZE,
        )

        if images.shape != expected_shape:
            raise RuntimeError(
                f"{split_name}: unexpected shape {images.shape}"
            )

        if not torch.isfinite(images).all():
            raise RuntimeError(
                f"{split_name}: generated images contain NaN/Inf."
            )

        for local_index, (dataset_index, image) in enumerate(
            zip(batch_indices, images)
        ):
            output_index = start + local_index

            output_path = (
                output_dir
                / f"{split_name.lower()}_{output_index:04d}.png"
            )

            save_image(image, output_path)

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
            len(indices),
        )

        print(
            f"Generated {completed}/{len(indices)}",
            flush=True,
        )

    elapsed = time.perf_counter() - start_time

    manifest_path = (
        output_dir
        / f"{split_name.lower()}_manifest.csv"
    )

    with manifest_path.open(
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

    generated_files = sorted(output_dir.glob("*.png"))

    if len(generated_files) != len(indices):
        raise RuntimeError(
            f"{split_name}: expected {len(indices)} images, "
            f"found {len(generated_files)}."
        )

    print(
        f"{split_name}: completed in {elapsed / 60:.2f} min"
    )

    return {
        "split": split_name,
        "samples": len(indices),
        "elapsed_seconds": elapsed,
        "images": len(generated_files),
        "manifest": str(manifest_path),
    }





# ============================================================
# MAIN
# ============================================================

def main():
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. "
            "Run this evaluation on the Kaggle T4 GPU."
        )

    device = torch.device("cuda")

    print("=" * 60)
    print("FINAL CHECKPOINT EVALUATION")
    print("=" * 60)

    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"Dataset: {DATASET_ROOT}")
    print(f"Checkpoint: {CHECKPOINT_PATH}")

    if not DATASET_ROOT.exists():
        raise FileNotFoundError(
            f"Dataset path does not exist:\n{DATASET_ROOT}"
        )

    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"Checkpoint path does not exist:\n{CHECKPOINT_PATH}"
        )

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Dataset and split
    # --------------------------------------------------------

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

    assert len(test_indices) == 515
    assert len(ood_indices) == 230

    # --------------------------------------------------------
    # Tokenizer
    # --------------------------------------------------------

    tokenizer = CaptionTokenizer()

    fit_tokenizer_on_indices(
        dataset,
        train_indices,
        tokenizer,
    )

    print("\n===== TOKENIZER =====")
    print(f"Vocabulary size: {tokenizer.vocab_size}")

    assert tokenizer.vocab_size == 219

    # --------------------------------------------------------
    # Models
    # --------------------------------------------------------

    models = build_models(
        tokenizer,
        device,
    )

    checkpoint = load_checkpoint(
        CHECKPOINT_PATH,
        models[0],
        models[1],
        models[2],
        models[3],
        device,
    )

    print("\n===== CHECKPOINT =====")
    print(f"Epoch: {checkpoint['epoch']}")
    print(f"Train loss: {checkpoint['train_loss']:.6f}")
    print(f"Validation loss: {checkpoint['val_loss']:.6f}")
    print(f"Config: {checkpoint['config']}")



    # --------------------------------------------------------
    # Generation
    # --------------------------------------------------------

    ordinary_dir = OUTPUT_ROOT / "ordinary_test"
    ood_dir = OUTPUT_ROOT / "ood"

    ordinary_result = generate_split(
        split_name="ordinary_test",
        indices=test_indices,
        dataset=dataset,
        tokenizer=tokenizer,
        models=models,
        device=device,
        output_dir=ordinary_dir,
        seed=SEED,
    )

    ood_result = generate_split(
        split_name="ood",
        indices=ood_indices,
        dataset=dataset,
        tokenizer=tokenizer,
        models=models,
        device=device,
        output_dir=ood_dir,
        seed=SEED,
    )



    # --------------------------------------------------------
    # Save results
    # --------------------------------------------------------

    results = {
        "checkpoint": str(CHECKPOINT_PATH),
        "device": torch.cuda.get_device_name(0),
        "image_size": IMAGE_SIZE,
        "num_timesteps": NUM_TIMESTEPS,
        "schedule": SCHEDULE_TYPE,
        "batch_size": BATCH_SIZE,
        "seed": SEED,
        "split": {
            "train": len(train_indices),
            "validation": len(val_indices),
            "test": len(test_indices),
            "ood": len(ood_indices),
        },
        "checkpoint_epoch": checkpoint["epoch"],
        "train_loss": checkpoint["train_loss"],
        "val_loss": checkpoint["val_loss"],
        "generation": {
            "ordinary_test": ordinary_result,
            "ood": ood_result,
        },

    }

    results_path = (
        OUTPUT_ROOT
        / "final_evaluation_results.json"
    )

    with results_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            results,
            file,
            indent=2,
        )

    print("\n" + "=" * 60)
    print("FINAL EVALUATION COMPLETED 99999")
    print("=" * 60)
    print(f"Results: {results_path}")


if __name__ == "__main__":
    main()
