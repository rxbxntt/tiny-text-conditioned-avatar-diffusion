from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(
    "/kaggle/input/datasets/rubinett/dl-avatar/"
    "tiny-text-conditioned-avatar-diffusion-development"
)

DATASET_ROOT = Path(
    "/kaggle/input/datasets/rubinett/full-10k-set/0"
)

CHECKPOINT_PATH = Path(
    "/kaggle/input/final_cfg_checkpoint/final_cfg_checkpoint.pt"
)

OUTPUT_DIR = Path(
    "/kaggle/working/final_evaluation"
)

DIVERSITY_DIR = (
    OUTPUT_DIR / "diversity"
)

NUM_TIMESTEPS = 1000
SCHEDULE_TYPE = "linear"

SEED = 42
GUIDANCE_SCALE = 2.0

DIVERSITY_SEEDS = [
    42,
    123,
    456,
    789,
    1000,
]

NUM_DIVERSITY_PROMPTS = 5


# ============================================================
# PROJECT IMPORTS
# ============================================================

sys.path.insert(0, str(PROJECT_ROOT))

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
# DEVICE
# ============================================================

if not torch.cuda.is_available():
    raise RuntimeError("CUDA non disponibile.")

device = torch.device("cuda")

print("Device:", torch.cuda.get_device_name(0))


# ============================================================
# DATASET AND TOKENIZER
# ============================================================

dataset = CartoonSetDataset(
    root_dir=DATASET_ROOT,
    expected_samples=10_000,
)

(
    train_indices,
    _,
    _,
    _,
) = create_dataset_splits(
    dataset=dataset,
    ood_hair_color=4,
    ood_eye_color=1,
    seed=42,
)

tokenizer = CaptionTokenizer()

train_captions = []

for index in train_indices:
    sample = dataset.samples[index]

    metadata = dataset._load_metadata(
        sample["metadata"]
    )

    train_captions.append(
        generate_caption(metadata)
    )

tokenizer.fit(train_captions)

print(
    "Tokenizer vocabulary size:",
    tokenizer.vocab_size,
)


# ============================================================
# MODEL
# ============================================================

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


# ============================================================
# LOAD CHECKPOINT
# ============================================================

checkpoint = torch.load(
    CHECKPOINT_PATH,
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
        f"Checkpoint missing keys: {sorted(missing)}"
    )

model_state = checkpoint["model_state_dict"]

text_encoder.load_state_dict(
    model_state["text_encoder"]
)

conditioning_projection.load_state_dict(
    model_state["conditioning_projection"]
)

timestep_embedding.load_state_dict(
    model_state["timestep_embedding"]
)

unet.load_state_dict(
    model_state["unet"]
)

text_encoder.eval()
conditioning_projection.eval()
timestep_embedding.eval()
unet.eval()

print("Checkpoint loaded successfully.")
print("Checkpoint epoch:", checkpoint["epoch"])


# ============================================================
# CFG SAMPLER
# ============================================================

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
    # --------------------------------------------------------
    # Tokenization
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Text conditioning
    # --------------------------------------------------------

    text_features = text_encoder(
        input_ids=input_ids,
        attention_mask=attention_mask,
    )

    conditional_context = conditioning_projection(
        text_features
    )

    # --------------------------------------------------------
    # Unconditional context
    # --------------------------------------------------------

    unconditional_context = torch.zeros_like(
        conditional_context
    )

    # --------------------------------------------------------
    # Initial Gaussian noise
    # --------------------------------------------------------

    batch_size = len(captions)

    x = torch.randn(
        batch_size,
        3,
        32,
        32,
        device=device,
        generator=generator,
    )

    # --------------------------------------------------------
    # Reverse diffusion with CFG
    # --------------------------------------------------------

    for timestep in reversed(range(NUM_TIMESTEPS)):

        timesteps = torch.full(
            (batch_size,),
            timestep,
            dtype=torch.long,
            device=device,
        )

        time_embedding = timestep_embedding(
            timesteps
        )

        eps_cond = unet(
            x=x,
            time_embedding=time_embedding,
            context=conditional_context,
        )

        eps_uncond = unet(
            x=x,
            time_embedding=time_embedding,
            context=unconditional_context,
        )

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


# ============================================================
# PREPARE OUTPUT DIRECTORY
# ============================================================

DIVERSITY_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# LOAD FIXED PROMPTS
# ============================================================

ordinary_manifest = (
    OUTPUT_DIR
    / "ordinary_test"
    / "ordinary_test_manifest.csv"
)

if not ordinary_manifest.exists():
    raise FileNotFoundError(
        f"Manifest non trovato: {ordinary_manifest}"
    )

selected_prompts = []

with ordinary_manifest.open(
    "r",
    encoding="utf-8",
) as file:

    reader = csv.DictReader(file)

    for row in reader:

        selected_prompts.append(
            row["caption"]
        )

        if len(selected_prompts) >= NUM_DIVERSITY_PROMPTS:
            break


if len(selected_prompts) < NUM_DIVERSITY_PROMPTS:
    raise RuntimeError(
        "Il manifest non contiene abbastanza prompt."
    )


print("\nSelected prompts:")

for index, caption in enumerate(
    selected_prompts
):
    print(
        f"{index}: {caption}"
    )


# ============================================================
# GENERATE SAME PROMPTS WITH DIFFERENT SEEDS
# ============================================================

all_images = {}

print("\n" + "=" * 60)
print("DIVERSITY GENERATION")
print("=" * 60)

print(
    f"Prompts: {NUM_DIVERSITY_PROMPTS}"
)

print(
    f"Seeds: {DIVERSITY_SEEDS}"
)

print(
    "Expected images:",
    NUM_DIVERSITY_PROMPTS
    * len(DIVERSITY_SEEDS),
)


for prompt_idx, caption in enumerate(
    selected_prompts
):

    print(
        f"\nPrompt "
        f"{prompt_idx + 1}/"
        f"{NUM_DIVERSITY_PROMPTS}"
    )

    print(caption)

    all_images[prompt_idx] = []

    for seed in DIVERSITY_SEEDS:

        # Dedicated generator for reproducibility.
        generator = torch.Generator(
            device=device
        )

        generator.manual_seed(seed)

        with torch.inference_mode():

            image_batch = sample_cfg_batch(
                captions=[caption],
                tokenizer=tokenizer,
                text_encoder=text_encoder,
                conditioning_projection=(
                    conditioning_projection
                ),
                timestep_embedding=(
                    timestep_embedding
                ),
                unet=unet,
                scheduler=scheduler,
                device=device,
                generator=generator,
                guidance_scale=(
                    GUIDANCE_SCALE
                ),
            )

        image = image_batch[0]

        # Convert [C, H, W] to [H, W, C].
        image_np = (
            image.detach()
            .cpu()
            .permute(1, 2, 0)
            .numpy()
            .astype(np.float32)
        )

        all_images[prompt_idx].append(
            image_np
        )

        # Save generated image.
        image_uint8 = np.clip(
            image_np * 255.0,
            0,
            255,
        ).astype(np.uint8)

        image_path = (
            DIVERSITY_DIR
            / (
                f"prompt_{prompt_idx:02d}"
                f"_seed_{seed}.png"
            )
        )

        Image.fromarray(
            image_uint8
        ).save(image_path)

        print(
            f"  Seed {seed}: saved"
        )


# ============================================================
# MEAN PAIRWISE PIXEL L1
# ============================================================

pairwise_distances = []


for prompt_idx in range(
    NUM_DIVERSITY_PROMPTS
):

    images = all_images[prompt_idx]

    # Compare every pair of seeds
    # for the same prompt.
    for i in range(len(images)):

        for j in range(
            i + 1,
            len(images),
        ):

            distance = np.mean(
                np.abs(
                    images[i]
                    - images[j]
                )
            )

            pairwise_distances.append(
                float(distance)
            )


expected_comparisons = (
    NUM_DIVERSITY_PROMPTS
    * (
        len(DIVERSITY_SEEDS)
        * (len(DIVERSITY_SEEDS) - 1)
        // 2
    )
)

if len(pairwise_distances) != expected_comparisons:
    raise RuntimeError(
        "Unexpected number of pairwise "
        "comparisons: "
        f"{len(pairwise_distances)} "
        f"(expected "
        f"{expected_comparisons})."
    )


diversity_mean = float(
    np.mean(pairwise_distances)
)

diversity_std = float(
    np.std(pairwise_distances)
)


# ============================================================
# PRINT RESULTS
# ============================================================

print("\n" + "=" * 60)
print("DIVERSITY RESULTS")
print("=" * 60)

print(
    f"Prompts:              "
    f"{NUM_DIVERSITY_PROMPTS}"
)

print(
    f"Seeds:                "
    f"{DIVERSITY_SEEDS}"
)

print(
    f"Generated images:     "
    f"{NUM_DIVERSITY_PROMPTS * len(DIVERSITY_SEEDS)}"
)

print(
    f"Pairwise comparisons: "
    f"{len(pairwise_distances)}"
)

print(
    f"Mean pairwise L1:     "
    f"{diversity_mean:.8f}"
)

print(
    f"Std pairwise L1:      "
    f"{diversity_std:.8f}"
)


# ============================================================
# UPDATE RESULTS JSON
# ============================================================

results_path = (
    OUTPUT_DIR
    / "results_without_kid.json"
)

if results_path.exists():

    with results_path.open(
        "r",
        encoding="utf-8",
    ) as file:

        results = json.load(file)

else:

    results = {}


results["diversity"] = {
    "metric": "mean_pairwise_pixel_L1",
    "num_prompts": (
        NUM_DIVERSITY_PROMPTS
    ),
    "seeds": DIVERSITY_SEEDS,
    "num_generated_images": (
        NUM_DIVERSITY_PROMPTS
        * len(DIVERSITY_SEEDS)
    ),
    "num_pairwise_comparisons": (
        len(pairwise_distances)
    ),
    "mean": diversity_mean,
    "std": diversity_std,
}


with results_path.open(
    "w",
    encoding="utf-8",
) as file:

    json.dump(
        results,
        file,
        indent=4,
    )


print(
    f"\nUpdated: {results_path}"
)