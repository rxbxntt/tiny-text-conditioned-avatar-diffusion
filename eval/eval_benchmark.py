from __future__ import annotations

import csv
import json
import statistics
import sys
import time
from pathlib import Path

import torch


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

NUM_TIMESTEPS = 1000
SCHEDULE_TYPE = "linear"

SEED = 42
GUIDANCE_SCALE = 2.0

WARMUP_RUNS = 1
MEASURED_RUNS = 5


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

print("Tokenizer vocabulary size:", tokenizer.vocab_size)


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
    # Initial noise
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
# BENCHMARK CAPTION
# ============================================================

manifest_path = (
    OUTPUT_DIR
    / "ordinary_test"
    / "ordinary_test_manifest.csv"
)

if not manifest_path.exists():
    raise FileNotFoundError(
        f"Manifest non trovato: {manifest_path}"
    )

with manifest_path.open(
    "r",
    encoding="utf-8",
) as file:

    reader = csv.DictReader(file)

    first_row = next(reader, None)

if first_row is None:
    raise RuntimeError(
        "Il manifest ordinary_test è vuoto."
    )

benchmark_caption = first_row["caption"]

print("\nBenchmark caption:")
print(benchmark_caption)


# ============================================================
# BENCHMARK CONFIGURATION
# ============================================================

print("\nBenchmark configuration:")
print("  Timesteps:", NUM_TIMESTEPS)
print("  Guidance scale:", GUIDANCE_SCALE)
print("  Warm-up runs:", WARMUP_RUNS)
print("  Measured runs:", MEASURED_RUNS)


# ============================================================
# GPU MEMORY RESET
# ============================================================

torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats(device)


# ============================================================
# WARM-UP
# ============================================================

print("\nRunning warm-up...")

with torch.inference_mode():

    for i in range(WARMUP_RUNS):

        generator = torch.Generator(
            device=device
        )

        generator.manual_seed(
            SEED + i
        )

        _ = sample_cfg_batch(
            [benchmark_caption],
            tokenizer,
            text_encoder,
            conditioning_projection,
            timestep_embedding,
            unet,
            scheduler,
            device,
            generator,
            guidance_scale=GUIDANCE_SCALE,
        )

torch.cuda.synchronize()

# Do not include warm-up in memory measurement.
torch.cuda.reset_peak_memory_stats(device)


# ============================================================
# TIMED RUNS
# ============================================================

times = []

print("\nRunning timed samples...")

with torch.inference_mode():

    for i in range(MEASURED_RUNS):

        generator = torch.Generator(
            device=device
        )

        generator.manual_seed(
            SEED + 100 + i
        )

        torch.cuda.synchronize()

        start = time.perf_counter()

        _ = sample_cfg_batch(
            [benchmark_caption],
            tokenizer,
            text_encoder,
            conditioning_projection,
            timestep_embedding,
            unet,
            scheduler,
            device,
            generator,
            guidance_scale=GUIDANCE_SCALE,
        )

        torch.cuda.synchronize()

        elapsed = time.perf_counter() - start

        times.append(elapsed)

        print(
            f"Run {i + 1}/{MEASURED_RUNS}: "
            f"{elapsed:.4f} s/image"
        )


# ============================================================
# RESULTS
# ============================================================

peak_allocated_mb = (
    torch.cuda.max_memory_allocated(device)
    / (1024 ** 2)
)

peak_reserved_mb = (
    torch.cuda.max_memory_reserved(device)
    / (1024 ** 2)
)

mean_time = sum(times) / len(times)

if len(times) > 1:
    std_time = statistics.stdev(times)
else:
    std_time = 0.0


print("\n" + "=" * 60)
print("CFG SAMPLING BENCHMARK")
print("=" * 60)

print(
    f"Mean sampling time : "
    f"{mean_time:.4f} s/image"
)

print(
    f"Std sampling time  : "
    f"{std_time:.4f} s/image"
)

print(
    f"Peak allocated     : "
    f"{peak_allocated_mb:.2f} MB"
)

print(
    f"Peak reserved      : "
    f"{peak_reserved_mb:.2f} MB"
)

print("=" * 60)


# ============================================================
# SAVE RESULTS
# ============================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

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


results["sampling_time"] = {
    "metric": "seconds_per_image",
    "guidance_scale": GUIDANCE_SCALE,
    "num_timesteps": NUM_TIMESTEPS,
    "warmup_runs": WARMUP_RUNS,
    "measured_runs": MEASURED_RUNS,
    "mean": mean_time,
    "std": std_time,
}

results["gpu_memory"] = {
    "device": torch.cuda.get_device_name(0),
    "peak_allocated_mb": peak_allocated_mb,
    "peak_reserved_mb": peak_reserved_mb,
}


with results_path.open(
    "w",
    encoding="utf-8",
) as file:

    json.dump(
        results,
        file,
        indent=2,
    )


print(
    f"\nUpdated: {results_path}"
)