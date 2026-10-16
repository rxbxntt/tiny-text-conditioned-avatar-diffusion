import time
from pathlib import Path

import torch

from data.cartoon_dataset import CartoonSetDataset
from data.dataset_split import create_dataset_splits
from text.caption_generator import generate_caption

from text.tokenizer import CaptionTokenizer
from text.text_encoder import TextEncoder
from text.text_conditioning import TextConditioningProjection

from diffusion.time_step_embedding import TimestepEmbedding
from diffusion.unet import DiffusionUNet
from diffusion.diffuser import DiffusionScheduler

from training.sample_conditional import sample_conditional

# ============================================================
# Configuration
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

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

NUM_SAMPLES = 1
MAX_LENGTH = 18
SEED = 42

def fit_tokenizer_on_train(dataset, train_indices, tokenizer):
    captions = []

    for index in train_indices:
        sample = dataset.samples[index]

        metadata = dataset._load_metadata(
            sample["metadata"]
        )

        caption = generate_caption(metadata)
        captions.append(caption)

    tokenizer.fit(captions)

def load_components(device):
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

    tokenizer = CaptionTokenizer()

    fit_tokenizer_on_train(
        dataset,
        train_indices,
        tokenizer,
    )

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
    ).to(device)

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

    return (
        dataset,
        train_indices,
        tokenizer,
        text_encoder,
        conditioning_projection,
        timestep_embedding,
        unet,
        scheduler,
    )

def get_test_caption(dataset, test_indices):
    index = test_indices[0]

    sample = dataset.samples[index]

    metadata = dataset._load_metadata(
        sample["metadata"]
    )

    return generate_caption(metadata)

def main():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Device: {device}")

    (
        dataset,
        train_indices,
        tokenizer,
        text_encoder,
        conditioning_projection,
        timestep_embedding,
        unet,
        scheduler,
    ) = load_components(device)

    # --------------------------------------------------------
    # Test caption
    # --------------------------------------------------------

    _, _, test_indices, _ = create_dataset_splits(
        dataset=dataset,
        ood_hair_color=4,
        ood_eye_color=1,
        seed=42,
    )

    caption = get_test_caption(
        dataset,
        test_indices,
    )

    print("\n===== CAPTION =====")
    print(caption)

    # --------------------------------------------------------
    # Sampling test
    # --------------------------------------------------------

    torch.manual_seed(SEED)

    print("\n===== SAMPLING BENCHMARK =====")

    # Warm-up: non viene misurato
    with torch.no_grad():
        _ = sample_conditional(
            captions=[caption],
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            conditioning_projection=conditioning_projection,
            timestep_embedding=timestep_embedding,
            unet=unet,
            scheduler=scheduler,
            device=device,
        )

    print("Warm-up completed.")

    # Misurazioni
    NUM_RUNS = 5
    sampling_times = []

    with torch.no_grad():
        for run in range(NUM_RUNS):
            start_time = time.perf_counter()

            images = sample_conditional(
                captions=[caption],
                tokenizer=tokenizer,
                text_encoder=text_encoder,
                conditioning_projection=conditioning_projection,
                timestep_embedding=timestep_embedding,
                unet=unet,
                scheduler=scheduler,
                device=device,
            )

            elapsed_time = time.perf_counter() - start_time
            sampling_times.append(elapsed_time)

            print(f"Run {run + 1}: {elapsed_time:.4f} s")

    mean_time = sum(sampling_times) / len(sampling_times)

    variance = sum(
        (t - mean_time) ** 2 for t in sampling_times
    ) / (len(sampling_times) - 1)

    std_time = variance ** 0.5

    print("\n===== RESULTS =====")
    print(f"Mean sampling time: {mean_time:.4f} s")
    print(f"Std sampling time:  {std_time:.4f} s")
    print(f"Mean time per image: {mean_time / NUM_SAMPLES:.4f} s")

    print(f"Output shape: {images.shape}")
    print(f"Min: {images.min().item():.4f}")
    print(f"Max: {images.max().item():.4f}")
    print(f"Finite: {torch.isfinite(images).all().item()}")


if __name__ == "__main__":
    main()