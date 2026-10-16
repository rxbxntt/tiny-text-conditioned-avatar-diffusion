import os

import torch
from PIL import Image

from data.cartoon_dataset import CartoonSetDataset
from data.dataset_split import create_dataset_splits

from text.caption_generator import generate_caption
from text.tokenizer import CaptionTokenizer
from text.text_encoder import TextEncoder
from text.text_conditioning import TextConditioningProjection

from diffusion.time_step_embedding import TimestepEmbedding
from diffusion.unet import DiffusionUNet
from diffusion.diffuser import DiffusionScheduler
from diffusion.sampler import sample


DATASET_ROOT = (
    r"C:\Users\utente\.cache\kagglehub\datasets"
    r"\paraglondhe\cartoonset100k\versions\2"
    r"\cartoonset100k\0"
)

CHECKPOINT_PATH = "checkpoint_conditioned_ep50/checkpoint_epoch_50.pt"
OUTPUT_DIR = "generated_samples"

NUM_SAMPLES = 4
MAX_LENGTH = 18


def fit_tokenizer_on_indices(dataset, indices, tokenizer):
    captions = []

    for index in indices:
        dataset_sample = dataset.samples[index]

        metadata = dataset._load_metadata(
            dataset_sample["metadata"]
        )

        caption = generate_caption(metadata)
        captions.append(caption)

    tokenizer.fit(captions)

    return tokenizer


def load_components(device):
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
        ood_hair_color=4,
        ood_eye_color=1,
        seed=42,
    )

    # --------------------------------------------------
    # Tokenizer
    # --------------------------------------------------

    tokenizer = CaptionTokenizer()

    fit_tokenizer_on_indices(
        dataset=dataset,
        indices=train_indices,
        tokenizer=tokenizer,
    )

    print(
        f"Tokenizer vocabulary size: "
        f"{tokenizer.vocab_size}"
    )

    # --------------------------------------------------
    # Models
    # --------------------------------------------------

    text_encoder = TextEncoder(
        vocab_size=tokenizer.vocab_size,
        hidden_size=64,
        num_layers=2,
        num_heads=4,
        max_length=18,
        dropout=0.0,
    ).to(device)

    text_conditioning = TextConditioningProjection(
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

    # --------------------------------------------------
    # Checkpoint
    # --------------------------------------------------

    if not os.path.exists(CHECKPOINT_PATH):
        raise FileNotFoundError(
            f"Checkpoint non trovato: {CHECKPOINT_PATH}"
        )

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
        f"Loaded checkpoint from epoch "
        f"{checkpoint['epoch']}"
    )

    return (
        dataset,
        val_indices,
        tokenizer,
        text_encoder,
        text_conditioning,
        timestep_embedding,
        unet,
        scheduler,
    )


def save_tensor_as_image(tensor, path):
    """
    tensor:
        [3, 32, 32], expected range approximately [0, 1]
    """

    tensor = tensor.detach().cpu()

    # The sampler can produce values slightly outside [0, 1].
    tensor = tensor.clamp(0.0, 1.0)

    tensor = tensor.permute(1, 2, 0)

    array = (tensor.numpy() * 255.0).round().astype("uint8")

    image = Image.fromarray(array)

    image.save(path)


def main():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Device: {device}")

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True,
    )

    (
        dataset,
        val_indices,
        tokenizer,
        text_encoder,
        text_conditioning,
        timestep_embedding,
        unet,
        scheduler,
    ) = load_components(device)

    # --------------------------------------------------
    # Select validation captions
    # --------------------------------------------------

    selected_indices = val_indices[:NUM_SAMPLES]

    captions = []

    for index in selected_indices:
        dataset_sample = dataset.samples[index]

        metadata = dataset._load_metadata(
            dataset_sample["metadata"]
        )

        caption = generate_caption(metadata)

        captions.append(caption)

    print("\n===== CAPTIONS =====")

    for i, caption in enumerate(captions):
        print(f"{i}: {caption}")

    # --------------------------------------------------
    # Sampling
    # --------------------------------------------------

    text_encoder.eval()
    text_conditioning.eval()
    unet.eval()

    print("\n===== SAMPLING =====")

    with torch.no_grad():

        generated_images = sample(
            captions=captions,
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            text_conditioning=text_conditioning,
            timestep_embedding=timestep_embedding,
            unet=unet,
            scheduler=scheduler,
            device=device,
            max_length=MAX_LENGTH,
        )

    # --------------------------------------------------
    # Save images
    # --------------------------------------------------

    print("\n===== SAVING =====")

    for i, image_tensor in enumerate(generated_images):
        # ----------------------------------------------
        # Generated image
        # ----------------------------------------------

        generated_path = os.path.join(
            OUTPUT_DIR,
            f"generated_{i}.png",
        )

        save_tensor_as_image(
            image_tensor,
            generated_path,
        )

        # ----------------------------------------------
        # Ground-truth image
        # ----------------------------------------------

        ground_truth_path = os.path.join(
            OUTPUT_DIR,
            f"ground_truth_{i}.png",
        )

        save_ground_truth_image(
            dataset,
            selected_indices[i],
            ground_truth_path,
        )

        print(f"Generated:    {generated_path}")
        print(f"Ground truth: {ground_truth_path}")

def save_ground_truth_image(dataset, index, path):
    """
    Salva l'immagine originale del dataset.
    """

    sample = dataset[index]

    image = sample["image"]

    if not isinstance(image, Image.Image):
        raise TypeError(
            "Il dataset deve restituire una PIL Image."
        )

    image = image.convert("RGB")
    image.save(path)


if __name__ == "__main__":
    main()