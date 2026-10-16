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

from pathlib import Path
from torchvision.utils import save_image

def fit_tokenizer_on_indices(dataset, indices, tokenizer):
    captions = []

    for index in indices:
        sample = dataset.samples[index]
        metadata = dataset._load_metadata(sample["metadata"])
        caption = generate_caption(metadata)
        captions.append(caption)

    tokenizer.fit(captions)

    return tokenizer

@torch.no_grad()
def sample_conditional(
    captions,
    tokenizer,
    text_encoder,
    conditioning_projection,
    timestep_embedding,
    unet,
    scheduler,
    device,
    seed=None,
):
    """
    Generate images conditioned on textual captions.

    Args:
        captions: list[str]
        tokenizer: train-fitted tokenizer
        text_encoder: trained text encoder
        conditioning_projection: 64 -> 128
        timestep_embedding: timestep embedding module
        unet: trained conditional U-Net
        scheduler: DDPM scheduler

    Returns:
        Tensor [B, 3, 32, 32]
    """

    # ---------------------------------------------------------
    # 1. Tokenize captions
    # ---------------------------------------------------------
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
        attention_masks.append(tokenizer.attention_mask(token_ids))

    input_ids = torch.tensor(input_ids, dtype=torch.long).to(device)
    attention_mask = torch.tensor(attention_masks, dtype=torch.bool).to(device)

    # ---------------------------------------------------------
    # 2. Encode text
    # ---------------------------------------------------------
    text_features = text_encoder(
        input_ids=input_ids,
        attention_mask=attention_mask,
    )

    # ---------------------------------------------------------
    # 3. Project text features for U-Net conditioning
    # ---------------------------------------------------------
    context = conditioning_projection(text_features)

    batch_size = len(captions)

    # ---------------------------------------------------------
    # 4. Start from Gaussian noise
    # ---------------------------------------------------------
    generator = None
    if seed is not None:
        generator = torch.Generator(device=device)
        generator.manual_seed(seed)

    x = torch.randn(
        batch_size,
        3,
        32,
        32,
        device=device,
        generator=generator,
    )

    # ---------------------------------------------------------
    # 5. Reverse diffusion
    # ---------------------------------------------------------
    for timestep in reversed(range(1000)):

        timesteps = torch.full(
            (batch_size,),
            timestep,
            dtype=torch.long,
            device=device,
        )

        # timestep conditioning
        time_embedding = timestep_embedding(timesteps)

        # predict noise conditioned on text
        predicted_noise = unet(
            x=x,
            time_embedding=time_embedding,
            context=context,
        )

        # DDPM reverse step
        x = scheduler.reverse_step(
            x_t=x,
            predicted_noise=predicted_noise,
            timesteps=timesteps,
        )

    # ---------------------------------------------------------
    # 6. Convert to valid image range
    # ---------------------------------------------------------
    x = x.clamp(0.0, 1.0)

    return x



def main():

    dataset_root = (
        r"C:\Users\utente\.cache\kagglehub\datasets"
        r"\paraglondhe\cartoonset100k\versions\2"
        r"\cartoonset100k\0"
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Device: {device}")

    # --------------------------------------------------
    # Dataset
    # --------------------------------------------------

    dataset = CartoonSetDataset(
        root_dir=dataset_root,
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

    print("\n===== DATASET SPLIT =====")
    print(f"Train: {len(train_indices)}")
    print(f"Validation: {len(val_indices)}")
    print(f"Test: {len(test_indices)}")
    print(f"OOD: {len(ood_indices)}")

    # --------------------------------------------------
    # Tokenizer
    # --------------------------------------------------

    tokenizer = CaptionTokenizer()

    fit_tokenizer_on_indices(
        dataset=dataset,
        indices=train_indices,
        tokenizer=tokenizer,
    )

    print("\n===== TOKENIZER =====")
    print(f"Vocabulary size: {tokenizer.vocab_size}")

    # --------------------------------------------------
    # Model components
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
        "checkpoint_conditioned_ep50/checkpoint_epoch_50.pt",
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
    print(f"Validation loss: {checkpoint['validation_loss']:.4f}")

    # --------------------------------------------------
    # Test captions
    # --------------------------------------------------

    caption = (
        "eye_angle_0 eye_lashes_0 eye_lid_0 chin_length_2 "
        "eyebrow_weight_1 eyebrow_shape_0 eyebrow_thickness_1 "
        "face_shape_2 facial_hair_0 hair_1 eye_color_3 face_color_2 "
        "hair_color_5 glasses_0 glasses_color_0 eye_slant_0 "
        "eyebrow_width_2 eye_eyebrow_distance_2"
    )

    seeds = [42, 123, 456, 789, 1000]

    print("\n===== DIVERSITY ACROSS SEEDS =====")
    print(f"Caption: {caption}")
    print(f"Seeds: {seeds}")

    output_dir = Path("outputs/diversity")
    output_dir.mkdir(parents=True, exist_ok=True)

    for seed in seeds:
        images = sample_conditional(
            captions=[caption],
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            conditioning_projection=conditioning_projection,
            timestep_embedding=timestep_embedding,
            unet=unet,
            scheduler=scheduler,
            device=device,
            seed=seed,
        )

        image = images[0]

        print(
            f"Seed {seed} | "
            f"shape={tuple(image.shape)} | "
            f"min={image.min().item():.4f} | "
            f"max={image.max().item():.4f} | "
            f"finite={torch.isfinite(image).all().item()}"
        )

        save_image(
            image,
            output_dir / f"sample_seed_{seed}.png",
        )

    print(f"\nImages saved in: {output_dir}")

    print(f"Output shape: {images.shape}")
    print(f"Min: {images.min().item():.4f}")
    print(f"Max: {images.max().item():.4f}")
    print(f"Mean: {images.mean().item():.4f}")
    print(f"Finite: {torch.isfinite(images).all().item()}")

    # --------------------------------------------------
    # Save images
    # --------------------------------------------------

    output_dir = Path("outputs/conditional")
    output_dir.mkdir(parents=True, exist_ok=True)

    for i, image in enumerate(images):
        save_image(
            image,
            output_dir / f"sample_{i}.png",
        )

    print(f"\nImages saved in: {output_dir}")


if __name__ == "__main__":
    main()