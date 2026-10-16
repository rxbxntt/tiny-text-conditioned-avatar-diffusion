from training.trainer import training_step

import time

from text.caption_generator import generate_caption
import torch
from training.dataloader import create_dataloader

from data.dataset_split import create_dataset_splits, create_torch_subsets
from data.cartoon_dataset import CartoonSetDataset
from preprocessing.image_processor import ImagePreprocessor
from text.tokenizer import CaptionTokenizer
from text.text_encoder import TextEncoder
from text.text_conditioning import TextConditioningProjection
from diffusion.time_step_embedding import TimestepEmbedding
from diffusion.unet import DiffusionUNet
from diffusion.diffuser import DiffusionScheduler, DiffusionForwardProcess


def fit_tokenizer_on_indices(dataset, indices, tokenizer):
    """
    Fits the tokenizer using captions generated only from
    the samples identified by the provided indices.
    """
    captions = []

    for index in indices:
        sample = dataset.samples[index]

        metadata = dataset._load_metadata(sample["metadata"])
        caption = generate_caption(metadata)

        captions.append(caption)

    tokenizer.fit(captions)

    return tokenizer



def create_training_components(dataset, train_indices, device):
    tokenizer = CaptionTokenizer()

    fit_tokenizer_on_indices(
        dataset=dataset,
        indices=train_indices,
        tokenizer=tokenizer,
    )

    print(f"Tokenizer training samples: {len(train_indices)}")
    print(f"Tokenizer vocabulary size: {tokenizer.vocab_size}")
    print("Tokenizer fitted: OK")

    # Preprocessing
    preprocessor = ImagePreprocessor(
        image_size=32,
        normalization="zero_one",
    )

    # Text encoder
    text_encoder = TextEncoder(
        vocab_size=tokenizer.vocab_size,
        hidden_size=64,
        num_layers=2,
        num_heads=4,
        max_length=18,
        dropout=0.0,
    ).to(device)

    # Text conditioning
    text_conditioning = TextConditioningProjection(
        input_dim=64,
        output_dim=128,
    ).to(device)

    # Timestep embedding
    timestep_embedding = TimestepEmbedding(
        num_timesteps=1000,
        embedding_dim=64,
    ).to(device)

    # Diffusion
    scheduler = DiffusionScheduler(
        num_timesteps=1000,
        schedule_type="linear",
    )

    diffusion_process = DiffusionForwardProcess(
        scheduler=scheduler,
    )

    # U-Net
    unet = DiffusionUNet(
        time_dim=64,
        context_dim=128,
        num_heads=4,
    ).to(device)

    # Optimizer
    optimizer = torch.optim.Adam(
        list(text_encoder.parameters())
        + list(text_conditioning.parameters())
        + list(unet.parameters()),
        lr=1e-4,
    )

    return (
        tokenizer,
        preprocessor,
        text_encoder,
        text_conditioning,
        timestep_embedding,
        unet,
        diffusion_process,
        optimizer,
    )

def train_one_epoch(
    dataloader,
    preprocessor,
    tokenizer,
    text_encoder,
    text_conditioning,
    timestep_embedding,
    unet,
    diffusion_process,
    optimizer,
    device,
    max_length=18,
    max_batches=None,
):
    text_encoder.train()
    text_conditioning.train()
    unet.train()

    total_loss = 0.0
    num_batches = 0

    for batch_idx, samples in enumerate(dataloader):

        if max_batches is not None and batch_idx >= max_batches:
            break

        result = training_step(
            samples=samples,
            preprocessor=preprocessor,
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            text_conditioning=text_conditioning,
            timestep_embedding=timestep_embedding,
            unet=unet,
            diffusion_process=diffusion_process,
            optimizer=optimizer,
            device=device,
            max_length=max_length,
        )

        loss = result["loss"].item()

        total_loss += loss
        num_batches += 1

        if (batch_idx + 1) % 100 == 0 or batch_idx == 0:
            print(
                f"Batch {batch_idx + 1}/{len(dataloader)} | "
                f"Loss: {loss:.4f}"
            )

    if num_batches == 0:
        raise ValueError("No batches were processed.")

    average_loss = total_loss / num_batches

    return average_loss

def validate(
    dataloader,
    preprocessor,
    tokenizer,
    text_encoder,
    text_conditioning,
    timestep_embedding,
    unet,
    diffusion_process,
    device,
    max_length=18,
    max_batches = 3,
):
    text_encoder.eval()
    text_conditioning.eval()
    unet.eval()

    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch_idx, samples in enumerate(dataloader):

            if max_batches is not None and batch_idx >= max_batches:
                break
            result = training_step(
                samples=samples,
                preprocessor=preprocessor,
                tokenizer=tokenizer,
                text_encoder=text_encoder,
                text_conditioning=text_conditioning,
                timestep_embedding=timestep_embedding,
                unet=unet,
                diffusion_process=diffusion_process,
                optimizer=None,
                device=device,
                max_length=max_length,
            )

            loss = result["loss"].item()

            total_loss += loss
            num_batches += 1

    if num_batches == 0:
        raise ValueError("No validation batches were processed.")

    return total_loss / num_batches

def train(
    train_dataloader,
    val_dataloader,
    preprocessor,
    tokenizer,
    text_encoder,
    text_conditioning,
    timestep_embedding,
    unet,
    diffusion_process,
    optimizer,
    device,
    num_epochs,
    max_length=18,
    max_batches=None,
):
    for epoch in range(num_epochs):

        train_loss = train_one_epoch(
            dataloader=train_dataloader,
            preprocessor=preprocessor,
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            text_conditioning=text_conditioning,
            timestep_embedding=timestep_embedding,
            unet=unet,
            diffusion_process=diffusion_process,
            optimizer=optimizer,
            device=device,
            max_length=max_length,
            max_batches=max_batches,
        )

        validation_loss = validate(
            dataloader=val_dataloader,
            preprocessor=preprocessor,
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            text_conditioning=text_conditioning,
            timestep_embedding=timestep_embedding,
            unet=unet,
            diffusion_process=diffusion_process,
            device=device,
            max_length=max_length,
            max_batches=max_batches,
        )

        print(
            f"\nEpoch {epoch + 1}/{num_epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Validation Loss: {validation_loss:.4f}"
        )

        torch.save(
            {
                "epoch": epoch + 1,
                "text_encoder": text_encoder.state_dict(),
                "text_conditioning": text_conditioning.state_dict(),
                "unet": unet.state_dict(),
                "optimizer": optimizer.state_dict(),
                "train_loss": train_loss,
                "validation_loss": validation_loss,
            },
            f"checkpoint_epoch_{epoch + 1}.pt",
        )




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

    (
        train_subset,
        val_subset,
        test_subset,
        ood_subset,
    ) = create_torch_subsets(
        dataset=dataset,
        train_indices=train_indices,
        val_indices=val_indices,
        test_indices=test_indices,
        ood_indices=ood_indices,
    )

    print("\n===== DATASET SPLIT =====")
    print(f"Train: {len(train_indices)}")
    print(f"Validation: {len(val_indices)}")
    print(f"Test: {len(test_indices)}")
    print(f"OOD: {len(ood_indices)}")

    # DataLoader
    train_dataloader = create_dataloader(
        dataset=train_subset,
        batch_size=2,
        shuffle=True,
        num_workers=0,
    )

    val_dataloader = create_dataloader(
        dataset=val_subset,
        batch_size=2,
        shuffle=False,
        num_workers=0,
    )

    # Model + training components
    (
        tokenizer,
        preprocessor,
        text_encoder,
        text_conditioning,
        timestep_embedding,
        unet,
        diffusion_process,
        optimizer,
    ) = create_training_components(
        dataset=dataset,
        train_indices=train_indices,
        device=device,
    )
    print("\n===== FULL EXPLORATORY TRAINING =====")

    train(
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        preprocessor=preprocessor,
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        text_conditioning=text_conditioning,
        timestep_embedding=timestep_embedding,
        unet=unet,
        diffusion_process=diffusion_process,
        optimizer=optimizer,
        device=device,
        num_epochs=50,
        max_length=18,
        max_batches=None,
    )

    print("Full exploratory training: OK")



if __name__ == "__main__":
    main()