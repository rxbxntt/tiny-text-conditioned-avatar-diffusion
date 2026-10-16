import torch

from data.cartoon_dataset import CartoonSetDataset
from data.dataset_split import (
    create_dataset_splits,
    create_torch_subsets,
)

from preprocessing.image_processor import ImagePreprocessor

from training.dataloader import create_dataloader
from training.unconditional_trainer import (
    unconditional_training_step,
)

from diffusion.time_step_embedding import TimestepEmbedding
from diffusion.unet import DiffusionUNet
from diffusion.diffuser import (
    DiffusionScheduler,
    DiffusionForwardProcess,
)


DATASET_ROOT = (
    r"C:\Users\utente\.cache\kagglehub\datasets"
    r"\paraglondhe\cartoonset100k\versions\2"
    r"\cartoonset100k\0"
)

CHECKPOINT_PATH = "unconditional_checkpoint.pt"


def create_components(device):
    """
    Creates all components required by the
    unconditional diffusion model.
    """

    # ---------------------------------------------------------
    # Image preprocessing
    # ---------------------------------------------------------

    preprocessor = ImagePreprocessor(
        image_size=32,
        normalization="zero_one",
    )

    # ---------------------------------------------------------
    # Timestep embedding
    # ---------------------------------------------------------

    timestep_embedding = TimestepEmbedding(
        num_timesteps=1000,
        embedding_dim=64,
    ).to(device)

    # ---------------------------------------------------------
    # Diffusion
    # ---------------------------------------------------------

    scheduler = DiffusionScheduler(
        num_timesteps=1000,
        schedule_type="linear",
    )

    diffusion_process = DiffusionForwardProcess(
        scheduler=scheduler,
    )

    # ---------------------------------------------------------
    # Unconditional U-Net
    # ---------------------------------------------------------

    unet = DiffusionUNet(
        time_dim=64,
        context_dim=None,
        num_heads=4,
    ).to(device)

    # ---------------------------------------------------------
    # Optimizer
    # ---------------------------------------------------------

    optimizer = torch.optim.Adam(
        unet.parameters(),
        lr=1e-4,
    )

    return (
        preprocessor,
        timestep_embedding,
        unet,
        diffusion_process,
        optimizer,
    )


def train_one_epoch(
    dataloader,
    preprocessor,
    timestep_embedding,
    unet,
    diffusion_process,
    optimizer,
    device,
    max_batches=None,
):
    """
    Trains the unconditional model for one epoch.
    """

    unet.train()

    total_loss = 0.0
    num_batches = 0

    for batch_idx, samples in enumerate(dataloader):

        if (
            max_batches is not None
            and batch_idx >= max_batches
        ):
            break

        result = unconditional_training_step(
            samples=samples,
            preprocessor=preprocessor,
            timestep_embedding=timestep_embedding,
            unet=unet,
            diffusion_process=diffusion_process,
            optimizer=optimizer,
            device=device,
        )

        loss = result["loss"].item()

        total_loss += loss
        num_batches += 1

        if (
            batch_idx == 0
            or (batch_idx + 1) % 100 == 0
        ):
            print(
                f"Batch {batch_idx + 1}/{len(dataloader)} "
                f"| Loss: {loss:.4f}"
            )

    if num_batches == 0:
        raise ValueError(
            "No batches were processed."
        )

    return total_loss / num_batches


@torch.no_grad()
def validate(
    dataloader,
    preprocessor,
    timestep_embedding,
    unet,
    diffusion_process,
    device,
    max_batches=3,
):
    """
    Computes the unconditional validation loss.
    """

    unet.eval()

    total_loss = 0.0
    num_batches = 0

    for batch_idx, samples in enumerate(dataloader):

        if (
            max_batches is not None
            and batch_idx >= max_batches
        ):
            break

        result = unconditional_training_step(
            samples=samples,
            preprocessor=preprocessor,
            timestep_embedding=timestep_embedding,
            unet=unet,
            diffusion_process=diffusion_process,
            optimizer=None,
            device=device,
        )

        loss = result["loss"].item()

        total_loss += loss
        num_batches += 1

    if num_batches == 0:
        raise ValueError(
            "No validation batches were processed."
        )

    return total_loss / num_batches


def main():

    # ---------------------------------------------------------
    # Device
    # ---------------------------------------------------------

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(f"Device: {device}")

    # ---------------------------------------------------------
    # Dataset
    # ---------------------------------------------------------

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

    print("\n===== DATASET SPLIT =====")
    print(f"Train: {len(train_indices)}")
    print(f"Validation: {len(val_indices)}")
    print(f"Test: {len(test_indices)}")
    print(f"OOD: {len(ood_indices)}")

    (
        train_subset,
        val_subset,
        _,
        _,
    ) = create_torch_subsets(
        dataset=dataset,
        train_indices=train_indices,
        val_indices=val_indices,
        test_indices=test_indices,
        ood_indices=ood_indices,
    )

    # ---------------------------------------------------------
    # DataLoader
    # ---------------------------------------------------------

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

    # ---------------------------------------------------------
    # Components
    # ---------------------------------------------------------

    (
        preprocessor,
        timestep_embedding,
        unet,
        diffusion_process,
        optimizer,
    ) = create_components(device)

    # ---------------------------------------------------------
    # Full unconditional baseline run
    # ---------------------------------------------------------

    NUM_EPOCHS = 50
    MAX_TRAIN_BATCHES = None
    MAX_VAL_BATCHES = None

    print("\n===== UNCONDITIONAL FULL CPU RUN =====")

    for epoch in range(NUM_EPOCHS):
        train_loss = train_one_epoch(
            dataloader=train_dataloader,
            preprocessor=preprocessor,
            timestep_embedding=timestep_embedding,
            unet=unet,
            diffusion_process=diffusion_process,
            optimizer=optimizer,
            device=device,
            max_batches=MAX_TRAIN_BATCHES,
        )

        validation_loss = validate(
            dataloader=val_dataloader,
            preprocessor=preprocessor,
            timestep_embedding=timestep_embedding,
            unet=unet,
            diffusion_process=diffusion_process,
            device=device,
            max_batches=MAX_VAL_BATCHES,
        )

        print(
            f"\nEpoch {epoch + 1}/{NUM_EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Validation Loss: {validation_loss:.4f}"
        )

        # Save checkpoint after every epoch
        torch.save(
            {
                "epoch": epoch + 1,
                "unet": unet.state_dict(),
                "optimizer": optimizer.state_dict(),
                "train_loss": train_loss,
                "validation_loss": validation_loss,
            },
            CHECKPOINT_PATH,
        )

        print(
            f"Checkpoint saved after epoch {epoch + 1}"
        )

    # ---------------------------------------------------------
    # Save mini-run checkpoint
    # ---------------------------------------------------------

    torch.save(
        {
            "epoch": NUM_EPOCHS,
            "unet": unet.state_dict(),
            "optimizer": optimizer.state_dict(),
            "train_loss": train_loss,
            "validation_loss": validation_loss,
        },
        CHECKPOINT_PATH,
    )

    print(
        f"\nCheckpoint saved: "
        f"{CHECKPOINT_PATH}"
    )

    print(
        "\nUNCONDITIONAL MINI RUN: PASSED"
    )


if __name__ == "__main__":
    main()