import torch

from data.cartoon_dataset import CartoonSetDataset
from data.dataset_split import create_dataset_splits
from preprocessing.image_processor import ImagePreprocessor
from training.dataloader import create_dataloader

from diffusion.time_step_embedding import TimestepEmbedding
from diffusion.unet import DiffusionUNet
from diffusion.diffuser import (
    DiffusionScheduler,
    DiffusionForwardProcess,
)

from training.unconditional_trainer import (
    unconditional_training_step,
)


DATASET_ROOT = (
    r"C:\Users\utente\.cache\kagglehub\datasets"
    r"\paraglondhe\cartoonset100k\versions\2"
    r"\cartoonset100k\0"
)


def main():

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Device:", device)

    # ---------------------------------------------------------
    # Dataset
    # ---------------------------------------------------------

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

    # Usiamo solo pochi sample per lo smoke test
    train_indices = train_indices[:2]

    samples = [
        dataset[index]
        for index in train_indices
    ]

    print("Number of samples:", len(samples))

    # ---------------------------------------------------------
    # Preprocessing
    # ---------------------------------------------------------

    preprocessor = ImagePreprocessor(
        image_size=32,
        normalization="zero_one",
    )

    # ---------------------------------------------------------
    # Diffusion
    # ---------------------------------------------------------

    timestep_embedding = TimestepEmbedding(
        num_timesteps=1000,
        embedding_dim=64,
    ).to(device)

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

    # ---------------------------------------------------------
    # Training step
    # ---------------------------------------------------------

    result = unconditional_training_step(
        samples=samples,
        preprocessor=preprocessor,
        timestep_embedding=timestep_embedding,
        unet=unet,
        diffusion_process=diffusion_process,
        optimizer=optimizer,
        device=device,
    )

    loss = result["loss"]

    print("x0 shape:", result["x0"].shape)
    print("xt shape:", result["xt"].shape)
    print("noise shape:", result["noise"].shape)
    print("predicted noise shape:", result["predicted_noise"].shape)
    print("timesteps shape:", result["timesteps"].shape)
    print("time embedding shape:", result["time_embedding"].shape)
    print("loss:", loss.item())

    # ---------------------------------------------------------
    # Sanity checks
    # ---------------------------------------------------------

    assert result["x0"].shape == (
        2, 3, 32, 32
    )

    assert result["xt"].shape == (
        2, 3, 32, 32
    )

    assert result["noise"].shape == (
        2, 3, 32, 32
    )

    assert result["predicted_noise"].shape == (
        2, 3, 32, 32
    )

    assert result["timesteps"].shape == (
        2,
    )

    assert result["time_embedding"].shape == (
        2, 64
    )

    assert torch.isfinite(loss)

    assert torch.isfinite(
        result["predicted_noise"]
    ).all()

    # Verifica che il backward abbia prodotto gradienti
    has_gradients = False

    for name, parameter in unet.named_parameters():

        if parameter.grad is not None:

            has_gradients = True

            print(
                "Gradient found:",
                name,
            )

            break

    assert has_gradients

    print(
        "\nUNCONDITIONAL TRAINING STEP "
        "SANITY CHECK: PASSED"
    )


if __name__ == "__main__":
    main()