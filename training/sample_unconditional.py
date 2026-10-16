import torch

from diffusion.time_step_embedding import TimestepEmbedding
from diffusion.unet import DiffusionUNet
from diffusion.diffuser import DiffusionScheduler

from pathlib import Path
from torchvision.utils import save_image, make_grid




CHECKPOINT_PATH = "unconditional_checkpoint.pt"

IMAGE_SIZE = 32
NUM_CHANNELS = 3
NUM_TIMESTEPS = 1000
TIME_DIM = 64
CONTEXT_DIM = None

NUM_SAMPLES = 2


@torch.no_grad()
def sample_unconditional(
    unet,
    timestep_embedding,
    scheduler,
    num_samples,
    device,
):
    """
    Generates unconditional samples using DDPM reverse diffusion.
    """



    # Start from pure Gaussian noise x_T
    x = torch.randn(
        num_samples,
        NUM_CHANNELS,
        IMAGE_SIZE,
        IMAGE_SIZE,
        device=device,
    )

    # Reverse diffusion: T-1 -> 0
    for timestep in reversed(
        range(NUM_TIMESTEPS)
    ):

        timesteps = torch.full(
            (num_samples,),
            timestep,
            dtype=torch.long,
            device=device,
        )

        time_embedding = timestep_embedding(
            timesteps
        )

        predicted_noise = unet(
            x=x,
            time_embedding=time_embedding,
            context=None,
        )

        x = scheduler.reverse_step(
            x_t=x,
            predicted_noise=predicted_noise,
            timesteps=timesteps,
        )

    # Convert to valid image range
    x = x.clamp(0.0, 1.0)

    return x


def main():

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(f"Device: {device}")

    # ---------------------------------------------------------
    # Create model components
    # ---------------------------------------------------------

    timestep_embedding = TimestepEmbedding(
        num_timesteps=NUM_TIMESTEPS,
        embedding_dim=TIME_DIM,
    ).to(device)

    unet = DiffusionUNet(
        time_dim=TIME_DIM,
        context_dim=CONTEXT_DIM,
        num_heads=4,
    ).to(device)

    scheduler = DiffusionScheduler(
        num_timesteps=NUM_TIMESTEPS,
        schedule_type="linear",
    )

    # ---------------------------------------------------------
    # Load unconditional checkpoint
    # ---------------------------------------------------------

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device,
    )

    unet.load_state_dict(
        checkpoint["unet"]
    )

    print(
        f"Loaded checkpoint from epoch "
        f"{checkpoint['epoch']}"
    )

    print(
        f"Checkpoint validation loss: "
        f"{checkpoint['validation_loss']:.4f}"
    )

    unet.eval()
    timestep_embedding.eval()

    # ---------------------------------------------------------
    # Sampling
    # ---------------------------------------------------------

    print("\n===== UNCONDITIONAL SAMPLING =====")

    samples = sample_unconditional(
        unet=unet,
        timestep_embedding=timestep_embedding,
        scheduler=scheduler,
        num_samples=NUM_SAMPLES,
        device=device,
    )

    # ---------------------------------------------------------
    # Save generated images
    # ---------------------------------------------------------

    output_dir = Path("outputs/unconditional")
    output_dir.mkdir(parents=True, exist_ok=True)

    for i, sample in enumerate(samples):
        save_image(
            sample,
            output_dir / f"sample_{i:02d}.png",
        )

    grid = make_grid(
        samples,
        nrow=NUM_SAMPLES,
    )

    save_image(
        grid,
        output_dir / "unconditional_samples.png",
    )

    print(
        f"\nImages saved to: {output_dir.resolve()}"
    )

    # ---------------------------------------------------------
    # Sanity checks
    # ---------------------------------------------------------

    print(
        f"Sample shape: {samples.shape}"
    )

    print(
        f"Min value: {samples.min().item():.4f}"
    )

    print(
        f"Max value: {samples.max().item():.4f}"
    )

    print(
        f"Mean value: {samples.mean().item():.4f}"
    )

    print(
        f"Finite: {torch.isfinite(samples).all().item()}"
    )

    assert samples.shape == (
        NUM_SAMPLES,
        NUM_CHANNELS,
        IMAGE_SIZE,
        IMAGE_SIZE,
    )

    assert torch.isfinite(samples).all()

    assert samples.min() >= 0.0
    assert samples.max() <= 1.0

    print(
        "\nUNCONDITIONAL SAMPLING "
        "SANITY CHECK: PASSED"
    )


if __name__ == "__main__":
    main()