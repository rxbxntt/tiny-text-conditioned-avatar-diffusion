import torch
import torch.nn.functional as F


def unconditional_training_step(
    samples,
    preprocessor,
    timestep_embedding,
    unet,
    diffusion_process,
    optimizer,
    device,
):
    """
    Executes one unconditional diffusion training step.

    Pipeline:

        samples
            ↓
        image preprocessing
            ↓
        timestep sampling
            ↓
        forward diffusion
            ↓
        timestep embedding
            ↓
        U-Net without text context
            ↓
        epsilon prediction
            ↓
        MSE loss
            ↓
        backward
            ↓
        optimizer step
    """

    if len(samples) == 0:
        raise ValueError("samples must contain at least one sample.")

    # ---------------------------------------------------------
    # 1. Prepare images
    # ---------------------------------------------------------

    images = []

    for sample in samples:
        image = preprocessor(sample["image"])
        images.append(image)

    x0 = torch.stack(
        images,
        dim=0,
    ).to(device)

    # x0:
    # [B, 3, 32, 32]

    # ---------------------------------------------------------
    # 2. Sample diffusion timesteps
    # ---------------------------------------------------------

    batch_size = x0.size(0)

    timesteps = diffusion_process.scheduler.sample_timesteps(
        batch_size=batch_size,
        device=device,
    )

    # timesteps:
    # [B]

    # ---------------------------------------------------------
    # 3. Forward diffusion
    # ---------------------------------------------------------

    xt, noise = diffusion_process.add_noise(
        x0=x0,
        timesteps=timesteps,
    )

    # xt:
    # [B, 3, 32, 32]
    #
    # noise:
    # [B, 3, 32, 32]

    # ---------------------------------------------------------
    # 4. Timestep embedding
    # ---------------------------------------------------------

    time_embedding = timestep_embedding(
        timesteps
    )

    # time_embedding:
    # [B, 64]

    # ---------------------------------------------------------
    # 5. U-Net forward
    # ---------------------------------------------------------

    predicted_noise = unet(
        x=xt,
        time_embedding=time_embedding,
        context=None,
    )

    # predicted_noise:
    # [B, 3, 32, 32]

    # ---------------------------------------------------------
    # 6. Epsilon prediction loss
    # ---------------------------------------------------------

    loss = F.mse_loss(
        predicted_noise,
        noise,
    )

    # ---------------------------------------------------------
    # 7. Backward + optimizer step
    # ---------------------------------------------------------

    if optimizer is not None:
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return {
        "loss": loss.detach(),
        "x0": x0.detach(),
        "xt": xt.detach(),
        "noise": noise.detach(),
        "predicted_noise": predicted_noise.detach(),
        "timesteps": timesteps.detach(),
        "time_embedding": time_embedding.detach(),
    }