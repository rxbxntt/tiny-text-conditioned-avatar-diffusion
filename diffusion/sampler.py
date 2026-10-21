import torch


@torch.no_grad()
def sample(
    captions,
    tokenizer,
    text_encoder,
    text_conditioning,
    timestep_embedding,
    unet,
    scheduler,
    device,
    max_length=18,
    guidance_scale=2.0,
):
    """
    Generate images from text captions using reverse DDPM
    sampling with classifier-free guidance.

    Args:
        captions:
            List of structured training-style captions.

        tokenizer:
            Caption tokenizer fitted on the training split.

        text_encoder:
            Transformer text encoder.

        text_conditioning:
            Projection from text features to U-Net context.

        timestep_embedding:
            Timestep embedding module.

        unet:
            Conditional diffusion U-Net.

        scheduler:
            DDPM scheduler.

        device:
            Device used for sampling.

        max_length:
            Maximum caption length.

        guidance_scale:
            Classifier-free guidance scale.

    Returns:
        Tensor with shape [B, 3, 32, 32].
    """

    if not captions:
        raise ValueError(
            "captions must contain at least one caption."
        )

    batch_size = len(captions)

    # ---------------------------------------------------------
    # 1. Tokenize captions
    # ---------------------------------------------------------

    token_ids = []
    attention_masks = []

    for caption in captions:
        ids = tokenizer.encode(
            caption,
            max_length=max_length,
            padding=True,
            truncation=False,
        )

        mask = tokenizer.attention_mask(ids)

        token_ids.append(ids)
        attention_masks.append(mask)

    input_ids = torch.tensor(
        token_ids,
        dtype=torch.long,
        device=device,
    )

    attention_mask = torch.tensor(
        attention_masks,
        dtype=torch.long,
        device=device,
    )

    # ---------------------------------------------------------
    # 2. Encode text
    # ---------------------------------------------------------

    text_features = text_encoder(
        input_ids=input_ids,
        attention_mask=attention_mask,
    )

    context = text_conditioning(
        text_features
    )

    # Unconditional CFG conditioning.
    # The final model was trained using zero conditioning
    # for the unconditional branch.
    unconditional_context = torch.zeros_like(
        context
    )

    # ---------------------------------------------------------
    # 3. Start from Gaussian noise
    # ---------------------------------------------------------

    x_t = torch.randn(
        batch_size,
        3,
        32,
        32,
        device=device,
    )

    # ---------------------------------------------------------
    # 4. Reverse diffusion
    # ---------------------------------------------------------

    for timestep in reversed(
        range(scheduler.num_timesteps)
    ):

        if (
            timestep % 100 == 0
            or timestep == scheduler.num_timesteps - 1
        ):
            print(
                f"Reverse diffusion: timestep "
                f"{timestep}/{scheduler.num_timesteps - 1}",
                flush=True,
            )

        timesteps = torch.full(
            (batch_size,),
            timestep,
            dtype=torch.long,
            device=device,
        )

        time_embedding = timestep_embedding(
            timesteps
        )

        # Conditional prediction.
        predicted_noise_cond = unet(
            x_t,
            time_embedding,
            context,
        )

        # Unconditional prediction.
        predicted_noise_uncond = unet(
            x_t,
            time_embedding,
            unconditional_context,
        )

        # Classifier-Free Guidance.
        predicted_noise = (
            predicted_noise_uncond
            + guidance_scale
            * (
                predicted_noise_cond
                - predicted_noise_uncond
            )
        )

        # DDPM reverse step.
        x_t = scheduler.reverse_step(
            x_t=x_t,
            predicted_noise=predicted_noise,
            timesteps=timesteps,
        )

    return x_t.clamp(0.0, 1.0)

