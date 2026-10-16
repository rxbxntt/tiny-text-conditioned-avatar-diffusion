import torch

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
):
    """
    Generate images from text captions using reverse DDPM sampling.

    Returns:
        Tensor with shape [B, 3, 32, 32].
    """

    if not captions:
        raise ValueError("captions must contain at least one caption.")

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
            truncation=True,
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

    context = text_conditioning(text_features)

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

    for timestep in reversed(range(scheduler.num_timesteps)):

        if timestep % 100 == 0 or timestep == scheduler.num_timesteps - 1:
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

        time_embedding = timestep_embedding(timesteps)

        predicted_noise = unet(
            x_t,
            time_embedding,
            context,
        )

        x_t = scheduler.reverse_step(
            x_t=x_t,
            predicted_noise=predicted_noise,
            timesteps=timesteps,
        )

    return x_t