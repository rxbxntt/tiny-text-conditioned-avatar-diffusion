import torch
import torch.nn.functional as F

from text.caption_generator import generate_caption


def training_step(
    samples,
    preprocessor,
    tokenizer,
    text_encoder,
    text_conditioning,
    timestep_embedding,
    unet,
    diffusion_process,
    optimizer,
    device,
    max_length=128,
):
    """
    Executes one minimal diffusion training step.

    Pipeline:

        samples
            ↓
        image preprocessing
            ↓
        caption generation
            ↓
        tokenization
            ↓
        text encoder
            ↓
        text conditioning projection
            ↓
        timestep sampling
            ↓
        forward diffusion
            ↓
        timestep embedding
            ↓
        U-Net
            ↓
        epsilon prediction
            ↓
        MSE loss
            ↓
        backward
            ↓
        optimizer step

    Args:
        samples:
            List of dataset samples.

            Each sample is expected to contain:
                sample["image"]
                sample["metadata"]

        preprocessor:
            ImagePreprocessor instance.

        tokenizer:
            Fitted CaptionTokenizer instance.

        text_encoder:
            TextEncoder instance.

        text_conditioning:
            TextConditioningProjection instance.

        timestep_embedding:
            TimestepEmbedding instance.

        unet:
            DiffusionUNet instance.

        diffusion_process:
            DiffusionForwardProcess instance.

        optimizer:
            PyTorch optimizer containing all trainable model parameters.

        device:
            torch.device used for training.

        max_length:
            Maximum caption sequence length.

    Returns:
        Dictionary containing the training loss and useful tensors
        for the sanity checks.
    """

    # ---------------------------------------------------------
    # 1. Basic validation
    # ---------------------------------------------------------

    if len(samples) == 0:
        raise ValueError("samples must contain at least one sample.")

    # ---------------------------------------------------------
    # 2. Prepare images
    # ---------------------------------------------------------

    images = []

    for sample in samples:
        image = preprocessor(sample["image"])
        images.append(image)

    x0 = torch.stack(images, dim=0).to(device)

    # x0:
    # [B, 3, 32, 32]

    # ---------------------------------------------------------
    # 3. Generate captions
    # ---------------------------------------------------------

    captions = [
        generate_caption(sample["metadata"])
        for sample in samples
    ]

    # ---------------------------------------------------------
    # 4. Tokenize captions
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

    # input_ids:
    # [B, L]
    #
    # attention_mask:
    # [B, L]

    # ---------------------------------------------------------
    # 5. Text encoder
    # ---------------------------------------------------------

    text_features = text_encoder(
        input_ids=input_ids,
        attention_mask=attention_mask,
    )

    # text_features:
    # [B, L, 64]

    # ---------------------------------------------------------
    # 6. Project text features for U-Net
    # ---------------------------------------------------------

    context = text_conditioning(text_features)

    # context:
    # [B, L, 128]

    # ---------------------------------------------------------
    # 7. Sample diffusion timesteps
    # ---------------------------------------------------------

    batch_size = x0.size(0)

    timesteps = diffusion_process.scheduler.sample_timesteps(
        batch_size=batch_size,
        device=device,
    )

    # timesteps:
    # [B]

    # ---------------------------------------------------------
    # 8. Forward diffusion
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
    # 9. Timestep embedding
    # ---------------------------------------------------------

    time_embedding = timestep_embedding(timesteps)

    # time_embedding:
    # [B, 64]

    # ---------------------------------------------------------
    # 10. U-Net forward
    # ---------------------------------------------------------

    predicted_noise = unet(
        x=xt,
        time_embedding=time_embedding,
        context=context,
    )

    # predicted_noise:
    # [B, 3, 32, 32]

    # ---------------------------------------------------------
    # 11. Epsilon prediction loss
    # ---------------------------------------------------------

    loss = F.mse_loss(
        predicted_noise,
        noise,
    )

    if optimizer is not None:
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # ---------------------------------------------------------
    # 14. Return useful information
    # ---------------------------------------------------------

    return {
        "loss": loss.detach(),
        "x0": x0.detach(),
        "xt": xt.detach(),
        "noise": noise.detach(),
        "predicted_noise": predicted_noise.detach(),
        "timesteps": timesteps.detach(),
        "input_ids": input_ids.detach(),
        "attention_mask": attention_mask.detach(),
        "text_features": text_features.detach(),
        "context": context.detach(),
        "time_embedding": time_embedding.detach(),
    }