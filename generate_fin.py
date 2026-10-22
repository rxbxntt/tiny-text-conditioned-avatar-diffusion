from pathlib import Path

import torch
from torchvision.utils import save_image

from text.tokenizer import CaptionTokenizer
from text.text_encoder import TextEncoder
from text.text_conditioning import TextConditioningProjection

from diffusion.time_step_embedding import TimestepEmbedding
from diffusion.unet import DiffusionUNet
from diffusion.diffuser import DiffusionScheduler

from prompt.prompt_to_caption import prompt_to_caption


PROJECT_ROOT = Path(__file__).resolve().parent

CHECKPOINT_PATH = PROJECT_ROOT / "final_cfg_checkpoint.pt"
VOCAB_PATH = PROJECT_ROOT / "vocab.json"
GENERATED_DIR = PROJECT_ROOT / "generated"

MAX_LENGTH = 18

TEXT_HIDDEN_SIZE = 64
TEXT_NUM_LAYERS = 2
TEXT_NUM_HEADS = 4

CONDITIONING_INPUT_DIM = 64
CONDITIONING_OUTPUT_DIM = 128

TIME_EMBEDDING_DIM = 64
NUM_TIMESTEPS = 1000

UNET_NUM_HEADS = 4
GUIDANCE_SCALE = 2.0


def set_seed(seed):
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_components(device):
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(f"Checkpoint not found:\n{CHECKPOINT_PATH}")

    if not VOCAB_PATH.exists():
        raise FileNotFoundError(
            f"Vocabulary not found:\n{VOCAB_PATH}\n\n"
            "Run build_vocab.py once before using generate_v2.py."
        )

    tokenizer = CaptionTokenizer.load(VOCAB_PATH)

    text_encoder = TextEncoder(
        vocab_size=tokenizer.vocab_size,
        hidden_size=TEXT_HIDDEN_SIZE,
        num_layers=TEXT_NUM_LAYERS,
        num_heads=TEXT_NUM_HEADS,
        max_length=128,
        dropout=0.0,
    ).to(device)

    conditioning_projection = TextConditioningProjection(
        input_dim=CONDITIONING_INPUT_DIM,
        output_dim=CONDITIONING_OUTPUT_DIM,
    ).to(device)

    timestep_embedding = TimestepEmbedding(
        num_timesteps=NUM_TIMESTEPS,
        embedding_dim=TIME_EMBEDDING_DIM,
    ).to(device)

    unet = DiffusionUNet(
        time_dim=TIME_EMBEDDING_DIM,
        context_dim=CONDITIONING_OUTPUT_DIM,
        num_heads=UNET_NUM_HEADS,
    ).to(device)

    scheduler = DiffusionScheduler(
        num_timesteps=NUM_TIMESTEPS,
        schedule_type="linear",
    ).to(device)

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device,
    )
    model_state = checkpoint["model_state_dict"]

    text_encoder.load_state_dict(model_state["text_encoder"])
    conditioning_projection.load_state_dict(
        model_state["conditioning_projection"]
    )
    timestep_embedding.load_state_dict(model_state["timestep_embedding"])
    unet.load_state_dict(model_state["unet"])

    scheduler.betas = scheduler.betas.to(device)
    scheduler.alphas = scheduler.alphas.to(device)
    scheduler.alpha_bars = scheduler.alpha_bars.to(device)

    text_encoder.eval()
    conditioning_projection.eval()
    timestep_embedding.eval()
    unet.eval()

    print(f"Vocabulary size: {tokenizer.vocab_size}")
    print(f"Checkpoint epoch: {checkpoint['epoch']}")
    print(f"Validation loss: {checkpoint['val_loss']:.6f}")

    return (
        tokenizer,
        text_encoder,
        conditioning_projection,
        timestep_embedding,
        unet,
        scheduler,
    )


@torch.inference_mode()
def sample_cfg(
    caption,
    tokenizer,
    text_encoder,
    conditioning_projection,
    timestep_embedding,
    unet,
    scheduler,
    device,
    guidance_scale=GUIDANCE_SCALE,
    seed=42,
):
    set_seed(seed)

    token_ids = tokenizer.encode(
        caption,
        max_length=MAX_LENGTH,
        padding=True,
        truncation=False,
    )
    attention_mask = tokenizer.attention_mask(token_ids)

    input_ids = torch.tensor(
        [token_ids],
        dtype=torch.long,
        device=device,
    )
    attention_mask = torch.tensor(
        [attention_mask],
        dtype=torch.long,
        device=device,
    )

    text_features = text_encoder(
        input_ids=input_ids,
        attention_mask=attention_mask,
    )
    context = conditioning_projection(text_features)
    unconditional_context = torch.zeros_like(context)

    x_t = torch.randn(
        1,
        3,
        32,
        32,
        device=device,
    )

    for timestep in reversed(range(scheduler.num_timesteps)):
        if timestep % 100 == 0 or timestep == scheduler.num_timesteps - 1:
            print(
                f"Reverse diffusion: "
                f"{timestep}/{scheduler.num_timesteps - 1}",
                flush=True,
            )

        timesteps = torch.full(
            (1,),
            timestep,
            dtype=torch.long,
            device=device,
        )

        time_embedding = timestep_embedding(timesteps)

        predicted_noise_cond = unet(
            x_t,
            time_embedding,
            context,
        )
        predicted_noise_uncond = unet(
            x_t,
            time_embedding,
            unconditional_context,
        )

        predicted_noise = (
            predicted_noise_uncond
            + guidance_scale
            * (predicted_noise_cond - predicted_noise_uncond)
        )

        x_t = scheduler.reverse_step(
            x_t=x_t,
            predicted_noise=predicted_noise,
            timesteps=timesteps,
        )

    return x_t.clamp(0.0, 1.0)


def generate_image(
    prompt,
    seed,
    device,
    tokenizer,
    text_encoder,
    conditioning_projection,
    timestep_embedding,
    unet,
    scheduler,
):
    if not prompt.strip():
        raise ValueError("Prompt cannot be empty.")

    caption = prompt_to_caption(prompt)

    print()
    print("User prompt:")
    print(prompt)

    print()
    print("Training-style caption:")
    print(caption)

    token_ids = tokenizer.encode(
        caption,
        max_length=MAX_LENGTH,
        padding=True,
        truncation=False,
    )

    print()
    print("Token IDs:")
    print(token_ids)

    print()
    print("Decoded:")
    print(tokenizer.decode(token_ids))

    return sample_cfg(
        caption=caption,
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        conditioning_projection=conditioning_projection,
        timestep_embedding=timestep_embedding,
        unet=unet,
        scheduler=scheduler,
        device=device,
        guidance_scale=GUIDANCE_SCALE,
        seed=seed,
    )


def get_next_output_path():
    GENERATED_DIR.mkdir(parents=True, exist_ok=True)

    counter = 1
    while True:
        output_path = GENERATED_DIR / f"generation_{counter:03d}.png"

        if not output_path.exists():
            return output_path

        counter += 1


def main():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("=" * 60)
    print("Tiny Text-Conditioned Avatar Diffusion")
    print("=" * 60)
    print(f"Device: {device}")

    (
        tokenizer,
        text_encoder,
        conditioning_projection,
        timestep_embedding,
        unet,
        scheduler,
    ) = load_components(device)

    prompt = input("\nPrompt: ").strip()
    seed = int(input("Seed: ").strip())

    print()
    print(f"CFG guidance scale: {GUIDANCE_SCALE}")
    print()
    print("Generating...")

    image = generate_image(
        prompt=prompt,
        seed=seed,
        device=device,
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        conditioning_projection=conditioning_projection,
        timestep_embedding=timestep_embedding,
        unet=unet,
        scheduler=scheduler,
    )

    print()
    print("Generation completed.")
    print(f"Shape: {tuple(image.shape)}")
    print(f"Min: {image.min().item():.4f}")
    print(f"Max: {image.max().item():.4f}")
    print(f"Mean: {image.mean().item():.4f}")
    print(f"Finite values: {torch.isfinite(image).all().item()}")

    output_path = get_next_output_path()
    save_image(image, output_path)

    print()
    print(f"Image saved to:\n{output_path}")


if __name__ == "__main__":
    main()
