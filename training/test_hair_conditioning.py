import os

import torch

from sample_checkpoint import load_components
from diffusion.sampler import sample


SEED = 42

HAIR_COLOR_A = "hair_color_9"
HAIR_COLOR_B = "hair_color_2"

OUTPUT_DIR = "generated_hair_test"


def change_hair_color(caption, new_value):
    tokens = caption.split()

    for i, token in enumerate(tokens):
        if token.startswith("hair_color_"):
            tokens[i] = new_value
            break
    else:
        raise ValueError(
            "Token 'hair_color_' non trovato nella caption."
        )

    return " ".join(tokens)


def main():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Device: {device}")

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True,
    )

    (
        dataset,
        val_indices,
        tokenizer,
        text_encoder,
        text_conditioning,
        timestep_embedding,
        unet,
        scheduler,
    ) = load_components(device)

    text_encoder.eval()
    text_conditioning.eval()
    unet.eval()

    # --------------------------------------------------
    # Base caption
    # --------------------------------------------------

    index = val_indices[0]

    dataset_sample = dataset.samples[index]

    metadata = dataset._load_metadata(
        dataset_sample["metadata"]
    )

    base_caption = (
        " ".join(
            [
                f"eye_angle_{metadata['eye_angle']}",
                f"eye_lashes_{metadata['eye_lashes']}",
                f"eye_lid_{metadata['eye_lid']}",
                f"chin_length_{metadata['chin_length']}",
                f"eyebrow_weight_{metadata['eyebrow_weight']}",
                f"eyebrow_shape_{metadata['eyebrow_shape']}",
                f"eyebrow_thickness_{metadata['eyebrow_thickness']}",
                f"face_shape_{metadata['face_shape']}",
                f"facial_hair_{metadata['facial_hair']}",
                f"hair_{metadata['hair']}",
                f"eye_color_{metadata['eye_color']}",
                f"face_color_{metadata['face_color']}",
                f"hair_color_{metadata['hair_color']}",
                f"glasses_{metadata['glasses']}",
                f"glasses_color_{metadata['glasses_color']}",
                f"eye_slant_{metadata['eye_slant']}",
                f"eyebrow_width_{metadata['eyebrow_width']}",
                f"eye_eyebrow_distance_{metadata['eye_eyebrow_distance']}",
            ]
        )
    )

    # --------------------------------------------------
    # Change only hair
    # --------------------------------------------------

    caption_a = change_hair_color(
        base_caption,
        HAIR_COLOR_A,
    )

    caption_b = change_hair_color(
        base_caption,
        HAIR_COLOR_B,
    )

    print("\n===== CONDITIONING TEST =====")

    print("\nHAIR A:")
    print(caption_a)

    print("\nHAIR B:")
    print(caption_b)

    # --------------------------------------------------
    # Same seed = same initial noise
    # --------------------------------------------------

    with torch.no_grad():

        print("\n===== HAIR A =====")

        torch.manual_seed(SEED)

        image_a = sample(
            captions=[caption_a],
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            text_conditioning=text_conditioning,
            timestep_embedding=timestep_embedding,
            unet=unet,
            scheduler=scheduler,
            device=device,
            max_length=18,
        )

        print("\n===== HAIR B =====")

        torch.manual_seed(SEED)

        image_b = sample(
            captions=[caption_b],
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            text_conditioning=text_conditioning,
            timestep_embedding=timestep_embedding,
            unet=unet,
            scheduler=scheduler,
            device=device,
            max_length=18,
        )

    # --------------------------------------------------
    # Save
    # --------------------------------------------------

    path_a = os.path.join(
        OUTPUT_DIR,
        "hair_a.png",
    )

    path_b = os.path.join(
        OUTPUT_DIR,
        "hair_b.png",
    )

    from sample_checkpoint import save_tensor_as_image

    save_tensor_as_image(
        image_a[0],
        path_a,
    )

    save_tensor_as_image(
        image_b[0],
        path_b,
    )

    print("\n===== RESULTS =====")

    print(f"Hair A: {path_a}")
    print(f"Hair B: {path_b}")

    print("\nTest completed.")


if __name__ == "__main__":
    main()