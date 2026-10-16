import os

import torch

from sample_checkpoint import load_components
from diffusion.sampler import sample


NUM_SAMPLES = 1
SEED = 42

GLASSES_WITHOUT = "glasses_11"
GLASSES_WITH = "glasses_1"

OUTPUT_DIR = "generated_glasses_test"


def change_glasses(caption, new_value):
    tokens = caption.split()

    for i, token in enumerate(tokens):
        if token.startswith("glasses_"):
            tokens[i] = new_value
            break
    else:
        raise ValueError(
            "Token 'glasses_' non trovato nella caption."
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

    caption_without = change_glasses(
        base_caption,
        GLASSES_WITHOUT,
    )

    caption_with = change_glasses(
        base_caption,
        GLASSES_WITH,
    )

    print("\n===== CONDITIONING TEST =====")

    print("\nWITHOUT GLASSES:")
    print(caption_without)

    print("\nWITH GLASSES:")
    print(caption_with)

    # --------------------------------------------------
    # Same seed = same initial noise
    # --------------------------------------------------

    with torch.no_grad():

        print("\n===== WITHOUT GLASSES =====")

        torch.manual_seed(SEED)

        image_without = sample(
            captions=[caption_without],
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            text_conditioning=text_conditioning,
            timestep_embedding=timestep_embedding,
            unet=unet,
            scheduler=scheduler,
            device=device,
            max_length=18,
        )

        print("\n===== WITH GLASSES =====")

        torch.manual_seed(SEED)

        image_with = sample(
            captions=[caption_with],
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

    path_without = os.path.join(
        OUTPUT_DIR,
        "without_glasses.png",
    )

    path_with = os.path.join(
        OUTPUT_DIR,
        "with_glasses.png",
    )

    from sample_checkpoint import save_tensor_as_image

    save_tensor_as_image(
        image_without[0],
        path_without,
    )

    save_tensor_as_image(
        image_with[0],
        path_with,
    )

    print("\n===== RESULTS =====")

    print(f"Without glasses: {path_without}")
    print(f"With glasses:    {path_with}")

    print("\nTest completed.")


if __name__ == "__main__":
    main()