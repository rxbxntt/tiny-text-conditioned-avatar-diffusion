import torch

from training.trainer import training_step
from PIL import Image

from data.cartoon_dataset import CartoonSetDataset
from preprocessing.image_processor import ImagePreprocessor
from text.tokenizer import CaptionTokenizer
from text.text_encoder import TextEncoder
from text.text_conditioning import TextConditioningProjection
from diffusion.time_step_embedding import TimestepEmbedding
from diffusion.unet import DiffusionUNet
from diffusion.diffuser import DiffusionScheduler, DiffusionForwardProcess
from dataloader import create_dataloader


def main():

    # =========================================================
    # 1. Device
    # =========================================================

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Device:", device)

    # =========================================================
    # TEST REAL DATASET
    # =========================================================

    dataset_root = (
        r"C:\Users\utente\.cache\kagglehub\datasets"
        r"\paraglondhe\cartoonset100k\versions\2"
        r"\cartoonset100k\0"
    )

    dataset = CartoonSetDataset(
        root_dir=dataset_root,
        expected_samples=10_000,
    )

    print("Dataset size:", len(dataset))

    assert len(dataset) == 10_000

    sample = dataset[0]

    print("Image type:", type(sample["image"]))
    print("Image size:", sample["image"].size)
    print("Image mode:", sample["image"].mode)

    print("Metadata attributes:", len(sample["metadata"]))

    assert isinstance(sample["image"], Image.Image)
    assert len(sample["metadata"]) == 18

    print("Dataset sample check: OK")

    # =========================================================
    # TEST DATALOADER
    # =========================================================

    dataloader = create_dataloader(
        dataset_root=dataset_root,
        batch_size=4,
        shuffle=True,
        num_workers=0,
    )

    batch = next(iter(dataloader))

    print()
    print("===== DATALOADER TEST =====")
    print("Batch size:", len(batch))

    assert len(batch) == 4

    for sample in batch:

        assert isinstance(sample["image"], Image.Image)
        assert isinstance(sample["metadata"], dict)
        assert len(sample["metadata"]) == 18

    print("Images and metadata: OK")
    print("DataLoader test: OK")

    # =========================================================
    # 2. Minimal fake dataset
    # =========================================================

    # We do not load the real dataset yet.
    #
    # Each sample must have:
    #   - image
    #   - metadata
    #
    # The metadata keys must match the caption generator.

    samples = [
        {
            "image": Image.new("RGB", (32, 32)),
            "metadata": {
                "eye_angle": 2,
                "eye_lashes": 1,
                "eye_lid": 1,
                "chin_length": 2,
                "eyebrow_weight": 1,
                "eyebrow_shape": 2,
                "eyebrow_thickness": 1,
                "face_shape": 3,
                "facial_hair": 1,
                "hair": 43,
                "eye_color": 2,
                "face_color": 1,
                "hair_color": 4,
                "glasses": 1,
                "glasses_color": 2,
                "eye_slant": 1,
                "eyebrow_width": 2,
                "eye_eyebrow_distance": 1,
            },
        },
        {
            "image": Image.new("RGB", (32, 32)),
            "metadata": {
                "eye_angle": 1,
                "eye_lashes": 2,
                "eye_lid": 1,
                "chin_length": 1,
                "eyebrow_weight": 2,
                "eyebrow_shape": 1,
                "eyebrow_thickness": 2,
                "face_shape": 2,
                "facial_hair": 1,
                "hair": 12,
                "eye_color": 3,
                "face_color": 2,
                "hair_color": 5,
                "glasses": 2,
                "glasses_color": 1,
                "eye_slant": 2,
                "eyebrow_width": 1,
                "eye_eyebrow_distance": 2,
            },
        },
    ]

    batch_size = len(samples)

    print("Batch size:", batch_size)

    # =========================================================
    # 3. Image preprocessor
    # =========================================================

    preprocessor = ImagePreprocessor(
        image_size=32,
        normalization="zero_one",
    )

    # NOTE:
    # "zero_one" is used here only for this smoke test.
    #
    # The final normalization strategy is still an open
    # experimental decision and is NOT being fixed here.

    # =========================================================
    # 4. Tokenizer
    # =========================================================

    tokenizer = CaptionTokenizer()

    # Generate the same type of captions used by training.py.
    from text.caption_generator import generate_caption

    captions = [
        generate_caption(sample["metadata"])
        for sample in samples
    ]

    # Fit only on the captions used by this smoke test.
    tokenizer.fit(captions)

    print("Vocabulary size:", tokenizer.vocab_size)

    # =========================================================
    # 5. Text encoder
    # =========================================================

    text_encoder = TextEncoder(
        vocab_size=tokenizer.vocab_size,
        hidden_size=64,
        num_layers=2,
        num_heads=4,
        max_length=128,
        dropout=0.0,
    ).to(device)

    # =========================================================
    # 6. Text conditioning
    # =========================================================

    text_conditioning = TextConditioningProjection(
        input_dim=64,
        output_dim=128,
    ).to(device)

    # =========================================================
    # 7. Timestep embedding
    # =========================================================

    num_timesteps = 1000

    timestep_embedding = TimestepEmbedding(
        num_timesteps=num_timesteps,
        embedding_dim=64,
    ).to(device)

    # =========================================================
    # 8. Diffusion scheduler
    # =========================================================

    scheduler = DiffusionScheduler(
        num_timesteps=num_timesteps,
        schedule_type="linear",
    )

    diffusion_process = DiffusionForwardProcess(
        scheduler=scheduler,
    )

    # =========================================================
    # 9. U-Net
    # =========================================================

    unet = DiffusionUNet(
        time_dim=64,
        context_dim=128,
        num_heads=4,
    ).to(device)

    # =========================================================
    # 10. Optimizer
    # =========================================================

    # Temporary optimizer for the smoke test.
    #
    # This is NOT a final training decision.

    optimizer = torch.optim.Adam(
        list(text_encoder.parameters())
        + list(text_conditioning.parameters())
        + list(unet.parameters()),
        lr=1e-4,
    )

    # =========================================================
    # 11. Execute one training step
    # =========================================================

    result = training_step(
        samples=samples,
        preprocessor=preprocessor,
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        text_conditioning=text_conditioning,
        timestep_embedding=timestep_embedding,
        unet=unet,
        diffusion_process=diffusion_process,
        optimizer=optimizer,
        device=device,
    )

    # =========================================================
    # 12. Basic checks
    # =========================================================

    loss = result["loss"]

    print()
    print("===== TRAINING STEP RESULT =====")
    print("Loss:", loss.item())

    assert loss.ndim == 0
    assert torch.isfinite(loss)

    print("Loss check: OK")

    # ---------------------------------------------------------
    # Image tensors
    # ---------------------------------------------------------

    assert result["x0"].shape == (
        batch_size,
        3,
        32,
        32,
    )

    assert result["xt"].shape == (
        batch_size,
        3,
        32,
        32,
    )

    assert result["noise"].shape == (
        batch_size,
        3,
        32,
        32,
    )

    assert result["predicted_noise"].shape == (
        batch_size,
        3,
        32,
        32,
    )

    print("Image tensor shapes: OK")

    # ---------------------------------------------------------
    # Timestep
    # ---------------------------------------------------------

    assert result["timesteps"].shape == (
        batch_size,
    )

    assert result["timesteps"].dtype == torch.long

    print("Timestep shape/type: OK")

    # ---------------------------------------------------------
    # Text tensors
    # ---------------------------------------------------------

    assert result["input_ids"].shape[0] == batch_size
    assert result["attention_mask"].shape == result["input_ids"].shape

    assert result["text_features"].shape == (
        batch_size,
        result["input_ids"].shape[1],
        64,
    )

    assert result["context"].shape == (
        batch_size,
        result["input_ids"].shape[1],
        128,
    )

    print("Text tensor shapes: OK")

    # ---------------------------------------------------------
    # Time embedding
    # ---------------------------------------------------------

    assert result["time_embedding"].shape == (
        batch_size,
        64,
    )

    print("Time embedding shape: OK")

    # =========================================================
    # 13. Check gradients
    # =========================================================

    models = [
        ("text_encoder", text_encoder),
        ("text_conditioning", text_conditioning),
        ("unet", unet),
    ]

    for model_name, model in models:

        has_gradient = False

        for parameter in model.parameters():

            if parameter.grad is not None:

                has_gradient = True

                assert torch.isfinite(
                    parameter.grad
                ).all()

        assert has_gradient, (
            f"No gradients found for {model_name}"
        )

        print(
            f"Gradients for {model_name}: OK"
        )

    # =========================================================
    # 14. Final message
    # =========================================================

    print()
    print("================================")
    print("TRAINING STEP SMOKE TEST PASSED")
    print("================================")


if __name__ == "__main__":
    main()