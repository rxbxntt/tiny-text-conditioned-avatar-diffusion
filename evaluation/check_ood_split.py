import time
from collections import Counter
from pathlib import Path

from data.cartoon_dataset import CartoonSetDataset
from data.dataset_split import create_dataset_splits
from text.caption_generator import generate_caption
from text.tokenizer import CaptionTokenizer


# =========================================================
# Configuration
# =========================================================

DATASET_ROOT = (
    r"C:\Users\utente\.cache\kagglehub\datasets"
    r"\paraglondhe\cartoonset100k\versions\2"
    r"\cartoonset100k\0"
)

EXPECTED_SAMPLES = 10_000

OOD_HAIR_COLOR = 4
OOD_EYE_COLOR = 1

SPLIT_SEED = 42


# =========================================================
# Main
# =========================================================

def main():

    start_time = time.perf_counter()

    print("===== OOD SPLIT CHECK =====")

    # -----------------------------------------------------
    # 1. Load dataset
    # -----------------------------------------------------

    dataset = CartoonSetDataset(
        root_dir=DATASET_ROOT,
        expected_samples=EXPECTED_SAMPLES,
    )

    print(f"Dataset samples: {len(dataset.samples)}")

    # -----------------------------------------------------
    # 2. Recreate official split
    # -----------------------------------------------------

    (
        train_indices,
        val_indices,
        test_indices,
        ood_indices,
    ) = create_dataset_splits(
        dataset=dataset,
        ood_hair_color=OOD_HAIR_COLOR,
        ood_eye_color=OOD_EYE_COLOR,
        seed=SPLIT_SEED,
    )

    print("\n===== DATASET SPLIT =====")
    print(f"Train:      {len(train_indices)}")
    print(f"Validation: {len(val_indices)}")
    print(f"Test:       {len(test_indices)}")
    print(f"OOD:        {len(ood_indices)}")

    # -----------------------------------------------------
    # 3. Check expected split sizes
    # -----------------------------------------------------

    assert len(train_indices) == 8227
    assert len(val_indices) == 1028
    assert len(test_indices) == 515
    assert len(ood_indices) == 230

    print("Split sizes: OK")

    # -----------------------------------------------------
    # 4. Check OOD attributes
    # -----------------------------------------------------

    ood_hair_values = []
    ood_eye_values = []

    for index in ood_indices:

        sample = dataset.samples[index]
        metadata = dataset._load_metadata(sample["metadata"])

        ood_hair_values.append(metadata["hair_color"])
        ood_eye_values.append(metadata["eye_color"])

    hair_values = set(ood_hair_values)
    eye_values = set(ood_eye_values)

    print("\n===== OOD ATTRIBUTES =====")
    print(f"OOD hair_color values: {hair_values}")
    print(f"OOD eye_color values:  {eye_values}")

    assert hair_values == {OOD_HAIR_COLOR}
    assert eye_values == {OOD_EYE_COLOR}

    print("OOD attribute check: OK")

    # -----------------------------------------------------
    # 5. Verify individual attributes occur outside OOD
    # -----------------------------------------------------

    remaining_indices = (
        train_indices
        + val_indices
        + test_indices
    )

    remaining_hair_values = set()
    remaining_eye_values = set()

    for index in remaining_indices:

        sample = dataset.samples[index]
        metadata = dataset._load_metadata(sample["metadata"])

        remaining_hair_values.add(metadata["hair_color"])
        remaining_eye_values.add(metadata["eye_color"])

    print("\n===== COMPOSITIONAL CHECK =====")
    print(
        f"hair_color={OOD_HAIR_COLOR} "
        f"appears outside OOD: "
        f"{OOD_HAIR_COLOR in remaining_hair_values}"
    )

    print(
        f"eye_color={OOD_EYE_COLOR} "
        f"appears outside OOD: "
        f"{OOD_EYE_COLOR in remaining_eye_values}"
    )

    assert OOD_HAIR_COLOR in remaining_hair_values
    assert OOD_EYE_COLOR in remaining_eye_values

    print("Individual attribute check: OK")

    # -----------------------------------------------------
    # 6. Verify held-out combination is absent from training
    # -----------------------------------------------------

    train_combinations = set()

    for index in train_indices:

        sample = dataset.samples[index]
        metadata = dataset._load_metadata(sample["metadata"])

        combination = (
            metadata["hair_color"],
            metadata["eye_color"],
        )

        train_combinations.add(combination)

    held_out_combination = (
        OOD_HAIR_COLOR,
        OOD_EYE_COLOR,
    )

    print("\n===== HELD-OUT COMBINATION =====")
    print(f"Held-out combination: {held_out_combination}")
    print(
        "Present in training:",
        held_out_combination in train_combinations,
    )

    assert held_out_combination not in train_combinations

    print("Held-out combination check: OK")

    # -----------------------------------------------------
    # 7. Fit tokenizer ONLY on training captions
    # -----------------------------------------------------

    train_captions = []

    for index in train_indices:

        sample = dataset.samples[index]
        metadata = dataset._load_metadata(sample["metadata"])

        caption = generate_caption(metadata)
        train_captions.append(caption)

    tokenizer = CaptionTokenizer()
    tokenizer.fit(train_captions)

    print("\n===== TOKENIZER =====")
    print(f"Training captions: {len(train_captions)}")
    print(f"Vocabulary size: {tokenizer.vocab_size}")

    assert len(train_captions) == 8227
    assert tokenizer.vocab_size == 219

    print("Tokenizer check: OK")

    # -----------------------------------------------------
    # 8. Generate one example OOD caption
    # -----------------------------------------------------

    example_index = ood_indices[0]

    example_sample = dataset.samples[example_index]
    example_metadata = dataset._load_metadata(
        example_sample["metadata"]
    )

    example_caption = generate_caption(example_metadata)

    print("\n===== OOD CAPTION EXAMPLE =====")
    print(f"Dataset index: {example_index}")
    print(f"Caption: {example_caption}")

    # -----------------------------------------------------
    # 9. Timing
    # -----------------------------------------------------

    elapsed = time.perf_counter() - start_time

    print("\n===== TIMING =====")
    print(f"Elapsed time: {elapsed:.3f} seconds")

    # -----------------------------------------------------
    # 10. Final result
    # -----------------------------------------------------

    print("\n===== FINAL RESULT =====")
    print("OOD split verification completed successfully.")


if __name__ == "__main__":
    main()