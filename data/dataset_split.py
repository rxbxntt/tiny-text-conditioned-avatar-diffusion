from collections import Counter
from torch.utils.data import Subset

from data.cartoon_dataset import CartoonSetDataset

def get_hair_eye_combination(metadata):
    """
    Returns the (hair_color, eye_color) combination of a sample.

    Args:
        metadata:
            Dictionary containing the sample attributes.

    Returns:
        Tuple containing hair_color and eye_color.
    """
    hair_color = metadata["hair_color"]
    eye_color = metadata["eye_color"]

    return hair_color, eye_color

def count_hair_eye_combinations(dataset):
    """
    Counts how many samples belong to each hair_color + eye_color
    combination.

    Args:
        dataset:
            CartoonSetDataset instance.

    Returns:
        Counter mapping (hair_color, eye_color) -> number of samples.
    """
    combinations = Counter()

    for sample in dataset.samples:
        metadata = dataset._load_metadata(sample["metadata"])

        combination = get_hair_eye_combination(metadata)
        combinations[combination] += 1

    return combinations

def print_hair_eye_combinations(combinations):
    """
    Prints hair_color + eye_color combinations ordered by frequency.
    """
    sorted_combinations = combinations.most_common()

    print("\n===== HAIR + EYE COLOR COMBINATIONS =====")

    for (hair_color, eye_color), count in sorted_combinations:
        print(
            f"hair_color={hair_color}, "
            f"eye_color={eye_color}: "
            f"{count} samples"
        )

def check_compositional_ood_candidate(
    dataset,
    hair_color,
    eye_color,
):
    """
    Checks whether both attribute values appear in other
    combinations, so that the selected pair can be used
    as a compositional OOD combination.
    """
    hair_combinations = set()
    eye_combinations = set()

    target_combination = (hair_color, eye_color)

    for sample in dataset.samples:
        metadata = dataset._load_metadata(sample["metadata"])

        combination = get_hair_eye_combination(metadata)

        if combination == target_combination:
            continue

        hair_combinations.add(combination[0])
        eye_combinations.add(combination[1])

    hair_seen = hair_color in hair_combinations
    eye_seen = eye_color in eye_combinations

    print("\n===== OOD CANDIDATE CHECK =====")
    print(f"Target combination: {target_combination}")
    print(f"Hair color seen elsewhere: {hair_seen}")
    print(f"Eye color seen elsewhere: {eye_seen}")

    return hair_seen and eye_seen

def create_dataset_splits(
    dataset,
    ood_hair_color,
    ood_eye_color,
    train_ratio=0.80,
    val_ratio=0.10,
    test_ratio=0.05,
    seed=42,
):
    """
    Creates train, validation, ordinary test and compositional OOD splits.

    The selected hair_color + eye_color combination is completely
    excluded from train, validation and ordinary test.

    Args:
        dataset:
            CartoonSetDataset instance.

        ood_hair_color:
            Hair color value used for the OOD combination.

        ood_eye_color:
            Eye color value used for the OOD combination.

        train_ratio:
            Target fraction of the non-OOD samples assigned to training.

        val_ratio:
            Target fraction of the non-OOD samples assigned to validation.

        test_ratio:
            Target fraction of the non-OOD samples assigned to
            ordinary test.

        seed:
            Random seed used to make the split reproducible.

    Returns:
        Four lists of sample indices:
            train_indices,
            val_indices,
            test_indices,
            ood_indices.
    """
    import random

    target_combination = (
        ood_hair_color,
        ood_eye_color,
    )

    ood_indices = []
    remaining_indices = []

    for index, sample in enumerate(dataset.samples):
        metadata = dataset._load_metadata(sample["metadata"])

        combination = get_hair_eye_combination(metadata)

        if combination == target_combination:
            ood_indices.append(index)
        else:
            remaining_indices.append(index)

    if len(ood_indices) == 0:
        raise ValueError(
            f"No samples found for OOD combination "
            f"{target_combination}."
        )

    total_ratio = train_ratio + val_ratio + test_ratio

    if abs(total_ratio - 0.95) > 1e-9:
        raise ValueError(
            "train_ratio + val_ratio + test_ratio must equal 0.95."
        )

    rng = random.Random(seed)
    rng.shuffle(remaining_indices)

    num_remaining = len(remaining_indices)

    num_train = round(
        num_remaining * train_ratio / total_ratio
    )

    num_val = round(
        num_remaining * val_ratio / total_ratio
    )

    train_indices = remaining_indices[:num_train]

    val_start = num_train
    val_end = val_start + num_val

    val_indices = remaining_indices[val_start:val_end]
    test_indices = remaining_indices[val_end:]

    return (
        train_indices,
        val_indices,
        test_indices,
        ood_indices,
    )

def print_split_statistics(
    train_indices,
    val_indices,
    test_indices,
    ood_indices,
):
    """
    Prints the size of each dataset split and verifies
    that the splits are mutually exclusive.
    """
    train_set = set(train_indices)
    val_set = set(val_indices)
    test_set = set(test_indices)
    ood_set = set(ood_indices)

    if train_set & val_set:
        raise ValueError("Train and validation overlap.")

    if train_set & test_set:
        raise ValueError("Train and test overlap.")

    if train_set & ood_set:
        raise ValueError("Train and OOD overlap.")

    if val_set & test_set:
        raise ValueError("Validation and test overlap.")

    if val_set & ood_set:
        raise ValueError("Validation and OOD overlap.")

    if test_set & ood_set:
        raise ValueError("Test and OOD overlap.")

    total_samples = (
        len(train_indices)
        + len(val_indices)
        + len(test_indices)
        + len(ood_indices)
    )

    print("\n===== DATASET SPLIT =====")
    print(f"Train:       {len(train_indices)}")
    print(f"Validation:  {len(val_indices)}")
    print(f"Test:        {len(test_indices)}")
    print(f"OOD:         {len(ood_indices)}")
    print(f"Total:       {total_samples}")

    print("\n===== SPLIT CHECK =====")
    print("All splits are mutually exclusive: OK")

def create_torch_subsets(
    dataset,
    train_indices,
    val_indices,
    test_indices,
    ood_indices,
):
    """
    Creates PyTorch Subset objects from the dataset indices.
    """
    train_subset = Subset(dataset, train_indices)
    val_subset = Subset(dataset, val_indices)
    test_subset = Subset(dataset, test_indices)
    ood_subset = Subset(dataset, ood_indices)

    return (
        train_subset,
        val_subset,
        test_subset,
        ood_subset,
    )

if __name__ == "__main__":
    dataset_root = (
        r"C:\Users\utente\.cache\kagglehub\datasets"
        r"\paraglondhe\cartoonset100k\versions\2"
        r"\cartoonset100k\0"
    )

    dataset = CartoonSetDataset(
        root_dir=dataset_root,
        expected_samples=10_000,
    )

    combinations = count_hair_eye_combinations(dataset)

    print_hair_eye_combinations(combinations)

    is_valid = check_compositional_ood_candidate(
        dataset,
        hair_color=4,
        eye_color=1,
    )

    print(f"Valid compositional OOD candidate: {is_valid}")

    (
        train_indices,
        val_indices,
        test_indices,
        ood_indices,
    ) = create_dataset_splits(
        dataset=dataset,
        ood_hair_color=4,
        ood_eye_color=1,
        seed=42,
    )

    print_split_statistics(
        train_indices=train_indices,
        val_indices=val_indices,
        test_indices=test_indices,
        ood_indices=ood_indices,
    )

    (
        train_subset,
        val_subset,
        test_subset,
        ood_subset,
    ) = create_torch_subsets(
        dataset=dataset,
        train_indices=train_indices,
        val_indices=val_indices,
        test_indices=test_indices,
        ood_indices=ood_indices,
    )

    print("\n===== TORCH SUBSETS =====")
    print(f"Train subset:      {len(train_subset)}")
    print(f"Validation subset: {len(val_subset)}")
    print(f"Test subset:       {len(test_subset)}")
    print(f"OOD subset:        {len(ood_subset)}")