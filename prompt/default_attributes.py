from collections import Counter

from data.cartoon_dataset import CartoonSetDataset
from data.dataset_split import create_dataset_splits


ATTRIBUTE_FIELDS = [
    "eye_angle",
    "eye_lashes",
    "eye_lid",
    "chin_length",
    "eyebrow_weight",
    "eyebrow_shape",
    "eyebrow_thickness",
    "face_shape",
    "facial_hair",
    "hair",
    "eye_color",
    "face_color",
    "hair_color",
    "glasses",
    "glasses_color",
    "eye_slant",
    "eyebrow_width",
    "eye_eyebrow_distance",
]


DATASET_ROOT = (
    r"C:\Users\utente\.cache\kagglehub\datasets"
    r"\paraglondhe\cartoonset100k\versions\2"
    r"\cartoonset100k\0"
)


def compute_default_attributes():
    dataset = CartoonSetDataset(
        root_dir=DATASET_ROOT,
        expected_samples=10_000,
    )

    (
        train_indices,
        _,
        _,
        _,
    ) = create_dataset_splits(
        dataset=dataset,
        ood_hair_color=4,
        ood_eye_color=1,
        seed=42,
    )

    counters = {
        attribute: Counter()
        for attribute in ATTRIBUTE_FIELDS
    }

    for index in train_indices:
        metadata = dataset._load_metadata(
            dataset.samples[index]["metadata"]
        )

        for attribute in ATTRIBUTE_FIELDS:
            counters[attribute][metadata[attribute]] += 1

    default_attributes = {
        attribute: counter.most_common(1)[0][0]
        for attribute, counter in counters.items()
    }

    return default_attributes


if __name__ == "__main__":
    defaults = compute_default_attributes()

    print("\n===== DEFAULT ATTRIBUTES =====")

    for attribute, value in defaults.items():
        print(f"{attribute}: {value}")