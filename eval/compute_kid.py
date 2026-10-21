"""
KID evaluation for the Tiny Text-Conditioned Avatar Diffusion project.

The script is intentionally runnable directly from PyCharm with the Run
(▶) button. Project paths are configured below.

KID is implemented explicitly as an unbiased polynomial-kernel MMD^2.
Inception-v3 is used only as the evaluation feature extractor.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.models import Inception_V3_Weights, inception_v3


# ============================================================
# PROJECT PATHS
# ============================================================

DATASET_ROOT = Path(
    r"C:\Users\utente\.cache\kagglehub\datasets"
    r"\paraglondhe\cartoonset100k\versions\2"
    r"\cartoonset100k\0"
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

TEST_MANIFEST = PROJECT_ROOT / "final_evaluation" / "ordinary_test" / "ordinary_test_manifest.csv"
TEST_GENERATED = PROJECT_ROOT / "final_evaluation" / "ordinary_test"

OOD_MANIFEST = PROJECT_ROOT / "final_evaluation" / "ood" / "ood_manifest.csv"
OOD_GENERATED = PROJECT_ROOT / "final_evaluation" / "ood"

OUTPUT_DIR = PROJECT_ROOT / "final_evaluation" / "kid_results"

BATCH_SIZE = 32

KID_NUM_SUBSETS = 10
KID_SUBSET_SIZE = 100

SEED = 42


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}

class ImageDataset(Dataset):
    """Dataset of explicitly selected image paths."""

    def __init__(self, paths: List[Path]):
        if not paths:
            raise ValueError("No images were provided.")

        self.paths = paths

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> Image.Image:
        with Image.open(self.paths[index]) as image:
            image = image.convert("RGB")
            image = image.resize((32, 32), Image.Resampling.BILINEAR)
            return image.copy()


def get_sorted_dataset_images() -> List[Path]:
    """
    Build the same deterministic image ordering used by the dataset loader:
    sorted PNG files from the dataset root.
    """
    if not DATASET_ROOT.exists():
        raise FileNotFoundError(
            f"Dataset root not found:\n{DATASET_ROOT}"
        )

    paths = sorted(
        path
        for path in DATASET_ROOT.glob("*.png")
        if path.is_file()
    )

    if not paths:
        raise ValueError(
            f"No PNG images found in dataset root:\n{DATASET_ROOT}"
        )

    return paths


def get_real_paths_from_manifest(
    manifest_path: Path,
    dataset_images: List[Path],
) -> List[Path]:
    """
    Recover real-image paths using dataset_index from the evaluation manifest.
    """
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Manifest not found:\n{manifest_path}"
        )

    real_paths = []

    with manifest_path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as file:
        reader = csv.DictReader(file)

        required_columns = {"dataset_index", "output_index"}
        if not required_columns.issubset(reader.fieldnames or set()):
            raise ValueError(
                f"Manifest {manifest_path.name} must contain "
                f"{required_columns}."
            )

        for row in reader:
            dataset_index = int(row["dataset_index"])

            if dataset_index < 0 or dataset_index >= len(dataset_images):
                raise IndexError(
                    f"dataset_index={dataset_index} is outside the "
                    f"dataset range [0, {len(dataset_images) - 1}]."
                )

            real_paths.append(dataset_images[dataset_index])

    if not real_paths:
        raise ValueError(
            f"No samples found in manifest:\n{manifest_path}"
        )

    return real_paths


def get_generated_paths(generated_dir: Path) -> List[Path]:
    """Load generated PNGs in deterministic filename order."""
    if not generated_dir.exists():
        raise FileNotFoundError(
            f"Generated directory not found:\n{generated_dir}"
        )

    paths = sorted(
        path
        for path in generated_dir.glob("*.png")
        if path.is_file()
    )

    if not paths:
        raise ValueError(
            f"No generated PNGs found in:\n{generated_dir}"
        )

    return paths


def build_inception_feature_extractor(
    device: torch.device,
) -> Tuple[nn.Module, object]:
    """Build Inception-v3 and expose its 2048-dimensional features."""
    weights = Inception_V3_Weights.DEFAULT

    model = inception_v3(
        weights=weights,
        aux_logits=True,
    )
    model.fc = nn.Identity()
    model.eval()
    model.to(device)

    return model, weights.transforms()


def collate_images(batch: List[Image.Image]) -> List[Image.Image]:
    return batch


@torch.no_grad()
def extract_features(
    image_paths: List[Path],
    model: nn.Module,
    transform,
    device: torch.device,
) -> torch.Tensor:
    """Extract 2048-dimensional Inception features."""
    dataset = ImageDataset(image_paths)

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_images,
    )

    all_features = []

    for images in loader:
        tensors = torch.stack(
            [transform(image) for image in images]
        ).to(device)

        features = model(tensors)

        if features.ndim != 2 or features.shape[1] != 2048:
            raise RuntimeError(
                f"Unexpected feature shape: {tuple(features.shape)}"
            )

        all_features.append(features.cpu())

    return torch.cat(all_features, dim=0)


def polynomial_kernel(
    x: torch.Tensor,
    y: torch.Tensor,
) -> torch.Tensor:
    """KID polynomial kernel."""
    d = x.shape[1]
    return (1.0 + (x @ y.T) / d) ** 3


def unbiased_mmd2(
    real_features: torch.Tensor,
    generated_features: torch.Tensor,
) -> torch.Tensor:
    """Unbiased MMD^2 estimator used by KID."""
    n = real_features.shape[0]
    m = generated_features.shape[0]

    if n < 2 or m < 2:
        raise ValueError("At least two samples are required per set.")

    k_xx = polynomial_kernel(real_features, real_features)
    k_yy = polynomial_kernel(generated_features, generated_features)
    k_xy = polynomial_kernel(real_features, generated_features)

    sum_xx = (
        k_xx.sum() - torch.diagonal(k_xx).sum()
    ) / (n * (n - 1))

    sum_yy = (
        k_yy.sum() - torch.diagonal(k_yy).sum()
    ) / (m * (m - 1))

    sum_xy = k_xy.mean()

    return sum_xx + sum_yy - 2.0 * sum_xy


def compute_kid(
    real_features: torch.Tensor,
    generated_features: torch.Tensor,
) -> Tuple[float, float]:
    """
    Compute KID using multiple random subsets.

    The same subset size and number of subsets are used for all splits.
    """
    if real_features.shape[0] != generated_features.shape[0]:
        raise ValueError(
            "Real and generated sets must have the same size."
        )

    num_samples = real_features.shape[0]

    if num_samples < KID_SUBSET_SIZE:
        raise ValueError(
            f"Not enough samples for KID subset size: "
            f"{num_samples} < {KID_SUBSET_SIZE}"
        )

    generator = torch.Generator(device="cpu")
    generator.manual_seed(SEED)

    kid_values = []

    for _ in range(KID_NUM_SUBSETS):
        indices = torch.randperm(
            num_samples,
            generator=generator,
        )[:KID_SUBSET_SIZE]

        real_subset = real_features[indices]
        generated_subset = generated_features[indices]

        value = unbiased_mmd2(
            real_subset,
            generated_subset,
        ).item()

        kid_values.append(value)

    values_tensor = torch.tensor(kid_values, dtype=torch.float32)

    mean = values_tensor.mean().item()
    std = values_tensor.std(unbiased=True).item()

    return float(mean), float(std)


def evaluate_split(
    name: str,
    manifest_path: Path,
    generated_dir: Path,
    dataset_images: List[Path],
    model: nn.Module,
    transform,
    device: torch.device,
) -> dict:
    """Evaluate one split: ordinary test or OOD."""

    print("\n" + "=" * 60)
    print(f"KID — {name}")
    print("=" * 60)

    real_paths = get_real_paths_from_manifest(
        manifest_path,
        dataset_images,
    )
    generated_paths = get_generated_paths(generated_dir)

    print(f"Real images:      {len(real_paths)}")
    print(f"Generated images: {len(generated_paths)}")

    if len(real_paths) != len(generated_paths):
        raise ValueError(
            f"{name}: real/generated sample count mismatch."
        )

    print("Extracting real features...")
    real_features = extract_features(
        real_paths,
        model,
        transform,
        device,
    )

    print("Extracting generated features...")
    generated_features = extract_features(
        generated_paths,
        model,
        transform,
        device,
    )

    kid_mean, kid_std = compute_kid(
        real_features,
        generated_features,
    )

    result = {
        "split": name,
        "metric": "KID",
        "feature_extractor": "Inception-v3",
        "feature_dimension": 2048,
        "kernel": "(1 + <x,y>/d)^3",
        "estimator": "unbiased MMD^2",
        "real_images": len(real_paths),
        "generated_images": len(generated_paths),
        "seed": SEED,
        "device": str(device),
        "kid_mean": kid_mean,
        "kid_std": kid_std,
    }

    print(f"\nKID: {kid_mean:.8f}")

    return result


def main() -> None:
    print("=" * 60)
    print("Tiny Text-Conditioned Avatar Diffusion — KID Evaluation")
    print("=" * 60)

    if not DATASET_ROOT.exists():
        raise FileNotFoundError(
            f"\nDataset root not found:\n{DATASET_ROOT}\n"
        )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"\nDevice: {device}")
    print(f"Dataset: {DATASET_ROOT}")

    dataset_images = get_sorted_dataset_images()

    print(f"Dataset images found: {len(dataset_images)}")

    model, transform = build_inception_feature_extractor(device)

    results = []

    results.append(
        evaluate_split(
            name="ordinary_test",
            manifest_path=TEST_MANIFEST,
            generated_dir=TEST_GENERATED,
            dataset_images=dataset_images,
            model=model,
            transform=transform,
            device=device,
        )
    )

    results.append(
        evaluate_split(
            name="ood",
            manifest_path=OOD_MANIFEST,
            generated_dir=OOD_GENERATED,
            dataset_images=dataset_images,
            model=model,
            transform=transform,
            device=device,
        )
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    output_path = OUTPUT_DIR / "kid_results.json"

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(results, file, indent=2)

    print("\n" + "=" * 60)
    print("FINAL KID RESULTS")
    print("=" * 60)

    for result in results:
        print(
            f"{result['split']}: "
            f"KID = {result['kid_mean']:.8f}"
        )

    print(f"\nSaved to: {output_path}")


if __name__ == "__main__":
    main()
