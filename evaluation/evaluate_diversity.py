from pathlib import Path
from PIL import Image
import numpy as np


def load_image(path):
    """Load a PNG as a float array in [0, 1]."""
    image = Image.open(path).convert("RGB")
    return np.asarray(image, dtype=np.float32) / 255.0


def mean_absolute_difference(image_a, image_b):
    """Mean pixel-wise absolute difference between two RGB images."""
    return np.mean(np.abs(image_a - image_b))


def main():
    project_root = Path(__file__).resolve().parent.parent
    output_dir = project_root / "training" / "outputs" / "diversity"

    seeds = [42, 123, 456, 789, 1000]
    images = {}

    for seed in seeds:
        path = output_dir / f"sample_seed_{seed}.png"

        if not path.exists():
            raise FileNotFoundError(f"Missing image: {path}")

        images[seed] = load_image(path)

    print("===== DIVERSITY EVALUATION =====")
    print(f"Images: {len(images)}")
    print(f"Seeds: {seeds}")

    # Check that all images have the same shape.
    shapes = {seed: image.shape for seed, image in images.items()}
    print(f"Shapes: {shapes}")

    if len(set(shapes.values())) != 1:
        raise ValueError("Images do not have the same shape.")

    # Compute all pairwise distances.
    distances = []

    for i, seed_a in enumerate(seeds):
        for seed_b in seeds[i + 1:]:
            distance = mean_absolute_difference(
                images[seed_a],
                images[seed_b],
            )

            distances.append(distance)

            print(
                f"Seed {seed_a} vs {seed_b} | "
                f"mean absolute difference = {distance:.6f}"
            )

    diversity_score = np.mean(distances)

    print()
    print(f"Pairwise comparisons: {len(distances)}")
    print(f"Mean pairwise L1 diversity: {diversity_score:.6f}")


if __name__ == "__main__":
    main()
