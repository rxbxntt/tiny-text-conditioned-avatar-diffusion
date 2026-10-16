"""Deterministic caption generation from Cartoon Set metadata."""

from typing import Mapping


# Fixed order required by decision D006.
ATTRIBUTE_ORDER = (
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
)


def generate_caption(metadata: Mapping[str, int]) -> str:
    """Convert one metadata dictionary into a deterministic caption."""
    if not isinstance(metadata, Mapping):
        raise TypeError("metadata must be a mapping of attribute names to integers")

    expected = set(ATTRIBUTE_ORDER)
    actual = set(metadata.keys())

    if actual != expected:
        missing = expected - actual
        unexpected = actual - expected
        raise ValueError(
            "Invalid metadata attributes. "
            f"Missing: {sorted(missing)}. "
            f"Unexpected: {sorted(unexpected)}."
        )

    tokens = []

    for attribute in ATTRIBUTE_ORDER:
        value = metadata[attribute]

        if not isinstance(value, int) or isinstance(value, bool):
            raise TypeError(
                f"Metadata value for '{attribute}' must be an integer, "
                f"got {type(value).__name__}"
            )

        tokens.append(f"{attribute}_{value}")

    return " ".join(tokens)
