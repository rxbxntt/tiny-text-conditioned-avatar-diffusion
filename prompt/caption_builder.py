from prompt.default_attributes import compute_default_attributes


CAPTION_FIELDS = [
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


# Compute the training-set defaults once.
DEFAULT_ATTRIBUTES = compute_default_attributes()


def build_caption(attributes):
    """
    Build a complete training-style caption.

    User-specified attributes override the training-set
    default values for unspecified attributes.
    """

    complete_attributes = DEFAULT_ATTRIBUTES.copy()
    complete_attributes.update(attributes)

    missing_fields = [
        field
        for field in CAPTION_FIELDS
        if field not in complete_attributes
    ]

    if missing_fields:
        raise ValueError(
            f"Missing attributes: {missing_fields}"
        )

    tokens = [
        f"{field}_{complete_attributes[field]}"
        for field in CAPTION_FIELDS
    ]

    return " ".join(tokens)


if __name__ == "__main__":
    test_attributes = {
        "hair_color": 3,
        "eye_color": 3,
        "glasses": 2,
    }

    caption = build_caption(test_attributes)

    print("\n===== GENERATED CAPTION =====")
    print(caption)