from prompt.attribute_mapping import (
    HAIR_COLOR_MAP,
    EYE_COLOR_MAP,
    GLASSES_MAP,
    FACIAL_HAIR_MAP,
    GLASSES_COLOR_MAP,
    HAIR_MAP,
)

import re


def normalize_prompt(prompt):
    prompt = prompt.lower().strip()

    replacements = {
        "light blue": "light_blue",
        "dark green": "dark_green",
        "light green": "light_green",
        "light brown": "light_brown",
        "red orange": "red_orange",
    }

    for old, new in replacements.items():
        prompt = prompt.replace(old, new)

    return prompt


def _find_matches(prompt, mapping, pattern):
    matches = []

    for word, value in mapping.items():
        if pattern.format(word=word) in prompt:
            matches.append((word, value))

    return matches


def _set_single_attribute(attributes, attribute, matches):
    if len(matches) > 1:
        values = [word for word, _ in matches]
        raise ValueError(
            f"Conflicting values for '{attribute}': {values}"
        )

    if len(matches) == 1:
        attributes[attribute] = matches[0][1]


def parse_prompt(prompt):
    prompt = normalize_prompt(prompt)

    attributes = {}

    # --------------------------------------------------
    # Hair color
    # --------------------------------------------------

    hair_color_matches = _find_matches(
        prompt,
        HAIR_COLOR_MAP,
        "{word} hair",
    )

    _set_single_attribute(
        attributes,
        "hair_color",
        hair_color_matches,
    )

    # --------------------------------------------------
    # Eye color
    # --------------------------------------------------

    eye_color_matches = _find_matches(
        prompt,
        EYE_COLOR_MAP,
        "{word} eyes",
    )

    # Natural-language aliases
    if (
            "blue eyes" in prompt
            and "light_blue eyes" not in prompt
    ):
        eye_color_matches.append(
            ("blue", EYE_COLOR_MAP["light_blue"])
        )

    if (
            "green eyes" in prompt
            and "light_green eyes" not in prompt
            and "dark_green eyes" not in prompt
    ):
        eye_color_matches.append(
            ("green", EYE_COLOR_MAP["light_green"])
        )

    _set_single_attribute(
        attributes,
        "eye_color",
        eye_color_matches,
    )

    # --------------------------------------------------
    # Glasses
    # --------------------------------------------------

    if "no glasses" in prompt:
        attributes["glasses"] = GLASSES_MAP["no glasses"]

    elif "sunglasses" in prompt:
        attributes["glasses"] = GLASSES_MAP["sunglasses"]

    elif "glasses" in prompt:
        attributes["glasses"] = GLASSES_MAP["glasses"]

    # --------------------------------------------------
    # Glasses color
    # --------------------------------------------------

    glasses_color_matches = []

    for color, value in GLASSES_COLOR_MAP.items():
        if (
            f"{color} glasses" in prompt
            or f"{color} sunglasses" in prompt
        ):
            glasses_color_matches.append((color, value))

    _set_single_attribute(
        attributes,
        "glasses_color",
        glasses_color_matches,
    )

    # --------------------------------------------------
    # Facial hair
    # --------------------------------------------------

    if "no facial hair" in prompt:
        attributes["facial_hair"] = FACIAL_HAIR_MAP["none"]
    else:
        facial_hair_matches = _find_matches(
            prompt,
            FACIAL_HAIR_MAP,
            "{word}",
        )

        _set_single_attribute(
            attributes,
            "facial_hair",
            facial_hair_matches,
        )

    # --------------------------------------------------
    # Hair style
    # --------------------------------------------------

    hair_style_matches = _find_matches(
        prompt,
        HAIR_MAP,
        "{word}",
    )

    # Handle hair color inserted between hair-style words and "hair".
    # Example:
    # "long curly red hair" -> "long curly hair"
    if not hair_style_matches:

        hair_style_patterns = [
            (r"\bshort curly(?: [a-z_]+)? hair\b", "short curly hair"),
            (r"\blong curly(?: [a-z_]+)? hair\b", "long curly hair"),
            (r"\blong straight(?: [a-z_]+)? hair\b", "long straight hair"),
        ]

        for pattern, hair_style in hair_style_patterns:
            if re.search(pattern, prompt):
                hair_style_matches.append(
                    (hair_style, HAIR_MAP[hair_style])
                )

    _set_single_attribute(
        attributes,
        "hair",
        hair_style_matches,
    )

    # --------------------------------------------------
    # Known ambiguous expressions
    # --------------------------------------------------

    if "curly hair" in prompt:
        if (
            "short curly hair" not in prompt
            and "long curly hair" not in prompt
        ):
            raise ValueError(
                "Ambiguous hair style: 'curly hair'. "
                "Use 'short curly hair' or 'long curly hair'."
            )

    return attributes


if __name__ == "__main__":

    test_prompts = [
        "a person with red hair, light green eyes and glasses",
        "a person with black hair and blue eyes",
        "a person with blonde hair and sunglasses",
        "a person with black hair, brown eyes and no glasses",
        "a person with red hair, blue eyes and black glasses",
        "a person with black hair and green eyes",
    ]

    for prompt in test_prompts:
        print("\nPrompt:")
        print(prompt)

        attributes = parse_prompt(prompt)

        print("Attributes:")
        print(attributes)