from prompt.parser import parse_prompt
from prompt.caption_builder import build_caption


def prompt_to_caption(prompt):
    """
    Converts a user-facing prompt into a complete
    training-style caption.
    """

    attributes = parse_prompt(prompt)

    return build_caption(attributes)


if __name__ == "__main__":
    test_prompts = [
        "a person with red hair, light green eyes and glasses",
        "a person with black hair and brown eyes",
        "a person with blonde hair and sunglasses",
    ]

    for prompt in test_prompts:
        print("\nPrompt:")
        print(prompt)

        caption = prompt_to_caption(prompt)

        print("Caption:")
        print(caption)