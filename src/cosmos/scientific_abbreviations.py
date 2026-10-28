
import re


ABBREVIATIONS = {
    "JWST": "James Webb Space Telescope",
    "NIRSpec": "Near Infrared Spectrograph",
    "CO2": "carbon dioxide",
}


def normalize_abbreviations(text):
    """
    Expand known scientific abbreviations while
    preserving the meaning of the original text.
    """

    normalized = text

    for abbreviation, full_name in ABBREVIATIONS.items():
        pattern = (
            r"(?<!\w)"
            + re.escape(abbreviation)
            + r"(?!\w)"
        )

        normalized = re.sub(
            pattern,
            full_name,
            normalized,
            flags=re.IGNORECASE
        )

    return normalized


if __name__ == "__main__":
    examples = [
        "JWST observed WASP-39b.",
        "The James Webb Space Telescope observed WASP-39b.",
        "NIRSpec detected CO2.",
    ]

    for example in examples:
        print("\nOriginal:", example)
        print("Normalized:", normalize_abbreviations(example))
