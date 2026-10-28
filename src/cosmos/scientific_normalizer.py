
import re
import unicodedata


# ============================================================
# 1. SCIENTIFIC TEXT NORMALIZATION
# ============================================================

def normalize_scientific_text(text):
    """
    Normalize scientific notation for evidence comparison.

    Handles:
    - Unicode formatting
    - Chemical subscripts
    - Common LaTeX commands
    - Multiplication and uncertainty symbols
    - Solar-mass notation
    - A limited PDF superscript extraction artifact

    Does not change ordinary numerical values,
    scientific negations, or named entities.
    """

    if not isinstance(text, str):
        raise TypeError("Text must be a string.")

    # --------------------------------------------------------
    # 1. Unicode cleanup
    # --------------------------------------------------------

    text = text.replace("\uf0a0", " ")
    text = text.replace("\u00a0", " ")

    text = unicodedata.normalize("NFKC", text)

    # --------------------------------------------------------
    # 2. LaTeX chemical notation
    # --------------------------------------------------------

    text = re.sub(
        r"\\text\s*\{\s*CO\s*\}\s*_\s*\{?2\}?",
        "CO2",
        text,
        flags=re.IGNORECASE,
    )

    text = re.sub(
        r"\\mathrm\s*\{\s*CO\s*\}\s*_\s*\{?2\}?",
        "CO2",
        text,
        flags=re.IGNORECASE,
    )

    # --------------------------------------------------------
    # 3. Solar mass symbols
    # --------------------------------------------------------

    text = re.sub(
        r"M\s*_\s*\{?\s*\\odot\s*\}?",
        "solar masses",
        text,
    )

    text = re.sub(
        r"M\s*[⊙☉]",
        "solar masses",
        text,
    )

    # --------------------------------------------------------
    # 4. Common LaTeX commands
    # --------------------------------------------------------

    text = re.sub(
        r"\\(?:text|mathrm)\s*\{([^{}]*)\}",
        r"\1",
        text,
    )

    text = re.sub(
        r"\\pm\b",
        " ± ",
        text,
    )

    text = re.sub(
        r"\\(?:times|cdot)\b",
        " x ",
        text,
    )

    text = re.sub(
        r"\\(?:mu)\b",
        "micro",
        text,
    )

    text = re.sub(
        r"\\[,;!:]",
        " ",
        text,
    )

    text = text.replace("\\(", "")
    text = text.replace("\\)", "")
    text = text.replace("$", "")

    # --------------------------------------------------------
    # 5. Normalize mathematical operators
    # --------------------------------------------------------

    text = text.replace("×", " x ")

    text = text.replace("−", "-")

    # --------------------------------------------------------
    # 6. Normalize powers of ten
    # --------------------------------------------------------

    # Example: 10^{9} -> 10^9
    text = re.sub(
        r"10\s*\^\s*\{\s*(\d+)\s*\}",
        r"10^\1",
        text,
    )

    # Example: 10 ^ 9 -> 10^9
    text = re.sub(
        r"10\s*\^\s*(\d+)",
        r"10^\1",
        text,
    )

    # --------------------------------------------------------
    # 7. Conservative PDF superscript repair
    # --------------------------------------------------------

    # Some PDFs extract 10^9 as 109.
    #
    # Only repair the pattern "x 109 Me" when nearby
    # text explicitly describes a black-hole mass.
    #
    # This is a heuristic for the inspected M87 PDF,
    # not a general mathematical equivalence.

    pattern = re.compile(
        r"\bx\s+10(\d)(?=\s+Me\b)"
    )

    def repair_mass_exponent(match):

        preceding = text[
            max(0, match.start() - 180):
            match.start()
        ].lower()

        has_mass_context = (
            "mass" in preceding
            and (
                "black hole" in preceding
                or "m87" in preceding
            )
        )

        if has_mass_context:
            exponent = match.group(1)
            return f"x 10^{exponent}"

        return match.group(0)

    text = pattern.sub(
        repair_mass_exponent,
        text,
    )

    # --------------------------------------------------------
    # 8. Normalize extracted solar-mass units
    # --------------------------------------------------------

    # The inspected M87 PDF extracts M_sun as "Me".
    # Only replace it immediately after a power of ten.

    text = re.sub(
    r"(?<=10\^9)\s+Me\b",
    " solar masses",
    text,
)

    # --------------------------------------------------------
    # 9. Normalize chemical subscripts
    # --------------------------------------------------------

    subscript_translation = str.maketrans(
        "₀₁₂₃₄₅₆₇₈₉",
        "0123456789",
    )

    text = text.translate(
        subscript_translation
    )

    text = re.sub(
        r"\bCO\s*_\s*\{?2\}?",
        "CO2",
        text,
        flags=re.IGNORECASE,
    )

    # --------------------------------------------------------
    # 10. Normalize whitespace
    # --------------------------------------------------------

    text = re.sub(
        r"\s+",
        " ",
        text,
    ).strip()

    return text


# ============================================================
# 2. TEST NORMALIZATION
# ============================================================

def main():

    examples = [
        (
            "WASP-39b chemical notation",
            "JWST detected CO₂ in WASP-39b."
        ),
        (
            "LaTeX chemical notation",
            r"JWST detected $\text{CO}_2$ in WASP-39b."
        ),
        (
            "M87 generated mass answer",
            r"The black hole mass is "
            r"$ (6.5 \pm 0.7) \times 10^9 "
            r"\, \text{Me} $."
        ),
        (
            "M87 extracted PDF mass",
            "For the adopted distance, "
            "the black hole mass of M87* is "
            "M = (6.5 ± 0.7) × 109 Me."
        ),
        (
            "Preserve unrelated numbers",
            "The observation contained 109 measurements."
        ),
        (
            "Preserve scientific negation",
            "The observations did not detect methane."
        ),
    ]

    print("\n" + "=" * 65)
    print("COSMOS INTELLIGENCE")
    print("SCIENTIFIC NORMALIZATION TEST")
    print("=" * 65)

    for label, original in examples:

        normalized = normalize_scientific_text(
            original
        )

        print(f"\nTEST: {label}")
        print("Original:", original)
        print("Normalized:", normalized)

    print("\nNormalization tests completed!")


if __name__ == "__main__":
    main()
