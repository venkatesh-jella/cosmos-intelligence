
"""
Cosmos Intelligence
Claim Splitter

Splits a generated scientific answer into individual claims
and extracts citations associated with each claim.

Supported citation formats:
    [Evidence 1]
    [Evidence 4]
    [wasp39b_co2_2022_abstract_paragraph_token_v1_chunk_001]
"""

import re


# Recognize numbered evidence references and chunk IDs.
CITATION_PATTERN = re.compile(
    r"\[(?:Evidence\s+\d+|[A-Za-z0-9_]+_chunk_\d+)\]",
    flags=re.IGNORECASE,
)


def extract_claim_citations(text):
    """Return all recognized citations in a claim."""
    return CITATION_PATTERN.findall(text)


def remove_citations(text):
    """Remove citations while preserving the scientific claim."""
    text = CITATION_PATTERN.sub("", text)

    # Remove extra spaces left behind by citation removal.
    text = re.sub(r"\s+", " ", text)

    # Fix spaces before punctuation.
    text = re.sub(r"\s+([.,;:!?])", r"\1", text)

    return text.strip()


def split_into_sentences(text):
    """
    Split an answer into sentences.

    Avoid splitting decimal values such as 4.3 or 3.0.
    """
    text = text.strip()

    if not text:
        return []

    sentences = re.split(
        r"(?<=[.!?])\s+(?=[A-Z])",
        text,
    )

    return [
        sentence.strip()
        for sentence in sentences
        if sentence.strip()
    ]


def split_claims(answer):
    """
    Convert a generated answer into claim dictionaries.

    Each claim contains:
        claim_id
        claim
        citations
    """
    if not isinstance(answer, str):
        raise TypeError("Answer must be a string.")

    sentences = split_into_sentences(answer)
    claims = []

    for sentence in sentences:
        citations = extract_claim_citations(sentence)
        clean_claim = remove_citations(sentence)

        if not clean_claim:
            continue

        claims.append(
            {
                "claim_id": len(claims) + 1,
                "claim": clean_claim,
                "citations": citations,
            }
        )

    return claims


if __name__ == "__main__":

    print("\nCOSMOS INTELLIGENCE - CLAIM SPLITTING TEST")

    answer = (
        "Based on the provided evidence, JWST detected "
        "carbon dioxide in WASP-39b's atmosphere "
        "[wasp39b_co2_2022_abstract_paragraph_token_v1_chunk_001]. "
        "Transmission spectroscopy revealed a CO2 absorption "
        "feature at 4.3 micrometres with 26-sigma significance "
        "[Evidence 4]."
    )

    claims = split_claims(answer)

    for item in claims:
        print("\n" + "=" * 60)
        print("Claim ID:", item["claim_id"])
        print("Claim:", item["claim"])
        print("Citations:", item["citations"])

    print("\nTotal claims:", len(claims))
