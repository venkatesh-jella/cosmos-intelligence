
import re


CHUNK_CITATION_PATTERN = re.compile(
    r"\[([A-Za-z0-9_]+_chunk_\d+)\]"
)

EVIDENCE_CITATION_PATTERN = re.compile(
    r"\[Evidence\s+(\d+)\]",
    re.IGNORECASE
)


def extract_citations(answer):
    """
    Extract both chunk-ID citations and numbered
    evidence citations from a generated answer.
    """
    chunk_ids = CHUNK_CITATION_PATTERN.findall(answer)

    evidence_numbers = [
        int(number)
        for number in EVIDENCE_CITATION_PATTERN.findall(answer)
    ]

    return {
        "chunk_ids": chunk_ids,
        "evidence_numbers": evidence_numbers
    }


def verify_citations(answer, retrieved_chunks):
    """
    Verify citations against evidence supplied to Gemini.

    Evidence numbers correspond to the ordering used
    by build_augmented_prompt().

    This checks citation references, not factual support.
    """
    citations = extract_citations(answer)

    allowed_ids = {
        chunk["chunk_id"]
        for chunk in retrieved_chunks
    }

    valid = []
    invalid = []

    for chunk_id in citations["chunk_ids"]:
        if chunk_id in allowed_ids:
            valid.append(chunk_id)
        else:
            invalid.append(chunk_id)

    for number in citations["evidence_numbers"]:
        if 1 <= number <= len(retrieved_chunks):
            chunk_id = retrieved_chunks[number - 1]["chunk_id"]
            valid.append(chunk_id)
        else:
            invalid.append(f"Evidence {number}")

    valid = list(dict.fromkeys(valid))
    invalid = list(dict.fromkeys(invalid))

    has_citations = bool(
        citations["chunk_ids"] or citations["evidence_numbers"]
    )

    return {
        "citations_found": citations,
        "valid_citations": valid,
        "invalid_citations": invalid,
        "has_citations": has_citations,
        "all_citations_valid": (
            has_citations and not invalid
        )
    }


if __name__ == "__main__":

    retrieved_chunks = [
        {
            "chunk_id": "wasp39b_co2_2022_abstract_paragraph_token_v1_chunk_001",
            "text": "JWST detected carbon dioxide in WASP-39b."
        }
    ]

    tests = [
        (
            "Correct numbered citation",
            "JWST detected carbon dioxide [Evidence 1]."
        ),
        (
            "Correct chunk-ID citation",
            "JWST detected carbon dioxide "
            "[wasp39b_co2_2022_abstract_paragraph_token_v1_chunk_001]."
        ),
        (
            "Invalid numbered citation",
            "JWST detected methane [Evidence 99]."
        ),
        (
            "Missing citation",
            "JWST detected carbon dioxide."
        )
    ]

    for name, answer in tests:
        result = verify_citations(answer, retrieved_chunks)

        print(f"\nTEST: {name}")
        print("Valid citations:", result["valid_citations"])
        print("Invalid citations:", result["invalid_citations"])
        print("All citations valid:", result["all_citations_valid"])
