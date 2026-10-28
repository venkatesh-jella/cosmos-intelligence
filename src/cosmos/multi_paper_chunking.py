
import json
from pathlib import Path
from dataclasses import asdict
from collections import Counter

from chunking import (
    build_chunk_records,
    validate_chunks,
    save_chunks,
    count_tokens,
)


# ============================================================
# 1. CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

MODEL_TOKEN_LIMIT = 256

# Keep the existing WASP-39b chunks unchanged.
WASP_TOKEN_LIMIT = 240

# Leave room for the M87 title and section metadata.
M87_TOKEN_LIMIT = 200

WASP_DOCUMENT_ID = "wasp39b_co2_2022"
M87_DOCUMENT_ID = "m87_eht_2019"

WASP_CHUNKS_PATH = (
    PROCESSED_DIR / "wasp39b_token_chunks.json"
)

M87_SECTIONS_PATH = (
    PROCESSED_DIR / "m87_sections.json"
)

M87_CHUNKS_PATH = (
    PROCESSED_DIR / "m87_token_chunks.json"
)

COMBINED_CHUNKS_PATH = (
    PROCESSED_DIR / "multi_paper_token_chunks.json"
)

CHUNKING_CONFIG = "paragraph_token_200_v1"


# ============================================================
# 2. LOAD JSON
# ============================================================

def load_json(path):

    if not path.exists():
        raise FileNotFoundError(
            f"Required file not found: {path}"
        )

    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


# ============================================================
# 3. LOAD EXISTING WASP-39b CHUNKS
# ============================================================

def load_wasp_chunks():

    print("\n[1] Loading existing WASP-39b chunks...")

    chunks = load_json(WASP_CHUNKS_PATH)

    if not chunks:
        raise ValueError("WASP-39b chunks are empty.")

    for chunk in chunks:

        if chunk["document_id"] != WASP_DOCUMENT_ID:
            raise ValueError(
                "Incorrect WASP-39b document ID."
            )

        actual_tokens = count_tokens(chunk["text"])

        if actual_tokens != chunk["token_count"]:
            raise ValueError(
                "Incorrect WASP-39b token metadata."
            )

        if actual_tokens > WASP_TOKEN_LIMIT:
            raise ValueError(
                "WASP-39b chunk exceeds its token limit."
            )

    print("WASP-39b chunks:", len(chunks))

    return chunks


# ============================================================
# 4. REBUILD M87 CHUNKS
# ============================================================

def create_m87_chunks():

    print("\n[2] Loading M87 sections...")

    paper = load_json(M87_SECTIONS_PATH)

    if paper["document"]["document_id"] != M87_DOCUMENT_ID:
        raise ValueError("Incorrect M87 document ID.")

    sections = paper["sections"]

    if not sections:
        raise ValueError("M87 has no sections.")

    print("M87 sections:", len(sections))
    print("M87 chunk token limit:", M87_TOKEN_LIMIT)

    print("\n[3] Rebuilding M87 chunks...")

    chunks = build_chunk_records(
        sections=sections,
        strategy="token",
        limit=M87_TOKEN_LIMIT,
        config_id=CHUNKING_CONFIG,
    )

    if not chunks:
        raise ValueError("No M87 chunks were generated.")

    print("\n[4] Validating M87 chunks...")

    validate_chunks(
        chunks,
        strategy="token",
        limit=M87_TOKEN_LIMIT,
    )

    save_chunks(chunks, M87_CHUNKS_PATH)

    return chunks


# ============================================================
# 5. COMBINE BOTH PAPERS
# ============================================================

def combine_papers(wasp_chunks, m87_chunks):

    print("\n[5] Combining papers...")

    m87_records = [
        asdict(chunk)
        for chunk in m87_chunks
    ]

    combined = wasp_chunks + m87_records

    chunk_ids = [
        chunk["chunk_id"]
        for chunk in combined
    ]

    if len(chunk_ids) != len(set(chunk_ids)):
        raise ValueError("Duplicate chunk IDs detected.")

    document_counts = Counter(
        chunk["document_id"]
        for chunk in combined
    )

    expected_documents = {
        WASP_DOCUMENT_ID,
        M87_DOCUMENT_ID,
    }

    if set(document_counts) != expected_documents:
        raise ValueError(
            "Combined dataset must contain both papers."
        )

    for chunk in combined:

        actual_tokens = count_tokens(chunk["text"])

        if actual_tokens != chunk["token_count"]:
            raise ValueError(
                f"Token metadata mismatch: {chunk['chunk_id']}"
            )

        limit = (
            M87_TOKEN_LIMIT
            if chunk["document_id"] == M87_DOCUMENT_ID
            else WASP_TOKEN_LIMIT
        )

        if actual_tokens > limit:
            raise ValueError(
                f"Token limit exceeded: {chunk['chunk_id']}"
            )

    COMBINED_CHUNKS_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        COMBINED_CHUNKS_PATH,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            combined,
            file,
            indent=2,
            ensure_ascii=False,
        )

    return combined, document_counts


# ============================================================
# 6. MAIN
# ============================================================

def main():

    print("\n" + "=" * 65)
    print("COSMOS INTELLIGENCE")
    print("MULTI-PAPER CHUNKING - UPDATED")
    print("=" * 65)

    wasp_chunks = load_wasp_chunks()

    m87_chunks = create_m87_chunks()

    combined, document_counts = combine_papers(
        wasp_chunks,
        m87_chunks,
    )

    print("\n" + "=" * 65)
    print("FINAL SUMMARY")
    print("=" * 65)

    for document_id, count in document_counts.items():
        print(f"{document_id}: {count} chunks")

    print("\nTotal combined chunks:", len(combined))

    print(
        "Largest M87 chunk:",
        max(chunk.token_count for chunk in m87_chunks),
        "tokens"
    )

    print("\nSaved to:")
    print(M87_CHUNKS_PATH)
    print(COMBINED_CHUNKS_PATH)

    print("\nAll chunking validations passed!")


if __name__ == "__main__":
    main()
