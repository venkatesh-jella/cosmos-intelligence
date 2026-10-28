
import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer


# ============================================================
# 1. CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

WASP_CHUNKS_PATH = (
    PROCESSED_DIR / "wasp39b_token_chunks.json"
)

WASP_SECTIONS_PATH = (
    PROCESSED_DIR / "wasp39b_sections.json"
)

WASP_EMBEDDINGS_PATH = (
    PROCESSED_DIR / "wasp39b_embeddings.npy"
)

M87_CHUNKS_PATH = (
    PROCESSED_DIR / "m87_token_chunks.json"
)

M87_SECTIONS_PATH = (
    PROCESSED_DIR / "m87_sections.json"
)

M87_EMBEDDINGS_PATH = (
    PROCESSED_DIR / "m87_embeddings.npy"
)

COMBINED_CHUNKS_PATH = (
    PROCESSED_DIR / "multi_paper_token_chunks.json"
)

COMBINED_EMBEDDINGS_PATH = (
    PROCESSED_DIR / "multi_paper_embeddings.npy"
)

EXPECTED_DIMENSION = 384
BATCH_SIZE = 16


# ============================================================
# 2. LOAD HELPERS
# ============================================================

def load_json(path):

    if not path.exists():
        raise FileNotFoundError(
            f"File not found: {path}"
        )

    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def validate_embeddings(embeddings, chunks, label):

    if embeddings.ndim != 2:
        raise ValueError(
            f"{label}: embeddings must be a 2D matrix."
        )

    expected_shape = (
        len(chunks),
        EXPECTED_DIMENSION,
    )

    if embeddings.shape != expected_shape:
        raise ValueError(
            f"{label}: expected {expected_shape}, "
            f"got {embeddings.shape}"
        )

    if not np.isfinite(embeddings).all():
        raise ValueError(
            f"{label}: non-finite embedding values found."
        )

    norms = np.linalg.norm(embeddings, axis=1)

    if not np.allclose(norms, 1.0, atol=1e-3):
        raise ValueError(
            f"{label}: embeddings are not normalized."
        )

    print(
        f"{label} validation passed: "
        f"{embeddings.shape}"
    )


# ============================================================
# 3. PREPARE CONTEXTUALIZED INPUTS
# ============================================================

def build_embedding_texts(chunks, title):

    return [
        (
            f"Title: {title}\n"
            f"Section: {chunk['section_name']}\n\n"
            f"{chunk['text']}"
        )
        for chunk in chunks
    ]


def validate_input_lengths(model, texts, label):

    max_length = model.max_seq_length
    oversized = []

    for index, text in enumerate(texts):

        token_length = len(
            model.tokenizer.encode(
                text,
                add_special_tokens=True,
                truncation=False,
            )
        )

        if token_length > max_length:
            oversized.append(
                (index, token_length)
            )

    print(
        f"{label} oversized embedding inputs: "
        f"{len(oversized)}"
    )

    if oversized:
        raise ValueError(
            f"{label}: {len(oversized)} inputs exceed "
            f"the model limit of {max_length} tokens. "
            f"First oversized input: {oversized[0]}"
        )


# ============================================================
# 4. MAIN PIPELINE
# ============================================================

def main():

    print("\n" + "=" * 65)
    print("COSMOS INTELLIGENCE")
    print("MULTI-PAPER EMBEDDINGS")
    print("=" * 65)

    # --------------------------------------------------------
    # Load existing WASP-39b data
    # --------------------------------------------------------

    print("\n[1] Loading WASP-39b data...")

    wasp_chunks = load_json(WASP_CHUNKS_PATH)
    wasp_paper = load_json(WASP_SECTIONS_PATH)

    wasp_embeddings = np.load(
        WASP_EMBEDDINGS_PATH,
        allow_pickle=False,
    )

    validate_embeddings(
        wasp_embeddings,
        wasp_chunks,
        "WASP-39b",
    )

    # --------------------------------------------------------
    # Load M87 data
    # --------------------------------------------------------

    print("\n[2] Loading M87 black-hole data...")

    m87_chunks = load_json(M87_CHUNKS_PATH)
    m87_paper = load_json(M87_SECTIONS_PATH)

    if not m87_chunks:
        raise ValueError(
            "M87 chunk file is empty."
        )

    # --------------------------------------------------------
    # Load combined chunk dataset
    # --------------------------------------------------------

    print("\n[3] Loading combined chunk dataset...")

    combined_chunks = load_json(
        COMBINED_CHUNKS_PATH
    )

    expected_ids = [
        chunk["chunk_id"]
        for chunk in wasp_chunks + m87_chunks
    ]

    combined_ids = [
        chunk["chunk_id"]
        for chunk in combined_chunks
    ]

    if combined_ids != expected_ids:
        raise ValueError(
            "Combined chunk order does not match "
            "WASP-39b followed by M87."
        )

    if len(set(combined_ids)) != len(combined_ids):
        raise ValueError(
            "Duplicate chunk IDs found."
        )

    print(
        "Combined chunk order validated:",
        len(combined_chunks),
        "chunks"
    )

    # --------------------------------------------------------
    # Load embedding model
    # --------------------------------------------------------

    print("\n[4] Loading embedding model...")

    model = SentenceTransformer(
        MODEL_NAME,
        device="cpu",
    )

    if model.get_sentence_embedding_dimension() != EXPECTED_DIMENSION:
        raise ValueError(
            "Unexpected embedding dimension."
        )

    print("Model loaded:", MODEL_NAME)
    print("Maximum sequence length:", model.max_seq_length)

    # --------------------------------------------------------
    # Validate contextualized inputs
    # --------------------------------------------------------

    wasp_title = wasp_paper["document"]["title"]
    m87_title = m87_paper["document"]["title"]

    wasp_texts = build_embedding_texts(
        wasp_chunks,
        wasp_title,
    )

    m87_texts = build_embedding_texts(
        m87_chunks,
        m87_title,
    )

    validate_input_lengths(
        model,
        wasp_texts,
        "WASP-39b",
    )

    validate_input_lengths(
        model,
        m87_texts,
        "M87",
    )

    # --------------------------------------------------------
    # Encode M87 passages
    # --------------------------------------------------------

    print("\n[5] Generating M87 embeddings...")

    m87_embeddings = model.encode(
        m87_texts,
        batch_size=BATCH_SIZE,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True,
    )

    m87_embeddings = np.asarray(
        m87_embeddings,
        dtype=np.float32,
    )

    validate_embeddings(
        m87_embeddings,
        m87_chunks,
        "M87",
    )

    # --------------------------------------------------------
    # Combine matrices
    # --------------------------------------------------------

    print("\n[6] Combining embedding matrices...")

    combined_embeddings = np.vstack(
        [
            wasp_embeddings,
            m87_embeddings,
        ]
    ).astype(np.float32)

    validate_embeddings(
        combined_embeddings,
        combined_chunks,
        "Combined",
    )

    # --------------------------------------------------------
    # Save files
    # --------------------------------------------------------

    print("\n[7] Saving embeddings...")

    np.save(
        M87_EMBEDDINGS_PATH,
        m87_embeddings,
    )

    np.save(
        COMBINED_EMBEDDINGS_PATH,
        combined_embeddings,
    )

    # --------------------------------------------------------
    # Reload validation
    # --------------------------------------------------------

    print("\n[8] Reloading saved embeddings...")

    loaded_m87 = np.load(
        M87_EMBEDDINGS_PATH,
        allow_pickle=False,
    )

    loaded_combined = np.load(
        COMBINED_EMBEDDINGS_PATH,
        allow_pickle=False,
    )

    assert np.array_equal(
        loaded_m87,
        m87_embeddings,
    ), "M87 saved embeddings differ."

    assert np.array_equal(
        loaded_combined,
        combined_embeddings,
    ), "Combined saved embeddings differ."

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print("\n" + "=" * 65)
    print("FINAL SUMMARY")
    print("=" * 65)

    print(
        "WASP-39b chunks:",
        len(wasp_chunks)
    )

    print(
        "M87 chunks:",
        len(m87_chunks)
    )

    print(
        "Combined chunks:",
        len(combined_chunks)
    )

    print(
        "Combined embedding shape:",
        combined_embeddings.shape
    )

    print("\nSaved files:")
    print(M87_EMBEDDINGS_PATH)
    print(COMBINED_EMBEDDINGS_PATH)

    print("\nAll embedding validations passed!")


if __name__ == "__main__":
    main()
