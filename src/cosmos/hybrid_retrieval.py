
import json
import re
from pathlib import Path
from collections import Counter

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer


# ============================================================
# 1. CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

CHUNKS_PATH = (
    PROCESSED_DIR / "multi_paper_token_chunks.json"
)

EMBEDDINGS_PATH = (
    PROCESSED_DIR / "multi_paper_embeddings.npy"
)

RRF_K = 60

EXPECTED_DIMENSION = 384


# ============================================================
# 2. TEXT TOKENIZATION
# ============================================================

def tokenize(text):
    """
    Normalize scientific terms for keyword retrieval.
    """

    text = text.lower()

    text = text.replace("co₂", "co2")
    text = text.replace("co2", "carbon dioxide")
    text = text.replace("jwst", "james webb space telescope")

    return re.findall(
        r"\w+(?:\.\w+)?",
        text
    )


# ============================================================
# 3. LOAD DATA
# ============================================================

def load_index():

    if not CHUNKS_PATH.exists():
        raise FileNotFoundError(
            f"Chunk file not found: {CHUNKS_PATH}"
        )

    if not EMBEDDINGS_PATH.exists():
        raise FileNotFoundError(
            f"Embedding file not found: {EMBEDDINGS_PATH}"
        )

    with open(
        CHUNKS_PATH,
        "r",
        encoding="utf-8"
    ) as file:

        loaded_chunks = json.load(file)

    loaded_embeddings = np.load(
        EMBEDDINGS_PATH,
        allow_pickle=False
    )

    if len(loaded_chunks) != len(loaded_embeddings):
        raise ValueError(
            "Number of chunks and embeddings does not match."
        )

    if loaded_embeddings.ndim != 2:
        raise ValueError(
            "Embedding matrix must be two-dimensional."
        )

    if loaded_embeddings.shape[1] != EXPECTED_DIMENSION:
        raise ValueError(
            "Unexpected embedding dimension."
        )

    chunk_ids = [
        chunk["chunk_id"]
        for chunk in loaded_chunks
    ]

    if len(chunk_ids) != len(set(chunk_ids)):
        raise ValueError(
            "Duplicate chunk IDs found."
        )

    if not np.isfinite(loaded_embeddings).all():
        raise ValueError(
            "Non-finite embedding values detected."
        )

    norms = np.linalg.norm(
        loaded_embeddings,
        axis=1,
        keepdims=True
    )

    loaded_embeddings = (
        loaded_embeddings / np.maximum(norms, 1e-12)
    )

    return loaded_chunks, loaded_embeddings


# ============================================================
# 4. BUILD RETRIEVERS
# ============================================================

print("\nCOSMOS INTELLIGENCE")
print("Loading multi-paper retrieval index...")

chunks, embeddings = load_index()

print("Loading embedding model...")

model = SentenceTransformer(
    MODEL_NAME,
    device="cpu"
)

tokenized_corpus = [
    tokenize(chunk["text"])
    for chunk in chunks
]

bm25 = BM25Okapi(tokenized_corpus)

document_counts = Counter(
    chunk["document_id"]
    for chunk in chunks
)

print("\nIndexed research papers:")

for document_id, count in document_counts.items():
    print(f"  {document_id}: {count} chunks")

print("\nTotal indexed chunks:", len(chunks))
print("Embedding shape:", embeddings.shape)


# ============================================================
# 5. HYBRID SEARCH
# ============================================================

def hybrid_search(query, top_k=5):
    """
    Search across all indexed research papers.

    Combines:
        - Dense semantic retrieval
        - BM25 keyword retrieval
        - Reciprocal Rank Fusion

    Returns ranked passages with document metadata.
    """

    if not isinstance(query, str) or not query.strip():
        raise ValueError(
            "Query must be a nonempty string."
        )

    if top_k < 1:
        raise ValueError(
            "top_k must be at least 1."
        )

    # --------------------------------------------------------
    # Step 1: Dense retrieval
    # --------------------------------------------------------

    query_vector = model.encode(
        query,
        normalize_embeddings=True,
        convert_to_numpy=True
    )

    dense_scores = embeddings @ query_vector

    dense_ranking = np.argsort(
        -dense_scores,
        kind="stable"
    )

    # --------------------------------------------------------
    # Step 2: BM25 keyword retrieval
    # --------------------------------------------------------

    query_tokens = tokenize(query)

    bm25_scores = bm25.get_scores(query_tokens)

    bm25_ranking = np.argsort(
        -bm25_scores,
        kind="stable"
    )

    # --------------------------------------------------------
    # Step 3: Reciprocal Rank Fusion
    # --------------------------------------------------------

    rrf_scores = np.zeros(
        len(chunks),
        dtype=np.float64
    )

    dense_ranks = np.empty(
        len(chunks),
        dtype=np.int32
    )

    bm25_ranks = np.empty(
        len(chunks),
        dtype=np.int32
    )

    for rank, index in enumerate(
        dense_ranking,
        start=1
    ):

        dense_ranks[index] = rank

        rrf_scores[index] += (
            1.0 / (RRF_K + rank)
        )

    for rank, index in enumerate(
        bm25_ranking,
        start=1
    ):

        bm25_ranks[index] = rank

        rrf_scores[index] += (
            1.0 / (RRF_K + rank)
        )

    # --------------------------------------------------------
    # Step 4: Rank by fused scores
    # --------------------------------------------------------

    final_ranking = np.argsort(
        -rrf_scores,
        kind="stable"
    )

    results = []

    for index in final_ranking[:top_k]:

        index = int(index)

        chunk = chunks[index]

        results.append({
            "chunk_id": chunk["chunk_id"],
            "document_id": chunk["document_id"],
            "section": chunk["section_name"],
            "start_page": chunk.get("start_page"),
            "end_page": chunk.get("end_page"),
            "rrf_score": float(
                rrf_scores[index]
            ),
            "dense_score": float(
                dense_scores[index]
            ),
            "bm25_score": float(
                bm25_scores[index]
            ),
            "dense_rank": int(
                dense_ranks[index]
            ),
            "bm25_rank": int(
                bm25_ranks[index]
            ),
            "text": chunk["text"]
        })

    return results


# ============================================================
# 6. TEST QUERIES
# ============================================================

def test_retrieval():

    questions = [
        (
            "EXOPLANET QUESTION",
            "What molecule did JWST detect "
            "in WASP-39b's atmosphere?",
            "wasp39b_co2_2022"
        ),
        (
            "BLACK HOLE QUESTION",
            "What did the Event Horizon Telescope "
            "observe in the galaxy M87?",
            "m87_eht_2019"
        ),
        (
            "BLACK HOLE MASS QUESTION",
            "What is the estimated mass of "
            "the black hole M87*?",
            "m87_eht_2019"
        )
    ]

    print("\n" + "=" * 65)
    print("MULTI-PAPER HYBRID RETRIEVAL TEST")
    print("=" * 65)

    for label, question, expected_document in questions:

        print("\n" + "-" * 65)
        print(label)
        print("-" * 65)

        print("Question:", question)

        results = hybrid_search(
            question,
            top_k=5
        )

        print("\nTop 5 retrieved passages:")

        for rank, result in enumerate(
            results,
            start=1
        ):

            print(f"\nResult {rank}")
            print("Document:", result["document_id"])
            print("Chunk:", result["chunk_id"])
            print("Section:", result["section"])

            if result["start_page"] is not None:
                print("Page:", result["start_page"])

            print(
                "Dense rank:",
                result["dense_rank"]
            )

            print(
                "BM25 rank:",
                result["bm25_rank"]
            )

            print(
                "RRF score:",
                round(result["rrf_score"], 5)
            )

            print(
                "Passage:",
                result["text"][:350]
            )

        correct_document_count = sum(
            result["document_id"] == expected_document
            for result in results
        )

        print("\nExpected document:", expected_document)

        print(
            "Correct-document passages in top 5:",
            correct_document_count
        )

        print(
            "Top result from expected document:",
            results[0]["document_id"] == expected_document
        )


# ============================================================
# 7. MAIN
# ============================================================

if __name__ == "__main__":
    test_retrieval()
