import re

from sentence_transformers import CrossEncoder

from hybrid_retrieval import hybrid_search


# ---------- CONFIGURATION ----------

RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L6-v2"

CANDIDATE_COUNT = 20
FINAL_COUNT = 5


# ---------- LOAD RERANKER ----------

print("\nLoading cross-encoder reranker...")

reranker = CrossEncoder(
    RERANKER_MODEL,
    device="cpu",
    max_length=512
)

print("Reranker loaded!")


# ---------- RERANKING FUNCTION ----------

def rerank(query, top_k=FINAL_COUNT):

    # 1. Retrieve 20 candidates using hybrid search
    candidates = hybrid_search(
        query,
        top_k=CANDIDATE_COUNT
    )

    # 2. Create question-passage pairs
    pairs = [
        (query, candidate["text"])
        for candidate in candidates
    ]

    # 3. Score every pair with the cross-encoder
    scores = reranker.predict(
        pairs,
        batch_size=8,
        show_progress_bar=False
    )

    # 4. Attach scores and original hybrid ranks
    for rank, (candidate, score) in enumerate(
        zip(candidates, scores),
        start=1
    ):
        candidate["hybrid_rank"] = rank
        candidate["reranker_score"] = float(score)

    # 5. Sort by reranker score, highest first
    reranked = sorted(
        candidates,
        key=lambda item: item["reranker_score"],
        reverse=True
    )

    return reranked[:top_k]


# ---------- TEST ----------

if __name__ == "__main__":

    question = (
        "What molecule did JWST detect "
        "in WASP-39b's atmosphere?"
    )

    print("\nQuestion:", question)

    results = rerank(question)

    for rank, result in enumerate(results, start=1):

        text = result["text"]

        mentions_co2 = (
            "carbon dioxide" in text.lower()
            or "co₂" in text.lower()
            or "co2" in text.lower()
        )

        print(f"\n--- Reranked Result {rank} ---")
        print("Chunk ID:", result["chunk_id"])
        print("Section:", result["section"])
        print("Original hybrid rank:", result["hybrid_rank"])
        print(
            "Reranker score:",
            round(result["reranker_score"], 3)
        )
        print("Mentions CO2:", mentions_co2)
        print("Passage:", text[:500])

        match = re.search(
            r"carbon dioxide|co₂|co2",
            text,
            flags=re.IGNORECASE
        )

        if match:
            start = max(0, match.start() - 150)
            end = min(len(text), match.end() + 200)

            print("CO2 evidence context:", text[start:end])
        else:
            print("CO2 evidence context: No CO2 mention found")
