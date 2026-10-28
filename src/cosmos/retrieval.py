import json
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

CHUNKS_PATH = Path("data/processed/wasp39b_token_chunks.json")
EMBEDDINGS_PATH = Path("data/processed/wasp39b_embeddings.npy")

# 1. Load the saved chunks and embeddings
with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
    chunks = json.load(f)

embeddings = np.load(EMBEDDINGS_PATH)

assert len(chunks) == len(embeddings), "Chunk/embedding count mismatch"

# 2. Load the same embedding model
model = SentenceTransformer(MODEL_NAME, device="cpu")


def search(query, top_k=3):
    # Convert the user's question into a vector
    query_vector = model.encode(
        query,
        normalize_embeddings=True
    )

    # Compare the question against all chunk vectors
    scores = embeddings @ query_vector

    # Select the highest-scoring chunk positions
    top_indices = np.argsort(scores)[::-1][:top_k]

    results = []

    for index in top_indices:
        chunk = chunks[int(index)]

        results.append({
            "chunk_id": chunk["chunk_id"],
            "section": chunk["section_name"],
            "score": float(scores[index]),
            "text": chunk["text"]
        })

    return results


if __name__ == "__main__":
    question = "What molecule did JWST detect in WASP-39b's atmosphere?"

    print("\nQuestion:", question)

    results = search(question, top_k=10)

    for rank, result in enumerate(results, start=1):
        text = result["text"]
        has_co2 = (
            "carbon dioxide" in text.lower()
            or "co₂" in text.lower()
            or "co2" in text.lower()
        )

        print(f"\n--- Result {rank} ---")
        print("Chunk ID:", result["chunk_id"])
        print("Section:", result["section"])
        print("Similarity:", round(result["score"], 3))
        print("Mentions CO2:", has_co2)
        print("Passage:", text[:250])
