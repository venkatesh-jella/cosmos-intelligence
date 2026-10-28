from sentence_transformers import SentenceTransformer

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

print("Loading embedding model...")

model = SentenceTransformer(MODEL_NAME, device="cpu")

print("Model loaded!")
print("Maximum sequence length:", model.max_seq_length)

sentences = [
    "JWST detected carbon dioxide in WASP-39b's atmosphere.",
    "The James Webb Space Telescope identified CO2 in an exoplanet atmosphere.",
    "The Moon orbits the Earth."
]

embeddings = model.encode(
    sentences,
    normalize_embeddings=True
)

print("\nEmbedding shape:", embeddings.shape)

similarity = embeddings @ embeddings.T

print("\nSimilarity between sentence 1 and 2:", round(float(similarity[0, 1]), 3))
print("Similarity between sentence 1 and 3:", round(float(similarity[0, 2]), 3))

import json
from pathlib import Path

chunks_path = Path("data/processed/wasp39b_token_chunks.json")
sections_path = Path("data/processed/wasp39b_sections.json")

with open(chunks_path, "r", encoding="utf-8") as f:
    chunks = json.load(f)

with open(sections_path, "r", encoding="utf-8") as f:
    paper = json.load(f)

title = paper["document"]["title"]

print("\n--- FULL EMBEDDING INPUT TEST ---")

oversized = 0
max_length = model.max_seq_length

for chunk in chunks:
    embedding_text = (
        f"Title: {title}\n"
        f"Section: {chunk['section_name']}\n\n"
        f"{chunk['text']}"
    )

    token_length = len(
        model.tokenizer.encode(
            embedding_text,
            add_special_tokens=True,
            truncation=False
        )
    )

    if token_length > max_length:
        oversized += 1

print("Total chunks:", len(chunks))
print("Model token limit:", max_length)
print("Oversized embedding inputs:", oversized)


import numpy as np
# Build the same contextualized inputs we validated
embedding_texts = [
    (
        f"Title: {title}\n"
        f"Section: {chunk['section_name']}\n\n"
        f"{chunk['text']}"
    )
    for chunk in chunks
]

# Convert the scientific passages into vectors
chunk_embeddings = model.encode(
    embedding_texts,
    batch_size=16,
    normalize_embeddings=True,
    convert_to_numpy=True,
    show_progress_bar=True

)

print("\n--- REAL PAPER EMBEDDINGS ---")
print("Embedding matrix shape:", chunk_embeddings.shape)

# Save the vectors
output_path = Path("data/processed/wasp39b_embeddings.npy")

np.save(output_path, chunk_embeddings)

# Reload to verify the saved file
loaded_embeddings = np.load(output_path)

assert loaded_embeddings.shape == (len(chunks), 384)
assert np.allclose(loaded_embeddings,chunk_embeddings)

print("Saved to:", output_path)
print("Reload validation passed!")
