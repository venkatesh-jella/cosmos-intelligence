import json
import re
from pathlib import Path

from rank_bm25 import BM25Okapi


def tokenize(text):
    """Convert text into normalized terms for BM25."""
    text = text.lower()
    text = text.replace("co₂", "co2")
    return re.findall(r"\w+(?:\.\w+)?", text)


json_path = Path("data/processed/wasp39b_token_chunks.json")

with open(json_path, "r", encoding="utf-8") as f:
    chunks = json.load(f)

# Turn every chunk into a list of searchable terms.
tokenized_corpus = [tokenize(chunk["text"]) for chunk in chunks]

# Build the BM25 index.
bm25 = BM25Okapi(tokenized_corpus)

question = "What molecule did JWST detect in WASP-39b's atmosphere?"

query_tokens = tokenize(question)
scores = bm25.get_scores(query_tokens)

# Rank chunks by their BM25 scores, highest first.
ranked_indices = sorted(
    range(len(scores)),
    key=lambda i: scores[i],
    reverse=True,
)

print("Question:", question)
print("Total indexed chunks:", len(chunks))

for rank, index in enumerate(ranked_indices[:10], start=1):
    chunk = chunks[index]

    print(f"\n--- Result {rank} ---")
    print("Chunk ID:", chunk["chunk_id"])
    print("Section:", chunk["section_name"])
    print("BM25 score:", round(float(scores[index]), 3))
    print("Passage:", chunk["text"][:250])
