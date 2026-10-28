
import json
import re
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer


PROJECT_ROOT = Path(__file__).resolve().parents[2]

CHUNKS_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "wasp39b_token_chunks.json"
)

EVIDENCE_CHUNK_ID = (
    "wasp39b_co2_2022_abstract_paragraph_token_v1_chunk_001"
)

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def split_sentences(text):
    """
    Split scientific text into sentences.

    Avoid splitting decimal numbers such as 4.3 or 3.0.
    This is a basic splitter, not a full scientific
    sentence tokenizer.
    """
    text = re.sub(r"\s+", " ", text).strip()

    return [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+", text)
        if sentence.strip()
    ]


class EvidenceSelector:
    def __init__(self):
        print("Loading evidence selection model...")

        self.model = SentenceTransformer(
            MODEL_NAME,
            device="cpu"
        )

        print("Evidence selection model loaded!")

    def select(self, claim, evidence_text, top_k=1):
        """
        Return the top-k evidence sentences most
        semantically similar to the scientific claim.
        """
        if top_k < 1:
            raise ValueError("top_k must be at least 1")

        sentences = split_sentences(evidence_text)

        if not sentences:
            return []

        claim_embedding = self.model.encode(
            [claim],
            normalize_embeddings=True
        )[0]

        sentence_embeddings = self.model.encode(
            sentences,
            normalize_embeddings=True
        )

        similarities = np.dot(
            sentence_embeddings,
            claim_embedding
        )

        ranked_indices = np.argsort(-similarities)

        selected = []

        for index in ranked_indices[:top_k]:
            index = int(index)

            selected.append({
                "sentence": sentences[index],
                "score": float(similarities[index]),
                "sentence_index": index
            })

        return selected


def load_evidence_chunk():
    with open(CHUNKS_PATH, "r", encoding="utf-8-sig") as file:
        chunks = json.load(file)

    if isinstance(chunks, dict):
        chunks = chunks["chunks"]

    for chunk in chunks:
        if chunk["chunk_id"] == EVIDENCE_CHUNK_ID:
            return chunk

    raise ValueError(
        f"Evidence chunk not found: {EVIDENCE_CHUNK_ID}"
    )


def main():
    from evidence_verifier import verify_claim_support

    claim = (
        "The CO2 absorption feature was detected "
        "at 4.3 micrometres."
    )

    chunk = load_evidence_chunk()

    selector = EvidenceSelector()

    selected = selector.select(
        claim=claim,
        evidence_text=chunk["text"],
        top_k=1
    )

    print("\nCOSMOS INTELLIGENCE")
    print("SENTENCE-LEVEL EVIDENCE SELECTION TEST")
    print("=" * 65)

    print("\nCLAIM:")
    print(claim)

    print("\nSOURCE CHUNK:")
    print(chunk["chunk_id"])

    if not selected:
        print("\nNo evidence sentences found.")
        return

    best = selected[0]

    print("\nSELECTED EVIDENCE:")
    print(best["sentence"])

    print("\nSemantic similarity:", round(best["score"], 4))

    result = verify_claim_support(
        claim,
        best["sentence"]
    )

    print("\nNLI PREDICTION:", result["predicted_label"])

    print(
        "Entailment:",
        round(result["scores"]["entailment"], 4)
    )
    print(
        "Neutral:",
        round(result["scores"]["neutral"], 4)
    )
    print(
        "Contradiction:",
        round(result["scores"]["contradiction"], 4)
    )


if __name__ == "__main__":
    main()
