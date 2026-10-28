
import json
from pathlib import Path

from sentence_transformers import CrossEncoder


MODEL_NAME = "cross-encoder/nli-deberta-v3-small"

PROJECT_ROOT = Path(__file__).resolve().parents[2]

CHUNKS_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "wasp39b_token_chunks.json"
)

TARGET_CHUNK_ID = (
    "wasp39b_co2_2022_abstract_paragraph_token_v1_chunk_001"
)


# Load actual scientific evidence
with open(CHUNKS_PATH, "r", encoding="utf-8") as file:
    chunks = json.load(file)

matching_chunks = [
    chunk for chunk in chunks
    if chunk["chunk_id"] == TARGET_CHUNK_ID
]

if len(matching_chunks) != 1:
    raise ValueError(
        f"Expected one matching chunk, found {len(matching_chunks)}"
    )

evidence = matching_chunks[0]["text"]


# Test scientific claims
claims = [
    "JWST detected carbon dioxide in WASP-39b's atmosphere.",
    "JWST detected methane instead of carbon dioxide in WASP-39b's atmosphere.",
    "WASP-39b has an atmospheric temperature of exactly 500 K."
]


print("Loading local NLI model...")

model = CrossEncoder(
    MODEL_NAME,
    device="cpu",
    max_length=512
)

print("NLI model loaded!")
print("\nEvidence chunk:", TARGET_CHUNK_ID)
print("Evidence:", evidence)

pairs = [
    (evidence, claim)
    for claim in claims
]

scores = model.predict(
    pairs,
    apply_softmax=True
)

id2label = model.model.config.id2label

for claim, probabilities in zip(claims, scores):
    predicted_index = int(probabilities.argmax())
    predicted_label = id2label[predicted_index]

    print("\n" + "-" * 60)
    print("CLAIM:", claim)
    print("PREDICTION:", predicted_label)

    print("CLASS SCORES:")

    for index, probability in enumerate(probabilities):
        print(
            f"  {id2label[index]}: {float(probability):.4f}"
        )
