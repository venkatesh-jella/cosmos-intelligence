
import json
from pathlib import Path

from evidence_verifier import verify_claim_support
from scientific_normalizer import normalize_scientific_text


PROJECT_ROOT = Path(__file__).resolve().parents[2]

CHUNKS_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "wasp39b_token_chunks.json"
)

TARGET_ID = (
    "wasp39b_co2_2022_abstract_paragraph_token_v1_chunk_001"
)

with open(CHUNKS_PATH, encoding="utf-8") as file:
    chunks = json.load(file)

matching = [
    chunk for chunk in chunks
    if chunk["chunk_id"] == TARGET_ID
]

if len(matching) != 1:
    raise ValueError("Expected exactly one evidence chunk.")

evidence = matching[0]["text"]


claims = {
    "A - Original Gemini answer": (
        r"Based on the provided evidence, JWST detected "
        r"carbon dioxide ($\text{CO}_2$) in the atmosphere "
        r"of the exoplanet WASP-39b."
    ),

    "B - Same claim without redundant chemical notation": (
        "Based on the provided evidence, JWST detected "
        "carbon dioxide in the atmosphere of the exoplanet WASP-39b."
    ),

    "C - Simple supported claim": (
        "JWST detected carbon dioxide in WASP-39b's atmosphere."
    ),

    "D - Incorrect methane claim": (
        "JWST detected methane instead of carbon dioxide "
        "in WASP-39b's atmosphere."
    ),

    "E - Unsupported temperature claim": (
        "WASP-39b has an atmospheric temperature of exactly 500 K."
    )
}


for name, original_claim in claims.items():

    normalized_claim = normalize_scientific_text(original_claim)

    original_result = verify_claim_support(
        original_claim,
        evidence
    )

    normalized_result = verify_claim_support(
        normalized_claim,
        evidence
    )

    print("\n" + "=" * 60)
    print(name)
    print("ORIGINAL:", original_claim)
    print("NORMALIZED:", normalized_claim)

    print(
        "ORIGINAL PREDICTION:",
        original_result["predicted_label"]
    )

    print(
        "ORIGINAL ENTAILMENT:",
        round(original_result["scores"]["entailment"], 4)
    )

    print(
        "NORMALIZED PREDICTION:",
        normalized_result["predicted_label"]
    )

    print(
        "NORMALIZED ENTAILMENT:",
        round(normalized_result["scores"]["entailment"], 4)
    )
