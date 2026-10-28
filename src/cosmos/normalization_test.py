
import json
from pathlib import Path

from evidence_verifier import verify_claim_support


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
    "A - Simple claim":
        "JWST detected carbon dioxide in WASP-39b's atmosphere.",

    "B - Full scientific wording":
        "The James Webb Space Telescope (JWST) detected "
        "carbon dioxide in the atmosphere of the exoplanet WASP-39b.",

    "C - With introductory wording":
        "Based on the provided evidence, JWST detected "
        "carbon dioxide in the atmosphere of WASP-39b.",

    "D - With chemical notation":
        "JWST detected carbon dioxide (CO2) "
        "in the atmosphere of WASP-39b.",

    "E - With LaTeX notation":
        r"JWST detected carbon dioxide ($\text{CO}_2$) "
        "in the atmosphere of WASP-39b."
}

for name, claim in claims.items():
    result = verify_claim_support(claim, evidence)

    print("\n" + "=" * 55)
    print(name)
    print("CLAIM:", claim)
    print("PREDICTION:", result["predicted_label"])
    print(
        "ENTAILMENT SCORE:",
        round(result["scores"]["entailment"], 4)
    )
    print(
        "NEUTRAL SCORE:",
        round(result["scores"]["neutral"], 4)
    )
