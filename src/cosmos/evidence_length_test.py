
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

EVIDENCE_CHUNK_ID = (
    "wasp39b_co2_2022_abstract_paragraph_token_v1_chunk_001"
)

CLAIM = (
    "The CO2 absorption feature was detected "
    "at 4.3 micrometres."
)


def load_full_evidence():
    with open(CHUNKS_PATH, "r", encoding="utf-8-sig") as file:
        chunks = json.load(file)

    if isinstance(chunks, dict):
        chunks = chunks["chunks"]

    for chunk in chunks:
        if chunk["chunk_id"] == EVIDENCE_CHUNK_ID:
            return chunk["text"]

    raise ValueError(
        f"Evidence chunk not found: {EVIDENCE_CHUNK_ID}"
    )


def run_test(name, evidence):
    print("\n" + "=" * 65)
    print(name)
    print("=" * 65)

    print("\nEVIDENCE:")
    print(evidence)

    print("\nCLAIM:")
    print(CLAIM)

    result = verify_claim_support(CLAIM, evidence)

    print("\nPREDICTION:", result["predicted_label"])

    scores = result["scores"]

    print("Entailment:", round(scores["entailment"], 4))
    print("Neutral:", round(scores["neutral"], 4))
    print("Contradiction:", round(scores["contradiction"], 4))

    return result


def main():
    full_evidence = load_full_evidence()

    focused_evidence = (
        "The data used in this study span 3.0–5.5 micrometres "
        "in wavelength and show a prominent CO2 absorption "
        "feature at 4.3 micrometres (26-sigma significance)."
    )

    print("\nCOSMOS INTELLIGENCE")
    print("EVIDENCE LENGTH DIAGNOSTIC")

    full_result = run_test(
        "TEST A: FULL EVIDENCE CHUNK",
        full_evidence
    )

    focused_result = run_test(
        "TEST B: FOCUSED EVIDENCE SENTENCE",
        focused_evidence
    )

    print("\n" + "=" * 65)
    print("COMPARISON")
    print("=" * 65)

    print(
        "Full evidence prediction:",
        full_result["predicted_label"]
    )

    print(
        "Focused evidence prediction:",
        focused_result["predicted_label"]
    )

    if (
        full_result["predicted_label"] != "entailment"
        and focused_result["predicted_label"] == "entailment"
    ):
        print(
            "\nRESULT: Shorter evidence improved the prediction."
        )
        print(
            "Investigate context length and input truncation."
        )

    elif (
        full_result["predicted_label"] == "entailment"
        and focused_result["predicted_label"] == "entailment"
    ):
        print(
            "\nRESULT: Both evidence versions support the claim."
        )

    else:
        print(
            "\nRESULT: Evidence length alone did not resolve "
            "the classification."
        )
        print(
            "Further tests of wording, tokenization, "
            "and model behavior are needed."
        )


if __name__ == "__main__":
    main()
