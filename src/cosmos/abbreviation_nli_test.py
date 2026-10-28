
from evidence_verifier import verify_claim_support
from scientific_abbreviations import normalize_abbreviations


evidence = (
    "Here we present the detection of CO2 in the atmosphere "
    "of the gas giant exoplanet WASP-39b from transmission "
    "spectroscopy observations obtained with JWST as part "
    "of the Early Release Science programme."
)

claims = [
    "The observations used the James Webb Space Telescope.",
    "The observations were part of the Early Release Science programme.",
    "JWST detected carbon dioxide in WASP-39b's atmosphere.",
    "The observations used the Hubble Space Telescope.",
]


def display_result(label, result):
    print(f"{label} prediction: {result['predicted_label']}")
    print(
        f"{label} entailment: "
        f"{result['scores']['entailment']:.4f}"
    )


for index, claim in enumerate(claims, start=1):
    original = verify_claim_support(claim, evidence)

    normalized_claim = normalize_abbreviations(claim)
    normalized_evidence = normalize_abbreviations(evidence)

    normalized = verify_claim_support(
        normalized_claim,
        normalized_evidence
    )

    print("\n" + "=" * 65)
    print(f"TEST {index}")
    print("Claim:", claim)

    display_result("Original", original)
    display_result("Normalized", normalized)

    print(
        "Prediction changed:",
        original["predicted_label"]
        != normalized["predicted_label"]
    )


print("\nComparison completed.")
