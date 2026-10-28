
from evidence_verifier import verify_claim_support

evidence = (
    "Here we present the detection of CO2 in the atmosphere "
    "of the gas giant exoplanet WASP-39b from transmission "
    "spectroscopy observations obtained with JWST as part "
    "of the Early Release Science programme."
)

tests = [
    (
        "Test 5 - Original",
        "The observations used the James Webb Space Telescope."
    ),
    (
        "Test 5 - Abbreviation",
        "The observations were obtained with JWST."
    ),
    (
        "Test 6 - Original",
        "The observations were part of the Early Release Science programme."
    ),
    (
        "Test 6 - Explicit context",
        "The JWST observations of WASP-39b were part of "
        "the Early Release Science programme."
    ),
]

print("\nCOSMOS INTELLIGENCE")
print("NLI FAILURE DIAGNOSTIC")
print("=" * 60)

for name, claim in tests:
    result = verify_claim_support(claim, evidence)

    print("\n" + "-" * 60)
    print(name)
    print("Claim:", claim)
    print("Prediction:", result["predicted_label"])

    for label, score in result["scores"].items():
        print(f"{label}: {score:.4f}")
