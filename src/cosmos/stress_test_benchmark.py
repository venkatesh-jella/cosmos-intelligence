
import re

from evidence_selector import EvidenceSelector, load_evidence_chunk
from evidence_verifier import verify_claim_support


# ============================================================
# CONFIGURATION
# ============================================================

TOP_K = 3


# ============================================================
# SCIENTIFIC TEXT NORMALIZATION
# ============================================================

def normalize_scientific_text(text):
    """
    Expand common scientific abbreviations.

    Normalize both the claim and the evidence before
    sending them to the NLI model.
    """

    replacements = [
        (r"\bJWST\b", "James Webb Space Telescope"),
        (r"\bNIRSpec\b", "Near Infrared Spectrograph"),
        (r"\bCO[₂2]\b", "carbon dioxide"),
    ]

    normalized = text

    for pattern, replacement in replacements:
        normalized = re.sub(
            pattern,
            replacement,
            normalized,
            flags=re.IGNORECASE
        )

    return normalized


# ============================================================
# STRESS TEST CASES
# ============================================================

TEST_CASES = [
    {
        "id": 1,
        "category": "supported_detection",
        "claim": "JWST detected carbon dioxide in WASP-39b's atmosphere.",
        "expected": "entailment",
    },
    {
        "id": 2,
        "category": "supported_wavelength",
        "claim": "The carbon dioxide absorption feature was observed at 4.3 micrometres.",
        "expected": "entailment",
    },
    {
        "id": 3,
        "category": "supported_significance",
        "claim": "The carbon dioxide absorption feature had a significance of 26 sigma.",
        "expected": "entailment",
    },
    {
        "id": 4,
        "category": "supported_planet_type",
        "claim": "WASP-39b is a gas giant exoplanet.",
        "expected": "entailment",
    },
    {
        "id": 5,
        "category": "supported_telescope",
        "claim": "The observations used the James Webb Space Telescope.",
        "expected": "entailment",
    },
    {
        "id": 6,
        "category": "supported_programme",
        "claim": "The observations were part of the Early Release Science programme.",
        "expected": "entailment",
    },
    {
        "id": 7,
        "category": "supported_range",
        "claim": "The observed spectral data covered wavelengths from 3.0 to 5.5 micrometres.",
        "expected": "entailment",
    },
    {
        "id": 8,
        "category": "incorrect_molecule",
        "claim": "JWST detected methane instead of carbon dioxide in WASP-39b.",
        "expected": "not_entailment",
    },
    {
        "id": 9,
        "category": "incorrect_wavelength",
        "claim": "The carbon dioxide absorption feature was observed at 8.6 micrometres.",
        "expected": "not_entailment",
    },
    {
        "id": 10,
        "category": "incorrect_significance",
        "claim": "The carbon dioxide absorption feature had a significance of 2.6 sigma.",
        "expected": "not_entailment",
    },
    {
        "id": 11,
        "category": "incorrect_planet",
        "claim": "The carbon dioxide feature was detected in the atmosphere of Mars.",
        "expected": "not_entailment",
    },
    {
        "id": 12,
        "category": "incorrect_telescope",
        "claim": "The observations were made using the Hubble Space Telescope rather than JWST.",
        "expected": "not_entailment",
    },
    {
        "id": 13,
        "category": "incorrect_planet_type",
        "claim": "WASP-39b is a rocky terrestrial planet.",
        "expected": "not_entailment",
    },
    {
        "id": 14,
        "category": "incorrect_range",
        "claim": "The observed wavelength range was 30 to 55 micrometres.",
        "expected": "not_entailment",
    },
    {
        "id": 15,
        "category": "unsupported_temperature",
        "claim": "WASP-39b has an atmospheric temperature of exactly 500 K.",
        "expected": "not_entailment",
    },
    {
        "id": 16,
        "category": "unsupported_biosignature",
        "claim": "JWST detected oxygen produced by biological organisms on WASP-39b.",
        "expected": "not_entailment",
    },
    {
        "id": 17,
        "category": "unsupported_ocean",
        "claim": "WASP-39b has a confirmed liquid-water ocean.",
        "expected": "not_entailment",
    },
    {
        "id": 18,
        "category": "unsupported_life",
        "claim": "The observations prove that WASP-39b supports intelligent life.",
        "expected": "not_entailment",
    },
    {
        "id": 19,
        "category": "partially_supported",
        "claim": "JWST detected carbon dioxide in WASP-39b and confirmed that the planet is habitable.",
        "expected": "not_entailment",
    },
    {
        "id": 20,
        "category": "partially_incorrect_number",
        "claim": "The carbon dioxide feature was detected at 4.3 micrometres with 100-sigma significance.",
        "expected": "not_entailment",
    },
]


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(expected_labels, predicted_labels):

    if len(expected_labels) != len(predicted_labels):
        raise ValueError("Prediction count mismatch.")

    if not expected_labels:
        raise ValueError("No predictions to evaluate.")

    tp = fp = tn = fn = 0

    for expected, predicted in zip(
        expected_labels,
        predicted_labels
    ):
        if expected and predicted:
            tp += 1

        elif not expected and predicted:
            fp += 1

        elif not expected and not predicted:
            tn += 1

        else:
            fn += 1

    total = len(expected_labels)

    accuracy = (tp + tn) / total

    precision = (
        tp / (tp + fp)
        if tp + fp > 0
        else 0.0
    )

    recall = (
        tp / (tp + fn)
        if tp + fn > 0
        else 0.0
    )

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall > 0
        else 0.0
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }


def print_metrics(name, metrics):

    print(f"\n{name}")
    print("-" * 60)

    print(f"Accuracy:  {metrics['accuracy']:.2%}")
    print(f"Precision: {metrics['precision']:.2%}")
    print(f"Recall:    {metrics['recall']:.2%}")
    print(f"F1 score:  {metrics['f1']:.2%}")

    print(
        "Confusion matrix:",
        f"TP={metrics['tp']}",
        f"FP={metrics['fp']}",
        f"TN={metrics['tn']}",
        f"FN={metrics['fn']}"
    )


# ============================================================
# SUPPORT CHECK
# ============================================================

def is_supported(result):
    return result["predicted_label"] == "entailment"


# ============================================================
# EVALUATION
# ============================================================

def evaluate():

    print("\nLoading evidence...")

    chunk = load_evidence_chunk()
    evidence_text = chunk["text"]

    selector = EvidenceSelector()

    expected_labels = []

    full_predictions = []
    top1_predictions = []
    hybrid_predictions = []

    print("\nCOSMOS INTELLIGENCE")
    print("NORMALIZED TOP-3 HYBRID STRESS TEST")
    print("=" * 65)

    print("Number of tests:", len(TEST_CASES))
    print("Top-K sentences:", TOP_K)

    for item in TEST_CASES:

        claim = item["claim"]
        expected_label = item["expected"]
        expected = expected_label == "entailment"

        # ----------------------------------------------------
        # METHOD A: ORIGINAL FULL EVIDENCE
        # ----------------------------------------------------

        full_result = verify_claim_support(
            claim,
            evidence_text
        )

        full_supported = is_supported(full_result)

        # ----------------------------------------------------
        # METHOD B: ORIGINAL TOP-1 SENTENCE
        # ----------------------------------------------------

        selected_top1 = selector.select(
            claim=claim,
            evidence_text=evidence_text,
            top_k=1
        )

        top1_supported = False

        if selected_top1:

            top1_result = verify_claim_support(
                claim,
                selected_top1[0]["sentence"]
            )

            top1_supported = is_supported(top1_result)

        # ----------------------------------------------------
        # METHOD C: NORMALIZED TOP-3 HYBRID
        # ----------------------------------------------------

        normalized_claim = normalize_scientific_text(
            claim
        )

        normalized_evidence = normalize_scientific_text(
            evidence_text
        )

        normalized_full_result = verify_claim_support(
            normalized_claim,
            normalized_evidence
        )

        normalized_full_supported = is_supported(
            normalized_full_result
        )

        selected_top3 = selector.select(
            claim=normalized_claim,
            evidence_text=normalized_evidence,
            top_k=TOP_K
        )

        top3_supported = False
        top3_diagnostics = []

        for candidate in selected_top3:

            original_sentence = candidate["sentence"]

            normalized_sentence = normalize_scientific_text(
                original_sentence
            )

            result = verify_claim_support(
                normalized_claim,
                normalized_sentence
            )

            supported = is_supported(result)

            if supported:
                top3_supported = True

            top3_diagnostics.append({
                "sentence": original_sentence,
                "normalized_sentence": normalized_sentence,
                "similarity": candidate["score"],
                "prediction": result["predicted_label"],
                "entailment": result["scores"]["entailment"],
                "supported": supported,
            })

        hybrid_supported = (
            normalized_full_supported
            or top3_supported
        )

        # ----------------------------------------------------
        # RECORD PREDICTIONS
        # ----------------------------------------------------

        expected_labels.append(expected)

        full_predictions.append(
            full_supported
        )

        top1_predictions.append(
            top1_supported
        )

        hybrid_predictions.append(
            hybrid_supported
        )

        # ----------------------------------------------------
        # PRINT TEST RESULT
        # ----------------------------------------------------

        print("\n" + "-" * 65)

        print(
            f"TEST {item['id']}: {item['category']}"
        )

        print("Claim:", claim)
        print("Expected:", expected_label)

        print("Full supported:", full_supported)
        print("Top-1 supported:", top1_supported)

        print(
            "Normalized full supported:",
            normalized_full_supported
        )

        print("Top-3 supported:", top3_supported)
        print("Hybrid supported:", hybrid_supported)

        passed = hybrid_supported == expected

        print(
            "HYBRID RESULT:",
            "PASS" if passed else "FAIL"
        )

        # ----------------------------------------------------
        # DETAILED DIAGNOSTICS FOR TEST 5
        # ----------------------------------------------------

        if item["id"] == 5:

            print("\n" + "=" * 65)
            print("TEST 5: DETAILED DIAGNOSTICS")
            print("=" * 65)

            print("\nOriginal claim:")
            print(claim)

            print("\nNormalized claim:")
            print(normalized_claim)

            print("\nOriginal full prediction:")
            print(full_result["predicted_label"])

            print(
                "Entailment:",
                round(
                    full_result["scores"]["entailment"],
                    4
                )
            )

            print("\nNormalized full prediction:")
            print(
                normalized_full_result["predicted_label"]
            )

            print(
                "Entailment:",
                round(
                    normalized_full_result[
                        "scores"
                    ]["entailment"],
                    4
                )
            )

            print("\nTOP-3 HYBRID DETAILS")

            for index, diagnostic in enumerate(
                top3_diagnostics,
                start=1
            ):

                print(f"\nCandidate {index}")

                print("\nOriginal sentence:")
                print(diagnostic["sentence"])

                print("\nNormalized sentence:")
                print(
                    diagnostic["normalized_sentence"]
                )

                print(
                    "\nSimilarity:",
                    round(
                        diagnostic["similarity"],
                        4
                    )
                )

                print(
                    "Prediction:",
                    diagnostic["prediction"]
                )

                print(
                    "Entailment:",
                    round(
                        diagnostic["entailment"],
                        4
                    )
                )

                print(
                    "Supported:",
                    diagnostic["supported"]
                )

            print("\nNormalized full supported:",
                  normalized_full_supported)

            print("Top-3 supported:",
                  top3_supported)

            print("Final hybrid supported:",
                  hybrid_supported)

            print("=" * 65)

    # ========================================================
    # FINAL METRICS
    # ========================================================

    full_metrics = calculate_metrics(
        expected_labels,
        full_predictions
    )

    top1_metrics = calculate_metrics(
        expected_labels,
        top1_predictions
    )

    hybrid_metrics = calculate_metrics(
        expected_labels,
        hybrid_predictions
    )

    print("\n" + "=" * 65)
    print("FINAL STRESS TEST RESULTS")
    print("=" * 65)

    print_metrics(
        "METHOD A: FULL EVIDENCE",
        full_metrics
    )

    print_metrics(
        "METHOD B: TOP-1 SENTENCE SELECTION",
        top1_metrics
    )

    print_metrics(
        "METHOD C: NORMALIZED TOP-3 HYBRID",
        hybrid_metrics
    )

    print("\n" + "=" * 65)
    print("HYBRID VS FULL EVIDENCE")
    print("=" * 65)

    for metric in [
        "accuracy",
        "precision",
        "recall",
        "f1"
    ]:

        difference = (
            hybrid_metrics[metric]
            - full_metrics[metric]
        ) * 100

        print(
            f"{metric.capitalize():10s}: "
            f"{difference:+.2f} percentage points"
        )

    print("\nStress test completed.")


if __name__ == "__main__":
    evaluate()
