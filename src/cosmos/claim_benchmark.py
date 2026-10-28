
import json
from pathlib import Path

from evidence_verifier import verify_claim_support


# Project root: cosmos-intelligence/
PROJECT_ROOT = Path(__file__).resolve().parents[2]

BENCHMARK_PATH = (
    PROJECT_ROOT
    / "data"
    / "evaluation"
    / "claim_verification_benchmark.json"
)

CHUNKS_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "wasp39b_token_chunks.json"
)

EVIDENCE_CHUNK_ID = (
    "wasp39b_co2_2022_abstract_paragraph_token_v1_chunk_001"
)


def load_json(path):
    with open(path, "r", encoding="utf-8-sig") as file:
        return json.load(file)


def load_evidence():
    chunks = load_json(CHUNKS_PATH)

    # Support both a list and a {"chunks": [...]} structure.
    if isinstance(chunks, dict):
        chunks = chunks["chunks"]

    for chunk in chunks:
        if chunk["chunk_id"] == EVIDENCE_CHUNK_ID:
            return chunk["text"]

    raise ValueError(
        f"Evidence chunk not found: {EVIDENCE_CHUNK_ID}"
    )


def normalize_prediction(label):
    if label == "entailment":
        return "entailment"

    return "not_entailment"


def evaluate():
    benchmark = load_json(BENCHMARK_PATH)
    evidence = load_evidence()

    print("\nCOSMOS INTELLIGENCE")
    print("REAL SCIENTIFIC CLAIM VERIFICATION BENCHMARK")
    print("=" * 65)

    print(f"\nEvidence chunk: {EVIDENCE_CHUNK_ID}")
    print(f"Total claims: {len(benchmark)}")

    correct = 0
    tp = fp = tn = fn = 0

    for item in benchmark:
        claim = item["claim"]
        expected = item["expected"]

        result = verify_claim_support(claim, evidence)

        raw_prediction = result["predicted_label"]
        prediction = normalize_prediction(raw_prediction)
        scores = result["scores"]

        passed = prediction == expected

        if passed:
            correct += 1

        if expected == "entailment":
            if prediction == "entailment":
                tp += 1
            else:
                fn += 1
        else:
            if prediction == "entailment":
                fp += 1
            else:
                tn += 1

        print("\n" + "-" * 65)
        print(f"TEST {item['id']}")
        print(f"Claim: {claim}")
        print(f"Expected: {expected}")
        print(f"Prediction: {raw_prediction}")

        print(
            "Entailment score:",
            round(scores["entailment"], 4)
        )
        print(
            "Contradiction score:",
            round(scores["contradiction"], 4)
        )
        print(
            "Neutral score:",
            round(scores["neutral"], 4)
        )

        print("RESULT:", "PASS" if passed else "FAIL")

    total = len(benchmark)

    if total == 0:
        raise ValueError("Benchmark is empty.")

    accuracy = correct / total
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )

    print("\n" + "=" * 65)
    print("FINAL BENCHMARK RESULTS")
    print("=" * 65)

    print(f"Total tests: {total}")
    print(f"Passed: {correct}")
    print(f"Failed: {total - correct}")
    print(f"Accuracy: {accuracy:.2%}")
    print(f"Precision: {precision:.2%}")
    print(f"Recall: {recall:.2%}")
    print(f"F1 score: {f1:.2%}")

    print("\nCONFUSION MATRIX")
    print(f"True positives:  {tp}")
    print(f"False positives: {fp}")
    print(f"True negatives:  {tn}")
    print(f"False negatives: {fn}")

    print("\nBenchmark execution completed.")


if __name__ == "__main__":
    evaluate()
