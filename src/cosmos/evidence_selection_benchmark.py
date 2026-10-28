
import json
from pathlib import Path

from hybrid_evidence_verifier import HybridEvidenceVerifier


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
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    with open(path, "r", encoding="utf-8-sig") as file:
        return json.load(file)


def load_evidence():
    chunks = load_json(CHUNKS_PATH)

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


def calculate_metrics(expected_labels, predicted_labels):
    if len(expected_labels) != len(predicted_labels):
        raise ValueError("Label counts do not match.")

    if not expected_labels:
        raise ValueError("No predictions to evaluate.")

    tp = fp = tn = fn = 0

    for expected, predicted in zip(
        expected_labels,
        predicted_labels
    ):
        if expected == "entailment":
            if predicted == "entailment":
                tp += 1
            else:
                fn += 1
        else:
            if predicted == "entailment":
                fp += 1
            else:
                tn += 1

    total = len(expected_labels)

    accuracy = (tp + tn) / total
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
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


def evaluate():
    benchmark = load_json(BENCHMARK_PATH)

    if not isinstance(benchmark, list) or not benchmark:
        raise ValueError(
            "Benchmark must be a non-empty JSON list."
        )

    full_evidence = load_evidence()

    # Loads the sentence selector once.
    # The NLI model is loaded by evidence_verifier.
    verifier = HybridEvidenceVerifier()

    expected_labels = []
    full_predictions = []
    selected_predictions = []
    hybrid_predictions = []

    print("\n" + "=" * 70)
    print("COSMOS INTELLIGENCE")
    print("THREE-METHOD EVIDENCE VERIFICATION BENCHMARK")
    print("=" * 70)

    for item in benchmark:
        claim = item["claim"]
        expected = item["expected"]

        if expected not in {
            "entailment",
            "not_entailment"
        }:
            raise ValueError(
                f"Invalid expected label for test {item['id']}: "
                f"{expected}"
            )

        result = verifier.verify(
            claim=claim,
            evidence_text=full_evidence
        )

        full_result = result["full_result"]
        selected_result = result["selected_result"]

        full_prediction = normalize_prediction(
            full_result["predicted_label"]
        )

        if selected_result is None:
            selected_raw_label = "no_evidence"
            selected_prediction = "not_entailment"
            selected_score = 0.0
        else:
            selected_raw_label = selected_result[
                "predicted_label"
            ]

            selected_prediction = normalize_prediction(
                selected_raw_label
            )

            selected_score = selected_result[
                "scores"
            ]["entailment"]

        hybrid_prediction = (
            "entailment"
            if result["supported"]
            else "not_entailment"
        )

        expected_labels.append(expected)
        full_predictions.append(full_prediction)
        selected_predictions.append(selected_prediction)
        hybrid_predictions.append(hybrid_prediction)

        full_correct = full_prediction == expected
        selected_correct = selected_prediction == expected
        hybrid_correct = hybrid_prediction == expected

        print("\n" + "-" * 70)
        print(f"TEST {item['id']}")
        print("Claim:", claim)
        print("Expected:", expected)

        print(
            "Full evidence:",
            full_result["predicted_label"],
            f"(entailment={full_result['scores']['entailment']:.4f})"
        )

        print(
            "Selected evidence:",
            selected_raw_label,
            f"(entailment={selected_score:.4f})"
        )

        print("Hybrid prediction:", hybrid_prediction)
        print("Methods disagree:", result["disagreement"])
        print("Selected sentence:", result["selected_sentence"])

        print(
            "Correct:",
            f"Full={full_correct}",
            f"Selected={selected_correct}",
            f"Hybrid={hybrid_correct}"
        )

        if hybrid_correct and not full_correct:
            print("Hybrid vs full: IMPROVED")
        elif full_correct and not hybrid_correct:
            print("Hybrid vs full: REGRESSED")
        else:
            print("Hybrid vs full: UNCHANGED")

    full_metrics = calculate_metrics(
        expected_labels,
        full_predictions
    )

    selected_metrics = calculate_metrics(
        expected_labels,
        selected_predictions
    )

    hybrid_metrics = calculate_metrics(
        expected_labels,
        hybrid_predictions
    )

    print("\n" + "=" * 70)
    print("FINAL COMPARISON")
    print("=" * 70)

    print_metrics(
        "METHOD A: FULL EVIDENCE",
        full_metrics
    )

    print_metrics(
        "METHOD B: SENTENCE SELECTION",
        selected_metrics
    )

    print_metrics(
        "METHOD C: HYBRID VERIFICATION",
        hybrid_metrics
    )

    print("\n" + "=" * 70)
    print("HYBRID VS FULL EVIDENCE")
    print("=" * 70)

    for metric in [
        "accuracy",
        "precision",
        "recall",
        "f1"
    ]:
        difference = (
            hybrid_metrics[metric] - full_metrics[metric]
        ) * 100

        print(
            f"{metric.capitalize():10s}: "
            f"{difference:+.2f} percentage points"
        )

    print("\nBenchmark completed.")


if __name__ == "__main__":
    evaluate()
