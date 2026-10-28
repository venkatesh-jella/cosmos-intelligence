
import json
from pathlib import Path

from hybrid_retrieval import hybrid_search
from reranker import rerank


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "retrieval_evaluation.json"
)

TOP_K = 5


# Small, manually labeled evaluation dataset.
# Each question has one expected source document.

TEST_CASES = [
    {
        "question": "What molecule did JWST detect in WASP-39b?",
        "expected_document": "wasp39b_co2_2022",
    },
    {
        "question": "At what wavelength was CO2 detected in WASP-39b?",
        "expected_document": "wasp39b_co2_2022",
    },
    {
        "question": "Which telescope observed carbon dioxide in WASP-39b?",
        "expected_document": "wasp39b_co2_2022",
    },
    {
        "question": "What is the significance of the CO2 absorption feature in WASP-39b?",
        "expected_document": "wasp39b_co2_2022",
    },
    {
        "question": "What instrument was used to study the atmosphere of WASP-39b?",
        "expected_document": "wasp39b_co2_2022",
    },
    {
        "question": "What did the Event Horizon Telescope observe in M87?",
        "expected_document": "m87_eht_2019",
    },
    {
        "question": "What is the estimated mass of the black hole M87*?",
        "expected_document": "m87_eht_2019",
    },
    {
        "question": "What is the diameter of the emission ring around M87*?",
        "expected_document": "m87_eht_2019",
    },
    {
        "question": "What observing wavelength did the Event Horizon Telescope use for M87?",
        "expected_document": "m87_eht_2019",
    },
    {
        "question": "What is the black hole shadow in the galaxy M87?",
        "expected_document": "m87_eht_2019",
    },
]


def evaluate_search(search_function, name):
    results = []

    print("\n" + "=" * 65)
    print(name.upper())
    print("=" * 65)

    for index, test in enumerate(TEST_CASES, start=1):
        question = test["question"]
        expected = test["expected_document"]

        retrieved = search_function(
            question,
            top_k=TOP_K,
        )

        document_ids = [
            item["document_id"]
            for item in retrieved
        ]

        correct_ranks = [
            rank
            for rank, document_id in enumerate(
                document_ids,
                start=1,
            )
            if document_id == expected
        ]

        first_correct_rank = (
            correct_ranks[0]
            if correct_ranks
            else None
        )

        hit_at_1 = first_correct_rank == 1
        hit_at_5 = first_correct_rank is not None

        reciprocal_rank = (
            1.0 / first_correct_rank
            if first_correct_rank
            else 0.0
        )

        result = {
            "question": question,
            "expected_document": expected,
            "retrieved_documents": document_ids,
            "first_correct_rank": first_correct_rank,
            "hit_at_1": hit_at_1,
            "hit_at_5": hit_at_5,
            "reciprocal_rank": reciprocal_rank,
        }

        results.append(result)

        status = "PASS" if hit_at_1 else "CHECK"

        print(f"\nQuestion {index}: {status}")
        print("Query:", question)
        print("Expected:", expected)
        print(
            "Top document:",
            document_ids[0] if document_ids else "NONE",
        )
        print("First correct rank:", first_correct_rank)

    count = len(results)

    metrics = {
        "total_questions": count,
        "hit_at_1": sum(
            result["hit_at_1"]
            for result in results
        ) / count,
        "hit_at_5": sum(
            result["hit_at_5"]
            for result in results
        ) / count,
        "mrr_at_5": sum(
            result["reciprocal_rank"]
            for result in results
        ) / count,
    }

    print("\n" + "-" * 65)
    print("EVALUATION METRICS")
    print("-" * 65)

    print(f"Hit@1: {metrics['hit_at_1']:.2%}")
    print(f"Hit@5: {metrics['hit_at_5']:.2%}")
    print(f"MRR@5: {metrics['mrr_at_5']:.4f}")

    return {
        "metrics": metrics,
        "results": results,
    }


def main():
    print("\nCOSMOS INTELLIGENCE")
    print("RETRIEVAL EVALUATION")
    print(f"Test questions: {len(TEST_CASES)}")

    hybrid_results = evaluate_search(
        hybrid_search,
        "Hybrid Retrieval",
    )

    reranked_results = evaluate_search(
        rerank,
        "Cross-Encoder Reranking",
    )

    report = {
        "evaluation_type": "document_retrieval",
        "top_k": TOP_K,
        "dataset_size": len(TEST_CASES),
        "hybrid": hybrid_results,
        "reranked": reranked_results,
        "limitations": [
            "Small, manually curated question set",
            "Evaluates correct-document retrieval only",
            "Does not evaluate passage relevance or answer correctness",
        ],
    }

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            report,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print("\n" + "=" * 65)
    print("EVALUATION COMPLETED")
    print("=" * 65)
    print("Saved report:", OUTPUT_PATH)


if __name__ == "__main__":
    main()
