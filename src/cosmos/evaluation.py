import json
from pathlib import Path

from hybrid_retrieval import hybrid_search
from reranker import rerank
from hybrid_retrieval import hybrid_search
from reranker import rerank


# ==================================================
# COSMOS INTELLIGENCE - RETRIEVAL EVALUATION
# ==================================================


# ---------- CONFIGURATION ----------

TOP_K_VALUES = [1, 3, 5, 10, 20]
RETRIEVAL_LIMIT = 20


# ---------- LOAD EVALUATION BENCHMARK ----------

BENCHMARK_PATH = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "evaluation"
    / "wasp39b_benchmark.json"
)


def load_benchmark():
    with open(BENCHMARK_PATH, "r", encoding="utf-8-sig") as file:
        benchmark = json.load(file)

    if not benchmark:
        raise ValueError("Benchmark cannot be empty")

    print(f"Loaded {len(benchmark)} benchmark question(s)")

    return benchmark


benchmark = load_benchmark()

# ---------- RECALL@K ----------

def recall_at_k(retrieved_ids, gold_ids, k):
    """
    Fraction of verified relevant chunks
    found in the top k retrieved results.
    """
    if k <= 0:
        raise ValueError("k must be greater than zero")

    gold_set = set(gold_ids)

    if not gold_set:
        raise ValueError("Gold evidence cannot be empty")

    retrieved_top_k = set(retrieved_ids[:k])

    found = retrieved_top_k.intersection(gold_set)

    return len(found) / len(gold_set)


# ---------- HIT@K ----------

def hit_at_k(retrieved_ids, gold_ids, k):
    """
    Returns 1 if at least one verified relevant
    chunk appears in the top k results.

    Otherwise returns 0.
    """
    if k <= 0:
        raise ValueError("k must be greater than zero")

    gold_set = set(gold_ids)

    if not gold_set:
        raise ValueError("Gold evidence cannot be empty")

    retrieved_top_k = set(retrieved_ids[:k])

    return int(bool(retrieved_top_k.intersection(gold_set)))


# ---------- RECIPROCAL RANK ----------

def reciprocal_rank(retrieved_ids, gold_ids):
    """
    Reciprocal of the rank of the first
    verified relevant chunk.

    Returns 0 if no relevant chunk is found.
    """
    gold_set = set(gold_ids)

    if not gold_set:
        raise ValueError("Gold evidence cannot be empty")

    for rank, chunk_id in enumerate(retrieved_ids, start=1):
        if chunk_id in gold_set:
            return 1.0 / rank

    return 0.0


# ---------- MEAN RECIPROCAL RANK ----------

def mean_reciprocal_rank(all_retrieved_ids, all_gold_ids):
    """
    Average reciprocal rank across all
    benchmark questions.
    """
    if not all_retrieved_ids:
        raise ValueError("Evaluation dataset cannot be empty")

    if len(all_retrieved_ids) != len(all_gold_ids):
        raise ValueError(
            "Retrieved rankings and gold labels must match"
        )

    scores = [
        reciprocal_rank(retrieved_ids, gold_ids)
        for retrieved_ids, gold_ids in zip(
            all_retrieved_ids,
            all_gold_ids
        )
    ]

    return sum(scores) / len(scores)


# ---------- METRIC UNIT TESTS ----------

def test_metrics():
    """
    Verify metrics using simulated rankings.
    """

    test_retrieved = [
        ["gold_a", "other_1", "other_2", "other_3"],
        ["other_1", "gold_b", "other_2", "other_3"],
        ["other_1", "other_2", "other_3", "gold_c"],
    ]

    test_gold = [
        ["gold_a"],
        ["gold_b"],
        ["gold_c"],
    ]

    mrr = mean_reciprocal_rank(
        test_retrieved,
        test_gold
    )

    assert abs(mrr - (1 + 0.5 + 0.25) / 3) < 1e-9

    assert recall_at_k(
        test_retrieved[1], test_gold[1], 1
    ) == 0.0

    assert recall_at_k(
        test_retrieved[1], test_gold[1], 3
    ) == 1.0

    assert hit_at_k(
        test_retrieved[2], test_gold[2], 3
    ) == 0

    assert hit_at_k(
        test_retrieved[2], test_gold[2], 4
    ) == 1

    print("\nMetric unit tests passed!")
    print(f"MRR unit test: {mrr:.4f}")


# ---------- DISPLAY METRIC TABLE ----------

def print_metric_comparison(
    metric_name,
    metric_function,
    hybrid_ids,
    reranked_ids,
    gold_ids
):
    """
    Print Recall@K or Hit@K results
    for both retrieval methods.
    """

    print(f"\n---------- {metric_name} COMPARISON ----------")

    print(
        f"{'Metric':<15}"
        f"{'Hybrid':<15}"
        f"{'Reranked':<15}"
    )

    print("-" * 45)

    for k in TOP_K_VALUES:

        hybrid_score = metric_function(
            hybrid_ids,
            gold_ids,
            k
        )

        reranked_score = metric_function(
            reranked_ids,
            gold_ids,
            k
        )

        print(
            f"{metric_name + '@' + str(k):<15}"
            f"{hybrid_score:<15.2f}"
            f"{reranked_score:<15.2f}"
        )


# ---------- GOLD EVIDENCE RANKS ----------

def print_gold_ranks(name, retrieved_ids, gold_ids):
    """
    Display the rank of each verified
    evidence chunk in a retrieved ranking.
    """

    ranks = [
        retrieved_ids.index(gold_id) + 1
        for gold_id in gold_ids
        if gold_id in retrieved_ids
    ]

    if ranks:
        print(f"{name}: {ranks}")
    else:
        print(f"{name}: Not found in top {RETRIEVAL_LIMIT}")


# ---------- REAL RETRIEVAL EVALUATION ----------

def evaluate():

    print("\n========== COSMOS INTELLIGENCE ==========")
    print("         RETRIEVAL EVALUATION")
    print("=========================================\n")

    hybrid_all_ids = []
    reranked_all_ids = []
    all_gold_ids = []

    for example in benchmark:

        question = example["question"]
        gold_ids = example["gold_chunk_ids"]

        print("Question:", question)
        print("Expected answer:", example["answer"])
        print("Gold evidence:", gold_ids)

        # 1. Hybrid retrieval
        hybrid_results = hybrid_search(
            question,
            top_k=RETRIEVAL_LIMIT
        )

        hybrid_ids = [
            result["chunk_id"]
            for result in hybrid_results
        ]

        # 2. Cross-encoder reranking
        reranked_results = rerank(
            question,
            top_k=RETRIEVAL_LIMIT
        )

        reranked_ids = [
            result["chunk_id"]
            for result in reranked_results
        ]

        # 3. Store rankings for overall MRR
        hybrid_all_ids.append(hybrid_ids)
        reranked_all_ids.append(reranked_ids)
        all_gold_ids.append(gold_ids)

        # 4. Recall@K comparison
        print_metric_comparison(
            "Recall",
            recall_at_k,
            hybrid_ids,
            reranked_ids,
            gold_ids
        )

        # 5. Hit@K comparison
        print_metric_comparison(
            "Hit",
            hit_at_k,
            hybrid_ids,
            reranked_ids,
            gold_ids
        )

        # 6. Reciprocal rank
        hybrid_rr = reciprocal_rank(
            hybrid_ids,
            gold_ids
        )

        reranked_rr = reciprocal_rank(
            reranked_ids,
            gold_ids
        )

        print("\n---------- RECIPROCAL RANK ----------")
        print(f"Hybrid RR:   {hybrid_rr:.4f}")
        print(f"Reranked RR: {reranked_rr:.4f}")

        # 7. Gold evidence positions
        print("\n---------- GOLD EVIDENCE RANKS ----------")

        print_gold_ranks(
            "Hybrid",
            hybrid_ids,
            gold_ids
        )

        print_gold_ranks(
            "Reranked",
            reranked_ids,
            gold_ids
        )

        print("\n" + "=" * 45)

    # 8. Overall MRR
    hybrid_mrr = mean_reciprocal_rank(
        hybrid_all_ids,
        all_gold_ids
    )

    reranked_mrr = mean_reciprocal_rank(
        reranked_all_ids,
        all_gold_ids
    )

    print("\n========== OVERALL RESULTS ==========")
    print("Benchmark questions:", len(benchmark))
    print(f"Hybrid MRR:   {hybrid_mrr:.4f}")
    print(f"Reranked MRR: {reranked_mrr:.4f}")
    print("=====================================")


# ---------- RUN ----------

if __name__ == "__main__":
    test_metrics()
    evaluate()
