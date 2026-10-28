
import os

from google import genai

from reranker import rerank
from augmentation import build_augmented_prompt
from verification_pipeline import verify_answer


# Gemini model
MODEL_NAME = "gemini-3.5-flash-lite"


def generate_answer(question, top_k=5):
    """
    Complete Cosmos Intelligence RAG pipeline.

    1. Retrieve and rerank scientific evidence.
    2. Augment the question with evidence.
    3. Generate an answer using Gemini.
    4. Verify citations and evidence support.
    5. Return the answer and verification results.
    """

    # STEP 1: Check Gemini API key
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY is not configured."
        )

    if not question.strip():
        raise ValueError("Question cannot be empty.")

    # STEP 2: Retrieve and rerank evidence
    print("\nRetrieving scientific evidence...")

    retrieved_chunks = rerank(
        question,
        top_k=top_k
    )

    if not retrieved_chunks:
        return {
            "question": question,
            "answer": "No scientific evidence was found.",
            "verification": None,
        }

    # STEP 3: Build augmented prompt
    prompt = build_augmented_prompt(
        question,
        retrieved_chunks
    )

    # STEP 4: Generate answer with Gemini
    print("Generating answer with Gemini...")

    client = genai.Client(api_key=api_key)

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt
    )

    if not response.text:
        raise RuntimeError(
            "Gemini returned an empty response."
        )

    answer = response.text.strip()

    # STEP 5: Verify citations and scientific evidence
    print("Verifying citations and evidence support...")

    verification = verify_answer(
        answer,
        retrieved_chunks
    )

    # STEP 6: Return complete result
    return {
        "question": question,
        "answer": answer,
        "verification": verification,
    }


if __name__ == "__main__":

    question = (
        "What molecule did JWST detect in "
        "WASP-39b's atmosphere?"
    )

    print("\n" + "=" * 60)
    print("COSMOS INTELLIGENCE")
    print("=" * 60)

    print("\nQUESTION:")
    print(question)

    result = generate_answer(question)

    print("\n" + "=" * 60)
    print("COSMOS INTELLIGENCE ANSWER")
    print("=" * 60)

    print(result["answer"])

    print("\n" + "=" * 60)
    print("VERIFICATION RESULTS")
    print("=" * 60)

    verification = result["verification"]

    if verification is None:
        print("No scientific evidence retrieved.")

    else:
        citation_result = verification[
            "citation_verification"
        ]

        print(
            "\nCitations valid:",
            citation_result["all_citations_valid"]
        )

        print(
            "Valid citations:",
            citation_result["valid_citations"]
        )

        print(
            "Invalid citations:",
            citation_result["invalid_citations"]
        )

        print(
            "\nOverall status:",
            verification["overall_status"]
        )


        print("\nCLAIM-LEVEL VERIFICATION:")

        for item in verification["claim_results"]:

            print("\n" + "-" * 55)
            print("Claim ID:", item["claim_id"])
            print("Claim:", item["claim"])
            print("Citations:", item["citations"])
            print("Supported:", item["supported"])

            if not item["citations_valid"]:
                print("Citation issue detected.")

            for evidence in item["evidence_support"]:

                print("Evidence chunk:", evidence["chunk_id"])
                print("NLI prediction:", evidence["predicted_label"])

                print(
                    "Entailment score:",
                    round(evidence["scores"]["entailment"], 4)
                )

                print(
                    "Neutral score:",
                    round(evidence["scores"]["neutral"], 4)
                )

                print(
                    "Contradiction score:",
                    round(evidence["scores"]["contradiction"], 4)
                )
