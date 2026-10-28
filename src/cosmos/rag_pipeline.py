
import re

from reranker import rerank
from answer_generator import AnswerGenerator
from evidence_selector import EvidenceSelector
from evidence_verifier import verify_claim_support
from scientific_normalizer import normalize_scientific_text


# ============================================================
# CONFIGURATION
# ============================================================

RETRIEVAL_TOP_K = 5
VERIFICATION_TOP_K = 3

INSUFFICIENT_EVIDENCE = "Insufficient scientific evidence."

# Prevent extremely long evidence from being sent to the LLM.
MAX_PASSAGE_CHARACTERS = 1600


# ============================================================
# TEXT UTILITIES
# ============================================================

def clean_generated_answer(answer):
    """
    Remove source markers and citation-only commentary
    before checking factual claims.
    """

    text = answer.strip()

    # Remove sentences that only discuss citation support.
    text = re.sub(
        r"(?i)\bthis is supported by (?:the )?"
        r"(?:scientific )?evidence(?: provided)?"
        r"(?: in)?\s*\[\d+\]\s*\.?",
        "",
        text,
    )

    text = re.sub(
        r"(?i)\baccording to (?:the )?"
        r"(?:provided )?evidence\s*\[\d+\]\s*[,.:]?",
        "",
        text,
    )

    # Remove citation markers such as [1], [2].
    text = re.sub(r"\[\d+\]", "", text)

    # Normalize whitespace.
    text = re.sub(r"\s+", " ", text).strip()

    return text


def split_answer_into_claims(answer):
    """
    Split the generated answer into sentences.

    Each sentence is treated as a candidate claim.
    """

    cleaned = clean_generated_answer(answer)

    sentences = re.split(
        r"(?<=[.!?])\s+",
        cleaned,
    )

    return [
        sentence.strip()
        for sentence in sentences
        if sentence.strip()
    ]


def build_evidence_context(passages):
    """
    Build numbered scientific evidence passages for Qwen.

    Each source number maps back to a retrieved chunk.
    """

    context_parts = []

    for index, passage in enumerate(passages, start=1):

        passage_text = passage["text"][
            :MAX_PASSAGE_CHARACTERS
        ]

        context_parts.append(
            f"[Source {index}]\n"
            f"Document ID: {passage['document_id']}\n"
            f"Section: {passage['section']}\n"
            f"Chunk ID: {passage['chunk_id']}\n"
            f"Evidence: {passage_text}"
        )

    return "\n\n".join(context_parts)


# ============================================================
# SCIENTIFIC RAG PIPELINE
# ============================================================

class CosmosRAGPipeline:

    def __init__(self):

        print("\nInitializing Cosmos Intelligence RAG...")

        self.generator = AnswerGenerator()
        self.selector = EvidenceSelector()

        print("\nCosmos Intelligence RAG ready!")

    # --------------------------------------------------------
    # VERIFY A CLAIM AGAINST ONE PASSAGE
    # --------------------------------------------------------

    def verify_claim_against_passage(
        self,
        claim,
        passage,
    ):

        normalized_claim = normalize_scientific_text(
            claim
        )

        normalized_evidence = normalize_scientific_text(
            passage["text"]
        )

        full_result = verify_claim_support(
            normalized_claim,
            normalized_evidence,
        )

        full_supported = (
            full_result["predicted_label"] == "entailment"
        )

        selected = self.selector.select(
            claim=normalized_claim,
            evidence_text=normalized_evidence,
            top_k=VERIFICATION_TOP_K,
        )

        sentence_results = []

        for candidate in selected:

            result = verify_claim_support(
                normalized_claim,
                candidate["sentence"],
            )

            sentence_results.append({
                "sentence": candidate["sentence"],
                "similarity": candidate["score"],
                "prediction": result["predicted_label"],
                "entailment_score": result["scores"]["entailment"],
            })

        sentence_supported = any(
            item["prediction"] == "entailment"
            for item in sentence_results
        )

        supported = full_supported or sentence_supported

        supporting_sentence = None

        for item in sentence_results:

            if item["prediction"] == "entailment":
                supporting_sentence = item["sentence"]
                break

        if supporting_sentence is None and full_supported:
            supporting_sentence = normalized_evidence

        return {
            "supported": supported,
            "full_prediction": full_result["predicted_label"],
            "supporting_sentence": supporting_sentence,
            "sentence_results": sentence_results,
        }

    # --------------------------------------------------------
    # VERIFY EACH CLAIM AGAINST ALL RETRIEVED PASSAGES
    # --------------------------------------------------------

    def verify_answer(self, answer, passages):

        claims = split_answer_into_claims(answer)

        if not claims:
            return {
                "verified": False,
                "claims": [],
            }

        claim_results = []

        for claim in claims:

            supporting_sources = []

            for source_index, passage in enumerate(
                passages,
                start=1,
            ):

                result = self.verify_claim_against_passage(
                    claim=claim,
                    passage=passage,
                )

                if result["supported"]:

                    supporting_sources.append({
                        "source_number": source_index,
                        "document_id": passage["document_id"],
                        "chunk_id": passage["chunk_id"],
                        "section": passage["section"],
                        "supporting_sentence": (
                            result["supporting_sentence"]
                        ),
                    })

            claim_results.append({
                "claim": claim,
                "supported": bool(supporting_sources),
                "supporting_sources": supporting_sources,
            })

        all_supported = all(
            result["supported"]
            for result in claim_results
        )

        return {
            "verified": all_supported,
            "claims": claim_results,
        }

    # --------------------------------------------------------
    # FORMAT VERIFIED ANSWER WITH CITATIONS
    # --------------------------------------------------------

    def format_verified_answer(self, verification):

        answer_parts = []

        for result in verification["claims"]:

            claim = result["claim"]

            source_numbers = sorted({
                source["source_number"]
                for source in result["supporting_sources"]
            })

            citations = "".join(
                f"[{number}]"
                for number in source_numbers
            )

            answer_parts.append(
                f"{claim} {citations}".strip()
            )

        return " ".join(answer_parts)

    # --------------------------------------------------------
    # ANSWER A QUESTION END TO END
    # --------------------------------------------------------

    def ask(self, question):

        if not question.strip():
            raise ValueError("Question cannot be empty.")

        print("\n" + "=" * 65)
        print("COSMOS INTELLIGENCE")
        print("=" * 65)

        print("\nQUESTION:")
        print(question)

        # ----------------------------------------------------
        # STEP 1: RETRIEVE AND RERANK
        # ----------------------------------------------------

        print("\n[1] Retrieving and reranking evidence...")

        passages = rerank(
            query=question,
            top_k=RETRIEVAL_TOP_K,
        )

        if not passages:

            print("\nNo scientific evidence retrieved.")

            return {
                "question": question,
                "answer": INSUFFICIENT_EVIDENCE,
                "verified": False,
                "status": "no_evidence",
                "sources": [],
            }

        print("\nRetrieved sources:")

        for index, passage in enumerate(
            passages,
            start=1,
        ):

            print(f"\nSource [{index}]")
            print("Chunk ID:", passage["chunk_id"])
            print("Section:", passage["section"])

            print(
                "Reranker score:",
                round(passage["reranker_score"], 4),
            )

        # ----------------------------------------------------
        # STEP 2: BUILD MULTI-PASSAGE CONTEXT
        # ----------------------------------------------------

        evidence_context = build_evidence_context(
            passages
        )

        # ----------------------------------------------------
        # STEP 3: GENERATE ANSWER
        # ----------------------------------------------------

        print("\n[2] Generating answer with Qwen3...")

        generated_answer = self.generator.generate_answer(
            question=question,
            evidence_text=evidence_context,
        )

        print("\nGENERATED ANSWER:")
        print(generated_answer)

        # ----------------------------------------------------
        # STEP 4: HANDLE INSUFFICIENT EVIDENCE
        # ----------------------------------------------------

        if (
            not generated_answer.strip()
            or INSUFFICIENT_EVIDENCE.lower()
            in generated_answer.lower()
        ):

            print("\nSTATUS: Insufficient evidence")

            return {
                "question": question,
                "answer": INSUFFICIENT_EVIDENCE,
                "generated_draft": generated_answer,
                "verified": False,
                "status": "insufficient_evidence",
                "sources": passages,
            }

        # ----------------------------------------------------
        # STEP 5: VERIFY GENERATED CLAIMS
        # ----------------------------------------------------

        print("\n[3] Verifying generated claims...")

        verification = self.verify_answer(
            answer=generated_answer,
            passages=passages,
        )

        for index, result in enumerate(
            verification["claims"],
            start=1,
        ):

            print(f"\nClaim {index}:")
            print(result["claim"])

            print("Supported:", result["supported"])

            for source in result["supporting_sources"]:

                print(
                    f"Supporting source [{source['source_number']}]:",
                    source["chunk_id"],
                )

                print(
                    "Evidence:",
                    source["supporting_sentence"],
                )

        # ----------------------------------------------------
        # STEP 6: FORMAT FINAL ANSWER
        # ----------------------------------------------------

        if verification["verified"]:

            final_answer = self.format_verified_answer(
                verification
            )

            status = "verified"

            print("\nVERIFICATION STATUS: SUPPORTED")

        else:

            final_answer = INSUFFICIENT_EVIDENCE
            status = "unverified"

            print(
                "\nVERIFICATION STATUS: "
                "COULD NOT ESTABLISH SUPPORT"
            )

        # ----------------------------------------------------
        # STEP 7: PRINT SOURCE REFERENCES
        # ----------------------------------------------------

        print("\n" + "=" * 65)
        print("FINAL ANSWER")
        print("=" * 65)

        print(final_answer)

        if verification["verified"]:

            print("\nSOURCE REFERENCES")

            cited_numbers = set()

            for claim_result in verification["claims"]:

                for source in claim_result["supporting_sources"]:

                    cited_numbers.add(
                        source["source_number"]
                    )

            for source_number in sorted(cited_numbers):

                passage = passages[source_number - 1]

                print(f"\n[{source_number}]")
                print("Document:", passage["document_id"])
                print("Section:", passage["section"])
                print("Chunk:", passage["chunk_id"])

        return {
            "question": question,
            "answer": final_answer,
            "generated_draft": generated_answer,
            "verified": verification["verified"],
            "status": status,
            "sources": passages,
            "verification": verification,
        }


# ============================================================
# TEST QUESTIONS
# ============================================================

def main():

    pipeline = CosmosRAGPipeline()

    questions = [
    "What molecule did JWST detect in WASP-39b's atmosphere?",
    "At what wavelength was the carbon dioxide feature observed?",
    "Did JWST confirm intelligent life on WASP-39b?",
    "What did the Event Horizon Telescope observe in M87?",
    "What is the estimated mass of the black hole M87*?"
]

    for question in questions:
        pipeline.ask(question)

    print("\nEnd-to-end multi-passage RAG test completed.")


if __name__ == "__main__":
    main()
