
from evidence_selector import EvidenceSelector
from evidence_verifier import verify_claim_support
from scientific_normalizer import normalize_scientific_text


class HybridEvidenceVerifier:
    def __init__(self):
        self.selector = EvidenceSelector()

    def verify(self, claim, evidence_text):
        """
        Verify a scientific claim using:
        1. Full-chunk NLI
        2. NLI on the top 3 relevant evidence sentences

        Experimental decision rule:
        Support is found if any NLI check predicts entailment.
        """

        normalized_claim = normalize_scientific_text(claim)
        normalized_evidence = normalize_scientific_text(
            evidence_text
        )

        # Method A: Full evidence
        full_result = verify_claim_support(
            normalized_claim,
            normalized_evidence
        )

        full_supported = (
            full_result["predicted_label"] == "entailment"
        )

        # Method B: Retrieve top 3 evidence sentences
        selected = self.selector.select(
            claim=claim,
            evidence_text=evidence_text,
            top_k=3
        )

        sentence_results = []

        for candidate in selected:
            sentence = candidate["sentence"]

            normalized_sentence = normalize_scientific_text(
                sentence
            )

            nli_result = verify_claim_support(
                normalized_claim,
                normalized_sentence
            )

            sentence_results.append({
                "sentence": sentence,
                "semantic_similarity": candidate["score"],
                "predicted_label": nli_result["predicted_label"],
                "scores": nli_result["scores"],
                "supported": (
                    nli_result["predicted_label"] == "entailment"
                ),
            })

        selected_supported = any(
            item["supported"]
            for item in sentence_results
        )

        hybrid_supported = (
            full_supported or selected_supported
        )

        disagreement = (
            full_supported != selected_supported
        )

        # Preserve older output keys for compatibility.
        first_result = (
            sentence_results[0]
            if sentence_results
            else None
        )

        selected_result = None

        if first_result is not None:
            selected_result = {
                "predicted_label": first_result["predicted_label"],
                "scores": first_result["scores"],
            }

        # Identify the strongest supporting sentence, if any.
        supporting_sentences = [
            item for item in sentence_results
            if item["supported"]
        ]

        best_supporting_sentence = None

        if supporting_sentences:
            best_supporting_sentence = max(
                supporting_sentences,
                key=lambda item: item["scores"]["entailment"]
            )

        return {
            "supported": hybrid_supported,
            "full_result": full_result,
            "selected_result": selected_result,
            "selected_sentence": (
                first_result["sentence"]
                if first_result
                else None
            ),
            "semantic_similarity": (
                first_result["semantic_similarity"]
                if first_result
                else None
            ),
            "full_supported": full_supported,
            "selected_supported": selected_supported,
            "disagreement": disagreement,
            "normalized_claim": normalized_claim,
            "sentence_results": sentence_results,
            "best_supporting_sentence": best_supporting_sentence,
        }


if __name__ == "__main__":
    from evidence_selector import load_evidence_chunk

    verifier = HybridEvidenceVerifier()
    chunk = load_evidence_chunk()

    test_claims = [
        "JWST detected carbon dioxide in WASP-39b's atmosphere.",
        "The CO2 absorption feature was detected at 4.3 micrometres.",
        "The observations used the James Webb Space Telescope.",
        "The observations were part of the Early Release Science programme.",
        "JWST detected methane instead of carbon dioxide.",
        "The observations used the Hubble Space Telescope.",
        "WASP-39b has an atmospheric temperature of exactly 500 K.",
    ]

    print("\nCOSMOS INTELLIGENCE")
    print("TOP-3 HYBRID EVIDENCE VERIFICATION")
    print("=" * 65)

    for claim in test_claims:
        result = verifier.verify(
            claim,
            chunk["text"]
        )

        print("\n" + "-" * 65)
        print("CLAIM:", claim)
        print("NORMALIZED:", result["normalized_claim"])

        print(
            "Full evidence:",
            result["full_result"]["predicted_label"],
            "| Entailment:",
            round(
                result["full_result"]["scores"]["entailment"],
                4
            )
        )

        print("\nTOP-3 SENTENCE RESULTS:")

        for index, item in enumerate(
            result["sentence_results"],
            start=1
        ):
            print(f"\nSentence {index}:")
            print(item["sentence"])
            print("NLI:", item["predicted_label"])
            print(
                "Entailment:",
                round(item["scores"]["entailment"], 4)
            )

        print("\nFull supported:", result["full_supported"])
        print("Top-3 supported:", result["selected_supported"])
        print("Hybrid supported:", result["supported"])
        print("Methods disagree:", result["disagreement"])

        if result["best_supporting_sentence"]:
            print(
                "Best supporting sentence:",
                result["best_supporting_sentence"]["sentence"]
            )
