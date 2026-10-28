
"""
Cosmos Intelligence
Claim-Level Citation and Evidence Verification

Supports:
1. Numbered citations: [Evidence 1]
2. Chunk-ID citations: [wasp39b_co2_2022_abstract_paragraph_token_v1_chunk_001]

Each scientific claim is checked against its cited evidence.
"""

import re

from citation_verifier import verify_citations
from evidence_verifier import verify_claim_support
from claim_splitter import split_claims


def resolve_citation(citation, retrieved_chunks):
    """
    Resolve a citation to a retrieved evidence chunk.

    Supported formats:
        [Evidence 1]
        [Evidence 4]
        [actual_chunk_id]
    """

    citation = citation.strip()

    # Format 1: [Evidence N]
    numbered_match = re.fullmatch(
        r"\[\s*Evidence\s+(\d+)\s*\]",
        citation,
        flags=re.IGNORECASE,
    )

    if numbered_match:
        index = int(numbered_match.group(1)) - 1

        if 0 <= index < len(retrieved_chunks):
            return retrieved_chunks[index]

        return None

    # Format 2: [actual_chunk_id]
    chunk_id = citation.strip("[]").strip()

    for chunk in retrieved_chunks:
        if chunk.get("chunk_id") == chunk_id:
            return chunk

    return None


def verify_answer(answer, retrieved_chunks):
    """
    Verify generated scientific claims against their
    specifically cited evidence chunks.
    """

    # First, validate citations across the complete answer.
    citation_result = verify_citations(
        answer,
        retrieved_chunks,
    )

    # Split the answer into individual scientific claims.
    claims = split_claims(answer)

    claim_results = []

    for claim in claims:

        evidence_results = []
        resolved_ids = set()
        invalid_citations = []

        for citation in claim["citations"]:

            # Find the evidence chunk referenced by the citation.
            chunk = resolve_citation(
                citation,
                retrieved_chunks,
            )

            if chunk is None:
                invalid_citations.append(citation)
                continue

            chunk_id = chunk["chunk_id"]

            # Avoid checking the same chunk twice.
            if chunk_id in resolved_ids:
                continue

            resolved_ids.add(chunk_id)

            # Check whether the cited evidence supports the claim.
            result = verify_claim_support(
                claim["claim"],
                chunk["text"],
            )

            evidence_results.append(
                {
                    "chunk_id": chunk_id,
                    "predicted_label": result["predicted_label"],
                    "scores": result["scores"],
                }
            )

        # At least one cited chunk must support the claim.
        supported_by_evidence = any(
            item["predicted_label"] == "entailment"
            for item in evidence_results
        )

        # Every citation must resolve successfully.
        citations_valid = (
            bool(claim["citations"])
            and not invalid_citations
        )

        claim_results.append(
            {
                "claim_id": claim["claim_id"],
                "claim": claim["claim"],
                "citations": claim["citations"],
                "citations_valid": citations_valid,
                "invalid_citations": invalid_citations,
                "evidence_support": evidence_results,
                "supported": (
                    supported_by_evidence and citations_valid
                ),
            }
        )

    # Require every extracted claim to have model support.
    all_claims_supported = (
        bool(claim_results)
        and all(
            item["supported"]
            for item in claim_results
        )
    )

    # Require citations to pass the existing global verifier.
    all_citations_valid = citation_result[
        "all_citations_valid"
    ]

    overall_status = (
        "MODEL_SUPPORT_FOUND"
        if all_claims_supported and all_citations_valid
        else "NOT_VERIFIED"
    )

    return {
        "citation_verification": citation_result,
        "claim_results": claim_results,
        "all_claims_supported": all_claims_supported,
        "overall_status": overall_status,
    }


if __name__ == "__main__":

    sample_chunks = [
        {
            "chunk_id": "sample_chunk_001",
            "text": (
                "JWST detected carbon dioxide in "
                "WASP-39b's atmosphere. "
                "A CO2 absorption feature was observed "
                "at 4.3 micrometres with "
                "26-sigma significance."
            ),
        }
    ]

    sample_answer = (
        "JWST detected carbon dioxide in "
        "WASP-39b's atmosphere [Evidence 1]. "
        "A CO2 absorption feature was observed "
        "at 4.3 micrometres with "
        "26-sigma significance [sample_chunk_001]."
    )

    # Check both citation formats independently.
    assert (
        resolve_citation("[Evidence 1]", sample_chunks)
        == sample_chunks[0]
    )

    assert (
        resolve_citation("[sample_chunk_001]", sample_chunks)
        == sample_chunks[0]
    )

    assert (
        resolve_citation("[Evidence 99]", sample_chunks)
        is None
    )

    print("Citation resolution tests passed!")

    result = verify_answer(
        sample_answer,
        sample_chunks,
    )

    print("\nCOSMOS INTELLIGENCE - CLAIM VERIFICATION")

    for item in result["claim_results"]:

        print("\n" + "=" * 55)
        print("Claim ID:", item["claim_id"])
        print("Claim:", item["claim"])
        print("Citations:", item["citations"])
        print("Citations valid:", item["citations_valid"])
        print("Supported:", item["supported"])

        if item["invalid_citations"]:
            print(
                "Invalid citations:",
                item["invalid_citations"],
            )

        for evidence in item["evidence_support"]:

            print("Evidence chunk:", evidence["chunk_id"])

            print(
                "NLI prediction:",
                evidence["predicted_label"],
            )

            print(
                "Entailment score:",
                round(
                    evidence["scores"]["entailment"],
                    4,
                ),
            )

    print("\nOverall status:", result["overall_status"])
