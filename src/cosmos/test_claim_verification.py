
"""
Cosmos Intelligence
Automated Claim-Level Verification Tests

Tests:
1. Supported scientific claim
2. Contradictory scientific claim
3. Unsupported scientific claim
4. Mixed correct and incorrect claims
5. Missing citation
6. Invalid citation
7. Chunk-ID citation
"""

from verification_pipeline import verify_answer


CHUNK_ID = "wasp39b_co2_2022_abstract_paragraph_token_v1_chunk_001"

# Controlled evidence makes test outcomes easier to interpret.
TEST_CHUNKS = [
    {
        "chunk_id": CHUNK_ID,
        "text": (
            "JWST detected carbon dioxide in WASP-39b's atmosphere. "
            "A carbon dioxide absorption feature was observed "
            "at 4.3 micrometres with 26-sigma significance."
        ),
    }
]


TEST_CASES = [
    {
        "name": "Supported claim",
        "answer": (
            "JWST detected carbon dioxide in "
            "WASP-39b's atmosphere [Evidence 1]."
        ),
        "expected_supported": True,
    },
    {
        "name": "Contradictory claim",
        "answer": (
            "JWST did not detect carbon dioxide in "
            "WASP-39b's atmosphere [Evidence 1]."
        ),
        "expected_supported": False,
    },
    {
        "name": "Unsupported claim",
        "answer": (
            "WASP-39b has an atmospheric temperature "
            "of exactly 500 K [Evidence 1]."
        ),
        "expected_supported": False,
    },
    {
        "name": "Mixed correct and incorrect claims",
        "answer": (
            "JWST detected carbon dioxide in "
            "WASP-39b's atmosphere [Evidence 1]. "
            "JWST did not detect carbon dioxide in "
            "WASP-39b's atmosphere [Evidence 1]."
        ),
        "expected_supported": False,
    },
    {
        "name": "Missing citation",
        "answer": (
            "JWST detected carbon dioxide in "
            "WASP-39b's atmosphere."
        ),
        "expected_supported": False,
    },
    {
        "name": "Invalid citation",
        "answer": (
            "JWST detected carbon dioxide in "
            "WASP-39b's atmosphere [Evidence 99]."
        ),
        "expected_supported": False,
    },
    {
        "name": "Chunk-ID citation",
        "answer": (
            "JWST detected carbon dioxide in "
            f"WASP-39b's atmosphere [{CHUNK_ID}]."
        ),
        "expected_supported": True,
    },
]


def run_tests():
    passed = 0
    failed = 0

    print("\nCOSMOS INTELLIGENCE")
    print("AUTOMATED CLAIM VERIFICATION TEST SUITE")
    print("=" * 65)

    for index, test in enumerate(TEST_CASES, start=1):

        print(f"\nTEST {index}: {test['name']}")
        print("-" * 65)

        try:
            result = verify_answer(
                test["answer"],
                TEST_CHUNKS,
            )

            actual_supported = (
                result["overall_status"] == "MODEL_SUPPORT_FOUND"
            )

            expected_supported = test["expected_supported"]

            success = actual_supported == expected_supported

            print("Expected support:", expected_supported)
            print("Actual support:", actual_supported)
            print("Overall status:", result["overall_status"])

            for claim in result["claim_results"]:
                print(
                    f"Claim {claim['claim_id']}: "
                    f"supported={claim['supported']}, "
                    f"citations_valid={claim['citations_valid']}"
                )

                for evidence in claim["evidence_support"]:
                    print(
                        "  NLI:",
                        evidence["predicted_label"],
                        "| Entailment:",
                        round(
                            evidence["scores"]["entailment"],
                            4,
                        ),
                    )

            if success:
                passed += 1
                print("RESULT: PASS")
            else:
                failed += 1
                print("RESULT: FAIL")

        except Exception as error:
            failed += 1
            print("RESULT: ERROR")
            print(type(error).__name__ + ":", error)

    total = len(TEST_CASES)

    print("\n" + "=" * 65)
    print("FINAL TEST RESULTS")
    print("=" * 65)
    print("Total tests:", total)
    print("Passed:", passed)
    print("Failed:", failed)
    print(f"Pass rate: {passed / total * 100:.1f}%")

    if failed:
        print("\nSome tests failed. Review the diagnostics above.")
    else:
        print("\nAll tests passed on this controlled test suite.")


if __name__ == "__main__":
    run_tests()
