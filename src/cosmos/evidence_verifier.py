
from sentence_transformers import CrossEncoder


MODEL_NAME = "cross-encoder/nli-deberta-v3-small"

print("Loading evidence verification model...")

nli_model = CrossEncoder(
    MODEL_NAME,
    device="cpu",
    max_length=512
)

LABELS = nli_model.model.config.id2label

print("Evidence verification model loaded!")


def verify_claim_support(claim, evidence):
    """
    Predict whether a scientific passage supports a claim.

    Returns NLI classification scores and a preliminary
    support decision. This is not proof of factual accuracy.
    """

    if not claim.strip():
        raise ValueError("Claim cannot be empty.")

    if not evidence.strip():
        raise ValueError("Evidence cannot be empty.")

    probabilities = nli_model.predict(
        [(evidence, claim)],
        apply_softmax=True
    )[0]

    scores = {
        str(LABELS[index]).lower(): float(probability)
        for index, probability in enumerate(probabilities)
    }

    predicted_label = max(scores, key=scores.get)

    return {
        "claim": claim,
        "predicted_label": predicted_label,
        "scores": scores,
        "model_predicts_support": predicted_label == "entailment"
    }


if __name__ == "__main__":

    evidence = (
        "The James Webb Space Telescope detected carbon "
        "dioxide in the atmosphere of WASP-39b."
    )

    claims = [
        "JWST detected carbon dioxide in WASP-39b's atmosphere.",
        "JWST detected methane instead of carbon dioxide.",
        "WASP-39b has an atmospheric temperature of 500 K."
    ]

    for claim in claims:
        result = verify_claim_support(claim, evidence)

        print("\nCLAIM:", result["claim"])
        print("PREDICTION:", result["predicted_label"])
        print("SCORES:", result["scores"])
        print(
            "MODEL PREDICTS SUPPORT:",
            result["model_predicts_support"]
        )
