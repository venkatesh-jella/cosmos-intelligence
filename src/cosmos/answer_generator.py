
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

CHUNKS_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "wasp39b_token_chunks.json"
)

MODEL_NAME = "Qwen/Qwen3-0.6B"

EVIDENCE_CHUNK_ID = (
    "wasp39b_co2_2022_abstract_paragraph_token_v1_chunk_001"
)


# ============================================================
# LOAD SCIENTIFIC EVIDENCE
# ============================================================

def load_evidence():

    if not CHUNKS_PATH.exists():
        raise FileNotFoundError(
            f"Evidence file not found: {CHUNKS_PATH}"
        )

    with open(CHUNKS_PATH, "r", encoding="utf-8-sig") as file:
        chunks = json.load(file)

    if isinstance(chunks, dict):
        chunks = chunks["chunks"]

    for chunk in chunks:
        if chunk["chunk_id"] == EVIDENCE_CHUNK_ID:
            return chunk["text"]

    raise ValueError(
        f"Evidence chunk not found: {EVIDENCE_CHUNK_ID}"
    )


# ============================================================
# LLM ANSWER GENERATOR
# ============================================================

class AnswerGenerator:

    def __init__(self):

        print("Loading Qwen3 language model...")

        self.device = (
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        print(f"Device: {self.device}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            dtype=torch.float32 if self.device == "cpu"
            else torch.float16
        )

        self.model.to(self.device)
        self.model.eval()

        print("Qwen3 language model loaded!")

    def generate_answer(self, question, evidence_text):

        if not question.strip():
            raise ValueError("Question cannot be empty.")

        if not evidence_text.strip():
            return "Insufficient scientific evidence."

        system_prompt = (
            "You are Cosmos Intelligence, an astronomy "
            "research assistant. "
            "Answer the user's question using only the "
            "scientific evidence provided. "
            "Do not invent facts, numbers, or discoveries. "
            "If the evidence does not answer the question, "
            "say 'Insufficient scientific evidence.' "
            "Keep your answer clear, concise, and scientific. "
            "Use citation [1] for information supported "
            "by the provided evidence."
        )

        user_prompt = f"""
Scientific evidence [1]:
{evidence_text}

Question:
{question}

Answer:
"""

        messages = [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ]

        inputs = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
            return_tensors="pt",
            return_dict=True
        )

        inputs = {
            key: value.to(self.device)
            for key, value in inputs.items()
        }

        with torch.inference_mode():

            outputs = self.model.generate(
                **inputs,
                max_new_tokens=180,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id
            )

        input_length = inputs["input_ids"].shape[-1]

        generated_tokens = outputs[0][input_length:]

        answer = self.tokenizer.decode(
            generated_tokens,
            skip_special_tokens=True
        ).strip()

        return answer or "Insufficient scientific evidence."


# ============================================================
# TEST THE GENERATOR
# ============================================================

def main():

    print("\nCOSMOS INTELLIGENCE")
    print("LLM ANSWER GENERATION TEST")
    print("=" * 60)

    evidence_text = load_evidence()

    generator = AnswerGenerator()

    questions = [
        "What molecule did JWST detect in WASP-39b?",
        "At what wavelength was the carbon dioxide feature observed?",
        "Did the observations confirm intelligent life on WASP-39b?"
    ]

    for index, question in enumerate(questions, start=1):

        print("\n" + "-" * 60)
        print(f"QUESTION {index}: {question}")

        answer = generator.generate_answer(
            question=question,
            evidence_text=evidence_text
        )

        print("\nGENERATED ANSWER:")
        print(answer)

    print("\nGeneration test completed.")


if __name__ == "__main__":
    main()
