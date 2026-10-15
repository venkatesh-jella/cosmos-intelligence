
import json
import re
from pathlib import Path
from dataclasses import asdict

from transformers import AutoTokenizer
from schemas import Chunk


# ==================================================
# 1. CONFIGURATION
# ==================================================

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

WORD_LIMIT = 512
TOKEN_LIMIT = 240

INPUT_PATH = Path("data/processed/wasp39b_sections.json")

WORD_OUTPUT_PATH = Path(
    "data/processed/wasp39b_chunks.json"
)

TOKEN_OUTPUT_PATH = Path(
    "data/processed/wasp39b_token_chunks.json"
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


# ==================================================
# 2. TEXT UTILITIES
# ==================================================

def count_words(text):
    return len(text.split())


def count_tokens(text):
    return len(
        tokenizer.encode(
            text,
            add_special_tokens=True,
            truncation=False
        )
    )


def normalize_text(text):
    return " ".join(text.split())


def split_into_sentences(paragraph):
    return [
        sentence.strip()
        for sentence in re.split(
            r"(?<=[.!?])\s+",
            paragraph.strip()
        )
        if sentence.strip()
    ]


# ==================================================
# 3. ORIGINAL WORD-BASED CHUNKING
# ==================================================

def split_long_sentence_words(sentence, max_words):
    words = sentence.split()

    return [
        " ".join(words[i:i + max_words])
        for i in range(0, len(words), max_words)
    ]


def split_long_paragraph_words(paragraph, max_words):
    sentences = split_into_sentences(paragraph)

    pieces = []
    current = []

    for sentence in sentences:

        if count_words(sentence) > max_words:

            if current:
                pieces.append(" ".join(current))
                current = []

            pieces.extend(
                split_long_sentence_words(
                    sentence,
                    max_words
                )
            )
            continue

        candidate = " ".join(current + [sentence])

        if current and count_words(candidate) > max_words:
            pieces.append(" ".join(current))
            current = [sentence]
        else:
            current.append(sentence)

    if current:
        pieces.append(" ".join(current))

    return pieces


def chunk_paragraphs(paragraphs, max_words=512):
    """
    Original word-based chunker.
    Kept as our baseline.
    """

    if max_words <= 0:
        raise ValueError("Word limit must be positive")

    chunks = []
    current = []

    for paragraph in paragraphs:

        paragraph = normalize_text(paragraph)

        if not paragraph:
            continue

        if count_words(paragraph) > max_words:
            pieces = split_long_paragraph_words(
                paragraph,
                max_words
            )
        else:
            pieces = [paragraph]

        for piece in pieces:

            candidate = " ".join(current + [piece])

            if current and count_words(candidate) > max_words:
                chunks.append(" ".join(current))
                current = [piece]
            else:
                current.append(piece)

    if current:
        chunks.append(" ".join(current))

    return chunks


# ==================================================
# 4. TOKEN-AWARE SENTENCE SPLITTING
# ==================================================

def split_long_sentence_tokens(sentence, max_tokens):
    """
    Fallback for a sentence longer than the token limit.

    Prefer splitting at word boundaries.
    Never silently truncate text.
    """

    words = sentence.split()

    pieces = []
    current = []

    for word in words:

        if count_tokens(word) > max_tokens:
            raise ValueError(
                "A single word exceeds the token limit: "
                + word[:100]
            )

        candidate = " ".join(current + [word])

        if current and count_tokens(candidate) > max_tokens:
            pieces.append(" ".join(current))
            current = [word]
        else:
            current.append(word)

    if current:
        pieces.append(" ".join(current))

    return pieces


def split_paragraph_by_tokens(paragraph, max_tokens=240):
    """
    Split paragraphs using actual tokenizer counts.

    Prefer sentence boundaries.
    Fall back to word boundaries for long sentences.
    """

    if max_tokens < 2:
        raise ValueError("Token limit must be at least 2")

    sentences = split_into_sentences(paragraph)

    pieces = []
    current = []

    for sentence in sentences:

        if count_tokens(sentence) > max_tokens:

            if current:
                pieces.append(" ".join(current))
                current = []

            smaller_pieces = split_long_sentence_tokens(
                sentence,
                max_tokens
            )

            pieces.extend(smaller_pieces)
            continue

        candidate = " ".join(current + [sentence])

        if current and count_tokens(candidate) > max_tokens:
            pieces.append(" ".join(current))
            current = [sentence]
        else:
            current.append(sentence)

    if current:
        pieces.append(" ".join(current))

    return pieces


# ==================================================
# 5. TOKEN-AWARE PARAGRAPH CHUNKING
# ==================================================

def chunk_paragraphs_token_aware(
    paragraphs,
    max_tokens=240
):
    """
    Combine adjacent paragraphs when possible.

    Split oversized paragraphs into smaller pieces.

    All resulting chunks must fit the token limit.
    """

    if max_tokens < 2:
        raise ValueError("Token limit must be at least 2")

    chunks = []
    current = []

    for paragraph in paragraphs:

        paragraph = normalize_text(paragraph)

        if not paragraph:
            continue

        if count_tokens(paragraph) > max_tokens:

            pieces = split_paragraph_by_tokens(
                paragraph,
                max_tokens
            )

        else:
            pieces = [paragraph]

        for piece in pieces:

            if count_tokens(piece) > max_tokens:
                raise ValueError(
                    "A piece exceeds the token limit"
                )

            candidate = " ".join(current + [piece])

            if current and count_tokens(candidate) > max_tokens:
                chunks.append(" ".join(current))
                current = [piece]
            else:
                current.append(piece)

    if current:
        chunks.append(" ".join(current))

    return chunks


# ==================================================
# 6. CREATE STRUCTURED CHUNK RECORDS
# ==================================================

def build_chunk_records(
    sections,
    strategy,
    limit,
    config_id
):
    """
    Generate structured Chunk objects with metadata.
    """

    all_records = []

    for section in sections:

        paragraphs = section["paragraphs"]

        if strategy == "word":
            texts = chunk_paragraphs(
                paragraphs,
                max_words=limit
            )

        elif strategy == "token":
            texts = chunk_paragraphs_token_aware(
                paragraphs,
                max_tokens=limit
            )

        else:
            raise ValueError(
                "Unknown chunking strategy: " + strategy
            )

        # Check that no text was lost during chunking
        original_text = normalize_text(
            " ".join(paragraphs)
        )

        reconstructed_text = normalize_text(
            " ".join(texts)
        )

        assert original_text == reconstructed_text, (
            "Text preservation failed in section: "
            + section["section_name"]
        )

        print(
            f"{section['section_name']} | "
            f"Paragraphs: {len(paragraphs)} | "
            f"Chunks: {len(texts)}"
        )

        for i, chunk_text in enumerate(texts, start=1):

            actual_tokens = count_tokens(chunk_text)

            chunk = Chunk(
                chunk_id=(
                    f"{section['section_id']}_"
                    f"{config_id}_chunk_{i:03d}"
                ),
                document_id=section["document_id"],
                section_id=section["section_id"],
                section_name=section["section_name"],
                text=chunk_text,
                start_page=section.get("start_page"),
                end_page=section.get("end_page"),
                token_count=actual_tokens,
                chunking_config_id=config_id
            )

            all_records.append(chunk)

    return all_records


# ==================================================
# 7. VALIDATION AND SAVING
# ==================================================

def validate_chunks(chunks, strategy, limit):

    ids = [chunk.chunk_id for chunk in chunks]

    assert len(ids) == len(set(ids)), (
        "Duplicate chunk IDs found"
    )

    assert all(
        chunk.text.strip()
        for chunk in chunks
    ), "Empty chunk found"

    assert all(
        chunk.token_count == count_tokens(chunk.text)
        for chunk in chunks
    ), "Incorrect token count metadata"

    if strategy == "word":

        assert all(
            count_words(chunk.text) <= limit
            for chunk in chunks
        ), "Word limit exceeded"

    elif strategy == "token":

        assert all(
            chunk.token_count <= limit
            for chunk in chunks
        ), "Token limit exceeded"

    print("Unique IDs: Passed")
    print("Nonempty chunks: Passed")
    print("Token metadata: Passed")
    print("Chunk size limit: Passed")
    print("Text preservation: Checked during creation")


def save_chunks(chunks, output_path):

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(
            [asdict(chunk) for chunk in chunks],
            f,
            indent=2,
            ensure_ascii=False
        )

    print("Saved to:", output_path)
    print("Total saved chunks:", len(chunks))


# ==================================================
# 8. MAIN PIPELINE
# ==================================================

if __name__ == "__main__":

    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        paper_data = json.load(f)

    sections = paper_data["sections"]

    print("\n=== COSMOS INTELLIGENCE CHUNKING ===")
    print("Total sections:", len(sections))

    # ----------------------------------------------
    # TOKENIZER TEST
    # ----------------------------------------------

    print("\n--- TOKENIZER TEST ---")

    sample = "JWST detected carbon dioxide in WASP-39b."

    print("Words:", count_words(sample))
    print("Tokens:", count_tokens(sample))

    # ----------------------------------------------
    # WORD-BASED BASELINE
    # ----------------------------------------------

    print("\n--- WORD-BASED CHUNKING ---")

    word_chunks = build_chunk_records(
        sections=sections,
        strategy="word",
        limit=WORD_LIMIT,
        config_id="paragraph_word_v1"
    )

    validate_chunks(
        word_chunks,
        strategy="word",
        limit=WORD_LIMIT
    )

    save_chunks(
        word_chunks,
        WORD_OUTPUT_PATH
    )

    # ----------------------------------------------
    # TOKEN-AWARE CHUNKING
    # ----------------------------------------------

    print("\n--- TOKEN-AWARE CHUNKING ---")

    token_chunks = build_chunk_records(
        sections=sections,
        strategy="token",
        limit=TOKEN_LIMIT,
        config_id="paragraph_token_v1"
    )

    validate_chunks(
        token_chunks,
        strategy="token",
        limit=TOKEN_LIMIT
    )

    save_chunks(
        token_chunks,
        TOKEN_OUTPUT_PATH
    )

    # ----------------------------------------------
    # OVERSIZED PARAGRAPH TEST
    # ----------------------------------------------

    print("\n--- OVERSIZED PARAGRAPH TEST ---")

    long_paragraph = (
        "JWST detected carbon dioxide. " * 150
    )

    test_chunks = chunk_paragraphs_token_aware(
        [long_paragraph],
        max_tokens=TOKEN_LIMIT
    )

    for i, text in enumerate(test_chunks, start=1):
        print(
            f"Test chunk {i}: "
            f"{count_tokens(text)} tokens"
        )

    assert all(
        count_tokens(text) <= TOKEN_LIMIT
        for text in test_chunks
    )

    assert normalize_text(
        " ".join(test_chunks)
    ) == normalize_text(long_paragraph)

    print("Oversized paragraph test passed!")

    # ----------------------------------------------
    # FINAL SUMMARY
    # ----------------------------------------------

    print("\n=== FINAL SUMMARY ===")

    print("Word-based chunks:", len(word_chunks))
    print("Token-aware chunks:", len(token_chunks))

    print(
        "Largest token-aware chunk:",
        max(chunk.token_count for chunk in token_chunks),
        "tokens"
    )

    print("\nAll validations passed!")
