
import json
import re
from pathlib import Path
from dataclasses import asdict

from transformers import AutoTokenizer
from schemas import Chunk


# --------------------------------------------------
# 1. TOKENIZER SETUP
# --------------------------------------------------

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


def count_tokens(text):
    """Count actual tokens, including special tokens."""
    return len(
        tokenizer.encode(
            text,
            add_special_tokens=True,
            truncation=False
        )
    )


def count_words(text):
    """Count words for our existing prototype chunker."""
    return len(text.split())


# --------------------------------------------------
# 2. SENTENCE SPLITTING
# --------------------------------------------------

def split_into_sentences(paragraph):
    """Split at punctuation followed by whitespace."""
    return [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+", paragraph.strip())
        if sentence.strip()
    ]


def split_paragraph_by_tokens(paragraph, max_tokens=240):
    """
    Split a paragraph into pieces using actual token counts.

    Prefer sentence boundaries.
    Raise an error if one sentence exceeds the limit.
    """
    if max_tokens <= 0:
        raise ValueError("max_tokens must be greater than zero")

    sentences = split_into_sentences(paragraph)

    pieces = []
    current_sentences = []

    for sentence in sentences:

        if count_tokens(sentence) > max_tokens:
            raise ValueError(
                "A sentence exceeds the token limit."
            )

        candidate = " ".join(current_sentences + [sentence])

        if current_sentences and count_tokens(candidate) > max_tokens:
            pieces.append(" ".join(current_sentences))
            current_sentences = [sentence]
        else:
            current_sentences.append(sentence)

    if current_sentences:
        pieces.append(" ".join(current_sentences))

    return pieces


# --------------------------------------------------
# 3. EXISTING WORD-BASED SPLITTING
# --------------------------------------------------

def split_long_sentence(sentence, max_words):
    """Fallback for a sentence longer than the word limit."""
    words = sentence.split()

    return [
        " ".join(words[i:i + max_words])
        for i in range(0, len(words), max_words)
    ]


def split_long_paragraph(paragraph, max_words):
    """Split oversized paragraphs, preferring sentence boundaries."""
    sentences = split_into_sentences(paragraph)

    pieces = []
    current_sentences = []
    current_words = 0

    for sentence in sentences:
        sentence_words = count_words(sentence)

        if sentence_words > max_words:
            if current_sentences:
                pieces.append(" ".join(current_sentences))
                current_sentences = []
                current_words = 0

            pieces.extend(
                split_long_sentence(sentence, max_words)
            )
            continue

        if current_sentences and current_words + sentence_words > max_words:
            pieces.append(" ".join(current_sentences))
            current_sentences = []
            current_words = 0

        current_sentences.append(sentence)
        current_words += sentence_words

    if current_sentences:
        pieces.append(" ".join(current_sentences))

    return pieces


# --------------------------------------------------
# 4. EXISTING PARAGRAPH-AWARE CHUNKING
# --------------------------------------------------

def chunk_paragraphs(paragraphs, max_tokens=512):
    """
    Group neighboring paragraphs into chunks.

    NOTE:
    max_tokens still represents a WORD limit here.
    We'll convert this function to token-aware chunking next.
    """
    if max_tokens <= 0:
        raise ValueError("max_tokens must be greater than zero")

    chunks = []
    current_chunk = []
    current_words = 0

    for paragraph in paragraphs:
        paragraph = paragraph.strip()

        if not paragraph:
            continue

        if count_words(paragraph) > max_tokens:
            pieces = split_long_paragraph(
                paragraph,
                max_tokens
            )
        else:
            pieces = [paragraph]

        for piece in pieces:
            piece_words = count_words(piece)

            if current_chunk and current_words + piece_words > max_tokens:
                chunks.append(" ".join(current_chunk))
                current_chunk = []
                current_words = 0

            current_chunk.append(piece)
            current_words += piece_words

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


# --------------------------------------------------
# 5. MAIN PIPELINE
# --------------------------------------------------

if __name__ == "__main__":

    # Load processed scientific paper
    json_path = Path("data/processed/wasp39b_sections.json")

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    print("Available keys:", data.keys())

    sections = data["sections"]

    print("Total sections:", len(sections))

    # Tokenizer test
    print("\n--- TOKENIZER TEST ---")

    sample = "JWST detected carbon dioxide in WASP-39b."

    print("Words:", count_words(sample))
    print("Tokens:", count_tokens(sample))

    total_chunks = 0
    all_chunks = []

    # Process each section independently
    for section in sections:

        section_name = section["section_name"]
        paragraphs = section["paragraphs"]

        chunks = chunk_paragraphs(
            paragraphs,
            max_tokens=512
        )

        structured_chunks = []

        for i, chunk_text in enumerate(chunks, start=1):

            chunk_record = {
                "chunk_id": f"{section['section_id']}_chunk_{i:03d}",
                "document_id": section["document_id"],
                "section_id": section["section_id"],
                "section_name": section_name,
                "text": chunk_text,
                "start_page": section.get("start_page"),
                "end_page": section.get("end_page"),
                "token_count": None,
                "chunking_config_id": "paragraph_v1"
            }

            structured_chunks.append(
                Chunk(**chunk_record)
            )

        all_chunks.extend(structured_chunks)

        print(f"\n--- {section_name} ---")
        print("Paragraphs:", len(paragraphs))
        print("Chunks created:", len(chunks))

        for i, chunk in enumerate(chunks, start=1):
            print(f"Chunk {i}: {count_words(chunk)} words")

        total_chunks += len(chunks)

    print("\nTotal paper chunks:", total_chunks)

    # Save chunks
    assert len(all_chunks) == total_chunks

    output_path = Path("data/processed/wasp39b_chunks.json")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(
            [asdict(chunk) for chunk in all_chunks],
            f,
            indent=2,
            ensure_ascii=False
        )

    print("Saved chunks to:", output_path)
    print("Total saved chunks:", len(all_chunks))

    # --------------------------------------------------
    # 6. CHUNK VALIDATION
    # --------------------------------------------------

    print("\n--- CHUNK VALIDATION ---")

    with open(output_path, "r", encoding="utf-8") as f:
        saved_chunks = json.load(f)

    chunk_ids = [
        chunk["chunk_id"]
        for chunk in saved_chunks
    ]

    assert len(saved_chunks) == total_chunks
    assert len(chunk_ids) == len(set(chunk_ids))
    assert all(chunk["text"].strip() for chunk in saved_chunks)
    assert all(count_words(chunk["text"]) <= 512 for chunk in saved_chunks)

    print("All chunk IDs are unique.")
    print("No empty chunks found.")
    print("All chunks are within the 512-word limit.")
    print("Validation passed!")

    # --------------------------------------------------
    # 7. OVERSIZED PARAGRAPH TEST (WORD-BASED)
    # --------------------------------------------------

    print("\n--- OVERSIZED PARAGRAPH TEST ---")

    long_paragraph = "JWST detected carbon dioxide. " * 150

    print("Original words:", count_words(long_paragraph))

    sentences = split_into_sentences(long_paragraph)

    print("Total sentences:", len(sentences))
    print("First sentence:", sentences[0])
    print("Last sentence:", sentences[-1])

    test_chunks = chunk_paragraphs(
        [long_paragraph],
        max_tokens=100
    )

    print("Word-based chunks created:", len(test_chunks))

    for i, chunk in enumerate(test_chunks, start=1):
        print(f"Chunk {i}: {count_words(chunk)} words")

    assert all(count_words(chunk) <= 100 for chunk in test_chunks)
    assert sum(count_words(chunk) for chunk in test_chunks) == 600

    print("Oversized paragraph test passed!")

    # --------------------------------------------------
    # 8. NEW TOKEN-AWARE SPLITTING TEST
    # --------------------------------------------------

    print("\n--- TOKEN-AWARE SPLITTING TEST ---")

    token_pieces = split_paragraph_by_tokens(
        long_paragraph,
        max_tokens=240
    )

    for i, piece in enumerate(token_pieces, start=1):
        print(f"Piece {i}: {count_tokens(piece)} tokens")

    assert all(
        count_tokens(piece) <= 240
        for piece in token_pieces
    )

    # Check that all words are preserved
    assert sum(
        count_words(piece)
        for piece in token_pieces
    ) == 600

    print("Token-aware splitting test passed!")
