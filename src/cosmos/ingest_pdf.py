
import json
import re
from pathlib import Path
from dataclasses import asdict

import fitz

from schemas import Document, Section
from text_utils import normalize_whitespace


# ============================================================
# 1. CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DOCUMENT_ID = "m87_eht_2019"

PDF_PATH = (
    PROJECT_ROOT / "data" / "raw" / "m87_eht_2019.pdf"
)

OUTPUT_PATH = (
    PROJECT_ROOT / "data" / "processed" / "m87_sections.json"
)

PAPER_TITLE = (
    "First M87 Event Horizon Telescope Results. "
    "I. The Shadow of the Supermassive Black Hole"
)

PAPER_URL = "https://arxiv.org/abs/1906.11238"

# Minimum length of a useful text block.
MIN_BLOCK_CHARS = 25

# Blocks occupying most of the page width are treated
# as full-width content rather than column content.
WIDE_BLOCK_RATIO = 0.72


# ============================================================
# 2. TEXT CLEANING
# ============================================================

def clean_text(text):
    """
    Clean extracted text while retaining scientific
    notation as far as the PDF text layer allows.
    """

    # Rejoin words split by line-end hyphenation.
    text = re.sub(
        r"(?<=\w)-\s*\n\s*(?=\w)",
        "",
        text
    )

    # Preserve normal word boundaries.
    text = re.sub(r"\s+", " ", text)

    return normalize_whitespace(text)


# ============================================================
# 3. EXTRACT POSITIONED TEXT BLOCKS
# ============================================================

def extract_blocks(page):
    """
    Extract text blocks with their PDF coordinates.

    PyMuPDF block tuple:
        x0, y0, x1, y1, text, block_number, block_type
    """

    blocks = []

    for block in page.get_text("blocks", sort=False):

        x0, y0, x1, y1, text, _, block_type = block

        # Skip image blocks.
        if block_type != 0:
            continue

        cleaned = clean_text(text)

        if len(cleaned) < MIN_BLOCK_CHARS:
            continue

        blocks.append({
            "x0": x0,
            "y0": y0,
            "x1": x1,
            "y1": y1,
            "text": cleaned
        })

    return blocks


# ============================================================
# 4. DETECT COLUMN STRUCTURE
# ============================================================

def order_blocks(blocks, page_width):
    """
    Read full-width content and two-column content
    in a more natural order.

    Full-width blocks divide the page into vertical
    regions. Within each region, read the left column
    before the right column.
    """

    if not blocks:
        return []

    page_middle = page_width / 2

    wide_blocks = []
    column_blocks = []

    for block in blocks:

        width = block["x1"] - block["x0"]

        if width >= page_width * WIDE_BLOCK_RATIO:
            wide_blocks.append(block)
        else:
            column_blocks.append(block)

    wide_blocks.sort(
        key=lambda block: (block["y0"], block["x0"])
    )

    # Split the page into vertical regions around
    # full-width blocks.
    regions = []
    previous_y = float("-inf")

    for wide_block in wide_blocks:

        regions.append({
            "start": previous_y,
            "end": wide_block["y0"],
            "wide": None
        })

        regions.append({
            "start": wide_block["y0"],
            "end": wide_block["y1"],
            "wide": wide_block
        })

        previous_y = wide_block["y1"]

    regions.append({
        "start": previous_y,
        "end": float("inf"),
        "wide": None
    })

    ordered = []

    for region in regions:

        if region["wide"] is not None:
            ordered.append(region["wide"])
            continue

        region_blocks = []

        for block in column_blocks:

            block_middle_y = (
                block["y0"] + block["y1"]
            ) / 2

            if (
                region["start"]
                <= block_middle_y
                < region["end"]
            ):
                region_blocks.append(block)

        left = []
        right = []

        for block in region_blocks:

            block_middle_x = (
                block["x0"] + block["x1"]
            ) / 2

            if block_middle_x < page_middle:
                left.append(block)
            else:
                right.append(block)

        left.sort(
            key=lambda block: (block["y0"], block["x0"])
        )

        right.sort(
            key=lambda block: (block["y0"], block["x0"])
        )

        ordered.extend(left)
        ordered.extend(right)

    return ordered


# ============================================================
# 5. EXTRACT PAGE PARAGRAPHS
# ============================================================

def extract_page_paragraphs(page):

    blocks = extract_blocks(page)

    ordered_blocks = order_blocks(
        blocks,
        page.rect.width
    )

    paragraphs = [
        block["text"]
        for block in ordered_blocks
        if block["text"].strip()
    ]

    return paragraphs


# ============================================================
# 6. INGEST PDF
# ============================================================

def ingest_pdf():

    if not PDF_PATH.exists():
        raise FileNotFoundError(
            f"PDF not found: {PDF_PATH}"
        )

    print("\n" + "=" * 65)
    print("COSMOS INTELLIGENCE")
    print("M87 COLUMN-AWARE PDF INGESTION")
    print("=" * 65)

    sections = []
    total_paragraphs = 0

    with fitz.open(PDF_PATH) as pdf:

        total_pages = len(pdf)

        print("\nPDF pages:", total_pages)

        for page_index in range(total_pages):

            page = pdf[page_index]

            page_number = page_index + 1

            paragraphs = extract_page_paragraphs(page)

            if not paragraphs:
                print(
                    f"Page {page_number}: no usable text"
                )
                continue

            page_text = " ".join(paragraphs)

            section = Section(
                section_id=(
                    f"{DOCUMENT_ID}_page_{page_number}"
                ),
                document_id=DOCUMENT_ID,
                section_name=f"Page {page_number}",
                start_page=page_number,
                end_page=page_number,
                text=page_text,
                paragraphs=paragraphs
            )

            sections.append(section)

            total_paragraphs += len(paragraphs)

            print(
                f"Page {page_number}: "
                f"{len(paragraphs)} text blocks, "
                f"{len(page_text)} characters"
            )

    if not sections:
        raise ValueError(
            "No readable sections extracted from PDF."
        )

    document = Document(
        document_id=DOCUMENT_ID,
        title=PAPER_TITLE,
        authors=["Event Horizon Telescope Collaboration"],
        year=2019,
        source_type="research_paper",
        source_url=PAPER_URL,
        doi="10.3847/2041-8213/ab0ec7"
    )

    processed_data = {
        "document": asdict(document),
        "sections": [
            asdict(section)
            for section in sections
        ],
        "figures": []
    }

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            processed_data,
            file,
            indent=2,
            ensure_ascii=False
        )

    print("\n" + "=" * 65)
    print("INGESTION SUMMARY")
    print("=" * 65)

    print("Document:", document.document_id)
    print("Pages processed:", len(sections))
    print("Text blocks:", total_paragraphs)
    print("Saved to:", OUTPUT_PATH)

    print("\n--- PAGE 1 PREVIEW ---")

    for paragraph in sections[0].paragraphs[:5]:
        print("\n", paragraph[:350])

    print("\nIngestion completed!")


if __name__ == "__main__":
    ingest_pdf()
