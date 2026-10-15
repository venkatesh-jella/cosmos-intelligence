
import json
from pathlib import Path
import xml.etree.ElementTree as ET
from dataclasses import asdict

from schemas import Document, Section, Figure
from text_utils import extract_scientific_text, normalize_whitespace


# --------------------------------------------------
# 1. File paths and document information
# --------------------------------------------------

DOCUMENT_ID = "wasp39b_co2_2022"

XML_PATH = Path(
    "data/raw/papers/wasp39b_co2_2022/article.xml"
)

OUTPUT_PATH = Path(
    "data/processed/wasp39b_sections.json"
)

JATS = {
    "j": "https://jats.nlm.nih.gov/ns/archiving/1.4/"
}

OAI = {
    "oai": "http://www.openarchives.org/OAI/2.0/"
}


# --------------------------------------------------
# 2. Helper functions
# --------------------------------------------------

def local_tag(element):
    """Return the tag name without its XML namespace."""
    return element.tag.split("}")[-1]


def extract_paragraphs(element):
    """Extract paragraphs while preserving scientific notation."""
    paragraphs = []

    for p in element.iter():
        if local_tag(p) == "p":
            paragraph_text = normalize_whitespace(
                extract_scientific_text(p)
            )

            if paragraph_text:
                paragraphs.append(paragraph_text)

    return paragraphs


def find_direct_child(element, tag_name):
    """Find a direct child regardless of its namespace."""
    for child in element:
        if local_tag(child) == tag_name:
            return child

    return None


# --------------------------------------------------
# 3. Load the JATS XML article
# --------------------------------------------------

tree = ET.parse(XML_PATH)
root = tree.getroot()

get_record = root.find("oai:GetRecord", OAI)

if get_record is None:
    raise ValueError("OAI GetRecord element not found")

record = get_record.find("oai:record", OAI)

if record is None:
    raise ValueError("OAI record not found")

metadata = record.find("oai:metadata", OAI)

if metadata is None or len(metadata) == 0:
    raise ValueError("Article metadata not found")

article = next(iter(metadata))


# --------------------------------------------------
# 4. Extract the article title
# --------------------------------------------------

title_element = article.find(
    "j:front/j:article-meta/j:title-group/j:article-title",
    JATS
)

if title_element is None:
    raise ValueError("Article title not found")

title = normalize_whitespace(
    extract_scientific_text(title_element)
)

print("Paper title:", title)


# --------------------------------------------------
# 5. Extract the abstract
# --------------------------------------------------

abstract_element = article.find(
    "j:front/j:article-meta/j:abstract",
    JATS
)

if abstract_element is None:
    raise ValueError("Abstract not found")

abstract_text = normalize_whitespace(
    extract_scientific_text(abstract_element)
)

abstract_paragraphs = extract_paragraphs(
    abstract_element
)


# --------------------------------------------------
# 6. Create the Document object
# --------------------------------------------------

document = Document(
    document_id=DOCUMENT_ID,
    title=title,
    authors=[],
    year=2022,
    source_type="research_paper",
    source_url="https://pmc.ncbi.nlm.nih.gov/articles/PMC9946830/",
    doi="10.1038/s41586-022-05269-w"
)


# --------------------------------------------------
# 7. Create the Abstract section
# --------------------------------------------------

abstract_section = Section(
    section_id=f"{DOCUMENT_ID}_abstract",
    document_id=DOCUMENT_ID,
    section_name="Abstract",
    start_page=None,
    end_page=None,
    text=abstract_text,
    paragraphs=abstract_paragraphs
)

sections = [abstract_section]


# --------------------------------------------------
# 8. Locate the article body
# --------------------------------------------------

body = article.find("j:body", JATS)

if body is None:
    raise ValueError("Article body not found")


# --------------------------------------------------
# 9. Extract Main and Methods sections
# --------------------------------------------------

for section in body.findall("j:sec", JATS):

    heading = section.find("j:title", JATS)

    if heading is not None:
        section_name = normalize_whitespace(
            extract_scientific_text(heading)
        )
    else:
        section_name = "Untitled"

    if section_name not in ["Main", "Methods"]:
        continue

    section_text = normalize_whitespace(
        extract_scientific_text(section)
    )

    section_paragraphs = extract_paragraphs(
        section
    )

    new_section = Section(
        section_id=f"{DOCUMENT_ID}_{section_name.lower()}",
        document_id=DOCUMENT_ID,
        section_name=section_name,
        start_page=None,
        end_page=None,
        text=section_text,
        paragraphs=section_paragraphs
    )

    sections.append(new_section)


# --------------------------------------------------
# 10. Extract figures and captions
# --------------------------------------------------

figures = [
    element
    for element in article.iter()
    if local_tag(element) == "fig"
]

figure_records = []

for fig in figures:

    label = find_direct_child(fig, "label")
    caption = find_direct_child(fig, "caption")

    label_text = (
        normalize_whitespace(
            extract_scientific_text(label)
        )
        if label is not None
        else "No label"
    )

    caption_text = (
        normalize_whitespace(
            extract_scientific_text(caption)
        )
        if caption is not None
        else "No caption"
    )

    figure_record = Figure(
        figure_id=f"{DOCUMENT_ID}_fig{len(figure_records) + 1}",
        document_id=DOCUMENT_ID,
        label=label_text,
        caption=caption_text
    )

    figure_records.append(figure_record)


# --------------------------------------------------
# 11. Save processed data to JSON
# --------------------------------------------------

processed_data = {
    "document": asdict(document),
    "sections": [
        asdict(section)
        for section in sections
    ],
    "figures": [
        asdict(figure)
        for figure in figure_records
    ]
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


# --------------------------------------------------
# 12. Print ingestion summary
# --------------------------------------------------

print("\n--- INGESTION SUMMARY ---")

print("Document:", document.title)
print("Document ID:", document.document_id)

print("\nTotal sections:", len(sections))

for section in sections:
    print(
        f"{section.section_name} - "
        f"{len(section.text)} characters"
    )


# --------------------------------------------------
# 13. Paragraph statistics
# --------------------------------------------------

print("\n--- SECTION PARAGRAPH COUNTS ---")

for section in sections:
    print(
        section.section_name,
        len(section.paragraphs)
    )

all_paragraphs = [
    paragraph
    for section in sections
    for paragraph in section.paragraphs
]

if all_paragraphs:

    lengths = [
        len(paragraph)
        for paragraph in all_paragraphs
    ]

    print("\n--- PARAGRAPH STATISTICS ---")
    print("Total paragraphs:", len(all_paragraphs))
    print("Shortest:", min(lengths))
    print("Longest:", max(lengths))
    print("Average:", round(sum(lengths) / len(lengths)))

    print("\n--- PARAGRAPH PREVIEWS ---")

    for paragraph in all_paragraphs[:3]:
        print("Paragraph preview:", paragraph[:150])


# --------------------------------------------------
# 14. Figure summary
# --------------------------------------------------

print("\n--- FIGURE SUMMARY ---")

print("Total figures:", len(figure_records))

for figure in figure_records:

    print(
        f"{figure.figure_id} | {figure.label}"
    )

    print(
        "Caption preview:",
        figure.caption[:150]
    )


# --------------------------------------------------
# 15. Final output confirmation
# --------------------------------------------------

print("\nSaved to:", OUTPUT_PATH)
