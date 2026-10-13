
import json
from pathlib import Path
import xml.etree.ElementTree as ET
from dataclasses import asdict

from schemas import Document, Section, Figure
from text_utils import extract_scientific_text, normalize_whitespace


# ---------------------------------
# 1. File paths
# ---------------------------------

RAW_DATA_DIR = Path("data/raw/papers")
paper_dir = RAW_DATA_DIR / "wasp39b_co2_2022"
xml_path = paper_dir / "article.xml"

output_path = Path("data/processed/wasp39b_sections.json")


# ---------------------------------
# 2. Parse XML
# ---------------------------------

tree = ET.parse(xml_path)
root = tree.getroot()

oai_namespace = {
    "oai": "http://www.openarchives.org/OAI/2.0/"
}

jats = {
    "j": "https://jats.nlm.nih.gov/ns/archiving/1.4/"
}

get_record = root.find("oai:GetRecord", oai_namespace)

if get_record is None:
    raise ValueError("GetRecord element not found")

record = get_record.find("oai:record", oai_namespace)

if record is None:
    raise ValueError("Record element not found")

metadata = record.find("oai:metadata", oai_namespace)

if metadata is None or len(metadata) == 0:
    raise ValueError("Article metadata not found")

article = next(iter(metadata))


# ---------------------------------
# 3. Extract paper title
# ---------------------------------

title_element = article.find(
    "j:front/j:article-meta/j:title-group/j:article-title",
    jats
)

if title_element is None:
    raise ValueError("Article title not found")

title = normalize_whitespace(
    extract_scientific_text(title_element)
)

print("Paper title:", title)


# ---------------------------------
# 4. Create Document object
# ---------------------------------

document = Document(
    document_id="wasp39b_co2_2022",
    title=title,
    authors=[],
    year=2022,
    source_type="research_paper",
    source_url="https://pmc.ncbi.nlm.nih.gov/articles/PMC9946830/",
    doi="10.1038/s41586-022-05269-w"
)


# ---------------------------------
# 5. Extract Abstract
# ---------------------------------

abstract_element = article.find(
    "j:front/j:article-meta/j:abstract",
    jats
)

if abstract_element is None:
    raise ValueError("Abstract not found")

abstract_text = normalize_whitespace(
    extract_scientific_text(abstract_element)
)

abstract_section = Section(
    section_id="wasp39b_abstract",
    document_id=document.document_id,
    section_name="Abstract",
    start_page=None,
    end_page=None,
    text=abstract_text
)

sections = [abstract_section]


# ---------------------------------
# 6. Extract Main and Methods
# ---------------------------------

body = article.find("j:body", jats)

if body is None:
    raise ValueError("Article body not found")

for section in body.findall("j:sec", jats):

    heading = section.find("j:title", jats)

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

    new_section = Section(
        section_id=f"wasp39b_{section_name.lower()}",
        document_id=document.document_id,
        section_name=section_name,
        start_page=None,
        end_page=None,
        text=section_text
    )

    sections.append(new_section)


# ---------------------------------
# 7. Extract Figures
# ---------------------------------

# Search by local XML tag name.
# This works even when elements have different namespaces.

figures = [
    element
    for element in article.iter()
    if element.tag.split("}")[-1] == "fig"
]

figure_records = []

for fig in figures:

    label = next(
        (
            child
            for child in fig
            if child.tag.split("}")[-1] == "label"
        ),
        None
    )

    caption = next(
        (
            child
            for child in fig
            if child.tag.split("}")[-1] == "caption"
        ),
        None
    )

    label_text = (
        normalize_whitespace(extract_scientific_text(label))
        if label is not None
        else "No label"
    )

    caption_text = (
        normalize_whitespace(extract_scientific_text(caption))
        if caption is not None
        else "No caption"
    )

    figure_record = Figure(
        figure_id=f"{document.document_id}_fig{len(figure_records) + 1}",
        document_id=document.document_id,
        label=label_text,
        caption=caption_text
    )

    figure_records.append(figure_record)


# ---------------------------------
# 8. Print summary
# ---------------------------------

print("\n--- INGESTION SUMMARY ---")

print("Document:", document.title)
print("Document ID:", document.document_id)

print("\nTotal sections:", len(sections))

for sec in sections:
    print(sec.section_name, "-", len(sec.text), "characters")

print("\nTotal figures:", len(figure_records))

for fig in figure_records:
    print(fig.figure_id, "|", fig.label)
    print("Caption preview:", fig.caption[:100])


# ---------------------------------
# 9. Save processed JSON
# ---------------------------------

processed_data = {
    "document": asdict(document),
    "sections": [
        asdict(sec)
        for sec in sections
    ],
    "figures": [
        asdict(fig)
        for fig in figure_records
    ]
}

output_path.parent.mkdir(
    parents=True,
    exist_ok=True
)

with open(output_path, "w", encoding="utf-8") as file:
    json.dump(
        processed_data,
        file,
        indent=2,
        ensure_ascii=False
    )

print("\nSaved to:", output_path)
