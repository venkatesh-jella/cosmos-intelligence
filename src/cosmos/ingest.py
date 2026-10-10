import json
from pathlib import Path
import xml.etree.ElementTree as ET
from dataclasses import asdict

RAW_DATA_DIR = Path("data/raw/papers")

paper_dir = RAW_DATA_DIR / "wasp39b_co2_2022"
xml_path = paper_dir / "article.xml"

tree = ET.parse(xml_path)
root = tree.getroot()
print(root.tag)

namespace = {
    "oai": "http://www.openarchives.org/OAI/2.0/"
}

get_record = root.find("oai:GetRecord", namespace)

print(get_record.tag)

record = get_record.find("oai:record", namespace)
metadata = record.find("oai:metadata", namespace)

article = next(iter(metadata))

jats = {
    "j": "https://jats.nlm.nih.gov/ns/archiving/1.4/"
}
title_element = article.find(
    "j:front/j:article-meta/j:title-group/j:article-title",
    jats
)

if title_element is None:
    raise ValueError("Article title not found")

title = "".join(title_element.itertext()).strip()

print("Paper title:", title)

abstract_element = article.find(
    "j:front/j:article-meta/j:abstract",
    jats
)

if abstract_element is None:
    raise ValueError("Abstract not found")

abstract_text = " ".join(
    " ".join(abstract_element.itertext()).split()
)

print("Abstract:", abstract_text)

for element in abstract_element.iter():
    if  element.tag.endswith(("sub", "sup")):
        print("Tag:", element.tag)
        print("Text:", "".join(element.itertext()))


from schemas import Document,Section

abstract_section = Section(
    section_id="wasp39b_abstract",
    document_id="wasp39b_co2_2022",
    section_name="Abstract",
    start_page=None,
    end_page=None,
    text=abstract_text
)

print("Section:", abstract_section.section_name)
print("Characters:", len(abstract_section.text))

document = Document(
    document_id="wasp39b_co2_2022",
    title=title,
    authors=[],
    year=2022,
    source_type="research_paper",
    source_url="https://pmc.ncbi.nlm.nih.gov/articles/PMC9946830/",
    doi="10.1038/s41586-022-05269-w"
)

print("Document:", document.title)
print("Document ID:", document.document_id)
print("Section belongs to:", abstract_section.document_id)

body = article.find("j:body", jats)

if body is None:
    raise ValueError("Article body not found")
sections =[abstract_section]

for section in body.findall("j:sec", jats):
    heading = section.find("j:title", jats)

    if heading is not None:
        section_name = "".join(heading.itertext()).strip()
    else:
        section_name = "Untitled"

    section_text = " ".join(
        " ".join(section.itertext()).split()
    )

    #only keep Mian and Methods for now 
    if section_name in ["Main","Methods"]:
        new_section = Section(
            section_id=f"wasp39b_{section_name.lower()}",
            document_id=document.document_id,
            section_name=section_name,
            start_page=None,
            end_page=None,
            text=section_text
        )

        sections.append(new_section)

print("Total sections:", len(sections))

for sec in sections:
    print(sec.section_name, len(sec.text))

output_path = Path("data/processed/wasp39b_sections.json")

processed_data = {
    "document": asdict(document),
    "sections": [asdict(sec) for sec in sections]
}

with open(output_path, "w", encoding="utf-8") as file:
    json.dump(processed_data, file, indent=2, ensure_ascii=False)

print("Saved to:", output_path)

