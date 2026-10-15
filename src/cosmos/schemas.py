from dataclasses import dataclass

@dataclass 
class Document:
    document_id: str
    title: str
    authors: list[str]
    year: int
    source_type: str
    source_url: str
    doi: str | None

@dataclass
class Section:
    section_id: str
    document_id: str
    section_name: str
    start_page: int | None
    end_page: int | None
    text: str
    paragraphs: list[str]
    
@dataclass
class ChunkingConfig:
    config_id: str
    strategy: str
    max_tokens: int
    overlap_tokens: int
    include_section_heading: bool
    include_document_title: bool

@dataclass
class Chunk:
    chunk_id: str
    document_id: str
    section_id: str
    section_name: str
    text: str
    start_page: int | None
    end_page: int | None
    token_count: int | None
    chunking_config_id: str
@dataclass
class GoldEvidence:
    document_id: str
    section_name: str
    start_page: int | None
    end_page: int | None
    evidence_text: str
@dataclass
class EvaluationQuestion:
    question_id: str
    questionn: str
    reference_answer: str
    category: str
    is_answerable: bool
    gold_evidence: list[GoldEvidence]


@dataclass
class Figure:
    figure_id: str
    document_id: str
    label: str
    caption: str














  

