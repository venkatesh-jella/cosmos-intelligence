
import re
import sys
from pathlib import Path

import streamlit as st


# ============================================================
# 1. PROJECT SETUP
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent
COSMOS_SRC = PROJECT_ROOT / "src" / "cosmos"

if str(COSMOS_SRC) not in sys.path:
    sys.path.insert(0, str(COSMOS_SRC))

from rag_pipeline import CosmosRAGPipeline


# ============================================================
# 2. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Cosmos Intelligence",
    page_icon="🌌",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# 3. CUSTOM STYLING
# ============================================================

st.markdown(
    """
    <style>
        .block-container {
            max-width: 1100px;
            padding-top: 2rem;
        }

        h1, h2, h3 {
            letter-spacing: -0.025em;
        }

        .cosmos-subtitle {
            color: #94a3b8;
            font-size: 1.05rem;
            margin-top: -0.5rem;
            margin-bottom: 1.5rem;
        }

        .cosmos-footer {
            color: #94a3b8;
            font-size: 0.85rem;
            margin-top: 2rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# 4. SCIENTIFIC ANSWER FORMATTING
# ============================================================

def format_scientific_answer(answer):
    """
    Normalize display-only scientific notation.

    Does not change the answer used for verification.
    """

    if not isinstance(answer, str):
        return ""

    formatted = answer

    # Replace the known PDF-extracted solar-mass artifact.
    formatted = re.sub(
        r"\\text\s*\{\s*Me\s*\}",
        lambda match: r"\mathrm{M}_{\odot}",
        formatted,
    )

    # Convert PDF-extracted units outside LaTeX where possible.
    formatted = re.sub(
        r"(?<=10\^9)\s+Me\b",
        r" M_sun",
        formatted,
    )

    return formatted


def render_scientific_answer(answer):
    """
    Display text and mathematical expressions separately.

    Supports:
        $ ... $
        \\( ... \\)

    Mathematical expressions are rendered with st.latex().
    """

    formatted = format_scientific_answer(answer)

    # Match either dollar-delimited or parenthesis-delimited math.
    math_pattern = re.compile(
        r"\$([^$]+)\$|\\\((.*?)\\\)",
        flags=re.DOTALL,
    )

    matches = list(
        math_pattern.finditer(formatted)
    )

    if not matches:
        st.markdown(formatted)
        return

    previous_end = 0

    for match in matches:

        # Render text preceding the equation.
        before = formatted[
            previous_end:match.start()
        ].strip()

        if before:
            st.markdown(before)

        # Extract the equation without its delimiters.
        expression = (
            match.group(1)
            if match.group(1) is not None
            else match.group(2)
        )

        if expression:
            st.latex(expression.strip())

        previous_end = match.end()

    # Render text and citations after the final equation.
    remaining = formatted[
        previous_end:
    ].strip()

    if remaining:
        st.markdown(remaining)


# ============================================================
# 5. CACHE THE AI PIPELINE
# ============================================================

@st.cache_resource(
    show_spinner="Loading Cosmos Intelligence AI models..."
)
def load_pipeline():
    """
    Load the RAG pipeline once and reuse its models.
    """

    return CosmosRAGPipeline()


# ============================================================
# 6. SIDEBAR
# ============================================================

with st.sidebar:

    st.title("🌌 Cosmos Intelligence")

    st.caption(
        "Scientific Research Copilot · v1.0"
    )

    st.divider()

    st.subheader("Research Library")

    st.markdown(
        """
        **JWST / Exoplanet Atmospheres**

        - WASP-39b carbon dioxide detection (2022)

        **Event Horizon Telescope**

        - M87* black-hole shadow (2019)
        """
    )

    st.divider()

    st.subheader("AI Architecture")

    st.markdown(
        """
        - Hybrid BM25 + dense retrieval
        - Reciprocal Rank Fusion (RRF)
        - Cross-encoder reranking
        - Qwen3-0.6B local generation
        - NLI evidence verification
        - Source-grounded citations
        """
    )

    st.divider()

    st.caption(
        "Runs locally using Python, PyTorch, "
        "Sentence Transformers, and Streamlit."
    )


# ============================================================
# 7. MAIN HEADER
# ============================================================

st.title("🌌 Cosmos Intelligence")

st.markdown(
    '<p class="cosmos-subtitle">'
    'Explore the universe through scientific research. '
    'Ask questions grounded in published astronomy papers.'
    '</p>',
    unsafe_allow_html=True,
)

st.info(
    "Research prototype: answers are generated from "
    "two scientific papers and automatically checked "
    "against retrieved scientific evidence."
)


# ============================================================
# 8. RESEARCH QUESTIONS
# ============================================================

st.subheader("Explore research questions")

EXAMPLE_QUESTIONS = [
    "What molecule did JWST detect in WASP-39b's atmosphere?",
    "At what wavelength was the carbon dioxide feature observed?",
    "What did the Event Horizon Telescope observe in M87?",
    "What is the estimated mass of the black hole M87*?",
    "Did JWST confirm intelligent life on WASP-39b?",
]

selected_example = st.selectbox(
    "Choose an example or write your own question",
    options=["Custom question"] + EXAMPLE_QUESTIONS,
)

default_question = (
    ""
    if selected_example == "Custom question"
    else selected_example
)

question = st.text_area(
    "Your research question",
    value=default_question,
    placeholder="Ask a question about WASP-39b or M87*...",
    height=110,
)

ask_button = st.button(
    "🔭 Ask Cosmos",
    type="primary",
    use_container_width=True,
)


# ============================================================
# 9. RUN THE RAG PIPELINE
# ============================================================

if ask_button:

    if not question.strip():

        st.warning(
            "Please enter a research question."
        )

    else:

        try:

            pipeline = load_pipeline()

            with st.spinner(
                "Retrieving research papers, generating "
                "an answer, and verifying evidence..."
            ):

                result = pipeline.ask(
                    question.strip()
                )

            st.session_state["cosmos_result"] = result
            st.session_state["cosmos_question"] = question.strip()

        except Exception as error:

            st.session_state.pop(
                "cosmos_result",
                None,
            )

            st.error(
                "The research pipeline encountered an error."
            )

            with st.expander("Technical error details"):
                st.exception(error)


# ============================================================
# 10. DISPLAY THE RESEARCH ANSWER
# ============================================================

if "cosmos_result" in st.session_state:

    result = st.session_state["cosmos_result"]

    st.divider()

    st.subheader("Research Answer")

    status = result.get(
        "status",
        "unknown",
    )

    if status == "verified":

        st.success(
            "Evidence supported — the generated claims "
            "were supported by retrieved passages."
        )

    elif status in (
        "insufficient_evidence",
        "no_evidence",
    ):

        st.warning(
            "Insufficient scientific evidence in "
            "the current research library."
        )

    else:

        st.warning(
            "The available evidence could not establish "
            "support for the generated answer."
        )

    render_scientific_answer(
        result.get(
            "answer",
            "",
        )
    )


    # ========================================================
    # 11. SCIENTIFIC SOURCES
    # ========================================================

    st.divider()

    st.subheader("Scientific Sources")

    sources = result.get(
        "sources",
        [],
    )

    if not sources:

        st.info(
            "No scientific sources were retrieved."
        )

    else:

        verification = (
            result.get("verification") or {}
        )

        cited_source_numbers = set()

        for claim in verification.get(
            "claims",
            [],
        ):

            for source in claim.get(
                "supporting_sources",
                [],
            ):

                cited_source_numbers.add(
                    source["source_number"]
                )

        if status != "verified":

            st.caption(
                "Retrieved passages are provided "
                "for inspection and are not "
                "verified citations."
            )

        for index, source in enumerate(
            sources,
            start=1,
        ):

            is_cited = (
                status == "verified"
                and index in cited_source_numbers
            )

            document_id = source.get(
                "document_id",
                "Unknown document",
            )

            section = source.get(
                "section",
                "Unknown section",
            )

            label = (
                f"Source [{index}] · "
                f"{document_id} · "
                f"{section}"
            )

            if is_cited:
                label += " · CITED"

            with st.expander(
                label,
                expanded=is_cited,
            ):

                st.markdown(
                    f"**Document:** `{document_id}`"
                )

                st.markdown(
                    f"**Section:** {section}"
                )

                st.markdown(
                    "**Chunk ID:** "
                    f"`{source.get('chunk_id', 'N/A')}`"
                )

                start_page = source.get(
                    "start_page"
                )

                end_page = source.get(
                    "end_page"
                )

                if start_page is not None:

                    if (
                        end_page is not None
                        and end_page != start_page
                    ):

                        st.markdown(
                            f"**Pages:** "
                            f"{start_page}–{end_page}"
                        )

                    else:

                        st.markdown(
                            f"**Page:** {start_page}"
                        )

                reranker_score = source.get(
                    "reranker_score"
                )

                if reranker_score is not None:

                    st.caption(
                        "Reranker score: "
                        f"{reranker_score:.4f}"
                    )

                st.markdown(
                    "**Retrieved scientific passage**"
                )

                st.write(
                    source.get(
                        "text",
                        "No passage text available.",
                    )
                )


    # ========================================================
    # 12. EVIDENCE VERIFICATION DETAILS
    # ========================================================

    verification = (
        result.get("verification") or {}
    )

    if verification:

        with st.expander(
            "Evidence verification details"
        ):

            for index, claim in enumerate(
                verification.get(
                    "claims",
                    [],
                ),
                start=1,
            ):

                st.markdown(
                    f"**Claim {index}**"
                )

                st.write(
                    claim.get(
                        "claim",
                        "",
                    )
                )

                if claim.get(
                    "supported",
                    False,
                ):

                    st.success(
                        "Claim supported"
                    )

                else:

                    st.error(
                        "Support not established"
                    )

                for source in claim.get(
                    "supporting_sources",
                    [],
                ):

                    st.caption(
                        "Supporting source "
                        f"[{source['source_number']}]"
                    )

                    st.write(
                        source.get(
                            "supporting_sentence",
                            "",
                        )
                    )

                st.divider()


# ============================================================
# 13. FOOTER
# ============================================================

st.markdown(
    '<p class="cosmos-footer">'
    'Cosmos Intelligence v1.0 · '
    'Scientific RAG · '
    'Local Qwen3 · '
    'Evidence Verification'
    '</p>',
    unsafe_allow_html=True,
)
