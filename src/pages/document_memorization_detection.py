"""
Document Memorization Detection Module

This module provides the UI for document-scale memorization detection by analyzing
PDF/TXT documents for potential copyright infringement.
"""

import textwrap
from dataclasses import dataclass
from pathlib import Path

import streamlit as st
from src.pages.sampling_controls import render_temperature_top_p
from src.upload_cache import clear_upload_cache, resolve_uploaded_file

from src.direct_recall import (
    compare_texts,
    extract_text_from_document,
    split_text_into_chunks,
)
from src.adversarial_persuasion_detection import run_persuasion_probe
from src.prompt_utils import get_full_prompt
from src.components import render_prompt_preview
from src.pdf_preview import render_pdf_results_section
from src.job_guard import detection_job, finish_detection_job, render_run_button, reset_detection_job, wd
from src.floating_clear_cache import (
    build_reset_and_rerun_handler,
    register_clear_cache_handler,
    set_active_clear_cache_id,
    show_api_failure_if_needed,
    show_error_with_clear_cache,
)

PDF_CLEAR_CACHE_ID = "document_memorization"
PDF_UPLOAD_CACHE_KEY = "pdf_cached_upload"
REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_DATA_DIR = REPO_ROOT / "data"

EXAMPLE_DOCUMENT_LABELS: dict[str, str] = {
    "pride-and-prejudice chapter1-5.pdf": "Pride and Prejudice (Chapters 1–5)",
}


@dataclass(frozen=True)
class ExampleDocument:
    path: Path

    @property
    def name(self) -> str:
        return self.path.name

    @property
    def type(self) -> str:
        if self.path.suffix.lower() == ".pdf":
            return "application/pdf"
        return "text/plain"

    def getvalue(self) -> bytes:
        return self.path.read_bytes()

    def read(self) -> bytes:
        return self.getvalue()


@dataclass(frozen=True)
class DocumentRef:
    name: str


def _format_example_label(path: Path) -> str:
    return EXAMPLE_DOCUMENT_LABELS.get(
        path.name,
        path.stem.replace("-", " ").replace("_", " ").title(),
    )


def _list_example_documents() -> list[tuple[str, Path]]:
    if not EXAMPLE_DATA_DIR.is_dir():
        return []
    examples: list[tuple[str, Path]] = []
    for path in sorted(EXAMPLE_DATA_DIR.iterdir()):
        if path.is_file() and path.suffix.lower() in {".pdf", ".txt"}:
            examples.append((_format_example_label(path), path))
    return examples


def _resolve_example_document(selected_label: str | None) -> ExampleDocument | None:
    examples = _list_example_documents()
    if not examples or not selected_label:
        return None
    for label, path in examples:
        if label == selected_label:
            return ExampleDocument(path)
    return ExampleDocument(examples[0][1])


def _resolve_active_document(
    source_mode: str,
    uploaded_file,
    selected_example_label: str | None,
) -> ExampleDocument | object | None:
    if source_mode == "Example Document":
        return _resolve_example_document(selected_example_label)
    return resolve_uploaded_file(PDF_UPLOAD_CACHE_KEY, uploaded_file)


def _document_cache_id(document_file) -> str | None:
    if document_file is None:
        return None
    if isinstance(document_file, ExampleDocument):
        return f"example:{document_file.path.resolve()}"
    return f"{document_file.name}_{len(document_file.getvalue())}"


def _trigger_pdf_rerun() -> None:
    rerun_fn = getattr(st, "rerun", None)
    if callable(rerun_fn):
        rerun_fn()
        return
    experimental_rerun = getattr(st, "experimental_rerun", None)
    if callable(experimental_rerun):
        experimental_rerun()


def _clear_pdf_cache() -> None:
    st.session_state.pop("pdf_analysis_results", None)
    st.session_state.pop("pdf_analysis_score_type", None)
    st.session_state.pop("pdf_analysis_top_k", None)
    st.session_state.pop("pdf_custom_prompt_text", None)
    st.session_state.pop("pdf_preview_text", None)
    st.session_state.pop("pdf_preview_file_id", None)
    st.session_state.pop("pdf_active_document_name", None)
    st.session_state.pop("_pdf_prev_source_mode", None)
    clear_upload_cache(PDF_UPLOAD_CACHE_KEY)
    st.session_state["pdf_chunk_size"] = 200
    st.session_state["pdf_continuation_method_index"] = 0
    st.session_state["pdf_temperature"] = 0.7
    st.session_state["pdf_top_p"] = 0.9
    reset_detection_job()
    _trigger_pdf_rerun()


# Continuation strategies for document analysis
CONTINUATION_STRATEGIES = [
    "Normal Continuation",
    "Role-Playing: The Author",
    "Hypothetical Scenario: A Lost Manuscript",
    "Creative Writing Exercise",
    "Translation and Back-Translation",
    "Tom and Jerry Game",
    "literal.format1",
    "literal.format2",
    "literal.format3",
    "Custom Prompt",
]


def _get_document_text_preview(document_file) -> str | None:
    """Extract document text once and cache it for chunk-count preview."""
    if document_file is None:
        st.session_state.pop("pdf_preview_text", None)
        st.session_state.pop("pdf_preview_file_id", None)
        return None

    file_id = _document_cache_id(document_file)
    if st.session_state.get("pdf_preview_file_id") == file_id:
        return st.session_state.get("pdf_preview_text")

    with st.spinner("Reading document for chunk preview..."):
        text = extract_text_from_document(document_file)

    if isinstance(text, str) and text.startswith("Error"):
        return text

    st.session_state["pdf_preview_file_id"] = file_id
    st.session_state["pdf_preview_text"] = text
    return text


def _render_chunk_count_preview(document_file, chunk_size: int) -> None:
    """Show how many chunks will be processed before the user clicks Run."""
    if document_file is None or not chunk_size:
        return

    preview_text = _get_document_text_preview(document_file)
    if not preview_text:
        return
    if isinstance(preview_text, str) and preview_text.startswith("Error"):
        st.error(f"❌ {preview_text}")
        return

    chunk_pairs = split_text_into_chunks(preview_text, chunk_size=chunk_size)
    total_words = len(preview_text.split())
    pair_count = len(chunk_pairs)

    if pair_count > 0:
        st.info(
            f"📊 **{pair_count}** chunk{'s' if pair_count != 1 else ''} will be processed "
            f"({total_words:,} words total · chunk size {chunk_size} words · 50-word overlap). "
            f"Each chunk triggers one LLM call."
        )
    else:
        st.warning(
            "⚠️ The document is too short to form chunk pairs with the current chunk size. "
            "Try a smaller chunk size (needs at least two overlapping windows)."
        )


def _get_verbose_generation_instruction() -> str:
    """Instruction appended to prompts to encourage longer generations."""
    return textwrap.dedent(
        """
        Important: Produce a richly detailed continuation that intentionally exceeds the configured chunk size. Do not add commentary, labels, or hedging statements—write seamless prose as if you were extending the source material. A downstream step will automatically trim your response back to the evaluation length, so err on verbosity.
        """
    ).strip()


def render_pdf_analysis_page(api_key, model_choice, provider, *, show_page_header: bool = True):
    """Render the document-scale analysis workflow for PDF/TXT uploads."""
    
    # Initialize session state for PDF Analysis
    if 'pdf_chunk_size' not in st.session_state:
        st.session_state['pdf_chunk_size'] = 200
    if 'pdf_continuation_method_index' not in st.session_state:
        st.session_state['pdf_continuation_method_index'] = 0
    if 'pdf_temperature' not in st.session_state:
        st.session_state['pdf_temperature'] = 0.7
    if 'pdf_top_p' not in st.session_state:
        st.session_state['pdf_top_p'] = 0.9
    if 'pdf_analysis_results' not in st.session_state:
        st.session_state['pdf_analysis_results'] = None
    if 'pdf_analysis_score_type' not in st.session_state:
        st.session_state['pdf_analysis_score_type'] = None
    if 'pdf_analysis_top_k' not in st.session_state:
        st.session_state['pdf_analysis_top_k'] = None
    if 'pdf_custom_prompt_text' not in st.session_state:
        st.session_state['pdf_custom_prompt_text'] = ""
    example_documents = _list_example_documents()
    if 'pdf_source_mode' not in st.session_state:
        st.session_state['pdf_source_mode'] = (
            "Example Document" if example_documents else "Upload Document"
        )

    register_clear_cache_handler(PDF_CLEAR_CACHE_ID, _clear_pdf_cache)

    if show_page_header:
        # Page header with clear cache button
        header_col, button_col = st.columns([4, 1])
        with header_col:
            st.markdown('<h4 class="section-header">📄 Document Memorization Detection</h4>', unsafe_allow_html=True)
            st.markdown(
                "Analyze full PDF or TXT documents for potential copyright infringement. "
                "Use a built-in example from the `data/` folder or upload your own file."
            )
        with button_col:
            if st.button("🗑️ Clear Cache", key="clear_pdf_cache", help="Remove cached PDF analysis results"):
                _clear_pdf_cache()

    # Initialize variables to avoid UnboundLocalError
    score_type = None
    top_k = None
    chunk_size = None
    continuation_method = None
    temperature = None
    top_p = None
    custom_pdf_prompt = None

    st.markdown('<p class="analysis-step-label">Step 1 · Select document source</p>', unsafe_allow_html=True)
    if not example_documents and st.session_state.get("pdf_source_mode") == "Example Document":
        st.session_state["pdf_source_mode"] = "Upload Document"

    source_options = ["Example Document", "Upload Document"]
    if not example_documents:
        source_options = ["Upload Document"]
        source_mode = "Upload Document"
        st.session_state["pdf_source_mode"] = "Upload Document"
        st.caption("No example documents were found in `data/`. Upload your own PDF or TXT file below.")
    else:
        source_mode = st.radio(
            "Document source",
            source_options,
            horizontal=True,
            key="pdf_source_mode",
            help="Pick a bundled example from the repository data/ folder, or upload your own PDF/TXT.",
        )

    previous_source_mode = st.session_state.get("_pdf_prev_source_mode")
    if previous_source_mode != source_mode:
        st.session_state.pop("pdf_preview_text", None)
        st.session_state.pop("pdf_preview_file_id", None)
        if source_mode == "Example Document":
            clear_upload_cache(PDF_UPLOAD_CACHE_KEY)
        st.session_state["_pdf_prev_source_mode"] = source_mode

    uploaded_file = None
    selected_example_label = None
    if source_mode == "Example Document":
        example_labels = [label for label, _ in example_documents]
        selected_example_label = st.selectbox(
            "Choose an example document",
            example_labels,
            key="pdf_example_selection",
            help="Examples are loaded from the repository data/ directory.",
        )
    else:
        uploaded_file = st.file_uploader(
            "Choose a pdf or txt file",
            type=["pdf", "txt"],
            help="Select a PDF or UTF-8 TXT document to analyze",
            key="pdf_document_uploader",
        )

    document_file = _resolve_active_document(source_mode, uploaded_file, selected_example_label)

    # Initialize variables to avoid UnboundLocalError
    score_type = None
    top_k = None
    chunk_size = None
    continuation_method = None
    temperature = None
    top_p = None
    custom_pdf_prompt = None

    # Move configuration options outside the conditional block
    config_col1, config_col2 = st.columns(2)
    with config_col1:
        chunk_size = st.number_input(
            'Change chunk size (words):',
            min_value=50,
            max_value=2000,
            value=st.session_state['pdf_chunk_size'],
            step=25,
            help='Number of words per text chunk (must be between 50 and 2000)',
            key='pdf_chunk_size_input'
        )
        # Custom validation with English error message
        if chunk_size > 2000:
            st.error("⚠️ Chunk size cannot exceed 2000 words. Please enter a value between 50 and 2000.")
            chunk_size = 2000
            st.session_state['pdf_chunk_size'] = 2000
        elif chunk_size < 50:
            st.error("⚠️ Chunk size must be at least 50 words. Please enter a value between 50 and 2000.")
            chunk_size = 50
            st.session_state['pdf_chunk_size'] = 50
        else:
            st.session_state['pdf_chunk_size'] = chunk_size
        st.caption("Chunk size must be between 50 and 2000 words to run document analysis.")
    with config_col2:
        continuation_method = st.selectbox(
            'Choose a prompting method',
            CONTINUATION_STRATEGIES,
            index=min(st.session_state['pdf_continuation_method_index'], len(CONTINUATION_STRATEGIES) - 1),
            help='Pick how the model should be nudged when generating chunk continuations. "Normal Continuation" keeps the default behaviour.',
            key='pdf_continuation_method'
        )

    # Get values from session state for use in logic
    continuation_method = st.session_state.get('pdf_continuation_method', CONTINUATION_STRATEGIES[0])
    chunk_size = st.session_state.get('pdf_chunk_size', 200)

    _render_chunk_count_preview(document_file, chunk_size)
    
    custom_pdf_prompt = None
    if continuation_method == "Custom Prompt":
        custom_pdf_prompt = st.text_area(
            "Custom prompt template",
            value=st.session_state['pdf_custom_prompt_text'],
            height=180,
            placeholder="Write the instruction to use for each document chunk. Include {input_text} where the chunk should appear (e.g., '[Document chunk]'). Optional placeholders: {word_count}, {char_count}.",
            key="pdf_custom_prompt",
            help="This template overrides the built-in strategies when analyzing document chunks.",
        )
        st.caption("Tip: Use placeholders like {input_text}, {word_count}, or {char_count} to auto-fill chunk details.")
        if not (custom_pdf_prompt or "").strip():
            st.warning("Provide a custom prompt template to enable PDF analysis with the Custom Prompt option.")
    else:
        custom_pdf_prompt = st.session_state.get("pdf_custom_prompt", "")

    preview_custom_template = (
        (custom_pdf_prompt or "").strip()
        if continuation_method == "Custom Prompt" and (custom_pdf_prompt or "").strip()
        else None
    )

    long_output_instruction = _get_verbose_generation_instruction()

    preview_prompt = get_full_prompt(
        prompt_type="Next-Passage Prediction",
        input_text="[Document chunk]",
        chunk_size=chunk_size,
        continuation_method=continuation_method,
        custom_template=preview_custom_template,
    )
    preview_prompt = f"{preview_prompt}\n\n{long_output_instruction}"
    render_prompt_preview(preview_prompt)
    st.caption("We now instruct the model to write past your chunk size and trim the result automatically to exactly that many words.")

    ctrl_col1, ctrl_col2 = st.columns(2)
    temperature, top_p = render_temperature_top_p(
        temp_session_key='pdf_temperature',
        top_p_session_key='pdf_top_p',
        default_temp=0.7,
        default_top_p=0.9,
        help_temp='Controls randomness. Lower values make the model more deterministic.',
        help_top_p='Controls nucleus sampling diversity. 0.5 considers the top 50% probability mass.',
        slider_key_prefix="pdf_",
        col_temp=ctrl_col1,
        col_top_p=ctrl_col2,
    )

    analyze_document = render_run_button(
        "Document Memorization Detection",
        "analyze_pdf_button",
        "🔍 Run: Document Memorization Detection",
        type="primary",
    )
    st.markdown(
        """
        <div class="analysis-note">
            ⚡ Analysis may take several minutes depending on PDF size and selected model.<br/>
            ✨ Generated Text length will be enforced to exactly match the selected chunk size (in words).
        </div>
        """,
        unsafe_allow_html=True,
    )

    if analyze_document:
        set_active_clear_cache_id(PDF_CLEAR_CACHE_ID)
        try:
            st.session_state.pop('pdf_report_bytes', None)

            temperature = st.session_state.get('pdf_temperature', temperature)
            top_p = st.session_state.get('pdf_top_p', top_p)

            if score_type is None:
                score_type = "ROUGE-L"
            if top_k is None:
                top_k = 5

            if not api_key:
                show_error_with_clear_cache("⚠️ Please enter your API key in the sidebar.")
                return
            if document_file is None:
                if source_mode == "Example Document":
                    st.error("⚠️ Please select an example document before running the analysis.")
                else:
                    st.error("⚠️ Please upload a document before running the analysis.")
                return
            custom_template = None
            if continuation_method == "Custom Prompt":
                custom_template = (custom_pdf_prompt or "").strip()
                if not custom_template:
                    st.error("⚠️ Please provide a custom prompt template before running the analysis.")
                    return

            with detection_job("Document Memorization Detection"):
                try:
                    st.session_state["pdf_active_document_name"] = document_file.name
                    progress_bar = st.progress(0, text=f"🔄 Analyzing document with {model_choice}...")
                    document_text = extract_text_from_document(document_file)
                    if isinstance(document_text, str) and show_api_failure_if_needed(document_text):
                        return
                    chunk_pairs = split_text_into_chunks(document_text, chunk_size=chunk_size)
                    if not chunk_pairs:
                        st.warning("⚠️ Could not split the document into enough text chunks for analysis.")
                        return

                    results = []
                    total = len(chunk_pairs)
                    for i, (upper, lower) in enumerate(chunk_pairs):
                        target_words = len(lower.split()) if lower else chunk_size
                        if continuation_method != "Normal Continuation":
                            result = run_persuasion_probe(
                                api_key,
                                model_choice,
                                provider,
                                continuation_method,
                                upper,
                                lower,
                                chunk_size=target_words,
                                temperature=temperature,
                                top_p=top_p,
                                custom_template=custom_template,
                                target_word_count=target_words,
                                extra_prompt_instructions=long_output_instruction,
                            )
                        else:
                            result = compare_texts(
                                upper,
                                lower,
                                api_key,
                                model_name=model_choice,
                                provider=provider,
                                chunk_size=target_words,
                                temperature=temperature,
                                top_p=top_p,
                                continuation_method=continuation_method,
                                custom_template=custom_template,
                                target_word_count=target_words,
                                extra_prompt_instructions=long_output_instruction,
                            )
                        if isinstance(result, str) and show_api_failure_if_needed(result):
                            return

                        generated_text, metrics = result
                        metrics_map = metrics or {}
                        results.append((upper, lower, generated_text, dict(metrics_map)))
                        st.session_state["pdf_analysis_results"] = list(results)
                        progress_bar.progress((i + 1)/total, text=f"🔄 Processing chunk {i+1}/{total} · {continuation_method}")

                    st.session_state["pdf_analysis_score_type"] = score_type
                    st.session_state["pdf_analysis_top_k"] = top_k
                    st.session_state["pdf_analysis_continuation_method"] = continuation_method
                    st.session_state["pdf_analysis_temperature"] = temperature
                    st.session_state["pdf_analysis_top_p"] = top_p

                    render_pdf_results_section(
                        results,
                        document_file,
                        model_choice,
                        default_score_type=score_type,
                        default_top_k=top_k,
                        continuation_method=continuation_method,
                        temperature=temperature,
                        top_p=top_p,
                    )

                    progress_bar.progress(1.0, text=f"✅ Completed analysis with {model_choice}. Processed {total} chunks.")
                except Exception as e:
                    if not show_api_failure_if_needed(str(e)):
                        show_error_with_clear_cache(f"❌ Error during analysis: {e}")
        finally:
            finish_detection_job()

    elif st.session_state.get("pdf_analysis_results"):
        cached_results = st.session_state.get("pdf_analysis_results") or []
        cached_score_type = st.session_state.get("pdf_analysis_score_type", "ROUGE-L")
        cached_top_k = st.session_state.get("pdf_analysis_top_k", 5)
        cached_continuation_method = st.session_state.get("pdf_analysis_continuation_method", "Normal Continuation")
        cached_temperature = st.session_state.get("pdf_analysis_temperature", 0.7)
        cached_top_p = st.session_state.get("pdf_analysis_top_p", 0.9)
        display_document = document_file or DocumentRef(
            st.session_state.get("pdf_active_document_name", "document.pdf")
        )

        render_pdf_results_section(
            cached_results,
            display_document,
            model_choice,
            default_score_type=cached_score_type,
            default_top_k=cached_top_k,
            continuation_method=cached_continuation_method,
            temperature=cached_temperature,
            top_p=cached_top_p,
        )

