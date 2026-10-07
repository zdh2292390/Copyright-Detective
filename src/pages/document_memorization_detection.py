"""
Document Memorization Detection Module

This module provides the UI for document-scale memorization detection by analyzing
PDF/TXT documents for potential copyright infringement.
"""

import hashlib
import textwrap
from dataclasses import dataclass
from pathlib import Path

import streamlit as st
from src.pages.sampling_controls import render_temperature_top_p
from src.upload_cache import clear_upload_cache, resolve_uploaded_file

from src.direct_recall import (
    extract_text_from_document,
    split_text_into_chunks,
)
from src.prompt_utils import get_full_prompt
from src.components import render_prompt_preview
from src.pdf_preview import render_pdf_results_section
from src.document_analysis import (
    analysis_fingerprint,
    analysis_results,
    new_analysis_state,
)
from src.document_checkpoints import CheckpointError
from src.document_jobs import DOCUMENT_JOBS
from src.job_guard import finish_detection_job, render_run_button, reset_detection_job
from src.floating_clear_cache import (
    register_clear_cache_handler,
    set_active_clear_cache_id,
    show_error_with_clear_cache,
)

PDF_CLEAR_CACHE_ID = "document_memorization"
PDF_UPLOAD_CACHE_KEY = "pdf_cached_upload"
PDF_JOB_TOKEN_KEY = "pdf_analysis_job_token"
PDF_JOB_QUERY_KEY = "document_analysis"
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
    content_hash = hashlib.sha256(document_file.getvalue()).hexdigest()
    return f"{document_file.name}:{content_hash}"


def _trigger_pdf_rerun() -> None:
    rerun_fn = getattr(st, "rerun", None)
    if callable(rerun_fn):
        rerun_fn()
        return
    experimental_rerun = getattr(st, "experimental_rerun", None)
    if callable(experimental_rerun):
        experimental_rerun()


def _clear_pdf_cache() -> None:
    token = st.session_state.get(PDF_JOB_TOKEN_KEY)
    if token:
        try:
            if DOCUMENT_JOBS.is_running(token):
                DOCUMENT_JOBS.stop(token, owner_id=st.session_state.get("user_id"))
                st.warning("Stop requested. Wait for the current request to finish, then clear the cache.")
                return
            DOCUMENT_JOBS.delete(token, owner_id=st.session_state.get("user_id"))
        except (CheckpointError, ValueError) as exc:
            st.error(f"Could not remove the saved analysis: {exc}")
            return
    st.session_state.pop(PDF_JOB_TOKEN_KEY, None)
    st.query_params.pop(PDF_JOB_QUERY_KEY, None)
    for key in (
        "pdf_analysis_state", "pdf_report_bytes", "pdf_report_fingerprint",
        "pdf_analysis_model", "pdf_analysis_chunk_size",
        "pdf_analysis_continuation_method", "pdf_analysis_temperature",
        "pdf_analysis_top_p",
    ):
        st.session_state.pop(key, None)
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
            f"📊 **{pair_count:,}** chunk{'s' if pair_count != 1 else ''} will be processed "
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


def _sync_document_job(state) -> None:
    st.session_state["pdf_analysis_state"] = state
    st.session_state["pdf_analysis_results"] = analysis_results(state)
    settings = state["settings"]
    st.session_state["pdf_active_document_name"] = settings["filename"]
    st.session_state["pdf_analysis_model"] = settings["model"]
    st.session_state["pdf_analysis_chunk_size"] = settings["chunk_size"]


def _restore_document_job():
    token = st.query_params.get(PDF_JOB_QUERY_KEY) or st.session_state.get(PDF_JOB_TOKEN_KEY)
    if not token:
        return None
    try:
        state = DOCUMENT_JOBS.get(token, owner_id=st.session_state.get("user_id"))
        if state is None:
            raise CheckpointError("The saved analysis is no longer available. Start a new run.")
        st.session_state[PDF_JOB_TOKEN_KEY] = token
        _sync_document_job(state)
        return token
    except (CheckpointError, ValueError) as exc:
        # Do not display another account's previously cached document after sign-out.
        st.session_state.pop("pdf_analysis_state", None)
        st.session_state.pop("pdf_analysis_results", None)
        st.session_state.pop("pdf_report_bytes", None)
        st.error(f"Could not restore the saved analysis: {exc}")
        return None


def _render_saved_pdf_results(state) -> None:
    settings = state["settings"]
    render_pdf_results_section(
        analysis_results(state), DocumentRef(settings["filename"]), settings["model"],
        default_score_type=st.session_state.get("pdf_analysis_score_type") or "ROUGE-L",
        default_top_k=st.session_state.get("pdf_analysis_top_k") or 5,
        continuation_method=settings["continuation_method"],
        temperature=settings["temperature"], top_p=settings["top_p"],
        chunk_size=settings["chunk_size"], analysis_progress=state,
    )
    if state.get("failures"):
        with st.expander(f"Failed chunks ({len(state['failures'])})"):
            for index, error in sorted(state["failures"].items()):
                st.write(f"Chunk {index + 1}: {error}")


@st.fragment(run_every=2)
def _poll_document_job(token, owner_id) -> None:
    try:
        state = DOCUMENT_JOBS.get(token, owner_id=owner_id)
    except (CheckpointError, ValueError) as exc:
        st.error(f"Could not read analysis progress: {exc}")
        return
    if state is None:
        st.error("The saved analysis is no longer available.")
        return
    _sync_document_job(state)
    if not DOCUMENT_JOBS.is_running(token):
        # Remove timed polling after the task ends and render its final report.
        st.rerun()
    completed = len(state["results"])
    total = state["total_chunks"]
    st.progress(
        completed / max(total, 1),
        text=f"Analyzing chunk {state.get('current_chunk') or 1}/{total} · {completed} succeeded · {len(state['failures'])} failed",
    )
    st.caption("Analysis continues in the background if you refresh or switch pages. Progress is saved after each chunk.")
    if state.get("retry_in_seconds"):
        st.caption(f"Temporary API error; retry {state.get('retry_attempt')} after {state['retry_in_seconds']:.1f} seconds.")
    if state.get("stop_requested"):
        st.info("Stop requested. Analysis will stop when the current API request finishes; completed chunks are kept.")
    if st.button("Stop analysis", key="stop_pdf_analysis", disabled=bool(state.get("stop_requested"))):
        DOCUMENT_JOBS.stop(token, owner_id=owner_id)
        st.info("Stop requested; completed chunks will be preserved.")


def _render_document_job(token, api_key, provider) -> None:
    owner_id = st.session_state.get("user_id")
    state = DOCUMENT_JOBS.get(token, owner_id=owner_id)
    if state is None:
        return
    _sync_document_job(state)
    settings = state["settings"]
    st.markdown("---")
    st.markdown(f"**Saved analysis: {settings['filename']} · {settings['model']}**")
    st.caption("Bookmark this page to restore this analysis after reconnecting. The recovery link provides access to this document; keep it private.")
    if DOCUMENT_JOBS.is_running(token):
        _poll_document_job(token, owner_id)
        return
    if state["status"] == "running":
        # The worker may have finished between the snapshot and active check.
        state = DOCUMENT_JOBS.get(token, owner_id=owner_id)
        if state is None:
            return
        _sync_document_job(state)
    if state["status"] != "complete":
        if provider != settings["provider"]:
            st.info(f"Select {settings['provider']} in the sidebar to resume this saved analysis.")
        if st.button(
            "Resume saved analysis", key="resume_saved_pdf_analysis",
            disabled=(not api_key and settings["provider"] != "Local vLLM") or provider != settings["provider"],
            help="Uses the saved document and generation settings, even if the controls above have changed.",
        ):
            try:
                DOCUMENT_JOBS.submit(token, api_key, owner_id=owner_id)
                _trigger_pdf_rerun()
            except (CheckpointError, ValueError) as exc:
                st.error(f"Could not resume analysis: {exc}")
    _render_saved_pdf_results(state)


def render_pdf_analysis_page(api_key, model_choice, provider, *, show_page_header: bool = True):
    """Render the document-scale analysis workflow for PDF/TXT uploads."""
    
    token = _restore_document_job()
    job_running = bool(token and DOCUMENT_JOBS.is_running(token))

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
            if st.button("🗑️ Clear Cache", key="clear_pdf_cache", help="Remove cached PDF analysis results", disabled=job_running):
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
            min_value=75,
            max_value=2000,
            value=max(75, st.session_state['pdf_chunk_size']),
            step=25,
            help='Number of words per text chunk (must be between 75 and 2000)',
            key='pdf_chunk_size_input'
        )
        # Custom validation with English error message
        if chunk_size > 2000:
            st.error("⚠️ Chunk size cannot exceed 2000 words. Please enter a value between 75 and 2000.")
            chunk_size = 2000
            st.session_state['pdf_chunk_size'] = 2000
        elif chunk_size < 75:
            st.error("⚠️ Chunk size must be at least 75 words. Please enter a value between 75 and 2000.")
            chunk_size = 75
            st.session_state['pdf_chunk_size'] = 75
        else:
            st.session_state['pdf_chunk_size'] = chunk_size
        st.caption("Chunk size must be between 75 and 2000 words to run document analysis.")
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

    # The preview and execution use the same extracted text and chunk settings.
    document_text = _get_document_text_preview(document_file)
    analysis_settings = {
        "filename": document_file.name if document_file else "document.pdf",
        "model": model_choice,
        "provider": provider,
        "chunk_size": chunk_size,
        "overlap": 50,
        "continuation_method": continuation_method,
        "temperature": temperature,
        "top_p": top_p,
        "custom_template": preview_custom_template,
        "extra_prompt_instructions": long_output_instruction,
        "base_url": st.session_state.get("sidebar_local_vllm_base_url", "http://localhost:8000/v1") or "http://localhost:8000/v1"
        if provider == "Local vLLM" else None,
    }
    fingerprint = (
        analysis_fingerprint(document_text, analysis_settings)
        if document_text and not document_text.startswith("Error") else None
    )
    saved_analysis = st.session_state.get("pdf_analysis_state")
    resume_analysis = bool(
        saved_analysis and saved_analysis.get("fingerprint") == fingerprint
        and len(saved_analysis["results"]) < saved_analysis["total_chunks"]
    )
    button_label = "🔍 Run: Document Memorization Detection"
    if resume_analysis:
        button_label = (
            f"▶️ Resume document analysis "
            f"({len(saved_analysis['results'])}/{saved_analysis['total_chunks']} complete)"
        )
        st.caption(
            "Resume retries failed and unprocessed chunks using these same settings. "
            "Completed chunks are kept; do not clear the cache to resume."
        )
    elif saved_analysis and saved_analysis.get("status") != "complete":
        st.caption(
            "An incomplete analysis is saved. Restore its document and generation "
            "settings to resume, or use Run to start a new analysis with these settings."
        )

    analyze_document = render_run_button(
        "Document Memorization Detection",
        "analyze_pdf_button",
        button_label,
        type="primary",
        disabled=job_running,
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
            if not api_key and provider != "Local vLLM":
                show_error_with_clear_cache("⚠️ Please enter your API key in the sidebar.")
                return
            if document_file is None:
                st.error("⚠️ Please select or upload a document before running the analysis.")
                return
            if not document_text or document_text.startswith("Error"):
                st.error(document_text or "⚠️ The document contains no extractable text.")
                return
            if continuation_method == "Custom Prompt" and not preview_custom_template:
                st.error("⚠️ Please provide a custom prompt template before running the analysis.")
                return
            if not resume_analysis:
                chunk_pairs = split_text_into_chunks(document_text, chunk_size=chunk_size)
                if not chunk_pairs:
                    st.warning("⚠️ Could not split the document into enough text chunks for analysis.")
                    return
                state = new_analysis_state(fingerprint, analysis_settings, len(chunk_pairs))
                token = DOCUMENT_JOBS.create(
                    state, chunk_pairs, owner_id=st.session_state.get("user_id")
                )
                st.session_state[PDF_JOB_TOKEN_KEY] = token
                st.query_params[PDF_JOB_QUERY_KEY] = token
            elif not token:
                # Upgrade the preceding in-session checkpoint to a durable task.
                chunk_pairs = split_text_into_chunks(document_text, chunk_size=chunk_size)
                token = DOCUMENT_JOBS.create(
                    saved_analysis, chunk_pairs, owner_id=st.session_state.get("user_id")
                )
                st.session_state[PDF_JOB_TOKEN_KEY] = token
                st.query_params[PDF_JOB_QUERY_KEY] = token
            st.session_state["pdf_analysis_score_type"] = "ROUGE-L"
            st.session_state["pdf_analysis_top_k"] = 5
            st.session_state.pop("pdf_report_bytes", None)
            st.session_state.pop("pdf_report_fingerprint", None)
            DOCUMENT_JOBS.submit(token, api_key, owner_id=st.session_state.get("user_id"))
        except (CheckpointError, ValueError) as exc:
            st.error(f"Could not start document analysis: {exc}")
        finally:
            finish_detection_job()

    if token:
        try:
            _render_document_job(token, api_key, provider)
        except (CheckpointError, ValueError) as exc:
            st.error(f"Could not read the saved analysis: {exc}")
    elif st.session_state.get("pdf_analysis_results"):
        # Historical results lack a durable task; report their coverage as unverified.
        state = st.session_state.get("pdf_analysis_state")
        if state:
            _render_saved_pdf_results(state)
        else:
            render_pdf_results_section(
                st.session_state["pdf_analysis_results"],
                DocumentRef(st.session_state.get("pdf_active_document_name", "document.pdf")),
                st.session_state.get("pdf_analysis_model", "Unknown (legacy run)"),
                default_score_type=st.session_state.get("pdf_analysis_score_type") or "ROUGE-L",
                default_top_k=st.session_state.get("pdf_analysis_top_k") or 5,
                continuation_method=st.session_state.get("pdf_analysis_continuation_method", "Normal Continuation"),
                temperature=st.session_state.get("pdf_analysis_temperature", 0.7),
                top_p=st.session_state.get("pdf_analysis_top_p", 0.9),
                chunk_size=st.session_state.get("pdf_analysis_chunk_size", 200),
            )
