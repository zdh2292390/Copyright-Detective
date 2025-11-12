import io
import math
import random
import textwrap
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import streamlit as st
import pandas as pd
from Levenshtein import distance
import html
from datasets import load_dataset, concatenate_datasets
from src.copyright_detective.comparison import (
    compare_texts,
    enforce_exact_char_count,
    get_llm_completion,
    calculate_rouge_score,
    calculate_jaccard_index,
)
from src.copyright_detective.pdf_utils import extract_text_from_pdf, split_text_into_chunks
from src.config import DEFAULT_OPENROUTER_KEY

import matplotlib.pyplot as plt
from src.copyright_detective.jailbreak_probe import (
    run_persuasion_probe,
)
from src.copyright_detective.unlearning import (
    list_representational_features,
    run_representational_analysis,
    is_representational_analysis_available,
)
from src.copyright_detective.adversarial_prompting import (
    list_persuasion_strategies,
    get_mutation_instruction,
    run_inference_scaling,
    assess_intention_preservation,
    mutate_strategies,
    rank_by_rouge,
    list_baseline_prompts,
    ExperimentMode,
    DEFAULT_HP_REFERENCE_EXCERPT,
    serialise_mutation_with_judge,
    deserialise_mutation_with_judge,
    MutationWithJudge,
    MutationEvaluation,
    SimilarityMetrics,
)
from src.metrics.logger import RougeEvalLogger
from src.prompt_utils import get_full_prompt
from src.components import (
    render_collapsible_panel,
    render_prompt_preview,
    render_prompt_style_panel,
    render_table_card,
    render_collapsible_table_card,
    render_top_sample_distribution,
    render_direct_recall_diff,
    render_streamlit_accordion,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


MUSE_DATASET_ID = "muse-bench/MUSE-Books"
MUSE_DATASET_CONFIG = "knowmem"
PREFERRED_QUESTION_FIELDS = [
    "question",
    "prompt",
    "query",
    "input",
    "qa_question",
    "question_text",
]
PREFERRED_ANSWER_FIELDS = [
    "answer",
    "response",
    "target",
    "qa_answer",
    "output",
    "completion",
]

QA_INPUT_SESSION_KEY = "qa_input_text"
QA_GROUND_SESSION_KEY = "qa_ground_truth_text"
QA_ICL_SESSION_KEY = "qa_icl_examples"
QA_MUSE_SAMPLE_KEY_PREFIX = "qa_muse_sample_indices"
QA_EVAL_QUEUE_KEY = "qa_eval_examples"

# Predefined QA examples for few-shot selection
PREDEFINED_QA_EXAMPLES = [
    {
        "question": "What is the capital of France?",
        "answer": "Paris"
    },
    {
        "question": "Who wrote Romeo and Juliet?",
        "answer": "William Shakespeare"
    },
    {
        "question": "What is the largest planet in our solar system?",
        "answer": "Jupiter"
    },
    {
        "question": "What is the chemical symbol for water?",
        "answer": "H2O"
    },
    {
        "question": "In what year did World War II end?",
        "answer": "1945"
    },
    {
        "question": "What is the square root of 16?",
        "answer": "4"
    },
    {
        "question": "Who painted the Mona Lisa?",
        "answer": "Leonardo da Vinci"
    },
    {
        "question": "What is the longest river in the world?",
        "answer": "The Nile River"
    },
    {
        "question": "What is the currency used in Japan?",
        "answer": "Japanese Yen"
    },
    {
        "question": "What is the boiling point of water in Celsius?",
        "answer": "100 degrees Celsius"
    }
]


def _resolve_dataset_column(columns: List[str], candidates: List[str], fallback_keyword: str) -> Optional[str]:
    lowered_map = {column.lower(): column for column in columns}
    for candidate in candidates:
        if candidate in lowered_map:
            return lowered_map[candidate]
    for column in columns:
        if fallback_keyword in column.lower():
            return column
    return None


def _trigger_rerun() -> None:
    rerun_fn = getattr(st, "rerun", None)
    if callable(rerun_fn):
        rerun_fn()
        return
    experimental_rerun = getattr(st, "experimental_rerun", None)
    if callable(experimental_rerun):
        experimental_rerun()


@st.cache_data(show_spinner=False)
def load_cached_muse_knowmem() -> pd.DataFrame:
    dataset = load_dataset(MUSE_DATASET_ID, MUSE_DATASET_CONFIG)
    combined_dataset = concatenate_datasets([dataset[split] for split in dataset.keys()])
    df = combined_dataset.to_pandas().reset_index(drop=True)

    question_col = _resolve_dataset_column(df.columns.tolist(), PREFERRED_QUESTION_FIELDS, "question")
    answer_col = _resolve_dataset_column(df.columns.tolist(), PREFERRED_ANSWER_FIELDS, "answer")
    if not question_col or not answer_col:
        raise RuntimeError("Unable to resolve question/answer columns in the MUSE knowmem dataset.")

    df = df.rename(columns={question_col: "question", answer_col: "answer"})
    df.insert(0, "row_id", df.index.astype(int))
    ordered_columns = [
        "row_id",
        "question",
        "answer",
        *[column for column in df.columns if column not in {"row_id", "question", "answer"}],
    ]
    df = df[ordered_columns]
    return df


def generate_muse_example_options(num_examples: int = 5) -> Tuple[List[str], Dict[str, Dict[str, str]]]:
    """Generate random MUSE knowmem example options for dropdown.
    
    Returns:
        Tuple of (options_list, option_to_example_mapping)
    """
    try:
        df = load_cached_muse_knowmem()
        if len(df) == 0:
            return [], {}
        
        # Sample random examples
        sampled_indices = random.sample(range(len(df)), min(num_examples, len(df)))
        options = []
        option_mapping = {}
        for i, idx in enumerate(sampled_indices, 1):
            row = df.iloc[idx]
            question_preview = row["question"].split("\n", 1)[0][:50]  # First 50 chars of question
            option_text = f"Example {i}: {question_preview}..."
            options.append(option_text)
            option_mapping[option_text] = {
                "question": row["question"],
                "answer": row["answer"]
            }
        return options, option_mapping
    except Exception:
        return [], {}


def ensure_qa_session_defaults() -> None:
    st.session_state.setdefault(QA_INPUT_SESSION_KEY, "")
    st.session_state.setdefault(QA_GROUND_SESSION_KEY, "")
    st.session_state.setdefault(QA_ICL_SESSION_KEY, [])
    st.session_state.setdefault(QA_EVAL_QUEUE_KEY, [])
    st.session_state.setdefault("qa_knowmem_model_path", "")
    st.session_state.setdefault("qa_knowmem_tokenizer_path", "")
    st.session_state.setdefault("qa_knowmem_device", "cpu")
    st.session_state.setdefault("qa_knowmem_max_new_tokens", 64)
    st.session_state.setdefault("qa_eval_scope_radio", "Current QA pair")
    st.session_state.setdefault("qa_prompt_mode", "Zero-Shot")
    st.session_state.setdefault("qa_selected_few_shot_examples", [])


KNOWMEM_STOP_SEQUENCES: List[str] = ["\n\n", "\nQuestion", "Question:"]


def _trim_knowmem_completion(output: Optional[str]) -> str:
    """Trim model output to the first answer span, mirroring reference knowmem logic."""

    if not output:
        return ""

    trimmed = output
    for marker in KNOWMEM_STOP_SEQUENCES:
        if marker in trimmed:
            trimmed = trimmed.split(marker, 1)[0]

    # Remove leading "Answer:" if included by the model.
    lowered = trimmed.lstrip()
    if lowered.lower().startswith("answer:"):
        trimmed = lowered[len("answer:"):].lstrip()
    else:
        trimmed = trimmed.strip()

    return trimmed


def add_icl_example_from_row(row: pd.Series) -> None:
    ensure_qa_session_defaults()

    try:
        row_id = int(row.get("row_id", 0))
    except (TypeError, ValueError):
        row_id = 0

    signature = row_id
    icl_examples: List[Dict[str, Any]] = st.session_state[QA_ICL_SESSION_KEY]
    if any(example.get("signature") == signature for example in icl_examples):
        st.info("Example already added to in-context list.")
        return

    if len(icl_examples) >= 5:
        st.warning("You can keep up to 5 in-context QA examples. Remove one before adding more.")
        return

    metadata = {
        column: row[column]
        for column in row.index
        if column not in {"row_id", "question", "answer"} and pd.notna(row[column])
    }

    icl_examples.append(
        {
            "question": row["question"],
            "answer": row["answer"],
            "signature": signature,
            "metadata": metadata,
        }
    )


def add_eval_example_from_row(row: pd.Series) -> None:
    ensure_qa_session_defaults()

    try:
        row_id = int(row.get("row_id", 0))
    except (TypeError, ValueError):
        row_id = 0

    signature = row_id
    eval_examples: List[Dict[str, Any]] = st.session_state[QA_EVAL_QUEUE_KEY]
    if any(example.get("signature") == signature for example in eval_examples):
        st.info("Example already present in the evaluation batch.")
        return

    metadata = {
        column: row[column]
        for column in row.index
        if column not in {"row_id", "question", "answer"} and pd.notna(row[column])
    }

    eval_examples.append(
        {
            "question": row["question"],
            "answer": row["answer"],
            "signature": signature,
            "metadata": metadata,
        }
    )


def render_selected_icl_examples() -> None:
    ensure_qa_session_defaults()
    icl_examples: List[Dict[str, Any]] = st.session_state[QA_ICL_SESSION_KEY]
    if not icl_examples:
        return

    st.markdown("##### 📚 Selected in-context QA examples")
    st.caption("These examples will be prepended when running knowmem-style evaluations.")

    for idx, example in enumerate(list(icl_examples)):
        question_preview = example["question"].split("\n", 1)[0]
        header = f"ICL {idx + 1}: {question_preview[:80]}" if question_preview else f"ICL {idx + 1}"
        with st.expander(header, expanded=False):
            st.markdown("**Question**")
            st.write(example["question"])
            st.markdown("**Answer**")
            st.write(example["answer"])
            metadata = example.get("metadata") or {}
            if metadata:
                st.markdown("**Metadata**")
                st.json(metadata)
            if st.button("Remove", key=f"qa_remove_icl_{idx}"):
                icl_examples.pop(idx)
                _trigger_rerun()

    if st.button("Clear all in-context examples", key="qa_clear_all_icl"):
        st.session_state[QA_ICL_SESSION_KEY] = []
        _trigger_rerun()


def run_knowmem_evaluation(api_key, model_choice, provider) -> None:
    """Run knowmem evaluation on the queued QA examples using API."""
    ensure_qa_session_defaults()
    
    eval_examples: List[Dict[str, Any]] = st.session_state[QA_EVAL_QUEUE_KEY]
    if not eval_examples:
        st.warning("No examples in evaluation queue.")
        return
    
    if not api_key or not model_choice:
        st.error("Please configure API key and model in the sidebar.")
        return
    
    # Get ICL examples
    icl_examples: List[Dict[str, Any]] = st.session_state[QA_ICL_SESSION_KEY]
    icl_qs = [ex["question"] for ex in icl_examples]
    icl_as = [ex["answer"] for ex in icl_examples]
    
    # Prepare evaluation data
    questions = [ex["question"] for ex in eval_examples]
    answers = [ex["answer"] for ex in eval_examples]
    
    try:
        from src.copyright_detective.comparison import get_llm_completion, calculate_rouge_score, calculate_jaccard_index
        from Levenshtein import distance
        
        st.markdown("### 🧠 Running Knowmem Evaluation")
        progress_text = st.empty()
        progress_bar = st.progress(0.0)
        
        progress_text.text("Setting up evaluation...")
        progress_bar.progress(0.1)
        
        # Create logger for results
        logger = RougeEvalLogger()
        general_prompt: str = ""

        # Determine if few-shot based on mode
        qa_prompt_mode = st.session_state.get("qa_prompt_mode", "Zero-Shot")
        if qa_prompt_mode == "Few-Shot":
            # Use all predefined examples for few-shot prompting
            few_shot_examples = PREDEFINED_QA_EXAMPLES
            
            # Build few-shot prompt with all examples
            for example in few_shot_examples:
                general_prompt += f"Question: {example['question']}\nAnswer: {example['answer']}\n\n"

        progress_text.text("Running evaluation...")
        progress_bar.progress(0.3)

        max_new_tokens = int(st.session_state.get("qa_knowmem_max_new_tokens", 64) or 64)
        
        for i, (question, answer) in enumerate(zip(questions, answers)):
            prompt = general_prompt + f"Question: {question}\nAnswer: "
            
            progress_text.text(f"Generating answer for question {i+1}/{len(questions)}...")
            progress_bar.progress(0.3 + (i / len(questions)) * 0.6)
            
            # Use API to generate answer
            generated_text = get_llm_completion(
                prompt, 
                api_key, 
                model_choice, 
                provider,
                temperature=0.0,  # Deterministic for evaluation
                top_p=1.0,
                max_output_tokens=max_new_tokens,
                stop_sequences=KNOWMEM_STOP_SEQUENCES,
            )
            
            if isinstance(generated_text, str) and generated_text.startswith("Error"):
                st.error(f"❌ API error for question {i+1}: {generated_text}")
                continue
            
            trimmed_output = _trim_knowmem_completion(generated_text)
            if not trimmed_output:
                trimmed_output = generated_text.strip()
            
            # Log the result
            logger.log(prompt, answer, trimmed_output, question=question)
        
        progress_bar.progress(1.0)
        progress_text.empty()
        progress_bar.empty()
        
        # Get results
        results = logger
        
        # Display results
        st.success("✅ Knowmem evaluation completed!")
        
        # Show summary metrics
        st.markdown("#### 📊 Evaluation Results")
        
        # Get the report
        report = results.report()
        st.markdown("**Summary Metrics:**")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Mean ROUGE-1", f"{report.get('mean_rouge1', 0) * 100:.2f}%")
        with col2:
            st.metric("Mean ROUGE-2", f"{report.get('mean_rouge2', 0) * 100:.2f}%")
        with col3:
            st.metric("Mean ROUGE-L", f"{report.get('mean_rougeL', 0) * 100:.2f}%")
        
        # Display detailed results for each example
        st.markdown("**Detailed Results:**")
        for i, entry in enumerate(results.entries):
            question = entry.get('question', f'Question {i+1}')
            with st.expander(f"Example {i+1}: {question[:50]}...", expanded=False):
                st.markdown("**Question:**")
                st.write(entry.get('question', 'N/A'))
                st.markdown("**Expected Answer:**")
                st.write(entry.get('gt', 'N/A'))
                st.markdown("**Generated Answer:**")
                st.write(entry.get('pred', 'N/A'))
                st.markdown("**Metrics:**")
                st.json({
                    'rouge1': entry.get('rouge1', 0),
                    'rouge2': entry.get('rouge2', 0),
                    'rougeL': entry.get('rougeL', 0)
                })
        
    except Exception as e:
        st.error(f"❌ Error during knowmem evaluation: {str(e)}")
        import traceback
        st.code(traceback.format_exc())


def render_evaluation_queue(api_key, model_choice, provider) -> None:
    ensure_qa_session_defaults()
    eval_examples: List[Dict[str, Any]] = st.session_state[QA_EVAL_QUEUE_KEY]
    if not eval_examples:
        return

    st.markdown("##### 🧪 Evaluation batch (knowmem)")
    st.caption("These QA pairs will be evaluated together when running the local knowmem scorer.")

    for idx, example in enumerate(list(eval_examples)):
        question_preview = example["question"].split("\n", 1)[0]
        header = f"Eval {idx + 1}: {question_preview[:80]}" if question_preview else f"Eval {idx + 1}"
        with st.expander(header, expanded=False):
            st.markdown("**Question**")
            st.write(example["question"])
            st.markdown("**Answer**")
            st.write(example["answer"])
            metadata = example.get("metadata") or {}
            if metadata:
                st.markdown("**Metadata**")
                st.json(metadata)
            action_cols = st.columns((1, 1))
            with action_cols[0]:
                if st.button("Set as active QA", key=f"qa_eval_set_active_{idx}"):
                    st.session_state[QA_INPUT_SESSION_KEY] = example["question"]
                    st.session_state[QA_GROUND_SESSION_KEY] = example["answer"]
                    _trigger_rerun()
            with action_cols[1]:
                if st.button("Remove", key=f"qa_eval_remove_{idx}"):
                    eval_examples.pop(idx)
                    _trigger_rerun()

    if st.button("🚀 Run Knowmem Evaluation", key="qa_run_knowmem_eval"):
        run_knowmem_evaluation(api_key, model_choice, provider)
    
    if st.button("Clear evaluation batch", key="qa_clear_eval_batch"):
        st.session_state[QA_EVAL_QUEUE_KEY] = []
        _trigger_rerun()


def render_muse_examples_panel() -> None:
    ensure_qa_session_defaults()
    st.markdown("#### 🎓 Browse MUSE knowmem QA examples")

    try:
        df = load_cached_muse_knowmem()
    except Exception as exc:  # noqa: BLE001
        st.error(f"Failed to load the MUSE knowmem dataset: {exc}")
        if not st.session_state[QA_INPUT_SESSION_KEY]:
            st.session_state[QA_INPUT_SESSION_KEY] = "What is the capital of France?"
        if not st.session_state[QA_GROUND_SESSION_KEY]:
            st.session_state[QA_GROUND_SESSION_KEY] = "Paris"
        return

    active_meta_columns = [column for column in df.columns if column not in {"row_id", "question", "answer"}]
    title_column = next((column for column in active_meta_columns if "title" in column.lower()), None)
    if title_column:
        title_options = ["All"] + sorted({str(value) for value in df[title_column].dropna().unique().tolist()})
        selected_title = st.selectbox("Filter by title", title_options, index=0)
        if selected_title != "All":
            df = df[df[title_column].astype(str) == selected_title]

    filtered_df = df

    total_rows = len(filtered_df)
    if total_rows == 0:
        st.info("No examples match the current filters.")
        return

    max_examples = min(10, total_rows)
    sample_count = st.slider("Examples to preview", 1, max_examples, min(3, max_examples))
    sample_mode = st.radio("Sampling", ("Top", "Random"), horizontal=True)

    sample_state_key = QA_MUSE_SAMPLE_KEY_PREFIX
    if sample_mode == "Random":
        refresh = st.button("🔁 Refresh random sample", key=f"qa_refresh_random")
        if refresh or sample_state_key not in st.session_state:
            st.session_state[sample_state_key] = random.sample(range(total_rows), k=min(sample_count, total_rows))
    else:
        st.session_state[sample_state_key] = list(range(min(sample_count, total_rows)))

    indices: List[int] = st.session_state.get(sample_state_key, [])[:sample_count]
    if not indices:
        st.info("Unable to find sample rows for display.")
        return

    for display_index, row_position in enumerate(indices, start=1):
        try:
            row = filtered_df.iloc[row_position]
        except IndexError:
            continue

        question_preview = row["question"].split("\n", 1)[0]
        header = f"Example {display_index}: {question_preview[:90]}" if question_preview else f"Example {display_index}"
        with st.expander(header, expanded=False):
            st.markdown("**Question**")
            st.write(row["question"])
            st.markdown("**Answer**")
            st.write(row["answer"])

            metadata_payload = {
                column: row[column]
                for column in active_meta_columns
                if column in row and pd.notna(row[column])
            }
            if metadata_payload:
                st.markdown("**Metadata**")
                st.json(metadata_payload)

            button_cols = st.columns((1, 1, 1, 1))
            with button_cols[0]:
                if st.button("Use QA pair", key=f"qa_use_muse_{int(row['row_id'])}"):
                    st.session_state[QA_INPUT_SESSION_KEY] = row["question"]
                    st.session_state[QA_GROUND_SESSION_KEY] = row["answer"]
                    _trigger_rerun()
            with button_cols[1]:
                if st.button("Add to ICL", key=f"qa_add_icl_{int(row['row_id'])}"):
                    add_icl_example_from_row(row)
                    _trigger_rerun()
            with button_cols[2]:
                if st.button("Queue for eval", key=f"qa_add_eval_{int(row['row_id'])}"):
                    add_eval_example_from_row(row)
                    _trigger_rerun()
            with button_cols[3]:
                st.caption(f"Row #{int(row['row_id'])}")

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


def render_header():
    """Render the app header with title and description."""
    st.markdown(
        """
        <div class="app-header">
            <div class="title">🕵️‍♂️ Copyright Detective</div>
            <div class="subtitle" style="font-size: 1.1em;">Analyze and find evidence for potential text copyright infringement in LLM application</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar():
    """Render the sidebar with API configuration, model selection, and navigation."""
    with st.sidebar:
        # API Key Management
        st.markdown("### 🔑 API Configuration")
        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        openai_api_key = st.text_input("OpenAI API Key", type="password", help="Enter your OpenAI API key")
        openrouter_api_key = st.text_input(
            "OpenRouter API Key",
            type="password",
            help="Leave blank to use the built-in default key (for quick testing)",
            placeholder="Will fallback automatically if empty"
        )
        anthropic_api_key = st.text_input("Anthropic API Key", type="password", help="Enter your Anthropic API key")
        google_api_key = st.text_input("Google Gemini API Key", type="password", help="Enter your Google Gemini API key")
        st.markdown('</div>', unsafe_allow_html=True)

        # Model Selection
        st.markdown("### 🤖 Model Selection")
        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        provider = st.selectbox("Select Provider", ["OpenAI", "OpenRouter", "Anthropic", "Google Gemini"], help="Choose your AI provider")

        model_choice = None
        if provider == "OpenAI":
            model_choice = st.selectbox(
                "Choose a model",
                [
                    "gpt-3.5-turbo",
                    "gpt-3.5-turbo-instruct",
                    "gpt-4o",
                    "gpt-4o-mini",
                ],
                help="Select an OpenAI model. Perplexity probes work best with instruct-style or mini models that support logprobs.",
            )
            api_key = openai_api_key
        elif provider == "OpenRouter":
            model_choice = st.selectbox(
                "Choose a model",
                [
                    "moonshotai/kimi-k2:free",
                    "meta-llama/llama-3.1-405b-instruct:free",
                    "qwen/qwen3-235b-a22b:free",
                    "meta-llama/llama-3.3-70b-instruct:free",
                    "mistralai/mistral-small-24b-instruct-2501:free",
                    "qwen/qwen-2.5-72b-instruct:free",
                    "nvidia/nemotron-nano-9b-v2:free",
                    "microsoft/wizardlm-2-8x22b:free",
                    "google/gemma-7b-it:free",
                    "meta-llama/llama-3.2-3b-instruct:free",
                ],
            )
            api_key = openrouter_api_key.strip() if openrouter_api_key.strip() else DEFAULT_OPENROUTER_KEY
        elif provider == "Anthropic":
            model_choice = st.selectbox("Choose a model", ["claude-3-haiku-20240307", "claude-3-sonnet-20240229", "claude-3-opus-20240229"])
            api_key = anthropic_api_key
        elif provider == "Google Gemini":
            model_choice = st.selectbox("Choose a model", ["gemini-1.5-flash", "gemini-1.5-pro"])
            api_key = google_api_key
        st.markdown('</div>', unsafe_allow_html=True)

        # Detection Mode
        st.markdown("### 🧭 Detection Mode")
        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        page = st.radio(
            "Go to",
            [
                "Direct Recall Test",
                "Persuasive Jailbreak Test",
                "Unlearning Detection Test",
            ],
            label_visibility="collapsed",
        )
        st.markdown('</div>', unsafe_allow_html=True)

    return api_key, model_choice, provider, page


def render_snippet_to_document_page(api_key, model_choice, provider):
    """Render the combined snippet-to-document analysis workspace."""

    st.markdown("### 🔎 Direct Recall Test")
    st.markdown(
        "Probe AI models for copyright infringement by testing recall of text excerpts. Analyze snippets or documents using various strategies and compare outputs against ground truth with similarity metrics."
    )

    snippet_tab, pdf_tab = st.tabs([
        "Text Detection",
        "Document Detection",
    ])

    with snippet_tab:
        render_text_analysis_page(api_key, model_choice, provider, show_page_header=False)

    with pdf_tab:
        render_pdf_analysis_page(api_key, model_choice, provider, show_page_header=False)


def render_text_analysis_page(api_key, model_choice, provider, *, show_page_header: bool = True):
    """Render the text detection workflow."""

    if show_page_header:
        st.markdown("### 📝 Text Detection")
        st.markdown(
            "Analyze text snippets to detect potential copyright infringement by comparing generated text with ground truth."
        )

    # Prompt Selection (moved from sidebar to main page)
    st.markdown(
        """
        <div class=\"analysis-callout\">
            <div class=\"analysis-callout__title\">How the Direct Recall Test works</div>
            <ul class=\"analysis-callout__list\">
                <li>Provide an input snippet and the expected ground-truth passage.</li>
                <li>Select a prompting strategy to probe potential memorization.</li>
                <li>Run inference and inspect overlap metrics with side-by-side diffs.</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<p class="analysis-step-label">Step 1 · Choose recall framing</p>', unsafe_allow_html=True)
    prompt_type = st.selectbox(
        "🎛️ Choose the Recall Type:",
        [
            "Next-Passage Prediction",
            "Prior-Context Reconstruction",
            "Title Prediction",
            "QA",
        ],
        help="Select the type of prompt to guide the Text Detection. (Choose only; typing custom values is not allowed.)",
    )

    # Explanatory notes for each prompt type
    if prompt_type == "Next-Passage Prediction":
        st.markdown(
            "_Next-Passage Prediction: Provide the current excerpt and ask the model to generate the following passage. This surfaces whether the model recalls memorized continuations from source texts._"
        )
    elif prompt_type == "Prior-Context Reconstruction":
        st.markdown(
            "_Prior-Context Reconstruction: Provide the continuation or subsequent passage and ask the model to recreate the most likely preceding context. This helps reveal whether the model can recover earlier text from memory._"
        )
    elif prompt_type == "Title Prediction":
        st.markdown(
            "_Title Prediction: Based on the provided snippet, ask the model to infer the most likely title or attribution for the work. This can surface potential source identification signals._"
        )
    elif prompt_type == "QA":
        st.markdown(
            "_QA: Use Question-Answer pairs to evaluate knowledge memorization. Supports custom input for user-defined QA pairs._"
        )

    st.markdown('<p class="analysis-step-label">Step 2 · Provide comparison texts</p>', unsafe_allow_html=True)
    
    input_options = [
        "Custom Input", 
        "Example: A Tale of Two Cities", 
        "Example: Harry Potter", 
        "Example: Pride and Prejudice", 
        "Example: 1984", 
        "Example: To Kill a Mockingbird", 
        "Example: The Great Gatsby", 
        "Example: The Catcher in the Rye"
    ]
    if prompt_type == "QA":
        # Check if MUSE examples are already cached in session state
        if "muse_example_options" not in st.session_state or "muse_example_mapping" not in st.session_state:
            muse_options, muse_mapping = generate_muse_example_options(num_examples=5)
            st.session_state["muse_example_options"] = muse_options
            st.session_state["muse_example_mapping"] = muse_mapping
        else:
            muse_options = st.session_state["muse_example_options"]
            muse_mapping = st.session_state["muse_example_mapping"]
        
        input_options = ["Custom Input"] + muse_options
    
    # Determine default index for QA mode
    default_index = 0
    if prompt_type == "QA" and muse_options:
        # For QA mode, default to custom input (index 0)
        default_index = 0

    input_method = st.selectbox(
        "Select input content:",
        input_options,
        index=default_index,
        help="Select custom input or choose from examples."
    )

    examples = {
        "Example: A Tale of Two Cities": {
            "input": "It was the best of times, it was the worst of times, it was the age of wisdom,",
            "ground_truth": "it was the age of foolishness, it was the epoch of belief,"
        },
        "Example: Harry Potter": {
            "input": "Mr and Mrs Dursley, of number four, Privet Drive, were proud to say that they were perfectly normal, thank you very much.",
            "ground_truth": "They were the last people you’d expect to be involved in anything strange or mysterious, because they just didn’t hold with such nonsense."
        },
        "Example: Pride and Prejudice": {
            "input": "It is a truth universally acknowledged, that a single man in possession of a good fortune, must be in want of a wife. However little known the feelings or views of such a man may be on his",
            "ground_truth": "first entering a neighbourhood, this truth is so well fixed in the minds of the surrounding families that he is considered as the rightful property of some one or other of their daughters."
        },
        "Example: 1984": {
            "input": "It was a bright cold day in April, and the clocks were striking thirteen. Winston Smith, his chin nuzzled into his breast in an effort to escape the vile wind,",
            "ground_truth": "slipped quickly through the glass doors of Victory Mansions, though not quickly enough to prevent a swirl of gritty dust from entering along with him."
        },
        "Example: To Kill a Mockingbird": {
            "input": "When he was nearly thirteen, my brother Jem got his arm badly broken at the elbow. When it healed, and Jem's fears of never being able to play football were assuaged, he was seldom self-conscious about his injury.",
            "ground_truth": "His left arm was somewhat shorter than his right; when he stood or walked, the back of his hand was at right angles to his body, his thumb parallel to his thigh. He couldn't have cared less, so long as he could pass and punt."
        },
        "Example: The Great Gatsby": {
            "input": "Only Gatsby, the man who gives his name to this book, was exempt from my reaction—Gatsby, who represented everything for which I have an unaffected scorn. If personality is an unbroken series of successful gestures,",
            "ground_truth": "then there was something gorgeous about him, some heightened sensitivity to the promises of life, as if he were related to one of those intricate machines that register earthquakes ten thousand miles away."
        },
        "Example: The Catcher in the Rye": {
            "input": "If you really want to hear about it, the first thing you'll probably want to know is where I was born, and what my lousy childhood was like,",
            "ground_truth": "and how my parents were occupied and all before they had me, and all that David Copperfield kind of crap, but I don't feel like going into it, if you want to know the truth."
        }
    }

    # Adjust examples based on prompt type for Prior-Context Reconstruction and Title Prediction
    adjusted_examples = {}
    for key, val in examples.items():
        if prompt_type == "Prior-Context Reconstruction":
            adjusted_examples[key] = {"input": val["ground_truth"], "ground_truth": val["input"]}
        elif prompt_type == "Title Prediction":
            title = key.split(": ", 1)[1] if ": " in key else key
            adjusted_examples[key] = {"input": val["input"], "ground_truth": title}
        else:
            adjusted_examples[key] = val

    if prompt_type == "QA":
        ensure_qa_session_defaults()

        if input_method != "Custom Input" and input_method in adjusted_examples:
            example = adjusted_examples[input_method]
            st.session_state[QA_INPUT_SESSION_KEY] = example["input"]
            st.session_state[QA_GROUND_SESSION_KEY] = example["ground_truth"]
        elif input_method.startswith("Example"):
            # Handle MUSE example selection
            muse_mapping = st.session_state.get("muse_example_mapping", {})
            if input_method in muse_mapping:
                example = muse_mapping[input_method]
                st.session_state[QA_INPUT_SESSION_KEY] = example["question"]
                st.session_state[QA_GROUND_SESSION_KEY] = example["answer"]

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Question**")
            st.text_area(
                "Question",
                key=QA_INPUT_SESSION_KEY,
                height=150,
                placeholder="Enter the question you want to probe.",
                label_visibility="collapsed",
            )
        with col2:
            st.markdown("**Ground Truth**")
            st.text_area(
                "Answer",
                key=QA_GROUND_SESSION_KEY,
                height=150,
                placeholder="Enter the expected reference answer.",
                label_visibility="collapsed",
            )

        text1 = st.session_state[QA_INPUT_SESSION_KEY]
        text2 = st.session_state[QA_GROUND_SESSION_KEY]
        
        # Add zero-shot/few-shot selector
        qa_prompt_mode = st.selectbox(
            "Choose Zero-Shot/Few-Shot:",
            ["Zero-Shot", "Few-Shot"],
            index=["Zero-Shot", "Few-Shot"].index(st.session_state.get("qa_prompt_mode", "Zero-Shot")),
            help="Select 'Zero-Shot' for no examples or 'Few-Shot' for including example demonstrations in the prompt.",
            key="qa_prompt_mode_selector",
        )
        st.session_state["qa_prompt_mode"] = qa_prompt_mode

        render_selected_icl_examples()
        render_evaluation_queue(api_key, model_choice, provider)

    else:
        if input_method == "Custom Input":
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Input Text**")
                text1 = st.text_area(
                    "Input Text",
                    height=150,
                    placeholder="Enter the input snippet (e.g., a previous sentence, a continuation, or an excerpt). The role of this field depends on the selected prompt type.",
                    label_visibility="collapsed",
                )
            with col2:
                st.markdown("**Ground Truth**")
                text2 = st.text_area(
                    "Ground Truth",
                    height=150,
                    placeholder="Enter the ground truth text or expected target to compare against (e.g., the known reference or target continuation). Leave blank if not applicable.",
                    label_visibility="collapsed",
                )
        else:
            example = adjusted_examples[input_method]
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Input Text**")
                text1 = st.text_area(
                    "Input Text",
                    value=example["input"],
                    height=150,
                    label_visibility="collapsed",
                )
            with col2:
                st.markdown("**Ground Truth**")
                text2 = st.text_area(
                    "Ground Truth",
                    value=example["ground_truth"],
                    height=150,
                    label_visibility="collapsed",
                )

    input_word_count = len(text1.split()) if text1 else 0
    ground_word_count = len(text2.split()) if text2 else 0
    input_char_count = len(text1) if text1 else 0
    ground_char_count = len(text2) if text2 else 0

    # Explanatory notes for each prompt type
    if prompt_type == "Next-Passage Prediction":
        
        col1, col2 = st.columns(2)
        with col1:
            continuation_method = st.selectbox(
                "Choose a Prompting Method:",
                CONTINUATION_STRATEGIES,
                help="Select 'Normal Continuation' for a direct prompt or a persuasion strategy to frame the request differently.",
                key="continuation_method_selector",
            )
        with col2:
            prompt_mode = st.selectbox(
                "Choose Zero-Shot/Few-Shot:",
                ["Zero-Shot", "Few-Shot"],
                help="Select 'Zero-Shot' for no examples or 'Few-Shot' for including example demonstrations in the prompt.",
                key="prompt_mode_selector",
            )

        custom_continuation_prompt: Optional[str] = None
        if continuation_method == "Custom Prompt":
            custom_continuation_prompt = st.text_area(
                "Custom prompt template",
                height=180,
                placeholder="Write the full instruction the model should follow. Use {input_text} where the snippet should appear. Optional placeholders: {word_count}, {char_count}.",
                key="custom_continuation_prompt",
                help="This template replaces the built-in continuation prompt. It should contain {input_text} so the snippet is inserted correctly.",
            )
            st.caption("Tip: Include placeholders like {input_text}, {word_count}, or {char_count} to auto-fill the preview values.")
        else:
            custom_continuation_prompt = st.session_state.get("custom_continuation_prompt", "")

        # Immediately preview the prompt after selecting the continuation method
        # Use placeholder text if the input or ground truth is empty
        chunk_size_preview = len(text2.split()) if text2 else None
        char_count_preview = len(text2) if text2 else None
        prompt_to_preview = get_full_prompt(
            prompt_type="Next-Passage Prediction",
            input_text=text1,
            chunk_size=chunk_size_preview,
            continuation_method=continuation_method,
            char_count=char_count_preview,
            custom_template=custom_continuation_prompt if continuation_method == "Custom Prompt" else None,
            mode=prompt_mode,
        )
        st.markdown(
            "ℹ️ The length of the generated text will be adjusted to match the character count of your **Ground Truth** input."
        )
        # Render preview immediately so users can confirm the exact prompt that will be sent
        render_prompt_preview(prompt_to_preview)
        
    elif prompt_type == "Prior-Context Reconstruction":
        preceding_method = st.selectbox(
            "Choose a prompting method:",
            CONTINUATION_STRATEGIES,
            help="Select a reconstruction framing. Each strategy nudges the model toward recreating the missing preceding context.",
            key="preceding_method_selector",
        )

        custom_preceding_prompt: Optional[str] = None
        if preceding_method == "Custom Prompt":
            custom_preceding_prompt = st.text_area(
                "Custom prompt template",
                height=180,
                placeholder="Describe how the model should reconstruct the preceding context. Use {input_text} for the continuation and {word_count}/{char_count} if needed.",
                key="custom_preceding_prompt",
                help="Your custom template replaces the selected strategy. Remember to include {input_text} to reference the continuation snippet.",
            )
            st.caption("Tip: Use {char_count} or {word_count} to remind the model of desired length.")
        else:
            custom_preceding_prompt = st.session_state.get("custom_preceding_prompt", "")
        chunk_size_preview = len(text2.split()) if text2 else None
        char_count_preview = len(text2) if text2 else None
        prompt_to_preview = get_full_prompt(
            prompt_type,
            text1,
            chunk_size=chunk_size_preview,
            continuation_method=preceding_method,
            char_count=char_count_preview,
            custom_template=custom_preceding_prompt if preceding_method == "Custom Prompt" else None,
        )
        st.markdown(
            "ℹ️ The length of the generated text will be adjusted to match the character count of your **Ground Truth** input."
        )
        # Show the preview immediately after the continuation method selection so users can edit if needed
        render_prompt_preview(prompt_to_preview)

    elif prompt_type == "Title Prediction":
        chunk_size_preview = len(text2.split()) if text2 else None
        char_count_preview = len(text2) if text2 else None
        prompt_to_preview = get_full_prompt(
            prompt_type,
            text1,
            chunk_size=chunk_size_preview,
            char_count=char_count_preview,
        )
        st.markdown(
            "ℹ️ The length of the generated text will be adjusted to match the character count of your **Ground Truth** input."
        )
        render_prompt_preview(prompt_to_preview)

    elif prompt_type == "QA":
        # Check if few-shot mode is selected
        qa_prompt_mode = st.session_state.get("qa_prompt_mode", "Zero-Shot")
        if qa_prompt_mode == "Few-Shot":
            # Build few-shot prompt preview with all predefined examples
            prompt_to_preview = ""
            for example in PREDEFINED_QA_EXAMPLES:
                prompt_to_preview += f"Question: {example['question']}\nAnswer: {example['answer']}\n\n"
            prompt_to_preview += f"Question: {text1}\nAnswer:"
        else:
            prompt_to_preview = f"Question: {text1}\nAnswer:"
        
        st.markdown(
            "ℹ️ The model will generate an answer to the question and it will be compared against the ground truth answer."
        )
        render_prompt_preview(prompt_to_preview)

    # Prompt Preview - This is now handled within each prompt_type section
    # if text1:
    #     # Define continuation_method for the preview logic even if it's not set
    #     continuation_method = "Normal Continuation"
    #     chunk_size = len(text2.split()) if text2 else None
    #     if prompt_type == "Next-Passage Prediction":
    #         continuation_method = st.session_state.get("continuation_method_selector", "Normal Continuation")
    #         prompt_to_preview = get_full_prompt(continuation_method, text1, chunk_size=chunk_size)
    #     else:
    #         prompt_to_preview = get_full_prompt(prompt_type, text1, chunk_size=chunk_size)

    #     prompt_preview(prompt_to_preview)

    st.divider()
    st.markdown('<p class="analysis-step-label">Step 3 · Configure generation</p>', unsafe_allow_html=True)

    st.markdown(
        '<p class="analysis-step-caption">Adjust the number of inference passes and how exploratory the sampling should be.</p>',
        unsafe_allow_html=True,
    )
    col1, col2, col3 = st.columns(3)
    with col1:
        inference_runs = st.number_input(
            "Number of Inference Runs",
            min_value=1,
            max_value=100,
            value=1,
            step=1,
            help="Specify how many times to run the inference for statistical analysis.",
        )
    with col2:
        temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=2.0,
            value=0.7,
            step=0.01,
            help="Controls randomness. Lower values make the model more deterministic.",
        )
    with col3:
        top_p = st.slider(
            "Top-P",
            min_value=0.0,
            max_value=1.0,
            value=1.0,
            step=0.01,
            help="Controls diversity via nucleus sampling. 0.5 means half of all likelihood-weighted options are considered.",
        )


    st.caption("Provide both snippets and an API key, then launch the run to view overlap diagnostics.")

    run_analysis = st.button("🚀 Run Analysis", key="run_snippet_analysis_button", use_container_width=True)

    if run_analysis:
        if not api_key:
            st.error(f"⚠️ Please enter your API key in the sidebar.")
        elif not text1 or not text2:
            st.warning("⚠️ Please enter both input text and ground truth.")
        else:
            # Define a variable for continuation_method if it's not set
            if prompt_type == "Prior-Context Reconstruction":
                continuation_method = st.session_state.get("preceding_method_selector", "Normal Continuation")
                custom_template = (
                    st.session_state.get("custom_preceding_prompt", "").strip()
                    if continuation_method == "Custom Prompt"
                    else None
                )
            else:
                continuation_method = st.session_state.get("continuation_method_selector", "Normal Continuation")
                custom_template = (
                    st.session_state.get("custom_continuation_prompt", "").strip()
                    if continuation_method == "Custom Prompt"
                    else None
                )
            prompt_mode = st.session_state.get("prompt_mode_selector", "Zero-Shot")
            target_char_count = len(text2)
            chunk_size = len(text2.split())

            if continuation_method == "Custom Prompt" and not custom_template:
                st.error("⚠️ Please provide a custom prompt template before running the analysis.")
                return

            if inference_runs == 1:
                # Single run: Original Analysis Results
                with st.spinner(
                    f"🔄 Generating text with {model_choice} and calculating scores..."
                ):
                    if prompt_type == "QA":
                        # Check if few-shot mode is selected for QA
                        qa_prompt_mode = st.session_state.get("qa_prompt_mode", "Zero-Shot")
                        if qa_prompt_mode == "Few-Shot":
                            # Use all predefined examples for few-shot prompting
                            few_shot_examples = PREDEFINED_QA_EXAMPLES
                            
                            # Build few-shot prompt with all examples
                            general_prompt = ""
                            for example in few_shot_examples:
                                general_prompt += f"Question: {example['question']}\nAnswer: {example['answer']}\n\n"
                            
                            prompt = general_prompt + f"Question: {text1}\nAnswer:"
                        else:
                            prompt = f"Question: {text1}\nAnswer:"
                        generated_text = get_llm_completion(
                            prompt,
                            api_key,
                            model_choice,
                            provider=provider,
                            temperature=temperature,
                            top_p=top_p
                        )
                        if isinstance(generated_text, str) and generated_text.startswith("Error"):
                            result = generated_text
                        else:
                            trimmed_output = _trim_knowmem_completion(generated_text) or generated_text.strip()
                            final_output = trimmed_output
                            if prompt_type not in {"Title Prediction", "QA"}:
                                final_output = enforce_exact_char_count(trimmed_output, target_char_count)
                            # Calculate similarity metrics using RougeEvalLogger for consistency
                            from src.metrics.logger import RougeEvalLogger
                            logger = RougeEvalLogger()
                            logger.log(prompt, text2, final_output, question=text1)
                            report = logger.report()
                            metrics_map = {
                                "rouge_1": report.get('mean_rouge1', 0),
                                "rouge_l": report.get('mean_rougeL', 0),
                                "jaccard_index": calculate_jaccard_index(final_output, text2),
                                "levenshtein": distance(final_output, text2)
                            }
                            result = (final_output, metrics_map)
                    elif prompt_type == "Next-Passage Prediction" and continuation_method != "Normal Continuation":
                        result = run_persuasion_probe(
                            api_key,
                            model_choice,
                            provider,
                            continuation_method,
                            text1,
                            text2,
                            chunk_size=chunk_size,
                            temperature=temperature,
                            top_p=top_p,
                            custom_template=custom_template,
                            mode=prompt_mode,
                        )
                    else:
                        result = compare_texts(
                            text1,
                            text2,
                            api_key,
                            model_name=model_choice,
                            provider=provider,
                            prompt_type=prompt_type,
                            chunk_size=chunk_size,
                            temperature=temperature,
                            top_p=top_p,
                            continuation_method=continuation_method,
                            custom_template=custom_template,
                            mode=prompt_mode,
                        )
                    
                    # Handle potential errors from both functions
                    error_occurred = False
                    if isinstance(result, str) and result.startswith("Error"):
                        st.error(f"❌ {result}")
                        error_occurred = True

                    if not error_occurred:
                        generated_text, metrics = result
                        metrics_map = metrics or {}
                        rouge_score = float(metrics_map.get("rouge_l", 0.0) or 0.0)
                        jaccard_index = float(metrics_map.get("jaccard_index", 0.0) or 0.0)
                        if prompt_type not in {"Title Prediction", "QA"}:
                            generated_text = enforce_exact_char_count(generated_text, target_char_count)

                        # Results section
                        st.divider()
                        st.markdown('<p class="analysis-step-label">Results</p>', unsafe_allow_html=True)
                        st.markdown("### 📊 Analysis Results")
                        st.caption(
                            "Metrics reported: ROUGE-1, ROUGE-L, LCS (character/word), ACS (word), Levenshtein distance, semantic similarity, MinHash similarity, and Jaccard index."
                        )

                        # Highlighted overlap view
                        st.markdown("**🧠 Direct Recall Overlap**")
                        render_direct_recall_diff(text2, generated_text, metrics=metrics_map)

                        # Conclusion
                        if rouge_score > 0.5 or jaccard_index > 0.5:
                            st.success(
                                "🎯 **High similarity detected!** This may indicate potential copyright concerns."
                            )
                        else:
                            st.info(
                                "✅ **Low to moderate similarity.** The generated text appears sufficiently different."
                            )
            else:
                # Multiple runs: Inference Results Over Multiple Runs
                st.divider()
                st.markdown('<p class="analysis-step-label">Results</p>', unsafe_allow_html=True)
                st.markdown('<h3 class="multi-run-title">🔄 Inference Results Over Multiple Runs</h3>', unsafe_allow_html=True)
                similarity_scores = []
                generated_texts = []  # Store generated texts for each run
                progress_bar = st.progress(0, text="Starting inference runs...")
                for i in range(inference_runs):
                    progress_bar.progress(
                        (i) / inference_runs,
                        text=f"🔄 Generating text for run {i+1}/{inference_runs}...",
                    )
                    if prompt_type == "QA":
                        # For QA, generate answer
                        prompt = f"Question: {text1}\nAnswer:"
                        generated_text = get_llm_completion(
                            prompt,
                            api_key,
                            model_choice,
                            provider=provider,
                            temperature=temperature,
                            top_p=top_p
                        )
                        if isinstance(generated_text, str) and generated_text.startswith("Error"):
                            result = generated_text
                        else:
                            trimmed_output = _trim_knowmem_completion(generated_text) or generated_text.strip()
                            final_output = trimmed_output
                            if prompt_type not in {"Title Prediction", "QA"}:
                                final_output = enforce_exact_char_count(trimmed_output, target_char_count)
                            # Calculate metrics using RougeEvalLogger for consistency
                            from src.metrics.logger import RougeEvalLogger
                            logger = RougeEvalLogger()
                            logger.log(prompt, text2, final_output, question=text1)
                            report = logger.report()
                            metrics_map = {
                                "rouge_1": report.get('mean_rouge1', 0),
                                "rouge_l": report.get('mean_rougeL', 0),
                                "jaccard_index": calculate_jaccard_index(final_output, text2),
                                "levenshtein": distance(final_output, text2)
                            }
                            result = (final_output, metrics_map)
                    elif prompt_type == "Next-Passage Prediction" and continuation_method != "Normal Continuation":
                        result = run_persuasion_probe(
                            api_key,
                            model_choice,
                            provider,
                            continuation_method,
                            text1,
                            text2,
                            chunk_size=chunk_size,
                            temperature=temperature,
                            top_p=top_p,
                            custom_template=custom_template,
                            mode=prompt_mode,
                        )
                    else:
                        result = compare_texts(
                            text1,
                            text2,
                            api_key,
                            model_name=model_choice,
                            provider=provider,
                            prompt_type=prompt_type,
                            chunk_size=chunk_size,
                            temperature=temperature,
                            top_p=top_p,
                            continuation_method=continuation_method,
                            custom_template=custom_template,
                            mode=prompt_mode,
                        )

                    # Handle potential errors from both functions
                    error_occurred = False
                    if isinstance(result, str) and result.startswith("Error"):
                        st.error(f"❌ {result}")
                        error_occurred = True
                        break
                    
                    if not error_occurred:
                        generated_text, metrics = result
                        metrics_map = metrics or {}
                        if prompt_type not in {"Title Prediction", "QA"}:
                            generated_text = enforce_exact_char_count(generated_text, target_char_count)
                        similarity_scores.append(dict(metrics_map))
                        generated_texts.append(generated_text)
                
                progress_bar.progress(1.0, text="✅ All runs completed!")

                if similarity_scores:
                    # Display generated texts for each run
                    st.markdown('<h3 class="section-header sm">🤖 Generated Texts for Each Run</h3>', unsafe_allow_html=True)
                    st.caption(
                        "Each run reports ROUGE-1, ROUGE-L, LCS (character/word), ACS (word), Levenshtein distance, semantic similarity, MinHash similarity, and Jaccard index."
                    )
                    for i, text in enumerate(generated_texts):
                        metrics_for_run = similarity_scores[i] if i < len(similarity_scores) else {}
                        with st.expander(f"Run {i+1}", expanded=False):
                            render_direct_recall_diff(text2, text, title=f"Run {i+1}", metrics=metrics_for_run)

                    metrics_df = pd.DataFrame(similarity_scores).apply(pd.to_numeric, errors="coerce")

                    if not metrics_df.empty:
                        # Set index to start from 1 instead of 0
                        metrics_df.index = range(1, len(metrics_df) + 1)
                        st.markdown('<h4 class="section-header sm">📄 Run Metrics Overview</h4>', unsafe_allow_html=True)
                        column_order = [
                            "rouge_l",
                            "rouge_1",
                            "jaccard_index",
                            "lcs_char_ratio",
                            "lcs_char_length",
                            "lcs_word_ratio",
                            "lcs_word_length",
                            "acs_word",
                            "semantic_similarity",
                            "minhash_similarity",
                            "levenshtein",
                        ]
                        available_columns = [col for col in column_order if col in metrics_df.columns]
                        if available_columns:
                            st.dataframe(metrics_df[available_columns].round(4))
                        else:
                            st.dataframe(metrics_df.round(4))

                        summary_labels = [
                            ("rouge_l", "ROUGE-L"),
                            ("rouge_1", "ROUGE-1"),
                            ("jaccard_index", "Jaccard"),
                            ("lcs_char_ratio", "LCS (Character)"),
                            ("lcs_word_ratio", "LCS (Word)"),
                            ("acs_word", "ACS (Word)"),
                            ("levenshtein", "Levenshtein"),
                            ("semantic_similarity", "Semantic Similarity"),
                            ("minhash_similarity", "MinHash Similarity"),
                        ]

                        summary_rows = []
                        for key, label in summary_labels:
                            if key not in metrics_df.columns:
                                continue
                            series = metrics_df[key].dropna()
                            if series.empty:
                                continue
                            summary_rows.append(
                                {
                                    "Metric": label,
                                    "Min": float(series.min()),
                                    "Max": float(series.max()),
                                    "Avg": float(series.mean()),
                                }
                            )

                        st.markdown("---")
                        st.markdown('<h3 class="section-header sm">📊 Statistical Results</h3>', unsafe_allow_html=True)
                        if summary_rows:
                            summary_df = pd.DataFrame(summary_rows).set_index("Metric")
                            st.dataframe(summary_df.round(4))
                        else:
                            st.info("No similarity statistics could be computed for the current runs.")

                        plot_df = metrics_df.fillna(0.0)
                        rouge_scores = plot_df.get("rouge_l", pd.Series([0.0] * len(plot_df))).tolist()
                        jaccard_scores = plot_df.get("jaccard_index", pd.Series([0.0] * len(plot_df))).tolist()
                        levenshtein_scores = plot_df.get("levenshtein", pd.Series([0.0] * len(plot_df))).tolist()

                        fig, ax = plt.subplots(1, 3, figsize=(15, 5))

                        ax[0].plot(rouge_scores, marker='o', label='ROUGE-L')
                        ax[0].set_title('ROUGE-L Scores')
                        ax[0].set_xlabel('Run')
                        ax[0].set_ylabel('Score')
                        ax[0].legend()

                        ax[1].plot(jaccard_scores, marker='o', label='Jaccard Index', color='orange')
                        ax[1].set_title('Jaccard Index')
                        ax[1].set_xlabel('Run')
                        ax[1].set_ylabel('Score')
                        ax[1].legend()

                        ax[2].plot(levenshtein_scores, marker='o', label='Levenshtein Distance', color='green')
                        ax[2].set_title('Levenshtein Distance')
                        ax[2].set_xlabel('Run')
                        ax[2].set_ylabel('Distance')
                        ax[2].legend()

                        st.pyplot(fig)

                        additional_cols = [
                            column
                            for column in ["semantic_similarity", "minhash_similarity", "acs_word", "lcs_word_ratio"]
                            if column in plot_df.columns
                        ]
                        if additional_cols:
                            st.markdown('<h4 class="section-header sm">📈 Additional Metric Trends</h4>', unsafe_allow_html=True)
                            st.line_chart(plot_df[additional_cols])

                    # Output stability metrics
                    st.markdown("---")
                    st.markdown('<h3 class="diversity-title">🌈 Output Diversity Diagnostics</h3>', unsafe_allow_html=True)
                    st.markdown('<div class="diversity-diagnostics">', unsafe_allow_html=True)

                    unique_counts = Counter(generated_texts)
                    total_runs = len(generated_texts)
                    probabilities = [count / total_runs for count in unique_counts.values()]
                    entropy_bits = -sum(p * math.log(p, 2) for p in probabilities if p > 0)
                    max_entropy = math.log(len(unique_counts), 2) if len(unique_counts) > 1 else 0.0
                    normalized_entropy = (entropy_bits / max_entropy) if max_entropy > 0 else 0.0
                    max_probability = max(probabilities) if probabilities else 0.0

                    diversity_metrics = [
                        {
                            "label": "Unique Variants",
                            "value": f"{len(unique_counts)}",
                            "detail": "Distinct generations observed",
                        },
                        {
                            "label": "Entropy (bits)",
                            "value": f"{entropy_bits:.3f}",
                            "detail": "Shannon entropy across unique outputs",
                        },
                        {
                            "label": "Top Probability",
                            "value": f"{max_probability:.3f}",
                            "detail": "Share of the most common generation",
                        },
                    ]

                    metrics_rows = "\n".join(
                        (
                            "<tr>"
                            f"<td class=\"diversity-metrics-label\">{metric['label']}</td>"
                            f"<td class=\"diversity-metrics-value\">{metric['value']}</td>"
                            f"<td class=\"diversity-metrics-detail\">{metric['detail']}</td>"
                            "</tr>"
                        )
                        for metric in diversity_metrics
                    )

                    table_html = textwrap.dedent(
                        f"""
                        <div class=\"diversity-metrics-card\">
                            <table class=\"diversity-metrics-table\">
                                <thead>
                                    <tr>
                                        <th>Metric</th>
                                        <th>Value</th>
                                        <th>Details</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    {metrics_rows}
                                </tbody>
                            </table>
                        </div>
                        """
                    ).strip()

                    st.markdown(table_html, unsafe_allow_html=True)

                    st.caption(
                        f"Normalized entropy: {normalized_entropy * 100:.1f}% of the theoretical maximum for {len(unique_counts)} unique outputs."
                    )

                    if unique_counts:
                        top_k_limit = min(5, len(unique_counts))
                        most_common = unique_counts.most_common(top_k_limit)
                        top_k_records = []
                        for rank, (sample_text, count) in enumerate(most_common, start=1):
                            probability = count / total_runs
                            preview = sample_text.strip()
                            if len(preview) > 120:
                                preview = preview[:117].rstrip() + "…"
                            top_k_records.append(
                                {
                                    "Rank": rank,
                                    "Frequency": count,
                                    "Probability": probability,
                                    "Sample Preview": preview,
                                }
                            )

                        st.markdown('<h4 class="diversity-subtitle">Top Sample Distribution</h4>', unsafe_allow_html=True)
                        render_top_sample_distribution(top_k_records)

                        labels = []
                        for record in top_k_records:
                            preview_label = record["Sample Preview"]
                            if preview_label:
                                labels.append(f"#{record['Rank']} {preview_label}")
                            else:
                                labels.append(f"#{record['Rank']}")
                        prob_series = pd.Series(
                            [count / total_runs for _, count in most_common],
                            index=labels,
                        )
                        st.bar_chart(prob_series, height=260)

                    st.markdown("</div>", unsafe_allow_html=True)

                    if total_runs >= 3 and (normalized_entropy < 0.3 or max_probability > 0.6):
                        st.warning(
                            "⚠️ Observed low output entropy or high mode concentration across runs. Stable generations may suggest residual memorization — consider increasing temperature or probing with alternative prompts."
                        )

                    st.markdown('</div>', unsafe_allow_html=True)



    st.markdown("---")
    # The Jailbreak Persuasion Probe section is now integrated above.
    # render_jailbreak_persuasion_probe_section(api_key, model_choice, provider)


def render_pdf_analysis_page(api_key, model_choice, provider, *, show_page_header: bool = True):
    """Render the document-scale PDF analysis workflow."""

    if show_page_header:
        # Page header with clear cache button
        header_col, button_col = st.columns([4, 1])
        with header_col:
            st.markdown("### 📄 Document Detection")
            st.markdown(
                "Upload a whole PDF document to automatically analyze text chunks for potential copyright infringement."
            )
        with button_col:
            if st.button("🗑️ Clear Cache", key="clear_pdf_cache", help="Remove cached PDF analysis results"):
                st.session_state.pop("pdf_analysis_results", None)
                st.session_state.pop("pdf_analysis_score_type", None)
                st.session_state.pop("pdf_analysis_top_k", None)
                st.session_state.pop("pdf_analysis_continuation_method", None)
                st.session_state.pop("pdf_analysis_temperature", None)
                st.session_state.pop("pdf_analysis_top_p", None)
                rerun_fn = getattr(st, "rerun", None)
                if callable(rerun_fn):
                    rerun_fn()
                else:
                    experimental_rerun = getattr(st, "experimental_rerun", None)
                    if callable(experimental_rerun):
                        experimental_rerun()

    # Initialize variables to avoid UnboundLocalError
    score_type = None
    top_k = None
    chunk_size = None
    continuation_method = None
    temperature = None
    top_p = None
    custom_pdf_prompt = None

    def render_pdf_results_section(
        results_data: List[Tuple[str, str, str, Dict[str, float]]],
        *,
        default_score_type: str,
        default_top_k: int,
        continuation_method: str,
        temperature: float,
        top_p: float,
    ) -> None:
        """Render ranked PDF chunk results with adjustable controls."""

        if not results_data:
            st.info("No comparable chunks were produced for ranking.")
            return

        metrics_options = [
            "ROUGE-L",
            "ROUGE-1",
            "Jaccard Index",
            "LCS (Character)",
            "LCS (Word)",
            "ACS (Word)",
            "Semantic Similarity",
            "MinHash Similarity",
            "Levenshtein Distance",
        ]

        # Seed widget defaults from session state when available
        current_score_type = st.session_state.get("pdf_analysis_score_type", default_score_type) or default_score_type
        if current_score_type not in metrics_options:
            current_score_type = metrics_options[0]

        current_top_k = st.session_state.get("pdf_analysis_top_k", default_top_k)
        if not isinstance(current_top_k, int) or current_top_k < 1:
            current_top_k = max(1, int(default_top_k or 5))

        st.markdown("---")
        col_rank1, col_rank2 = st.columns(2)
        with col_rank1:
            display_score_type = st.selectbox(
                "Ranking Metric",
                metrics_options,
                index=metrics_options.index(current_score_type),
                help="Choose how to rank the most similar sections",
                key="display_score_type",
            )
            st.session_state["pdf_analysis_score_type"] = display_score_type

        with col_rank2:
            display_top_k = st.number_input(
                "Display Count",
                min_value=1,
                max_value=20,
                value=min(max(current_top_k, 1), 20),
                step=1,
                help="Select how many of the highest scoring chunks to show",
                key="display_top_k",
            )
            st.session_state["pdf_analysis_top_k"] = int(display_top_k)

        score_mapping = {
            "ROUGE-L": ("rouge_l", True),
            "ROUGE-1": ("rouge_1", True),
            "Jaccard Index": ("jaccard_index", True),
            "LCS (Character)": ("lcs_char_ratio", True),
            "LCS (Word)": ("lcs_word_ratio", True),
            "ACS (Word)": ("acs_word", True),
            "Semantic Similarity": ("semantic_similarity", True),
            "MinHash Similarity": ("minhash_similarity", True),
            "Levenshtein Distance": ("levenshtein", False),
        }
        metric_key, descending = score_mapping.get(display_score_type, ("rouge_l", True))

        # Work on a copy to avoid mutating session state accidentally
        sorted_results = sorted(
            results_data,
            key=lambda entry: float(entry[3].get(metric_key, float("-inf") if descending else float("inf"))),
            reverse=descending,
        )

        final_display_limit = min(int(display_top_k), len(sorted_results))

        st.markdown(f"#### 🏆 Top {final_display_limit} Most Similar Sections")
        st.caption(
            f"Ranking by {display_score_type}. Showing top {final_display_limit} of {len(sorted_results)} chunks. "
            f"Generation strategy: {continuation_method} · Temperature {temperature:.2f} · Top-P {top_p:.2f}.\n"
            "Metrics tracked: ROUGE-1, ROUGE-L, LCS (character/word), ACS (word), Levenshtein distance, semantic similarity, MinHash similarity, and Jaccard index."
        )

        for rank, (upper, lower, gen, metric_values) in enumerate(sorted_results[:final_display_limit], start=1):
            metrics_for_display = metric_values or {}
            rouge_l = float(metrics_for_display.get("rouge_l", 0.0) or 0.0)
            jaccard = float(metrics_for_display.get("jaccard_index", 0.0) or 0.0)
            levenshtein_val = int(metrics_for_display.get("levenshtein", 0.0) or 0.0)
            with render_streamlit_accordion(
                f"Rank {rank}",
                key=f"pdf_top_section_{rank}",
                expanded=False,
            ):
                st.markdown("**📝 Prefix Context**")
                st.markdown(
                    f"""
                    <div style="
                        background: rgba(255, 255, 255, 0.85);
                        border: 1px solid rgba(191, 219, 254, 0.6);
                        border-left: 4px solid #2563eb;
                        border-radius: 12px;
                        padding: 0.65rem 0.75rem;
                        font-size: 0.9rem;
                        line-height: 1.7;
                        color: #1f2937;
                        white-space: pre-wrap;
                        word-break: break-word;
                        margin: 0.5rem 0;
                    ">
                    {html.escape(upper)}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                st.markdown("**🧠 Direct Recall Overlap**")
                render_direct_recall_diff(
                    lower,
                    gen,
                    title="Ground Truth vs. Generated Output",
                    metrics=metrics_for_display,
                )

    uploaded_file = st.file_uploader("📎 Choose a PDF file:", type="pdf", help="Select a PDF document to analyze")
    if uploaded_file is not None:
        st.markdown('<h3 class="section-header sm">⚙️ Analysis Configuration</h3>', unsafe_allow_html=True)
        config_col1, config_col2 = st.columns(2)
        with config_col1:
            chunk_size = st.number_input(
                'Change Chunk Size (words):',
                min_value=50,
                max_value=2000,
                value=200,
                step=25,
                help='Number of words per text chunk'
            )
        with config_col2:
            continuation_method = st.selectbox(
                'Choose a Prompting Method:',
                CONTINUATION_STRATEGIES,
                index=0,
                help='Pick how the model should be nudged when generating chunk continuations. "Normal Continuation" keeps the default behaviour.',
                key='pdf_continuation_method'
            )

        custom_pdf_prompt = None
        if continuation_method == "Custom Prompt":
            custom_pdf_prompt = st.text_area(
                "Custom prompt template",
                height=180,
                placeholder="Write the instruction to use for each PDF chunk. Include {input_text} where the chunk should appear (e.g., '[PDF chunk]'). Optional placeholders: {word_count}, {char_count}.",
                key="pdf_custom_prompt",
                help="This template overrides the built-in strategies when analyzing PDF chunks.",
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
        preview_prompt = get_full_prompt(
            prompt_type="Next-Passage Prediction",
            input_text="[PDF chunk]",
            chunk_size=chunk_size,
            continuation_method=continuation_method,
            custom_template=preview_custom_template,
        )
        render_prompt_preview(preview_prompt)

        st.markdown('<h3 class="section-header sm">🛠️ Generation Controls</h3>', unsafe_allow_html=True)
        ctrl_col1, ctrl_col2 = st.columns(2)
        with ctrl_col1:
            temperature = st.slider(
                'Temperature',
                min_value=0.0,
                max_value=2.0,
                value=0.7,
                step=0.01,
                help='Controls randomness. Lower values make the model more deterministic.',
                key='pdf_temperature'
            )
        with ctrl_col2:
            top_p = st.slider(
                'Top-P',
                min_value=0.0,
                max_value=1.0,
                value=1.0,
                step=0.01,
                help='Controls nucleus sampling diversity. 0.5 considers the top 50% probability mass.',
                key='pdf_top_p'
            )

    else:
        score_type = "ROUGE-L"  # Default ranking metric
        chunk_size = None
        continuation_method = "Normal Continuation"
        temperature = 0.7
        top_p = 1.0
        top_k = 5  # Default number of ranks to display
        custom_pdf_prompt = ""

    if uploaded_file is not None:
        st.markdown("---")
        analyze_pdf = st.button(
            "🔍 Analyze PDF",
            width='stretch',
            type="primary",
            key="analyze_pdf_button",
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
    else:
        analyze_pdf = False

    if analyze_pdf:
        # Set default values for ranking parameters
        if score_type is None:
            score_type = "ROUGE-L"
        if top_k is None:
            top_k = 5
            
        if not api_key:
            st.error("⚠️ Please enter your API key in the sidebar.")
            return
        custom_template = None
        if continuation_method == "Custom Prompt":
            custom_template = (custom_pdf_prompt or "").strip()
            if not custom_template:
                st.error("⚠️ Please provide a custom prompt template before running the analysis.")
                return

        try:
            progress_bar = st.progress(0, text=f"🔄 Analyzing PDF with {model_choice}... Preparing document...")
            pdf_buffer = io.BytesIO(uploaded_file.getvalue())
            pdf_text = extract_text_from_pdf(pdf_buffer)
            if "Error" in pdf_text:
                st.error(f"❌ {pdf_text}")
                return
            chunk_pairs = split_text_into_chunks(pdf_text, chunk_size=chunk_size)
            if not chunk_pairs:
                st.warning("⚠️ Could not split the PDF into enough text chunks for analysis.")
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
                    )
                if isinstance(result, str) and result.startswith("Error"):
                    st.error(f"❌ {result}")
                    return

                generated_text, metrics = result
                metrics_map = metrics or {}
                results.append((upper, lower, generated_text, dict(metrics_map)))
                progress_bar.progress((i + 1)/total, text=f"🔄 Processing chunk {i+1}/{total} · {continuation_method}")

            # Store results in session state for post-analysis adjustment and subsequent reruns
            st.session_state["pdf_analysis_results"] = list(results)
            st.session_state["pdf_analysis_score_type"] = score_type
            st.session_state["pdf_analysis_top_k"] = top_k
            st.session_state["pdf_analysis_continuation_method"] = continuation_method
            st.session_state["pdf_analysis_temperature"] = temperature
            st.session_state["pdf_analysis_top_p"] = top_p

            render_pdf_results_section(
                results,
                default_score_type=score_type,
                default_top_k=top_k,
                continuation_method=continuation_method,
                temperature=temperature,
                top_p=top_p,
            )

            progress_bar.progress(1.0, text=f"✅ Completed analysis with {model_choice}. Processed {total} chunks.")
        except Exception as e:
            st.error(f"❌ Error during analysis: {e}")

    elif st.session_state.get("pdf_analysis_results"):
        cached_results = st.session_state.get("pdf_analysis_results") or []
        cached_score_type = st.session_state.get("pdf_analysis_score_type", "ROUGE-L")
        cached_top_k = st.session_state.get("pdf_analysis_top_k", 5)
        cached_continuation_method = st.session_state.get("pdf_analysis_continuation_method", "Normal Continuation")
        cached_temperature = st.session_state.get("pdf_analysis_temperature", 0.7)
        cached_top_p = st.session_state.get("pdf_analysis_top_p", 1.0)

        render_pdf_results_section(
            cached_results,
            default_score_type=cached_score_type,
            default_top_k=cached_top_k,
            continuation_method=cached_continuation_method,
            temperature=cached_temperature,
            top_p=cached_top_p,
        )




def render_adversarial_persuasion_page(api_key, model_choice, provider):
    """Render the adversarial persuasive prompting workspace."""

    st.markdown("### 🔓 Persuasive Jailbreak Test")
    strategies = list_persuasion_strategies()
    baseline_prompts = list_baseline_prompts()
    strategy_count = len(strategies)

    def _slugify_filename(value: str) -> str:
        safe = "".join(ch.lower() if ch.isalnum() else "_" for ch in value)
        safe = "_".join(filter(None, safe.split("_")))
        return safe[:80] or "records"

    def _extract_top_few_shot_examples(
        prompt_text: str,
        mutation_store: Dict[str, List[Dict]],
        limit: int = 5,
    ) -> List[str]:
        """Extract top 5 mutated prompts by ROUGE-L score for few-shot examples."""
        records = mutation_store.get(prompt_text, [])
        if not records:
            return []
        
        scored = []
        for record in records:
            data = record.get("data") or {}
            evaluation_data = data.get("evaluation") or {}
            parsed_data = evaluation_data.get("parsed") or {}
            metrics_data = evaluation_data.get("metrics") or {}
            
            mutated_text = parsed_data.get("mutated_text", "").strip()
            rouge_l = metrics_data.get("rouge_l")
            
            if mutated_text and rouge_l is not None:
                scored.append((rouge_l, mutated_text))
        
        # Sort by ROUGE-L descending
        scored.sort(reverse=True, key=lambda x: x[0])
        return [text for _, text in scored[:limit]]

    if strategy_count:
        st.markdown(
            f"Profile LLM copyright leakage using the {strategy_count} persuasion strategies and released evaluation templates."
        )
    else:
        st.markdown(
            "Profile LLM copyright leakage using the persuasion strategies and released evaluation templates."
        )

    template_coverage_bullet = (
        f"- **Template coverage** – toggle among {strategy_count} persuasion strategies sourced from the zero-shot and few-shot JSON frameworks."
        if strategy_count
        else "- **Template coverage** – load persuasion strategies from the zero-shot and few-shot JSON frameworks as soon as the template exports are available."
    )

    st.markdown("#### Key capabilities")
    st.markdown(
        "\n".join(
            [
                template_coverage_bullet,
                "- **Zero-shot vs. few-shot workflows** – generate single exemplars or sweep/ rank batches to curate mutation sets.",
                "- **Automated intention judging** – reuse your selected provider to verify the mutation preserves the original intent.",
                "- **Inference scaling analytics** – repeat generations per strategy and inspect variance/summary statistics.",
                "- **ROUGE, Jaccard, Levenshtein scoring** – quantify similarity against copyrighted reference excerpts on every run.",
            ]
        )
    )

    if not strategies:
        st.error("No persuasion strategies were found. Ensure the persuasion templates are available.")
        return

    if "last_mutated_text" not in st.session_state:
        st.session_state["last_mutated_text"] = ""
    if "last_core_intention" not in st.session_state:
        st.session_state["last_core_intention"] = ""

    adversarial_prompt = st.session_state.get("adversarial_prompt", "")
    reference_text = st.session_state.get("reference_text", "")
    reference_excerpt = reference_text.strip() if reference_text else ""

    st.divider()
    st.markdown("### 🧪 Evaluation Experiments")
    st.markdown(
        """
        <div class=\"analysis-callout\">
            <div class=\"analysis-callout__title\">Two-step persuasion workflow</div>
            <ul class=\"analysis-callout__list\">
                <li><strong>Step 1 · Zero-shot mutation</strong> — Generate baseline adversarial prompt variations and score them against your reference excerpt.</li>
                <li><strong>Step 2 · Few-shot refinement</strong> — Reuse the highest-scoring Step&nbsp;1 exemplars as in-context prompts to craft stronger mutations.</li>
                <li><strong>Review intention judging</strong> — Inspect stored results, confirm intent preservation, and iterate on the strongest findings.</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Get mutation store for accessing results
    mutation_store = st.session_state.setdefault("generated_persuasion_mutations", {})
    stage1_reference_map = st.session_state.setdefault("stage1_reference_texts", {})
    
    # ========== STEP 1: Zero-Shot Mutation & Evaluation ==========
    header_col, spacer_col, button_col = st.columns([4, 1, 1])
    with header_col:
        st.markdown('<p class="analysis-step-label">Step 1 · Configure zero-shot mutation</p>', unsafe_allow_html=True)
        st.markdown("#### 🧪 Zero-Shot Mutation & Evaluation")
    with spacer_col:
        st.write("")
    with button_col:
        if st.button("🗑️ Clear Cache", key="clear_stage1_cache_top", help="Remove cached Step 1/2 results and reference excerpts"):
            st.session_state.pop("generated_persuasion_mutations", None)
            st.session_state.pop("stage1_reference_texts", None)
            st.session_state.pop("stage1_results_prompt_selector", None)
            st.session_state.pop("last_stage1_prompt", None)
            st.session_state.pop("stage2_results", None)
            rerun_fn = getattr(st, "rerun", None)
            if callable(rerun_fn):
                rerun_fn()
            else:
                experimental_rerun = getattr(st, "experimental_rerun", None)
                if callable(experimental_rerun):
                    experimental_rerun()

    st.markdown(
        '<p class="analysis-step-caption">Configure your baseline adversarial prompt, select zero-shot persuasion strategies, and choose the reference excerpt for scoring.</p>',
        unsafe_allow_html=True,
    )

    st.markdown("#### 📝 Baseline prompt setup")
    st.caption("Add or load the adversarial prompt to mutate in Step 1.")
    stage1_input_prompt = st.text_area(
        "Original adversarial prompt",
        height=150,
        placeholder="e.g., Reproduce verbatim the first 100 words of Harry Potter and the Sorcerer's Stone.",
        key="stage1_input_prompt",
        help="Enter the baseline prompt to mutate. This will be transformed using persuasion strategies.",
    )

    if baseline_prompts:
        no_preset_option = "Keep current text"

        def _apply_stage1_preset() -> None:
            selected = st.session_state.get("stage1_baseline_selector")
            if selected and selected != no_preset_option:
                st.session_state["stage1_input_prompt"] = selected
            else:
                st.session_state["stage1_input_prompt"] = ""

        st.selectbox(
            "Load a sample adversarial prompt",
            [no_preset_option, *baseline_prompts],
            key="stage1_baseline_selector",
            on_change=_apply_stage1_preset,
            help="These presets mirror the baseline requests from the paper's extraction study.",
        )

    st.markdown("#### ⚙️ Sampling & evaluation parameters")
    st.caption("Choose persuasion strategies, attempt counts, and the reference excerpt for scoring.")
    zero_shot_strategies = st.multiselect(
        "Persuasion strategies",
        strategies,
        default=strategies[:3] if len(strategies) >= 3 else strategies,
        key="stage1_strategies",
        help="Select one or more persuasion strategies to apply in zero-shot mode.",
    )

    col_attempts, _ = st.columns([1, 2])
    with col_attempts:
        zero_shot_attempts = st.number_input(
            "Attempts per strategy",
            min_value=1,
            max_value=20,
            value=5,
            step=1,
            key="stage1_attempts",
            help="Number of mutation attempts for each strategy (more attempts = broader exploration).",
        )

    st.markdown("**Reference text for scoring**")
    zero_shot_reference = st.text_area(
        "Reference text (for ROUGE-L scoring)",
        value=DEFAULT_HP_REFERENCE_EXCERPT,
        height=150,
        key="stage1_reference",
        help="Ground-truth copyrighted text. ROUGE-L measures how well mutations induce the LLM to reproduce this content.",
    )

    with render_streamlit_accordion(
        "📋 Step 1 checklist",
        key="pj_step1_checklist",
        expanded=False,
    ):
        st.markdown(
            """
            1. <strong>Generate</strong> – Apply persuasion strategies to create mutated prompts (zero-shot, no examples).
            2. <strong>Evaluate</strong> – Send each mutation to the LLM and collect its response.
            3. <strong>Rank</strong> – Score responses against the reference excerpt (ROUGE-L, Jaccard, Levenshtein).
            4. <strong>Judge</strong> – Assess whether each mutation preserves the original intention.
            """,
            unsafe_allow_html=True,
        )

    run_stage1 = st.button(
        "🚀 Run Step 1 · Generate & Evaluate",
        key="run_stage1",
        type="primary",
        width='stretch'
    )
    
    if run_stage1:
        # Validation
        if not stage1_input_prompt.strip():
            st.warning("⚠️ Please enter an adversarial prompt.")
        elif not zero_shot_strategies:
            st.warning("⚠️ Select at least one persuasion strategy.")
        elif not zero_shot_reference.strip():
            st.warning("⚠️ Please provide reference text for evaluation.")
        elif not api_key or not model_choice:
            st.error("⚠️ Enter your API key and choose a model in the sidebar.")
        else:
            original_prompt = stage1_input_prompt.strip()
            prompt_reference_text = zero_shot_reference.strip()

            if prompt_reference_text:
                stage1_reference_map[original_prompt] = prompt_reference_text
            
            # Display processing header
            st.markdown("---")
            st.markdown(f"**Processing:** {textwrap.shorten(original_prompt, width=120, placeholder='…')}")
            st.caption(f"📊 {len(zero_shot_strategies)} strategy(ies) × {zero_shot_attempts} attempt(s) = {len(zero_shot_strategies) * zero_shot_attempts} mutations")
            
            successful_count = 0
            
            # ===== STEP 1: Generate Mutations =====
            st.markdown("**🔄 Step 1/4: Generating zero-shot mutations**")
            st.caption(f"Generating {len(zero_shot_strategies)} strategy(ies) × {zero_shot_attempts} attempt(s) = {len(zero_shot_strategies) * zero_shot_attempts} total mutations")
            
            generation_progress = st.progress(0.0)
            total_to_generate = len(zero_shot_strategies) * zero_shot_attempts
            
            evaluations = mutate_strategies(
                api_key,
                model_choice,
                provider,
                zero_shot_strategies,
                original_prompt,
                reference_text=None,  # Don't calculate ROUGE during generation
                few_shot_examples=None,  # Zero-shot: no examples
                attempts_per_strategy=zero_shot_attempts,
                temperature=1.0,  # Higher temperature for diverse mutation generation
                top_p=1.0,
                dry_run=False,
            )
            
            generation_progress.progress(1.0)
            generation_progress.empty()
            
            if not evaluations:
                st.error("❌ No mutations produced. Check your API key and model settings.")
            else:
                st.success(f"✅ Generated {len(evaluations)} mutations")
                
                # ===== STEP 2: Evaluate Mutations =====
                st.markdown("**🔄 Step 2/4: Evaluating mutations against reference text**")
                st.caption("Sending each mutation to the LLM and calculating ROUGE-L with reference output...")
                
                evaluated_mutations = []
                progress_bar = st.progress(0.0)
                
                for eval_idx, evaluation in enumerate(evaluations):
                    if evaluation is None or evaluation.mutation.error:
                        continue
                    
                    parsed = evaluation.parsed
                    if not parsed or not parsed.mutated_text:
                        continue
                    
                    mutated_text = parsed.mutated_text.strip()
                    
                    # Send mutated prompt to LLM to get its response
                    try:
                        llm_response = get_llm_completion(
                            mutated_text,
                            api_key,
                            model_choice,
                            provider=provider,
                            temperature=0.0,  # Deterministic for evaluation
                            top_p=0.8,
                        )
                        
                        # Calculate similarity metrics
                        rouge_score = calculate_rouge_score(llm_response, zero_shot_reference.strip())
                        jaccard = calculate_jaccard_index(llm_response, zero_shot_reference.strip())
                        levenshtein = distance(llm_response, zero_shot_reference.strip())
                        
                        eval_metrics = SimilarityMetrics(
                            rouge_l=rouge_score,
                            jaccard=jaccard,
                            levenshtein=levenshtein,
                        )
                        
                        updated_evaluation = MutationEvaluation(
                            mutation=evaluation.mutation,
                            parsed=evaluation.parsed,
                            metrics=eval_metrics,
                            attempt=evaluation.attempt,
                        )
                        
                        evaluated_mutations.append({
                            "evaluation": updated_evaluation,
                            "llm_response": llm_response,
                        })
                        
                    except Exception as e:
                        st.warning(f"⚠️ Failed to evaluate mutation {eval_idx + 1}: {str(e)}")
                        continue
                    
                    progress_bar.progress((eval_idx + 1) / len(evaluations))
                
                progress_bar.empty()
                
                if not evaluated_mutations:
                    st.error("❌ No mutations were successfully evaluated.")
                else:
                    # ===== STEP 3: Rank & Store Results =====
                    st.markdown("**🔄 Step 3/4: Ranking mutations by ROUGE-L score**")
                    
                    # Sort by ROUGE-L score (descending)
                    evaluated_mutations.sort(
                        key=lambda x: x["evaluation"].metrics.rouge_l if x["evaluation"].metrics else 0,
                        reverse=True
                    )
                    
                    # Store results in mutation_store
                    for eval_item in evaluated_mutations:
                        evaluation = eval_item["evaluation"]
                        llm_response = eval_item["llm_response"]
                        
                        record_entries = mutation_store.setdefault(original_prompt, [])
                        
                        mutation_entry = MutationWithJudge(
                            evaluation=evaluation,
                            judge=None,
                            judge_passed=None,
                        )
                        serialised_entry = serialise_mutation_with_judge(mutation_entry)
                        
                        # Check for duplicates
                        mutated_text = evaluation.parsed.mutated_text.strip()
                        entry_exists = False
                        for stored in record_entries:
                            stored_config = stored.get("config") or []
                            if stored_config and stored_config[0] != "zero":
                                continue
                            
                            stored_data = stored.get("data") or {}
                            stored_eval = stored_data.get("evaluation") or {}
                            stored_parsed = stored_eval.get("parsed") or {}
                            stored_mutated_text = stored_parsed.get("mutated_text", "").strip()
                            
                            if stored_mutated_text == mutated_text:
                                entry_exists = True
                                break
                        
                        if not entry_exists:
                            record_entries.append({
                                "config": ["zero", False],
                                "data": serialised_entry,
                                "llm_response": llm_response,
                            })
                            successful_count += 1
                    
                    # ===== STEP 4: Intention Preservation Judging =====
                    st.markdown("---")
                    st.markdown("**🔄 Step 4/4: Intention Preservation Judging**")
                    st.caption("Assessing whether mutated prompts preserve the original harmful intention...")
                    
                    judging_progress = st.progress(0.0)
                    
                    for judge_idx, eval_item in enumerate(evaluated_mutations):
                        evaluation = eval_item["evaluation"]
                        mutated_text = evaluation.parsed.mutated_text.strip()
                        strategy = evaluation.mutation.strategy
                        
                        with st.spinner(f"Judging mutation {judge_idx + 1}/{len(evaluated_mutations)} ({strategy})..."):
                            try:
                                assessment = assess_intention_preservation(
                                    api_key,
                                    model_choice,
                                    provider,
                                    original_prompt,
                                    mutated_text,
                                    temperature=0.0,  # Deterministic for judging
                                    top_p=0.0,
                                    dry_run=False,
                                )
                                
                                # Update mutation store with judging results
                                record_entries = mutation_store.get(original_prompt, [])
                                for stored in record_entries:
                                    stored_config = stored.get("config") or []
                                    if stored_config and stored_config[0] != "zero":
                                        continue
                                    
                                    stored_data = stored.get("data") or {}
                                    stored_eval = stored_data.get("evaluation") or {}
                                    stored_parsed = stored_eval.get("parsed") or {}
                                    stored_mutated_text = stored_parsed.get("mutated_text", "").strip()
                                    
                                    if stored_mutated_text == mutated_text:
                                        # Update with judging results
                                        judged_entry = MutationWithJudge(
                                            evaluation=evaluation,
                                            judge=assessment.secondary,
                                            judge_passed=assessment.judge_passed,
                                        )
                                        stored["data"] = serialise_mutation_with_judge(judged_entry)
                                        stored["config"] = ["zero", True]  # Mark as judged
                                        stored["judge_meta"] = {
                                            "core_intention": assessment.core_intention,
                                            "restated_mutated_text": assessment.restated_mutated_text,
                                            "primary_error": assessment.primary.error,
                                            "secondary_error": assessment.secondary.error,
                                        }
                                        break
                                
                                # Store assessment for display
                                eval_item["assessment"] = assessment
                                
                            except Exception as e:
                                st.warning(f"⚠️ Failed to judge mutation {judge_idx + 1}: {str(e)}")
                                eval_item["assessment"] = None
                        
                        judging_progress.progress((judge_idx + 1) / len(evaluated_mutations))
                    
                    judging_progress.empty()
                    st.success(f"✅ Completed intention judging for {len(evaluated_mutations)} mutations")
                    
                    st.session_state["last_stage1_prompt"] = original_prompt
                    st.session_state["stage1_results_prompt_selector"] = original_prompt
                    st.markdown("---")
                    st.success(f"✅ **Step 1 Complete:** Evaluated {successful_count} mutations (ranked by ROUGE-L)")
    
    st.divider()

    # ===== Persistent Step 1 Results =====
    stage1_prompts = [
        prompt_text
        for prompt_text, records in mutation_store.items()
        if any((entry.get("config") or [None])[0] == "zero" for entry in records)
    ]

    if stage1_prompts:
        st.markdown('<p class="analysis-step-label">Step 1 · Results explorer</p>', unsafe_allow_html=True)
        st.markdown("#### 📚 Step 1 Results Library")
        st.caption("Step 1 results are cached in session state so you can revisit them while configuring Step 2.")

        default_prompt = st.session_state.get("stage1_results_prompt_selector")
        if default_prompt not in stage1_prompts:
            default_prompt = stage1_prompts[0]

        selected_prompt = st.selectbox(
            "Select a Step 1 prompt to inspect",
            options=stage1_prompts,
            format_func=lambda x: textwrap.shorten(x, width=100, placeholder="…"),
            index=stage1_prompts.index(default_prompt) if default_prompt in stage1_prompts else 0,
            key="stage1_results_prompt_selector",
        )

        stage1_records = [
            entry for entry in mutation_store.get(selected_prompt, [])
            if (entry.get("config") or [None])[0] == "zero"
        ]

        if stage1_records:
            ranked_rows: List[Dict[str, Any]] = []
            stored_panels: List[Dict[str, Any]] = []

            for entry in stage1_records:
                serialised = entry.get("data") or {}
                llm_response = entry.get("llm_response", "")
                judge_meta = entry.get("judge_meta") or {}
                config = entry.get("config") or []
                judged_flag = bool(config[1]) if len(config) > 1 else False

                deserialised = deserialise_mutation_with_judge(serialised)
                evaluation = deserialised.evaluation
                parsed = evaluation.parsed
                metrics = evaluation.metrics

                mutated_text = parsed.mutated_text.strip() if parsed and parsed.mutated_text else ""
                rouge_l = metrics.rouge_l if metrics else 0.0
                jaccard = metrics.jaccard if metrics else 0.0
                levenshtein = metrics.levenshtein if metrics else None

                judge_passed = deserialised.judge_passed if judged_flag else None
                if not judged_flag:
                    status_icon = "⏳"
                    status_text = "Pending — Not yet judged"
                elif judge_passed is True:
                    status_icon = "✅"
                    status_text = "PASSED — Preserves original intention"
                elif judge_passed is False:
                    status_icon = "❌"
                    status_text = "FAILED — Does not preserve intention"
                else:
                    status_icon = "⚠️"
                    status_text = "UNCLEAR — Unable to determine"

                ranked_rows.append({
                    "score": rouge_l,
                    "strategy": evaluation.mutation.strategy,
                    "attempt": evaluation.attempt,
                    "mutated_text": mutated_text,
                    "mutated_display": textwrap.shorten(mutated_text, width=120, placeholder="…") if mutated_text else "",
                    "llm_response": llm_response or "",
                    "llm_display": textwrap.shorten(llm_response, width=120, placeholder="…") if llm_response else "",
                    "rouge_l": f"{rouge_l:.4f}" if metrics else "N/A",
                    "jaccard": f"{jaccard:.4f}" if metrics else "N/A",
                    "levenshtein": str(levenshtein) if levenshtein is not None else "N/A",
                    "judge_status": f"{status_icon} {status_text}",
                })

                stored_panels.append({
                    "score": rouge_l,
                    "evaluation": evaluation,
                    "metrics": metrics,
                    "judge_passed": judge_passed,
                    "judge": deserialised.judge,
                    "judge_meta": judge_meta,
                    "llm_response": llm_response,
                    "status_icon": status_icon,
                    "status_text": status_text,
                    "judged": judged_flag,
                })

            ranked_rows.sort(key=lambda item: item["score"], reverse=True)
            stored_panels.sort(key=lambda item: item["score"], reverse=True)

            if ranked_rows:
                df_data = []
                for idx, row in enumerate(ranked_rows, start=1):
                    df_data.append({
                        "rank": idx,
                        "strategy": row["strategy"],
                        "attempt": row["attempt"],
                        "mutated_text": row["mutated_display"],
                        "llm_response": row["llm_display"],
                        "rouge_l": row["rouge_l"],
                        "jaccard": row["jaccard"],
                        "levenshtein": row["levenshtein"],
                        "judge_status": row["judge_status"],
                    })

                df_stage1 = pd.DataFrame(df_data)
                st.dataframe(
                    df_stage1,
                    width='stretch',
                    hide_index=True,
                    column_config={
                        "rank": st.column_config.NumberColumn("Rank", width="small"),
                        "strategy": st.column_config.TextColumn("Strategy", width="medium"),
                        "attempt": st.column_config.NumberColumn("Attempt", width="small"),
                        "mutated_text": st.column_config.TextColumn("Mutated Prompt", width="large"),
                        "llm_response": st.column_config.TextColumn("LLM Response", width="large"),
                        "rouge_l": st.column_config.TextColumn("ROUGE-L", width="small"),
                        "jaccard": st.column_config.TextColumn("Jaccard", width="small"),
                        "levenshtein": st.column_config.TextColumn("Levenshtein", width="small"),
                        "judge_status": st.column_config.TextColumn("Judge Status", width="medium"),
                    },
                )

                st.markdown("##### 🎯 Intention Preservation Judging Results")
                st.caption("Click to expand each mutation result and view detailed intention preservation analysis.")

                for idx, panel_payload in enumerate(stored_panels, start=1):
                    evaluation = panel_payload["evaluation"]
                    metrics = panel_payload.get("metrics")
                    parsed = evaluation.parsed
                    mutated_text = parsed.mutated_text.strip() if parsed and parsed.mutated_text else ""
                    rouge_score = metrics.rouge_l if metrics else 0.0
                    jaccard_value = metrics.jaccard if metrics else 0.0
                    levenshtein_value = metrics.levenshtein if metrics else None
                    status_icon = panel_payload["status_icon"]
                    status_text = panel_payload["status_text"]
                    judged_flag = panel_payload["judged"]
                    judge_meta = panel_payload.get("judge_meta") or {}
                    judge_result = panel_payload.get("judge")
                    llm_response = panel_payload.get("llm_response") or ""

                    sections: List[Tuple[str,str,Optional[str]]] = []
                    
                    # Summary section
                    summary_lines = [
                        f"Strategy: {evaluation.mutation.strategy}",
                        f"Attempt: {evaluation.attempt}",
                        f"Judge Status: {status_icon} {status_text}",
                    ]
                    sections.append(("📄 Mutation Summary", "\n".join(summary_lines), None))
                    
                    # Mutated prompt
                    sections.append(("📝 Mutated Prompt", mutated_text, "generated"))
                    
                    # Metrics
                    if metrics:
                        metrics_lines = [
                            f"ROUGE-L: {rouge_score:.4f}",
                            f"Jaccard: {jaccard_value:.4f}",
                            f"Levenshtein: {levenshtein_value}",
                        ]
                        sections.append(("📊 Similarity Metrics", "\n".join(metrics_lines), None))
                    
                    # LLM Response
                    if llm_response:
                        sections.append(("🧪 Evaluation Model Response", llm_response, "generated"))
                    
                    # Intention judging results
                    if judged_flag:
                        primary_error = judge_meta.get("primary_error")
                        if primary_error:
                            primary_content = f"Error: {primary_error}"
                        else:
                            core_intention = judge_meta.get("core_intention")
                            restated_mutated_text = judge_meta.get("restated_mutated_text")
                            bits = []
                            if core_intention:
                                bits.append(f"Core Intention Extracted:\n{core_intention}")
                            if restated_mutated_text:
                                bits.append(f"Restated Mutated Text:\n{restated_mutated_text}")
                            primary_content = "\n\n".join(bits) if bits else "No assessment data available"
                        sections.append(("🧠 Primary Intention Assessment", primary_content, "generated"))
                        
                        secondary_error = judge_meta.get("secondary_error")
                        if secondary_error:
                            secondary_content = f"Error: {secondary_error}"
                        else:
                            secondary_content = f"{status_icon} {status_text}"
                        sections.append(("⚖️ Secondary Validation", secondary_content, None))
                        
                        judge_response = judge_result.response if judge_result else ""
                        if judge_response:
                            sections.append(("🗳️ Judge Raw Response", judge_response, "generated"))
                    else:
                        sections.append(("🧠 Primary Intention Assessment", "⏳ Pending — judge not run yet.", None))
                        sections.append(("⚖️ Secondary Validation", "⏳ Pending — judge not run yet.", None))

                    # Build meta string with metrics
                    meta_parts = [f"{status_icon.strip()} {status_text.strip()}"]
                    if metrics:
                        levenshtein_display = (
                            str(levenshtein_value)
                            if levenshtein_value is not None
                            else "N/A"
                        )
                        meta_parts.append(f"ROUGE-L {rouge_score:.4f}")
                        meta_parts.append(f"Jaccard {jaccard_value:.4f}")
                        meta_parts.append(f"Levenshtein {levenshtein_display}")
                    meta_text = " | ".join(meta_parts)

                    render_prompt_style_panel(
                        title=f"Mutation #{idx} — {evaluation.mutation.strategy}",
                        sections=sections,
                        meta=meta_text,
                        expanded=False,
                    )

    st.divider()

    # ========== STEP 2: Few-Shot Generation ==========
    st.markdown('<p class="analysis-step-label">Step 2 · Refine with few-shot examples</p>', unsafe_allow_html=True)
    st.markdown("#### 🎯 Few-Shot Generation")
    st.markdown(
        '<p class="analysis-step-caption">Reuse the strongest Step 1 mutations as exemplars to guide new adversarial variants. Complete Step 1 before proceeding.</p>',
        unsafe_allow_html=True,
    )

    with render_streamlit_accordion(
        "📋 Step 2 checklist",
        key="pj_step2_checklist",
        expanded=False,
    ):
        st.markdown(
            """
            1. <strong>Select prompts</strong> – Choose which Step 1 prompts you want to refine.
            2. <strong>Pick strategies</strong> – Decide which persuasion strategies to reuse in few-shot mode.
            3. <strong>Review exemplars</strong> – Inspect the top-ranked Step 1 outputs that will seed the few-shot prompt.
            4. <strong>Generate</strong> – Run Step 2 to produce refined mutations with in-context examples.
            """,
            unsafe_allow_html=True,
        )
    
    # Check if Step 1 has been run
    if not mutation_store:
        st.warning("⚠️ No Step 1 results found. Please run Step 1: Zero-shot mutation first.")
    else:
        # Show available prompts from Step 1
        available_prompts = list(mutation_store.keys())
        st.markdown(f"**Available prompts from Step 1:** {len(available_prompts)}")
        
        selected_stage2_prompts = st.multiselect(
            "Select prompts for Stage 2",
            available_prompts,
            default=available_prompts[:3] if len(available_prompts) >= 3 else available_prompts,
            format_func=lambda x: textwrap.shorten(x, width=80, placeholder="…"),
            key="stage2_prompt_selection",
            help="Choose which Stage 1 prompts to use for few-shot generation.",
        )
        
        few_shot_strategies = st.multiselect(
            "Persuasion strategies",
            strategies,
            default=strategies[:2] if len(strategies) >= 2 else strategies,
            key="stage2_strategies",
            help="Select strategies for few-shot mode.",
        )

        few_shot_attempts = st.number_input(
            "Attempts per strategy",
            min_value=1,
            max_value=20,
            value=5,
            step=1,
            key="stage2_attempts",
            help="Number of few-shot mutation attempts per strategy.",
        )
        
        run_stage2 = st.button(
            "🎯 Run Step 2 · Few-Shot Generation",
            key="run_stage2",
            type="primary",
            width='stretch'
        )
        
        if run_stage2:
            if not selected_stage2_prompts:
                st.warning("⚠️ Select at least one prompt for Step 2.")
            elif not few_shot_strategies:
                st.warning("⚠️ Select at least one strategy.")
            elif not api_key or not model_choice:
                st.error("⚠️ Enter your API key and choose a model in the sidebar.")
            else:
                st.markdown(f"**Processing {len(selected_stage2_prompts)} prompt(s) with {len(few_shot_strategies)} strategy(ies)...**")
                
                total_few_shot = 0
                stage2_results = []
                
                for prompt_idx, original_prompt in enumerate(selected_stage2_prompts, 1):
                    st.markdown(f"##### Prompt {prompt_idx}/{len(selected_stage2_prompts)}")
                    st.caption(f"📝 {textwrap.shorten(original_prompt, width=100, placeholder='…')}")
                    
                    # Extract top 5 examples from Stage 1
                    few_shot_examples = _extract_top_few_shot_examples(original_prompt, mutation_store, limit=5)
                    
                    if not few_shot_examples:
                        st.warning(f"⚠️ No Stage 1 results for this prompt. Skipping.")
                        continue
                    
                    st.caption(f"📚 Using {len(few_shot_examples)} top-ranked Stage 1 examples as demonstrations")
                    prompt_reference_text = stage1_reference_map.get(original_prompt)
                    if prompt_reference_text:
                        reference_sections = [
                            (
                                "Reference Excerpt",
                                prompt_reference_text,
                                "generated",
                            )
                        ]
                        reference_meta = f"Length {len(prompt_reference_text)} chars"
                        render_prompt_style_panel(
                            title="📄 Stage 1 Reference Text",
                            sections=reference_sections,
                            meta=reference_meta,
                            expanded=False,
                        )
                    else:
                        st.caption("⚠️ No Stage 1 reference text found—falling back to the current reference input field.")

                    example_sections = []
                    for ex_idx, example in enumerate(few_shot_examples, 1):
                        example_sections.append(
                            (
                                f"Example #{ex_idx}",
                                example,
                                "generated",
                            )
                        )

                    render_prompt_style_panel(
                        title="🧾 Stage 1 Top Examples",
                        sections=example_sections,
                        meta=f"{len(few_shot_examples)} exemplars",
                        expanded=False,
                    )
                    
                    st.markdown(f"**Generating {len(few_shot_strategies)} strategy(ies) × {few_shot_attempts} attempt(s)...**")
                    stage2_progress = st.progress(0.0)

                    if prompt_reference_text:
                        st.caption("📏 Stage 2 ROUGE reference inherited from Stage 1 run")
                    else:
                        prompt_reference_text = zero_shot_reference.strip() or None
                        if prompt_reference_text:
                            st.caption("📏 Stage 2 fallback reference: current Stage 1 reference field")
                    
                    evaluations = mutate_strategies(
                        api_key,
                        model_choice,
                        provider,
                        few_shot_strategies,
                        original_prompt,
                        reference_text=prompt_reference_text,
                        few_shot_examples=few_shot_examples,  # Pass top 5 examples
                        attempts_per_strategy=few_shot_attempts,
                        temperature=1.0,  # Higher temperature for diverse mutation generation
                        top_p=1.0,
                        dry_run=False,
                    )
                    
                    stage2_progress.progress(1.0)
                    stage2_progress.empty()
                    
                    if not evaluations:
                        st.error(f"❌ No few-shot mutations for prompt {prompt_idx}.")
                        continue
                    
                    # Store few-shot results
                    successful_count = 0
                    for evaluation in evaluations:
                        if evaluation is None or evaluation.mutation.error:
                            continue
                        
                        parsed = evaluation.parsed
                        if not parsed or not parsed.mutated_text:
                            continue
                        
                        mutated_text = parsed.mutated_text.strip()
                        reference_for_metrics = (prompt_reference_text or "").strip()

                        llm_response: Optional[str] = None
                        eval_metrics: Optional[SimilarityMetrics] = None

                        if reference_for_metrics:
                            try:
                                llm_response = get_llm_completion(
                                    mutated_text,
                                    api_key,
                                    model_choice,
                                    provider=provider,
                                    temperature=0.0,
                                    top_p=0.8,
                                )

                                rouge_score = calculate_rouge_score(llm_response, reference_for_metrics)
                                jaccard = calculate_jaccard_index(llm_response, reference_for_metrics)
                                levenshtein = distance(llm_response, reference_for_metrics)

                                eval_metrics = SimilarityMetrics(
                                    rouge_l=rouge_score,
                                    jaccard=jaccard,
                                    levenshtein=levenshtein,
                                )
                            except Exception as exc:
                                st.warning(
                                    f"⚠️ Failed to score few-shot mutation (strategy: {evaluation.mutation.strategy}): {exc}"
                                )
                                eval_metrics = evaluation.metrics
                        else:
                            eval_metrics = evaluation.metrics

                        evaluation_to_store = MutationEvaluation(
                            mutation=evaluation.mutation,
                            parsed=evaluation.parsed,
                            metrics=eval_metrics,
                            attempt=evaluation.attempt,
                        )

                        record_entries = mutation_store.setdefault(original_prompt, [])
                        
                        few_mutation_entry = MutationWithJudge(
                            evaluation=evaluation_to_store,
                            judge=None,
                            judge_passed=None,
                        )
                        serialised_few = serialise_mutation_with_judge(few_mutation_entry)
                        
                        record_entries.append({
                            "config": ["few", False],
                            "data": serialised_few,
                            "llm_response": llm_response,
                        })
                        
                        successful_count += 1
                        stage2_results.append({
                            "prompt": original_prompt,
                            "strategy": evaluation.mutation.strategy,
                            "mutated_text": mutated_text,
                            "llm_response": llm_response or "",
                            "rouge_l": eval_metrics.rouge_l if eval_metrics else None,
                            "jaccard": eval_metrics.jaccard if eval_metrics else None,
                            "levenshtein": eval_metrics.levenshtein if eval_metrics else None,
                        })
                    
                    total_few_shot += successful_count
                    st.success(f"✅ Prompt {prompt_idx}: Generated {successful_count} few-shot mutations")
                
                st.success(f"🎉 **Stage 2 Complete!** Total few-shot mutations: {total_few_shot}")
                
                # Store results in session state for persistence
                st.session_state["stage2_results"] = stage2_results

    # Display summary table if results exist
    stage2_results = st.session_state.get("stage2_results", [])
    if stage2_results:
        df_stage2 = pd.DataFrame(stage2_results)
        st.markdown("**Stage 2 Summary**")
        st.dataframe(df_stage2, width='stretch')

def render_unlearning_detection_page(api_key, model_choice, provider):
    """Render the representational analysis experience."""
    st.markdown("### 🧬 Representational Analysis")
    st.markdown(
        "Run Fisher Information, PCA shift/sim, and layer-wise CKA probes to quantify how unlearning reshapes the reference versus adapted model across every layer."
    )

    st.warning(
        "⚠️ **Important Notes:**\n\n"
        "- **API Incompatibility**: This feature is not applicable for API-based-only models as it requires access to the model's internal state\n"
        "- **Memory Requirements**: Large models (>7B parameters) may cause crashes due to insufficient RAM/VRAM\n"
        "- **Network**: First-time use requires internet to download models from HuggingFace\n"
        "- **Recommendation**: Start with small models (≤1B parameters) like `Qwen/Qwen2-0.5B` or `gpt2`\n"
        "- **Offline Mode**: Pre-download models using `huggingface-cli` for offline use"
    )

    dependencies_available = is_representational_analysis_available()
    if not dependencies_available:
        st.warning(
            "Representational analysis requires optional dependencies (PyTorch, Transformers, scikit-learn, matplotlib). Install the GPU toolkit extras before using this feature."
        )

    features = list_representational_features()
    if not features:
        st.info("No representational analysis features are currently available.")
        return

    feature_lookup = {feature.id: feature for feature in features}

    with st.form("representational_analysis_form"):
        selected_feature_id = st.selectbox(
            "Select representational probe",
            options=[feature.id for feature in features],
            index=0,
            format_func=lambda feature_id: f"{feature_lookup[feature_id].name} — {feature_lookup[feature_id].description}",
            key="representational_feature_selection",
            help="Maps directly to the `feature` argument of `run_feature_analysis`.",
        )

        selected_feature = feature_lookup[selected_feature_id]

        st.markdown("##### Model checkpoints")
        st.info("💡 **Model Path Format**: Use Hugging Face model IDs (e.g., 'gpt2', 'microsoft/DialoGPT-medium') or absolute paths to local directories containing `config.json` and model files. Do not use Hugging Face cache paths directly.")
        col_ref, col_upd = st.columns(2)
        with col_ref:
            reference_model_path = st.text_input(
                "Reference model (baseline)",
                placeholder="e.g. gpt2, Qwen/Qwen2.5-7B, or /path/to/local/model",
                help="Hugging Face model ID (e.g., 'gpt2') or absolute path to local model directory containing config.json",
                key="representational_reference_model",
            )
        with col_upd:
            updated_model_path = st.text_input(
                "Updated / deployed model",
                placeholder="Path or HF repo ID for the model under audit",
                help="Hugging Face model ID (e.g., 'microsoft/DialoGPT-medium') or absolute path to local model directory",
                key="representational_updated_model",
            )

        st.markdown("##### Evaluation prompts")
        query_text = st.text_area(
            "Evaluation prompts",
            height=180,
            placeholder="Enter one query per line that probes the model's behaviour post-unlearning.\n\nExample:\nThe quick brown fox jumps over the lazy dog.\nUnlearning LLMs is an active area of research.\nWhat is the capital of France?",
            help="Each non-empty line is passed as an element of the `query` list. Enter multiple queries (one per line) to test different prompts.",
            key="representational_query_text",
        )
        query_preview = [line.strip() for line in query_text.splitlines() if line.strip()]

        if query_preview:
            st.caption(f"📝 **{len(query_preview)} query(ies) will be processed:**")
            for i, query in enumerate(query_preview, 1):
                st.caption(f"{i}. {query}")
        else:
            st.caption("📝 No queries entered yet. Add at least one query above.")

        st.markdown("##### Runtime parameters")
        st.caption("Device is set to `cuda` (GPU enabled).")
        device = "cuda"

        col_batch, col_batches, col_length = st.columns([1, 1, 1])
        with col_batch:
            batch_size = st.number_input(
                "Batch size",
                min_value=1,
                max_value=128,
                value=4,
                step=1,
                help="Mini-batch size for analyses that stream batches (FIM, CKA).",
                key="representational_batch_size",
            )
        with col_batches:
            num_batches = st.number_input(
                "Batches",
                min_value=1,
                max_value=200,
                value=10,
                step=1,
                help="Number of dataloader batches to use when estimating statistics (FIM, CKA).",
                key="representational_num_batches",
            )
        with col_length:
            max_length = st.number_input(
                "Max length",
                min_value=16,
                max_value=4096,
                value=128,
                step=16,
                help="Maximum sequence length for tokenization.",
                key="representational_max_length",
            )

        st.caption("Preview of the backend call that will be executed with your settings:")
        query_list_preview = ", ".join(f'"{q}"' for q in query_preview) or '"<enter at least one query>"'
        call_preview = textwrap.dedent(
            f"""
            run_feature_analysis(
                feature="{selected_feature.id}",
                model_reference_path="{reference_model_path.strip() or '<reference_model>'}",
                model_path="{updated_model_path.strip() or '<updated_model>'}",
                query=[{query_list_preview}],
                device="{device}",
                batch_size={int(batch_size)},
                num_batches={int(num_batches)},
                max_length={int(max_length)},
            )
            """.strip()
        )
        st.code(call_preview, language="python")

        submit_run = st.form_submit_button(
            "🧬 Run Representational Analysis",
            width='stretch',
            help="Submit the parameters above and execute the representational probe on the backend.",
        )

    rep_result = None
    analysis_request = None
    if submit_run:
            queries = query_preview
            if not reference_model_path.strip():
                st.warning("⚠️ Provide the reference model path before running representational analysis.")
            elif not updated_model_path.strip():
                st.warning("⚠️ Provide the updated model path before running representational analysis.")
            elif not queries:
                st.warning("⚠️ Enter at least one non-empty query prompt.")
            else:
                # Validate model paths
                import os
                ref_path = reference_model_path.strip()
                upd_path = updated_model_path.strip()
                
                ref_valid = False
                upd_valid = False
                
                # Check if it's a Hugging Face model ID (contains slash or is simple name)
                if '/' in ref_path or ref_path in ['gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl']:
                    ref_valid = True
                # Check if it's a local directory with config.json
                elif os.path.isdir(ref_path) and os.path.exists(os.path.join(ref_path, 'config.json')):
                    ref_valid = True
                else:
                    st.error(f"❌ Reference model path '{ref_path}' is not valid. Use a Hugging Face model ID (e.g., 'gpt2') or a local directory containing config.json")
                
                if '/' in upd_path or upd_path in ['gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl']:
                    upd_valid = True
                elif os.path.isdir(upd_path) and os.path.exists(os.path.join(upd_path, 'config.json')):
                    upd_valid = True
                else:
                    st.error(f"❌ Updated model path '{upd_path}' is not valid. Use a Hugging Face model ID (e.g., 'gpt2') or a local directory containing config.json")
                
                if ref_valid and upd_valid:
                    analysis_request = {
                        "feature": selected_feature.id,
                        "model_reference_path": ref_path,
                        "model_path": upd_path,
                        "query": queries,
                        "device": device,
                        "batch_size": int(batch_size),
                        "num_batches": int(num_batches),
                        "max_length": int(max_length),
                    }
                    with st.spinner("🔎 Computing representational differences... this may take several minutes for large models."):
                        try:
                            rep_result = run_representational_analysis(**analysis_request)
                            st.session_state["representational_last_run_request"] = analysis_request
                        except ValueError as exc:
                            st.error(f"❌ {exc}")
                            rep_result = None
                        except RuntimeError as exc:
                            # The RuntimeError raised by run_representational_analysis includes
                            # a detailed diagnostic containing captured stdout/stderr and the traceback.
                            err_text = str(exc)
                            st.error("❌ Representational analysis failed. Expand for full diagnostics below.")
                            # Parse the diagnostic into sections for the custom component
                            sections = []
                            parts = err_text.split("--- Captured stdout ---")
                            if len(parts) == 2:
                                before_stdout = parts[0]
                                after_stdout = parts[1]
                                parts2 = after_stdout.split("--- Captured stderr ---")
                                if len(parts2) == 2:
                                    stdout_content = parts2[0]
                                    after_stderr = parts2[1]
                                    parts3 = after_stderr.split("--- Traceback ---")
                                    if len(parts3) == 2:
                                        stderr_content = parts3[0]
                                        tb_content = parts3[1]
                                        exception_part = before_stdout.strip()
                                        sections.append(("Exception", exception_part, None))
                                        if stdout_content.strip():
                                            sections.append(("Captured stdout", stdout_content.strip(), None))
                                        if stderr_content.strip():
                                            sections.append(("Captured stderr", stderr_content.strip(), None))
                                        if tb_content.strip():
                                            sections.append(("Traceback", tb_content.strip(), None))
                                    else:
                                        sections.append(("Full Diagnostics", err_text, None))
                                else:
                                    sections.append(("Full Diagnostics", err_text, None))
                            else:
                                sections.append(("Full Diagnostics", err_text, None))
                            # Add slider for panel height
                            panel_max_height = st.slider(
                                "Panel Display Height (pixels)",
                                min_value=200,
                                max_value=1200,
                                value=600,
                                step=50,
                                help="Adjust the maximum height of the error details panel.",
                                key="error_panel_height_slider",
                            )
                            render_collapsible_panel(
                                title="Representational Analysis Logs and Traceback",
                                sections=sections,
                                expanded=False,
                                max_height=panel_max_height,
                            )
                            rep_result = None

            if rep_result:
                st.markdown("---")
                st.success(
                    f"Completed {rep_result.feature_name} analysis. Review the generated artifacts below."
                )

                if analysis_request:
                    st.markdown("##### Parameters sent to the backend")
                    st.json(analysis_request)

                if rep_result.warnings:
                    for warning in rep_result.warnings:
                        st.warning(warning)

                if rep_result.inline_artifacts:
                    st.markdown("##### Visualisations")

                    # Group visualizations in columns for better layout
                    num_artifacts = len(rep_result.inline_artifacts)
                    if num_artifacts <= 3:
                        # For few artifacts, show in a single row
                        cols = st.columns(num_artifacts)
                        for idx, artifact in enumerate(rep_result.inline_artifacts):
                            with cols[idx]:
                                caption = artifact.title or f"Visualisation {idx + 1}"
                                if artifact.mime_type.startswith("image/"):
                                    st.image(artifact.data, caption=caption, width='content')
                                else:
                                    st.download_button(
                                        label=f"⬇️ Download {caption}",
                                        data=artifact.data,
                                        file_name=f"representational_artifact_{idx + 1}",
                                        mime=artifact.mime_type,
                                        key=f"representational_inline_{idx + 1}",
                                    )
                                if artifact.description:
                                    st.caption(artifact.description)
                    else:
                        # For many artifacts, show in a grid
                        cols_per_row = 3
                        for i in range(0, num_artifacts, cols_per_row):
                            row_artifacts = rep_result.inline_artifacts[i:i + cols_per_row]
                            cols = st.columns(len(row_artifacts))
                            for j, artifact in enumerate(row_artifacts):
                                with cols[j]:
                                    caption = artifact.title or f"Visualisation {i + j + 1}"
                                    if artifact.mime_type.startswith("image/"):
                                        st.image(artifact.data, caption=caption, width='content')
                                    else:
                                        st.download_button(
                                            label=f"⬇️ Download {caption}",
                                            data=artifact.data,
                                            file_name=f"representational_artifact_{i + j + 1}",
                                            mime=artifact.mime_type,
                                            key=f"representational_inline_{i + j + 1}",
                                        )
                                    if artifact.description:
                                        st.caption(artifact.description)

                if rep_result.generated_artifacts:
                    st.markdown("##### Generated artifacts")
                    for artifact_path in rep_result.generated_artifacts:
                        st.markdown(f"- `{artifact_path}`")

                    from pathlib import Path as _ResultPath  # local import to avoid polluting module namespace
                    import zipfile

                    output_dir_path = _ResultPath(rep_result.output_path)
                    if output_dir_path.is_dir() and len(rep_result.generated_artifacts) > 1:
                        zip_target = output_dir_path / f"{rep_result.feature_id}_artifacts.zip"
                        try:
                            with zipfile.ZipFile(zip_target, "w") as zipf:
                                for artifact in rep_result.generated_artifacts:
                                    artifact_path_obj = _ResultPath(artifact)
                                    if artifact_path_obj.exists():
                                        zipf.write(artifact_path_obj, arcname=artifact_path_obj.name)
                            with open(zip_target, "rb") as zip_bytes:
                                st.download_button(
                                    label="⬇️ Download all artifacts (ZIP)",
                                    data=zip_bytes.read(),
                                    file_name=zip_target.name,
                                    mime="application/zip",
                                    key="representational_zip_download",
                                )
                        except Exception as exc:  # pragma: no cover - file IO errors
                            st.warning(f"Unable to bundle artifacts for download: {exc}")

                    for artifact_path in rep_result.generated_artifacts[:5]:
                        artifact_obj = _ResultPath(artifact_path)
                        if artifact_obj.is_file() and artifact_obj.suffix.lower() == ".pdf":
                            try:
                                with open(artifact_obj, "rb") as pdf_bytes:
                                    st.download_button(
                                        label=f"⬇️ Download {artifact_obj.name}",
                                        data=pdf_bytes.read(),
                                        file_name=artifact_obj.name,
                                        mime="application/pdf",
                                        key=f"representational_pdf_{artifact_obj.name}",
                                    )
                            except Exception as exc:  # pragma: no cover - file IO errors
                                st.warning(f"Could not open {artifact_obj.name} for download: {exc}")
                    if len(rep_result.generated_artifacts) > 5:
                        st.caption(
                            "Additional artifacts are available in the output directory. Download them from the filesystem if needed."
                        )
                if not rep_result.generated_artifacts and not rep_result.inline_artifacts:
                    st.info("No artifacts were detected. Check the logs and ensure the selected feature produces outputs.")


def render_jailbreak_persuasion_probe_section(api_key, model_choice, provider):
    """Render the Jailbreak Persuasion Probe section for text continuation and comparison."""
    st.markdown("### 🕵️ Jailbreak Persuasion Probe")
    st.markdown(
        """
        This tool uses various prompt engineering techniques to persuade an LLM to continue a given text.
        The generated continuation is then compared against a "Ground Truth" text to measure similarity,
        helping to detect potential memorization of copyrighted content.
        """
    )

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Input Text**")
        input_text_probe = st.text_area(
            "Input Text",
            height=150,
            placeholder="Enter the input snippet to be continued (e.g., 'It was the best of times...').",
            label_visibility="collapsed",
            key="probe_input_text",
        )
    with col2:
        st.markdown("**Ground Truth**")
        ground_truth_probe = st.text_area(
            "Ground Truth",
            height=150,
            placeholder="Enter the expected ground truth continuation to compare against.",
            label_visibility="collapsed",
            key="probe_ground_truth",
        )

    persuasion_strategy = st.selectbox(
        "**Choose a persuasion strategy:**",
        [
            "Role-Playing: The Author",
            "Hypothetical Scenario: A Lost Manuscript",
            "Creative Writing Exercise",
            "Translation and Back-Translation",
            "Stylistic Transformation",
            "Tom and Jerry Game",
        ],
        help="Select a technique to encourage the model to generate a continuation.",
    )

    # Explanations for strategies
    if persuasion_strategy:
        template_text = get_persuasion_template(persuasion_strategy)
        if template_text:
            # Fill placeholders for display
            display_template = template_text.replace("{input_text}", "[Your input text will go here]").replace("{word_count}", "[word_count]")
            st.info(display_template)

    if input_text_probe:
        chunk_size = len(ground_truth_probe.split()) if ground_truth_probe else "{word_count}"
        prompt_to_preview = get_persuasion_prompt(persuasion_strategy, input_text_probe, chunk_size=chunk_size)
        render_prompt_preview(prompt_to_preview)

    if st.button("🚀 Run Probe", width='stretch', key="run_probe_button"):
        if not api_key:
            st.error("⚠️ Please enter your API key in the sidebar.")
        elif not input_text_probe or not ground_truth_probe:
            st.warning("⚠️ Please enter both the Input Text and the Ground Truth text.")
        else:
            with st.spinner(f"🕵️ Running persuasion probe with {model_choice}..."):
                chunk_size = len(ground_truth_probe.split())
                result = run_persuasion_probe(
                    api_key,
                    model_choice,
                    provider,
                    persuasion_strategy,
                    input_text_probe,
                    ground_truth_probe,
                    chunk_size=chunk_size,
                )

                if isinstance(result, str) and result.startswith("Error"):
                    st.error(f"❌ {result}")
                else:
                    generated_text, metrics = result
                    metrics_map = metrics or {}
                    st.success("✅ Probe completed. Review the overlap below.")
                    render_direct_recall_diff(
                        ground_truth_probe,
                        generated_text,
                        title="Ground Truth vs. Probe Output",
                        metrics=metrics_map,
                    )

def render_footer():
    """Renders a footer section."""
    # This is a placeholder for any footer content you might want to add later.
    pass

def main():
    """Main function to run the Streamlit app."""
    render_header()
    api_key, model_choice, provider, page = render_sidebar()

    if page == "Direct Recall Test":
        render_snippet_to_document_page(api_key, model_choice, provider)
    elif page == "Unlearning Detection Test":
        render_unlearning_detection_page(api_key, model_choice, provider)
    elif page == "Persuasive Jailbreak Test":
        render_adversarial_persuasion_page(api_key, model_choice, provider)

    # Footer (currently empty, can be customized)
    render_footer()