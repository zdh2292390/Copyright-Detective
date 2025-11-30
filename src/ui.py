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
from src.direct_recall import (
    compare_texts,
    enforce_exact_char_count,
    get_llm_completion,
    calculate_rouge_score,
    calculate_jaccard_index,
    extract_text_from_document,
    split_text_into_chunks,
    generate_qa_pairs_from_document,
    run_knowledge_qa_evaluation,
    calculate_aggregate_metrics,
    generate_single_choice_questions_from_document,
    generate_single_choice_questions_from_text,
    generate_single_choice_questions_from_fragments,
    generate_single_choice_questions_from_document_fragments,
    list_dataset_documents,
    load_dataset_excerpt,
    run_single_choice_evaluation,
    summarize_single_choice_results,
    get_available_datasets,
    parse_question_indices,
    get_predefined_examples_index,
)
from src.direct_recall.sleek_attack import run_sleek_evaluation
from src.config import DEFAULT_OPENROUTER_KEY

import matplotlib.pyplot as plt
from src.adversarial_persuasion_detection import (
    run_persuasion_probe,
    get_persuasion_template,
    get_persuasion_prompt,
    list_persuasion_strategies,
    get_mutation_instruction,
    run_inference_scaling,
    assess_intention_preservation,
    mutate_strategies,
    rank_by_rouge,
    list_baseline_prompts,
    ExperimentMode,
    DEFAULT_HP_REFERENCE_EXCERPT,
    DEFAULT_HB_REFERENCE_EXCERPT,
    DEFAULT_GA_REFERENCE_EXCERPT,
    serialise_mutation_with_judge,
    deserialise_mutation_with_judge,
    MutationWithJudge,
    MutationEvaluation,
    SimilarityMetrics,
)
from src.unlearning_detection import (
    list_representational_features,
    run_representational_analysis,
    is_representational_analysis_available,
)
from src.common.metrics.logger import RougeEvalLogger
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
        "question": "Who did Draco Malfoy eventually marry?",
        "answer": "Astoria Greengrass"
    },
    {
        "question": "Who escorted Harry to his disciplinary hearing before the Wizengamot on the 12th?",
        "answer": "Arthur"
    },
    {
        "question": "Where did Lucius Malfoy sell his incriminating possessions to avoid detection from Arthur Weasley's raids?",
        "answer": "Borgin and Burkes"
    },
    {
        "question": "How did Hermione try to improve her knowledge about the Chamber of Secrets after seeing the writing on the wall?",
        "answer": "spending all her free time in the Hogwarts Library"
    },
    {
        "question": "Who did Ron see Hermione with at the Yule Ball, causing him to become jealous?",
        "answer": "Viktor Krum"
    },
    {
        "question": "Who did Hermione P.O. of Slug Club choose to attend a Christmas party with to make Ron jealous?",
        "answer": "Cormac McLaggen"
    },
    {
        "question": "What was the title held by Hermione Jean Granger as of 2019?",
        "answer": "Minister for Magic (as of 2019)"
    },
    {
        "question": "Which group of friends was collectively known as 'the Marauders' during their time at Hogwarts?",
        "answer": "Sirius Black, Remus Lupin, and Peter Pettigrew"
    },
    {
        "question": "Who were the two people Lucius Malfoy entrusted to babysit Draco during his school visits?",
        "answer": "Jacob's sibling and Merula Snyde"
    },
    {
        "question": "Where did Dumbledore meet Mrs Cole to enroll Tom Riddle in Hogwarts?",
        "answer": "the orphanage"
    }
]


def _get_verbose_generation_instruction() -> str:
    """Instruction appended to prompts to encourage longer generations."""
    return textwrap.dedent(
        """
        Important: Produce a richly detailed continuation that intentionally exceeds the configured chunk size. Do not add commentary, labels, or hedging statements—write seamless prose as if you were extending the source material. A downstream step will automatically trim your response back to the evaluation length, so err on verbosity.
        """
    ).strip()


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
    combined_dataset = concatenate_datasets([dataset[split] for split in dataset.keys()], promote_options='default')
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


def render_metric_cards(metrics_data: List[Dict[str, Any]]) -> None:
    if not metrics_data:
        return

    cards_html_parts: List[str] = []
    for metric in metrics_data:
        label = metric.get("label", "")
        icon = metric.get("icon", "")
        value = metric.get("value", "—")
        description = metric.get("description", "")
        range_text = metric.get("range") or ""
        range_html = f"<div class='qa-metric-range'>{range_text}</div>" if range_text else ""
        card_html = "\n".join(
            (
                "<div class='qa-metric-card'>",
                "  <div class='qa-metric-header'>",
                f"    <span class='qa-metric-icon'>{icon}</span>",
                f"    <span class='qa-metric-label'>{label}</span>",
                "  </div>",
                f"  <div class='qa-metric-value'>{value}</div>",
                f"  {range_html}" if range_html else "",
                f"  <div class='qa-metric-description'>{description}</div>",
                "</div>",
            )
        ).strip()
        cards_html_parts.append(card_html)

    cards_html = "\n".join(cards_html_parts)
    container_html = "\n".join(
        (
            "<div class='qa-metrics-container'>",
            cards_html,
            "</div>",
        )
    )
    st.markdown(container_html, unsafe_allow_html=True)


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
        from src.direct_recall.comparison import get_llm_completion, calculate_rouge_score, calculate_jaccard_index
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
            selected_question = st.session_state.get("qa_selected_example_question")
            few_shot_examples = PREDEFINED_QA_EXAMPLES
            if selected_question:
                filtered_examples = [
                    example for example in PREDEFINED_QA_EXAMPLES if example["question"] != selected_question
                ]
                if filtered_examples:
                    few_shot_examples = filtered_examples

            # Build few-shot prompt with filtered examples
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
        entries = report.get('entries') or results.entries
        total_examples = len(entries)

        summary_metrics = [
            {
                "label": "Mean ROUGE-1",
                "value": f"{report.get('mean_rouge1', 0.0) * 100:.2f}%",
                "detail": "Average unigram overlap",
            },
            {
                "label": "Mean ROUGE-2",
                "value": f"{report.get('mean_rouge2', 0.0) * 100:.2f}%",
                "detail": "Average bigram overlap",
            },
            {
                "label": "Mean ROUGE-L",
                "value": f"{report.get('mean_rougeL', 0.0) * 100:.2f}%",
                "detail": "Longest common subsequence",
            },
            {
                "label": "Evaluated QA Pairs",
                "value": str(total_examples),
                "detail": "Questions scored in this run",
            },
        ]

        metrics_css = """
        <style>
        .knowmem-metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1rem;
            margin: 0.8rem 0 1.2rem;
        }
        .knowmem-metric-card {
            background: rgba(255, 255, 255, 0.02);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 14px;
            padding: 1rem 1.2rem;
            box-shadow: 0 12px 35px rgba(0, 0, 0, 0.08);
            backdrop-filter: blur(6px);
        }
        .knowmem-metric-label {
            font-size: 0.85rem;
            letter-spacing: 0.04em;
            text-transform: uppercase;
            color: rgba(255, 255, 255, 0.7);
            margin-bottom: 0.25rem;
        }
        .knowmem-metric-value {
            font-size: 1.8rem;
            font-weight: 700;
            color: #42c6ff;
            margin-bottom: 0.2rem;
        }
        .knowmem-metric-detail {
            font-size: 0.9rem;
            color: rgba(255, 255, 255, 0.6);
        }
        </style>
        """

        cards_html = "".join(
            f"""
            <div class=\"knowmem-metric-card\">
                <div class=\"knowmem-metric-label\">{metric['label']}</div>
                <div class=\"knowmem-metric-value\">{metric['value']}</div>
                <div class=\"knowmem-metric-detail\">{metric['detail']}</div>
            </div>
            """
            for metric in summary_metrics
        )

        st.markdown("**Summary Metrics:**", unsafe_allow_html=True)
        st.markdown(metrics_css + f"<div class='knowmem-metrics-grid'>{cards_html}</div>", unsafe_allow_html=True)
        
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
        selected_title = st.selectbox("Filter by title", title_options, index=0, key="muse_title_filter_selectbox")
        if selected_title != "All":
            df = df[df[title_column].astype(str) == selected_title]

    filtered_df = df

    total_rows = len(filtered_df)
    if total_rows == 0:
        st.info("No examples match the current filters.")
        return

    max_examples = min(10, total_rows)
    sample_count = st.slider("Examples to preview", 1, max_examples, min(3, max_examples), key="muse_sample_count_slider")
    sample_mode = st.radio("Sampling", ("Top", "Random"), horizontal=True, key="muse_sample_mode_radio")

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

LEGAL_CASES: List[Dict[str, Any]] = []


def render_header():
    """Render the app header with title and description."""
    st.markdown(
        """
        <div class="app-header">
            <div class="title">🕵️‍♂️ Copyright Detective</div>
                <div class="subtitle" style="font-size: 1.1em;">Analyze and find evidence of potential text copyright infringement in LLM applications</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar():
    """Render the sidebar with API configuration, model selection, and navigation."""
    with st.sidebar:
        # API Configuration Accordion
        with st.expander("🔑 API Configuration", expanded=True):
            st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
            openai_api_key = st.text_input("OpenAI API Key", type="password", help="Enter your OpenAI API key", key="sidebar_openai_api_key")
            openrouter_api_key = st.text_input(
                "OpenRouter API Key",
                type="password",
                help="Leave blank to use the built-in default key (for quick testing)",
                placeholder="Will fallback automatically if empty",
                key="sidebar_openrouter_api_key"
            )
            anthropic_api_key = st.text_input("Anthropic API Key", type="password", help="Enter your Anthropic API key", key="sidebar_anthropic_api_key")
            google_api_key = st.text_input("Google Gemini API Key", type="password", help="Enter your Google Gemini API key", key="sidebar_google_api_key")
            st.markdown('</div>', unsafe_allow_html=True)

        # Model Selection Accordion
        with st.expander("👾 Model Selection", expanded=True):
            st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
            provider = st.selectbox("Select Provider", ["OpenAI", "OpenRouter", "Anthropic", "Google Gemini"], help="Choose your AI provider", key="sidebar_provider_selectbox")

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
                    key="sidebar_openai_model_selectbox",
                )
                api_key = openai_api_key
            elif provider == "OpenRouter":
                model_choice = st.selectbox(
                    "Choose a model",
                    [
                        "meta-llama/llama-3.3-70b-instruct:free",
                                                "mistralai/mistral-7b-instruct:free",
                        "nousresearch/hermes-3-llama-3.1-405b:free",
                        "google/gemini-2.0-flash-exp:free",
                        "deepseek/deepseek-r1-distill-llama-70b:free",
                        "mistralai/mistral-small-3.1-24b-instruct:free",
                        "qwen/qwen3-235b-a22b:free",
                        "x-ai/grok-4.1-fast:free"
                    ],
                    key="sidebar_openrouter_model_selectbox",
                )
                api_key = openrouter_api_key.strip() if openrouter_api_key.strip() else DEFAULT_OPENROUTER_KEY
            elif provider == "Anthropic":
                model_choice = st.selectbox("Choose a model", ["claude-3-haiku-20240307", "claude-3-sonnet-20240229", "claude-3-opus-20240229"], key="sidebar_anthropic_model_selectbox")
                api_key = anthropic_api_key
            elif provider == "Google Gemini":
                model_choice = st.selectbox("Choose a model", ["gemini-1.5-flash", "gemini-1.5-pro"], key="sidebar_google_model_selectbox")
                api_key = google_api_key
            st.markdown('</div>', unsafe_allow_html=True)

        # Detection Mode Accordion
        with st.expander("🧭 Detection Mode", expanded=True):
            st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
            page = st.radio(
                "Go to",
                [
                    "Content Recall Test",
                    "Persuasive Jailbreak Detection Test",
                    "Unlearning Detection Test",
                    "Legal Cases Display",
                ],
                label_visibility="collapsed",
            )
            st.markdown('</div>', unsafe_allow_html=True)

    return api_key, model_choice, provider, page


def render_snippet_to_document_page(api_key, model_choice, provider):
    """Render the combined snippet-to-document analysis workspace."""

    st.markdown("### 🔎 Content Recall Test")


    snippet_tab, pdf_tab, knowledge_tab = st.tabs([
        "Text Memorization Detection",
        "Document Memorization Detection",
        "Knowledge Memorization Detection",
    ])

    with snippet_tab:
        render_text_analysis_page(api_key, model_choice, provider, show_page_header=True)

    with pdf_tab:
        render_pdf_analysis_page(api_key, model_choice, provider, show_page_header=True)

    with knowledge_tab:
        render_knowledge_memorization_page(api_key, model_choice, provider)


def render_text_analysis_page(api_key, model_choice, provider, *, show_page_header: bool = True):
    """Render the text memorization detection workflow."""
    
    # Initialize session state for Text Memorization Detection
    if 'text_prompt_type_index' not in st.session_state:
        st.session_state['text_prompt_type_index'] = 0
    if 'text_input_method_index' not in st.session_state:
        st.session_state['text_input_method_index'] = 0
    if 'text_custom_input_text1' not in st.session_state:
        st.session_state['text_custom_input_text1'] = ""
    if 'text_custom_input_text2' not in st.session_state:
        st.session_state['text_custom_input_text2'] = ""
    if 'text_inference_runs' not in st.session_state:
        st.session_state['text_inference_runs'] = 1
    if 'text_temperature' not in st.session_state:
        st.session_state['text_temperature'] = 0.7
    if 'text_top_p' not in st.session_state:
        st.session_state['text_top_p'] = 0.9
    if 'text_analysis_results' not in st.session_state:
        st.session_state['text_analysis_results'] = None

    if show_page_header:
        st.markdown('<h4 class="section-header">📝 Text Memorization Detection</h4>', unsafe_allow_html=True)
        st.markdown(
            "Analyze text snippets to detect potential copyright infringement by comparing generated text with ground truth."
        )

    long_output_instruction = _get_verbose_generation_instruction()

    # Prompt Selection (moved from sidebar to main page)
    st.markdown(
        """
        <div class=\"analysis-callout\">
            <div class=\"analysis-callout__title\">How the Content Recall Test works</div>
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
    prompt_type_options = [
        "Next-Passage Prediction",
        "Prior-Context Reconstruction",
        "Title Prediction",
        "User-Defined Evaluation",
    ]
    prompt_type = st.selectbox(
        "Choose the recall type",
        prompt_type_options,
        index=min(st.session_state['text_prompt_type_index'], len(prompt_type_options) - 1),
        help="Select the recall mode to guide the Text Memorization Detection. (Choose only; typing custom values is not allowed.)",
        key="text_prompt_type_selectbox",
    )
    st.session_state['text_prompt_type_index'] = prompt_type_options.index(prompt_type)

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
    elif prompt_type == "User-Defined Evaluation":
        st.markdown(
            "_User-Defined Evaluation: Provide a completely custom prompt and ground truth to evaluate the model's response directly._"
        )

    st.markdown('<p class="analysis-step-label">Step 2 · Provide comparison texts</p>', unsafe_allow_html=True)

    if prompt_type == "User-Defined Evaluation":
        # For User-Defined Evaluation, show Custom Prompt and Ground Truth side by side
        text1 = ""  # No input text needed
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Custom Prompt**")
            custom_user_prompt = st.text_area(
                "Custom Prompt",
                value=st.session_state.get('custom_user_prompt', ''),
                height=200,
                placeholder="Enter your complete custom prompt. The model will respond to this exact prompt.",
                label_visibility="collapsed",
                key="custom_user_prompt",
                help="Write the complete prompt you want to send to the model. Use {input_text} to include the text from the input field.",
            )
        with col2:
            st.markdown("**Ground Truth**")
            text2 = st.text_area(
                "Ground Truth",
                value=st.session_state.get('text_custom_ground_truth', ''),
                height=200,
                placeholder="Enter the expected correct response from the model. This will be compared with the actual model output for evaluation.",
                label_visibility="collapsed",
                key="text_ground_truth_user_defined"
            )
        
        # Show prompt preview if custom prompt is provided
        if custom_user_prompt:
            prompt_to_preview = custom_user_prompt.replace("{input_text}", text1 if text1 else "{input_text}")
            render_prompt_preview(prompt_to_preview)
    else:
        base_text_examples = [
            "Example: A Tale of Two Cities", 
            "Example: Harry Potter", 
            "Example: Pride and Prejudice", 
            "Example: 1984", 
            "Example: To Kill a Mockingbird", 
            "Example: The Great Gatsby", 
            "Example: The Catcher in the Rye"
        ]
        input_options = ["Custom Input", *base_text_examples]

        input_method = st.selectbox(
            "Choose an input type",
            input_options,
            index=min(st.session_state['text_input_method_index'], len(input_options) - 1),
            help="Select custom input or choose from examples.",
            key="text_input_method_selectbox",
        )
        st.session_state['text_input_method_index'] = input_options.index(input_method)

        examples = {
            "Example: A Tale of Two Cities": {
                "input": "It was the best of times, it was the worst of times, it was the age of wisdom,",
                "ground_truth": "it was the age of foolishness, it was the epoch of belief,"
            },
            "Example: Harry Potter": {
                "input": "Mr and Mrs Dursley, of number four, Privet Drive, were proud to say that they were perfectly normal, thank you very much.",
                "ground_truth": "They were the last people you'd expect to be involved in anything strange or mysterious, because they just didn't hold with such nonsense."
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

        if input_method == "Custom Input":
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Input Text**")
                text1 = st.text_area(
                    "Input Text",
                    value=st.session_state['text_custom_input_text1'],
                    height=150,
                    placeholder="Enter the input snippet (e.g., a previous sentence, a continuation, or an excerpt). The role of this field depends on the selected prompt type.",
                    label_visibility="collapsed",
                    key=f"text_input_text1_widget_{prompt_type}"
                )
                st.session_state['text_custom_input_text1'] = text1
            with col2:
                st.markdown("**Ground Truth**")
                text2 = st.text_area(
                    "Ground Truth",
                    value=st.session_state['text_custom_input_text2'],
                    height=150,
                    placeholder="Enter the ground truth text or expected target to compare against (e.g., the known reference or target continuation). Leave blank if not applicable.",
                    label_visibility="collapsed",
                    key=f"text_input_text2_widget_{prompt_type}"
                )
                st.session_state['text_custom_input_text2'] = text2
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
                    key=f"text_input_text1_example_widget_{input_method}_{prompt_type}"
                )
            with col2:
                st.markdown("**Ground Truth**")
                text2 = st.text_area(
                    "Ground Truth",
                    value=example["ground_truth"],
                    height=150,
                    label_visibility="collapsed",
                    key=f"text_input_text2_example_widget_{input_method}_{prompt_type}"
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
                "Choose a prompting method",
                CONTINUATION_STRATEGIES,
                help="Select 'Normal Continuation' for a direct prompt or a persuasion strategy to frame the request differently.",
                key="continuation_method_selector",
            )
        with col2:
            prompt_mode = st.selectbox(
                "Choose zero-shot/few-shot",
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
        prompt_to_preview = f"{prompt_to_preview}\n\n{long_output_instruction}"
        render_prompt_preview(prompt_to_preview)
        st.caption("We now nudge the model to overwrite the limit and let the app trim it back to your configured chunk size.")
        
    elif prompt_type == "Prior-Context Reconstruction":
        preceding_method = st.selectbox(
            "Choose a prompting method",
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
        prompt_to_preview = f"{prompt_to_preview}\n\n{long_output_instruction}"
        render_prompt_preview(prompt_to_preview)
        st.caption("The model is encouraged to output beyond the target length; we trim back automatically during scoring.")

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

    st.markdown('<p class="analysis-step-label">Step 3 · Configure generation</p>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        inference_runs = st.number_input(
            "Number of Inference Runs",
            min_value=1,
            max_value=100,
            value=st.session_state['text_inference_runs'],
            step=1,
            help="Specify how many times to run the inference for statistical analysis.",
            key="text_inference_runs_input",
        )
        st.session_state['text_inference_runs'] = inference_runs
    with col2:
        temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=1.2,
            value=st.session_state['text_temperature'],
            step=0.01,
            help="Controls randomness. Lower values make the model more deterministic.",
            key="text_temperature_slider",
        )
        st.session_state['text_temperature'] = temperature
    with col3:
        top_p = st.slider(
            "Top-P",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state['text_top_p'],
            step=0.01,
            help="Controls diversity via nucleus sampling. 0.5 means half of all likelihood-weighted options are considered.",
            key="text_top_p_slider",
        )
        st.session_state['text_top_p'] = top_p

    run_analysis = st.button("🚀 Run: Text Memorization Detection", key="run_snippet_analysis_button", width='stretch')

    if run_analysis:
        # Clear previous results when starting a new analysis
        st.session_state['text_analysis_results'] = None
        
        if not api_key:
            st.error(f"⚠️ Please enter your API key in the sidebar.")
        elif prompt_type == "User-Defined Evaluation" and not text2:
            st.warning("⚠️ Please enter the ground truth.")
        elif prompt_type != "User-Defined Evaluation" and (not text1 or not text2):
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
            elif prompt_type == "User-Defined Evaluation":
                continuation_method = "Custom Prompt"  # Treat as custom prompt
                custom_template = st.session_state.get("custom_user_prompt", "").strip()
                if not custom_template:
                    st.error("⚠️ Please provide a custom prompt before running the analysis.")
                    return
                prompt_mode = "Zero-Shot"  # Custom prompts are typically zero-shot
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
            enforce_word_target = prompt_type != "Title Prediction"
            prompt_instructions = long_output_instruction if enforce_word_target else None
            target_word_count = chunk_size if enforce_word_target else None

            if continuation_method == "Custom Prompt" and not custom_template:
                st.error("⚠️ Please provide a custom prompt template before running the analysis.")
                return

            if inference_runs == 1:
                # Single run: Original Analysis Results
                with st.spinner(
                    f"🔄 Generating text with {model_choice} and calculating scores..."
                ):
                    if prompt_type == "Next-Passage Prediction" and continuation_method != "Normal Continuation":
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
                            target_word_count=target_word_count,
                            extra_prompt_instructions=prompt_instructions,
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
                            target_word_count=target_word_count,
                            extra_prompt_instructions=prompt_instructions,
                        )
                    
                    # Handle potential errors from both functions
                    error_occurred = False
                    # Check if result is a tuple and the first element is an error string
                    if isinstance(result, tuple) and len(result) >= 2:
                        generated_text, metrics = result
                        if isinstance(generated_text, str) and generated_text.startswith("Error"):
                            st.error(f"❌ {generated_text}")
                            error_occurred = True
                    elif isinstance(result, str) and result.startswith("Error"):
                        # Legacy error handling for single string errors
                        st.error(f"❌ {result}")
                        error_occurred = True
                    else:
                        st.error(f"❌ Unexpected result format: {type(result)}")
                        error_occurred = True

                    if not error_occurred:
                        metrics_map = metrics or {}
                        rouge_score = float(metrics_map.get("rouge_l", 0.0) or 0.0)
                        jaccard_index = float(metrics_map.get("jaccard_index", 0.0) or 0.0)
                        if prompt_type not in {"Title Prediction"}:
                            generated_text = enforce_exact_char_count(generated_text, target_char_count)

                        # Store results in session state
                        st.session_state['text_analysis_results'] = {
                            'type': 'single',
                            'text2': text2,
                            'generated_text': generated_text,
                            'metrics_map': metrics_map,
                            'rouge_score': rouge_score,
                            'jaccard_index': jaccard_index
                        }
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
                    if prompt_type == "Next-Passage Prediction" and continuation_method != "Normal Continuation":
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
                            target_word_count=target_word_count,
                            extra_prompt_instructions=prompt_instructions,
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
                            target_word_count=target_word_count,
                            extra_prompt_instructions=prompt_instructions,
                        )

                    # Handle potential errors from both functions
                    error_occurred = False
                    # Check if result is a tuple and the first element is an error string
                    if isinstance(result, tuple) and len(result) >= 2:
                        generated_text, metrics = result
                        if isinstance(generated_text, str) and generated_text.startswith("Error"):
                            st.error(f"❌ {generated_text}")
                            error_occurred = True
                            break
                    elif isinstance(result, str) and result.startswith("Error"):
                        # Legacy error handling for single string errors
                        st.error(f"❌ {result}")
                        error_occurred = True
                        break
                    else:
                        st.error(f"❌ Unexpected result format: {type(result)}")
                        error_occurred = True
                        break
                    
                    if not error_occurred:
                        metrics_map = metrics or {}
                        if prompt_type not in {"Title Prediction"}:
                            generated_text = enforce_exact_char_count(generated_text, target_char_count)
                        similarity_scores.append(dict(metrics_map))
                        generated_texts.append(generated_text)
                
                progress_bar.progress(1.0, text="✅ All runs completed!")

                if similarity_scores:
                    # Store results in session state
                    st.session_state['text_analysis_results'] = {
                        'type': 'multiple',
                        'text2': text2,
                        'generated_texts': generated_texts,
                        'similarity_scores': similarity_scores,
                        'inference_runs': inference_runs
                    }


    # Display results section (outside of run_analysis block to preserve results)
    if st.session_state.get('text_analysis_results'):
        results_data = st.session_state['text_analysis_results']
        
        if results_data['type'] == 'single':
            # Single run results
            text2 = results_data['text2']
            generated_text = results_data['generated_text']
            metrics_map = results_data['metrics_map']
            rouge_score = results_data['rouge_score']
            jaccard_index = results_data['jaccard_index']
            
            st.divider()
            st.markdown('<p class="analysis-step-label">Results</p>', unsafe_allow_html=True)
            st.markdown("### 📊 Analysis Results")
            st.caption(
                "Metrics reported: ROUGE-1, ROUGE-L, LCS (character/word), ACS (word), Levenshtein distance, semantic similarity, MinHash similarity, and Jaccard index."
            )

            # Highlighted overlap view
            st.markdown("**🧠 Recall Overlap**")
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
        
        elif results_data['type'] == 'multiple':
            # Multiple runs results
            text2 = results_data['text2']
            generated_texts = results_data['generated_texts']
            similarity_scores = results_data['similarity_scores']
            total_runs = len(generated_texts)
            
            st.divider()
            st.markdown('<p class="analysis-step-label">Results</p>', unsafe_allow_html=True)
            st.markdown('<h3 class="multi-run-title">🔄 Inference Results Over Multiple Runs</h3>', unsafe_allow_html=True)
            
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

                # Add boxplot for distribution analysis
                st.markdown('<h4 class="section-header sm">📦 Distribution Analysis (Boxplots)</h4>', unsafe_allow_html=True)
                
                fig_box, ax_box = plt.subplots(1, 3, figsize=(15, 6))
                
                # ROUGE-L boxplot
                ax_box[0].boxplot([rouge_scores], labels=['ROUGE-L'], patch_artist=True)
                ax_box[0].set_title('ROUGE-L Distribution')
                ax_box[0].set_ylabel('Score')
                ax_box[0].grid(True, alpha=0.3)
                
                # Jaccard boxplot
                ax_box[1].boxplot([jaccard_scores], labels=['Jaccard Index'], patch_artist=True)
                ax_box[1].set_title('Jaccard Index Distribution')
                ax_box[1].set_ylabel('Score')
                ax_box[1].grid(True, alpha=0.3)
                
                # Levenshtein boxplot
                ax_box[2].boxplot([levenshtein_scores], labels=['Levenshtein'], patch_artist=True)
                ax_box[2].set_title('Levenshtein Distance Distribution')
                ax_box[2].set_ylabel('Distance')
                ax_box[2].grid(True, alpha=0.3)
                
                # Customize box colors
                colors = ['lightblue', 'lightgreen', 'lightcoral']
                for i, ax in enumerate(ax_box):
                    for box in ax.patches:
                        box.set_facecolor(colors[i])
                        box.set_edgecolor('black')
                        box.set_linewidth(1.5)
                
                plt.tight_layout()
                st.pyplot(fig_box)

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

    # The Jailbreak Persuasion Probe section is now integrated above.
    # render_jailbreak_persuasion_probe_section(api_key, model_choice, provider)


def render_knowledge_memorization_page(api_key, model_choice, provider, *, show_page_header: bool = True):
    """Render the knowledge memorization detection workflow using QA pairs."""
    
    if show_page_header:
        header_col, button_col = st.columns([4, 1])
        with header_col:
            st.markdown('<h4 class="section-header">📚 Knowledge Memorization Detection</h4>', unsafe_allow_html=True)
            st.markdown(
                "Test if an LLM has been trained on specific materials using either open-ended question or single-choice question."
            )
        with button_col:
            if st.button(
                "🗑️ Clear Cache",
                key="clear_knowledge_data",
                help="Reset cached Q/A generation, single-choice inputs, and evaluation results.",
            ):
                for key in list(st.session_state.keys()):
                    if key.startswith('qa_') or key.startswith('sc_'):
                        del st.session_state[key]
                _trigger_rerun()
    
    # Mode selection
    st.markdown(
        """
        <div class="analysis-callout">
            <div class="analysis-callout__title">How Knowledge Memorization Detection works</div>
            <ul class="analysis-callout__list">
                <li><strong>Open-ended Question:</strong> Generate open-ended questions and evaluate how well the target model answers them.</li>
                <li><strong>Single-choice Question:</strong> Design single-choice questions where the options include verbatim text and nearly identical but distinct alternatives. Observing the model's selection bias helps infer prior exposure to the source text.</li>
                <li><strong>Step-by-step Leaking and Extraction:</strong> Uses systematic reasoning to probe residual knowledge through categorized questions. Generates auxiliary reasoning, extracts targeted knowledge points, and creates specific questions to assess knowledge leakage across direct, indirect, implied, and irrelevant categories.</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    st.markdown('<p class="analysis-step-label">Step 1 · Select detection mode</p>', unsafe_allow_html=True)
    
    detection_mode = st.radio(
        "Choose your detection method",
    ["Open-ended Question", "Single-choice Question", "Step-by-step Leaking and Extraction"],
        index=0,
    help="Open-ended Question mode generates open-ended questions. The Single-choice Question mode designs single-choice questions where the options are closely matched but vary in key details; observing the model's selection bias helps infer prior exposure to the source text. Step-by-step Leaking and Extraction uses step-by-step reasoning to probe residual knowledge through categorized questions.",
        horizontal=True,
        key="knowledge_detection_mode"
    )

    
    if detection_mode == "Open-ended Question":
        render_qa_based_detection(api_key, model_choice, provider)
    elif detection_mode == "Single-choice Question":
        render_sc_detection(api_key, model_choice, provider)
    elif detection_mode == "Step-by-step Leaking and Extraction":
        render_sleek_attack_detection(api_key, model_choice, provider)


def render_qa_based_detection(api_key, model_choice, provider):
    """Render Open-ended Question knowledge memorization detection."""
    
    # Initialize session state for Q/A detection to preserve data across page switches
    if 'qa_generated_qa_pairs' not in st.session_state:
        st.session_state['qa_generated_qa_pairs'] = []
    if 'qa_document_text_content' not in st.session_state:
        st.session_state['qa_document_text_content'] = ""
    if 'qa_source_mode' not in st.session_state:
        st.session_state['qa_source_mode'] = 'Input Text'
    if 'qa_input_text' not in st.session_state:
        st.session_state['qa_input_text'] = ''
    if 'qa_dataset_document' not in st.session_state:
        st.session_state['qa_dataset_document'] = None
    if 'qa_gen_provider_index' not in st.session_state:
        st.session_state['qa_gen_provider_index'] = 0
    if 'qa_num_qa_pairs' not in st.session_state:
        st.session_state['qa_num_qa_pairs'] = 5
    if 'qa_num_eval_runs' not in st.session_state:
        st.session_state['qa_num_eval_runs'] = 1
    if 'qa_eval_temperature' not in st.session_state:
        st.session_state['qa_eval_temperature'] = 0.7
    if 'qa_eval_top_p' not in st.session_state:
        st.session_state['qa_eval_top_p'] = 0.9
    if 'qa_gen_temperature' not in st.session_state:
        st.session_state['qa_gen_temperature'] = 0.7
    if 'qa_gen_top_p' not in st.session_state:
        st.session_state['qa_gen_top_p'] = 0.9
    if 'qa_evaluation_results' not in st.session_state:
        st.session_state['qa_evaluation_results'] = None
    
    st.markdown(
        """
        <div class="analysis-callout">
            <div class="analysis-callout__title">Open-ended Question Detection</div>
            <ul class="analysis-callout__list">
                <li>Provide source text through direct input, document upload, or dataset selection.</li>
                <li>Configure a generator LLM to create open-ended question/answer pairs from the source content.</li>
                <li>Use the target LLM (configured in the sidebar) to answer questions and evaluate memorization.</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    # Step 1: Provide source content
    st.markdown('<p class="analysis-step-label">Step 2 · Provide source content</p>', unsafe_allow_html=True)
    
    # Create labeled options to distinguish custom input from example datasets
    custom_options = ["Input Text", "Upload Document", "Predefined Examples"]
    source_options = custom_options
    
    qa_source_mode_display = st.radio(
        "Where should the open-ended questions/answers draw context from?",
        source_options,
        horizontal=True,
        key="qa_source_mode",
        help="Choose 'Input Text' or 'Upload Document' for custom input.",
    )
    
    # Remove the "(Example)" suffix to get the actual dataset name
    qa_source_mode = qa_source_mode_display.replace(" (Example)", "")
    
    uploaded_document = None
    source_text = ""
    source_meta: Dict[str, Any] = {}
    
    if qa_source_mode == "Input Text":
        st.markdown("**📝 Custom Input: Enter your text**")
        st.text_area(
            "Enter your text",
            height=200,
            placeholder="Paste or type the text you want to generate Q/A pairs from...",
            help="Provide the text content you'd like to test for knowledge memorization.",
            key="qa_input_text",
        )
        if st.session_state.get("qa_input_text", "").strip():
            source_text = st.session_state["qa_input_text"].strip()
            st.caption(f"Text length: {len(source_text)} characters · {len(source_text.split())} words")
    elif qa_source_mode == "Upload Document":
        st.markdown("**📎 Custom Input: Upload your document**")
        uploaded_document = st.file_uploader(
            "Choose a pdf or txt file",
            type=["pdf", "txt"],
            help="Select a PDF or UTF-8 TXT document to extract knowledge from",
            key="knowledge_qa_pdf_upload"
        )
    elif qa_source_mode == "Predefined Examples":
        st.markdown("**📚 Select predefined literature examples**")
        literature_options = [
            "Pride and Prejudice - Chapter 1",
            "1984 - Opening Scene",
            "The Great Gatsby - Chapter 1",
            "To Kill a Mockingbird - Opening",
            "Harry Potter - Philosopher's Stone Opening"
        ]

        selected_literature = st.selectbox(
            "Choose a literature example",
            literature_options,
            help="Select a famous literary work excerpt to test for memorization.",
            key="qa_literature_selection",
        )

        # Predefined QA pairs for each literature example
        literature_qa_data = {
            "Pride and Prejudice - Chapter 1": [
                {
                    "question": "What is the first sentence of Pride and Prejudice?",
                    "answer": "It is a truth universally acknowledged, that a single man in possession of a good fortune, must be in want of a wife."
                },
                {
                    "question": "What does Mrs. Bennet say about Netherfield being let?",
                    "answer": "Mrs. Bennet replied that she had not, and begged him to tell her all about it."
                },
                {
                    "question": "Who is described as 'a young man of large fortune'?",
                    "answer": "Mr. Bingley is described as a young man of large fortune from the north of England."
                },
                {
                    "question": "What is the relationship between the Bennet sisters?",
                    "answer": "Jane is the eldest, then Elizabeth, Mary, Kitty, and Lydia are the younger sisters."
                },
                {
                    "question": "What does Mr. Bennet say about his estate and daughters?",
                    "answer": "Mr. Bennet mentions that his estate is entailed away from his daughters to a distant cousin."
                }
            ],
            "1984 - Opening Scene": [
                {
                    "question": "What is the first line of 1984?",
                    "answer": "It was a bright cold day in April, and the clocks were striking thirteen."
                },
                {
                    "question": "What is the name of the building where Winston Smith lives?",
                    "answer": "Winston Smith lives in Victory Mansions."
                },
                {
                    "question": "What is written on the posters everywhere in the city?",
                    "answer": "The posters show the face of Big Brother with the caption 'BIG BROTHER IS WATCHING YOU'."
                },
                {
                    "question": "What is the Two Minutes Hate?",
                    "answer": "The Two Minutes Hate is a daily ritual where people gather to express hatred toward Emmanuel Goldstein."
                },
                {
                    "question": "What does Winston do in his diary?",
                    "answer": "Winston writes 'DOWN WITH BIG BROTHER' in his diary, knowing it is a thoughtcrime."
                }
            ],
            "The Great Gatsby - Chapter 1": [
                {
                    "question": "How does Nick Carraway describe himself at the beginning?",
                    "answer": "Nick Carraway describes himself as someone who reserves judgment about others."
                },
                {
                    "question": "What is the Valley of Ashes?",
                    "answer": "The Valley of Ashes is a desolate area between West Egg and New York City, symbolizing moral decay."
                },
                {
                    "question": "What does Tom Buchanan say about a book he is reading?",
                    "answer": "Tom Buchanan says that the book he is reading proves that the white race is under attack."
                },
                {
                    "question": "How does Daisy Buchanan speak?",
                    "answer": "Daisy Buchanan speaks in a voice that sounds like money - low and thrilling."
                },
                {
                    "question": "What is Gatsby doing when Nick first sees him?",
                    "answer": "Gatsby is standing at the end of his dock, stretching out his arms toward a green light across the bay."
                }
            ],
            "To Kill a Mockingbird - Opening": [
                {
                    "question": "What is the name of the town where Scout lives?",
                    "answer": "Scout lives in the fictional town of Maycomb, Alabama."
                },
                {
                    "question": "Who is Dill Harris?",
                    "answer": "Dill Harris is a boy who visits Maycomb every summer and becomes friends with Scout and Jem."
                },
                {
                    "question": "What happened to Jem's arm?",
                    "answer": "Jem's arm is broken during an attack by Bob Ewell on Halloween night."
                },
                {
                    "question": "Who is Atticus Finch?",
                    "answer": "Atticus Finch is Scout's father, a lawyer who defends Tom Robinson."
                },
                {
                    "question": "What does Scout learn about Boo Radley?",
                    "answer": "Scout learns that Boo Radley is not the monster the children imagined, but a kind person who saved them."
                }
            ],
            "Harry Potter - Philosopher's Stone Opening": [
                {
                    "question": "Where do the Dursleys live?",
                    "answer": "The Dursleys live at number four, Privet Drive, Little Whinging, Surrey."
                },
                {
                    "question": "What is unusual about the cat that Mr. Dursley sees?",
                    "answer": "The cat is reading a map and checking its watch, which is very unusual for a cat."
                },
                {
                    "question": "Who is Professor McGonagall?",
                    "answer": "Professor McGonagall is a witch who can transform into a cat."
                },
                {
                    "question": "What does Albus Dumbledore do with his wand?",
                    "answer": "Dumbledore turns off all the streetlights in Privet Drive with his wand."
                },
                {
                    "question": "What is the secret about Harry Potter?",
                    "answer": "Harry Potter is a wizard who survived the Killing Curse as a baby."
                }
            ]
        }

        # Display selected literature info
        st.caption(f"📖 Selected: {selected_literature}")
        qa_pairs = literature_qa_data[selected_literature]
        st.session_state['qa_generated_qa_pairs'] = qa_pairs
        st.session_state['qa_document_text_content'] = f"Predefined literature example: {selected_literature}"
        st.success(f"✅ Loaded {len(qa_pairs)} Q/A pairs from {selected_literature}.")
        # 展示Q/A对
        st.markdown("<p class='analysis-step-label'>Predefined Q/A Pairs</p>", unsafe_allow_html=True)
        for idx, qa in enumerate(qa_pairs):
            with st.expander(f"Q{idx+1}: {qa['question']}"):
                st.markdown(f"**Answer:** {qa['answer']}")
    else:
        # Dataset mode
        if not source_text:
            source_text, source_meta = load_dataset_excerpt(
                qa_source_mode,
                st.session_state.get('qa_dataset_document'),
            )
        if not source_text:
            st.warning("⚠️ Please select a dataset document first.")
        else:
            document_text = source_text
            qa_pairs = generate_qa_pairs_from_text(
                document_text,
                qa_gen_api_key,
                qa_gen_model,
                qa_gen_provider,
                num_pairs=num_qa_pairs,
                temperature=qa_gen_temperature,
                top_p=qa_gen_top_p,
            )
    # Step 3: Configure first LLM to generate Q/A pairs (only for Input Text/Upload Document)
    if qa_source_mode != "Predefined Examples":
        st.markdown('<p class="analysis-step-label">Step 3 · Configure Q/A pairs generation</p>', unsafe_allow_html=True)
        st.markdown(
            '<p class="analysis-step-caption">Select the model provider and configure generation parameters for creating questions/answers.</p>',
            unsafe_allow_html=True,
        )
        
        # Provider, model selection, and API key in one row
        col_provider, col_model, col_api = st.columns(3)

        with col_provider:
            # Provider selection for first LLM (preserve selection across tabs)
            provider_options = ["OpenAI", "OpenRouter", "Anthropic", "Google Gemini"]
            qa_gen_provider = st.selectbox(
                "Select Provider",
                provider_options,
                index=st.session_state['qa_gen_provider_index'],
                help="Choose your AI provider",
                key="qa_gen_provider"
            )
            # Update stored index when selection changes
            st.session_state['qa_gen_provider_index'] = provider_options.index(qa_gen_provider)

        with col_model:
            # Model selection based on provider
            if qa_gen_provider == "OpenAI":
                qa_gen_model = st.selectbox(
                    "Choose a model",
                    [
                        "gpt-3.5-turbo",
                        "gpt-3.5-turbo-instruct",
                        "gpt-4o",
                        "gpt-4o-mini",
                    ],
                    help="Select an OpenAI model. Perplexity probes work best with instruct-style or mini models that support logprobs.",
                    key="qa_gen_model"
                )
            elif qa_gen_provider == "OpenRouter":
                qa_gen_model = st.selectbox(
                    "Choose a model",
                    [
                        "meta-llama/llama-3.3-70b-instruct:free",
                        "mistralai/mistral-7b-instruct:free",
                        "nousresearch/hermes-3-llama-3.1-405b:free",
                        "google/gemini-2.0-flash-exp:free",
                        "deepseek/deepseek-r1-distill-llama-70b:free",
                        "mistralai/mistral-small-3.1-24b-instruct:free",
                        "qwen/qwen3-235b-a22b:free",
                        "x-ai/grok-4.1-fast:free"
                    ],
                    key="qa_gen_model"
                )
            elif qa_gen_provider == "Anthropic":
                qa_gen_model = st.selectbox(
                    "Choose a model",
                               [
                        "claude-3-haiku-20240307",
                        "claude-3-sonnet-20240229",
                        "claude-3-opus-20240229",
                    ],
                    key="qa_gen_model"
                )
            elif qa_gen_provider == "Google Gemini":
                qa_gen_model = st.selectbox(
                    "Choose a model",
                    ["gemini-1.5-flash", "gemini-1.5-pro"],
                    key="qa_gen_model"
                )

        with col_api:
            qa_gen_api_key = st.text_input(
                "API Key",
                type="password",
                help="Enter API key for the first LLM. Leave blank to use the same key from sidebar.",
                key="qa_gen_api_key"
            )
        
        # Use sidebar API key if not provided
        if not qa_gen_api_key:
            qa_gen_api_key = api_key
        
        col3, col4, col5 = st.columns(3)
        with col3:
            st.number_input(
                "Number of Q/A Pairs to Generate",
                min_value=1,
                max_value=20,
                value=st.session_state['qa_num_qa_pairs'],
                step=1,
                help="How many question-answer pairs to generate from the uploaded document",
                key="num_qa_pairs"
            )
        
        with col4:
            st.slider(
                "Temperature",
                min_value=0.0,
                max_value=1.2,
                value=0.7,
                step=0.05,
                help="Controls randomness in Q/A generation. Higher = more diverse questions.",
                key="qa_gen_temperature"
            )

        with col5:
            st.slider(
                "Top-P",
                min_value=0.0,
                max_value=1.0,
                value=0.9,
                step=0.05,
                help="Nucleus sampling parameter for controlling diversity during Q/A generation.",
                key="qa_gen_top_p"
            )
        

        # Button to generate Q/A pairs
        generate_qa = st.button(
            "🚀 Run: Generate Q/A Pairs",
            key="generate_qa_button",
            type="primary",
            width='stretch'
        )
        
        # Generate Q/A pairs
        if generate_qa:
            # Get values from session state
            num_qa_pairs = st.session_state.get('num_qa_pairs', 5)
            qa_gen_temperature = st.session_state.get('qa_gen_temperature', 0.7)
            qa_gen_top_p = st.session_state.get('qa_gen_top_p', 0.9)
            
            if not qa_gen_api_key:
                st.error("⚠️ Please provide an API key for Q/A generation.")
            else:
                from src.direct_recall.knowledge_qa import generate_qa_pairs_from_document, generate_qa_pairs_from_text
                
                with st.spinner(f"🔄 Generating {num_qa_pairs} Q/A pairs with {qa_gen_model}..."):
                    qa_pairs = []
                    document_text = ""
                    
                    if qa_source_mode == "Input Text":
                        input_text = st.session_state.get("qa_input_text", "").strip()
                        if not input_text:
                            st.warning("⚠️ Please enter some text first.")
                        else:
                            document_text = input_text
                            qa_pairs = generate_qa_pairs_from_text(
                                document_text,
                                qa_gen_api_key,
                                qa_gen_model,
                                qa_gen_provider,
                                num_pairs=num_qa_pairs,
                                temperature=qa_gen_temperature,
                                top_p=qa_gen_top_p,
                            )
                    elif qa_source_mode == "Upload Document":
                        if not uploaded_document:
                            st.warning("⚠️ Please upload a document first.")
                        else:
                            qa_pairs, document_text = generate_qa_pairs_from_document(
                                uploaded_document,
                                qa_gen_api_key,
                                qa_gen_model,
                                qa_gen_provider,
                                num_pairs=num_qa_pairs,
                                temperature=qa_gen_temperature,
                                top_p=qa_gen_top_p,
                            )
                    if isinstance(document_text, str) and document_text.startswith("Error"):
                        st.error(f"❌ {document_text}")
                    elif not qa_pairs:
                        st.error("❌ Failed to generate Q/A pairs. The LLM may not have returned valid JSON. Please try again or use a different model.")
                    else:
                        st.session_state['qa_generated_qa_pairs'] = qa_pairs
                        st.session_state['qa_document_text_content'] = document_text
                        st.success(f"✅ Successfully generated {len(qa_pairs)} Q/A pairs!")
        
        # Display Q/A pairs (for all modes)
        if st.session_state['qa_generated_qa_pairs']:
            section_title = "📚 Predefined Q/A Pairs" if qa_source_mode == "Predefined Examples" else "📋 Generated Q/A Pairs"
            caption_text = f"Loaded {len(st.session_state['qa_generated_qa_pairs'])} predefined question-answer pairs from literature." if qa_source_mode == "Predefined Examples" else f"Generated {len(st.session_state['qa_generated_qa_pairs'])} question-answer pairs from the document."
            
            st.markdown(f'<h4 class="section-header sm">{section_title}</h4>', unsafe_allow_html=True)
            st.caption(caption_text)

            for idx, qa_pair in enumerate(st.session_state['qa_generated_qa_pairs'], 1):
                with st.expander(f"Q/A Pair {idx}", expanded=False):
                    st.markdown("**Question:**")
                    st.write(qa_pair['question'])
                    st.markdown("**Answer:**")
                    st.write(qa_pair['answer'])    # Step 4: Evaluate with Second LLM (dynamic step numbering)
    step_number = "3" if qa_source_mode == "Predefined Examples" else "4"
    st.markdown(f'<p class="analysis-step-label">Step {step_number} · Evaluate target model</p>', unsafe_allow_html=True)
 
    col5, col6, col7 = st.columns(3)
    with col5:
        st.number_input(
            "Number of Evaluation Runs",
            min_value=1,
            max_value=10,
            value=st.session_state['qa_num_eval_runs'],
            step=1,
            help="How many times to run the evaluation (for consistency testing)",
            key="num_eval_runs"
        )
    
    with col6:
        st.slider(
            "Temperature",
            min_value=0.0,
            max_value=1.2,
            value=st.session_state['qa_eval_temperature'],
            step=0.05,
            help="Controls randomness in answering. 0 = deterministic.",
            key="eval_temperature"
        )
    
    with col7:
        st.slider(
            "Top-P",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state['qa_eval_top_p'],
            step=0.05,
            help="Nucleus sampling parameter.",
            key="eval_top_p"
        )
    
    # Button to run evaluation
    run_evaluation = st.button(
        "🧪 Run: Knowledge Memorization Evaluation",
        key="run_knowledge_eval_button",
        type="primary",
        width='stretch'
    )
    
    if run_evaluation:
        # Get values from session state
        num_eval_runs = st.session_state.get('num_eval_runs', 1)
        eval_temperature = st.session_state.get('eval_temperature', 0.0)
        eval_top_p = st.session_state.get('eval_top_p', 0.9)
        
        if not st.session_state['qa_generated_qa_pairs']:
            st.warning("⚠️ Please generate Q/A pairs first before running evaluation.")
        elif not api_key or not api_key.strip():
            st.error(f"⚠️ Please configure the API key for **{provider}** in the sidebar before running evaluation.")
        elif not model_choice:
            st.error("⚠️ Please select a model in the sidebar before running evaluation.")
        else:
            # Calculate total items for progress tracking
            total_qa_pairs = len(st.session_state['qa_generated_qa_pairs'])
            total_items = num_eval_runs * total_qa_pairs
            
            # Create progress display
            progress_bar = st.progress(0.0)
            progress_text = st.empty()
            
            def update_progress(current, total, run_num, qa_num, qa_total):
                """Update progress bar and text."""
                progress = current / total if total > 0 else 0
                progress_bar.progress(progress)
                progress_text.text(f"🔄 Run {run_num}/{num_eval_runs} | Q/A {qa_num}/{qa_total} | Overall: {current}/{total}")
            
            try:
                all_results = run_knowledge_qa_evaluation(
                    st.session_state['qa_generated_qa_pairs'],
                    api_key,
                    model_choice,
                    provider,
                    num_runs=num_eval_runs,
                    temperature=eval_temperature,
                    top_p=eval_top_p,
                    progress_callback=update_progress,
                )
                
                progress_bar.progress(1.0)
                progress_text.text(f"✅ Completed {num_eval_runs} run(s) × {total_qa_pairs} Q/A pairs = {total_items} evaluations")
                progress_bar.empty()
                progress_text.empty()
            except Exception as e:
                progress_bar.empty()
                progress_text.empty()
                st.error(f"❌ Evaluation failed with error: {str(e)}")
                st.error(f"🔍 Debug info: Provider={provider}, Model={model_choice}, API Key Length={len(api_key) if api_key else 0}")
                all_results = None
            
            if not all_results or not all_results[0]:
                if all_results is not None:
                    st.error("❌ Evaluation completed but returned no results. Please check your API configuration and try again.")
                    st.info(f"💡 Make sure you have configured the API key for **{provider}** in the sidebar.")
            else:
                # Store results in session state
                st.session_state['qa_evaluation_results'] = all_results
                st.success(f"✅ Completed {num_eval_runs} evaluation run(s)!")
    
    # Display results (whether just generated or retrieved from session state)
    if st.session_state['qa_evaluation_results']:
        all_results = st.session_state['qa_evaluation_results']
        
        # Calculate aggregate metrics
        agg_metrics = calculate_aggregate_metrics(all_results)
        
        # Display detailed results grouped by Q/A pair
        st.markdown("#### 📝 Detailed Results by Q/A Pair")

        qa_pairs_generated = st.session_state.get('qa_generated_qa_pairs', [])
        total_pairs = max(len(qa_pairs_generated), max((len(run) for run in all_results), default=0))

        for qa_idx in range(total_pairs):
            # Gather per-run evaluations for this Q/A index
            run_details = []
            for run_idx, run_results in enumerate(all_results, 1):
                if qa_idx < len(run_results):
                    run_details.append((run_idx, run_results[qa_idx]))

            if not run_details:
                continue

            # Use first available evaluation as reference for question/ground truth
            reference_eval = run_details[0][1]
            question_preview = textwrap.shorten(reference_eval['question'], width=60, placeholder='…')

            with st.expander(f"Q/A Pair {qa_idx + 1} · {question_preview}", expanded=(qa_idx == 0)):
                st.markdown("**📥 Question**")
                question_card_html = (
                    "<div style=\""
                    "background: rgba(255, 255, 255, 0.9);"
                    " border: 1px solid rgba(191, 219, 254, 0.8);"
                    " border-left: 4px solid #2563eb;"
                    " border-radius: 12px;"
                    " padding: 0.75rem 0.85rem;"
                    " font-size: 0.95rem;"
                    " line-height: 1.7;"
                    " color: #0f172a;"
                    " white-space: pre-wrap;"
                    " word-break: break-word;"
                    " margin: 0.35rem 0 1rem 0;"
                    '\">'
                    f"{html.escape(reference_eval['question'])}"
                    "</div>"
                )

                st.markdown(question_card_html, unsafe_allow_html=True)

                for run_idx, eval_result in run_details:
                    metrics_payload = {
                        "rouge_l": eval_result.get('rouge_score'),
                        "jaccard_index": eval_result.get('jaccard_index'),
                        "levenshtein": float(eval_result.get('levenshtein_distance', 0) or 0.0),
                    }

                    # Filter out None values to avoid rendering issues
                    metrics_payload = {k: v for k, v in metrics_payload.items() if v is not None}

                    render_direct_recall_diff(
                        reference_eval['ground_truth'],
                        eval_result['llm_answer'],
                        title=f"Run #{run_idx}",
                        metrics=metrics_payload,
                    )
        
        # Interpretation
        st.markdown("#### 🔍 Interpretation")
        avg_rouge = agg_metrics.get('avg_rouge_score', 0)
        avg_jaccard = agg_metrics.get('avg_jaccard_index', 0)
        
        if avg_rouge > 0.5 or avg_jaccard > 0.5:
            st.error(
                "⚠️ **High Memorization Detected**: The LLM shows strong similarity to the ground truth answers, "
                "suggesting it may have memorized content from the document or similar sources."
            )
        elif avg_rouge > 0.3 or avg_jaccard > 0.3:
            st.warning(
                "⚠️ **Moderate Memorization**: The LLM shows some similarity to ground truth answers, "
                "which could indicate partial memorization or general knowledge overlap."
            )
        else:
            st.success(
                "✅ **Low Memorization**: The LLM's answers differ significantly from ground truth, "
                "suggesting it is not recalling memorized content from this specific document."
            )
    elif not st.session_state['qa_generated_qa_pairs']:
        st.info("👆 Upload a PDF or TXT file and generate Q/A pairs to begin the knowledge memorization detection process.")


def render_sc_detection(api_key, model_choice, provider):
    """Render Single-choice question test for copyright detection."""

    default_state = {
        'sc_source_mode': 'Input Text',
        'sc_generated_mcqs': [],
        'sc_document_text': '',
        'sc_input_text': '',
        'sc_dataset_document': None,
        'sc_num_questions': 5,
        'sc_gen_temperature': 0.4,
        'sc_gen_top_p': 0.85,
        'sc_gen_provider_index': 0,
        'sc_evaluation_results': None,
        'sc_eval_runs': 1,
        'sc_eval_temperature': 0.0,
        'sc_eval_top_p': 0.9,
    }
    for key, value in default_state.items():
        if key not in st.session_state:
            st.session_state[key] = value

    # Import pandas at the top of the function
    import pandas as pd

    st.markdown(
        """
        <div class="analysis-callout">
            <div class="analysis-callout__title">Single-choice Question Detection</div>
            <ul class="analysis-callout__list">
                <li>Provide source text through direct input or document upload.</li>
                <li>Extract text fragments from your content as correct answers.</li>
                <li>Use a generator LLM to create distractor options from the fragments.</li>
                <li>Evaluate your target LLM to see whether it consistently prefers the verbatim option.</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Step 1: Provide source content
    st.markdown('<p class="analysis-step-label">Step 1 · Provide source content</p>', unsafe_allow_html=True)
    
    # Create options for custom input or predefined examples
    custom_options = ["Input Text", "Upload Document", "Predefined Examples"]
    
    source_mode_display = st.radio(
        "Where should the text fragments come from?",
        custom_options,
        horizontal=True,
        key="sc_source_mode",
        help="Choose 'Input Text' or 'Upload Document' for custom input, or 'Predefined Examples' to use built-in evaluation datasets.",
    )
    
    # No need to remove suffix since we don't have datasets
    source_mode = source_mode_display

    uploaded_document = None
    excerpt_preview = ""
    excerpt_meta: Dict[str, Any] = {}

    if source_mode == "Input Text":
        st.markdown("**📝 Input your text**")
        st.text_area(
            "Enter your text",
            height=200,
            placeholder="Paste or type the text you want to use for generating single-choice questions...",
            help="Provide the text content you'd like to probe for memorization detection.",
            key="sc_input_text",
        )
        if st.session_state.get("sc_input_text", "").strip():
            excerpt_preview = st.session_state["sc_input_text"].strip()
            st.caption(f"Text length: {len(excerpt_preview)} characters · {len(excerpt_preview.split())} words")
    elif source_mode == "Upload Document":
        st.markdown("**📎 Upload your document**")
        uploaded_document = st.file_uploader(
            "Upload PDF or TXT",
            type=["pdf", "txt"],
            help="Provide the copyrighted material you'd like to probe.",
            key="sc_document_upload",
        )
    elif source_mode == "Predefined Examples":
        st.markdown("**📚 Select predefined evaluation dataset**")
        dataset_options = ["arXivTection", "BookTection"]
        
        # Put dataset selection and question indices on the same row
        col_dataset, col_indices = st.columns([1, 1])
        
        with col_dataset:
            selected_dataset = st.selectbox(
                "Choose evaluation dataset",
                dataset_options,
                help="Select a predefined dataset containing single-choice questions for copyright detection evaluation.",
                key="sc_dataset_selection",
            )
        
        with col_indices:
            question_indices = st.text_input(
                "Question indices",
                placeholder="e.g., 1,5,10-15,20",
                help="Enter question indices (comma-separated, ranges with hyphens). Leave empty to load all questions.",
                key="sc_question_indices",
            )
        # Show selected questions count
        if question_indices.strip():
            try:
                indices = parse_question_indices(question_indices.strip())
                st.caption(f"Selected {len(indices)} questions: {indices[:10]}{'...' if len(indices) > 10 else ''}")
            except ValueError as e:
                st.error(f"Invalid format: {e}")
        else:
            pass
        
        # Add accordion to display CSV content
        with st.expander("📊 Preview Dataset Content", expanded=False):
            try:
                import pandas as pd
                from pathlib import Path
                csv_path = Path("src/direct_recall/decop/data") / f"{selected_dataset}.csv"
                if csv_path.exists():
                    df = pd.read_csv(csv_path)
                    # Reset index to start from 1 for display
                    df.index = range(1, len(df) + 1)
                    st.caption(f"📊 Dataset contains {len(df)} questions (indices: 1-{len(df)})")
                    dataset_info = {
                        "arXivTection": "Academic paper excerpts (label=1: appeared in training, label=0: not seen)",
                        "BookTection": "Book excerpts (label=1: appeared in training, label=0: not seen)"
                    }
                    st.caption(f"📖 {dataset_info[selected_dataset]}")
                    st.dataframe(df, width='stretch')
                    st.caption(f"Total rows: {len(df)} | Columns: {', '.join(df.columns.tolist())}")
                else:
                    st.error(f"CSV file not found: {csv_path}")
            except Exception as e:
                st.error(f"Error loading CSV: {e}")
        
        # Load button for predefined examples
        load_examples = st.button(
            "📥 Load Selected Questions",
            key="sc_load_examples_button",
            width='stretch',
        )
        
        if load_examples:
            if not question_indices.strip():
                st.warning("⚠️ Please enter question indices before loading.")
                return
            
            try:
                from src.direct_recall.single_choice import load_predefined_examples
                indices_to_load = None
                if question_indices.strip():
                    try:
                        indices_to_load = parse_question_indices(question_indices.strip())
                    except ValueError as e:
                        st.error(f"Invalid question indices format: {e}")
                        indices_to_load = None
                
                generated_mcqs = load_predefined_examples(selected_dataset, indices_to_load)
                if generated_mcqs:
                    st.session_state['sc_generated_mcqs'] = generated_mcqs
                    st.session_state['sc_document_text'] = f"Predefined dataset: {selected_dataset}"
                    if indices_to_load:
                        st.session_state['sc_document_text'] += f" (questions: {indices_to_load})"
                    st.session_state['sc_evaluation_results'] = None
                    st.success(f"✅ Loaded {len(generated_mcqs)} predefined single-choice questions from {selected_dataset}.")
                else:
                    st.error(f"❌ No questions found for the specified indices in {selected_dataset}.")
            except Exception as exc:
                st.error(f"❌ Failed to load predefined examples: {exc}")

    # Step 2: Configure generation model and parameters (only for custom input)
    if source_mode in ["Input Text", "Upload Document"]:
        st.markdown('<p class="analysis-step-label">Step 2 · Configure text fragment extraction and distractor generation</p>', unsafe_allow_html=True)
        st.markdown(
            '<p class="analysis-step-caption">Extract text fragments and use a generator LLM to create distractor options.</p>',
            unsafe_allow_html=True,
        )

        col_provider, col_model, col_api = st.columns(3)
        provider_options = ["OpenAI", "OpenRouter", "Anthropic", "Google Gemini"]

        with col_provider:
            generation_provider = st.selectbox(
                "Generation provider",
                provider_options,
                index=min(st.session_state['sc_gen_provider_index'], len(provider_options) - 1),
                key="sc_gen_provider",
            )
            st.session_state['sc_gen_provider_index'] = provider_options.index(generation_provider)

        def _provider_models(provider_name: str) -> List[str]:
            if provider_name == "OpenAI":
                return [
                    "gpt-3.5-turbo",
                    "gpt-3.5-turbo-instruct",
                    "gpt-4o",
                    "gpt-4o-mini",
                ]
            if provider_name == "OpenRouter":
                return [
                    "moonshotai/kimi-k2:free",
                    "meta-llama/llama-3.1-405b-instruct:free",
                    "qwen/qwen3-235b-a22b:free",
                    "meta-llama/llama-3.3-70b-instruct:free",
                    "mistralai/mistral-small-24b-instruct-2501:free",
                    "qwen/qwen-2.5-72b-instruct:free",
                ]
            if provider_name == "Anthropic":
                return [
                    "claude-3-haiku-20240307",
                    "claude-3-sonnet-20240229",
                    "claude-3-opus-20240229",
                ]
            if provider_name == "Google Gemini":
                return ["gemini-1.5-flash", "gemini-1.5-pro"]
            return ["custom-model"]

        with col_model:
            generation_model = st.selectbox(
                "Generation model",
                _provider_models(generation_provider),
                key="sc_gen_model",
            )

        with col_api:
            generation_api_key = st.text_input(
                "Generation API key",
                type="password",
                help="Leave blank to reuse the sidebar API key.",
                key="sc_gen_api_key",
            )

        col_qty, col_dist, col_temp, col_top_p = st.columns(4)
        with col_qty:
            st.number_input(
                "Number of questions",
                min_value=1,
                max_value=20,
                step=1,
                key="sc_num_questions",
            )
        with col_dist:
            st.number_input(
                "Number of distractors",
                min_value=1,
                value=3,
                step=1,
                help="Number of incorrect options to generate for each question.",
                key="sc_num_distractors",
            )
        with col_temp:
            st.slider(
                "Generation temperature",
                min_value=0.0,
                max_value=1.2,
                step=0.05,
                key="sc_gen_temperature",
            )
        with col_top_p:
            st.slider(
                "Generation Top-P",
                min_value=0.0,
                max_value=1.0,
                step=0.05,
                key="sc_gen_top_p",
            )

        generate_questions = st.button(
            "🚀 Generate single-choice questions",
            key="sc_generate_mcq_button",
            width='stretch',
        )
    else:
        # For predefined examples, skip generation and go directly to evaluation
        generate_questions = False

    if generate_questions:
        effective_api_key = generation_api_key or api_key
        if not effective_api_key:
            st.error("⚠️ Provide an API key for the generation model or reuse the sidebar key.")
        else:
            # Calculate total operations for progress bar
            num_questions = st.session_state['sc_num_questions']
            num_distractors = st.session_state['sc_num_distractors']
            total_operations = num_questions * (num_distractors + 1)  # +1 for question creation
            
            progress_bar = st.progress(0.0)
            progress_text = st.empty()
            
            def update_generation_progress(current, total, question_num):
                pct = current / total if total else 0
                progress_bar.progress(pct)
                progress_text.text(
                    f"🔄 Generating question {question_num}/{num_questions} | "
                    f"Creating distractors... ({current}/{total})"
                )
            
            try:
                if source_mode == "Input Text":
                    input_text = st.session_state.get("sc_input_text", "").strip()
                    if not input_text:
                        st.warning("⚠️ Please enter some text first.")
                        generated_mcqs, document_text = [], ""
                    else:
                        document_text = input_text
                        generated_mcqs = generate_single_choice_questions_from_fragments(
                            document_text,
                            effective_api_key,
                            generation_model,
                            generation_provider,
                            num_questions=st.session_state['sc_num_questions'],
                            num_distractors=st.session_state['sc_num_distractors'],
                            temperature=st.session_state['sc_gen_temperature'],
                            top_p=st.session_state['sc_gen_top_p'],
                            progress_callback=update_generation_progress,
                        )
                elif source_mode == "Upload Document":
                    if not uploaded_document:
                        st.warning("⚠️ Upload a PDF/TXT document first.")
                        generated_mcqs, document_text = [], ""
                    else:
                        generated_mcqs, document_text = generate_single_choice_questions_from_document_fragments(
                            uploaded_document,
                            effective_api_key,
                            generation_model,
                            generation_provider,
                            num_questions=st.session_state['sc_num_questions'],
                            num_distractors=st.session_state['sc_num_distractors'],
                            temperature=st.session_state['sc_gen_temperature'],
                            top_p=st.session_state['sc_gen_top_p'],
                            progress_callback=update_generation_progress,
                        )
                else:
                    generated_mcqs, document_text = [], ""
                
                progress_bar.empty()
                progress_text.empty()
                
                if not generated_mcqs:
                    st.error("❌ Failed to generate single-choice questions. Try adjusting the model or prompt parameters.")
                else:
                    st.session_state['sc_generated_mcqs'] = generated_mcqs
                    st.session_state['sc_document_text'] = document_text
                    st.session_state['sc_evaluation_results'] = None
                    st.success(f"✅ Generated {len(generated_mcqs)} single-choice questions.")
                    
            except Exception as exc:
                progress_bar.empty()
                progress_text.empty()
                st.error(f"❌ Generation failed: {exc}")

    # Handle predefined examples - load them directly
    if source_mode == "Predefined Examples":
        pass  # Loading is now handled by the load button above

    if st.session_state['sc_generated_mcqs']:
        section_title = "🧩 Generated Single-choice Questions" if source_mode in ["Input Text", "Upload Document"] else "📚 Predefined Single-choice Questions"
        st.markdown(f'<h4 class="section-header sm">{section_title}</h4>', unsafe_allow_html=True)
        for idx, mcq in enumerate(st.session_state['sc_generated_mcqs'], start=1):
            question_title = mcq['question']
            if source_mode == "Predefined Examples":
                # For predefined examples, show more descriptive title
                question_title = f"Question {idx} ({mcq['question']})"
            
            with st.expander(question_title, expanded=False):
                st.markdown(f"**Question:** {mcq['question']}")
                for option in mcq['options']:
                    badge = "✅" if option['label'] == mcq['correct_option'] else ""
                    st.write(f"{option['label']}. {option['text']} {badge}")
                if mcq.get('explanation'):
                    st.caption(f"Rationale: {mcq['explanation']}")
                # Show label for predefined examples
                if source_mode == "Predefined Examples" and 'label' in mcq:
                    label_text = "Training data (appeared in training)" if mcq['label'] == 1 else "Non-training data (not seen during training)"
                    original_id = mcq.get('original_id', '')
                    st.caption(f"Label: {mcq['label']} - {label_text}" + (f" | Original ID: {original_id}" if original_id else ""))

    # Step 3: Evaluate with target model
    step_label = "Step 3" if source_mode in ["Input Text", "Upload Document"] else "Step 2"
    st.markdown(f'<p class="analysis-step-label">{step_label} · Evaluate target model</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="analysis-step-caption">Run the model configured in the sidebar and look for biased option selections.</p>',
        unsafe_allow_html=True,
    )

    eval_cols = st.columns(3)
    with eval_cols[0]:
        st.number_input(
            "Evaluation runs",
            min_value=1,
            step=1,
            key="sc_eval_runs",
        )
    with eval_cols[1]:
        st.slider(
            "Evaluation temperature",
            min_value=0.0,
            max_value=1.2,
            step=0.05,
            key="sc_eval_temperature",
        )
    with eval_cols[2]:
        st.slider(
            "Evaluation Top-P",
            min_value=0.0,
            max_value=1.0,
            step=0.05,
            key="sc_eval_top_p",
        )

    run_single_choice_eval = st.button(
        "🧪 Run Single-Choice Evaluation",
        key="sc_run_eval_button",
        width='stretch',
    )

    if run_single_choice_eval:
        if not st.session_state['sc_generated_mcqs']:
            st.warning("⚠️ Generate single-choice questions before running the evaluation.")
        elif not api_key or not api_key.strip():
            st.error(f"⚠️ Configure an API key for {provider} in the sidebar.")
        elif not model_choice:
            st.error("⚠️ Select a target model in the sidebar before running evaluation.")
        else:
            total_questions = len(st.session_state['sc_generated_mcqs'])
            total_items = total_questions * st.session_state['sc_eval_runs']
            progress_bar = st.progress(0.0)
            progress_text = st.empty()

            def update_progress(current, total, run_num, question_num, question_total):
                pct = current / total if total else 0
                progress_bar.progress(pct)
                progress_text.text(
                    f"🔄 Run {run_num}/{st.session_state['sc_eval_runs']} | "
                    f"Question {question_num}/{question_total} | {current}/{total} evaluations"
                )

            try:
                results = run_single_choice_evaluation(
                    st.session_state['sc_generated_mcqs'],
                    api_key,
                    model_choice,
                    provider,
                    num_runs=st.session_state['sc_eval_runs'],
                    temperature=st.session_state['sc_eval_temperature'],
                    top_p=st.session_state['sc_eval_top_p'],
                    progress_callback=update_progress,
                )
                progress_bar.empty()
                progress_text.empty()
                if not results:
                    st.error("❌ Evaluation returned no results. Please try again.")
                else:
                    st.session_state['sc_evaluation_results'] = results
                    st.success(f"✅ Completed {total_items} single-choice evaluations.")
            except Exception as exc:  # noqa: BLE001
                progress_bar.empty()
                progress_text.empty()
                st.error(f"❌ Evaluation failed: {exc}")

    if st.session_state['sc_evaluation_results']:
        results = st.session_state['sc_evaluation_results']
        metrics = summarize_single_choice_results(results)
        if metrics:
            st.markdown('<h4 class="section-header sm">📊 Evaluation summary</h4>', unsafe_allow_html=True)
            accuracy = metrics.get('overall_accuracy', 0)
            avg_conf = metrics.get('avg_correct_confidence')
            sc_metrics = [
                {
                    "label": "Runs",
                    "icon": "🔁",
                    "value": str(metrics.get('total_runs', 0)),
                    "description": "Evaluation passes",
                    "range": "",
                },
                {
                    "label": "Attempts",
                    "icon": "🧪",
                    "value": str(metrics.get('total_attempts', 0)),
                    "description": "Questions × runs",
                    "range": "",
                },
                {
                    "label": "Accuracy",
                    "icon": "🎯",
                    "value": f"{accuracy * 100:.1f}%",
                    "description": "Correct option rate",
                    "range": "",
                },
                {
                    "label": "Avg confidence (correct)",
                    "icon": "📈",
                    "value": (
                        f"{avg_conf * 100:.1f}%" if isinstance(avg_conf, (int, float)) else "—"
                    ),
                    "description": "Mean probability when right",
                    "range": "",
                },
            ]

    if st.session_state['sc_evaluation_results']:
        results = st.session_state['sc_evaluation_results']
        metrics = summarize_single_choice_results(results)
        if metrics:

            # Add analysis for predefined examples
            if source_mode == "Predefined Examples" and st.session_state.get('sc_generated_mcqs'):
                pass

            if accuracy >= 0.75:
                st.error(
                    "⚠️ **High memorization risk** — the model consistently prefers the verbatim option."
                )
            elif accuracy >= 0.5:
                st.warning(
                    "⚠️ **Moderate memorization** — the model shows a noticeable bias toward the correct option."
                )
            else:
                st.success(
                    "✅ **Low memorization signal** — selections look close to chance level."
                )

            per_question = metrics.get('per_question', [])
            if per_question:
                st.markdown("#### Question-level accuracy")
                per_question_df = pd.DataFrame(
                    [
                        {
                            "Question #": item['index'] + 1,
                            "Accuracy": f"{item['accuracy'] * 100:.1f}%",
                            "Attempts": item['attempts'],
                            "Question": item['question'][:120] + ('…' if len(item['question']) > 120 else ''),
                        }
                        for item in per_question
                    ]
                )
                st.dataframe(per_question_df, hide_index=True)

        st.markdown("#### Detailed responses")
        for question_idx, mcq in enumerate(st.session_state['sc_generated_mcqs'], start=1):
            with st.expander(f"Question {question_idx}: {textwrap.shorten(mcq['question'], width=80, placeholder='…')}"):
                st.markdown(f"**Question:** {mcq['question']}")
                for option in mcq['options']:
                    badge = "✅" if option['label'] == mcq['correct_option'] else ""
                    st.write(f"{option['label']}. {option['text']} {badge}")
                for run_idx, run_results in enumerate(results, start=1):
                    if question_idx - 1 < len(run_results):
                        eval_result = run_results[question_idx - 1]
                        status = "✅" if eval_result.get('is_correct') else "❌"
                        st.write(
                            f"Run {run_idx}: chose {eval_result.get('llm_choice', '?')} {status}"
                        )
                        if eval_result.get('raw_response'):
                            st.caption(f"Raw response: {eval_result['raw_response']}")
                        probs = eval_result.get('option_probabilities')
                        if isinstance(probs, dict):
                            ordered = []
                            for label in ["A", "B", "C", "D"]:
                                if label in probs:
                                    ordered.append(f"{label}: {probs[label] * 100:.1f}%")
                            leftovers = [
                                f"{label}: {value * 100:.1f}%"
                                for label, value in probs.items()
                                if label not in {"A", "B", "C", "D"}
                            ]
                            prob_line = ", ".join(ordered + leftovers)
                            st.caption(f"Option probabilities » {prob_line}")

                if st.session_state['sc_document_text']:
                    with st.expander("📄 Source excerpt", expanded=False):
                        st.write(st.session_state['sc_document_text'][:5000])


def render_sleek_attack_detection(api_key, model_choice, provider):
    """Render Step-by-step Leaking and Extraction knowledge memorization detection."""
    
    # Initialize session state for Step-by-step Leaking and Extraction
    if 'sleek_source_mode' not in st.session_state:
        st.session_state['sleek_source_mode'] = 'Input Text'
    if 'sleek_document_text' not in st.session_state:
        st.session_state['sleek_document_text'] = ''
    if 'sleek_input_text' not in st.session_state:
        st.session_state['sleek_input_text'] = ''
    if 'sleek_evaluation_results' not in st.session_state:
        st.session_state['sleek_evaluation_results'] = None
    if 'sleek_temperature' not in st.session_state:
        st.session_state['sleek_temperature'] = 0.7
    if 'sleek_top_p' not in st.session_state:
        st.session_state['sleek_top_p'] = 0.9
    
    st.markdown(
        """
        <div class="analysis-callout">
            <div class="analysis-callout__title">Step-by-step Leaking and Extraction Detection</div>
            <ul class="analysis-callout__list">
                <li>Provide source text through direct input or document upload.</li>
                <li>Generate auxiliary reasoning and extract targeted knowledge points.</li>
                <li>Create specific questions to probe residual knowledge in the target model.</li>
                <li>Execute the attack and assess knowledge leakage through systematic evaluation.</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    # Step 1: Provide source content
    st.markdown('<p class="analysis-step-label">Step 1 · Provide source content</p>', unsafe_allow_html=True)
    
    source_mode = st.radio(
        "Where should the knowledge come from?",
        ["Input Text", "Upload Document"],
        horizontal=True,
        key="sleek_source_mode",
        help="Choose 'Input Text' for custom input or 'Upload Document' for PDF/TXT files.",
    )
    
    document_text = ""
    
    if source_mode == "Input Text":
        st.markdown("**📝 Input your text**")
        st.text_area(
            "Enter your text",
            height=200,
            placeholder="Paste or type the text you want to test for knowledge leakage...",
            help="Provide the text content you'd like to probe for residual knowledge.",
            key="sleek_input_text",
        )
        if st.session_state.get("sleek_input_text", "").strip():
            document_text = st.session_state["sleek_input_text"].strip()
            st.caption(f"Text length: {len(document_text)} characters · {len(document_text.split())} words")
    else:  # Upload Document
        st.markdown("**📎 Upload your document**")
        uploaded_file = st.file_uploader(
            "Choose a pdf or txt file",
            type=["pdf", "txt"],
            help="Select a PDF or UTF-8 TXT document to extract knowledge from",
            key="sleek_document_upload"
        )
        if uploaded_file:
            try:
                from src.direct_recall.pdf_utils import extract_text_from_document
                document_text = extract_text_from_document(uploaded_file)
                if isinstance(document_text, str) and document_text.startswith("Error"):
                    st.error(f"❌ {document_text}")
                    document_text = ""
                else:
                    st.caption(f"Document length: {len(document_text)} characters · {len(document_text.split())} words")
            except Exception as e:
                st.error(f"❌ Failed to extract text from document: {e}")
                document_text = ""
    
    # Step 2: Configure evaluation parameters
    st.markdown('<p class="analysis-step-label">Step 2 · Configure evaluation parameters</p>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.slider(
            "Temperature",
            min_value=0.0,
            max_value=1.2,
            value=st.session_state['sleek_temperature'],
            step=0.05,
            help="Controls randomness in LLM responses. Lower values = more deterministic.",
            key="sleek_temperature_slider"
        )
    
    with col2:
        st.slider(
            "Top-P",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state['sleek_top_p'],
            step=0.05,
            help="Nucleus sampling parameter for controlling diversity.",
            key="sleek_top_p_slider"
        )
    
    # Run evaluation button
    run_sleek = st.button(
        "🔍 Run: Step-by-step Leaking and Extraction Evaluation",
        key="run_sleek_button",
        type="primary",
        width='stretch'
    )
    
    if run_sleek:
        temperature = st.session_state.get('sleek_temperature_slider', 0.7)
        top_p = st.session_state.get('sleek_top_p_slider', 0.9)
        
        if not document_text:
            st.warning("⚠️ Please provide source text first.")
        elif not api_key or not api_key.strip():
            st.error(f"⚠️ Please configure the API key for **{provider}** in the sidebar before running evaluation.")
        elif not model_choice:
            st.error("⚠️ Please select a model in the sidebar before running evaluation.")
        else:
            try:
                from src.direct_recall.sleek_attack import run_sleek_evaluation
                
                with st.spinner("🔄 Running Step-by-step Leaking and Extraction evaluation... This may take several minutes."):
                    results = run_sleek_evaluation(
                        document_text=document_text,
                        api_key=api_key,
                        model_name=model_choice,
                        provider=provider,
                        temperature=temperature,
                        top_p=top_p
                    )
                
                st.session_state['sleek_evaluation_results'] = results
                st.success("✅ Step-by-step Leaking and Extraction evaluation completed!")
                
            except Exception as e:
                st.error(f"❌ Step-by-step Leaking and Extraction evaluation failed: {str(e)}")
                st.session_state['sleek_evaluation_results'] = None
    
    # Display results
    if st.session_state.get('sleek_evaluation_results'):
        results = st.session_state['sleek_evaluation_results']
        
        # Summary metrics
        st.markdown("#### 📊 Evaluation Summary")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Questions", results.get('total_questions', 0))
        with col2:
            st.metric("Questions with Leakage", results.get('questions_with_leakage', 0))
        with col3:
            leakage_rate = results.get('leakage_rate', 0)
            st.metric("Leakage Rate", f"{leakage_rate:.1%}")
        
        # Summary info
        summary = results.get('summary', {})
        st.caption(f"Model: {summary.get('model', 'N/A')} | Provider: {summary.get('provider', 'N/A')} | Document Length: {summary.get('document_length', 0)} chars")
        
        # Detailed results by question
        st.markdown("#### 📝 Detailed Results by Question")
        
        questions = results.get('questions', [])
        if questions:
            for i, question_data in enumerate(questions, 1):
                with st.expander(f"Question {i}: {question_data.get('question', 'N/A')[:80]}...", expanded=(i == 1)):
                    st.markdown(f"**Category:** {question_data.get('category', 'N/A')}")
                    st.markdown(f"**Knowledge Point:** {question_data.get('knowledge_point', 'N/A')}")
                    
                    response = question_data.get('response', '')
                    if response:
                        st.markdown("**Model Response:**")
                        st.write(response)
                    
                    leakage_score = question_data.get('leakage_score', 0)
                    has_leakage = question_data.get('has_leakage', False)
                    
                    if has_leakage:
                        st.error(f"⚠️ **Leakage Detected** (Score: {leakage_score:.2f})")
                    else:
                        st.success(f"✅ **No Leakage** (Score: {leakage_score:.2f})")
        else:
            st.info("No questions were generated during the evaluation.")
        
        # Category breakdown analysis
        category_breakdown = results.get('category_breakdown', {})
        if category_breakdown:
            st.markdown("#### 📊 Category Analysis")
            st.caption("Breakdown of leakage detection by question category")
            
            categories = []
            total_questions = []
            leaked_questions = []
            leakage_rates = []
            
            for cat, stats in category_breakdown.items():
                categories.append(cat)
                total_questions.append(stats.get('total', 0))
                leaked_questions.append(stats.get('leaked', 0))
                rate = stats.get('leaked', 0) / stats.get('total', 1) * 100
                leakage_rates.append(f"{rate:.1f}%")
            
            # Create a summary table
            import pandas as pd
            category_df = pd.DataFrame({
                'Category': categories,
                'Total Questions': total_questions,
                'Leaked Questions': leaked_questions,
                'Leakage Rate': leakage_rates
            })
            st.dataframe(category_df, width='stretch')
            
            # Interpretation by category
            st.markdown("**Category Interpretation:**")
            for cat in categories:
                stats = category_breakdown[cat]
                total = stats.get('total', 0)
                leaked = stats.get('leaked', 0)
                rate = leaked / total if total > 0 else 0
                
                if cat == "Direct":
                    if rate > 0.5:
                        st.error(f"• **Direct Questions**: High leakage ({rate:.1%}) - Model directly reveals forgotten knowledge")
                    elif rate > 0.2:
                        st.warning(f"• **Direct Questions**: Moderate leakage ({rate:.1%}) - Some direct knowledge exposure")
                    else:
                        st.success(f"• **Direct Questions**: Low leakage ({rate:.1%}) - Effective forgetting of direct facts")
                        
                elif cat == "Indirect":
                    if rate > 0.4:
                        st.error(f"• **Indirect Questions**: High leakage ({rate:.1%}) - Model shows associative knowledge retention")
                    elif rate > 0.2:
                        st.warning(f"• **Indirect Questions**: Moderate leakage ({rate:.1%}) - Some associative connections remain")
                    else:
                        st.success(f"• **Indirect Questions**: Low leakage ({rate:.1%}) - Limited associative knowledge")
                        
                elif cat == "Implied":
                    if rate > 0.6:
                        st.error(f"• **Implied Questions**: High leakage ({rate:.1%}) - Model can infer forgotten knowledge")
                    elif rate > 0.3:
                        st.warning(f"• **Implied Questions**: Moderate leakage ({rate:.1%}) - Some logical inference capability")
                    else:
                        st.success(f"• **Implied Questions**: Low leakage ({rate:.1%}) - Limited inference from remaining knowledge")
                        
                elif cat == "Irrelevant":
                    if rate > 0.1:
                        st.warning(f"• **Irrelevant Questions**: Unexpected leakage ({rate:.1%}) - Control questions show false positives")
                    else:
                        st.success(f"• **Irrelevant Questions**: Clean control ({rate:.1%}) - No false positive detections")
        
        # Interpretation
        st.markdown("#### 🔍 Overall Interpretation")
        leakage_rate = results.get('leakage_rate', 0)
        
        if leakage_rate > 0.5:
            st.error(
                "⚠️ **High Knowledge Leakage Detected**: The model shows significant residual knowledge "
                "about the source content, suggesting incomplete unlearning or memorization."
            )
        elif leakage_rate > 0.2:
            st.warning(
                "⚠️ **Moderate Knowledge Leakage**: The model shows some residual knowledge "
                "that may indicate partial memorization of the content."
            )
        else:
            st.success(
                "✅ **Low Knowledge Leakage**: The model appears to have limited or no residual knowledge "
                "about the specific content provided."
            )


def render_legal_case_display_page():
    """Showcase real-world lawsuits that underscore memorization risk."""

    st.markdown("### ⚖️ Legal Cases Display")
    st.caption(
        "Curated legal milestones that illustrate why Copyright Detective workflows are essential."
    )

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

    if show_page_header:
        # Page header with clear cache button
        header_col, button_col = st.columns([4, 1])
        with header_col:
            st.markdown('<h4 class="section-header">📄 Document Memorization Detection</h4>', unsafe_allow_html=True)
            st.markdown(
                "Upload a full PDF or TXT document to automatically analyze text chunks for potential copyright infringement."
            )
        with button_col:
            if st.button("🗑️ Clear Cache", key="clear_pdf_cache", help="Remove cached PDF analysis results"):
                st.session_state.pop("pdf_analysis_results", None)
                st.session_state.pop("pdf_analysis_score_type", None)
                st.session_state.pop("pdf_analysis_top_k", None)
                st.session_state['pdf_chunk_size'] = 200
                st.session_state['pdf_continuation_method_index'] = 0
                st.session_state['pdf_temperature'] = 0.7
                st.session_state['pdf_top_p'] = 0.9
                st.session_state['pdf_custom_prompt_text'] = ""
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
        """Render ranked document chunk results with adjustable controls."""

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
            levenshtein = metrics_for_display.get("levenshtein", None)
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
                st.markdown("**🧠 Recall Overlap**")
                render_direct_recall_diff(
                    lower,
                    gen,
                    title="Ground Truth vs. Generated Output",
                    metrics=metrics_for_display,
                )

    uploaded_file = st.file_uploader(
        "Choose a pdf or txt file",
        type=["pdf", "txt"],
        help="Select a PDF or UTF-8 TXT document to analyze"
    )

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
            'Change Chunk Size (words):',
            min_value=50,
            max_value=2000,
            value=st.session_state['pdf_chunk_size'],
            step=25,
            help='Number of words per text chunk',
            key='pdf_chunk_size_input'
        )
        st.session_state['pdf_chunk_size'] = chunk_size
        st.caption("Chunk size must be at least 50 words to run document analysis.")
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
    with ctrl_col1:
        st.slider(
            'Temperature',
            min_value=0.0,
            max_value=1.2,
            value=st.session_state['pdf_temperature'],
            step=0.01,
            help='Controls randomness. Lower values make the model more deterministic.',
            key='pdf_temperature_slider'
        )
    with ctrl_col2:
        st.slider(
            'Top-P',
            min_value=0.0,
            max_value=1.0,
            value=st.session_state['pdf_top_p'],
            step=0.01,
            help='Controls nucleus sampling diversity. 0.5 considers the top 50% probability mass.',
            key='pdf_top_p_slider'
        )

    analyze_document = st.button(
        "🔍 Run: Document Memorization Detection",
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

    if analyze_document:
        # Get values from session state
        temperature = st.session_state.get('pdf_temperature_slider', 0.7)
        top_p = st.session_state.get('pdf_top_p_slider', 0.9)
        
        # Set default values for ranking parameters
        if score_type is None:
            score_type = "ROUGE-L"
        if top_k is None:
            top_k = 5
            
        if not api_key:
            st.error("⚠️ Please enter your API key in the sidebar.")
            return
        if uploaded_file is None:
            st.error("⚠️ Please upload a document before running the analysis.")
            return
        custom_template = None
        if continuation_method == "Custom Prompt":
            custom_template = (custom_pdf_prompt or "").strip()
            if not custom_template:
                st.error("⚠️ Please provide a custom prompt template before running the analysis.")
                return

        try:
            progress_bar = st.progress(0, text=f"🔄 Analyzing document with {model_choice}...")
            document_text = extract_text_from_document(uploaded_file)
            if isinstance(document_text, str) and document_text.startswith("Error"):
                st.error(f"❌ {document_text}")
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
        cached_top_p = st.session_state.get("pdf_analysis_top_p", 0.9)

        render_pdf_results_section(
            cached_results,
            default_score_type=cached_score_type,
            default_top_k=cached_top_k,
            continuation_method=cached_continuation_method,
            temperature=cached_temperature,
            top_p=cached_top_p,
        )


def render_adversarial_persuasion_page(api_key, model_choice, provider):
    """Render the persuasive jailbreak detection test workspace."""
    
    # Initialize session state for Adversarial Persuasion
    if 'adv_stage1_input_prompt' not in st.session_state:
        st.session_state['adv_stage1_input_prompt'] = ""
    if 'adv_reference_excerpt' not in st.session_state:
        st.session_state['adv_reference_excerpt'] = DEFAULT_HP_REFERENCE_EXCERPT
    if 'adv_stage1_attempts' not in st.session_state:
        st.session_state['adv_stage1_attempts'] = 3
    if 'adv_stage1_temperature' not in st.session_state:
        st.session_state['adv_stage1_temperature'] = 0.7
    if 'adv_stage1_top_p' not in st.session_state:
        st.session_state['adv_stage1_top_p'] = 0.9
    if 'adv_stage2_attempts' not in st.session_state:
        st.session_state['adv_stage2_attempts'] = 5
    if 'adv_stage2_temperature' not in st.session_state:
        st.session_state['adv_stage2_temperature'] = 0.8
    if 'adv_stage2_top_p' not in st.session_state:
        st.session_state['adv_stage2_top_p'] = 0.95

    # Get available strategies and baseline prompts
    strategies = list_persuasion_strategies()
    baseline_prompts = list_baseline_prompts()

    # Page header with clear cache button
    header_col, button_col = st.columns([4, 1])
    with header_col:
        st.markdown('<h4 class="section-header">🔓 Persuasive Jailbreak Detection Test</h4>', unsafe_allow_html=True)
        st.markdown(
            "An evaluation framework that uses persuasion techniques to assess copyright infringement risks in LLMs."
        )
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

    st.markdown(
        """
        <div class=\"analysis-callout\">
            <div class=\"analysis-callout__title\">How the Persuasive Jailbreak Detection Test works</div>
            <ul class=\"analysis-callout__list\">
                <li><strong>One-shot mutation</strong> — Generate baseline adversarial prompt variations without examples.</li>
                <li><strong>Few-shot refinement</strong> — Use predefined examples from few-shot.json to guide stronger mutations.</li>
                <li><strong>Review intention judging</strong> — Inspect stored results, confirm intent preservation, and iterate on findings.</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Get mutation store for accessing results
    mutation_store = st.session_state.setdefault("generated_persuasion_mutations", {})
    stage1_reference_map = st.session_state.setdefault("stage1_reference_texts", {})
    
    # ========== Unified Generation ==========
    header_col, spacer_col, button_col = st.columns([4, 1, 1])
    with header_col:
        st.markdown('<p class="analysis-step-label">Adversarial Prompt Generation</p>', unsafe_allow_html=True)
    with spacer_col:
        st.write("")
    with button_col:
        st.write("")  # Empty column for alignment

    st.markdown("**📝 Prompt setup**")

    if baseline_prompts:
        no_preset_option = "Custom Input"

        # Create display options with "Example:" prefix and mapping to clean prompts
        display_options = [no_preset_option]
        prompt_mapping = {}
        
        for prompt in baseline_prompts:
            if prompt.startswith("Example: "):
                display_text = prompt  # Keep "Example:" for display
                clean_prompt = prompt[9:].strip()  # Remove "Example: " prefix
            else:
                display_text = f"Example: {prompt}"  # Add prefix if not present
                clean_prompt = prompt
            display_options.append(display_text)
            prompt_mapping[display_text] = clean_prompt

        def _apply_preset() -> None:
            selected = st.session_state.get("baseline_selector")
            if selected and selected != no_preset_option:
                clean_prompt = prompt_mapping.get(selected, selected)
                st.session_state["input_prompt"] = clean_prompt
                # Set reference text based on the selected prompt
                if "Hobbit" in selected:
                    st.session_state["reference"] = DEFAULT_HB_REFERENCE_EXCERPT
                elif "Thrones" in selected:
                    st.session_state["reference"] = DEFAULT_GA_REFERENCE_EXCERPT
                else:
                    st.session_state["reference"] = DEFAULT_HP_REFERENCE_EXCERPT
            else:
                st.session_state["input_prompt"] = ""
                st.session_state["reference"] = ""

        st.selectbox(
            "Choose an adversarial prompt type",
            display_options,
            key="baseline_selector",
            on_change=_apply_preset,
            help="These presets mirror the baseline requests from the paper's extraction study.",
        )

    input_prompt = st.text_area(
        "Original adversarial prompt",
        value=st.session_state.get('adv_input_prompt', ""),
        height=150,
        placeholder="e.g., Reproduce verbatim the first 100 words of Harry Potter and the Sorcerer's Stone.",
        key="input_prompt",
        help="Enter the baseline prompt to mutate. This will be transformed using persuasion strategies.",
    )

    st.markdown("**⚙️ Sampling & evaluation parameters**")
    
    col_mode, col_strategies, col_attempts_strategy, col_attempts_prompt = st.columns([1, 2, 1, 1])
    with col_mode:
        generation_mode = st.selectbox(
            "Choose zero-shot/few-shot",
            ["One-Shot", "Few-Shot"],
            index=0,
            key="generation_mode",
            help="Select generation mode: One-Shot (zero-shot) or Few-Shot.",
        )
        generation_modes = [generation_mode]  # Convert to list for compatibility
    
    with col_strategies:
        selected_strategies = st.multiselect(
            "Persuasion strategies",
            strategies,
            default=[],
            key="strategies",
            help="Select one or more persuasion strategies to apply.",
        )
    
    with col_attempts_strategy:
        attempts = st.number_input(
            "Attempts per strategy",
            min_value=1,
            max_value=20,
            value=st.session_state.get('adv_attempts', 3),
            step=1,
            key="attempts",
            help="Number of mutation attempts for each strategy (more attempts = broader exploration).",
        )
    
    with col_attempts_prompt:
        attempts_per_prompt = st.number_input(
            "Attempts per mutated prompt",
            min_value=1,
            max_value=10,
            value=st.session_state.get('adv_attempts_per_prompt', 1),
            step=1,
            key="attempts_per_prompt",
            help="Number of generation attempts for each mutated prompt.",
        )

    st.text_area(
        "Reference text",
        value=st.session_state.get('adv_reference_excerpt', DEFAULT_HP_REFERENCE_EXCERPT),
        height=150,
        key="reference",
        help="Ground-truth copyrighted text. ROUGE-L measures how well mutations induce the LLM to reproduce this content.",
    )

    # Prompt Preview Accordion
    with render_streamlit_accordion(
        "👀 Prompt Preview",
        key="prompt_preview",
        expanded=False,
    ):
        input_prompt = st.session_state.get('input_prompt', '')
        generation_mode = st.session_state.get('generation_mode', 'One-Shot')
        generation_modes = [generation_mode]  # Convert to list for compatibility
        selected_strategies = st.session_state.get('strategies', [])
        attempts = st.session_state.get('attempts', 3)
        attempts_per_prompt = st.session_state.get('attempts_per_prompt', 1)
        reference_text = st.session_state.get('reference', '')
        
        st.markdown("**📋 Generation Configuration Summary:**")
        st.markdown(f"- **Mode:** {generation_mode}")
        st.markdown(f"- **Strategies:** {', '.join(selected_strategies) if selected_strategies else 'None selected'}")
        st.markdown(f"- **Attempts per strategy:** {attempts}")
        st.markdown(f"- **Attempts per mutated prompt:** {attempts_per_prompt}")
        st.markdown(f"- **Total mutations:** {len(selected_strategies) * attempts if selected_strategies else 0}")
        st.markdown(f"- **Total generations:** {len(selected_strategies) * attempts * attempts_per_prompt if selected_strategies else 0}")
        
        st.markdown("**📝 Original Prompt:**")
        if input_prompt.strip():
            st.text_area(
                "Original adversarial prompt",
                value=input_prompt,
                height=100,
                disabled=True,
                key="preview_input_prompt",
            )
        else:
            st.info("⚠️ No prompt entered yet.")
        
        # Show preview for each selected strategy and mode
        if selected_strategies and input_prompt.strip() and generation_modes:
            st.markdown("**🎯 Strategy-Specific Prompts Preview:**")
            
            # Load few-shot data once for all previews
            few_shot_data = {}
            try:
                import json
                from pathlib import Path
                few_shot_path = Path(__file__).parent / "adversarial_persuasion_detection" / "few-shot.json"
                with open(few_shot_path) as f:
                    few_shot_data = json.load(f)
            except Exception as e:
                st.warning(f"⚠️ Could not load few-shot data: {e}")
            
            for strategy in selected_strategies:
                for mode in generation_modes:
                    with st.expander(f"🔧 {strategy} ({mode})", expanded=False):
                        try:
                            # Import the persuasion strategies
                            from src.adversarial_persuasion_detection.adversarial_prompting import PERSUASIVE_MUTATION_TEMPLATES
                            
                            if mode == "One-Shot":
                                # Show one-shot prompt preview
                                if strategy in few_shot_data:
                                    template = few_shot_data[strategy]
                                    
                                    # Extract the prefix before examples for one-shot
                                    prefix = template.split("\n\nNow, I will provide")[0]
                                    one_shot_prompt = prefix + "\n\nOriginal Adversarial Prompt - " + input_prompt.strip()
                                    
                                    st.markdown("**One-Shot Prompt Preview:**")
                                    st.text_area(
                                        f"One-shot prompt for {strategy}",
                                        value=one_shot_prompt,
                                        height=300,
                                        disabled=True,
                                        key=f"preview_{strategy}_{mode.lower().replace('-', '_')}",
                                    )
                                else:
                                    st.warning(f"⚠️ Template not found for strategy: {strategy}")
                            
                            elif mode == "Few-Shot":
                                # Show complete few-shot prompt preview
                                try:
                                    if strategy in few_shot_data:
                                        template = few_shot_data[strategy]
                                        
                                        # Extract original examples from the #Example section for few-shot formatting
                                        examples = []
                                        start = template.find("#Example\n")
                                        if start != -1:
                                            start += len("#Example\n")
                                            end = template.find("\n\nNow, I will provide", start)
                                            if end == -1:
                                                end = len(template)
                                            example_block = template[start:end].strip()
                                            
                                            # Extract original and mutated prompts from the example
                                            original_start = example_block.find("Original Adversarial Prompt: ")
                                            original = ""
                                            if original_start != -1:
                                                original_start += len("Original Adversarial Prompt: ")
                                                original_end = example_block.find("\nThe Core", original_start)
                                                if original_end == -1:
                                                    original_end = example_block.find("\nMutated", original_start)
                                                if original_end == -1:
                                                    original_end = len(example_block)
                                                original = example_block[original_start:original_end].strip()
                                            
                                            mutated_start = example_block.find("Mutated Adversarial Prompt (with the same harmful intention):")
                                            mutated = ""
                                            if mutated_start != -1:
                                                mutated_start += len("Mutated Adversarial Prompt (with the same harmful intention):")
                                                mutated = example_block[mutated_start:].strip()
                                            
                                            if original and mutated:
                                                examples = [original, mutated] * 5
                                        
                                        # Format the complete few-shot prompt using the original example
                                        if examples:
                                            complete_fewshot_prompt = template % tuple(examples + [original])
                                        else:
                                            # Fallback if extraction failed
                                            prompt_stripped = input_prompt.strip()
                                            complete_fewshot_prompt = template % (
                                                prompt_stripped, prompt_stripped,
                                                prompt_stripped, prompt_stripped,
                                                prompt_stripped, prompt_stripped,
                                                prompt_stripped, prompt_stripped,
                                                prompt_stripped, prompt_stripped,
                                                prompt_stripped,
                                            )
                                        
                                        st.markdown("**Few-Shot Complete Prompt Preview:**")
                                        st.text_area(
                                            f"Complete few-shot prompt for {strategy}",
                                            value=complete_fewshot_prompt,
                                            height=400,
                                            disabled=True,
                                            key=f"preview_{strategy}_{mode.lower().replace('-', '_')}_complete",
                                        )
                                        
                                    else:
                                        st.warning(f"⚠️ Few-shot data not found for strategy: {strategy}")
                                
                                except Exception as e:
                                    st.warning(f"⚠️ Could not load few-shot preview for {strategy}: {e}")
                        
                        except Exception as e:
                            st.error(f"❌ Error loading preview for {strategy}: {e}")
        elif not selected_strategies:
            st.info("⚠️ No strategies selected yet.")
        elif not generation_modes:
            st.info("⚠️ No generation modes selected yet.")
        elif not input_prompt.strip():
            st.info("⚠️ Enter a prompt above to see strategy previews.")
        
        st.markdown("**📄 Reference Text (truncated):**")
        if reference_text.strip():
            truncated_ref = textwrap.shorten(reference_text, width=200, placeholder="...")
            st.text_area(
                "Reference excerpt",
                value=truncated_ref,
                height=80,
                disabled=True,
                key="preview_reference",
            )
        else:
            st.info("⚠️ No reference text entered yet.")

    with render_streamlit_accordion(
        "📋 Generation checklist",
        key="generation_checklist",
        expanded=False,
    ):
        st.markdown(
            """
            1. <strong>Generate</strong> – Apply persuasion strategies to create mutated prompts.
            2. <strong>Evaluate</strong> – Send each mutation to the LLM and collect its response.
            3. <strong>Rank</strong> – Score responses against the reference excerpt (ROUGE-L, Jaccard, Levenshtein).
            4. <strong>Judge</strong> – Assess whether each mutation preserves the original intention.
            """,
            unsafe_allow_html=True,
        )

    run_generation = st.button(
        "🚀 Generate & Evaluate",
        key="run_generation",
        type="primary",
        width='stretch'
    )
    
    if run_generation:
        # Clear previous results to only store current run data
        mutation_store.clear()
        
        # Get values from session state
        original_prompt = st.session_state.get('input_prompt', '')
        reference_text = st.session_state.get('reference', '')
        generation_mode = st.session_state.get('generation_mode', 'One-Shot')
        selected_strategies = st.session_state.get('strategies', [])
        attempts = st.session_state.get('attempts', 3)
        attempts_per_prompt = st.session_state.get('attempts_per_prompt', 1)
        generation_modes = [generation_mode]  # Convert to list for compatibility
        
        # Validation
        if not original_prompt.strip():
            st.warning("⚠️ Please enter an adversarial prompt.")
        elif not selected_strategies:
            st.warning("⚠️ Select at least one persuasion strategy.")
        elif not generation_mode:
            st.warning("⚠️ Select a generation mode.")
        elif not reference_text.strip():
            st.warning("⚠️ Please provide reference text for evaluation.")
        elif not api_key or not model_choice:
            st.error("⚠️ Enter your API key and choose a model in the sidebar.")
        else:
            original_prompt = original_prompt.strip()
            reference_text = reference_text.strip()

            if reference_text:
                stage1_reference_map[original_prompt] = reference_text
            
            # Display processing header
            st.markdown(f"**Processing:** {textwrap.shorten(original_prompt, width=120, placeholder='…')}")
            st.caption(f"📊 {generation_mode} × {len(selected_strategies)} strategy(ies) × {attempts} attempt(s) = {len(selected_strategies) * attempts} mutations")
            
            successful_count = 0
            
            # ===== Generate Mutations =====
            st.markdown("**🔄 Generating mutations**")
            st.caption(f"Generating {generation_mode} × {len(selected_strategies)} strategy(ies) × {attempts} attempt(s) = {len(selected_strategies) * attempts} total mutations")
            
            generation_progress = st.progress(0.0)
            
            all_evaluations = []
            
            # Process the single generation mode
            mode_idx = 0
            generation_mode = generation_modes[0]  # Always one element
            
            # Load few-shot examples if needed
            few_shot_examples = None
            if generation_mode == "Few-Shot":
                import json
                from pathlib import Path
                few_shot_path = Path(__file__).parent / "adversarial_persuasion_detection" / "few-shot.json"
                try:
                    with open(few_shot_path) as f:
                        few_shot_data = json.load(f)
                    
                    few_shot_examples = []
                    for strategy in selected_strategies:
                        if strategy in few_shot_data:
                            template = few_shot_data[strategy]
                            start = template.find("Mutated Adversarial Prompt (with the same harmful intention):")
                            if start != -1:
                                start += len("Mutated Adversarial Prompt (with the same harmful intention):")
                                end = template.find("\n\nNow, I will provide", start)
                                if end == -1:
                                    end = len(template)
                                example = template[start:end].strip()
                                if example:
                                    few_shot_examples.append(example)
                except Exception as e:
                    st.warning(f"⚠️ Could not load few-shot examples: {e}")
            
            # Generate mutations for each strategy individually to show progress
            progress_placeholder = st.empty()
            total_mutations = len(selected_strategies) * attempts
            cumulative = 0
            for strategy_idx, strategy in enumerate(selected_strategies, 1):
                # Update progress display
                progress_text = f"**🔄 Generating mutations ({cumulative + 1}-{cumulative + attempts}/{total_mutations}): {strategy}**"
                progress_placeholder.markdown(progress_text)
                generation_progress.progress(strategy_idx / len(selected_strategies))
                
                evaluations = mutate_strategies(
                    api_key,
                    model_choice,
                    provider,
                    [strategy],  # Process one strategy at a time
                    original_prompt,
                    reference_text=None,  # Don't calculate ROUGE during generation
                    few_shot_examples=few_shot_examples,  # Pass examples if Few-Shot
                    attempts_per_strategy=attempts,
                    attempts_per_prompt=attempts_per_prompt,
                    temperature=1.0,  # Higher temperature for diverse mutation generation
                    top_p=1.0,
                    dry_run=False,
                )
                
                # Add mode information to evaluations
                for i, evaluation in enumerate(evaluations):
                    if evaluation and evaluation.mutation:
                        evaluations[i] = MutationEvaluation(
                            mutation=evaluation.mutation,
                            parsed=evaluation.parsed,
                            metrics=evaluation.metrics,
                            attempt=evaluation.attempt,
                            mode=generation_mode,
                        )
                
                all_evaluations.extend(evaluations)
                cumulative += attempts
            
            generation_progress.progress(1.0)
            generation_progress.empty()
            progress_placeholder.empty()
            
            if not all_evaluations:
                st.error("❌ No mutations produced. Check your API key and model settings.")
            else:
                
                # ===== Evaluate Mutations =====
                st.markdown("**🔄 Evaluating mutations against reference text**")
                st.caption("Sending each mutation to the LLM and calculating ROUGE-L with reference output...")
                
                evaluated_mutations = []
                progress_bar = st.progress(0.0)
                progress_text = st.empty()
                
                total_evaluations = len(all_evaluations) * attempts_per_prompt
                eval_count = 0
                
                for eval_idx, evaluation in enumerate(all_evaluations):
                    if evaluation is None or evaluation.mutation.error:
                        continue
                    
                    parsed = evaluation.parsed
                    if not parsed or not parsed.mutated_text:
                        continue
                    
                    mutated_text = parsed.mutated_text.strip()
                    
                    # Send mutated prompt to LLM multiple times to get responses
                    for prompt_attempt in range(1, attempts_per_prompt + 1):
                        progress_text.text(f"Evaluating mutation {eval_count + 1}/{total_evaluations}")
                        progress_bar.progress((eval_count + 1) / total_evaluations)
                        
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
                            rouge_score = calculate_rouge_score(llm_response, reference_text.strip())
                            jaccard = calculate_jaccard_index(llm_response, reference_text.strip())
                            levenshtein = distance(llm_response, reference_text.strip())
                            
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
                                mode=evaluation.mode,
                            )
                            
                            evaluated_mutations.append({
                                'evaluation': updated_evaluation,
                                'llm_response': llm_response,
                                'rouge_l': rouge_score,
                                'jaccard': jaccard,
                                'levenshtein': levenshtein,
                                'prompt_attempt': prompt_attempt,
                            })
                            
                        except Exception as e:
                            st.warning(f"⚠️ Failed to evaluate mutation {eval_idx + 1}, attempt {prompt_attempt}: {e}")
                            continue
                        
                        eval_count += 1
                
                progress_bar.empty()
                progress_text.empty()
                
                if not evaluated_mutations:
                    st.error("❌ No mutations were successfully evaluated.")
                else:
                    # ===== Rank & Store Results =====
                    st.markdown("**🔄 Ranking mutations by ROUGE-L score**")
                    
                    # Sort by ROUGE-L score (descending)
                    evaluated_mutations.sort(
                        key=lambda x: x["evaluation"].metrics.rouge_l if x["evaluation"].metrics else 0,
                        reverse=True
                    )
                    
                    # Store results in mutation_store
                    for eval_item in evaluated_mutations:
                        evaluation = eval_item["evaluation"]
                        llm_response = eval_item["llm_response"]
                        
                        # Determine config_type from the evaluation's mode
                        config_type = "one" if evaluation.mode == "One-Shot" else "few"
                        
                        record_entries = mutation_store.setdefault(original_prompt, [])
                        
                        mutation_entry = MutationWithJudge(
                            evaluation=evaluation,
                            judge=None,
                            judge_passed=None,
                        )
                        serialised_entry = serialise_mutation_with_judge(mutation_entry)
                        
                        # Check for duplicates - now include prompt_attempt
                        mutated_text = evaluation.parsed.mutated_text.strip()
                        prompt_attempt = eval_item.get("prompt_attempt", 1)
                        entry_exists = False
                        for stored in record_entries:
                            stored_config = stored.get("config") or []
                            if stored_config and stored_config[0] == config_type:
                                stored_data = stored.get("data") or {}
                                stored_eval = stored_data.get("evaluation") or {}
                                stored_parsed = stored_eval.get("parsed") or {}
                                stored_mutated_text = stored_parsed.get("mutated_text", "").strip()
                                stored_prompt_attempt = stored.get("prompt_attempt", 1)
                                
                                if stored_mutated_text == mutated_text and stored_prompt_attempt == prompt_attempt:
                                    entry_exists = True
                                    break
                        
                        if not entry_exists:
                            record_entries.append({
                                "config": [config_type, False],
                                "data": serialised_entry,
                                "llm_response": llm_response,
                                "prompt_attempt": prompt_attempt,
                            })
                            successful_count += 1
                    
                    # ===== Intention Preservation Judging =====
                    st.markdown("**🔄 Intention Preservation Judging**")
                    st.caption("Assessing whether mutated prompts preserve the original harmful intention...")
                    
                    judging_progress = st.progress(0.0)
                    judging_text = st.empty()
                    
                    for judge_idx, eval_item in enumerate(evaluated_mutations):
                        evaluation = eval_item["evaluation"]
                        mutated_text = evaluation.parsed.mutated_text.strip()
                        strategy = evaluation.mutation.strategy
                        
                        judging_text.text(f"Judging mutation {judge_idx + 1}/{len(evaluated_mutations)} ({strategy})...")
                        with st.spinner(""):
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
                                # Determine config_type from the evaluation's mode
                                config_type = "one" if evaluation.mode == "One-Shot" else "few"
                                prompt_attempt = eval_item.get("prompt_attempt", 1)
                                for stored in record_entries:
                                    stored_config = stored.get("config") or []
                                    if stored_config and stored_config[0] == config_type:
                                        stored_data = stored.get("data") or {}
                                        stored_eval = stored_data.get("evaluation") or {}
                                        stored_parsed = stored_eval.get("parsed") or {}
                                        stored_mutated_text = stored_parsed.get("mutated_text", "").strip()
                                        stored_prompt_attempt = stored.get("prompt_attempt", 1)
                                        
                                        if stored_mutated_text == mutated_text and stored_prompt_attempt == prompt_attempt:
                                            # Update with judging results
                                            judged_entry = MutationWithJudge(
                                                evaluation=evaluation,
                                                judge=assessment.secondary,
                                                judge_passed=assessment.judge_passed,
                                            )
                                            stored["data"] = serialise_mutation_with_judge(judged_entry)
                                            stored["config"] = [config_type, True]  # Mark as judged
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
                                st.warning(f"⚠️ Failed to judge mutation {judge_idx + 1}: {e}")
                                eval_item["assessment"] = None
                        
                        judging_progress.progress((judge_idx + 1) / len(evaluated_mutations))
                    
                    judging_progress.empty()
                    judging_text.empty()
                    
                    st.session_state["last_prompt"] = original_prompt
                    st.session_state["results_prompt_selector"] = original_prompt
                    
                    st.success(f"✅ **Generation Complete:** Evaluated {successful_count} mutations (ranked by ROUGE-L)")
    
    st.divider()

    # ===== Results Explorer =====
    prompts = [
        prompt_text
        for prompt_text, records in mutation_store.items()
        if any((entry.get("config") or [None])[0] in ["one", "few"] for entry in records)
    ]

    if prompts:
        st.markdown('<p class="analysis-step-label">Results explorer</p>', unsafe_allow_html=True)
        st.markdown("**📚 Generation Results Library**")
        st.caption("Results are cached in session state so you can revisit them.")

        # Get the first (and only) prompt from current run
        selected_prompt = prompts[0]

        records = [
            entry for entry in mutation_store.get(selected_prompt, [])
            if (entry.get("config") or [None])[0] in ["one", "few"]
        ]

        if records:
            ranked_rows: List[Dict[str, Any]] = []
            stored_panels: List[Dict[str, Any]] = []

            for entry in records:
                serialised = entry.get("data") or {}
                llm_response = entry.get("llm_response", "")
                judge_meta = entry.get("judge_meta") or {}
                config = entry.get("config") or []
                judged_flag = bool(config[1]) if len(config) > 1 else False
                prompt_attempt = entry.get("prompt_attempt", 1)

                deserialised = deserialise_mutation_with_judge(serialised)
                evaluation = deserialised.evaluation
                parsed = evaluation.parsed
                metrics = evaluation.metrics

                mutated_text = parsed.mutated_text.strip() if parsed and parsed.mutated_text else ""
                rouge_l = float(metrics.rouge_l or 0.0)
                jaccard = float(metrics.jaccard or 0.0)
                levenshtein = metrics.levenshtein

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
                    "prompt_attempt": prompt_attempt,
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
                    "prompt_attempt": prompt_attempt,
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
                        "prompt_attempt": row["prompt_attempt"],
                        "mutated_text": row["mutated_display"],
                        "llm_response": row["llm_display"],
                        "rouge_l": row["rouge_l"],
                        "jaccard": row["jaccard"],
                        "levenshtein": row["levenshtein"],
                        "judge_status": row["judge_status"],
                    })

                df = pd.DataFrame(df_data)
                st.dataframe(
                    df,
                    width='stretch',
                    hide_index=True,
                    column_config={
                        "rank": st.column_config.NumberColumn("Rank", width="small"),
                        "strategy": st.column_config.TextColumn("Strategy", width="medium"),
                        "attempt": st.column_config.NumberColumn("Strategy Attempt", width="small"),
                        "prompt_attempt": st.column_config.NumberColumn("Prompt Attempt", width="small"),
                        "mutated_text": st.column_config.TextColumn("Mutated Prompt", width="large"),
                        "llm_response": st.column_config.TextColumn("LLM Response", width="large"),
                        "rouge_l": st.column_config.TextColumn("ROUGE-L", width="small"),
                        "jaccard": st.column_config.TextColumn("Jaccard", width="small"),
                        "levenshtein": st.column_config.TextColumn("Levenshtein", width="small"),
                        "judge_status": st.column_config.TextColumn("Judge Status", width="medium"),
                    },
                )

                # Add boxplot analysis by strategy
                st.markdown("**📦 Distribution Analysis by Strategy**")
                st.caption("Boxplots showing the distribution of ROUGE-L scores across different persuasion strategies.")
                
                # Prepare data for boxplots
                strategy_groups = {}
                for row in ranked_rows:
                    strategy = row["strategy"]
                    try:
                        rouge_score = float(row["rouge_l"])
                        if strategy not in strategy_groups:
                            strategy_groups[strategy] = []
                        strategy_groups[strategy].append(rouge_score)
                    except (ValueError, TypeError):
                        continue
                
                if strategy_groups:
                    fig_box, ax_box = plt.subplots(figsize=(12, 6))
                    
                    # Prepare data for boxplot
                    strategies = list(strategy_groups.keys())
                    data = [strategy_groups[strategy] for strategy in strategies]
                    
                    # Create boxplot
                    box = ax_box.boxplot(data, labels=strategies, patch_artist=True)
                    
                    # Customize box colors
                    colors = ['lightblue', 'lightgreen', 'lightcoral', 'lightyellow', 'lightpink', 'lightcyan']
                    for i, patch in enumerate(box['boxes']):
                        color = colors[i % len(colors)]
                        patch.set_facecolor(color)
                        patch.set_edgecolor('black')
                        patch.set_linewidth(1.5)
                    
                    # Customize median lines
                    for median in box['medians']:
                        median.set(color='red', linewidth=2)
                    
                    ax_box.set_title('ROUGE-L Score Distribution by Persuasion Strategy', fontsize=14, fontweight='bold')
                    ax_box.set_ylabel('ROUGE-L Score', fontsize=12)
                    ax_box.set_xlabel('Strategy', fontsize=12)
                    ax_box.grid(True, alpha=0.3)
                    plt.xticks(rotation=45, ha='right')
                    
                    plt.tight_layout()
                    st.pyplot(fig_box)
                else:
                    st.info("No valid ROUGE-L scores available for boxplot analysis.")

                st.markdown("**🎯 Intention Preservation Judging Results**")
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

                    # Use Streamlit's native accordion (expander) for each mutation
                    with st.expander(f"Mutation #{idx} — {evaluation.mutation.strategy} | {meta_text}", expanded=False):
                        # Summary and Metrics in compact format
                        col1, col2 = st.columns(2)
                        with col1:
                            st.caption(f"**Strategy:** {evaluation.mutation.strategy} | **Strategy Attempt:** {evaluation.attempt} | **Prompt Attempt:** {panel_payload.get('prompt_attempt', 1)}")
                        with col2:
                            if metrics:
                                st.caption(f"**ROUGE-L:** {rouge_score:.4f} | **Jaccard:** {jaccard_value:.4f} | **Levenshtein:** {levenshtein_value}")
                        
                        # Mutated prompt
                        st.markdown("**📝 Mutated Prompt**")
                        st.text(mutated_text)
                        
                        # Ground truth comparison - moved inside accordion
                        if llm_response:
                            reference_text = stage1_reference_map.get(selected_prompt, '')
                            if reference_text:
                                st.markdown("**🧠 Ground Truth Comparison**")
                                render_direct_recall_diff(
                                    reference_text,
                                    llm_response,
                                    title="Generated Text vs. Reference Text",
                                    metrics=metrics,
                                )
                        
                        # Intention judging results
                        if judged_flag:
                            st.markdown("**🧠 Primary Intention Assessment**")
                            primary_error = judge_meta.get("primary_error")
                            if primary_error:
                                st.caption(f"Error: {primary_error}")
                            else:
                                core_intention = judge_meta.get("core_intention")
                                restated_mutated_text = judge_meta.get("restated_mutated_text")
                                bits = []
                                if core_intention:
                                    bits.append(f"**Core Intention:** {core_intention}")
                                if restated_mutated_text:
                                    bits.append(f"**Restated:** {restated_mutated_text}")
                                if bits:
                                    st.markdown("\n\n".join(bits))
                                else:
                                    st.caption("No assessment data available")
                            
                            col3, col4 = st.columns(2)
                            with col3:
                                st.markdown("**🗳️ Judge Raw Response**")
                                judge_response = judge_result.response if judge_result else ""
                                if judge_response:
                                    st.text(judge_response)
                                else:
                                    st.caption("No response available")
                            
                            with col4:
                                st.markdown("**⚖️ Secondary Validation**")
                                secondary_error = judge_meta.get("secondary_error")
                                if secondary_error:
                                    st.caption(f"Error: {secondary_error}")
                                else:
                                    st.markdown(f"{status_icon} {status_text}")
                        else:
                            st.caption("⏳ **Primary Intention Assessment & Secondary Validation:** Pending — judge not run yet.")

def render_unlearning_detection_page(api_key, model_choice, provider):
    """Render the representational analysis experience."""
    
    # Initialize session state for Unlearning Detection
    if 'unlearn_feature_id_index' not in st.session_state:
        st.session_state['unlearn_feature_id_index'] = 0
    if 'unlearn_reference_model' not in st.session_state:
        st.session_state['unlearn_reference_model'] = ""
    if 'unlearn_updated_model' not in st.session_state:
        st.session_state['unlearn_updated_model'] = ""
    if 'unlearn_query_text' not in st.session_state:
        st.session_state['unlearn_query_text'] = ""
    if 'unlearn_batch_size' not in st.session_state:
        st.session_state['unlearn_batch_size'] = 4
    if 'unlearn_num_batches' not in st.session_state:
        st.session_state['unlearn_num_batches'] = 10
    if 'unlearn_max_length' not in st.session_state:
        st.session_state['unlearn_max_length'] = 128
    if 'unlearn_last_result' not in st.session_state:
        st.session_state['unlearn_last_result'] = None
    
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
        feature_options = [feature.id for feature in features]
        selected_feature_id = st.selectbox(
            "Select representational probe",
            options=feature_options,
            index=min(st.session_state['unlearn_feature_id_index'], len(feature_options) - 1),
            format_func=lambda feature_id: f"{feature_lookup[feature_id].name} — {feature_lookup[feature_id].description}",
            key="representational_feature_selection",
            help="Maps directly to the `feature` argument of `run_feature_analysis`.",
        )
        # Update index in session state
        if selected_feature_id in feature_options:
            st.session_state['unlearn_feature_id_index'] = feature_options.index(selected_feature_id)

        selected_feature = feature_lookup[selected_feature_id]

        st.markdown("##### Model checkpoints")
        st.info("💡 **Model Path Format**: Use Hugging Face model IDs (e.g., 'gpt2', 'microsoft/DialoGPT-medium') or absolute paths to local directories containing `config.json` and model files. Do not use Hugging Face cache paths directly.")
        col_ref, col_upd = st.columns(2)
        with col_ref:
            reference_model_path = st.text_input(
                "Reference model (baseline)",
                value=st.session_state['unlearn_reference_model'],
                placeholder="e.g. gpt2, Qwen/Qwen2.5-7B, or /path/to/local/model",
                help="Hugging Face model ID (e.g., 'gpt2') or absolute path to local model directory containing config.json",
                key="representational_reference_model",
            )
        with col_upd:
            st.text_input(
                "Updated / deployed model",
                value=st.session_state['unlearn_updated_model'],
                placeholder="Path or HF repo ID for the model under audit",
                help="Hugging Face model ID (e.g., 'microsoft/DialoGPT-medium') or absolute path to local model directory",
                key="representational_updated_model",
            )

        st.markdown("##### Evaluation prompts")
        st.text_area(
            "Evaluation prompts",
            value=st.session_state['unlearn_query_text'],
            height=180,
            placeholder="Enter one query per line that probes the model's behaviour post-unlearning.\n\nExample:\nThe quick brown fox jumps over the lazy dog.\nUnlearning LLMs is an active area of research.\nWhat is the capital of France?",
            help="Each non-empty line is passed as an element of the `query` list. Enter multiple queries (one per line) to test different prompts.",
            key="representational_query_text",
        )
        query_preview = [line.strip() for line in st.session_state.get('representational_query_text', '').splitlines() if line.strip()]

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
            st.number_input(
                "Batch size",
                min_value=1,
                max_value=128,
                value=st.session_state['unlearn_batch_size'],
                step=1,
                help="Mini-batch size for analyses that stream batches (FIM, CKA).",
                key="representational_batch_size",
            )
        with col_batches:
            st.number_input(
                "Batches",
                min_value=1,
                max_value=200,
                value=st.session_state['unlearn_num_batches'],
                step=1,
                help="Number of dataloader batches to use when estimating statistics (FIM, CKA).",
                key="representational_num_batches",
            )
        with col_length:
            st.number_input(
                "Max length",
                min_value=16,
                max_value=4096,
                value=st.session_state['unlearn_max_length'],
                step=16,
                help="Maximum sequence length for tokenization.",
                key="representational_max_length",
            )

        st.caption("Preview of the backend call that will be executed with your settings:")
        query_list_preview = ", ".join(f'"{q}"' for q in query_preview) or '"<enter at least one query>"'
        reference_model_path = st.session_state.get('representational_reference_model', '')
        updated_model_path = st.session_state.get('representational_updated_model', '')
        batch_size = st.session_state.get('representational_batch_size', 4)
        num_batches = st.session_state.get('representational_num_batches', 10)
        max_length = st.session_state.get('representational_max_length', 128)
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
        "**Choose a persuasion strategy**",
        [
            "Role-Playing: The Author",
            "Hypothetical Scenario: A Lost Manuscript",
            "Creative Writing Exercise",
            "Translation and Back-Translation",
            "Stylistic Transformation",
            "Tom and Jerry Game",
        ],
        help="Select a technique to encourage the model to generate a continuation.",
        key="persuasion_strategy_selectbox",
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

def render_sleek_attack_page(api_key, model_choice, provider):
    """Render the SLEEK Attack detection page."""
    
    # Initialize session state
    if 'sleek_document_text' not in st.session_state:
        st.session_state['sleek_document_text'] = ""
    if 'sleek_evaluation_results' not in st.session_state:
        st.session_state['sleek_evaluation_results'] = None
    
    st.markdown("### 🔍 SLEEK Attack")
    st.markdown(
        "Step-by-step Leaking and Extraction of 'Erased' Knowledge - A black-box attack framework for detecting residual knowledge in unlearned LLMs."
    )
    
    st.markdown(
        """
        <div class="analysis-callout">
            <div class="analysis-callout__title">How SLEEK Attack works</div>
            <ul class="analysis-callout__list">
                <li><strong>Step 1:</strong> Generate auxiliary LLM responses for step-by-step reasoning</li>
                <li><strong>Step 2:</strong> Extract knowledge points and generate targeted questions</li>
                <li><strong>Step 3:</strong> Categorize questions by knowledge type</li>
                <li><strong>Step 4:</strong> Execute attack and evaluate leakage</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    # Step 1: Provide source content
    st.markdown('<p class="analysis-step-label">Step 1 · Provide source content</p>', unsafe_allow_html=True)
    
    # Source mode selection
    source_mode = st.radio(
        "Where should the knowledge come from?",
        ["Input Text", "Upload Document"],
        horizontal=True,
        key="sleek_source_mode",
        help="Choose 'Input Text' for custom input or 'Upload Document' for PDF/TXT files.",
    )
    
    document_text = ""
    
    if source_mode == "Input Text":
        st.markdown("**📝 Input your text**")
        st.text_area(
            "Enter your text",
            height=200,
            placeholder="Paste or type the text content you want to test for knowledge leakage...",
            help="Provide the text content that may have been 'unlearned' from the target model.",
            key="sleek_input_text",
        )
        if st.session_state.get("sleek_input_text", "").strip():
            document_text = st.session_state["sleek_input_text"].strip()
            st.caption(f"Text length: {len(document_text)} characters · {len(document_text.split())} words")
    else:
        st.markdown("**📎 Upload your document**")
        uploaded_document = st.file_uploader(
            "Choose a pdf or txt file",
            type=["pdf", "txt"],
            help="Select a PDF or UTF-8 TXT document to extract knowledge from",
            key="sleek_document_upload"
        )
        if uploaded_document:
            try:
                from src.direct_recall.pdf_utils import extract_text_from_document
                document_text = extract_text_from_document(uploaded_document)
                st.caption(f"Extracted text length: {len(document_text)} characters · {len(document_text.split())} words")
            except Exception as e:
                st.error(f"Error extracting text from document: {e}")
    
    # Step 2: Configure evaluation
    st.markdown('<p class="analysis-step-label">Step 2 · Configure evaluation</p>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.slider(
            "Temperature",
            min_value=0.0,
            max_value=1.0,
            value=0.7,
            step=0.05,
            help="Controls randomness in LLM responses during the attack.",
            key="sleek_temperature"
        )
    
    with col2:
        st.slider(
            "Top-P",
            min_value=0.0,
            max_value=1.0,
            value=0.9,
            step=0.05,
            help="Nucleus sampling parameter for controlling diversity.",
            key="sleek_top_p"
        )
    
    # Run SLEEK Attack button
    run_sleek = st.button(
        "🚀 Run SLEEK Attack",
        key="run_sleek_button",
        type="primary",
        width='stretch'
    )
    
    if run_sleek:
        if not document_text:
            st.warning("⚠️ Please provide source content first.")
        elif not api_key or not api_key.strip():
            st.error(f"⚠️ Please configure the API key for **{provider}** in the sidebar.")
        elif not model_choice:
            st.error("⚠️ Please select a model in the sidebar.")
        else:
            temperature = st.session_state.get('sleek_temperature', 0.7)
            top_p = st.session_state.get('sleek_top_p', 0.9)
            
            with st.spinner("🔍 Running SLEEK Attack evaluation..."):
                try:
                    results = run_sleek_evaluation(
                        document_text=document_text,
                        api_key=api_key,
                        model_name=model_choice,
                        provider=provider,
                        temperature=temperature,
                        top_p=top_p
                    )
                    
                    st.session_state['sleek_evaluation_results'] = results
                    st.success("✅ SLEEK Attack evaluation completed!")
                    
                except Exception as e:
                    st.error(f"❌ SLEEK Attack failed: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())
    
    # Display results
    if st.session_state.get('sleek_evaluation_results'):
        results = st.session_state['sleek_evaluation_results']
        
        st.markdown("#### 📊 SLEEK Attack Results")
        
        # Overall metrics
        st.markdown("**Overall Leakage Assessment:**")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Questions Generated", results.get('total_questions', 0))
        with col2:
            st.metric("Questions with Leakage", results.get('questions_with_leakage', 0))
        with col3:
            leakage_rate = results.get('leakage_rate', 0.0)
            st.metric("Leakage Rate", f"{leakage_rate:.2%}")
        
        # Leakage interpretation
        if leakage_rate > 0.5:
            st.error("⚠️ **High Knowledge Leakage Detected**: The model shows significant residual knowledge of the content.")
        elif leakage_rate > 0.2:
            st.warning("⚠️ **Moderate Knowledge Leakage**: The model shows some residual knowledge that may indicate incomplete unlearning.")
        else:
            st.success("✅ **Low Knowledge Leakage**: The model appears to have effectively unlearned the content.")
        
        # Detailed results by question
        st.markdown("**Detailed Results by Question:**")
        
        questions = results.get('questions', [])
        for i, question_data in enumerate(questions, 1):
            question = question_data.get('question', '')
            category = question_data.get('category', 'Unknown')
            leakage_score = question_data.get('leakage_score', 0.0)
            has_leakage = question_data.get('has_leakage', False)
            
            with st.expander(f"Question {i}: {question[:60]}...", expanded=False):
                st.markdown(f"**Category:** {category}")
                st.markdown(f"**Leakage Score:** {leakage_score:.3f}")
                st.markdown(f"**Leakage Detected:** {'Yes' if has_leakage else 'No'}")
                
                if 'response' in question_data:
                    st.markdown("**Model Response:**")
                    st.write(question_data['response'])
                
                if 'support_response' in question_data:
                    st.markdown("**Auxiliary Reasoning:**")
                    st.write(question_data['support_response'])

def render_footer():
    """Renders a footer section."""
    # This is a placeholder for any footer content you might want to add later.
    pass

def main():
    """Main function to run the Streamlit app."""
    render_header()
    api_key, model_choice, provider, page = render_sidebar()

    if page == "Content Recall Test":
        render_snippet_to_document_page(api_key, model_choice, provider)
    elif page == "Unlearning Detection Test":
        render_unlearning_detection_page(api_key, model_choice, provider)
    elif page == "Legal Cases Display":
        render_legal_case_display_page()

    # Footer (currently empty, can be customized)
    render_footer()