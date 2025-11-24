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
from src.config import DEFAULT_OPENROUTER_KEY

import matplotlib.pyplot as plt
from src.direct_recall.persuasive_jailbreak import (
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

LEGAL_CASES: List[Dict[str, Any]] = [
    {
        "id": "gema-openai-2025",
        "title": "GEMA v. OpenAI — Europe’s First Generative AI Copyright Judgment",
        "tagline": "Munich I Regional Court finds unauthorized lyric ingestion",
        "headline": "OpenAI loses the first-instance trial; appeal anticipated",
        "decision_date": "2025-11-11",
        "published_on": "2025-11-15",
        "jurisdiction": "Munich I Regional Court, Germany",
        "case_number": "42 O 14139/24",
        "plaintiff": "GEMA (German musical works collective management organization)",
        "defendant": "OpenAI",
        "summary": (
            "On 11 November 2025, the Munich I Regional Court issued the first-instance judgment in GEMA v. "
            "OpenAI. The panel held that OpenAI infringed reproduction rights by using copyrighted lyrics from "
            "nine popular German songs to train ChatGPT without licences."
        ),
        "status": "First-instance judgment; not yet final. OpenAI is expected to appeal.",
        "key_points": [
            "Training-stage memorisation and downstream lyric reproduction both infringe the reproduction and making-available rights under German law.",
            "The court rejected OpenAI’s reliance on the text and data mining exception (Section 44b UrhG) and the non-commercial research exemption (Section 60d UrhG).",
            "The ruling aligns with the EU AI Act’s emphasis on data governance and copyright compliance for foundation models.",
        ],
        "timeline": [
            {
                "date": "2024-11-13",
                "event": "GEMA files suit",
                "detail": "GEMA announces proceedings in Munich, alleging that ChatGPT was trained on member lyrics without licences.",
            },
            {
                "date": "2025-11-11",
                "event": "First-instance decision",
                "detail": "The Munich I Regional Court rules in favour of GEMA, finding copyright infringement by OpenAI.",
            },
            {
                "date": "2025-11-15",
                "event": "Industry briefing",
                "detail": "Observers label the case the first generative AI copyright verdict in Europe and anticipate appellate proceedings.",
            },
        ],
        "sections": [
            {
                "heading": "Part 01 · Case Snapshot",
                "intro": "How the dispute emerged and what each side argued.",
                "subsections": [
                    {
                        "title": "Parties",
                        "bullets": [
                            "GEMA — one of Europe’s largest collective management organisations, representing nearly 100,000 German members and more than two million rightsholders worldwide.",
                            "OpenAI — the developer and operator of ChatGPT, a commercial large language model platform.",
                        ],
                    },
                    {
                        "title": "Claims raised by GEMA",
                        "body": "On 13 November 2024 GEMA confirmed it had sued OpenAI for infringing members’ reproduction rights.",
                        "bullets": [
                            "GEMA alleges ChatGPT was trained on lyrics from nine high-value German hits, including Herbert Grönemeyer’s 1984 synth-pop classic “Männer” and Helene Fischer’s stadium anthem “Atemlos durch die Nacht”.",
                            "Tests showed that simple prompts triggered ChatGPT to regenerate the lyrics almost verbatim without accessing the live internet, demonstrating memorisation.",
                            "GEMA stresses that it had issued a machine-readable opt-out prohibiting text and data mining (TDM), which OpenAI allegedly ignored.",
                        ],
                    },
                    {
                        "title": "OpenAI’s defence",
                        "bullets": [
                            "OpenAI argued the lyrics served only as training data and that ChatGPT does not store or reproduce the works itself; any reproduction should be attributed to the user’s prompt.",
                            "The company claimed users, not the platform, bear infringement risk because ChatGPT is merely a tool.",
                            "It invoked Section 44b UrhG (text and data mining exception), insisting GEMA’s opt-out was ineffective, and further cited Section 60d UrhG by portraying OpenAI as a research institution.",
                            "OpenAI added that public availability of the songs implied consent and warned that an injunction would force it to discontinue its German services, harming the public interest.",
                        ],
                    },
                ],
            },
            {
                "heading": "Part 02 · Court’s Findings",
                "intro": "The court sided with GEMA and systematically rejected OpenAI’s arguments.",
                "subsections": [
                    {
                        "title": "1. Training-stage memorisation is reproduction",
                        "body": "The judges characterised the model’s “memory” of lyrics as a reproduction because the training pipeline extracts and stores protected expression beyond fleeting analysis.",
                    },
                    {
                        "title": "2. Generated outputs amount to reproduction and making available",
                        "body": "Producing complete lyrics in response to simple prompts demonstrates that ChatGPT itself recreates and offers the works to the public, making OpenAI responsible for the output.",
                    },
                    {
                        "title": "3. No shelter under Section 44b UrhG",
                        "body": "Text and data mining privileges apply only when copies are strictly instrumental to extracting information and do not impair the rightsholder’s exploitation. The court held that memorising lyrics for model use exceeds that limit, especially because GEMA had expressly opted out.",
                    },
                    {
                        "title": "4. No protection under Section 60d UrhG",
                        "body": "OpenAI is a commercial enterprise monetising ChatGPT; therefore it cannot rely on the non-commercial research exception. Public accessibility of lyrics does not equate to licence or waiver.",
                    },
                    {
                        "title": "5. Remedies and compliance expectations",
                        "body": (
                            "The court ordered OpenAI to cease using the disputed lyrics in the training and operation of ChatGPT and "
                            "to implement effective safeguards against further reproduction. While the precise level of damages is expected "
                            "to be clarified in subsequent proceedings or settlement, the decision signals that courts may impose "
                            "forward-looking technical obligations on model providers, including verifiable training-data provenance, "
                            "respect for machine-readable TDM opt-outs, and deployment of content filters that materially reduce verbatim "
                            "reproduction risk."
                        ),
                    },
                ],
            },
            {
                "heading": "Part 03 · Comparative Lens: Robert Kneschke v. LAION (Hamburg)",
                "intro": "The LAION dataset dispute illustrates how different facts can shift the legal analysis.",
                "subsections": [
                    {
                        "title": "Key contrasts",
                        "bullets": [
                            "The LAION dataset stored only URLs and textual descriptors of the plaintiff’s images rather than the images themselves.",
                            "LAION operates as a registered non-profit and distributes the dataset for free, unlike OpenAI’s commercial ChatGPT platform.",
                            "The challenged conduct concerned temporary downloads to verify metadata during dataset curation, not model training or public outputs.",
                            "Had the dataset bundled the images, been commercial in nature, or been relied upon by a model developer, the outcome might have been closer to GEMA v. OpenAI.",
                        ],
                    },
                ],
            },
            {
                "heading": "Part 04 · Policy Context and Market Dynamics",
                "intro": "Europe and the United States prioritise different interests in the AI copyright debate.",
                "subsections": [
                    {
                        "title": "Europe: Copyright-centric infrastructure",
                        "bullets": [
                            "Long-standing cultural heritage and early copyright legislation, from the 1709 Statute of Anne to modern civil-law author-right regimes.",
                            "Powerful collecting societies such as GEMA, SACEM, and PRS for Music coordinate licensing and enforcement at scale.",
                            "Recent regulatory initiatives like the EU AI Act stress data governance, transparency, and respect for creator rights.",
                        ],
                    },
                    {
                        "title": "United States: AI industry powerhouse",
                        "bullets": [
                            "Home to the world’s largest AI companies (OpenAI, Google, Meta) with deep datasets and capital to scale foundation models.",
                            "A strategic incentive to minimise licensing costs for training data, prompting friction with European rightsholders.",
                            "Divergent policy focus on innovation speed versus cultural remuneration amplifies transatlantic tension in cases like this.",
                        ],
                    },
                ],
            },
            {
                "heading": "Part 05 · Outlook",
                "body": (
                    "GEMA v. OpenAI marks the first European judgment squarely addressing generative AI training on protected works. "
                    "It strengthens creators’ negotiating position and will influence compliance expectations worldwide. For practitioners, "
                    "such cases increase the value of audit tooling—recall tests, QA-based memorisation probes, and single-choice diagnostics—"
                    "to document that a deployed model has been examined for verbatim reproduction risk and that mitigation efforts are "
                    "ongoing. Each emerging “first case” in AI copyright law—across Europe, the United States, and beyond—helps "
                    "define the balance between innovation and author rights."
                ),
            },
        ],
    }
    ,
    {
        "id": "getty-stabilityai-2025",
        "title": "Getty Images v. Stability AI — UK High Court on AI Training and Infringing Copies",
        "tagline": "Stable Diffusion is not, by itself, an infringing copy under UK law",
        "headline": "UK High Court clarifies when AI models are “articles” but not infringing copies",
        "decision_date": "2025-11-04",
        "published_on": "2025-11-04",
        "jurisdiction": "High Court of Justice of England and Wales (EWHC)",
        "case_number": "Getty Images (US), Inc. v. Stability AI Ltd.",
        "plaintiff": "Getty Images (US), Inc.",
        "defendant": "Stability AI Ltd.",
        "summary": (
            "Mrs Justice Joanna Smith DBE held that, on the record before the court, the Stable Diffusion model is not "
            "itself an “infringing copy” of Getty’s works for the purposes of sections 22 and 23 of the Copyright, Designs "
            "and Patents Act 1988 (CDPA), because the model does not store or reproduce the underlying images. The court "
            "nonetheless found limited trademark infringement where synthetic outputs contained Getty and iStock watermarks."
        ),
        "status": "High Court judgment; primary copyright and database claims remain unresolved on the merits.",
        "key_points": [
            "The court treated an AI model as an “article” under sections 22 and 23 CDPA, confirming that intangible, electronic artefacts can in principle be infringing copies.",
            "Stable Diffusion was found not to be an infringing copy because its weights do not store or reproduce copyrighted images, despite exposure during training.",
            "Getty obtained only a partial victory on trademark claims related to watermarked outputs; the broader copyright questions remain open.",
        ],
        "timeline": [
            {
                "date": "2019-11-04",
                "event": "Stability AI incorporated",
                "detail": "Stability AI Ltd is incorporated in England and Wales, later developing the Stable Diffusion model.",
            },
            {
                "date": "2023-01-16",
                "event": "Getty files UK suit",
                "detail": "Getty Images sues Stability AI in the High Court, alleging primary and secondary copyright infringement, database right infringement, trademark infringement, and passing off.",
            },
            {
                "date": "2025-06-01",
                "event": "Trial and narrowing of issues",
                "detail": "At trial in June 2025, Getty abandons its primary infringement claim after failing to show that Stable Diffusion’s training and development took place in the UK; Stability has also blocked certain prompts generating allegedly infringing output.",
            },
            {
                "date": "2025-11-04",
                "event": "High Court judgment",
                "detail": "Mrs Justice Joanna Smith DBE issues a 205-page judgment clarifying the status of AI models as possible “articles” and rejecting that Stable Diffusion is an infringing copy on the facts.",
            },
        ],
        "sections": [
            {
                "heading": "Part 01 · Case Background",
                "intro": "Who the parties are and how the dispute arose.",
                "subsections": [
                    {
                        "title": "Parties",
                        "bullets": [
                            "Getty Images — a leading global visual content licensor founded in 1995, owning extensive catalogues of photographs, video, illustrations, audio, and associated databases and trademarks.",
                            "Stability AI — a UK‑incorporated machine learning company (since 2019) best known for developing the Stable Diffusion image generation model and other generative AI systems.",
                        ],
                    },
                    {
                        "title": "Claims brought by Getty",
                        "body": (
                            "In January 2023 Getty sued Stability AI in London, alleging that Stable Diffusion had been trained on millions of Getty images and metadata scraped without licence, and asserting primary and secondary copyright "
                            "infringement, database right infringement, trademark infringement, and passing off."
                        ),
                        "bullets": [
                            "Getty argued that scraping and using large volumes of Getty content to train Stable Diffusion amounted to unlawful commercial exploitation of its catalogue.",
                            "It further claimed that downstream outputs sometimes reproduced Getty’s trademarks and watermarks (including “Getty Images” and “iStock”), misleading users and damaging brand value.",
                            "Getty maintained that the development and distribution of Stable Diffusion’s model weights constituted dealing in infringing copies.",
                        ],
                    },
                    {
                        "title": "Narrowing of the issues at trial",
                        "body": (
                            "By the June 2025 trial Getty withdrew its primary copyright claim because it could not prove that "
                            "Stable Diffusion’s initial training and development occurred within the UK. Stability had also blocked "
                            "certain prompts, achieving much of the practical relief Getty had sought on that front."
                        ),
                    },
                ],
            },
            {
                "heading": "Part 02 · Articles and Infringing Copies under CDPA",
                "intro": "How the court interpreted “article” and “infringing copy” for AI models.",
                "subsections": [
                    {
                        "title": "1. Can an AI model be an “article” under sections 22 and 23 CDPA?",
                        "body": (
                            "Sections 22 and 23 CDPA govern secondary infringement through importing and dealing in “articles” that are infringing "
                            "copies. Stability argued that an “article” must be a tangible item, but the court disagreed."
                        ),
                        "bullets": [
                            "Justice Smith read “article” in light of section 17 CDPA, which defines copying to include electronic storage of a work in any medium.",
                            "Limiting “article” to physical objects would undermine protection for electronic copies and conflict with the statutory scheme.",
                            "On that basis, she held that an AI model like Stable Diffusion can, in principle, qualify as an article for sections 22 and 23."
                        ],
                    },
                    {
                        "title": "2. Stable Diffusion is not, on these facts, an infringing copy",
                        "body": (
                            "The harder question was whether Stable Diffusion’s weights were themselves infringing copies of Getty’s works. "
                            "Justice Smith concluded they were not."
                        ),
                        "bullets": [
                            "An infringing copy must itself reproduce a work, not merely have been exposed to the work during training.",
                            "Drawing an analogy with Sony v. Ball (2004), the judge reasoned that RAM chips cease to be infringing copies when they no longer store gameplay data; similarly, Stable Diffusion’s weights do not store Getty images.",
                            "Accordingly, a model “which does not store or reproduce any Copyright Works (and has never done so) is not an ‘infringing copy’” under sections 22 and 23 CDPA."
                        ],
                    },
                ],
            },
            {
                "heading": "Part 03 · Trademark and Passing Off Findings",
                "intro": "Where Getty succeeded—and where it did not—on brand protection claims.",
                "subsections": [
                    {
                        "title": "1. Partial success on trademark infringement",
                        "body": (
                            "More than half of the judgment addresses Getty’s claims under sections 10(1)–(3) of the Trade Marks Act 1994, "
                            "focused on Getty and iStock watermarks appearing in generated images."
                        ),
                        "bullets": [
                            "The court found “double identity” infringement under section 10(1) and confusion-based infringement under section 10(2) for certain images containing Getty’s marks.",
                            "However, the scope of infringement was limited to the specific watermarked outputs evidenced in the record.",
                            "Justice Smith declined to infer infringement from test prompts (such as “news photo”) without proof that real users employed the same prompts."
                        ],
                    },
                    {
                        "title": "2. Reputation and passing off claims largely fail",
                        "bullets": [
                            "The court rejected Getty’s section 10(3) claim, noting that Stability actively attempted to filter out watermarks, which cut against any intent to ride on Getty’s reputation.",
                            "Passing off failed because Getty did not adequately rebut Stability’s argument that the tort does not extend to post-sale confusion; any confusion would arise only after a user chose to download or access outputs.",
                        ],
                    },
                ],
            },
            {
                "heading": "Part 04 · Unresolved Questions and Practical Takeaways",
                "intro": "What the judgment does not decide—and what it still signals for AI training.",
                "subsections": [
                    {
                        "title": "1. Open questions on primary infringement and datasets",
                        "bullets": [
                            "Because Getty withdrew its primary copyright and database-right claims, the court did not rule on whether scraping Getty’s catalogue for AI training would itself infringe UK copyright or database rights.",
                            "Fundamental questions about lawful boundaries for training data therefore remain open and may be litigated in future cases.",
                        ],
                    },
                    {
                        "title": "2. Signals for AI developers and rightsholders",
                        "bullets": [
                            "The decision confirms that AI models can be treated as electronic “articles” but will not be infringing copies unless they store or reproduce protected works.",
                            "Developers still face risk around outputs (e.g., watermarks, logos, distinctive styles) and around factual circumstances not tested in this case, such as UK‑based training or more direct reproductions.",
                            "Rightsholders gain guidance on how to structure evidence for future litigation, including logging training locations, dataset composition, and representative outputs."
                        ],
                    },
                ],
            },
            {
                "heading": "Part 05 · Relationship to GEMA v. OpenAI and Global Trends",
                "intro": "How the UK approach contrasts with emerging EU case law.",
                "subsections": [
                    {
                        "title": "1. Different answers to similar anxieties",
                        "bullets": [
                            "Where the Munich court in GEMA v. OpenAI framed training‑stage memorisation and output reproduction as copyright infringement, the UK High Court focused more narrowly on whether the model itself is an infringing copy.",
                            "Together, the cases show that jurisdictions can agree on the economic and ethical stakes of AI training while adopting different legal tests and thresholds for liability.",
                        ],
                    },
                    {
                        "title": "2. Implications for compliance and tooling",
                        "bullets": [
                            "UK developers cannot assume “no infringement” simply because a model does not store pixel‑level copies; outputs, trademark use, and training workflows remain scrutinised.",
                            "For both rightsholders and AI builders, systematic evaluation—recall tests, QA probes, and watermark/brand checks—helps document that models are monitored for memorisation and brand misuse, supporting more defensible positions under evolving case law.",
                        ],
                    },
                ],
            },
        ],
    }
    ,
    {
        "id": "kadrey-meta-2025",
        "title": "Kadrey v. Meta Platforms — N.D. Cal. Fair Use Ruling on LLM Training",
        "tagline": "Highly transformative use, but plaintiffs lose on market-harm evidence",
        "headline": "Court upholds Meta’s fair use defense for training Llama on 13 authors’ books",
        "decision_date": "2025-06-25",
        "published_on": "2025-06-25",
        "jurisdiction": "United States District Court for the Northern District of California",
        "case_number": "23-cv-03417-VC",
        "plaintiff": "Richard Kadrey et al. (13 authors, including Sarah Silverman, Junot Díaz, Ta-Nehisi Coates)",
        "defendant": "Meta Platforms, Inc.",
        "summary": (
            "Judge Vince Chhabria denied the authors’ motion for partial summary judgment and granted Meta’s cross-"
            "motion, holding on this record that Meta’s copying of 13 authors’ books to train its Llama models was fair "
            "use. The opinion stresses that although training generative AI on copyrighted books will often be illegal, "
            "fair use is fact-specific and evidence-driven. Here, Meta prevailed because the use was highly transformative "
            "and the plaintiffs failed to substantiate any of their market-harm theories—regurgitation, loss of a training-"
            "licence market, or market dilution from AI-generated competing works."
        ),
        "status": "Partial summary judgment for Meta on reproduction-based copyright claim; distribution and other issues remain.",
        "key_points": [
            "Training LLMs on books is a highly transformative use under U.S. fair use doctrine, but this does not automatically determine the outcome.",
            "The fourth factor (market effect) is the \"single most important\" element; on this record, plaintiffs offered almost no empirical evidence of direct or indirect market harm.",
            "Regurgitation and loss of a licensing market for AI training were rejected as sufficient theories of cognizable harm, while a more promising market-dilution theory was mentioned only in passing and left undeveloped.",
            "The court emphasised that Meta’s win is narrow: training on copyrighted books may often be unlawful when rightsholders can show that AI outputs meaningfully substitute for, or dilute demand for, human-authored works.",
            "The opinion explicitly distances itself from approaches that over-emphasise transformativeness (such as Bartz v. Anthropic) without squarely engaging with the risk that generative AI will flood markets with competing works.",
        ],
        "timeline": [
            {
                "date": "2023-07-",
                "event": "Case filed",
                "detail": "Thirteen authors sue Meta in N.D. Cal., alleging that Meta downloaded their books from shadow libraries and used them to train Llama without permission.",
            },
            {
                "date": "2024-2025",
                "event": "Claims narrowed",
                "detail": "Most non-copyright claims (vicarious infringement, unjust enrichment, negligence, CDAFA) are dismissed; a DMCA claim and a distribution theory (torrenting) remain alongside reproduction claims.",
            },
            {
                "date": "2025-03-",
                "event": "Cross-motions for partial summary judgment",
                "detail": "Parties file cross-motions focused on whether Meta’s reproduction of the books for LLM training is fair use as a matter of law.",
            },
            {
                "date": "2025-06-25",
                "event": "Fair use ruling",
                "detail": "Judge Chhabria issues an extensive opinion denying plaintiffs’ motion and granting Meta’s, holding that Meta’s reproduction for training is fair use on this record.",
            },
            {
                "date": "2025-07-11",
                "event": "Next steps",
                "detail": "Court schedules further case management to address the remaining claim that Meta unlawfully distributed plaintiffs’ works while torrenting shadow libraries.",
            },
        ],
        "sections": [
            {
                "heading": "Part 01 · Case Background",
                "intro": "Who sued Meta, what Llama is, how Meta used shadow libraries, and how the fair-use question was procedurally framed.",
                "subsections": [
                    {
                        "title": "Parties and technologies",
                        "bullets": [
                            "Plaintiffs: thirteen published authors, including Richard Kadrey, Sarah Silverman, Junot Díaz, Ta-Nehisi Coates and other prominent fiction and non-fiction writers.",
                            "Defendant: Meta Platforms, Inc., developer of the Llama family of large language models (Llama 1–4) and the Meta AI chatbot.",
                            "Llama models are trained on massive text datasets (Common Crawl, Wikipedia, GitHub, ArXiv, Stack Exchange, Books3, LibGen, Anna’s Archive) to learn statistical patterns in language.",
                        ],
                    },
                    {
                        "title": "How Meta obtained the books",
                        "body": (
                            "Meta initially explored licensing books from traditional publishers, but encountered fragmented rights, territorial limits, and sluggish negotiations. It then turned to so‑called "
                            "shadow libraries (LibGen and later Anna’s Archive) to obtain large book corpora, downloading them via BitTorrent in ways that raise separate piracy and distribution concerns."
                        ),
                        "bullets": [
                            "Two-thirds of Llama 1 and 2’s training data came from Common Crawl; books were sourced from Books3 and later from LibGen and Anna’s Archive.",
                            "Meta torrented LibGen in late 2022 and Anna’s Archive in early 2024, downloading at least 666 copies of books whose copyrights the plaintiffs hold.",
                            "Engineers wrote scripts to prevent \"seeding\" via BitTorrent, but default settings may have allowed some \"leeching\" (reuploading) of pieces of the datasets, a factual dispute that the court carved out for later proceedings.",
                        ],
                    },
                    {
                        "title": "Claims and narrowed issues",
                        "body": "The case evolved from a broad, multi-claim class action into a narrower test of fair use for training and a residual distribution theory.",
                        "bullets": [
                            "Plaintiffs alleged direct and vicarious copyright infringement, DMCA violations, unfair competition, unjust enrichment, negligence, and a CDAFA claim.",
                            "Most claims were dismissed early; plaintiffs were allowed to add a distribution theory (torrent reuploading) and a refined DMCA claim.",
                            "At the court’s invitation, the parties focused first on cross-motions for partial summary judgment on reproduction and fair use for the named plaintiffs only (no class certified).",
                        ],
                    },
                ],
            },
            {
                "heading": "Part 02 · Fair Use Framework and Factors 1–3",
                "intro": "Why the court saw LLM training as \"highly transformative\" and how that shapes—but does not end—the analysis under the first three factors.",
                "subsections": [
                    {
                        "title": "1. Overall fair use frame",
                        "body": (
                            "The opinion opens with an unusually candid discussion of generative AI: in many situations, copying books for LLM training will likely be illegal because it can erode incentives to create. But fair use is a flexible, "
                            "fact-specific defence focused on substitution, not a one‑size‑fits‑all permission slip for \"transformative\" technologies."
                        ),
                        "bullets": [
                            "Copyright’s goal is to preserve incentives to create by preventing uses that substantially substitute for the original works in the market.",
                            "Fair use is a mixed question of law and fact, organised around four statutory factors but ultimately a holistic substitution inquiry.",
                            "The court emphasises that factor four (market impact) is \"undoubtedly the single most important element\" and can override even highly transformative uses when plaintiffs demonstrate meaningful substitution or dilution.",
                        ],
                    },
                    {
                        "title": "2. Factor one — purpose, character, and transformativeness",
                        "body": "Meta’s use of the books to train LLMs is found \"highly transformative\" even though it is commercial and involved shadow libraries.",
                        "bullets": [
                            "Llama is an \"innovative tool\" that helps users do tasks (summarisation, coding, translation, research) very different from reading the books for entertainment or education.",
                            "An LLM \"reads\" books by learning statistical patterns rather than consuming expression as humans do; this, combined with its general-purpose outputs, supports a distinct purpose and character.",
                            "Commerciality and alleged bad faith (using shadow libraries) are noted but treated as secondary given the high degree of transformation and the absence of evidence that Meta’s downloads materially supported the shadow libraries’ operations.",
                        ],
                    },
                    {
                        "title": "3. Factor two and three — nature and amount of use",
                        "body": "The court recognises that books are core expressive works, so factor two favours plaintiffs but carries limited weight in this context.",
                        "bullets": [
                            "Meta’s use depends on the books’ expression, not just unprotected \"functional\" information, so intermediate-copying cases (like Sega and Connectix) do not apply cleanly.",
                            "Copying entire books is \"reasonably necessary\" for training, given that LLMs perform better when exposed to longer, coherent texts; factor three therefore favours Meta.",
                            "The opinion notes that the amount copied does not appear to drive market substitution here, because Llama does not output large, sequential portions of the plaintiffs’ works.",
                        ],
                    },
                ],
            },
            {
                "heading": "Part 03 · Market Harm Theories and Factor Four",
                "intro": "Three possible theories of market harm — and why only one is promising but badly underdeveloped on this record.",
                "subsections": [
                    {
                        "title": "1. Regurgitation and direct substitution",
                        "body": "The first theory is that users could prompt Llama to reproduce substantial passages from the plaintiffs’ books, substituting for paid access.",
                        "bullets": [
                            "Empirical testing showed that even adversarial prompting could not coax Llama to output more than ~50 tokens from any plaintiff’s book.",
                            "Both experts agreed that Llama could not reproduce any \"significant percentage\" of the books; this fell well below levels that raised concerns in Google Books.",
                            "On this record, direct substitution via regurgitation was rejected as a viable harm theory.",
                        ],
                    },
                    {
                        "title": "2. Loss of a licensing market for training data",
                        "body": "Plaintiffs’ primary briefing focused on the idea that unauthorised training harms a potential market for licensing books as training data, but the court rejected this as circular in the context of a new, transformative use.",
                        "bullets": [
                            "The court held that loss of fees for licences that enable a new, transformative use is generally not cognizable, to avoid circularly defining every fair use as a lost licence.",
                            "Citing Nimmer, Bill Graham Archives, and Oracle, the opinion concludes that harm to a hypothetical market for AI-training licences cannot by itself defeat fair use.",
                        ],
                    },
                    {
                        "title": "3. Market dilution from AI-generated competing works",
                        "body": (
                            "The court develops, in detail, a third theory: indirect substitution or market dilution, where LLM outputs flood the market with non‑infringing but competing works—a theory it views as highly serious in principle but "
                            "poorly supported by the plaintiffs’ actual evidence."
                        ),
                        "bullets": [
                            "LLMs could generate endless genre fiction, how-to guides, or news-like content that competes for readers’ attention and sales, especially for lesser-known authors.",
                            "Training on copyrighted books can make models significantly better at generating such competing outputs than models trained only on public domain works.",
                            "The court stresses that this type of market dilution is a cognizable harm under factor four — but the plaintiffs in this case barely raised it and offered virtually no evidence (e.g., sales data, genre-specific impact, or counter-expert analysis), so Meta’s expert submissions went effectively unrebutted.",
                        ],
                    },
                ],
            },
            {
                "heading": "Part 04 · Evidence Gaps, Shadow Libraries, and Limited Consequences",
                "intro": "Why Meta wins here — and why shadow-library facts, public-interest rhetoric, and abstract fears about \"killing AI innovation\" did not change the outcome.",
                "subsections": [
                    {
                        "title": "1. Plaintiffs’ evidentiary shortfalls",
                        "body": "After Meta offered expert analysis suggesting no observable sales impact, plaintiffs did not respond with empirical evidence of their own.",
                        "bullets": [
                            "The complaint and summary-judgment briefing focused on regurgitation and licensing, not market dilution from competing AI-generated works.",
                            "Plaintiffs’ expert mentioned reports of AI-generated books \"flooding Amazon\" but did not quantify effects on the markets for these plaintiffs’ specific works.",
                            "Because fair use is an affirmative defense, Meta bore the initial burden, but once it produced market-harm evidence, plaintiffs’ speculation without data was insufficient to create a triable issue.",
                        ],
                    },
                    {
                        "title": "2. Shadow libraries and bad faith",
                        "body": "The opinion acknowledges that LibGen and Anna’s Archive traffic in pirated books and that Meta’s downloads may involve reuploading via BitTorrent, but treats these facts carefully and separates them from the fair-use analysis.",
                        "bullets": [
                            "Downloading from shadow libraries does not automatically defeat fair use; otherwise courts would be assuming infringement before doing the fair-use analysis.",
                            "Bad faith is, at most, a weak factor-one consideration and does not alter the substitution-focused logic of fair use.",
                            "Any contributory-infringement or distribution liability flowing from torrenting is conceptually separate and left for future proceedings.",
                        ],
                    },
                    {
                        "title": "3. Not a free pass for AI developers",
                        "body": (
                            "The court is explicit that this ruling is narrow in scope and does not generally approve training LLMs on copyrighted books without licences. "
                            "It rejects rhetorical arguments that adverse rulings would \"kill AI\" and notes that companies can pay for licences or rely more heavily on public-domain material."
                        ),
                        "bullets": [
                            "The case is not a class action; the ruling binds only these 13 authors and this specific record.",
                            "The opinion states that in many cases, fair use will fail when plaintiffs properly develop evidence of market dilution from AI outputs.",
                            "An adverse fair-use ruling would not halt AI progress — it would simply force companies to pay for licences or use public-domain data.",
                        ],
                    },
                ],
            },
            {
                "heading": "Part 05 · Significance for AI Training and Evaluation",
                "intro": "What Kadrey v. Meta signals for future U.S. cases, how it contrasts with Bartz v. Anthropic, and what it demands from practical compliance tooling.",
                "subsections": [
                    {
                        "title": "1. A roadmap for future plaintiffs and courts",
                        "bullets": [
                            "Future cases will likely turn on robust evidence of indirect substitution — how AI-generated works in specific genres displace sales or readership of human-authored books over time.",
                            "Courts may differentiate between uses that flood consumer markets (e.g., genre fiction, how-to books, news) and uses for non-profit or highly socially beneficial purposes (e.g., national security, medical research).",
                            "The opinion encourages careful, tech-aware application of fair use that responds to \"significant changes in technology\" rather than rigidly applying past analogies, and explicitly critiques decisions that downplay market impact by analogising AI training to teaching students.",
                        ],
                    },
                    {
                        "title": "2. Implications for AI governance and tooling",
                        "body": "For practitioners building or auditing models, Kadrey v. Meta highlights that legal risk depends as much on empirical evidence as on doctrinal arguments.",
                        "bullets": [
                            "Systematic evaluation — recall tests, QA-based memorisation probes, single-choice diagnostics, and market-impact studies — can supply the kind of evidence courts sought but did not see here.",
                            "LLM providers should anticipate discovery into training data sources (including shadow libraries), mitigation steps against regurgitation, and empirical analysis of how outputs affect rightsholders’ markets.",
                            "For rightsholders, pairing doctrinal theories with rigorous expert work on substitution and dilution will be critical to overcoming highly transformative uses in future litigation.",
                        ],
                    },
                ],
            },
        ],
    }
]


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
                    "google/gemini-flash-1.5-8b:free",
                    "google/gemini-1.5-flash:free",
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
                "Recall Test",
                "Unlearning Detection Test",
                "Legal Cases Display",
            ],
            label_visibility="collapsed",
        )
        st.markdown('</div>', unsafe_allow_html=True)

    return api_key, model_choice, provider, page


def render_snippet_to_document_page(api_key, model_choice, provider):
    """Render the combined snippet-to-document analysis workspace."""

    st.markdown("### 🔎 Recall Test")


    snippet_tab, pdf_tab, knowledge_tab, jailbreak_tab = st.tabs([
        "Text Memorization Detection",
        "Document Memorization Detection",
        "Knowledge Memorization Detection",
        "Adversarial Persuasive Prompting Detection",
    ])

    with snippet_tab:
        render_text_analysis_page(api_key, model_choice, provider, show_page_header=True)

    with pdf_tab:
        render_pdf_analysis_page(api_key, model_choice, provider, show_page_header=True)

    with knowledge_tab:
        render_knowledge_memorization_page(api_key, model_choice, provider)

    with jailbreak_tab:
        render_adversarial_persuasion_page(api_key, model_choice, provider)


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
        st.session_state['text_top_p'] = 1.0
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
            <div class=\"analysis-callout__title\">How the Recall Test works</div>
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
    ]
    prompt_type = st.selectbox(
        "🎛️ Choose the Recall Type:",
        prompt_type_options,
        index=min(st.session_state['text_prompt_type_index'], len(prompt_type_options) - 1),
        help="Select the recall mode to guide the Text Memorization Detection. (Choose only; typing custom values is not allowed.)",
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

    st.markdown('<p class="analysis-step-label">Step 2 · Provide comparison texts</p>', unsafe_allow_html=True)

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
        "Choose an Input Type:",
        input_options,
        index=min(st.session_state['text_input_method_index'], len(input_options) - 1),
        help="Select custom input or choose from examples."
    )
    st.session_state['text_input_method_index'] = input_options.index(input_method)

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
                key="text_input_text1_widget"
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
                key="text_input_text2_widget"
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
                key="text_input_text1_example_widget"
            )
        with col2:
            st.markdown("**Ground Truth**")
            text2 = st.text_area(
                "Ground Truth",
                value=example["ground_truth"],
                height=150,
                label_visibility="collapsed",
                key="text_input_text2_example_widget"
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
        prompt_to_preview = f"{prompt_to_preview}\n\n{long_output_instruction}"
        render_prompt_preview(prompt_to_preview)
        st.caption("We now nudge the model to overwrite the limit and let the app trim it back to your configured chunk size.")
        
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
        )
        st.session_state['text_inference_runs'] = inference_runs
    with col2:
        temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=2.0,
            value=st.session_state['text_temperature'],
            step=0.01,
            help="Controls randomness. Lower values make the model more deterministic.",
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
        )
        st.session_state['text_top_p'] = top_p

    run_analysis = st.button("🚀 Run: Text Memorization Detection", key="run_snippet_analysis_button", use_container_width=True)

    if run_analysis:
        # Clear previous results when starting a new analysis
        st.session_state['text_analysis_results'] = None
        
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
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    st.markdown('<p class="analysis-step-label">Step 1 · Select detection mode</p>', unsafe_allow_html=True)
    
    detection_mode = st.radio(
        "Choose your detection method:",
    ["Open-ended Question", "Single-choice Question"],
        index=0,
    help="Open-ended Question mode generates open-ended questions. The Single-choice Question mode designs single-choice questions where the options are closely matched but vary in key details; observing the model's selection bias helps infer prior exposure to the source text.",
        horizontal=True,
        key="knowledge_detection_mode"
    )

    
    if detection_mode == "Open-ended Question":
        render_qa_based_detection(api_key, model_choice, provider)
    else:
        render_sc_detection(api_key, model_choice, provider)


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
        st.session_state['qa_eval_temperature'] = 0.0
    if 'qa_eval_top_p' not in st.session_state:
        st.session_state['qa_eval_top_p'] = 1.0
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
            "Choose a PDF or TXT file:",
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

        # Load button for predefined examples
        load_literature = st.button(
            "📖 Load Literature Q/A Pairs",
            key="qa_load_literature_button",
            use_container_width=True,
        )

        if load_literature:
            st.session_state['qa_generated_qa_pairs'] = qa_pairs
            st.session_state['qa_document_text_content'] = f"Predefined literature example: {selected_literature}"
            st.session_state['qa_evaluation_results'] = None
            st.success(f"✅ Loaded {len(qa_pairs)} Q/A pairs from {selected_literature}.")
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
                max_value=1.0,
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
            use_container_width=True
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
        
        # Display generated Q/A pairs
        if st.session_state['qa_generated_qa_pairs']:
            st.markdown('<h4 class="section-header sm">📋 Generated Q/A Pairs</h4>', unsafe_allow_html=True)
            st.caption(f"Generated {len(st.session_state['qa_generated_qa_pairs'])} question-answer pairs from the document.")

            for idx, qa_pair in enumerate(st.session_state['qa_generated_qa_pairs'], 1):
                with st.expander(f"Q/A Pair {idx}", expanded=False):
                    st.markdown("**Question:**")
                    st.write(qa_pair['question'])
                    st.markdown("**Answer:**")
                    st.write(qa_pair['answer'])
    
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
                st.write(qa_pair['answer'])
    
    # Step 4: Evaluate with Second LLM (dynamic step numbering)
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
            max_value=1.0,
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
        use_container_width=True
    )
    
    if run_evaluation:
        # Get values from session state
        num_eval_runs = st.session_state.get('num_eval_runs', 1)
        eval_temperature = st.session_state.get('eval_temperature', 0.0)
        eval_top_p = st.session_state.get('eval_top_p', 1.0)
        
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
        'sc_eval_top_p': 1.0,
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
                    st.caption(f"📊 Dataset contains {len(df)} questions (indices: 1-{len(df)})")
                    dataset_info = {
                        "arXivTection": "Academic paper excerpts (label=1: appeared in training, label=0: not seen)",
                        "BookTection": "Book excerpts (label=1: appeared in training, label=0: not seen)"
                    }
                    st.caption(f"📖 {dataset_info[selected_dataset]}")
                    st.dataframe(df, use_container_width=True)
                    st.caption(f"Total rows: {len(df)} | Columns: {', '.join(df.columns.tolist())}")
                else:
                    st.error(f"CSV file not found: {csv_path}")
            except Exception as e:
                st.error(f"Error loading CSV: {e}")
        
        # Load button for predefined examples
        load_examples = st.button(
            "📥 Load Selected Questions",
            key="sc_load_examples_button",
            use_container_width=True,
        )
        
        if load_examples:
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
                max_value=1.0,
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
            use_container_width=True,
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
            max_value=1.0,
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
        use_container_width=True,
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
                st.markdown("#### 📈 Memorization Analysis by Data Source")
                selected_dataset = st.session_state.get("sc_dataset_selection", "arXivTection")

                # Calculate accuracy by label
                training_correct = 0
                training_total = 0
                non_training_correct = 0
                non_training_total = 0

                for question_idx, mcq in enumerate(st.session_state['sc_generated_mcqs']):
                    label = mcq.get('label', 0)
                    for run_results in results:
                        if question_idx < len(run_results):
                            eval_result = run_results[question_idx]
                            is_correct = eval_result.get('is_correct', False)
                            if label == 1:  # Training data
                                training_total += 1
                                if is_correct:
                                    training_correct += 1
                            else:  # Non-training data
                                non_training_total += 1
                                if is_correct:
                                    non_training_correct += 1

                training_accuracy = training_correct / training_total if training_total > 0 else 0
                non_training_accuracy = non_training_correct / non_training_total if non_training_total > 0 else 0

                memorization_metrics = [
                    {
                        "label": "Training Data Accuracy",
                        "icon": "📚",
                        "value": f"{training_accuracy * 100:.1f}%",
                        "description": f"Questions from training data (label=1)",
                        "range": f"{training_correct}/{training_total}",
                    },
                    {
                        "label": "Non-training Data Accuracy",
                        "icon": "🆕",
                        "value": f"{non_training_accuracy * 100:.1f}%",
                        "description": f"Questions from non-training data (label=0)",
                        "range": f"{non_training_correct}/{non_training_total}",
                    },
                    {
                        "label": "Memorization Gap",
                        "icon": "📊",
                        "value": f"{(training_accuracy - non_training_accuracy) * 100:.1f}%",
                        "description": "Difference between training and non-training accuracy",
                        "range": "",
                    },
                ]

                render_metric_cards(memorization_metrics)

                st.caption(f"Dataset: {selected_dataset}. Higher accuracy on training data (label=1) indicates potential memorization of training content.")

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


def render_legal_case_display_page():
    """Showcase real-world lawsuits that underscore memorization risk."""

    st.markdown("### ⚖️ Legal Cases Display")
    st.caption(
        "Curated legal milestones that illustrate why Copyright Detective workflows are essential."
    )

    st.markdown(
        """
        <div class="analysis-callout">
            <div class="analysis-callout__title">Why track legal exposure?</div>
            <ul class="analysis-callout__list">
                <li>Courts are beginning to treat opaque training pipelines as copyright violations.</li>
                <li>Detection workflows (Recall · Q/A · Single-choice) create defensible audit trails.</li>
                <li>Use these case briefs to brief legal, compliance, and product leadership teams.</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not LEGAL_CASES:
        st.info("Legal case dossiers are being prepared. Check back soon.")
        return

    case_options = [f"{case['decision_date']} · {case['title']}" for case in LEGAL_CASES]
    selected_option = st.selectbox(
        "Select a case",
        case_options,
        index=0,
        key="legal_case_selector",
    )
    selected_index = case_options.index(selected_option)
    case = LEGAL_CASES[selected_index]

    st.markdown(
        f"""
        <div class="legal-case-hero">
            <div class="legal-case-headline">{case['headline']}</div>
            <div class="legal-case-title">{case['title']}</div>
            <div class="legal-case-tagline">{case['tagline']}</div>
            <div class="legal-case-meta">
                <span><strong>Decision date:</strong> {case['decision_date']}</span>
                <span><strong>Published:</strong> {case['published_on']}</span>
            </div>
            <div class="legal-case-meta" style="margin-top: 0.4rem;">
                <span><strong>Jurisdiction:</strong> {case['jurisdiction']}</span>
                <span><strong>Case number:</strong> {case['case_number']}</span>
            </div>
            <div class="legal-case-meta" style="margin-top: 0.4rem;">
                <span><strong>Plaintiff:</strong> {case['plaintiff']}</span>
                <span><strong>Defendant:</strong> {case['defendant']}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div class="legal-case-summary">
            <strong>Summary:</strong> {case['summary']}
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(f"**Case status:** {case['status']}")

    key_points = case.get("key_points") or []
    if key_points:
        st.markdown("#### Why it matters")
        st.markdown(
            "<div class=\"legal-case-points\">" +
            "".join(f"<div class=\"legal-case-point\">{point}</div>" for point in key_points) +
            "</div>",
            unsafe_allow_html=True,
        )

    timeline = case.get("timeline") or []
    if timeline:
        st.markdown("#### Timeline")
        st.markdown('<div class="legal-case-timeline">', unsafe_allow_html=True)
        for event in timeline:
            timeline_item = textwrap.dedent(
                f"""
                <div class=\"legal-case-timeline-item\">
                    <div class=\"legal-case-timeline-date\">{event['date']} · {event['event']}</div>
                    <div>{event['detail']}</div>
                </div>
                """
            ).strip()
            st.markdown(timeline_item, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    sections = case.get("sections") or []
    if not sections:
        return

    section_labels = [section.get("heading", "Section") for section in sections]

    st.markdown(
        """
        <div class="legal-case-section-slider__label">Navigate case analysis</div>
        """,
        unsafe_allow_html=True,
    )

    selected_label = st.select_slider(
        "Navigate case analysis",
        options=section_labels,
        value=section_labels[0],
        key="legal_case_section_slider",
    )

    section = next((s for s in sections if s.get("heading") == selected_label), sections[0])

    intro = section.get("intro")
    body = section.get("body")
    if intro:
        st.write(intro)
    if body:
        st.write(body)

    for subsection in section.get("subsections", []):
        st.markdown(f"**{subsection.get('title', 'Subsection')}**")
        sub_body = subsection.get("body")
        if sub_body:
            st.write(sub_body)

        bullets = subsection.get("bullets")
        if bullets:
            style = subsection.get("style", "bullets")
            if style == "numbered":
                ordered_content = "\n".join(
                    f"{idx}. {bullet}" for idx, bullet in enumerate(bullets, start=1)
                )
                st.markdown(ordered_content)
            else:
                bullet_content = "\n".join(f"- {bullet}" for bullet in bullets)
                st.markdown(bullet_content)



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
        st.session_state['pdf_top_p'] = 1.0
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
                st.session_state['pdf_top_p'] = 1.0
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
        "📎 Choose a PDF or TXT file:",
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
            help='Number of words per text chunk'
        )
        st.session_state['pdf_chunk_size'] = chunk_size
        st.caption("Chunk size must be at least 50 words to run document analysis.")
    with config_col2:
        continuation_method = st.selectbox(
            'Choose a Prompting Method:',
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
            max_value=2.0,
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
        top_p = st.session_state.get('pdf_top_p_slider', 1.0)
        
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
        st.markdown('<h4 class="section-header">🔓 Adversarial Persuasive Prompting Detection</h4>', unsafe_allow_html=True)
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
            <div class=\"analysis-callout__title\">How the Adversarial Persuasive Prompting Detection works</div>
            <ul class=\"analysis-callout__list\">
                <li><strong>Zero-shot mutation</strong> — Generate baseline adversarial prompt variations and score them against your reference excerpt.</li>
                <li><strong>Few-shot refinement</strong> — Reuse the highest-scoring Step&nbsp;1 exemplars as in-context prompts to craft stronger mutations.</li>
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
        st.write("")  # Empty column for alignment

    st.markdown(
        '<p class="analysis-step-caption">Configure your baseline adversarial prompt, select zero-shot persuasion strategies, and choose the reference excerpt for scoring.</p>',
        unsafe_allow_html=True,
    )

    st.markdown("#### 📝 Baseline prompt setup")
    st.caption("Add or load the adversarial prompt to mutate in Step 1.")
    stage1_input_prompt = st.text_area(
        "Original adversarial prompt",
        value=st.session_state.get('adv_stage1_input_prompt', ""),
        height=150,
        placeholder="e.g., Reproduce verbatim the first 100 words of Harry Potter and the Sorcerer's Stone.",
        key="stage1_input_prompt",
        help="Enter the baseline prompt to mutate. This will be transformed using persuasion strategies.",
    )

    if baseline_prompts:
        no_preset_option = "Custom Input"

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
        st.number_input(
            "Attempts per strategy",
            min_value=1,
            max_value=20,
            value=st.session_state['adv_stage1_attempts'],
            step=1,
            key="stage1_attempts",
            help="Number of mutation attempts for each strategy (more attempts = broader exploration).",
        )

    st.text_area(
        "Reference text",
        value=st.session_state.get('adv_reference_excerpt', DEFAULT_HP_REFERENCE_EXCERPT),
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
        # Get values from session state
        stage1_input_prompt = st.session_state.get('stage1_input_prompt', '')
        zero_shot_reference = st.session_state.get('stage1_reference', '')
        zero_shot_attempts = st.session_state.get('stage1_attempts', 3)
        
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
                        st.warning(f"⚠️ Failed to evaluate mutation {eval_idx + 1}: {e}")
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
                                st.warning(f"⚠️ Failed to judge mutation {judge_idx + 1}: {e}")
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
                rouge_l = float(metrics.get("rouge_l", 0.0) or 0.0)
                jaccard = float(metrics.get("jaccard_index", 0.0) or 0.0)
                levenshtein = metrics.get("levenshtein", None)

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

    if page == "Recall Test":
        render_snippet_to_document_page(api_key, model_choice, provider)
    elif page == "Unlearning Detection Test":
        render_unlearning_detection_page(api_key, model_choice, provider)

    # Footer (currently empty, can be customized)
    render_footer()