"""
Single-choice Question Detection Module

This module provides the UI for single-choice question-based knowledge memorization detection.
"""

import textwrap
from collections import Counter
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import streamlit as st

from src.pages.sampling_controls import render_temperature_top_p
from src.direct_recall import (
    generate_single_choice_questions_from_document_fragments,
    generate_single_choice_questions_from_fragments,
    parse_question_indices,
    run_single_choice_evaluation,
    summarize_single_choice_results,
)
from src.direct_recall.single_choice import load_predefined_examples, shuffle_tections_dataframe
from src.pdf_preview import (
    generate_single_choice_question_pdf_report,
    render_pdf_preview_with_blob,
)
from src.job_guard import detection_job, render_run_button, wd
from src.upload_cache import resolve_uploaded_file

SC_UPLOAD_CACHE_KEY = "sc_cached_upload"


def render_single_choice_detection_page(api_key, model_choice, provider):
    """Render Single-choice question test for copyright detection."""

    default_state = {
        'sc_source_mode': 'Predefined Examples',
        'sc_generated_mcqs': [],
        'sc_document_text': '',
        'sc_input_text': '',
        'sc_dataset_document': None,
        'sc_num_questions': 5,
        'sc_gen_temperature': 0.7,
        'sc_gen_top_p': 0.9,
        'sc_gen_provider_index': 0,
        'sc_evaluation_results': None,
        'sc_eval_runs': 1,
        'sc_eval_temperature': 0.7,
        'sc_eval_top_p': 0.9,
    }
    for key, value in default_state.items():
        if key not in st.session_state:
            st.session_state[key] = value

    # Step 2: Provide source content
    st.markdown('<p class="analysis-step-label">Step 2 · Provide source content</p>', unsafe_allow_html=True)
    
    # Create options for custom input or predefined examples
    custom_options = ["Predefined Examples", "Input Text", "Upload Document"]
    
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
        uploaded_document = resolve_uploaded_file(SC_UPLOAD_CACHE_KEY, uploaded_document)
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
                csv_path = Path("src/direct_recall/decop/data") / f"{selected_dataset}.csv"
                if csv_path.exists():
                    df = pd.read_csv(csv_path)
                    # Raw DE-COP files keep the verbatim passage in Example_A
                    # (Answer=A). Preview a shuffled copy so letter positions are
                    # not stuck on A before evaluation.
                    preview_seed = abs(hash(f"preview:{selected_dataset}")) % (2**31)
                    preview_df = shuffle_tections_dataframe(df, seed=preview_seed)
                    preview_df.index = range(1, len(preview_df) + 1)
                    st.caption(
                        f"📊 Dataset contains {len(preview_df)} questions "
                        f"(indices: 1-{len(preview_df)}). Options below are "
                        "shuffled for preview; Load Selected Questions reshuffles "
                        "again for evaluation."
                    )
                    dataset_info = {
                        "arXivTection": "Academic paper excerpts for single-choice probing",
                        "BookTection": "Book excerpts for single-choice probing",
                    }
                    st.caption(f"📖 {dataset_info[selected_dataset]}")
                    display_df = preview_df.rename(
                        columns={
                            column: f"Option_{column[len('Example_'):]}"
                            if str(column).startswith("Example_")
                            else column
                            for column in preview_df.columns
                        }
                    )
                    # Membership Label is unused in this experiment UI.
                    display_df = display_df.drop(
                        columns=[col for col in ("Label", "label") if col in display_df.columns]
                    )
                    answer_counts = (
                        display_df["Answer"]
                        .astype(str)
                        .str.strip()
                        .str.upper()
                        .value_counts()
                        .to_dict()
                        if "Answer" in display_df.columns
                        else {}
                    )
                    st.caption(
                        "Shuffled correct-letter counts in this preview: "
                        + ", ".join(
                            f"{letter}={answer_counts.get(letter, 0)}"
                            for letter in ("A", "B", "C", "D")
                        )
                    )
                    st.dataframe(display_df, width='stretch')
                    st.caption(
                        f"Total rows: {len(display_df)} | Columns: "
                        f"{', '.join(display_df.columns.tolist())}"
                    )
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
                    letter_counts = Counter(
                        str(mcq.get("correct_option") or "").upper()[:1]
                        for mcq in generated_mcqs
                    )
                    distribution = ", ".join(
                        f"{letter}={letter_counts.get(letter, 0)}"
                        for letter in ("A", "B", "C", "D")
                    )
                    st.success(
                        f"✅ Loaded {len(generated_mcqs)} predefined single-choice questions "
                        f"from {selected_dataset}. Options were shuffled "
                        f"(correct-letter distribution: {distribution})."
                    )
                else:
                    st.error(f"❌ No questions found for the specified indices in {selected_dataset}.")
            except Exception as exc:
                st.error(f"❌ Failed to load predefined examples: {exc}")

    # Step 3: Configure generation model and parameters (only for custom input)
    if source_mode in ["Input Text", "Upload Document"]:
        st.markdown('<p class="analysis-step-label">Step 3 · Configure text fragment extraction and distractor generation</p>', unsafe_allow_html=True)
        st.markdown(
            f'<p class="analysis-step-caption">Using the target model (<strong>{model_choice}</strong>) for text fragment extraction and distractor generation. Configure generation parameters below.</p>',
            unsafe_allow_html=True,
        )
        
        # Use the target model from sidebar for generation
        generation_provider = provider
        generation_model = model_choice
        generation_api_key = api_key

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
        render_temperature_top_p(
            temp_session_key='sc_gen_temperature',
            top_p_session_key='sc_gen_top_p',
            default_temp=0.7,
            default_top_p=0.9,
            temp_step=0.01,
            top_p_step=0.01,
            slider_key_prefix="sc_gen_",
            col_temp=col_temp,
            col_top_p=col_top_p,
        )

        generate_questions = render_run_button(
            "Single-Choice Question Generation",
            "sc_generate_mcq_button",
            "❓ Run: Generate single-choice questions",
        )
    else:
        # For predefined examples, skip generation and go directly to evaluation
        generate_questions = False

    if generate_questions:
        effective_api_key = generation_api_key or api_key
        if not effective_api_key:
            st.error("⚠️ Please provide an API key for question generation.")
        else:
            with detection_job("Single-Choice Question Generation"):
                num_questions = st.session_state['sc_num_questions']
                num_distractors = st.session_state['sc_num_distractors']
                progress_bar = st.progress(0, text=f"🔄 Starting question generation with target model ({model_choice})...")

                def update_generation_progress(current, total, question_num):
                    pct = current / total if total else 0
                    progress_bar.progress(
                        pct,
                        text=f"🔄 Generating question {question_num}/{num_questions} | Creating distractors... ({current}/{total})"
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

                    if not generated_mcqs:
                        st.error("❌ Failed to generate single-choice questions. Try adjusting the model or prompt parameters.")
                    else:
                        st.session_state['sc_generated_mcqs'] = generated_mcqs
                        st.session_state['sc_document_text'] = document_text
                        st.session_state['sc_evaluation_results'] = None
                        st.success(f"✅ Generated {len(generated_mcqs)} single-choice questions.")

                except Exception as exc:
                    progress_bar.empty()
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
                if source_mode == "Predefined Examples":
                    st.caption(
                        f"Correct option after shuffle: {mcq.get('correct_option')} "
                        f"(raw DE-COP CSV answer was {mcq.get('original_correct_option', 'A')})"
                    )
                if mcq.get('explanation'):
                    st.caption(f"Rationale: {mcq['explanation']}")
                if source_mode == "Predefined Examples" and mcq.get("original_id"):
                    st.caption(f"Original ID: {mcq['original_id']}")

    # Step 4: Evaluate with target model
    step_label = "Step 4" if source_mode in ["Input Text", "Upload Document"] else "Step 3"
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
            max_value=500,
            step=1,
            key="sc_eval_runs",
            help="How many times to evaluate each question. Maximum 500.",
        )
    render_temperature_top_p(
        temp_session_key='sc_eval_temperature',
        top_p_session_key='sc_eval_top_p',
        default_temp=0.7,
        default_top_p=0.9,
        temp_step=0.01,
        top_p_step=0.01,
        slider_key_prefix="sc_eval_",
        col_temp=eval_cols[1],
        col_top_p=eval_cols[2],
    )

    run_single_choice_eval = render_run_button(
        "Single-Choice Evaluation",
        "sc_run_eval_button",
        "📋 Run: Single-Choice Evaluation",
    )

    if run_single_choice_eval:
        if not st.session_state['sc_generated_mcqs']:
            st.warning("⚠️ Generate single-choice questions before running the evaluation.")
        elif not api_key or not api_key.strip():
            st.error(f"⚠️ Configure an API key for {provider} in the sidebar.")
        elif not model_choice:
            st.error("⚠️ Select a target model in the sidebar before running evaluation.")
        else:
            with detection_job("Single-Choice Evaluation"):
                total_questions = len(st.session_state['sc_generated_mcqs'])
                total_items = total_questions * st.session_state['sc_eval_runs']
                progress_bar = st.progress(0, text="🔄 Starting single-choice evaluation...")

                def update_progress(current, total, run_num, question_num, question_total):
                    pct = current / total if total else 0
                    progress_bar.progress(
                        pct,
                        text=f"🔄 Run {run_num}/{st.session_state['sc_eval_runs']} | Question {question_num}/{question_total} | {current}/{total} evaluations"
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
                    if not results:
                        st.error("❌ Evaluation returned no results. Please try again.")
                    else:
                        st.session_state['sc_evaluation_results'] = results
                        st.success(f"✅ Completed {total_items} single-choice evaluations.")
                except Exception as exc:  # noqa: BLE001
                    progress_bar.empty()
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

            per_question = metrics.get('per_question', [])
            if per_question:
                with st.expander("📊 Question-level accuracy", expanded=False):
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
                    st.dataframe(per_question_df, hide_index=True, width='stretch')

        st.markdown('<h3 class="section-header sm">📝 Detailed responses</h3>', unsafe_allow_html=True)
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

        # Display memorization risk assessment at the end (outside the loop)
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

        # PDF Report Section
        st.markdown("---")

        # Prepare data for PDF report
        pdf_data = {
            'results': results,
            'metrics': metrics,
            'generated_mcqs': st.session_state.get('sc_generated_mcqs', []),
            'document_text': st.session_state.get('sc_document_text', '')
        }

        # Generate PDF report
        pdf_bytes = generate_single_choice_question_pdf_report(pdf_data, model_choice, provider, source_mode)

        # PDF Preview
        render_pdf_preview_with_blob(pdf_bytes, title="📋 Audit Report Preview", iframe_height=450)

