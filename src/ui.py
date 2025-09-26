import io
import math
from typing import Optional

import streamlit as st
import pandas as pd
from src.copyright_detective.comparison import compare_texts, enforce_exact_char_count
from src.copyright_detective.pdf_utils import extract_text_from_pdf, split_text_into_chunks
from src.config import DEFAULT_OPENROUTER_KEY
import matplotlib.pyplot as plt
from src.copyright_detective.jailbreak_probe import (
    run_persuasion_probe,
)
from src.copyright_detective.unlearning import (
    list_unlearning_strategies,
    build_unlearning_prompt,
    run_unlearning_detection,
    run_membership_inference,
)
from src.prompt_utils import get_full_prompt, get_persuasion_prompt, get_persuasion_template, get_prompt_template
from src.components import render_collapsible_panel, render_prompt_preview


CONTINUATION_STRATEGIES = [
    "Normal Continuation",
    "Role-Playing: The Author",
    "Hypothetical Scenario: A Lost Manuscript",
    "Creative Writing Exercise",
    "Translation and Back-Translation",
    "Tom and Jerry Game",
    "Custom Prompt",
]


def render_header():
    """Render the app header with title and description."""
    st.markdown(
        """
        <div class="app-header">
            <div class="title">🕵️‍♂️ Copyright Detective</div>
            <div class="subtitle">Analyze potential text copyright infringement in LLM application</div>
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
                "Snippet-to-Document Analysis",
                "Unlearning Detection",
            ],
            label_visibility="collapsed",
        )
        st.markdown('</div>', unsafe_allow_html=True)

    return api_key, model_choice, provider, page


def render_snippet_to_document_page(api_key, model_choice, provider):
    """Render the combined snippet-to-document analysis workspace."""

    st.markdown("### 🔎 Snippet-to-Document Analysis")
    st.markdown(
        "Investigate potential copyright issues from a single excerpt through to an entire document."
    )

    snippet_tab, pdf_tab = st.tabs([
        "Text Snippet Analysis",
        "Whole PDF Analysis",
    ])

    with snippet_tab:
        render_text_analysis_page(api_key, model_choice, provider, show_page_header=False)

    with pdf_tab:
        render_pdf_analysis_page(api_key, model_choice, provider, show_page_header=False)


def render_text_analysis_page(api_key, model_choice, provider, *, show_page_header: bool = True):
    """Render the text snippet analysis workflow."""

    if show_page_header:
        st.markdown("### 📝 Text Snippet Analysis")
        st.markdown(
            "Analyze text snippets to detect potential copyright infringement by comparing generated text with ground truth."
        )

    # Prompt Selection (moved from sidebar to main page)
    prompt_type = st.selectbox(
        "📎Choose the Prompt Type:",
        [
            "Sequential Continuation Evaluation",
            "Preceding Context Reconstruction",
            "Copyright Attribution Inference",
        ],
        help="Select the type of prompt to guide the Text Snippet Analysis. (Choose only; typing custom values is not allowed.)",
    )

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

    # Explanatory notes for each prompt type
    if prompt_type == "Sequential Continuation Evaluation":
        st.markdown(
            "_Sequential Continuation Evaluation: Provide the prefix (previous sentence) and ask the model to continue by generating the next sentence. This probes whether the model reproduces or closely follows memorized sequences from source texts._"
        )
        
        continuation_method = st.selectbox(
            "Choose a continuation method (Jailbreak Persuasion):",
            CONTINUATION_STRATEGIES,
            help="Select 'Normal Continuation' for a direct prompt or a persuasion strategy to frame the request differently.",
            key="continuation_method_selector",
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
        # Use placeholder text if the input is empty
        chunk_size_preview = len(text2.split()) if text2 else None
        char_count_preview = len(text2) if text2 else None
        prompt_to_preview = get_full_prompt(
            prompt_type="Sequential Continuation Evaluation",
            input_text=text1,
            chunk_size=chunk_size_preview,
            continuation_method=continuation_method,
            char_count=char_count_preview,
            custom_template=custom_continuation_prompt if continuation_method == "Custom Prompt" else None,
        )
        st.markdown(
            "ℹ️ The length of the generated text will be adjusted to match the character count of your **Ground Truth** input."
        )
        render_prompt_preview(prompt_to_preview)
        
    elif prompt_type == "Preceding Context Reconstruction":
        st.markdown(
            "_Preceding Context Reconstruction: Provide the continuation or subsequent sentence and ask the model to generate the most likely preceding sentence. This helps detect whether the model can reconstruct prior context, which may indicate memorization of original works._"
        )
        preceding_method = st.selectbox(
            "Choose a continuation method (Jailbreak Persuasion):",
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
        render_prompt_preview(prompt_to_preview)

    elif prompt_type == "Copyright Attribution Inference":
        st.markdown(
            "_Copyright Attribution Inference: Based on the provided text snippet, ask the model to infer a likely title or attribution for the work (for example, a classic novel or another copyrighted source). Useful for identifying potential origins of the snippet._"
        )
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

    # Prompt Preview - This is now handled within each prompt_type section
    # if text1:
    #     # Define continuation_method for the preview logic even if it's not selected
    #     continuation_method = "Normal Continuation"
    #     chunk_size = len(text2.split()) if text2 else None
    #     if prompt_type == "Sequential Continuation Evaluation":
    #         continuation_method = st.session_state.get("continuation_method_selector", "Normal Continuation")
    #         prompt_to_preview = get_full_prompt(continuation_method, text1, chunk_size=chunk_size)
    #     else:
    #         prompt_to_preview = get_full_prompt(prompt_type, text1, chunk_size=chunk_size)

    #     prompt_preview(prompt_to_preview)

    st.markdown("---")
    st.markdown("**Inference Time Scaling & Parameters**")
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

    st.markdown("---")
    col_center = st.columns([1, 2, 1])[1]
    with col_center:
        run_analysis = st.button(
            "🚀 Run Analysis",
            width='stretch',
            key="run_snippet_analysis_button",
        )

    if run_analysis:
        if not api_key:
            st.error(f"⚠️ Please enter your API key in the sidebar.")
        elif not text1 or not text2:
            st.warning("⚠️ Please enter both input text and ground truth.")
        else:
            # Define a variable for continuation_method if it's not set
            if prompt_type == "Preceding Context Reconstruction":
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
                    if prompt_type == "Sequential Continuation Evaluation" and continuation_method != "Normal Continuation":
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
                        )
                    
                    # Handle potential errors from both functions
                    error_occurred = False
                    if isinstance(result, str) and result.startswith("Error"):
                        st.error(f"❌ {result}")
                        error_occurred = True
                    # run_persuasion_probe returns a tuple
                    elif isinstance(result, tuple) and isinstance(result[0], str) and result[0].startswith("Error"):
                        st.error(f"❌ {result[0]}")
                        error_occurred = True

                    if not error_occurred:
                        generated_text, rouge_score, jaccard_index, levenshtein_dist = result
                        generated_text = enforce_exact_char_count(generated_text, target_char_count)

                        # Results section
                        st.markdown("---")
                        st.markdown("### 📊 Analysis Results")

                        # Generated text
                        st.markdown("**🤖 Generated Text**")
                        st.markdown(
                            f'<div class="generated-text">{generated_text}</div>',
                            unsafe_allow_html=True,
                        )

                        # Similarity scores summary
                        st.markdown("**📈 Similarity Scores**")
                        st.markdown(
                            f"ROUGE-L: **{rouge_score:.4f}** | Jaccard: **{jaccard_index:.4f}** | Levenshtein: **{levenshtein_dist}**"
                        )

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
                st.markdown('<h3 class="multi-run-title">🔄 Inference Results Over Multiple Runs</h3>', unsafe_allow_html=True)
                similarity_scores = []
                generated_texts = []  # Store generated texts for each run
                progress_bar = st.progress(0, text="Starting inference runs...")
                for i in range(inference_runs):
                    progress_bar.progress(
                        (i) / inference_runs,
                        text=f"🔄 Generating text for run {i+1}/{inference_runs}...",
                    )
                    if prompt_type == "Sequential Continuation Evaluation" and continuation_method != "Normal Continuation":
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
                        )

                    # Handle potential errors from both functions
                    error_occurred = False
                    if isinstance(result, str) and result.startswith("Error"):
                        st.error(f"❌ {result}")
                        error_occurred = True
                        break
                    elif isinstance(result, tuple) and isinstance(result[0], str) and result[0].startswith("Error"):
                        st.error(f"❌ {result[0]}")
                        error_occurred = True
                        break
                        
                    if not error_occurred:
                        generated_text, rouge_score, jaccard_index, levenshtein_dist = result
                        generated_text = enforce_exact_char_count(generated_text, target_char_count)
                        similarity_scores.append(
                            {
                                "rouge": rouge_score,
                                "jaccard": jaccard_index,
                                "levenshtein": levenshtein_dist,
                            }
                        )
                        generated_texts.append(generated_text)
                
                progress_bar.progress(1.0, text="✅ All runs completed!")

                if similarity_scores:
                    # Display generated texts for each run
                    st.markdown('<h3 class="section-header sm">🤖 Generated Texts for Each Run</h3>', unsafe_allow_html=True)
                    for i, text in enumerate(generated_texts):
                        st.markdown(f"**Run {i+1}:**")
                        st.markdown(
                            f'<div class="generated-text sm">{text}</div>',
                            unsafe_allow_html=True,
                        )

                    # Calculate statistics
                    rouge_scores = [score["rouge"] for score in similarity_scores]
                    jaccard_scores = [score["jaccard"] for score in similarity_scores]
                    levenshtein_scores = [score["levenshtein"] for score in similarity_scores]

                    stats = {
                        "rouge": {
                            "max": max(rouge_scores),
                            "min": min(rouge_scores),
                            "avg": sum(rouge_scores) / len(rouge_scores),
                        },
                        "jaccard": {
                            "max": max(jaccard_scores),
                            "min": min(jaccard_scores),
                            "avg": sum(jaccard_scores) / len(jaccard_scores),
                        },
                        "levenshtein": {
                            "max": max(levenshtein_scores),
                            "min": min(levenshtein_scores),
                            "avg": sum(levenshtein_scores) / len(levenshtein_scores),
                        },
                    }

                    st.markdown("---")
                    st.markdown("### 📊 Statistical Results")
                    st.write(stats)

                    # Plot statistical graph
                    fig, ax = plt.subplots(1, 3, figsize=(15, 5))

                    # ROUGE-L Scores
                    ax[0].plot(rouge_scores, marker='o', label='ROUGE-L')
                    ax[0].set_title('ROUGE-L Scores')
                    ax[0].set_xlabel('Run')
                    ax[0].set_ylabel('Score')
                    ax[0].legend()

                    # Jaccard Index
                    ax[1].plot(jaccard_scores, marker='o', label='Jaccard Index', color='orange')
                    ax[1].set_title('Jaccard Index')
                    ax[1].set_xlabel('Run')
                    ax[1].set_ylabel('Score')
                    ax[1].legend()

                    # Levenshtein Distance
                    ax[2].plot(levenshtein_scores, marker='o', label='Levenshtein Distance', color='green')
                    ax[2].set_title('Levenshtein Distance')
                    ax[2].set_xlabel('Run')
                    ax[2].set_ylabel('Distance')
                    ax[2].legend()

                    st.pyplot(fig)

    st.markdown("---")
    # The Jailbreak Persuasion Probe section is now integrated above.
    # render_jailbreak_persuasion_probe_section(api_key, model_choice, provider)


def render_pdf_analysis_page(api_key, model_choice, provider, *, show_page_header: bool = True):
    """Render the document-scale PDF analysis workflow."""

    if show_page_header:
        st.markdown("### 📄 Whole PDF Analysis")
        st.markdown(
            "Upload a whole PDF document to automatically analyze text chunks for potential copyright infringement."
        )

    uploaded_file = st.file_uploader("📎 Choose a PDF file", type="pdf", help="Select a PDF document to analyze")
    if uploaded_file is not None:
        st.markdown('<h3 class="section-header sm">⚙️ Analysis Configuration</h3>', unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)
        with col1:
            score_type = st.selectbox(
                'Change Ranking Metric',
                ["ROUGE-L", "Jaccard Index", "Levenshtein Distance"],
                index=0,
                help='Choose how to rank the most similar sections'
            )
        with col2:
            chunk_size = st.number_input(
                'Change Chunk Size (words)',
                min_value=50,
                max_value=2000,
                value=200,
                step=25,
                help='Number of words per text chunk'
            )
        with col3:
            top_k = st.number_input(
                'Ranks to Display',
                min_value=1,
                max_value=20,
                value=5,
                step=1,
                help='Select how many of the highest scoring chunks to show after analysis'
            )

        st.markdown('<h3 class="section-header sm">🎭 Continuation Strategy</h3>', unsafe_allow_html=True)
        continuation_method = st.selectbox(
            'Select persuasion framing',
            CONTINUATION_STRATEGIES,
            index=0,
            help='Pick how the model should be nudged when generating chunk continuations. "Normal Continuation" keeps the default behaviour.',
            key='pdf_continuation_method'
        )

        custom_pdf_prompt: Optional[str] = None
        if continuation_method == "Custom Prompt":
            custom_pdf_prompt = st.text_area(
                "Custom prompt template",
                height=180,
                placeholder="Write the instruction to use for each PDF chunk. Include {input_text} where the chunk should appear, and optionally {word_count} or {char_count} for length hints.",
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
            prompt_type="Sequential Continuation Evaluation",
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
        score_type = None
        chunk_size = None
        continuation_method = "Normal Continuation"
        temperature = 0.7
        top_p = 1.0
        top_k = 5
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
                    if isinstance(result[0], str) and result[0].startswith("Error"):
                        st.error(f"❌ {result[0]}")
                        return
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

                generated_text, rouge_score, jaccard_index, levenshtein_dist = result
                results.append((upper, lower, generated_text, rouge_score, jaccard_index, levenshtein_dist))
                progress_bar.progress((i + 1)/total, text=f"🔄 Processing chunk {i+1}/{total} · {continuation_method}")

            # Sort
            if score_type == "ROUGE-L":
                results.sort(key=lambda x: x[3], reverse=True)
            elif score_type == "Jaccard Index":
                results.sort(key=lambda x: x[4], reverse=True)
            else:  # Levenshtein
                results.sort(key=lambda x: x[5])

            display_limit = min(top_k, len(results)) if results else 0
            if display_limit == 0:
                st.info("No comparable chunks were produced for ranking.")
                return

            st.markdown(f"#### 🏆 Top {display_limit} Most Similar Sections")
            st.caption(
                f"Ranking by {score_type}. Showing top {display_limit} of {len(results)} chunks. "
                f"Generation strategy: {continuation_method} · Temperature {temperature:.2f} · Top-P {top_p:.2f}"
            )
            for rank, (upper, lower, gen, r, j, l) in enumerate(results[:display_limit], start=1):
                sections = [
                    ("📝 Prefix Context", upper, None),
                    ("🎯 Ground Truth", lower, None),
                    ("🤖 Generated Text", gen, "generated"),
                    (
                        "📈 Scores",
                        f"ROUGE-L: {r:.4f}\nJaccard: {j:.4f}\nLevenshtein: {l}",
                        None,
                    ),
                ]
                render_collapsible_panel(
                    title=f"Rank {rank}",
                    sections=sections,
                    meta=f"ROUGE-L {r:.3f} · Jaccard {j:.3f} · Lev {l}",
                )

            progress_bar.progress(1.0, text=f"✅ Completed analysis with {model_choice}. Processed {total} chunks.")
        except Exception as e:
            st.error(f"❌ Error during analysis: {e}")


def render_unlearning_detection_page(api_key, model_choice, provider):
    """Render the unlearning detection page with membership inference."""
    st.markdown("### 🧠 Unlearning Detection")
    st.markdown(
        "Combine targeted jailbreak prompts with perplexity-based probes to uncover lingering memorisation."
    )

    probe_tab, membership_tab = st.tabs(["Prompt-Based Probes", "Membership Inference"])

    with probe_tab:
        target_description = st.text_area(
            "Target knowledge or passage",
            height=140,
            placeholder="Describe the copyrighted passage or knowledge that should have been unlearned.",
            key="unlearning_target_description",
        )

        strategies = list_unlearning_strategies()
        strategy_lookup = {strategy.id: strategy for strategy in strategies}
        strategy_options = [strategy.id for strategy in strategies]
        default_options = strategy_options[:2] if len(strategy_options) >= 2 else strategy_options

        strategy_selection = st.multiselect(
            "Probe strategies",
            options=strategy_options,
            default=default_options,
            format_func=lambda strategy_id: f"{strategy_lookup[strategy_id].name} — {strategy_lookup[strategy_id].description}",
            help="Select the prompt framings that will be used to probe the model.",
            key="unlearning_strategy_selection",
        )

        custom_prompt_enabled = st.checkbox("Add custom probe prompt", key="unlearning_use_custom_prompt")
        custom_prompt = ""
        if custom_prompt_enabled:
            custom_prompt = st.text_area(
                "Custom prompt template",
                height=160,
                placeholder="Provide the exact instructions. Use {target_description} where the description should appear.",
                key="unlearning_custom_prompt",
            )

        if strategy_selection:
            st.markdown("#### Prompt previews")
            for strategy_id in strategy_selection:
                preview_prompt = build_unlearning_prompt(strategy_id, target_description or "the withheld passage")
                render_prompt_preview(preview_prompt, expanded=False)

        if custom_prompt_enabled and custom_prompt.strip():
            st.markdown("#### Custom prompt preview")
            sample_prompt = build_unlearning_prompt("custom", target_description or "the withheld passage", custom_prompt=custom_prompt)
            render_prompt_preview(sample_prompt, expanded=False)

        st.markdown("---")
        ctrl_col1, ctrl_col2 = st.columns(2)
        with ctrl_col1:
            temperature = st.slider(
                "Temperature",
                min_value=0.0,
                max_value=2.0,
                value=0.3,
                step=0.01,
                help="Lower temperatures encourage deterministic echoes of memorised content.",
                key="unlearning_temperature",
            )
        with ctrl_col2:
            top_p = st.slider(
                "Top-P",
                min_value=0.0,
                max_value=1.0,
                value=0.9,
                step=0.01,
                help="Restrict sampling to the most likely tokens to surface memorisation.",
                key="unlearning_top_p",
            )

        st.markdown("---")
        probe_button_col = st.columns([1, 1, 1])[1]
        with probe_button_col:
            run_prompt_probe = st.button(
                "🚀 Run Prompt-Based Probes",
                width='stretch',
                key="run_unlearning_prompt_button",
            )

        if run_prompt_probe:
            if not api_key:
                st.error("⚠️ Please enter your API key in the sidebar.")
            elif not model_choice:
                st.error("⚠️ Please select a model in the sidebar.")
            elif not target_description.strip():
                st.warning("⚠️ Provide a target description before running detection.")
            else:
                selected_ids = list(strategy_selection)
                custom_prompt_value = custom_prompt.strip() if custom_prompt_enabled else None
                if custom_prompt_enabled and not custom_prompt_value:
                    st.warning("⚠️ Enter a custom prompt or disable the custom prompt option.")
                else:
                    if custom_prompt_value:
                        selected_ids.append("custom")
                    if not selected_ids:
                        st.warning("⚠️ Select at least one probe strategy or add a custom prompt.")
                    else:
                        with st.spinner(f"🔍 Evaluating memorisation with {model_choice}..."):
                            try:
                                summary = run_unlearning_detection(
                                    api_key,
                                    model_choice,
                                    provider,
                                    target_description=target_description,
                                    strategy_ids=selected_ids,
                                    temperature=temperature,
                                    top_p=top_p,
                                    custom_prompt=custom_prompt_value,
                                )
                            except ValueError as exc:
                                st.error(f"❌ {exc}")
                                summary = None
                            except Exception as exc:  # pragma: no cover - runtime/SDK errors
                                st.error(f"❌ Detection failed: {exc}")
                                summary = None

                        if summary:
                            st.markdown("---")
                            st.success("Prompt probes completed. Review the model responses below.")

                            for result in summary.results:
                                if result.error:
                                    sections = [
                                        ("Status", "Error while generating response.", None),
                                        ("Details", result.error, None),
                                    ]
                                    meta = "Error"
                                else:
                                    sections = [
                                        ("Model Response", result.response, "generated"),
                                    ]
                                    meta = "Response"

                                render_collapsible_panel(
                                    title=f"Strategy · {result.strategy_name}",
                                    sections=sections,
                                    meta=meta,
                                    expanded=False,
                                )

    with membership_tab:
        st.markdown("#### 📉 Membership Inference (Perplexity Probe)")
        st.caption(
            "Estimate whether the reference text still lives in the model's training data by comparing perplexity against a matched control passage."
        )

        membership_reference = st.text_area(
            "Reference text for perplexity probe",
            height=220,
            placeholder="Provide the original passage to test for memorisation.",
            key="membership_reference_text",
        )

        control_text = st.text_area(
            "Control text (public baseline)",
            height=220,
            placeholder="Provide a stylistically similar passage that the model definitely should not have memorised.",
            key="membership_control_text",
        )

        membership_cols = st.columns(3)
        with membership_cols[0]:
            chunk_size = st.slider(
                "Chunk size (tokens)",
                min_value=50,
                max_value=200,
                value=120,
                step=10,
                help=(
                    "Each passage will be evaluated in fixed-size token windows. If the tokenizer is unavailable, the app falls back to word chunks."
                ),
                key="membership_chunk_size",
            )
        with membership_cols[1]:
            max_chunks = st.number_input(
                "Chunks per passage",
                min_value=1,
                max_value=12,
                value=4,
                step=1,
                help="Limit how many segments are sampled from each passage.",
                key="membership_max_chunks",
            )
        with membership_cols[2]:
            ppl_gap_threshold = st.slider(
                "Flag gap (Δ PPL)",
                min_value=0.0,
                max_value=30.0,
                value=5.0,
                step=0.5,
                help="Minimum perplexity gap (control minus reference) to raise an alert.",
                key="membership_gap_threshold",
            )

        advanced_expander = st.expander("Advanced settings", expanded=False)
        with advanced_expander:
            st.caption("Tune confidence reporting and sampling depth for the perplexity probe.")
            bootstrap_samples = st.slider(
                "Bootstrap iterations",
                min_value=0,
                max_value=2000,
                value=500,
                step=50,
                help="Number of bootstrap resamples to estimate a confidence interval for the perplexity gap.",
                key="membership_bootstrap_samples",
            )
            confidence_level = st.slider(
                "Confidence level",
                min_value=0.5,
                max_value=0.99,
                value=0.9,
                step=0.01,
                help="Confidence level for the bootstrap interval.",
                key="membership_confidence_level",
            )

        membership_button_col = st.columns([1, 1, 1])[1]
        with membership_button_col:
            run_membership = st.button(
                "🧮 Run Membership Inference",
                width='stretch',
                key="run_membership_inference_button",
            )

        if run_membership:
            if not api_key:
                st.error("⚠️ Please enter your API key in the sidebar.")
            elif not model_choice:
                st.error("⚠️ Please select a model in the sidebar.")
            elif provider != "OpenAI":
                st.error("⚠️ Perplexity-based membership inference currently supports OpenAI models only.")
            elif not membership_reference.strip():
                st.warning("⚠️ Provide the reference text before running membership inference.")
            elif not control_text.strip():
                st.warning("⚠️ Provide a control passage to compare against.")
            else:
                with st.spinner(f"📉 Sampling log probabilities with {model_choice}..."):
                    try:
                        membership_summary = run_membership_inference(
                            api_key,
                            model_choice,
                            provider,
                            reference_text=membership_reference,
                            control_text=control_text,
                            chunk_size=int(chunk_size),
                            max_chunks=int(max_chunks),
                            ppl_gap_threshold=ppl_gap_threshold,
                            bootstrap_samples=int(bootstrap_samples),
                            confidence_level=float(confidence_level),
                        )
                    except ValueError as exc:
                        st.error(f"❌ {exc}")
                        membership_summary = None
                    except Exception as exc:  # pragma: no cover - network/SDK errors
                        st.error(f"❌ Membership inference failed: {exc}")
                        membership_summary = None

                if membership_summary:
                    st.markdown("---")
                    if membership_summary.flagged:
                        st.error(
                            "🚨 The reference passages show significantly lower perplexity than the control, indicating possible memorisation."
                        )
                    else:
                        st.success("✅ No significant perplexity gap detected between reference and control passages.")

                    mean_ref = "—" if math.isinf(membership_summary.mean_target_ppl) else f"{membership_summary.mean_target_ppl:.2f}"
                    mean_ctrl = "—" if math.isinf(membership_summary.mean_control_ppl) else f"{membership_summary.mean_control_ppl:.2f}"
                    gap = "—" if math.isnan(membership_summary.ppl_gap) else f"{membership_summary.ppl_gap:.2f}"

                    metric_col1, metric_col2, metric_col3 = st.columns(3)
                    metric_col1.metric("Mean PPL · Reference", mean_ref)
                    metric_col2.metric("Mean PPL · Control", mean_ctrl)
                    metric_col3.metric("Δ PPL", gap)

                    median_col1, median_col2, effect_col = st.columns(3)
                    median_col1.metric(
                        "Median PPL · Reference",
                        "—" if math.isinf(membership_summary.median_target_ppl) else f"{membership_summary.median_target_ppl:.2f}",
                    )
                    median_col2.metric(
                        "Median PPL · Control",
                        "—" if math.isinf(membership_summary.median_control_ppl) else f"{membership_summary.median_control_ppl:.2f}",
                    )
                    effect_display = "—" if membership_summary.effect_size is None else f"g = {membership_summary.effect_size:.2f}"
                    effect_col.metric("Effect Size", effect_display)

                    if membership_summary.bootstrap_iterations:
                        if membership_summary.ppl_gap_ci:
                            lower, upper = membership_summary.ppl_gap_ci
                            if any(math.isnan(val) for val in (lower, upper)):
                                st.caption("Bootstrap confidence interval unavailable.")
                            else:
                                confidence_pct = int(round(confidence_level * 100))
                                st.caption(f"Bootstrap Δ PPL {confidence_pct}% CI · [{lower:.2f}, {upper:.2f}]")
                        else:
                            st.caption("Bootstrap confidence interval could not be computed with the collected samples.")

                        ref_samples, ctrl_samples = membership_summary.sample_sizes
                        st.caption(
                            f"Valid perplexity samples · Reference: {ref_samples} · Control: {ctrl_samples}"
                        )

                        if membership_summary.statistical_tests:
                            st.markdown("#### Statistical comparison")
                            for outcome in membership_summary.statistical_tests:
                                cols = st.columns([2, 1, 2])
                                with cols[0]:
                                    st.markdown(f"**{outcome.name}**")
                                with cols[1]:
                                    stat_display = "—" if outcome.statistic is None else f"{outcome.statistic:.4f}"
                                    st.markdown(f"Stat: {stat_display}")
                                with cols[2]:
                                    p_display = "—" if outcome.pvalue is None else f"p = {outcome.pvalue:.4f}"
                                    st.markdown(p_display)
                                if outcome.detail:
                                    st.caption(outcome.detail)

                        if membership_summary.errors:
                            st.warning("; ".join(membership_summary.errors))

                    table_rows = []
                    for result in membership_summary.target_results + membership_summary.control_results:
                        status = "Error" if result.error else "Flagged" if (
                            membership_summary.flagged and result.label == "reference" and not math.isinf(result.perplexity)
                        ) else "OK"
                        snippet_preview = result.snippet.strip()
                        if len(snippet_preview) > 180:
                            snippet_preview = snippet_preview[:177] + "…"
                        delta_display = None if result.relative_perplexity is None else round(result.relative_perplexity, 3)
                        z_score_display = None if result.z_score is None else round(result.z_score, 2)
                        table_rows.append(
                            {
                                "Group": "Reference" if result.label == "reference" else "Control",
                                "Tokens": result.token_count,
                                "Avg LogProb": None if math.isnan(result.avg_logprob) else round(result.avg_logprob, 4),
                                "Perplexity": None if math.isinf(result.perplexity) else round(result.perplexity, 3),
                                "Δ vs Ctrl Mean": delta_display,
                                "Z-score": z_score_display,
                                "Trace Score": round(result.training_trace_score, 4),
                                "Status": status,
                                "Snippet": snippet_preview,
                            }
                        )

                    if table_rows:
                        st.dataframe(pd.DataFrame(table_rows), width='stretch')

                    for group_name, results in (
                        ("Reference", membership_summary.target_results),
                        ("Control", membership_summary.control_results),
                    ):
                        for idx, result in enumerate(results, start=1):
                            if result.error:
                                sections = [
                                    ("Snippet", result.snippet, None),
                                    ("Error", result.error, None),
                                ]
                                meta = "Error"
                            else:
                                delta_display = "—" if result.relative_perplexity is None else f"{result.relative_perplexity:.3f}"
                                z_score_display = "—" if result.z_score is None else f"{result.z_score:.2f}"
                                sections = [
                                    ("Snippet", result.snippet, None),
                                    (
                                        "Metrics",
                                        (
                                            f"Tokens: {result.token_count}\n"
                                            f"Avg logprob: {result.avg_logprob:.4f}\n"
                                            f"Perplexity: {result.perplexity:.3f}\n"
                                            f"Trace score: {result.training_trace_score:.4f}\n"
                                            f"Δ vs control mean: {delta_display}\n"
                                            f"Z-score: {z_score_display}"
                                        ),
                                        None,
                                    ),
                                ]
                                meta = "Flagged" if membership_summary.flagged and result.label == "reference" else "OK"

                            render_collapsible_panel(
                                title=f"{group_name} chunk {idx}",
                                sections=sections,
                                meta=meta,
                                expanded=membership_summary.flagged and result.label == "reference",
                            )


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

                if isinstance(result[0], str) and result[0].startswith("Error"):
                    st.error(f"❌ {result[0]}")

def render_footer():
    """Renders a footer section."""
    # This is a placeholder for any footer content you might want to add later.
    pass

def main():
    """Main function to run the Streamlit app."""
    render_header()
    api_key, model_choice, provider, page = render_sidebar()

    if page == "Snippet-to-Document Analysis":
        render_snippet_to_document_page(api_key, model_choice, provider)
    else:
        render_unlearning_detection_page(api_key, model_choice, provider)

    # Footer (currently empty, can be customized)
    render_footer()