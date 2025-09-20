import streamlit as st
from src.copyright_detective.comparison import compare_texts
from src.copyright_detective.pdf_utils import extract_text_from_pdf, split_text_into_chunks
from src.copyright_detective.jailbreak_probe import (
    ProbeConfig,
    available_strategies,
    run_probe_batch,
    list_templates,
    build_probe_prompt,
)
from src.config import DEFAULT_OPENROUTER_KEY
import matplotlib.pyplot as plt


def render_header():
    """Render the app header with title and description."""
    st.markdown(
        """
        <div class="app-header">
            <div class="title">🔍 Copyright Detective</div>
            <div class="subtitle">Analyze snippets or full PDFs for potential copyright overlap</div>
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
            model_choice = st.selectbox("Choose a model", ["gpt-3.5-turbo", "gpt-4o"])
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
                "Text Snippet Analysis",
                "Whole PDF Analysis",
            ],
            label_visibility="collapsed",
        )
        st.markdown('</div>', unsafe_allow_html=True)

    return api_key, model_choice, provider, page


def render_text_analysis_page(api_key, model_choice, provider):
    """Render the text snippet analysis page."""
    st.markdown("### 📝 Text Snippet Analysis")

    # Prompt Selection (moved from sidebar to main page)
    prompt_type = st.selectbox(
        "Choose the Prompt Type:",
        [
            "Sequential Continuation Evaluation",
            "Preceding Context Reconstruction",
            "Copyright Attribution Inference",
        ],
        help="Select the type of prompt to guide the Text Snippet Analysis. (Choose only; typing custom values is not allowed.)",
    )

    # Explanatory notes for each prompt type
    if prompt_type == "Sequential Continuation Evaluation":
        st.markdown(
            "_Sequential Continuation Evaluation: Provide the prefix (previous sentence) and ask the model to continue by generating the next sentence. This probes whether the model reproduces or closely follows memorized sequences from source texts._"
        )
    elif prompt_type == "Preceding Context Reconstruction":
        st.markdown(
            "_Preceding Context Reconstruction: Provide the continuation or subsequent sentence and ask the model to generate the most likely preceding sentence. This helps detect whether the model can reconstruct prior context, which may indicate memorization of original works._"
        )
    elif prompt_type == "Copyright Attribution Inference":
        st.markdown(
            "_Copyright Attribution Inference: Based on the provided text snippet, ask the model to infer a likely title or attribution for the work (for example, a classic novel or another copyrighted source). Useful for identifying potential origins of the snippet._"
        )

    st.markdown(
        "Analyze text snippets to detect potential copyright infringement by comparing generated text with ground truth."
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

    st.markdown("---")
    st.markdown("**Inference Time Scaling**")
    inference_runs = st.number_input(
        "Number of Inference Runs",
        min_value=1,
        max_value=100,
        value=1,
        step=1,
        help="Specify how many times to run the inference for statistical analysis.",
    )

    st.markdown("---")
    col_center = st.columns([1, 2, 1])[1]
    with col_center:
        run_analysis = st.button("🚀 Run Analysis", use_container_width=True)

    if run_analysis:
        if not api_key:
            st.error(f"⚠️ Please enter your API key in the sidebar.")
        elif not text1 or not text2:
            st.warning("⚠️ Please enter both prefix text and ground truth.")
        else:
            # Modify the analysis logic to incorporate the prompt type
            if prompt_type == "Sequential Continuation Evaluation":
                # Logic for continuing the next sentence
                pass
            elif prompt_type == "Preceding Context Reconstruction":
                # Logic for inferring the previous sentence
                pass
            elif prompt_type == "Copyright Attribution Inference":
                # Logic for generating the title of the work
                pass

            if inference_runs == 1:
                # Single run: Original Analysis Results
                with st.spinner(
                    f"🔄 Generating text with {model_choice} and calculating scores..."
                ):
                    result = compare_texts(
                        text1,
                        text2,
                        api_key,
                        model_name=model_choice,
                        provider=provider,
                        prompt_type=prompt_type,
                    )
                    if isinstance(result, str) and result.startswith("Error"):
                        st.error(f"❌ {result}")
                    else:
                        generated_text, rouge_score, jaccard_index, levenshtein_dist = result

                        # Results section
                        st.markdown("---")
                        st.markdown("### 📊 Analysis Results")

                        # Generated text
                        st.markdown("**🤖 Generated Text**")
                        st.markdown(
                            f'<div class="generated-text">{generated_text}</div>',
                            unsafe_allow_html=True,
                        )

                        # Similarity scores summary (boxes removed per request)
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
                    result = compare_texts(
                        text1,
                        text2,
                        api_key,
                        model_name=model_choice,
                        provider=provider,
                        prompt_type=prompt_type,
                    )
                    if isinstance(result, str) and result.startswith("Error"):
                        st.error(f"❌ {result}")
                        break
                    else:
                        generated_text, rouge_score, jaccard_index, levenshtein_dist = result
                        similarity_scores.append(
                            {
                                "rouge": rouge_score,
                                "jaccard": jaccard_index,
                                "levenshtein": levenshtein_dist,
                            }
                        )
                        generated_texts.append(generated_text)  # Append generated text
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

    # ------------------------------------------------------------------
    # Inline Jailbreak Persuasion Probe Section (migrated from standalone)
    # ------------------------------------------------------------------
    st.markdown("---")
    st.markdown("### 🧪 Jailbreak Persuasion Probe (Inline)")
    st.markdown(
        "This section evaluates whether crafted prompts can persuade the selected model to produce potentially copyright-violating content. "
        "Provide a research goal or apply a pre-defined template. For safety, you can redact model outputs (only a short prefix shown)."
    )

    mode = st.radio("Probe Mode", ["Strategy", "Template"], horizontal=True, key="jb_mode")

    col1, col2 = st.columns(2)
    if mode == "Strategy":
        with col1:
            strategy = st.selectbox("Probe Strategy", options=available_strategies(), key="jb_strategy")
            attempts = st.number_input("Attempts", min_value=1, max_value=50, value=3, step=1, key="jb_attempts")
            redact = st.checkbox(
                "Redact model outputs in UI",
                value=True,
                help="Show only a small prefix of outputs to avoid accidental exposure",
                key="jb_redact",
            )
        with col2:
            temperature = st.slider("Temperature (if supported)", 0.0, 1.5, 0.7, 0.1, key="jb_temp")
            dry_run = st.checkbox("Dry-run (no API calls)", value=False, key="jb_dry")

        goal = st.text_area(
            "Research Goal (what are we probing?)",
            placeholder="e.g., Attempt to get the model to restate a known copyrighted paragraph without attribution.",
            height=100,
            key="jb_goal",
        )
        seed = st.text_area(
            "Optional Seed/Context",
            placeholder="Provide seed context if needed (kept generic and research-oriented).",
            height=100,
            key="jb_seed",
        )
        template_id = None
        insertion_text = ""
    else:
        templates = list_templates()
        pattern_to_templates = {}
        for t in templates:
            pats = t.get("pattern") or ["(None)"]
            for p in pats:
                pattern_to_templates.setdefault(p, []).append(t)

        with st.expander("🔎 Template Filters", expanded=True):
            colf1, colf2 = st.columns([1, 1])
            with colf1:
                selected_group = st.selectbox(
                    "Primary Group (Pattern)",
                    options=["All"] + sorted(pattern_to_templates.keys()),
                    help="Filter by pattern tag group",
                    key="jb_group",
                )
            with colf2:
                search_kw = st.text_input(
                    "Secondary Search (ID / Name / Text)",
                    help="Enter keyword to perform secondary full-text filter (case-insensitive)",
                    placeholder="e.g. DAN / roleplay / internet",
                    key="jb_search",
                ).strip()

        filtered = []
        for t in templates:
            if selected_group != "All":
                pats = t.get("pattern") or []
                if selected_group not in pats:
                    continue
            if search_kw:
                blob = f"{t['id']} {t['name']} {t.get('text','')}".lower()
                if search_kw.lower() not in blob:
                    continue
            filtered.append(t)

        if not filtered:
            st.info("No templates matched current filters. Adjust criteria.")
            filtered = templates

        def label_for(t):
            pats = ", ".join(t.get("pattern") or [])
            return f"{t['id']} — {t['name']}" + (f"  [{pats}]" if pats else "")

        option_labels = [label_for(t) for t in filtered]
        selected_label = st.selectbox(
            "Template", options=option_labels, help="Select a template to apply insertion", key="jb_tpl"
        )
        selected_id = selected_label.split(" — ")[0]

        with st.expander("📄 模板原文 / Template Raw Text", expanded=False):
            tpl_obj = next((t for t in filtered if t["id"] == selected_id), None)
            if tpl_obj:
                st.code(tpl_obj.get("text", ""))

        insertion_text = st.text_area(
            "Insertion Text (replace [INSERT PROMPT HERE])",
            height=80,
            placeholder="Enter text that will replace the placeholder in the template",
            key="jb_insert",
        )
        attempts = st.number_input("Attempts", min_value=1, max_value=50, value=1, step=1, key="jb_attempts_tpl")
        redact = st.checkbox("Redact model outputs in UI", value=True, key="jb_redact_tpl")
        temperature = st.slider("Temperature (if supported)", 0.0, 1.5, 0.7, 0.1, key="jb_temp_tpl")
        dry_run = st.checkbox("Dry-run (no API calls)", value=False, key="jb_dry_tpl")
        strategy = available_strategies()[0]
        goal = "Template-driven probe"
        seed = ""
        template_id = selected_id

        preview_cfg = ProbeConfig(
            strategy=strategy,
            goal_description=goal,
            seed_context=seed,
            attempts=1,
            temperature=temperature,
            redact_outputs=redact,
            dry_run=True,
            template_id=template_id,
            insertion_text=insertion_text,
        )
        st.markdown("**Prompt Preview**")
        st.code(build_probe_prompt(preview_cfg))

    st.markdown("---")
    run_probe = st.button("▶ Run Probe", type="primary", key="jb_run")

    if run_probe:
        if not api_key:
            st.error("⚠️ Please enter your API key in the sidebar.")
        elif mode == "Strategy" and not goal.strip():
            st.warning("⚠️ Please describe the research goal.")
        else:
            cfg = ProbeConfig(
                strategy=strategy,
                goal_description=goal,
                seed_context=seed,
                attempts=attempts,
                temperature=temperature,
                redact_outputs=redact,
                dry_run=dry_run,
                template_id=template_id,
                insertion_text=insertion_text,
            )
            with st.spinner("Running probe attempts..."):
                results, error = run_probe_batch(cfg, api_key, model_choice, provider)
                if error:
                    st.error(f"❌ {error}")
                else:
                    risk_scores = [r.get("risk_score", 0) for r in results]
                    avg_risk = sum(risk_scores) / len(risk_scores) if risk_scores else 0
                    st.markdown("### 📊 Probe Summary")
                    st.metric("Average Risk Score", f"{avg_risk:.1f}/100")
                    st.markdown("### 🧾 Attempts")
                    for idx, r in enumerate(results, 1):
                        with st.expander(f"Attempt {idx} — Risk {r.get('risk_score', 0)}"):
                            st.markdown("**Prompt**")
                            st.code(r.get("prompt", ""))
                            if r.get("error"):
                                st.error(r["error"])
                            else:
                                st.markdown("**Model Response**")
                                st.write(r.get("response", ""))
                                st.caption(f"Raw response length: {r.get('raw_response_len', 0)} characters")


def render_pdf_analysis_page(api_key, model_choice, provider):
    """Render the whole PDF analysis page (restored)."""
    st.markdown("### 📄 Whole PDF Analysis")
    st.markdown(
        "Upload a whole PDF document to automatically analyze text chunks for potential copyright infringement."
    )

    uploaded_file = st.file_uploader("📎 Choose a PDF file", type="pdf", help="Select a PDF document to analyze")
    if uploaded_file is not None:
        st.markdown('<h3 class="section-header sm">⚙️ Analysis Configuration</h3>', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
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
                max_value=1000,
                value=200,
                step=25,
                help='Number of words per text chunk'
            )

        st.markdown('<h3 class="section-header sm">💡 Size Recommendations</h3>', unsafe_allow_html=True)
        st.markdown(
            """
            <div class="hint">
                <div style="margin-bottom: 0.5rem;"><strong>50-200:</strong> Precise analysis — detects specific phrases</div>
                <div style="margin-bottom: 0.5rem;"><strong>200-400:</strong> Balanced — general copyright detection</div>
                <div><strong>400-1000:</strong> Contextual — preserves broader context</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        score_type = None
        chunk_size = None

    if uploaded_file is not None:
        st.markdown("---")
        analyze_pdf = st.button("🔍 Analyze PDF", use_container_width=True, type="primary")
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
        try:
            progress_bar = st.progress(0, text=f"🔄 Analyzing PDF with {model_choice}... Preparing document...")
            pdf_text = extract_text_from_pdf(uploaded_file)
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
                generated_text, rouge_score, jaccard_index, levenshtein_dist = compare_texts(
                    upper, lower, api_key, model_name=model_choice, provider=provider, chunk_size=chunk_size
                )
                results.append((upper, lower, generated_text, rouge_score, jaccard_index, levenshtein_dist))
                progress_bar.progress((i + 1)/total, text=f"🔄 Processing chunk {i+1}/{total}")

            # Sort
            if score_type == "ROUGE-L":
                results.sort(key=lambda x: x[3], reverse=True)
            elif score_type == "Jaccard Index":
                results.sort(key=lambda x: x[4], reverse=True)
            else:  # Levenshtein
                results.sort(key=lambda x: x[5])

            st.markdown("### 🏆 Top 5 Most Similar Sections")
            for rank, (upper, lower, gen, r, j, l) in enumerate(results[:5], start=1):
                with st.expander(f"Rank {rank}"):
                    st.markdown("**📝 Prefix Context**")
                    st.write(upper)
                    st.markdown("**🎯 Ground Truth**")
                    st.write(lower)
                    st.markdown("**🤖 Generated Text**")
                    st.write(gen)
                    st.markdown(f"**Scores** — ROUGE-L: {r:.4f} | Jaccard: {j:.4f} | Levenshtein: {l}")

            progress_bar.progress(1.0, text=f"✅ Completed analysis with {model_choice}. Processed {total} chunks.")
        except Exception as e:
            st.error(f"❌ Error during analysis: {e}")




def render_footer():
    """Render a small, unobtrusive footer."""
    st.markdown(
        """
        <div class="app-footer">
            <div class="footer-left">© 2025 Copyright Detective</div>
            <div class="footer-right">
                <a href="https://github.com/changhu73/Copyright-Detective" target="_blank" rel="noopener">GitHub</a>
                <span>·</span>
                <a href="#" onclick="window.scrollTo({top: 0, behavior: 'smooth'}); return false;">Back to top</a>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )