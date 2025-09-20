import streamlit as st
from src.copyright_detective.comparison import compare_texts
from src.copyright_detective.pdf_utils import extract_text_from_pdf, split_text_into_chunks
from src.copyright_detective.jailbreak_probe import (
    ProbeConfig,
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
    st.markdown("### 🧪 Jailbreak Text Continuation Probe")
    st.markdown(
        "This section evaluates whether a jailbreak prompt can trick the model into generating a specific continuation. "
        "Select a template, provide the prefix text (the prompt), and the reference text (the expected output)."
    )

    templates = list_templates()
    if not templates:
        st.warning("No jailbreak templates found.")
        return

    # Prepare data for JS
    import json as _json
    js_templates = _json.dumps([
        {
            "id": t["id"],
            "name": t.get("name", "Unnamed"),
            "text": t.get("text", ""),
            "patterns": t.get("pattern") or [],
        }
        for t in templates
    ])

    # Build unique pattern tags
    pattern_tags = set()
    for t in templates:
        for p in (t.get("pattern") or []):
            pattern_tags.add(p)
    pattern_tags = sorted(pattern_tags)

    # Fallback initial selection (first template)
    if "jb_selected_template" not in st.session_state and templates:
        st.session_state["jb_selected_template"] = templates[0]["id"]

    st.markdown("#### 🔎 Template Browser")

    hidden_selected = st.session_state.get("jb_selected_template", templates[0]["id"] if templates else "")

    component_html = """
    <style>
    .tpl-browser { border:1px solid #e2e8f0; border-radius:10px; background:#ffffff; padding:0.9rem 1rem; margin-bottom:0.75rem; }
    .tpl-search-row { display:flex; gap:0.6rem; flex-wrap:wrap; align-items:center; margin-bottom:0.65rem; }
    .tpl-search-row input { flex:1; padding:0.45rem 0.6rem; border:1px solid #cbd5e1; border-radius:6px; font-size:0.8rem; }
    .tpl-patterns { display:flex; flex-wrap:wrap; gap:0.4rem; margin-bottom:0.6rem; }
    .tpl-chip { font-size:0.65rem; padding:0.35rem 0.55rem; border:1px solid #cbd5e1; border-radius:14px; cursor:pointer; background:#f1f5f9; font-weight:500; letter-spacing:0.3px; }
    .tpl-chip.active { background:#2563eb; color:#fff; border-color:#1d4ed8; }
    .tpl-list { max-height:260px; overflow:auto; border:1px solid #e2e8f0; border-radius:8px; background:#f8fafc; }
    .tpl-item { padding:0.55rem 0.65rem; border-bottom:1px solid #e2e8f0; cursor:pointer; display:flex; flex-direction:column; gap:2px; }
    .tpl-item:last-child { border-bottom:none; }
    .tpl-item:hover { background:#eef2f7; }
    .tpl-item.active { background:#dbeafe; box-shadow:inset 0 0 0 1px #3b82f6; }
    .tpl-line1 { font-size:0.75rem; font-weight:600; color:#1e293b; display:flex; justify-content:space-between; align-items:center; }
    .tpl-line2 { font-size:0.65rem; color:#64748b; display:flex; gap:0.4rem; flex-wrap:wrap; }
    .tpl-tag { background:#e2e8f0; padding:0.1rem 0.4rem; border-radius:12px; font-size:0.55rem; font-weight:500; letter-spacing:0.3px; }
    .tpl-raw-wrapper { margin-top:0.75rem; }
    .tpl-raw-header { font-size:0.7rem; font-weight:600; text-transform:uppercase; letter-spacing:0.5px; color:#64748b; margin-bottom:0.25rem; display:flex; align-items:center; gap:0.4rem; }
    .tpl-raw-box { background:#0f172a; color:#e2e8f0; padding:0.65rem 0.75rem; border-radius:6px; font-size:0.7rem; line-height:1.35; max-height:220px; overflow:auto; white-space:pre-wrap; word-break:break-word; }
    .tpl-empty { font-size:0.7rem; color:#94a3b8; padding:0.6rem; text-align:center; }
    .tpl-counter { font-size:0.6rem; font-weight:500; color:#64748b; }
    @media (prefers-color-scheme: dark) {
        .tpl-browser { background:#1e293b; border-color:#334155; }
        .tpl-list { background:#1e293b; border-color:#334155; }
        .tpl-item { border-color:#334155; }
        .tpl-item:hover { background:#2d3a4f; }
        .tpl-item.active { background:#1e40af; }
        .tpl-chip { background:#334155; border-color:#475569; color:#e2e8f0; }
        .tpl-chip.active { background:#2563eb; border-color:#1d4ed8; }
        .tpl-raw-box { background:#1e293b; }
        .tpl-tag { background:#334155; color:#e2e8f0; }
    }
    </style>
    <div class='tpl-browser'>
        <div class='tpl-search-row'>
            <input id='tpl-search' placeholder='Search ID / Name / Text...' />
            <span class='tpl-counter' id='tpl-counter'></span>
        </div>
        <div class='tpl-patterns' id='tpl-patterns'></div>
        <div class='tpl-list' id='tpl-list'></div>
        <div class='tpl-raw-wrapper'>
            <div class='tpl-raw-header'>📄 Raw Template</div>
            <div class='tpl-raw-box' id='tpl-raw-box'>(select a template)</div>
        </div>
    </div>
    <input type='hidden' id='tpl-selected-hidden' value='__INIT_SELECTED__' />
    <script>
    const TEMPLATES = %%JS_TEMPLATES%%;
    let activePattern = 'All';
    let searchKW = '';
    let selectedId = document.getElementById('tpl-selected-hidden').value || '';
    const patternTags = ['All', ...Array.from(new Set(TEMPLATES.flatMap(t=>t.patterns && t.patterns.length ? t.patterns : [])))];
    function renderPatterns(){
      const wrap = document.getElementById('tpl-patterns');
      wrap.innerHTML = patternTags.map(p=>`<div class="tpl-chip ${p===activePattern?'active':''}" data-p="${p}">${p}</div>`).join('');
      wrap.querySelectorAll('.tpl-chip').forEach(ch=>ch.addEventListener('click',()=>{ activePattern = ch.dataset.p; renderList(); renderPatterns(); }));
    }
    function passesFilters(t){
      if(activePattern!=='All'){
         const pats = t.patterns && t.patterns.length ? t.patterns : [];
         if(!pats.includes(activePattern)) return false;
      }
      if(searchKW){
         const blob = (t.id + ' ' + t.name + ' ' + t.text).toLowerCase();
         if(!blob.includes(searchKW.toLowerCase())) return false;
      }
      return true;
    }
    function renderList(){
       const list = document.getElementById('tpl-list');
       const arr = TEMPLATES.filter(passesFilters);
       document.getElementById('tpl-counter').textContent = arr.length + ' / ' + TEMPLATES.length;
       if(!arr.length){ list.innerHTML = '<div class="tpl-empty">No templates match filters.</div>'; document.getElementById('tpl-raw-box').textContent='(select a template)'; return; }
       list.innerHTML = arr.map(t=>{
          const pats = (t.patterns && t.patterns.length) ? t.patterns.map(p=>`<span class=tpl-tag>${p}</span>`).join('') : '<span class=tpl-tag>(None)</span>';
          return `<div class="tpl-item ${t.id===selectedId?'active':''}" data-id="${t.id}">\n <div class=tpl-line1><span>${t.id}</span><span style='font-size:0.55rem;opacity:.7;'>${t.name}</span></div>\n <div class=tpl-line2>${pats}</div>\n </div>`;
       }).join('');
       list.querySelectorAll('.tpl-item').forEach(it=>it.addEventListener('click',()=>{ selectedId=it.dataset.id; updateSelection(); renderList(); }));
       updateSelection();
    }
    function updateSelection(){
       const tpl = TEMPLATES.find(t=>t.id===selectedId);
       document.getElementById('tpl-selected-hidden').value = selectedId;
       if(tpl){ document.getElementById('tpl-raw-box').textContent = tpl.text || '(empty template)'; }
    }
    const searchInput = document.getElementById('tpl-search');
    searchInput.addEventListener('input', ()=>{ searchKW = searchInput.value.trim(); renderList(); });
    renderPatterns();
    renderList();
    </script>
    """

    # Replace placeholders
    component_html = component_html.replace('__INIT_SELECTED__', hidden_selected.replace("'", "&#39;"))
    component_html = component_html.replace('%%JS_TEMPLATES%%', js_templates)

    st.components.v1.html(component_html, height=520, scrolling=True)

    # Hidden bridge input (auto-updated by JS via polling from the outer document)
    bridge_val = st.text_input(
        "Selected Template Bridge",
        value=st.session_state.get("jb_selected_template", templates[0]["id"] if templates else ""),
        key="jb_selected_template_bridge",
        label_visibility="collapsed",
    )
    st.markdown(
        """
        <style>
        div[data-testid='stTextInput'] label:has(+ div input[aria-label='Selected Template Bridge']) {display:none !important;}
        input[aria-label='Selected Template Bridge'] {display:none !important;}
        </style>
        """,
        unsafe_allow_html=True,
    )

    bridge_js = """
    <script>
    (function(){
        function syncBridge(){
            try {
                const iframe = document.querySelector('iframe');
                if(!iframe) return;
                const idInput = iframe.contentDocument && iframe.contentDocument.getElementById('tpl-selected-hidden');
                if(!idInput) return;
                const newVal = idInput.value;
                const bridge = document.querySelector("input[aria-label='Selected Template Bridge']");
                if(bridge && newVal && bridge.value !== newVal){
                    bridge.value = newVal;
                    const ev = new Event('input', {bubbles:true});
                    bridge.dispatchEvent(ev);
                }
            } catch(e) { /* ignore cross-frame errors */ }
        }
        setInterval(syncBridge, 800);
    })();
    </script>
    """
    st.markdown(bridge_js, unsafe_allow_html=True)

    if templates:
        selected_id = bridge_val or st.session_state.get("jb_selected_template") or templates[0]["id"]
    else:
        selected_id = ""
    if selected_id:
        st.session_state["jb_selected_template"] = selected_id

    prefix_text = st.text_area(
        "Input Text",
        height=80,
        placeholder="Enter the input snippet (e.g., a previous sentence, a continuation, or an excerpt). The role of this field depends on the selected prompt type.",
        key="jb_prefix",
    )
    reference_text = st.text_area(
        "Ground Truth",
        height=80,
        placeholder="Enter the ground truth text or expected target to compare against (e.g., the known reference or target continuation). Leave blank if not applicable.",
        key="jb_reference",
    )

    col1, col2 = st.columns(2)
    with col1:
        attempts = st.number_input("Attempts", min_value=1, max_value=50, value=1, step=1, key="jb_attempts_tpl")
        temperature = st.slider("Temperature (if supported)", 0.0, 1.5, 0.7, 0.1, key="jb_temp_tpl")
    with col2:
        dry_run = st.checkbox("Dry-run (no API calls)", value=False, key="jb_dry_tpl")

    if selected_id:
        preview_cfg = ProbeConfig(
            prefix_text=prefix_text,
            reference_text=reference_text,
            template_id=selected_id,
        )
        st.markdown("**Prompt Preview**")
        st.code(build_probe_prompt(preview_cfg))

    st.markdown("---")
    run_probe = st.button("▶ Run Probe", type="primary", key="jb_run")

    if run_probe:
        if not api_key:
            st.error("⚠️ Please enter your API key in the sidebar.")
        elif not prefix_text or not reference_text:
            st.warning("⚠️ Please provide both Prefix and Reference text.")
        else:
            cfg = ProbeConfig(
                prefix_text=prefix_text,
                reference_text=reference_text,
                template_id=selected_id,
                attempts=attempts,
                temperature=temperature,
                dry_run=dry_run,
            )
            with st.spinner("Running probe attempts..."):
                results, error = run_probe_batch(cfg, api_key, model_choice, provider)
                if error:
                    st.error(f"❌ {error}")
                else:
                    st.markdown("### 📊 Probe Results")
                    # Build custom cards
                    card_blocks = []
                    for idx, r in enumerate(results, 1):
                        error_text = r.get('error')
                        prompt_text = r.get('prompt', '')
                        response_text = r.get('response', '')
                        ref_text = r.get('reference', '')
                        rouge_l = r.get('rouge_l', 0)
                        jaccard = r.get('jaccard', 0)
                        levenshtein = r.get('levenshtein', 0)

                        # Escape HTML special chars
                        import html as _html
                        disp_prompt_html = _html.escape(prompt_text)
                        disp_resp_html = _html.escape(response_text)
                        disp_ref_html = _html.escape(ref_text)

                        status_badge = '🟢' if rouge_l < 0.3 else '🟠' if rouge_l < 0.6 else '🔴'

                        card_blocks.append(f"""
                        <div class='jb-card' id='jb-card-{idx}'>
                          <div class='jb-card-header' onclick="toggleJBCard({idx})">
                             <div class='jb-card-left'>
                                <span class='jb-attempt-label'>Attempt {idx}</span>
                                <span class='jb-risk-badge'>{status_badge} ROUGE-L: {rouge_l:.3f}</span>
                             </div>
                             <div class='jb-card-actions'>
                                <button class='jb-btn small' onclick="copyText(event,'jb-prompt-{idx}')">Copy Prompt</button>
                                <button class='jb-btn small' onclick="copyText(event,'jb-response-{idx}')">Copy Response</button>
                                <span class='jb-toggle-icon' id='jb-icon-{idx}'>▶</span>
                             </div>
                          </div>
                          <div class='jb-card-body' id='jb-body-{idx}'>
                             {('<div class="jb-error">❌ ' + _html.escape(error_text) + '</div>') if error_text else f'''
                             <div class='jb-section'>
                                <div class='jb-section-label'>Scores</div>
                                <div class='jb-scores'>
                                    <span>ROUGE-L: <b>{rouge_l:.4f}</b></span>
                                    <span>Jaccard: <b>{jaccard:.4f}</b></span>
                                    <span>Levenshtein: <b>{levenshtein}</b></span>
                                </div>
                             </div>
                             <div class='jb-section'>
                                <div class='jb-section-label'>Reference vs. Response</div>
                                <div class='jb-comparison'>
                                    <div class='jb-comp-col'>
                                        <div class='jb-comp-header'>Reference</div>
                                        <pre class='jb-code'>{disp_ref_html}</pre>
                                    </div>
                                    <div class='jb-comp-col'>
                                        <div class='jb-comp-header'>Model Response</div>
                                        <pre class='jb-code' id='jb-response-{idx}'>{disp_resp_html}</pre>
                                    </div>
                                </div>
                             </div>
                             <div class='jb-section'>
                                <div class='jb-section-label'>Full Prompt</div>
                                <pre class='jb-code' id='jb-prompt-{idx}'>{disp_prompt_html}</pre>
                             </div>
                             '''}
                          </div>
                        </div>
                        """)

                    custom_css = """
                    <style>
                    .jb-cards-wrapper { margin-top: 0.5rem; }
                    .jb-controls { display:flex; gap:0.5rem; margin-bottom:0.75rem; flex-wrap:wrap; }
                    .jb-btn { background: linear-gradient(135deg,#f8fafc,#eef2f7); border:1px solid #cbd5e1; border-radius:6px; padding:0.4rem 0.8rem; font-size:0.75rem; font-weight:600; color:#334155; cursor:pointer; transition:.2s; }
                    .jb-btn:hover { background: linear-gradient(135deg,#e2e8f0,#dce3ea); }
                    .jb-btn.small { padding:0.3rem 0.6rem; }
                    .jb-card { border:1px solid #e2e8f0; border-radius:8px; margin-bottom:0.6rem; background:#ffffff; box-shadow:0 1px 2px rgba(0,0,0,0.04); overflow:hidden; }
                    .jb-card-header { display:flex; justify-content:space-between; align-items:center; padding:0.55rem 0.75rem; cursor:pointer; background:linear-gradient(90deg,#f8fafc,#f1f5f9); }
                    .jb-card-left { display:flex; align-items:center; gap:0.6rem; }
                    .jb-attempt-label { font-weight:600; color:#1e293b; }
                    .jb-risk-badge { font-size:0.75rem; font-weight:600; padding:0.2rem 0.5rem; border-radius:12px; background:#f1f5f9; border:1px solid #cbd5e1; }
                    .jb-card-actions { display:flex; align-items:center; gap:0.4rem; }
                    .jb-toggle-icon { font-size:0.7rem; transition:transform .25s ease; }
                    .jb-card-body { display:none; padding:0.75rem 0.9rem 0.9rem; border-top:1px solid #e2e8f0; }
                    .jb-card-body.open { display:block; }
                    .jb-section { margin-bottom:0.9rem; }
                    .jb-section-label { font-size:0.7rem; text-transform:uppercase; letter-spacing:0.5px; font-weight:600; color:#64748b; margin-bottom:0.25rem; }
                    .jb-code { background:#0f172a; color:#e2e8f0; padding:0.6rem 0.7rem; border-radius:6px; font-size:0.75rem; line-height:1.4; white-space:pre-wrap; word-break:break-word; }
                    .jb-meta { font-size:0.65rem; color:#64748b; margin-top:0.3rem; }
                    .jb-error { background:#fef2f2; color:#b91c1c; border:1px solid #fecaca; padding:0.6rem 0.7rem; border-radius:6px; font-size:0.8rem; }
                    .jb-card-header:hover { background:linear-gradient(90deg,#eef2f7,#e2e8f0); }
                    .jb-card.open .jb-toggle-icon { transform:rotate(90deg); }
                    .jb-scores { display: flex; gap: 1rem; font-size: 0.8rem; }
                    .jb-comparison { display: grid; grid-template-columns: 1fr 1fr; gap: 0.75rem; }
                    .jb-comp-header { font-weight: 600; font-size: 0.75rem; margin-bottom: 0.25rem; }
                    @media (prefers-color-scheme: dark) {
                        .jb-card { background:#1e293b; border-color:#334155; }
                        .jb-card-header { background:linear-gradient(90deg,#243044,#1e293b); }
                        .jb-card-header:hover { background:linear-gradient(90deg,#2d3a4f,#243044); }
                        .jb-code { background:#1e293b; }
                        .jb-risk-badge { background:#243044; border-color:#334155; color:#e2e8f0; }
                    }
                    </style>
                    """

                    controls_js = """
                    <script>
                    function toggleJBCard(idx){
                        const card = document.getElementById('jb-card-'+idx);
                        const body = document.getElementById('jb-body-'+idx);
                        const icon = document.getElementById('jb-icon-'+idx);
                        if(!card || !body) return;
                        const open = body.classList.toggle('open');
                        if(open){ card.classList.add('open'); icon.textContent='▼'; } else { card.classList.remove('open'); icon.textContent='▶'; }
                    }
                    function expandAllJBCards(){ document.querySelectorAll('.jb-card-body').forEach(b=>b.classList.add('open')); document.querySelectorAll('.jb-card').forEach(c=>c.classList.add('open')); document.querySelectorAll('.jb-toggle-icon').forEach(i=>i.textContent='▼'); }
                    function collapseAllJBCards(){ document.querySelectorAll('.jb-card-body').forEach(b=>b.classList.remove('open')); document.querySelectorAll('.jb-card').forEach(c=>c.classList.remove('open')); document.querySelectorAll('.jb-toggle-icon').forEach(i=>i.textContent='▶'); }
                    function copyText(ev,id){ ev.stopPropagation(); const el=document.getElementById(id); if(!el) return; const txt=el.innerText; navigator.clipboard.writeText(txt); const btn=ev.currentTarget; const old=btn.textContent; btn.textContent='Copied!'; setTimeout(()=>btn.textContent=old,1200); }
                    </script>
                    """

                    wrapper_html = custom_css + "<div class='jb-controls'>" + \
                        "<button class='jb-btn' onclick='expandAllJBCards()'>Expand All</button>" + \
                        "<button class='jb-btn' onclick='collapseAllJBCards()'>Collapse All</button>" + \
                        "</div><div class='jb-cards-wrapper'>" + ''.join(card_blocks) + "</div>" + controls_js

                    st.components.v1.html(wrapper_html, height=400+len(results)*250, scrolling=True)


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