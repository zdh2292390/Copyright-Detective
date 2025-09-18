import streamlit as st
import streamlit.components.v1 as components
from src.copyright_detective.comparison import compare_texts
from src.copyright_detective.pdf_utils import extract_text_from_pdf, split_text_into_chunks
import matplotlib.pyplot as plt

def render_header():
    """Render the main header and subtitle."""
    st.markdown('<h1 class="main-header">🔍 Copyright Detective</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Advanced AI-powered tool for detecting potential copyright infringement in large language models</p>', unsafe_allow_html=True)

def render_sidebar():
    """Render the sidebar with API configuration, model selection, and navigation."""
    with st.sidebar:
        # API Key Management
        st.markdown("### 🔑 API Configuration")
        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        openai_api_key = st.text_input("OpenAI API Key", type="password", help="Enter your OpenAI API key")
        openrouter_api_key = st.text_input("OpenRouter API Key", type="password", help="Enter your OpenRouter API key")
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
                    "meta-llama/llama-3.2-3b-instruct:free"
                ]
            )
            api_key = openrouter_api_key
        elif provider == "Anthropic":
            model_choice = st.selectbox("Choose a model", ["claude-3-haiku-20240307", "claude-3-sonnet-20240229", "claude-3-opus-20240229"])
            api_key = anthropic_api_key
        elif provider == "Google Gemini":
            model_choice = st.selectbox("Choose a model", ["gemini-1.5-flash", "gemini-1.5-pro"])
            api_key = google_api_key
        st.markdown('</div>', unsafe_allow_html=True)

        # Page Navigation
        st.markdown("### 🧭 Navigation")
        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        page = st.radio("Go to", ["Text Snippet Analysis", "Whole PDF Analysis"], label_visibility="collapsed")
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
            "Copyright Attribution Inference"
        ],
        help="Select the type of prompt to guide the Text Snippet Analysis. (Choose only; typing custom values is not allowed.)"
    )

    # Explanatory notes for each prompt type
    if prompt_type == "Sequential Continuation Evaluation":
        st.markdown("_Sequential Continuation Evaluation: Provide the prefix (previous sentence) and ask the model to continue by generating the next sentence. This probes whether the model reproduces or closely follows memorized sequences from source texts._")
    elif prompt_type == "Preceding Context Reconstruction":
        st.markdown("_Preceding Context Reconstruction: Provide the continuation or subsequent sentence and ask the model to generate the most likely preceding sentence. This helps detect whether the model can reconstruct prior context, which may indicate memorization of original works._")
    elif prompt_type == "Copyright Attribution Inference":
        st.markdown("_Copyright Attribution Inference: Based on the provided text snippet, ask the model to infer a likely title or attribution for the work (for example, a classic novel or another copyrighted source). Useful for identifying potential origins of the snippet._")

    st.markdown("Analyze text snippets to detect potential copyright infringement by comparing generated text with ground truth.")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Input Text**")
        text1 = st.text_area(
            "Input Text",
            height=150,
            placeholder="Enter the input snippet (e.g., a previous sentence, a continuation, or an excerpt). The role of this field depends on the selected prompt type.",
            label_visibility="collapsed"
        )
    with col2:
        st.markdown("**Ground Truth**")
        text2 = st.text_area(
            "Ground Truth",
            height=150,
            placeholder="Enter the ground truth text or expected target to compare against (e.g., the known reference or target continuation). Leave blank if not applicable.",
            label_visibility="collapsed"
        )

    st.markdown("---")
    st.markdown("**Inference Time Scaling**")
    inference_runs = st.number_input("Number of Inference Runs", min_value=1, max_value=100, value=1, step=1, help="Specify how many times to run the inference for statistical analysis.")

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
                with st.spinner(f"🔄 Generating text with {model_choice} and calculating scores..."):
                    result = compare_texts(text1, text2, api_key, model_name=model_choice, provider=provider, prompt_type=prompt_type)
                    if isinstance(result, str) and result.startswith("Error"):
                        st.error(f"❌ {result}")
                    else:
                        generated_text, rouge_score, jaccard_index, levenshtein_dist = result
                        
                        # Results section
                        st.markdown("---")
                        st.markdown("### 📊 Analysis Results")
                        
                        # Generated text
                        st.markdown("**🤖 Generated Text**")
                        st.markdown(f'<div style="padding: 1rem 0; font-family: Georgia, serif; line-height: 1.6; color: #2c3e50;">{generated_text}</div>', unsafe_allow_html=True)
                        
                        # Similarity scores in cards
                        st.markdown("**📈 Similarity Scores**")
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                            st.metric(label="ROUGE-L Score", value=f"{rouge_score:.4f}", 
                                    delta="High" if rouge_score > 0.5 else "Low")
                            st.markdown('</div>', unsafe_allow_html=True)
                        with col2:
                            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                            st.metric(label="Jaccard Index", value=f"{jaccard_index:.4f}",
                                    delta="High" if jaccard_index > 0.5 else "Low")
                            st.markdown('</div>', unsafe_allow_html=True)
                        with col3:
                            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                            st.metric(label="Levenshtein Distance", value=f"{levenshtein_dist}")
                            st.markdown('</div>', unsafe_allow_html=True)

                        # Conclusion
                        if rouge_score > 0.5 or jaccard_index > 0.5:
                            st.success("🎯 **High similarity detected!** This may indicate potential copyright concerns.")
                        else:
                            st.info("✅ **Low to moderate similarity.** The generated text appears sufficiently different.")
            else:
                # Multiple runs: Inference Results Over Multiple Runs
                st.markdown("### 🔄 Inference Results Over Multiple Runs")
                similarity_scores = []
                generated_texts = []  # Store generated texts for each run
                progress_bar = st.progress(0, text="Starting inference runs...")
                for i in range(inference_runs):
                    progress_bar.progress((i) / inference_runs, text=f"🔄 Generating text for run {i+1}/{inference_runs}...")
                    result = compare_texts(text1, text2, api_key, model_name=model_choice, provider=provider, prompt_type=prompt_type)
                    if isinstance(result, str) and result.startswith("Error"):
                        st.error(f"❌ {result}")
                        break
                    else:
                        generated_text, rouge_score, jaccard_index, levenshtein_dist = result
                        similarity_scores.append({
                            "rouge": rouge_score,
                            "jaccard": jaccard_index,
                            "levenshtein": levenshtein_dist
                        })
                        generated_texts.append(generated_text)  # Append generated text
                progress_bar.progress(1.0, text="✅ All runs completed!")

                if similarity_scores:
                    # Display generated texts for each run
                    st.markdown("### 🤖 Generated Texts for Each Run")
                    for i, text in enumerate(generated_texts):
                        st.markdown(f"**Run {i+1}:**")
                        st.markdown(f'<div style="padding: 1rem 0; font-family: Georgia, serif; line-height: 1.6; color: #2c3e50;">{text}</div>', unsafe_allow_html=True)

                    # Calculate statistics
                    rouge_scores = [score["rouge"] for score in similarity_scores]
                    jaccard_scores = [score["jaccard"] for score in similarity_scores]
                    levenshtein_scores = [score["levenshtein"] for score in similarity_scores]

                    stats = {
                        "rouge": {
                            "max": max(rouge_scores),
                            "min": min(rouge_scores),
                            "avg": sum(rouge_scores) / len(rouge_scores)
                        },
                        "jaccard": {
                            "max": max(jaccard_scores),
                            "min": min(jaccard_scores),
                            "avg": sum(jaccard_scores) / len(jaccard_scores)
                        },
                        "levenshtein": {
                            "max": max(levenshtein_scores),
                            "min": min(levenshtein_scores),
                            "avg": sum(levenshtein_scores) / len(levenshtein_scores)
                        }
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

def render_pdf_analysis_page(api_key, model_choice, provider):
    """Render the whole PDF analysis page."""
    st.markdown("### 📄 Whole PDF Analysis")
    st.markdown("Upload a whole PDF document to automatically analyze text chunks for potential copyright infringement.")
    
    # File Upload Section
    uploaded_file = st.file_uploader("📎 Choose a PDF file", type="pdf", 
                                   help="Select a PDF document to analyze")
    
    # Configuration Section
    if uploaded_file is not None:
        st.markdown('<h3 class="section-header">⚙️ Analysis Configuration</h3>', unsafe_allow_html=True)
        
        # Controls in a separate section
        col1, col2 = st.columns([1, 1])
        
        with col1:
            score_type = st.selectbox('Change Ranking Metric', ["ROUGE-L", "Jaccard Index", "Levenshtein Distance"], 
                                    help='Choose how to rank the most similar sections', 
                                    key='ranking_metric', index=0)
            
        with col2:
            chunk_size = st.number_input('Change Chunk Size', min_value=50, max_value=1000, value=200, step=25, 
                                       help='Number of words per text chunk', 
                                       key='chunk_size')
        
        # Recommendations
        st.markdown('<h3 class="section-header">💡 Size Recommendations</h3>', unsafe_allow_html=True)
        st.markdown("""
        <div style="background: #f0f8ff; padding: 1rem; border-radius: 8px; border-left: 4px solid #1f77b4;">
            <div style="margin-bottom: 0.5rem;"><strong>50-200:</strong> Precise analysis — detects specific phrases</div>
            <div style="margin-bottom: 0.5rem;"><strong>200-400:</strong> Balanced — general copyright detection</div>
            <div><strong>400-1000:</strong> Contextual — preserves broader context</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Analysis button - only show when file is uploaded
    if uploaded_file is not None:
        st.markdown("---")
        col_center = st.columns([1, 2, 1])[1]
        with col_center:
            analyze_pdf = st.button("🔍 Analyze PDF", use_container_width=True, type="primary")
        st.markdown("""
        <div class="analysis-note">
            ⚡ Analysis may take several minutes depending on PDF size and selected model
        </div>
        """, unsafe_allow_html=True)
    else:
        analyze_pdf = False

    if analyze_pdf:
        if not api_key:
            st.error(f"⚠️ Please enter your API key in the sidebar.")
        elif uploaded_file is not None:
            with st.spinner(""):
                spinner_placeholder = st.empty()
                spinner_placeholder.markdown(f'<div style="font-size: 0.85rem;">🔄 Analyzing PDF with {model_choice}... This may take a while.</div>', unsafe_allow_html=True)
                try:
                    pdf_text = extract_text_from_pdf(uploaded_file)
                    if "Error" in pdf_text:
                        st.error(f"❌ {pdf_text}")
                        spinner_placeholder.empty()
                    else:
                        chunk_pairs = split_text_into_chunks(pdf_text, chunk_size=chunk_size)
                        if not chunk_pairs:
                            st.warning("⚠️ Could not split the PDF into enough text chunks for analysis.")
                            spinner_placeholder.empty()
                        else:
                            results = []
                            progress_bar = st.progress(0, text="Processing text chunks...")
                            total_chunks = len(chunk_pairs)
                            
                            for i, (upper, lower) in enumerate(chunk_pairs):
                                generated_text, rouge_score, jaccard_index, levenshtein_dist = compare_texts(
                                    upper, lower, api_key, model_name=model_choice, provider=provider)
                                results.append(((upper, lower, generated_text), rouge_score, jaccard_index, levenshtein_dist))
                                progress_bar.progress((i + 1) / total_chunks, 
                                                    text=f"🔄 Processing chunk {i+1}/{total_chunks}")

                            # Sort results by the selected score type
                            if score_type == "ROUGE-L":
                                results.sort(key=lambda x: x[1], reverse=True)
                            elif score_type == "Jaccard Index":
                                results.sort(key=lambda x: x[2], reverse=True)
                            else:  # Levenshtein Distance
                                results.sort(key=lambda x: x[3])

                            st.markdown("---")
                            st.markdown("""
                            <div style="margin: 2rem 0;">
                                <h3 style="font-size: 1.5rem; font-weight: 700; color: #1e293b; margin-bottom: 0.5rem; display: flex; align-items: center; gap: 0.5rem;">
                                    🏆 Top 5 Most Similar Sections
                                </h3>
                                <p style="color: #64748b; font-size: 0.95rem; margin-bottom: 1.5rem;">Ranked by {score_type}</p>
                            </div>
                            """.format(score_type=score_type), unsafe_allow_html=True)

                            # Define rank styling
                            rank_styles = [
                                {"bg": "linear-gradient(135deg, #ffd700, #ffb347)", "color": "#8b4513", "shadow": "0 4px 15px rgba(255, 215, 0, 0.3)"},
                                {"bg": "linear-gradient(135deg, #c0c0c0, #a8a8a8)", "color": "#696969", "shadow": "0 4px 15px rgba(192, 192, 192, 0.3)"},
                                {"bg": "linear-gradient(135deg, #cd7f32, #a0522d)", "color": "#8b4513", "shadow": "0 4px 15px rgba(205, 127, 50, 0.3)"},
                                {"bg": "linear-gradient(135deg, #e8f4fd, #b3d9ff)", "color": "#1e40af", "shadow": "0 4px 15px rgba(59, 130, 246, 0.2)"},
                                {"bg": "linear-gradient(135deg, #f0f9ff, #bae6fd)", "color": "#0369a1", "shadow": "0 4px 15px rgba(14, 165, 233, 0.2)"}
                            ]

                            for i, (texts, rouge, jaccard, levenshtein) in enumerate(results[:5]):
                                upper, lower, generated = texts

                                # Get score value and display
                                score_value = (rouge if score_type == "ROUGE-L"
                                             else jaccard if score_type == "Jaccard Index"
                                             else levenshtein)
                                score_display = f"{score_value:.4f}" if score_type != "Levenshtein Distance" else f"{score_value}"

                                # Determine score color based on value
                                if score_type == "Levenshtein Distance":
                                    score_color = "#ef4444" if score_value > 50 else "#f59e0b" if score_value > 25 else "#10b981"
                                else:
                                    score_color = "#ef4444" if score_value > 0.7 else "#f59e0b" if score_value > 0.4 else "#10b981"

                                rank_style = rank_styles[i] if i < len(rank_styles) else rank_styles[-1]

                                # Create modern card layout
                                card_html = f"""
                                <style>
                                .similarity-card-{i} {{
                                    background: white;
                                    border: 1px solid #e2e8f0;
                                    border-radius: 12px;
                                    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1), 0 1px 2px rgba(0, 0, 0, 0.06);
                                    margin-bottom: 1.5rem;
                                    overflow: hidden;
                                    transition: all 0.2s ease;
                                }}
                                .similarity-card-{i}:hover {{
                                    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.07), 0 2px 4px rgba(0, 0, 0, 0.06);
                                    transform: translateY(-1px);
                                }}
                                .card-header-{i} {{
                                    background: {rank_style["bg"]};
                                    padding: 1rem 1.25rem;
                                    display: flex;
                                    align-items: center;
                                    justify-content: space-between;
                                    border-bottom: 1px solid rgba(255, 255, 255, 0.2);
                                }}
                                .rank-info-{i} {{
                                    display: flex;
                                    align-items: center;
                                    gap: 0.75rem;
                                }}
                                .rank-badge-{i} {{
                                    background: rgba(255, 255, 255, 0.9);
                                    border-radius: 50%;
                                    width: 40px;
                                    height: 40px;
                                    display: flex;
                                    align-items: center;
                                    justify-content: center;
                                    font-size: 1.25rem;
                                    font-weight: 700;
                                    color: {rank_style["color"]};
                                    box-shadow: {rank_style["shadow"]};
                                    border: 2px solid rgba(255, 255, 255, 0.8);
                                }}
                                .rank-text-{i} {{
                                    color: white;
                                    font-size: 1.1rem;
                                    font-weight: 600;
                                    text-shadow: 0 1px 2px rgba(0, 0, 0, 0.1);
                                }}
                                .score-display-{i} {{
                                    background: rgba(255, 255, 255, 0.95);
                                    padding: 0.5rem 1rem;
                                    border-radius: 20px;
                                    display: flex;
                                    flex-direction: column;
                                    align-items: center;
                                    min-width: 100px;
                                }}
                                .score-label-{i} {{
                                    font-size: 0.75rem;
                                    font-weight: 500;
                                    color: #64748b;
                                    text-transform: uppercase;
                                    letter-spacing: 0.5px;
                                    margin-bottom: 0.25rem;
                                }}
                                .score-value-{i} {{
                                    font-size: 1.1rem;
                                    font-weight: 700;
                                    color: {score_color};
                                }}
                                .card-content-{i} {{
                                    padding: 1.25rem;
                                }}
                                .text-preview-{i} {{
                                    background: #f8fafc;
                                    border: 1px solid #e2e8f0;
                                    border-radius: 8px;
                                    padding: 1rem;
                                    margin-bottom: 1rem;
                                    font-family: 'Georgia', 'Times New Roman', serif;
                                    font-size: 0.9rem;
                                    line-height: 1.6;
                                    color: #334155;
                                    max-height: 120px;
                                    overflow: hidden;
                                    position: relative;
                                }}
                                .text-preview-{i}::after {{
                                    content: '';
                                    position: absolute;
                                    bottom: 0;
                                    left: 0;
                                    right: 0;
                                    height: 40px;
                                    background: linear-gradient(transparent, #f8fafc);
                                    pointer-events: none;
                                }}
                                .expand-btn-{i} {{
                                    background: #f1f5f9;
                                    border: 1px solid #cbd5e1;
                                    border-radius: 6px;
                                    padding: 0.5rem 1rem;
                                    font-size: 0.85rem;
                                    font-weight: 500;
                                    color: #475569;
                                    cursor: pointer;
                                    transition: all 0.3s ease;
                                    display: flex;
                                    align-items: center;
                                    gap: 0.5rem;
                                    width: 100%;
                                    justify-content: center;
                                }}
                                .expand-btn-{i}:hover {{
                                    background: #e2e8f0;
                                    border-color: #94a3b8;
                                    transform: translateY(-1px);
                                    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
                                }}
                                .expand-icon-{i} {{
                                    transition: transform 0.3s ease;
                                }}
                                .details-panel-{i} {{
                                    margin-top: 1rem;
                                    padding-top: 1rem;
                                    border-top: 1px solid #e2e8f0;
                                    max-height: 0;
                                    overflow: hidden;
                                    transition: max-height 0.4s ease-in-out, padding-top 0.4s ease-in-out;
                                }}
                                .details-panel-{i}.expanded {{
                                    max-height: 800px;
                                    padding-top: 1rem;
                                }}
                                .detail-section-{i} {{
                                    margin-bottom: 1.25rem;
                                }}
                                .detail-label-{i} {{
                                    font-weight: 600;
                                    color: #1e293b;
                                    margin-bottom: 0.5rem;
                                    display: block;
                                    font-size: 0.9rem;
                                }}
                                .detail-text-{i} {{
                                    background: #f8fafc;
                                    border: 1px solid #e2e8f0;
                                    border-radius: 6px;
                                    padding: 0.75rem;
                                    font-family: 'Georgia', 'Times New Roman', serif;
                                    font-size: 0.85rem;
                                    line-height: 1.6;
                                    color: #334155;
                                    white-space: pre-wrap;
                                    word-wrap: break-word;
                                    max-height: 200px;
                                    overflow-y: auto;
                                }}
                                .generated-text-{i} {{
                                    background: #eff6ff;
                                    border-left: 4px solid #3b82f6;
                                    padding: 0.75rem;
                                    border-radius: 0 6px 6px 0;
                                    font-family: 'Georgia', 'Times New Roman', serif;
                                    font-size: 0.85rem;
                                    line-height: 1.6;
                                    color: #1e40af;
                                    max-height: 200px;
                                    overflow-y: auto;
                                }}
                                .scores-grid-{i} {{
                                    display: grid;
                                    grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
                                    gap: 0.75rem;
                                    margin-top: 0.5rem;
                                }}
                                .score-card-{i} {{
                                    background: #f8fafc;
                                    border: 1px solid #e2e8f0;
                                    border-radius: 6px;
                                    padding: 0.75rem;
                                    text-align: center;
                                }}
                                .score-name-{i} {{
                                    font-size: 0.8rem;
                                    font-weight: 500;
                                    color: #64748b;
                                    margin-bottom: 0.25rem;
                                    text-transform: uppercase;
                                    letter-spacing: 0.5px;
                                }}
                                .score-number-{i} {{
                                    font-size: 1rem;
                                    font-weight: 700;
                                    color: #1e293b;
                                }}
                                </style>

                                <div class="similarity-card-{i}">
                                    <div class="card-header-{i}">
                                        <div class="rank-info-{i}">
                                            <div class="rank-badge-{i}">{i+1}</div>
                                            <div class="rank-text-{i}">Rank {i+1}</div>
                                        </div>
                                        <div class="score-display-{i}">
                                            <div class="score-label-{i}">{score_type}</div>
                                            <div class="score-value-{i}">{score_display}</div>
                                        </div>
                                    </div>
                                    <div class="card-content-{i}">
                                        <div class="text-preview-{i}">{generated[:200]}{"..." if len(generated) > 200 else ""}</div>
                                        <button class="expand-btn-{i}" onclick="toggleDetails{i}()">
                                            <span>📋 View Full Details</span>
                                            <span class="expand-icon-{i}" id="icon-{i}">▶</span>
                                        </button>
                                        <div class="details-panel-{i}" id="details-{i}">
                                            <div class="detail-section-{i}">
                                                <span class="detail-label-{i}">📝 Prefix Context</span>
                                                <div class="detail-text-{i}">{upper.replace(chr(10), '<br>')}</div>
                                            </div>
                                            <div class="detail-section-{i}">
                                                <span class="detail-label-{i}">🎯 Ground Truth</span>
                                                <div class="detail-text-{i}">{lower.replace(chr(10), '<br>')}</div>
                                            </div>
                                            <div class="detail-section-{i}">
                                                <span class="detail-label-{i}">🤖 Generated Text</span>
                                                <div class="generated-text-{i}">{generated.replace(chr(10), '<br>')}</div>
                                            </div>
                                            <div class="detail-section-{i}">
                                                <span class="detail-label-{i}">📊 All Similarity Scores</span>
                                                <div class="scores-grid-{i}">
                                                    <div class="score-card-{i}">
                                                        <div class="score-name-{i}">ROUGE-L</div>
                                                        <div class="score-number-{i}">{rouge:.4f}</div>
                                                    </div>
                                                    <div class="score-card-{i}">
                                                        <div class="score-name-{i}">Jaccard</div>
                                                        <div class="score-number-{i}">{jaccard:.4f}</div>
                                                    </div>
                                                    <div class="score-card-{i}">
                                                        <div class="score-name-{i}">Levenshtein</div>
                                                        <div class="score-number-{i}">{levenshtein}</div>
                                                    </div>
                                                </div>
                                            </div>
                                        </div>
                                    </div>
                                </div>

                                <script>
                                function toggleDetails{i}() {{
                                    const panel = document.getElementById('details-{i}');
                                    const icon = document.getElementById('icon-{i}');
                                    const btn = document.querySelector('.expand-btn-{i}');

                                    if (panel.classList.contains('expanded')) {{
                                        panel.classList.remove('expanded');
                                        icon.textContent = '▶';
                                        btn.innerHTML = '<span>📋 View Full Details</span><span class="expand-icon-{i}" id="icon-{i}">▶</span>';
                                    }} else {{
                                        panel.classList.add('expanded');
                                        icon.textContent = '▼';
                                        btn.innerHTML = '<span>📋 Hide Details</span><span class="expand-icon-{i}" id="icon-{i}">▼</span>';
                                    }}
                                }}
                                </script>
                                """

                                components.html(card_html, height=600)
                            spinner_placeholder.empty()
                except Exception as e:
                    st.error(f"❌ An error occurred during PDF analysis: {e}")
                    spinner_placeholder.empty()
        else:
            st.warning("⚠️ Please upload a PDF file first.")