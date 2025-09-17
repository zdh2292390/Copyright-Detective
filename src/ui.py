import streamlit as st
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
    st.markdown('<div class="feature-card">', unsafe_allow_html=True)
    st.markdown("### 📄 Whole PDF Analysis")
    st.markdown("Upload a whole PDF document to automatically analyze text chunks for potential copyright infringement.")
    st.markdown('</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    with col1:
        uploaded_file = st.file_uploader("📎 Choose a PDF file", type="pdf", 
                                       help="Select a PDF document to analyze")
    with col2:
        score_type = st.selectbox("📊 Ranking Metric", 
                                ["ROUGE-L", "Jaccard Index", "Levenshtein Distance"],
                                help="Choose how to rank the most similar sections")

    st.markdown("---")
    col_center = st.columns([1, 2, 1])[1]
    with col_center:
        analyze_pdf = st.button("🔍 Analyze PDF", use_container_width=True)

    if analyze_pdf:
        if not api_key:
            st.error(f"⚠️ Please enter your {provider} API key in the sidebar.")
        elif uploaded_file is not None:
            with st.spinner(f"🔄 Analyzing PDF with {model_choice}... This may take a while."):
                try:
                    pdf_text = extract_text_from_pdf(uploaded_file)
                    if "Error" in pdf_text:
                        st.error(f"❌ {pdf_text}")
                    else:
                        chunk_pairs = split_text_into_chunks(pdf_text, chunk_size=100)
                        if not chunk_pairs:
                            st.warning("⚠️ Could not split the PDF into enough text chunks for analysis.")
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
                            st.markdown(f"### 🏆 Top 5 Most Similar Sections (ranked by {score_type})")
                            
                            for i, (texts, rouge, jaccard, levenshtein) in enumerate(results[:5]):
                                upper, lower, generated = texts
                                
                                # Rank card
                                rank_color = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else f"#{i+1}"
                                st.markdown(f'<div class="rank-header">{rank_color} Rank {i+1}</div>', unsafe_allow_html=True)
                                
                                # Score display
                                score_value = (rouge if score_type == "ROUGE-L" 
                                             else jaccard if score_type == "Jaccard Index" 
                                             else levenshtein)
                                score_display = f"{score_value:.4f}" if score_type != "Levenshtein Distance" else f"{score_value}"
                                
                                col1, col2 = st.columns([1, 3])
                                with col1:
                                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                                    st.metric(label=f"{score_type} Score", value=score_display)
                                    st.markdown('</div>', unsafe_allow_html=True)
                                with col2:
                                    with st.expander("📋 View Details"):
                                        st.markdown("**Prefix Context**")
                                        st.text_area("Prefix Context", upper, height=100, disabled=True, label_visibility="collapsed")
                                        st.markdown("**Ground Truth**")
                                        st.text_area("Ground Truth", lower, height=100, disabled=True, label_visibility="collapsed")
                                        st.markdown("**🤖 Generated Text**")
                                        st.markdown(f'<div style="padding: 1rem 0; font-family: Georgia, serif; line-height: 1.6; color: #2c3e50; margin-bottom: 1rem;">{generated}</div>', unsafe_allow_html=True)
                                        st.markdown("**📊 All Scores**")
                                        st.markdown(f"""
                                        <div style="padding: 1rem 0; font-family: Georgia, serif;">
                                            <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
                                                <span style="font-weight: 600; color: #2c3e50;">ROUGE-L:</span>
                                                <span style="font-weight: 700; color: #1f77b4;">{rouge:.4f}</span>
                                            </div>
                                            <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
                                                <span style="font-weight: 600; color: #2c3e50;">Jaccard Index:</span>
                                                <span style="font-weight: 700; color: #1f77b4;">{jaccard:.4f}</span>
                                            </div>
                                            <div style="display: flex; justify-content: space-between;">
                                                <span style="font-weight: 600; color: #2c3e50;">Levenshtein Distance:</span>
                                                <span style="font-weight: 700; color: #1f77b4;">{levenshtein}</span>
                                            </div>
                                        </div>
                                        """, unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"❌ An error occurred during PDF analysis: {e}")
        else:
            st.warning("⚠️ Please upload a PDF file first.")