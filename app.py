import streamlit as st
from src.copyright_detective.comparison import compare_texts
from src.copyright_detective.pdf_utils import extract_text_from_pdf, split_text_into_chunks
import matplotlib.pyplot as plt

# Custom CSS for better styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    @import url('https://fonts.googleapis.com/icon?family=Material+Icons');
    
    * {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
    }

    /* Material Icons override - ensure ligatures like "keyboard_double_arrow_right" render as icons
       This must come after the global * rule which forces 'Inter' with !important. */
    .material-icons,
    .material-icons-outlined,
    .material-icons-round,
    .material-icons-sharp,
    .material-icons-two-tone {
        font-family: 'Material Icons' !important;
        font-weight: normal;
        font-style: normal;
        font-size: 1.1rem; /* adjust size as needed */
        line-height: 1;
        letter-spacing: normal;
        text-transform: none;
        display: inline-block;
        white-space: nowrap;
        word-wrap: normal;
        direction: ltr;
        -webkit-font-feature-settings: 'liga';
        -webkit-font-smoothing: antialiased;
    }
    
    .main-header {
        font-size: 2.8rem;
        font-weight: 700;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 0.5rem;
        line-height: 1.2;
        letter-spacing: -0.02em;
    }
    
    .subtitle {
        font-size: 1.1rem;
        font-weight: 400;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
        line-height: 1.5;
        max-width: 600px;
        margin-left: auto;
        margin-right: auto;
    }
    
    .feature-card {
        padding: 1.5rem 0;
        margin-bottom: 1.5rem;
    }
    
    .feature-card h3 {
        font-size: 1.4rem;
        font-weight: 600;
        color: #2c3e50;
        margin-bottom: 0.5rem;
    }
    
    .feature-card p {
        font-size: 0.95rem;
        font-weight: 400;
        color: #666;
        line-height: 1.6;
    }
    
    .metric-card {
        padding: 1.2rem 0;
        text-align: center;
    }
    
    .metric-card .metric-label {
        font-size: 0.85rem;
        font-weight: 500;
        color: #495057;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 0.5rem;
    }
    
    .metric-card .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #1f77b4;
        line-height: 1;
    }
    
    .sidebar-section {
        padding: 0.5rem 0; /* reduced vertical padding */
        margin-bottom: 0.4rem; /* reduce gap between sidebar sections */
    }
    
    .sidebar-section h3 {
        font-size: 0.95rem;
        font-weight: 600;
        color: #2c3e50;
        margin-bottom: 0.35rem; /* tighten space under heading */
        display: flex;
        align-items: center;
        gap: 0.4rem;
    }
    
    .stButton>button {
        background: linear-gradient(135deg, #1f77b4 0%, #0e5a8a 100%);
        color: white;
        border-radius: 8px;
        border: none;
        padding: 0.6rem 1.2rem;
        font-weight: 600;
        font-size: 0.95rem;
        letter-spacing: 0.3px;
        transition: all 0.2s ease;
        box-shadow: 0 2px 4px rgba(31, 119, 180, 0.2);
    }
    
    .stButton>button:hover {
        background: linear-gradient(135deg, #0e5a8a 0%, #0a3d5c 100%);
        transform: translateY(-1px);
        box-shadow: 0 4px 8px rgba(31, 119, 180, 0.3);
    }
    
    .stTextArea>textarea {
        border-radius: 8px;
        border: 1px solid #e0e0e0;
        font-family: 'Inter', sans-serif;
        font-size: 0.95rem;
        line-height: 1.5;
        transition: border-color 0.2s ease;
    }
    
    .stTextArea>textarea:focus {
        border-color: #1f77b4;
        box-shadow: 0 0 0 3px rgba(31, 119, 180, 0.1);
    }
    
    .stTextArea label {
        font-weight: 600;
        color: #2c3e50;
        font-size: 1rem;
        margin-bottom: 0.5rem;
    }
    
    .stSelectbox label, .stRadio label {
        font-weight: 600;
        color: #2c3e50;
        font-size: 0.95rem;
    }
    
    .stExpander {
        border-radius: 8px;
    }
    
    .stExpander summary {
        font-weight: 600;
        color: #2c3e50;
        font-size: 0.95rem;
    }
    
    .stProgress .st-bo {
        background-color: #1f77b4;
    }

    /* Make Streamlit sidebar fixed and disable collapse/drag controls */
    /* Targets Streamlit's typical sidebar container and hides the collapse button */
    .css-1oe6wy4.e1fqkh3o0, /* old class fallback */
    .css-1d391kg.egzxvld2, /* another possible sidebar wrapper */
    .css-18e3th9.e1fqkh3o0 {
        position: fixed !important;
        left: 0 !important;
        top: 0 !important;
        height: 100vh !important;
        overflow: auto !important;
        z-index: 9999 !important;
        transition: transform 0.3s ease !important;
    }


    /* Hide all sidebar collapse/expand buttons (Streamlit and custom) */
    .sidebar-toggle-btn,
    [data-testid="stSidebarCollapseButton"] {
        display: none !important;
    }

    /* Force Streamlit's default sidebar collapse icon to use Material Icons font and show menu icon */
    [data-testid="stSidebarCollapseButton"] span[data-testid="stIconMaterial"] {
        font-family: 'Material Icons' !important;
        font-style: normal !important;
        font-weight: normal !important;
        font-size: 1.4rem !important;
        letter-spacing: normal !important;
        text-transform: none !important;
        display: inline-block !important;
        white-space: nowrap !important;
        word-wrap: normal !important;
        direction: ltr !important;
        -webkit-font-feature-settings: 'liga';
        -webkit-font-smoothing: antialiased;
        color: #1f77b4 !important;
    }
    [data-testid="stSidebarCollapseButton"] span[data-testid="stIconMaterial"]::before {
        content: "menu";
        font-family: 'Material Icons' !important;
    }

    /* Add left padding to main content so it's not covered by the fixed sidebar */
    .main > div[role="main"] {
        margin-left: 300px; /* adjust based on your sidebar width */
        transition: margin-left 0.3s ease !important;
    }

    /* When sidebar is collapsed, reduce main content margin */
    .sidebar-collapsed ~ .main > div[role="main"] {
        margin-left: 0 !important;
    }
    
    .stAlert {
        border-radius: 8px;
        border: none;
        box-shadow: none;
    }
    
    h1, h2, h3, h4 {
        color: #2c3e50;
        font-weight: 600;
        line-height: 1.3;
    }
    
    p {
        line-height: 1.6;
        color: #495057;
    }
    
    .rank-header {
        font-size: 1.3rem;
        font-weight: 700;
        color: #2c3e50;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# JavaScript fallback for Material Icons in deployed environments
st.markdown("""
<script>
    // Fallback for Material Icons ligatures in deployed environments
    document.addEventListener('DOMContentLoaded', function() {
        let sidebarCollapsed = false;
        let customToggleBtn = null;
        
        // Function to add material-icons class to elements containing icon ligatures
        function fixMaterialIcons() {
            // Find all text nodes containing icon ligatures
            const walker = document.createTreeWalker(
                document.body,
                NodeFilter.SHOW_TEXT,
                null,
                false
            );
            
            const nodes = [];
            let node;
            while (node = walker.nextNode()) {
                if (node.textContent.includes('keyboard_double_arrow_right') || 
                    node.textContent.includes('keyboard_double_arrow_left') ||
                    node.textContent.includes('menu')) {
                    nodes.push(node);
                }
            }
            
            // Wrap each found text node with a span having material-icons class
            nodes.forEach(textNode => {
                const span = document.createElement('span');
                span.className = 'material-icons';
                span.textContent = textNode.textContent;
                textNode.parentNode.replaceChild(span, textNode);
            });
        }
        
        // Only run icon fix for ligatures, no sidebar collapse logic
    });
</script>
""", unsafe_allow_html=True)

# Main header
st.markdown('<h1 class="main-header">🔍 Copyright Detective</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Advanced AI-powered tool for detecting potential copyright infringement in large language models</p>', unsafe_allow_html=True)

# --- API Key Management ---
with st.sidebar:
    st.markdown("### 🔑 API Configuration")
    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    openai_api_key = st.text_input("OpenAI API Key", type="password", help="Enter your OpenAI API key")
    openrouter_api_key = st.text_input("OpenRouter API Key", type="password", help="Enter your OpenRouter API key")
    st.markdown('</div>', unsafe_allow_html=True)

    # --- Model Selection ---
    st.markdown("### 🤖 Model Selection")
    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    provider = st.selectbox("Select Provider", ["OpenAI", "OpenRouter"], help="Choose your AI provider")

    model_choice = None
    if provider == "OpenAI":
        model_choice = st.sidebar.selectbox("Choose a model", ["gpt-3.5-turbo", "gpt-4o"])
        api_key = openai_api_key
    elif provider == "OpenRouter":
        model_choice = st.sidebar.selectbox(
            "Choose a model", 
            [
                "moonshotai/kimi-k2:free", 
                "meta-llama/llama-3.1-405b-instruct:free",
                "qwen/qwen3-235b-a22b:free",
                "meta-llama/llama-3.3-70b-instruct:free",
                "mistralai/mistral-small-24b-instruct-2501:free",
                "qwen/qwen-2.5-72b-instruct:free"
            ]
        )
        api_key = openrouter_api_key
    st.markdown('</div>', unsafe_allow_html=True)

    # --- Page Navigation ---
    st.markdown("### 🧭 Navigation")
    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    page = st.radio("Go to", ["Text Snippet Analysis", "Whole PDF Analysis"], label_visibility="collapsed")
    st.markdown('</div>', unsafe_allow_html=True)

if page == "Text Snippet Analysis":
    st.markdown('<div class="feature-card">', unsafe_allow_html=True)
    st.markdown("### 📝 Text Snippet Analysis")
    st.markdown("Analyze text snippets to detect potential copyright infringement by comparing generated text with ground truth.")
    st.markdown('</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Prefix Text**")
        text1 = st.text_area("Prefix Text", height=150, placeholder="Enter the prefix text that will be used to generate continuation...", label_visibility="collapsed")
    with col2:
        st.markdown("**Ground Truth**")
        text2 = st.text_area("Ground Truth", height=150, placeholder="Enter the ground truth text to compare against...", label_visibility="collapsed")

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
            if inference_runs == 1:
                # Single run: Original Analysis Results
                with st.spinner(f"🔄 Generating text with {model_choice} and calculating scores..."):
                    result = compare_texts(text1, text2, api_key, model_name=model_choice, provider=provider)
                    if isinstance(result, str) and result.startswith("Error"):
                        st.error(f"❌ {result}")
                    else:
                        generated_text, rouge_score, jaccard_index, levenshtein_dist = result
                        
                        # Results section
                        st.markdown("---")
                        st.markdown("### 📊 Analysis Results")
                        
                        # Generated text
                        st.markdown("**🤖 Generated Text**")
                        st.markdown(f'<div style="padding: 1rem 0; font-family: Inter, sans-serif; line-height: 1.6; color: #2c3e50;">{generated_text}</div>', unsafe_allow_html=True)
                        
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
                    result = compare_texts(text1, text2, api_key, model_name=model_choice, provider=provider)
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
                        st.markdown(f'<div style="padding: 1rem 0; font-family: Inter, sans-serif; line-height: 1.6; color: #2c3e50;">{text}</div>', unsafe_allow_html=True)

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

elif page == "Whole PDF Analysis":
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
                        chunk_pairs = split_text_into_chunks(pdf_text)
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
                                        st.markdown(f'<div style="padding: 1rem 0; font-family: Inter, sans-serif; line-height: 1.6; color: #2c3e50; margin-bottom: 1rem;">{generated}</div>', unsafe_allow_html=True)
                                        st.markdown("**📊 All Scores**")
                                        st.markdown(f"""
                                        <div style="padding: 1rem 0; font-family: Inter, sans-serif;">
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
