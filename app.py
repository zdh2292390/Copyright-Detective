import streamlit as st
from src.copyright_detective.comparison import compare_texts
from src.copyright_detective.pdf_utils import extract_text_from_pdf, split_text_into_chunks

# Custom CSS for better styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    @import url('https://fonts.googleapis.com/icon?family=Material+Icons');
    
    * {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
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
        padding: 1.2rem 0;
        margin-bottom: 1.2rem;
    }
    
    .sidebar-section h3 {
        font-size: 1rem;
        font-weight: 600;
        color: #2c3e50;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .stButton>button {
        background: linear-gradient(135deg, #1f77b4 0%, #0e5a8a 100%);
        color: white;
        border-radius: 8px;
        border: none;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        font-size: 0.95rem;
        letter-spacing: 0.3px;
        transition: all 0.3s ease;
        box-shadow: 0 2px 4px rgba(31, 119, 180, 0.2);
        display: inline-flex;
        align-items: center;
        justify-content: center;
        gap: 0.5rem;
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
    
    .icon-text {
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .material-icons {
        font-family: 'Material Icons';
        font-weight: normal;
        font-style: normal;
        font-size: 1.2em;
        line-height: 1;
        letter-spacing: normal;
        text-transform: none;
        display: inline-block;
        white-space: nowrap;
        word-wrap: normal;
        direction: ltr;
    }
    
    .card {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
        margin-bottom: 1rem;
        border: 1px solid #e9ecef;
    }
    
    .gradient-bg {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    
    @media (max-width: 768px) {
        .main-header {
            font-size: 2.2rem;
        }
        
        .card {
            padding: 1rem;
        }
        
        .stButton>button {
            padding: 0.6rem 1rem;
            font-size: 0.9rem;
        }
    }
</style>
""", unsafe_allow_html=True)

# Main header
st.markdown('<div class="card gradient-bg" style="text-align: center; margin-bottom: 2rem;">', unsafe_allow_html=True)
st.markdown('<h1 class="main-header"><span class="material-icons" style="vertical-align: middle; margin-right: 0.5rem;">search</span>Copyright Detective</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Advanced AI-powered tool for detecting potential copyright infringement in large language models</p>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# --- API Key Management ---
with st.sidebar:
    st.markdown('<div class="icon-text"><span class="material-icons">key</span><h3>API Configuration</h3></div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    openai_api_key = st.text_input("OpenAI API Key", type="password", help="Enter your OpenAI API key")
    openrouter_api_key = st.text_input("OpenRouter API Key", type="password", help="Enter your OpenRouter API key")
    st.markdown('</div>', unsafe_allow_html=True)

    # --- Model Selection ---
    st.markdown('<div class="icon-text"><span class="material-icons">smart_toy</span><h3>Model Selection</h3></div>', unsafe_allow_html=True)
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
    st.markdown('<div class="icon-text"><span class="material-icons">navigation</span><h3>Navigation</h3></div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    page = st.radio("Go to", ["Text Snippet Analysis", "Whole PDF Analysis"], label_visibility="collapsed")
    st.markdown('</div>', unsafe_allow_html=True)

if page == "Text Snippet Analysis":
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="icon-text" style="margin-bottom: 1rem;"><span class="material-icons">text_snippet</span><h3>Text Snippet Analysis</h3></div>', unsafe_allow_html=True)
    st.markdown('<p style="color: #666; margin-bottom: 1.5rem;">Analyze text snippets to detect potential copyright infringement by comparing generated text with ground truth.</p>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Prefix Text**")
        text1 = st.text_area("Prefix Text", height=150, placeholder="Enter the prefix text that will be used to generate continuation...", label_visibility="collapsed")
    with col2:
        st.markdown("**Ground Truth**")
        text2 = st.text_area("Ground Truth", height=150, placeholder="Enter the ground truth text to compare against...", label_visibility="collapsed")

    col_center = st.columns([1, 2, 1])[1]
    with col_center:
        run_comparison = st.button("🚀 Run Comparison", use_container_width=True)

    if run_comparison:
        if not api_key:
            st.error(f"⚠️ Please enter your API key in the sidebar.")
        elif not text1 or not text2:
            st.warning("⚠️ Please enter both prefix text and ground truth.")
        else:
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

elif page == "Whole PDF Analysis":
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="icon-text" style="margin-bottom: 1rem;"><span class="material-icons">picture_as_pdf</span><h3>Whole PDF Analysis</h3></div>', unsafe_allow_html=True)
    st.markdown('<p style="color: #666; margin-bottom: 1.5rem;">Upload a whole PDF document to automatically analyze text chunks for potential copyright infringement.</p>', unsafe_allow_html=True)
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
        analyze_pdf = st.button('<span class="material-icons" style="vertical-align: middle; margin-right: 0.5rem;">search</span>Analyze PDF', use_container_width=True)

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
