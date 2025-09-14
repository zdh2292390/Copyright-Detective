import streamlit as st
from src.copyright_detective.comparison import compare_texts
from src.copyright_detective.pdf_utils import extract_text_from_pdf, split_text_into_chunks

st.title("Copyright Detective")

st.write("This tool helps you find evidence of potential copyright infringement in large language models.")

# --- API Key Management ---
st.sidebar.header("API Keys")
openai_api_key = st.sidebar.text_input("OpenAI API Key", type="password")
openrouter_api_key = st.sidebar.text_input("OpenRouter API Key", type="password")


# --- Model Selection ---
st.sidebar.header("Model Selection")
provider = st.sidebar.selectbox("Select Provider", ["OpenAI", "OpenRouter"])

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


# --- Page Navigation ---
st.sidebar.header("Navigation")
page = st.sidebar.radio("Go to", ["Text Comparison", "PDF Analysis"])

if page == "Text Comparison":
    # --- Feature 1: Direct Text Comparison ---
    st.header("1. Compare two text snippets")
    text1 = st.text_area("Upper context text", height=150)
    text2 = st.text_area("Lower context text (the real one)", height=150)

    if st.button("Run Comparison"):
        if not api_key:
            st.warning(f"Please enter your {provider} API key in the sidebar.")
        elif not text1 or not text2:
            st.warning("Please enter both upper and lower context text.")
        else:
            with st.spinner(f"Generating text with {model_choice} and calculating scores..."):
                result = compare_texts(text1, text2, api_key, model_name=model_choice, provider=provider)
                if isinstance(result, str) and result.startswith("Error"):
                    st.error(result)
                else:
                    generated_text, rouge_score, jaccard_index, levenshtein_dist = result
                    st.subheader("Generated Text")
                    st.write(generated_text)

                    st.subheader("Similarity Scores")
                    col1, col2, col3 = st.columns(3)
                    col1.metric(label="ROUGE-L", value=f"{rouge_score:.4f}")
                    col2.metric(label="Jaccard Index", value=f"{jaccard_index:.4f}")
                    col3.metric(label="Levenshtein Distance", value=f"{levenshtein_dist}")

                    if rouge_score > 0.5 or jaccard_index > 0.5:
                        st.success("High similarity detected!")
                    else:
                        st.info("Low to moderate similarity.")

elif page == "PDF Analysis":
    # --- Feature 2: PDF Analysis ---
    st.header("2. Analyze a PDF document")
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    score_type = st.selectbox("Choose score for PDF analysis ranking", ["ROUGE-L", "Jaccard Index", "Levenshtein Distance"])

    if st.button("Analyze PDF"):
        if not api_key:
            st.warning(f"Please enter your {provider} API key in the sidebar.")
        elif uploaded_file is not None:
            with st.spinner(f"Analyzing PDF with {model_choice}... This may take a while."):
                try:
                    pdf_text = extract_text_from_pdf(uploaded_file)
                    if "Error" in pdf_text:
                        st.error(pdf_text)
                    else:
                        chunk_pairs = split_text_into_chunks(pdf_text)
                        if not chunk_pairs:
                            st.warning("Could not split the PDF into enough text chunks for analysis.")
                        else:
                            results = []
                            progress_bar = st.progress(0)
                            total_chunks = len(chunk_pairs)
                            for i, (upper, lower) in enumerate(chunk_pairs):
                                try:
                                    generated_text, rouge_score, jaccard_index, levenshtein_dist = compare_texts(upper, lower, api_key, model_name=model_choice, provider=provider)
                                except ValueError:
                                    generated_text, rouge_score, jaccard_index = compare_texts(upper, lower, api_key, model_name=model_choice, provider=provider)
                                    levenshtein_dist = 0
                                results.append(((upper, lower, generated_text), rouge_score, jaccard_index, levenshtein_dist))
                                progress_bar.progress((i + 1) / total_chunks, text=f"Processing chunk {i+1}/{total_chunks}")

                            # Sort results by the selected score type
                            if score_type == "ROUGE-L":
                                results.sort(key=lambda x: x[1], reverse=True)
                            elif score_type == "Jaccard Index":
                                results.sort(key=lambda x: x[2], reverse=True)
                            else: # Levenshtein Distance
                                results.sort(key=lambda x: x[3])

                            st.subheader(f"Top 5 Most Similar Sections (ranked by {score_type})")
                            for i, (texts, rouge, jaccard, levenshtein) in enumerate(results[:5]):
                                upper, lower, generated = texts
                                st.markdown(f"---")
                                st.metric(label=f"Rank {i+1} - {score_type} Score", value=f"{rouge if score_type == 'ROUGE-L' else jaccard if score_type == 'Jaccard Index' else levenshtein:.4f}")
                                with st.expander(f"Click to see text pair and all scores"):
                                    st.text_area("Upper Context", upper, height=150)
                                    st.text_area("Lower Context (Original)", lower, height=150)
                                    st.text_area("LLM Generated Text", generated, height=150)
                                    st.write(f"ROUGE-L Score: {rouge:.4f}")
                                    st.write(f"Jaccard Index: {jaccard:.4f}")
                                    st.write(f"Levenshtein Distance: {levenshtein}")
                except Exception as e:
                    st.error(f"An error occurred during PDF analysis: {e}")
        else:
            st.warning("Please upload a PDF file first.")
