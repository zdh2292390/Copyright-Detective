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
            generated_text, scores = compare_texts(text1, text2, api_key, model_name=model_choice, provider=provider)
            st.subheader("Generated Text")
            st.write(generated_text)
            st.subheader("Comparison Results")
            
            col1, col2, col3 = st.columns(3)
            col1.metric(label="ROUGE-L Score", value=f"{scores.get('ROUGE-L', 0.0):.4f}")
            col2.metric(label="BLEU Score", value=f"{scores.get('BLEU', 0.0):.4f}")
            col3.metric(label="Jaccard Similarity", value=f"{scores.get('Jaccard', 0.0):.4f}")

            if scores.get('ROUGE-L', 0.0) > 0.5 or scores.get('Jaccard', 0.0) > 0.5:
                st.success("High similarity detected!")
            else:
                st.info("Low to moderate similarity.")

# --- Feature 2: PDF Analysis ---
st.header("2. Analyze a PDF document")
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

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
                            _, scores = compare_texts(upper, lower, api_key, model_name=model_choice, provider=provider)
                            # For PDF analysis, we'll primarily sort by ROUGE-L for consistency
                            results.append(((upper, lower), scores))
                            progress_bar.progress((i + 1) / total_chunks, text=f"Processing chunk {i+1}/{total_chunks}")
                        
                        # Sort results by ROUGE score in descending order
                        results.sort(key=lambda x: x[1].get('ROUGE-L', 0.0), reverse=True)
                        
                        st.subheader("Top 5 Most Similar Sections (by ROUGE-L)")
                        for i, (pair, scores) in enumerate(results[:5]):
                            st.markdown(f"---")
                            st.write(f"**Rank {i+1}**")
                            col1, col2, col3 = st.columns(3)
                            col1.metric(label="ROUGE-L Score", value=f"{scores.get('ROUGE-L', 0.0):.4f}")
                            col2.metric(label="BLEU Score", value=f"{scores.get('BLEU', 0.0):.4f}")
                            col3.metric(label="Jaccard Similarity", value=f"{scores.get('Jaccard', 0.0):.4f}")
                            
                            with st.expander(f"Click to see text pair"):
                                st.text_area("Upper Context", pair[0], height=150, key=f"pdf_upper_{i}")
                                st.text_area("Lower Context (Original)", pair[1], height=150, key=f"pdf_lower_{i}")
            except Exception as e:
                st.error(f"An error occurred during PDF analysis: {e}")
    else:
        st.warning("Please upload a PDF file first.")
