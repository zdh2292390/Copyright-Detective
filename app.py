import streamlit as st
from src.ui import render_header, render_sidebar, render_text_analysis_page, render_pdf_analysis_page

# Ensure the sidebar starts expanded (and we'll hide the toggle via CSS)
st.set_page_config(
    page_title="Copyright Detective",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Load custom CSS
with open("assets/styles.css", "r") as f:
    css = f.read()
st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)

# Keep CSS-only approach; removed JS ligature fallback to avoid side effects

# Render header
render_header()

# Render sidebar and get configuration
api_key, model_choice, provider, page = render_sidebar()

# Render main content based on selected page
if page == "Text Snippet Analysis":
    render_text_analysis_page(api_key, model_choice, provider)
elif page == "Whole PDF Analysis":
    render_pdf_analysis_page(api_key, model_choice, provider)
