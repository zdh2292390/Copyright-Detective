import streamlit as st

def prompt_preview(prompt_text: str, expanded: bool = False):
    """
    Renders a styled preview of the prompt.
    """
    with st.expander("Prompt Preview", expanded=expanded):
        st.markdown(
            f"""
            <div style="background-color: #f0f2f6; border-radius: 10px; padding: 15px; margin-bottom: 15px; border: 1px solid #e0e0e0;">
                <p style="font-family: monospace; white-space: pre-wrap; margin: 0; color: #333;">{prompt_text}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
