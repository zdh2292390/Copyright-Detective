import streamlit as st
from src.ui import render_header, render_sidebar, render_text_analysis_page, render_pdf_analysis_page

# Load custom CSS
with open("assets/styles.css", "r") as f:
    css = f.read()
st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)

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

# Render header
render_header()

# Render sidebar and get configuration
api_key, model_choice, provider, page = render_sidebar()

# Render main content based on selected page
if page == "Text Snippet Analysis":
    render_text_analysis_page(api_key, model_choice, provider)
elif page == "Whole PDF Analysis":
    render_pdf_analysis_page(api_key, model_choice, provider)
