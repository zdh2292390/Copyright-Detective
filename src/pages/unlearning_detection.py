"""
Unlearning Detection Module

This module provides the UI for MIN-K% PROB and representational analysis to detect unlearning
in language models by comparing reference and updated models.
"""

import os
import textwrap
import zipfile
import math
import json
import random
import zlib
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict
from io import BytesIO

import requests
import streamlit as st
import pandas as pd
from src.job_guard import detection_job, render_run_button, reset_detection_job, wd
from src.floating_clear_cache import (
    register_clear_cache_handler,
    set_active_clear_cache_id,
    show_error_with_clear_cache,
)

MIN_K_CLEAR_CACHE_ID = "unlearning_min_k"
REPRESENTATIONAL_CLEAR_CACHE_ID = "unlearning_representational"


def _clear_min_k_cache() -> None:
    for key in list(st.session_state.keys()):
        if key.startswith("min_k_"):
            del st.session_state[key]
    st.session_state["min_k_input_mode"] = "Predefined Examples"
    reset_detection_job()
    from src.ui import _trigger_rerun

    _trigger_rerun()


def _clear_representational_cache() -> None:
    for key in list(st.session_state.keys()):
        if key.startswith("unlearn_"):
            del st.session_state[key]
    reset_detection_job()
    from src.ui import _trigger_rerun

    _trigger_rerun()

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

try:
    from sklearn.metrics import auc, roc_curve
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

try:
    import PyPDF2
    PDF_AVAILABLE = True
except ImportError:
    try:
        import pdfplumber
        PDF_AVAILABLE = True
    except ImportError:
        PDF_AVAILABLE = False

from src.components import render_collapsible_panel
from src.direct_recall.comparison import get_llm_completion
from src.pdf_preview import render_pdf_preview_with_blob, generate_min_k_prob_pdf_report
from src.unlearning_detection import (
    is_representational_analysis_available,
    list_representational_features,
    run_representational_analysis,
)


def load_wikimia_dataset(length: int) -> Optional[List[Dict[str, Any]]]:
    """Load WikiMIA dataset from parquet file.
    
    Args:
        length: Text length (32, 64, 128, or 256)
    
    Returns:
        List of dictionaries with 'text' and 'label' keys, or None if error
    """
    try:
        import pandas as pd
    except ImportError:
        raise ImportError("pandas is required to load WikiMIA dataset. Please install it: pip install pandas")
    
    # Construct path to parquet file
    data_dir = Path(__file__).parent.parent / "unlearning_detection" / "minkprob" / "WikiMIA"
    parquet_file = data_dir / f"WikiMIA_length{length}-00000-of-00001-*.parquet"
    
    # Find matching parquet file
    matching_files = list(data_dir.glob(f"WikiMIA_length{length}-*.parquet"))
    if not matching_files:
        raise FileNotFoundError(f"WikiMIA dataset file for length {length} not found in {data_dir}")
    
    parquet_path = matching_files[0]
    
    try:
        # Read parquet file
        df = pd.read_parquet(parquet_path)
        
        # Convert to expected format
        batch_data = []
        for _, row in df.iterrows():
            text = row.get('input', '')
            label = int(row.get('label', 0))
            
            if text:  # Only add non-empty texts
                batch_data.append({
                    'text': text,
                    'label': label,
                })
        
        return batch_data
    except Exception as e:
        raise RuntimeError(f"Error reading parquet file {parquet_path}: {str(e)}")


def load_bookmia_dataset() -> Optional[List[Dict[str, Any]]]:
    """Load BookMIA dataset from JSONL file.
    
    Returns:
        List of dictionaries with 'text' and 'label' keys, or None if error
    """
    # Construct path to JSONL file
    jsonl_path = Path(__file__).parent.parent / "unlearning_detection" / "minkprob" / "BookMIA" / "book_data.jsonl"
    
    if not jsonl_path.exists():
        raise FileNotFoundError(f"BookMIA dataset file not found: {jsonl_path}")
    
    try:
        batch_data = []
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    # BookMIA uses 'snippet' as text; fall back to 'input' if present
                    text = data.get('snippet') or data.get('input', '')
                    label = int(data.get('label', 0))
                    
                    if text:  # Only add non-empty texts
                        batch_data.append({
                            'text': text,
                            'label': label,
                        })
                except json.JSONDecodeError:
                    continue
        
        return batch_data
    except Exception as e:
        raise RuntimeError(f"Error reading JSONL file {jsonl_path}: {str(e)}")


def render_unlearning_detection_page(api_key, model_choice, provider):
    """Render the unlearning detection experience with tabs for MIN-K% PROB and Representational Analysis."""
    
    st.markdown('<h4 class="section-header">🔬 Unlearning Detection</h4>', unsafe_allow_html=True)
    st.markdown(
        "Detect unlearning in language models using MIN-K% PROB analysis and representational analysis."
    )

    unlearning_mode = st.radio(
        "Unlearning detection mode",
        ["MIN-K% PROB", "Representational Analysis"],
        horizontal=True,
        key="unlearning_detection_mode",
        label_visibility="collapsed",
    )

    if unlearning_mode == "MIN-K% PROB":
        render_min_k_prob_page(api_key, model_choice, provider)
    else:
        render_representational_analysis_page(api_key, model_choice, provider)


def _get_mode_key(input_mode: str, base_key: str) -> str:
    """Get mode-specific session state key."""
    mode_prefixes = {
        "User Input": "min_k_user_input_",
        "Upload Document": "min_k_upload_",
        "Predefined Examples": "min_k_predefined_",
    }
    prefix = mode_prefixes.get(input_mode, "min_k_")
    # Remove existing prefix if present
    if base_key.startswith("min_k_"):
        base_key = base_key[6:]  # Remove "min_k_"
    return prefix + base_key


def _clear_other_modes_data(current_mode: str):
    """Clear data from other modes when switching modes."""
    mode_keys = {
        "User Input": ["min_k_user_input_"],
        "Upload Document": ["min_k_upload_"],
        "Predefined Examples": ["min_k_predefined_"],
    }
    
    # Clear all keys that belong to other modes
    for mode, prefixes in mode_keys.items():
        if mode != current_mode:
            for key in list(st.session_state.keys()):
                if any(key.startswith(prefix) for prefix in prefixes):
                    st.session_state.pop(key, None)


def render_min_k_prob_page(api_key, model_choice, provider):
    """Render the MIN-K% PROB analysis page."""
    
    # Initialize session state for MIN-K% PROB
    if 'min_k_input_mode' not in st.session_state:
        st.session_state['min_k_input_mode'] = "Predefined Examples"
    
    # Shared configuration (not mode-specific)
    if 'min_k_deploy_agent_url' not in st.session_state:
        st.session_state['min_k_deploy_agent_url'] = ""
    if 'min_k_deploy_agent_key' not in st.session_state:
        st.session_state['min_k_deploy_agent_key'] = ""
    if 'min_k_model_path' not in st.session_state:
        st.session_state['min_k_model_path'] = ""
    
    # Initialize mode-specific session state with defaults
    default_states = {
        "User Input": {
            'min_k_user_input_prompt': "",
            'min_k_user_input_percentage': 10,
            'min_k_user_input_max_tokens': 50,
            'min_k_user_input_last_result': None,
            'min_k_user_input_batch_results': [],
            'min_k_user_input_evaluation_results': None,
        },
        "Upload Document": {
            'min_k_upload_chunk_size': 200,
            'min_k_upload_chunk_count': 10,
            'min_k_upload_percentage': 10,
            'min_k_upload_max_tokens': 50,
            'min_k_upload_batch_data': None,
            'min_k_upload_batch_results': [],
            'min_k_upload_evaluation_results': None,
            'min_k_upload_dataset_name': None,
            'min_k_upload_dataset_length': None,
            'min_k_uploaded_text': None,
            'min_k_uploaded_file_id': None,
            'min_k_cached_chunk_size': None,
            'min_k_cached_chunk_count': None,
        },
        "Predefined Examples": {
            'min_k_predefined_dataset_type': 'WikiMIA',
            'min_k_predefined_wikimia_length': 64,
            'min_k_predefined_sample_count': 50,
            'min_k_predefined_percentage': 10,
            'min_k_predefined_max_tokens': 50,
            'min_k_predefined_batch_data': None,
            'min_k_predefined_batch_results': [],
            'min_k_predefined_evaluation_results': None,
            'min_k_predefined_dataset_name': None,
            'min_k_predefined_dataset_length': None,
        },
    }
    
    # Initialize defaults for all modes
    for mode, defaults in default_states.items():
        for key, value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = value
    
    # Page header with clear cache button
    header_col, button_col = st.columns([4, 1])
    with header_col:
        st.markdown('<h4 class="section-header">📊 MIN-K% PROB</h4>', unsafe_allow_html=True)
        st.markdown(
            "MIN-K% PROB computes the average negative log probability of the lowest k% tokens to detect if text was in the pretraining data. "
            "(Shi et al., 2024)"
        )
        with st.expander("📚 Reference", expanded=False):
            st.markdown("""
            **Shi, W., Ajith, A., Xia, M., Huang, Y., Liu, D., Blevins, T., Chen, D., & Zettlemoyer, L. (2024).**  
            Detecting Pretraining Data from Large Language Models.  
            *International Conference on Learning Representations (ICLR 2024)*, 51826-51843.  
            [Paper](https://proceedings.iclr.cc/paper_files/paper/2024/file/e32ad85fa27be4a9868d55703f01323e-Paper-Conference.pdf)
            """)
    with button_col:
        register_clear_cache_handler(MIN_K_CLEAR_CACHE_ID, _clear_min_k_cache)
        if st.button(
            "🗑️ Clear Cache",
            key="clear_min_k_cache",
            help="Reset cached MIN-K% PROB analysis results, dataset data, and evaluation results.",
        ):
            _clear_min_k_cache()
    
    # Server deployment agent configuration
    st.markdown("##### 🚀 Server Deployment Agent Configuration")
    st.caption("Configure the URL and API key of your server deployment agent (e.g., Cloudflare Tunnel URL)")
    col_url, col_key = st.columns([2, 1])
    with col_url:
        agent_url = st.text_input(
            "Deployment Agent URL",
            value=st.session_state.get('min_k_deploy_agent_url', ''),
            placeholder="https://cool-server-link.trycloudflare.com",
            help="The URL of your server deployment agent (from Cloudflare Tunnel or similar)",
            type="password",
            key="min_k_deploy_agent_url_input",
        )
    with col_key:
        agent_key = st.text_input(
            "Key",
            value=st.session_state.get('min_k_deploy_agent_key', ''),
            placeholder="YOUR_API_KEY",
            help="API key set on your server (YOUR_API_KEY environment variable)",
            type="password",
            key="min_k_deploy_agent_key_input",
        )
    if agent_url:
        st.session_state['min_k_deploy_agent_url'] = agent_url.strip()
    else:
        st.session_state['min_k_deploy_agent_url'] = ""
    if agent_key:
        st.session_state['min_k_deploy_agent_key'] = agent_key.strip()
    else:
        st.session_state['min_k_deploy_agent_key'] = ""
    
    model_path = st.text_input(
        "Model path",
        value=st.session_state['min_k_model_path'],
        placeholder="e.g. gpt2, Qwen/Qwen2.5-7B, or /path/to/local/model",
        help="Hugging Face model ID (e.g., 'gpt2') or absolute path to local model directory containing config.json",
        type="password",
        key="min_k_model_path_input",
    )
    st.session_state['min_k_model_path'] = model_path
    
    # Input mode selection
    previous_mode = st.session_state.get('min_k_input_mode', "Predefined Examples")
    input_mode = st.radio(
        "Input Mode",
        ["Predefined Examples", "User Input", "Upload Document"],
        horizontal=True,
        key="min_k_input_mode",
    )
    
    # Clear other modes' data when switching modes
    if input_mode != previous_mode:
        _clear_other_modes_data(input_mode)
        st.session_state['min_k_input_mode'] = input_mode
    
    if input_mode == "User Input":
        st.markdown("##### Input Prompt")
        prompt = st.text_area(
            "Prompt",
            value=st.session_state[_get_mode_key(input_mode, 'prompt')],
            height=180,
            placeholder="Enter one or multiple lines; each non-empty line will be treated as a separate prompt.",
            help="Multiple lines supported: each non-empty line is treated as one prompt.",
            key="min_k_user_input_prompt_input",
        )
        st.session_state[_get_mode_key(input_mode, 'prompt')] = prompt
        batch_data = None
    elif input_mode == "Upload Document":
        st.markdown("##### Upload Document")
        
        uploaded_file = st.file_uploader("Upload a text file (.txt) or PDF (.pdf)", type=["txt", "pdf"], key="min_k_upload_file")
        
        batch_data = None
        if uploaded_file is not None:
            try:
                # Check if file is new or parameters changed
                file_id = f"{uploaded_file.name}_{uploaded_file.size}"
                current_chunk_size = int(st.session_state.get(_get_mode_key(input_mode, 'chunk_size'), 200))
                current_chunk_count = int(st.session_state.get(_get_mode_key(input_mode, 'chunk_count'), 10))
                
                # Parse file if it's new or not cached
                if (st.session_state.get(_get_mode_key(input_mode, 'uploaded_file_id')) != file_id or 
                    st.session_state.get(_get_mode_key(input_mode, 'uploaded_text')) is None):
                    raw_bytes = uploaded_file.read()
                    text = ""
                    
                    # Parse based on file type
                    if uploaded_file.type == "application/pdf" or uploaded_file.name.endswith('.pdf'):
                        if not PDF_AVAILABLE:
                            st.error("❌ PDF parsing library not available. Please install PyPDF2 or pdfplumber: pip install PyPDF2 or pip install pdfplumber")
                            batch_data = None
                        else:
                            try:
                                # Try PyPDF2 first
                                try:
                                    import PyPDF2
                                    pdf_file = BytesIO(raw_bytes)
                                    pdf_reader = PyPDF2.PdfReader(pdf_file)
                                    text_parts = []
                                    for page in pdf_reader.pages:
                                        text_parts.append(page.extract_text())
                                    text = "\n".join(text_parts)
                                except:
                                    # Fallback to pdfplumber
                                    import pdfplumber
                                    with pdfplumber.open(BytesIO(raw_bytes)) as pdf:
                                        text_parts = []
                                        for page in pdf.pages:
                                            text_parts.append(page.extract_text())
                                        text = "\n".join(text_parts)
                            except Exception as e:
                                st.error(f"❌ Failed to parse PDF: {str(e)}")
                                batch_data = None
                    else:
                        # Text file
                        text = raw_bytes.decode('utf-8', errors='ignore')
                    
                    # Cache the parsed text and file ID
                    st.session_state[_get_mode_key(input_mode, 'uploaded_text')] = text
                    st.session_state[_get_mode_key(input_mode, 'uploaded_file_id')] = file_id
                else:
                    # Use cached text
                    text = st.session_state.get(_get_mode_key(input_mode, 'uploaded_text'), '')
                
                # Check if parameters changed
                cached_chunk_size = st.session_state.get('min_k_cached_chunk_size')
                cached_chunk_count = st.session_state.get('min_k_cached_chunk_count')
                params_changed = (cached_chunk_size != current_chunk_size or 
                                 cached_chunk_count != current_chunk_count)
                
                if text:
                    # Process chunks with current parameters
                    words = text.split()
                    size = max(1, current_chunk_size)
                    chunks = []
                    for i in range(0, len(words), size):
                        chunk_words = words[i:i+size]
                        if chunk_words:
                            chunks.append(" ".join(chunk_words))
                    if not chunks:
                        st.error("❌ No text content found after processing the upload.")
                    else:
                        sample_n = min(len(chunks), current_chunk_count)
                        # Always regenerate chunks when parameters change or file is new
                        random.seed(42)  # Fixed seed for reproducibility
                        sampled_chunks = random.sample(chunks, sample_n) if sample_n < len(chunks) else chunks
                        random.seed()  # Reset to random
                        upload_batch = [{"text": c, "label": 0} for c in sampled_chunks]
                        
                        st.session_state[_get_mode_key(input_mode, 'batch_data')] = upload_batch
                        st.session_state[_get_mode_key(input_mode, 'dataset_name')] = "Uploaded Document"
                        st.session_state[_get_mode_key(input_mode, 'dataset_length')] = f"{size} words/chunk"
                        st.session_state[_get_mode_key(input_mode, 'cached_chunk_size')] = current_chunk_size
                        st.session_state[_get_mode_key(input_mode, 'cached_chunk_count')] = current_chunk_count
                    batch_data = st.session_state.get(_get_mode_key(input_mode, 'batch_data'), None)
                else:
                    batch_data = None
            except Exception as e:
                st.error(f"❌ Failed to process uploaded file: {str(e)}")
                batch_data = None
        elif st.session_state.get(_get_mode_key(input_mode, 'uploaded_text')):
            batch_data = st.session_state.get(_get_mode_key(input_mode, 'batch_data'), None)
    else:  # Predefined Examples
        # Initialize batch_data
        batch_data = None
        
        st.markdown("**📚 Select predefined dataset**")
        st.caption("Load dataset for evaluation. Label 0 = non-member (unseen), Label 1 = member (seen in pretraining). Note: These labels are dataset-specific and do not necessarily reflect membership status for any particular model.")
        
        # Get current dataset type for layout (using mode-specific key)
        dataset_type_key = _get_mode_key(input_mode, 'dataset_type')
        current_dataset_type = st.session_state.get(dataset_type_key, 'WikiMIA')
        
        # Place all controls in the same row
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col1:
            # Dataset selection
            dataset_type = st.selectbox(
                "Choose dataset",
                options=["WikiMIA", "BookMIA"],
                index=0 if current_dataset_type == 'WikiMIA' else 1,
                help="Select a predefined dataset for evaluation",
                key="min_k_predefined_dataset_type_select",
            )
        
        # Clear batch_data and cache if dataset type or length changed
        dataset_type_key = _get_mode_key(input_mode, 'dataset_type')
        if st.session_state.get(dataset_type_key) != dataset_type:
            st.session_state.pop(_get_mode_key(input_mode, 'batch_data'), None)
            st.session_state.pop(_get_mode_key(input_mode, 'dataset_name'), None)
            st.session_state.pop(_get_mode_key(input_mode, 'dataset_length'), None)
            st.session_state.pop(_get_mode_key(input_mode, 'sample_count'), None)
            # Clear cache
            for key in list(st.session_state.keys()):
                if key.startswith('min_k_full_data_'):
                    st.session_state.pop(key, None)
        
        # Clear cache if length changed for WikiMIA
        if dataset_type == "WikiMIA":
            wikimia_length_key = _get_mode_key(input_mode, 'wikimia_length')
            selected_length = st.session_state.get(wikimia_length_key, 64)
            dataset_length_key = _get_mode_key(input_mode, 'dataset_length')
            if st.session_state.get(dataset_length_key) != selected_length:
                cache_key = f"min_k_full_data_wikimia_{selected_length}"
                if cache_key not in st.session_state:
                    # Clear old cache
                    for key in list(st.session_state.keys()):
                        if key.startswith('min_k_full_data_wikimia_'):
                            st.session_state.pop(key, None)
                # Clear batch_data when length changes
                st.session_state.pop(_get_mode_key(input_mode, 'batch_data'), None)
        
        st.session_state[dataset_type_key] = dataset_type
        
        # WikiMIA specific: Text length selection (BookMIA has fixed length but uses same layout)
        if dataset_type == "WikiMIA":
            length_options = [32, 64, 128, 256]
            
            with col2:
                wikimia_length_key = _get_mode_key(input_mode, 'wikimia_length')
                selected_length = st.selectbox(
                    "Select Text Length",
                    options=length_options,
                    index=length_options.index(st.session_state[wikimia_length_key]) if st.session_state[wikimia_length_key] in length_options else 1,
                    help="Select the length of text sequences in the WikiMIA dataset",
                    key=f"min_k_predefined_wikimia_length_select",
                )
                st.session_state[wikimia_length_key] = selected_length
        else:  # BookMIA
            with col2:
                st.selectbox(
                    "Select Text Length",
                    options=[512],
                    index=0,
                    disabled=True,
                    help="BookMIA uses fixed text length of 512",
                    key="min_k_bookmia_length_select",
                )
        
        # Sample count input (balanced labels for both datasets)
        with col3:
            sample_count_key = _get_mode_key(input_mode, 'sample_count')
            sample_count = st.number_input(
                "Sample count (balanced labels)",
                min_value=2,
                max_value=5000,
                value=int(st.session_state.get(sample_count_key, 50)),
                step=2,
                help="Total number of samples to load; will try to split evenly between label=0 and label=1.",
                key="min_k_predefined_sample_count_input",
            )
            st.session_state[sample_count_key] = int(sample_count)
        
        # Load full dataset for preview (cached)
        full_batch_data = None
        if dataset_type == "WikiMIA":
            selected_length = st.session_state.get(_get_mode_key(input_mode, 'wikimia_length'), 64)
            cache_key = f"min_k_full_data_wikimia_{selected_length}"
            if cache_key not in st.session_state:
                with st.spinner(f"Loading WikiMIA dataset (length={selected_length})..."):
                    try:
                        full_batch_data = load_wikimia_dataset(selected_length)
                        if full_batch_data:
                            st.session_state[cache_key] = full_batch_data
                        else:
                            st.error("❌ Failed to load WikiMIA dataset")
                    except Exception as e:
                        st.error(f"❌ Error loading WikiMIA dataset: {str(e)}")
            else:
                full_batch_data = st.session_state[cache_key]
        else:  # BookMIA
            cache_key = "min_k_full_data_bookmia"
            if cache_key not in st.session_state:
                with st.spinner("Loading BookMIA dataset..."):
                    try:
                        full_batch_data = load_bookmia_dataset()
                        if full_batch_data:
                            st.session_state[cache_key] = full_batch_data
                        else:
                            st.error("❌ Failed to load BookMIA dataset")
                    except Exception as e:
                        st.error(f"❌ Error loading BookMIA dataset: {str(e)}")
            else:
                full_batch_data = st.session_state[cache_key]
        
        # No status filtering; use full dataset
        filtered_batch_data = full_batch_data if full_batch_data else None
        
        # Preview dataset content
        if filtered_batch_data:
            dataset_length = st.session_state.get(_get_mode_key(input_mode, 'wikimia_length'), 64) if dataset_type == "WikiMIA" else 512
            
            with st.expander("📊 Preview Dataset Content", expanded=True):
                label_counts = pd.Series([ex.get('label', 0) for ex in filtered_batch_data]).value_counts().to_dict()
                label0 = label_counts.get(0, 0)
                label1 = label_counts.get(1, 0)
                if dataset_type == 'WikiMIA':
                    st.caption(f"📖 WikiMIA dataset ({len(filtered_batch_data)} examples) | label=0: {label0}, label=1: {label1}")
                else:
                    st.caption(f"📘 BookMIA dataset ({len(filtered_batch_data)} examples) | label=0: {label0}, label=1: {label1}")
                
                # Show preview of filtered dataset (all examples, indexed from 1)
                preview_data = []
                for idx, example in enumerate(filtered_batch_data, start=1):
                    label_text = "Member (seen in pretraining)" if example.get('label', 0) == 1 else "Non-member (unseen)"
                    label_badge = "🔴" if example.get('label', 0) == 1 else "🟢"
                    text_content = example.get('text', '')
                    
                    preview_data.append({
                        "Example #": idx,
                        "Label": f"{label_badge} {example.get('label', 0)}",
                        "Status": label_text,
                        "Text": text_content,
                    })
                
                # Display as table
                preview_df = pd.DataFrame(preview_data)
                # Set index to start from 1
                preview_df.index = range(1, len(preview_df) + 1)
                st.dataframe(preview_df, width='stretch', column_config={
                    "Text": st.column_config.TextColumn(
                        "Text",
                        width="large",
                    )
                })
                st.caption(f"Showing {len(filtered_batch_data)} examples | Text length: {dataset_length}")
            
            # Load Selected Examples button
            load_selected_button = st.button(
                "📥 Load Dataset",
                key="min_k_load_selected",
                width='stretch',
                disabled=wd(),
            )
            
            if load_selected_button:
                requested_total = int(st.session_state.get(_get_mode_key(input_mode, 'sample_count'), 50))
                if not filtered_batch_data:
                    st.error("❌ No data available to load.")
                else:
                    label0_data = [ex for ex in filtered_batch_data if ex.get('label', 0) == 0]
                    label1_data = [ex for ex in filtered_batch_data if ex.get('label', 0) == 1]
                    
                    if not label0_data or not label1_data:
                        st.error("❌ Need both label=0 and label=1 samples to create a balanced batch.")
                    else:
                        # Ensure balanced labels: each label gets the same number of samples
                        # Calculate how many samples we can get from each label
                        max_per_label = min(len(label0_data), len(label1_data))
                        # Target total should be even and not exceed 2 * max_per_label
                        max_total = 2 * max_per_label
                        target_total = min(requested_total, max_total)
                        # Make sure target_total is even for perfect balance
                        if target_total % 2 != 0:
                            target_total -= 1
                        
                        # Each label gets exactly half
                        num_per_label = target_total // 2
                        
                        if target_total < requested_total:
                            if target_total == 0:
                                st.error(f"❌ Cannot create balanced batch: insufficient samples. Available: label=0: {len(label0_data)}, label=1: {len(label1_data)}")
                            else:
                                st.warning(f"Requested {requested_total} samples, but only {num_per_label} per label available. Using {target_total} total ({num_per_label} per label) instead.")
                        
                        if num_per_label > 0:
                            selected_label0 = random.sample(label0_data, num_per_label)
                            selected_label1 = random.sample(label1_data, num_per_label)
                            selected_batch_data = selected_label0 + selected_label1
                            random.shuffle(selected_batch_data)
                            
                            st.session_state[_get_mode_key(input_mode, 'batch_data')] = selected_batch_data
                            st.session_state[_get_mode_key(input_mode, 'dataset_name')] = dataset_type
                            st.session_state[_get_mode_key(input_mode, 'dataset_length')] = dataset_length
                            
                            st.success(f"✅ Loaded {len(selected_batch_data)} examples (label=0: {len(selected_label0)}, label=1: {len(selected_label1)})")
                        else:
                            st.error(f"❌ Cannot create balanced batch: insufficient samples. Available: label=0: {len(label0_data)}, label=1: {len(label1_data)}")
        
        # Display loaded examples for detection
        batch_data = st.session_state.get(_get_mode_key(input_mode, 'batch_data'), None)
        if batch_data:
            dataset_name = st.session_state.get(_get_mode_key(input_mode, 'dataset_name'), dataset_type)
            dataset_length = st.session_state.get(_get_mode_key(input_mode, 'dataset_length'), 'N/A')
            
            st.markdown(f'<h4 class="section-header sm">📚 Predefined Examples</h4>', unsafe_allow_html=True)
            st.caption(f"Total examples: {len(batch_data)} | Dataset: {dataset_name} | Text length: {dataset_length}")
            
            # Display selected examples in expander
            with st.expander("📋 View Selected Examples", expanded=True):
                # Create table for all selected data
                selected_data = []
                for idx, example in enumerate(batch_data, start=1):
                    label_text = "Member (seen in pretraining)" if example.get('label', 0) == 1 else "Non-member (unseen)"
                    label_badge = "🔴" if example.get('label', 0) == 1 else "🟢"
                    text_content = example.get('text', '')
                    
                    selected_data.append({
                        "Example #": idx,
                        "Label": f"{label_badge} {example.get('label', 0)}",
                        "Status": label_text,
                        "Text": text_content,
                    })
                
                # Display as table with full text
                selected_df = pd.DataFrame(selected_data)
                # Set index to start from 1
                selected_df.index = range(1, len(selected_df) + 1)
                st.dataframe(selected_df, width='stretch', column_config={
                    "Text": st.column_config.TextColumn(
                        "Text",
                        width="large",
                    )
                })
                st.caption(f"Total examples: {len(batch_data)} | Columns: {', '.join(selected_df.columns.tolist())}")
        else:
            batch_data = None
        
        prompt = None
    
    st.markdown("##### Parameters")
    # Show 4 columns if Upload Document mode, otherwise 2 columns
    if input_mode == "Upload Document":
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            chunk_size = st.number_input(
                "Chunk size (words)",
                min_value=10,
                max_value=5000,
                value=int(st.session_state.get('min_k_upload_chunk_size', 200)),
                step=10,
                help="Number of words per chunk.",
                key="min_k_upload_chunk_size_input",
            )
            st.session_state['min_k_upload_chunk_size'] = int(chunk_size)
        with col2:
            chunk_count = st.number_input(
                "Chunk count",
                min_value=1,
                max_value=200,
                value=int(st.session_state.get('min_k_upload_chunk_count', 10)),
                step=1,
                help="How many chunks to sample.",
                key="min_k_upload_chunk_count_input",
            )
            st.session_state['min_k_upload_chunk_count'] = int(chunk_count)
        with col3:
            k_percentage = st.number_input(
                "K Percentage (%)",
                min_value=1,
                max_value=50,
                value=st.session_state[_get_mode_key(input_mode, 'percentage')],
                step=1,
                help="Percentage of lowest probability tokens to analyze (e.g., 10 means bottom 10%).",
                key="min_k_upload_percentage_input",
            )
            st.session_state[_get_mode_key(input_mode, 'percentage')] = k_percentage
        with col4:
            max_tokens = st.number_input(
                "Max Tokens",
                min_value=1,
                max_value=4000,
                value=st.session_state[_get_mode_key(input_mode, 'max_tokens')],
                step=50,
                help="Maximum number of tokens to generate.",
                key="min_k_upload_max_tokens_input",
            )
            st.session_state[_get_mode_key(input_mode, 'max_tokens')] = max_tokens
    else:
        col1, col2 = st.columns(2)
        with col1:
            k_percentage = st.number_input(
                "K Percentage (%)",
                min_value=1,
                max_value=50,
                value=st.session_state[_get_mode_key(input_mode, 'percentage')],
                step=1,
                help="Percentage of lowest probability tokens to analyze (e.g., 10 means bottom 10%).",
                key=f"min_k_{input_mode.lower().replace(' ', '_')}_percentage_input",
            )
            st.session_state[_get_mode_key(input_mode, 'percentage')] = k_percentage
        with col2:
            max_tokens = st.number_input(
                "Max Tokens",
                min_value=1,
                max_value=4000,
                value=st.session_state[_get_mode_key(input_mode, 'max_tokens')],
                step=50,
                help="Maximum number of tokens to generate.",
                key=f"min_k_{input_mode.lower().replace(' ', '_')}_max_tokens_input",
            )
            st.session_state[_get_mode_key(input_mode, 'max_tokens')] = max_tokens
    
    # Fixed decoding parameters (not configurable)
    temperature = 1.0
    top_p = 1.0
    
    # Show Chunks Preview for Upload Document mode (after parameters are set)
    if input_mode == "Upload Document":
        uploaded_text = st.session_state.get(_get_mode_key(input_mode, 'uploaded_text'))
        if uploaded_text:
            current_chunk_size = int(st.session_state.get(_get_mode_key(input_mode, 'chunk_size'), 200))
            current_chunk_count = int(st.session_state.get(_get_mode_key(input_mode, 'chunk_count'), 10))
            
            # Process chunks with current parameters
            words = uploaded_text.split()
            size = max(1, current_chunk_size)
            chunks = []
            for i in range(0, len(words), size):
                chunk_words = words[i:i+size]
                if chunk_words:
                    chunks.append(" ".join(chunk_words))
            
            if chunks:
                sample_n = min(len(chunks), current_chunk_count)
                # Use fixed seed for reproducibility
                random.seed(42)
                sampled_chunks = random.sample(chunks, sample_n) if sample_n < len(chunks) else chunks
                random.seed()  # Reset to random
                upload_batch = [{"text": c, "label": 0} for c in sampled_chunks]
                
                # Update batch data
                st.session_state[_get_mode_key(input_mode, 'batch_data')] = upload_batch
                st.session_state[_get_mode_key(input_mode, 'dataset_name')] = "Uploaded Document"
                st.session_state[_get_mode_key(input_mode, 'dataset_length')] = f"{size} words/chunk"
                
                # Display preview
                with st.expander(f"📋 Chunks Preview ({len(upload_batch)} chunks)", expanded=True):
                    for idx, chunk in enumerate(upload_batch, start=1):
                        st.markdown(f"**Chunk {idx}:**")
                        st.caption(chunk['text'][:500] + ("..." if len(chunk['text']) > 500 else ""))
    
    submit_button = render_run_button(
        "MIN-K% PROB Analysis",
        "run_min_k_prob_analysis_button",
        "📊 Run: MIN-K% PROB Analysis",
        help="Run the MIN-K% PROB analysis with the specified parameters.",
    )
    
    result = st.session_state.get('min_k_last_result')
    evaluation_results = st.session_state.get('min_k_evaluation_results')
    
    if submit_button:
        set_active_clear_cache_id(MIN_K_CLEAR_CACHE_ID)
        if input_mode == "User Input":
            prompt_lines = [ln.strip() for ln in (prompt or "").splitlines() if ln.strip()]
            if not prompt_lines:
                st.warning("⚠️ Please enter a prompt before running the analysis.")
            elif not model_path.strip():
                st.warning("⚠️ Please provide the model path before running the analysis.")
            else:
                if len(prompt_lines) == 1:
                    _run_single_analysis(
                        prompt_lines[0], model_path, api_key, model_choice, provider,
                        k_percentage, temperature, top_p, max_tokens, input_mode,
                    )
                    # Clear batch evaluation states
                    st.session_state.pop(_get_mode_key(input_mode, 'evaluation_results'), None)
                    st.session_state.pop(_get_mode_key(input_mode, 'batch_results'), None)
                else:
                    # Treat multiple lines as batch data (labels set to 0 by default)
                    user_batch = [{"text": p, "label": 0} for p in prompt_lines]
                    st.session_state['min_k_batch_data'] = user_batch
                    st.session_state['min_k_dataset_name'] = "User Input"
                    st.session_state['min_k_dataset_length'] = "N/A"
                    st.session_state[_get_mode_key(input_mode, 'batch_data')] = user_batch
                    st.session_state[_get_mode_key(input_mode, 'dataset_name')] = "User Input"
                    st.session_state[_get_mode_key(input_mode, 'dataset_length')] = "N/A"
                    _run_batch_evaluation(
                        user_batch, model_path, api_key, model_choice, provider,
                        k_percentage, max_tokens, input_mode,
                    )
                    # Clear single prompt result
                    st.session_state.pop(_get_mode_key(input_mode, 'last_result'), None)
        elif input_mode == "Upload Document":
            effective_batch = batch_data or st.session_state.get(_get_mode_key(input_mode, 'batch_data'))
            if effective_batch is None or len(effective_batch) == 0:
                st.warning("⚠️ Please upload a document and generate chunks before running the analysis.")
            elif not model_path.strip():
                st.warning("⚠️ Please provide the model path before running the analysis.")
            else:
                _run_batch_evaluation(
                    effective_batch, model_path, api_key, model_choice, provider,
                    k_percentage, max_tokens, input_mode,
                )
        else:  # Predefined Examples
            if batch_data is None or len(batch_data) == 0:
                st.warning("⚠️ Please load dataset before running the analysis.")
            elif not model_path.strip():
                st.warning("⚠️ Please provide the model path before running the analysis.")
            else:
                _run_batch_evaluation(
                    batch_data, model_path, api_key, model_choice, provider,
                    k_percentage, max_tokens, input_mode,
                )
    
    # Display results using mode-specific keys
    if input_mode == "User Input":
        result = st.session_state.get(_get_mode_key(input_mode, 'last_result'))
        evaluation_results = st.session_state.get(_get_mode_key(input_mode, 'evaluation_results'))
        if result and not evaluation_results:
            _display_single_result(result, k_percentage)
        elif evaluation_results:
            dataset_name = st.session_state.get(_get_mode_key(input_mode, 'dataset_name'), 'User Input')
            # Use model_path if cloudflared is configured, otherwise use model_choice from sidebar
            agent_url = st.session_state.get('min_k_deploy_agent_url', '').strip()
            effective_model_path = model_path if agent_url else None
            _display_evaluation_results(evaluation_results, model_choice, provider, dataset_name, k_percentage, input_mode=input_mode, model_path=effective_model_path)
    elif input_mode == "Upload Document":
        evaluation_results = st.session_state.get(_get_mode_key(input_mode, 'evaluation_results'))
        if evaluation_results:
            dataset_name = st.session_state.get(_get_mode_key(input_mode, 'dataset_name'), 'Uploaded Document')
            # Use model_path if cloudflared is configured, otherwise use model_choice from sidebar
            agent_url = st.session_state.get('min_k_deploy_agent_url', '').strip()
            effective_model_path = model_path if agent_url else None
            _display_evaluation_results(evaluation_results, model_choice, provider, dataset_name, k_percentage, input_mode=input_mode, model_path=effective_model_path)
    else:  # Predefined Examples
        evaluation_results = st.session_state.get(_get_mode_key(input_mode, 'evaluation_results'))
        if evaluation_results:
            dataset_name = st.session_state.get(_get_mode_key(input_mode, 'dataset_name'), 'Unknown')
            # Use model_path if cloudflared is configured, otherwise use model_choice from sidebar
            agent_url = st.session_state.get('min_k_deploy_agent_url', '').strip()
            effective_model_path = model_path if agent_url else None
            _display_evaluation_results(evaluation_results, model_choice, provider, dataset_name, k_percentage, input_mode=input_mode, model_path=effective_model_path)


def _run_single_analysis(
    prompt, model_path, api_key, model_choice, provider,
    k_percentage, temperature, top_p, max_tokens, input_mode: str = "User Input",
):
    """Run single prompt analysis."""
    # Auto-deploy model if Deployment Agent URL is configured
    agent_url = st.session_state.get('min_k_deploy_agent_url', '').strip()
    agent_key = st.session_state.get('min_k_deploy_agent_key', '').strip()
    if agent_url:
        model_path_stripped = model_path.strip()
        
        # Prepare headers with API key if provided
        deploy_headers = {}
        if agent_key:
            deploy_headers["X-API-Key"] = agent_key
        
        # Deploy model
        if model_path_stripped:
            with st.spinner("Sending deployment request for model..."):
                try:
                    response = requests.post(
                        f"{agent_url}/deploy",
                        json={"model_path": model_path_stripped},
                        headers=deploy_headers,
                        timeout=10
                    )
                    if response.status_code == 200:
                        res_json = response.json()
                        if res_json.get("status") == "success":
                            st.success(f"✅ Model deployment initiated: {res_json.get('message', '')}")
                        else:
                            st.warning(f"⚠️ Model deployment warning: {res_json.get('message', 'Unknown error')}")
                    elif response.status_code == 401:
                        st.error(f"❌ Model deployment failed (401): Authentication failed. Please check your API key in the 'Key' field.")
                    elif response.status_code == 403:
                        st.error(f"❌ Model deployment failed (403): Invalid API key. Please check your API key in the 'Key' field.")
                    elif response.status_code == 530:
                        st.warning(
                            f"⚠️ Model deployment failed (530): Cloudflare Tunnel cannot reach the server. "
                            f"Please check:\n"
                            f"1. Is `deploy_agent.py` running on the server?\n"
                            f"2. Is Cloudflare Tunnel running and connected?\n"
                            f"3. Is the Tunnel URL still valid? (Tunnel URLs may expire)\n"
                            f"Continuing with analysis..."
                        )
                    else:
                        st.warning(f"⚠️ Model deployment failed with status code: {response.status_code}. Continuing with analysis...")
                except requests.exceptions.Timeout:
                    st.warning("⏱️ Model deployment timeout. The server may be slow or unreachable. Continuing with analysis...")
                except requests.exceptions.ConnectionError as e:
                    st.warning(f"🔌 Unable to connect to deployment agent: {str(e)}. Continuing with analysis...")
                except Exception as e:
                    st.warning(f"⚠️ Model deployment error: {str(e)}. Continuing with analysis...")
    
    # Validate model path
    model_path_stripped = model_path.strip()
    model_valid = False
    
    # Check if it's a Hugging Face model ID (contains slash or is simple name)
    if '/' in model_path_stripped or model_path_stripped in ['gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl']:
        model_valid = True
    # Check if it's a local directory with config.json
    elif os.path.isdir(model_path_stripped) and os.path.exists(os.path.join(model_path_stripped, 'config.json')):
        model_valid = True
    else:
        st.error(f"❌ Model path '{model_path_stripped}' is not valid. Use a Hugging Face model ID (e.g., 'gpt2') or a local directory containing config.json")
    
    if model_valid:
        # Get agent_url and agent_key from session state if configured
        agent_url = st.session_state.get('min_k_deploy_agent_url', '').strip()
        agent_key = st.session_state.get('min_k_deploy_agent_key', '').strip()

        with detection_job("MIN-K% PROB Analysis"):
            with st.spinner("🔄 Running MIN-K% PROB analysis..."):
                try:
                    # If using agent_url, use model_path as model_name (vLLM typically uses path or served-model-name)
                    # Otherwise use model_choice from sidebar
                    effective_model_name = model_path_stripped if agent_url else model_choice

                    result = run_min_k_prob_analysis(
                        prompt=prompt,
                        api_key=api_key,
                        model_name=effective_model_name,
                        provider=provider,
                        k_percentage=k_percentage,
                        temperature=temperature,
                        top_p=top_p,
                        max_tokens=max_tokens,
                        agent_url=agent_url if agent_url else None,
                        agent_key=agent_key if agent_key else None,
                    )
                    st.session_state[_get_mode_key(input_mode, 'last_result')] = result
                except Exception as e:
                    st.error(f"❌ Error running analysis: {str(e)}")
                    st.session_state[_get_mode_key(input_mode, 'last_result')] = None


def _run_batch_evaluation(
    batch_data, model_path, api_key, model_choice, provider,
    k_percentage, max_tokens, input_mode: str = "Predefined Examples",
):
    """Run batch evaluation on multiple examples."""
    # Fixed decoding parameters (temperature/top_p not configurable)
    temperature = 1.0
    top_p = 1.0
    # Validate model path (same as single analysis)
    model_path_stripped = model_path.strip()
    model_valid = False
    
    if '/' in model_path_stripped or model_path_stripped in ['gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl']:
        model_valid = True
    elif os.path.isdir(model_path_stripped) and os.path.exists(os.path.join(model_path_stripped, 'config.json')):
        model_valid = True
    else:
        st.error(f"❌ Model path '{model_path_stripped}' is not valid.")
        return
    
    if not model_valid:
        return

    with detection_job("MIN-K% PROB Batch Analysis"):
        # Process batch data
        all_output = []
        progress_bar = st.progress(0)
        status_text = st.empty()

        for idx, ex in enumerate(batch_data):
            text = ex.get('text', '')
            label = ex.get('label', None)

            if not text:
                continue

            if label is None:
                st.warning(f"⚠️ Example {idx+1} missing 'label' field. Skipping.")
                continue

            status_text.text(f"Processing example {idx+1}/{len(batch_data)}...")
            progress_bar.progress((idx + 1) / len(batch_data))

            try:
                agent_url = st.session_state.get('min_k_deploy_agent_url', '').strip()
                agent_key = st.session_state.get('min_k_deploy_agent_key', '').strip()
                effective_model_name = model_path_stripped if agent_url else model_choice

                result = run_min_k_prob_analysis(
                    prompt=text,
                    api_key=api_key,
                    model_name=effective_model_name,
                    provider=provider,
                    k_percentage=k_percentage,
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=max_tokens,
                    agent_url=agent_url if agent_url else None,
                    agent_key=agent_key if agent_key else None,
                )

                pred_dict = {
                    "ppl": result.get('perplexity', 0.0),
                    "Min_k%_Prob": result['min_k_prob'],
                }

                if 'min_k_probs' in result:
                    for key, value in result['min_k_probs'].items():
                        pred_dict[key] = value

                if result.get('ppl_lowercase') is not None:
                    pred_dict["ppl/lowercase_ppl"] = result['ppl_lowercase']
                if result.get('ppl_zlib') is not None:
                    pred_dict["ppl/zlib"] = result['ppl_zlib']

                all_output.append({
                    "text": text,
                    "label": label,
                    "pred": pred_dict,
                    "result": result,
                })
                st.session_state[_get_mode_key(input_mode, 'batch_results')] = list(all_output)
            except Exception as e:
                st.warning(f"⚠️ Error processing example {idx+1}: {str(e)}")
                continue

        progress_bar.empty()
        status_text.empty()

        if all_output:
            evaluation_results = compute_evaluation_metrics(all_output)
            st.session_state[_get_mode_key(input_mode, 'evaluation_results')] = evaluation_results
            st.session_state[_get_mode_key(input_mode, 'batch_results')] = all_output
        else:
            st.error("❌ No valid results to evaluate.")


def sweep(score, x):
    """
    Compute a ROC curve and then return the FPR, TPR, AUC, and ACC.
    """
    if not SKLEARN_AVAILABLE or not NUMPY_AVAILABLE:
        raise ImportError("scikit-learn and numpy are required for evaluation metrics")
    
    # Check if we have both positive and negative samples
    x_array = np.array(x, dtype=bool)
    if np.sum(x_array) == 0 or np.sum(x_array) == len(x_array):
        # All samples are the same class (no positive samples or no negative samples)
        # Return default values to avoid sklearn warnings
        fpr = np.array([0.0, 1.0])
        tpr = np.array([0.0, 1.0])
        auc_score = 0.5  # Random classifier performance
        acc = 0.5  # Random classifier accuracy
        return fpr, tpr, auc_score, acc
    
    # Suppress warnings for edge cases
    import warnings
    with warnings.catch_warnings():
        # Suppress UndefinedMetricWarning from sklearn
        warnings.filterwarnings("ignore", message="No positive samples in y_true")
        warnings.filterwarnings("ignore", message="No negative samples in y_true")
        fpr, tpr, _ = roc_curve(x, -np.array(score))
    acc = np.max(1 - (fpr + (1 - tpr)) / 2)
    return fpr, tpr, auc(fpr, tpr), acc


def do_plot(prediction, answers, sweep_fn=sweep, metric='auc', legend="", output_dir=None):
    """
    Generate the ROC curves by using ntest models as test models and the rest to train.
    """
    fpr, tpr, auc_score, acc = sweep_fn(np.array(prediction), np.array(answers, dtype=bool))
    
    low = 0.0
    fpr_indices = np.where(fpr < 0.05)
    if len(fpr_indices[0]) > 0:
        low = tpr[fpr_indices[0][-1]]
    
    metric_text = ''
    if metric == 'auc':
        metric_text = 'auc=%.3f' % auc_score
    elif metric == 'acc':
        metric_text = 'acc=%.3f' % acc
    
    return legend, auc_score, acc, low, fpr, tpr


def compute_evaluation_metrics(all_output):
    """Compute evaluation metrics using the sweep and do_plot functions."""
    if not SKLEARN_AVAILABLE or not NUMPY_AVAILABLE:
        return {
            "error": "scikit-learn and numpy are required for evaluation metrics. Please install them: pip install scikit-learn numpy"
        }
    
    answers = []
    metric2predictions = defaultdict(list)
    
    for ex in all_output:
        answers.append(ex["label"])
        for metric in ex["pred"].keys():
            if ("raw" in metric) and ("clf" not in metric):
                continue
            metric2predictions[metric].append(ex["pred"][metric])
    
    if not answers or not metric2predictions:
        return {"error": "No valid predictions found."}
    
    # Check if we have both positive and negative samples
    answers_array = np.array(answers, dtype=bool)
    unique_labels = np.unique(answers_array)
    has_mixed_labels = len(unique_labels) > 1
    
    if not has_mixed_labels:
        # All samples are the same class - cannot compute meaningful ROC curve
        label_value = int(answers_array[0]) if len(answers_array) > 0 else 0
        label_name = "member (seen in pretraining)" if label_value == 1 else "non-member (unseen)"
        return {
            "error": f"⚠️ All samples have the same label (label={label_value}, {label_name}). ROC curve and AUC cannot be computed with only one class. Please load samples with both label=0 (non-member) and label=1 (member) for meaningful evaluation.",
            "num_examples": len(answers),
            "all_labels_same": True,
            "label_value": label_value,
        }
    
    # Compute metrics for each metric type
    results = {}
    for metric, predictions in metric2predictions.items():
        legend, auc_score, acc, low, fpr, tpr = do_plot(
            predictions, answers, legend=metric, metric='auc'
        )
        results[metric] = {
            "legend": legend,
            "auc": float(auc_score),
            "accuracy": float(acc),
            "tpr_at_5fpr": float(low),
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist(),
        }
    
    # Use the first metric's results as primary
    primary_metric = list(metric2predictions.keys())[0]
    primary_results = results[primary_metric]
    
    return {
        "fpr": primary_results["fpr"],
        "tpr": primary_results["tpr"],
        "auc": primary_results["auc"],
        "accuracy": primary_results["accuracy"],
        "tpr_at_5fpr": primary_results["tpr_at_5fpr"],
        "predictions": metric2predictions[primary_metric],
        "answers": answers,
        "num_examples": len(answers),
        "all_metrics": results,
    }


def _display_single_result(result, k_percentage):
    """Display single analysis result (simplified for User Input mode)."""
    st.divider()
    st.markdown('<p class="analysis-step-label">Results</p>', unsafe_allow_html=True)
    
    # Use styled metrics similar to content recall detection
    # Calculate perplexity display value
    perplexity_value = "∞" if result['perplexity'] == float('inf') else f"{result['perplexity']:.2f}"
    
    st.markdown(
        f"""
        <div class="analysis-quick-stats">
            <div class="analysis-stat">
                <div class="analysis-stat__label">MIN-K% PROB</div>
                <div class="analysis-stat__value">{result['min_k_prob']:.4f}</div>
                <div class="analysis-stat__hint">Negative mean of the lowest {k_percentage}% log probabilities</div>
            </div>
            <div class="analysis-stat">
                <div class="analysis-stat__label">Perplexity</div>
                <div class="analysis-stat__value">{perplexity_value}</div>
                <div class="analysis-stat__hint">Perplexity of the input text</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    # Additional details in a single expander
    with st.expander("📊 Additional Details", expanded=False):
        # All Min-k% Prob metrics
        if 'min_k_probs' in result and result['min_k_probs']:
            st.markdown('<p class="analysis-step-caption">Min-k% Prob for different k values:</p>', unsafe_allow_html=True)
            # Build HTML for min-k% prob metrics
            min_k_html_parts = []
            for key, value in result['min_k_probs'].items():
                label = key.replace("Min_", "").replace("_Prob", "")
                min_k_html_parts.append(
                    f'<div class="analysis-stat">'
                    f'<div class="analysis-stat__label">{label}</div>'
                    f'<div class="analysis-stat__value">{value:.4f}</div>'
                    f'</div>'
                )
            st.markdown(
                f'<div class="analysis-quick-stats">{"".join(min_k_html_parts)}</div>',
                unsafe_allow_html=True,
            )
        
        # Additional metrics
        additional_metrics = []
        if result.get('ppl_lowercase') is not None:
            additional_metrics.append(("ppl/lowercase_ppl", result['ppl_lowercase']))
        if result.get('ppl_zlib') is not None:
            additional_metrics.append(("ppl/zlib", result['ppl_zlib']))
        
        if additional_metrics:
            st.markdown('<p class="analysis-step-caption">Additional Metrics:</p>', unsafe_allow_html=True)
            # Build HTML for additional metrics
            additional_html_parts = []
            for key, value in additional_metrics:
                additional_html_parts.append(
                    f'<div class="analysis-stat">'
                    f'<div class="analysis-stat__label">{key}</div>'
                    f'<div class="analysis-stat__value">{value:.4f}</div>'
                    f'</div>'
                )
            st.markdown(
                f'<div class="analysis-quick-stats">{"".join(additional_html_parts)}</div>',
                unsafe_allow_html=True,
            )
        
        # Statistics
        st.markdown('<p class="analysis-step-caption">Statistics:</p>', unsafe_allow_html=True)
        stats = [
            ("Total Tokens", str(result['total_tokens'])),
            ("K Tokens Analyzed", str(result['k_tokens_count'])),
            ("Avg Log Prob", f"{result['overall_avg_logprob']:.4f}"),
        ]
        # Build HTML for statistics
        stats_html_parts = []
        for label, value in stats:
            stats_html_parts.append(
                f'<div class="analysis-stat">'
                f'<div class="analysis-stat__label">{label}</div>'
                f'<div class="analysis-stat__value">{value}</div>'
                f'</div>'
            )
        st.markdown(
            f'<div class="analysis-quick-stats">{"".join(stats_html_parts)}</div>',
            unsafe_allow_html=True,
        )


def _display_evaluation_results(evaluation_results, model_choice: str, provider: str, dataset_name: str = "Unknown", k_percentage: int = 10, input_mode: Optional[str] = None, model_path: Optional[str] = None):
    """Display batch evaluation results with ROC curve.
    
    Args:
        evaluation_results: Evaluation results dictionary
        model_choice: Model name from sidebar (used if model_path is not provided)
        provider: Provider name from sidebar
        dataset_name: Name of the dataset
        k_percentage: K percentage value
        input_mode: Input mode (User Input, Upload Document, or Predefined Examples)
        model_path: Optional custom model path (e.g., from cloudflared). If provided, this will be used in PDF instead of model_choice.
    """
    if "error" in evaluation_results:
        batch_results = st.session_state.get(_get_mode_key(input_mode or "User Input", 'batch_results'), [])
        # For User Input and Upload Document, just show per-example metrics without warning
        if input_mode in ["User Input", "Upload Document"]:
            st.divider()
            st.markdown('<p class="analysis-step-label">Results</p>', unsafe_allow_html=True)
            st.markdown("**📊 Batch Analysis Results**")
            if batch_results:
                with st.expander("📋 Per-example Metrics", expanded=True):
                    try:
                        import pandas as pd
                        rows = []
                        for idx, item in enumerate(batch_results, start=1):
                            pred = item.get("pred", {})
                            rows.append({
                                "Example #": idx,
                                "Label": item.get("label", ""),
                                "ppl": pred.get("ppl"),
                                "ppl/lowercase_ppl": pred.get("ppl/lowercase_ppl"),
                                "ppl/zlib": pred.get("ppl/zlib"),
                                "Min_5%": pred.get("Min_5%_Prob"),
                                "Min_10%": pred.get("Min_10%_Prob"),
                                "Min_20%": pred.get("Min_20%_Prob"),
                                "Min_30%": pred.get("Min_30%_Prob"),
                                "Min_40%": pred.get("Min_40%_Prob"),
                                "Min_50%": pred.get("Min_50%_Prob"),
                                "Min_60%": pred.get("Min_60%_Prob"),
                                "Text (preview)": (item.get("text") or "")[:160] + ("..." if item.get("text") and len(item.get("text")) > 160 else "")
                            })
                        if rows:
                            df = pd.DataFrame(rows)
                            st.dataframe(df, width='stretch', column_config={
                                "Text (preview)": st.column_config.TextColumn("Text (preview)", width="large")
                            })
                    except Exception as e:
                        st.error(f"Error displaying per-example metrics: {str(e)}")
            return
        else:
            # For Predefined Examples, show warning and per-example metrics
            st.warning(f"⚠️ {evaluation_results['error']}")
            if batch_results:
                with st.expander("📋 Per-example Metrics", expanded=True):
                    try:
                        import pandas as pd
                        rows = []
                        for idx, item in enumerate(batch_results, start=1):
                            pred = item.get("pred", {})
                            rows.append({
                                "Example #": idx,
                                "Label": item.get("label", ""),
                                "ppl": pred.get("ppl"),
                                "ppl/lowercase_ppl": pred.get("ppl/lowercase_ppl"),
                                "ppl/zlib": pred.get("ppl/zlib"),
                                "Min_5%": pred.get("Min_5%_Prob"),
                                "Min_10%": pred.get("Min_10%_Prob"),
                                "Min_20%": pred.get("Min_20%_Prob"),
                                "Min_30%": pred.get("Min_30%_Prob"),
                                "Min_40%": pred.get("Min_40%_Prob"),
                                "Min_50%": pred.get("Min_50%_Prob"),
                                "Min_60%": pred.get("Min_60%_Prob"),
                                "Text (preview)": (item.get("text") or "")[:160] + ("..." if item.get("text") and len(item.get("text")) > 160 else "")
                            })
                        if rows:
                            df = pd.DataFrame(rows)
                            st.dataframe(df, width='stretch', column_config={
                                "Text (preview)": st.column_config.TextColumn("Text (preview)", width="large")
                            })
                    except Exception as e:
                        st.error(f"Error displaying per-example metrics: {str(e)}")
            return
    
    batch_results = st.session_state.get(_get_mode_key(input_mode or "Predefined Examples", 'batch_results'), [])
    
    st.divider()
    st.markdown('<p class="analysis-step-label">Results</p>', unsafe_allow_html=True)
    st.markdown("**📊 Batch Evaluation Results**")
    st.caption(
        "Evaluation metrics for MIN-K% PROB on batch data with member/non-member labels."
    )
    
    # Main metrics in columns (using smaller font for values)
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("**AUC**")
        st.markdown(f'<p style="font-size: 18px; margin-top: -10px;">{evaluation_results["auc"]:.4f}</p>', unsafe_allow_html=True)
        st.caption("Area Under the ROC Curve")
    with col2:
        st.markdown("**Accuracy**")
        st.markdown(f'<p style="font-size: 18px; margin-top: -10px;">{evaluation_results["accuracy"]:.4f}</p>', unsafe_allow_html=True)
        st.caption("Optimal accuracy based on ROC curve")
    with col3:
        st.markdown("**TPR@5%FPR**")
        st.markdown(f'<p style="font-size: 18px; margin-top: -10px;">{evaluation_results["tpr_at_5fpr"]:.4f}</p>', unsafe_allow_html=True)
        st.caption("True Positive Rate at 5% False Positive Rate")
    with col4:
        st.markdown("**Total Examples**")
        st.markdown(f'<p style="font-size: 18px; margin-top: -10px;">{evaluation_results["num_examples"]}</p>', unsafe_allow_html=True)
    
    # Per-example metrics table (first)
    current_input_mode = input_mode or st.session_state.get('min_k_input_mode', 'Predefined Examples')
    if batch_results and current_input_mode == "Predefined Examples":
        with st.expander("📋 Per-example Metrics", expanded=False):
            try:
                import pandas as pd
                rows = []
                for idx, item in enumerate(batch_results, start=1):
                    pred = item.get("pred", {})
                    rows.append({
                        "Example #": idx,
                        "Label": item.get("label", ""),
                        "ppl": pred.get("ppl"),
                        "ppl/lowercase_ppl": pred.get("ppl/lowercase_ppl"),
                        "ppl/zlib": pred.get("ppl/zlib"),
                        "Min_5%": pred.get("Min_5%_Prob"),
                        "Min_10%": pred.get("Min_10%_Prob"),
                        "Min_20%": pred.get("Min_20%_Prob"),
                        "Min_30%": pred.get("Min_30%_Prob"),
                        "Min_40%": pred.get("Min_40%_Prob"),
                        "Min_50%": pred.get("Min_50%_Prob"),
                        "Min_60%": pred.get("Min_60%_Prob"),
                        "Text (preview)": (item.get("text") or "")[:160] + ("..." if item.get("text") and len(item.get("text")) > 160 else "")
                    })
                if rows:
                    df = pd.DataFrame(rows)
                    st.dataframe(
                        df,
                        width='stretch',
                        column_config={
                            "Text (preview)": st.column_config.TextColumn("Text (preview)", width="large")
                        }
                    )
                else:
                    st.info("No per-example metrics available.")
            except ImportError:
                st.info("Install pandas to view per-example metrics table (pip install pandas).")

    # Detailed metrics summary (second)
    with st.expander("📄 Detailed Metrics Summary", expanded=False):
        # Display primary metric
        metrics_text = f"""
**Primary Metric (MIN-K% PROB) Evaluation Results:**
- AUC: {evaluation_results['auc']:.4f}
- Accuracy: {evaluation_results['accuracy']:.4f}
- TPR@5%FPR: {evaluation_results['tpr_at_5fpr']:.4f}
- Total Examples: {evaluation_results['num_examples']}
        """
        st.text(metrics_text)
        
        # Display all metrics if available
        if 'all_metrics' in evaluation_results and evaluation_results['all_metrics']:
            st.markdown("**All Metrics Comparison:**")
            all_metrics_data = []
            for metric_name, metric_data in evaluation_results['all_metrics'].items():
                all_metrics_data.append({
                    "Metric": metric_name,
                    "AUC": f"{metric_data['auc']:.4f}",
                    "Accuracy": f"{metric_data['accuracy']:.4f}",
                    "TPR@5%FPR": f"{metric_data['tpr_at_5fpr']:.4f}",
                })
            if all_metrics_data:
                try:
                    import pandas as pd
                    st.dataframe(pd.DataFrame(all_metrics_data), width='stretch')
                except ImportError:
                    # Fallback to markdown table
                    st.markdown("| Metric | AUC | Accuracy | TPR@5%FPR |")
                    st.markdown("|--------|-----|----------|-----------|")
                    for row in all_metrics_data:
                        st.markdown(f"| {row['Metric']} | {row['AUC']} | {row['Accuracy']} | {row['TPR@5%FPR']} |")
    
    # ROC curve (third)
    if MATPLOTLIB_AVAILABLE and NUMPY_AVAILABLE and 'fpr' in evaluation_results and 'tpr' in evaluation_results:
        with st.expander("📈 ROC Curve", expanded=True):
            try:
                fig, ax = plt.subplots(figsize=(6, 5))
                
                fpr = np.array(evaluation_results['fpr'])
                tpr = np.array(evaluation_results['tpr'])
                auc_score = evaluation_results['auc']
                
                legend_text = f"MIN-K% PROB (AUC={auc_score:.4f})"
                # Avoid log-scale issues when FPR/TPR contain zeros by clipping to a tiny epsilon for plotting only
                plot_fpr = np.clip(fpr, 1e-8, 1.0)
                plot_tpr = np.clip(tpr, 1e-8, 1.0)
                ax.plot(plot_fpr, plot_tpr, label=legend_text, linewidth=2)
                ax.plot([0, 1], [0, 1], ls='--', color='gray', label='Random')
                
                ax.set_xscale('log')
                ax.set_yscale('log')
                ax.set_xlim(1e-5, 1)
                ax.set_ylim(1e-5, 1)
                ax.set_xlabel("False Positive Rate", fontsize=10)
                ax.set_ylabel("True Positive Rate", fontsize=10)
                ax.legend(fontsize=9)
                ax.grid(True, alpha=0.3)
                
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)
            except Exception as e:
                st.warning(f"⚠️ Error plotting ROC curve: {str(e)}")

    # Generate PDF Report
    should_generate_pdf = batch_results and ("error" not in evaluation_results)
    if should_generate_pdf:
        try:
            # Use model_path if provided (e.g., from cloudflared), otherwise use model_choice from sidebar
            effective_model_name = model_path.strip() if model_path and model_path.strip() else model_choice
            
            # Prepare PDF data
            pdf_data = {
                'evaluation_results': evaluation_results,
                'batch_results': batch_results,
                'model_choice': effective_model_name,
                'provider': provider,
                'dataset_name': dataset_name,
                'k_percentage': k_percentage,
            }
            
            # Build plots for PDF (e.g., ROC curve)
            pdf_plots: Dict[str, bytes] = {}
            if MATPLOTLIB_AVAILABLE and NUMPY_AVAILABLE and 'fpr' in evaluation_results and 'tpr' in evaluation_results:
                try:
                    fig, ax = plt.subplots(figsize=(6, 5))
                    fpr = np.array(evaluation_results['fpr'])
                    tpr = np.array(evaluation_results['tpr'])
                    auc_score = float(evaluation_results.get('auc', 0.0) or 0.0)
                    legend_text = f"MIN-K% PROB (AUC={auc_score:.4f})"

                    # Avoid log-scale issues when FPR/TPR contain zeros by clipping to a tiny epsilon for plotting only
                    plot_fpr = np.clip(fpr, 1e-8, 1.0)
                    plot_tpr = np.clip(tpr, 1e-8, 1.0)
                    ax.plot(plot_fpr, plot_tpr, label=legend_text, linewidth=2)
                    ax.plot([0, 1], [0, 1], ls='--', color='gray', label='Random')

                    ax.set_xscale('log')
                    ax.set_yscale('log')
                    ax.set_xlim(1e-5, 1)
                    ax.set_ylim(1e-5, 1)
                    ax.set_xlabel("False Positive Rate", fontsize=10)
                    ax.set_ylabel("True Positive Rate", fontsize=10)
                    ax.legend(fontsize=9)
                    ax.grid(True, alpha=0.3)
                    plt.tight_layout()

                    buf = BytesIO()
                    fig.savefig(buf, format="png", dpi=180, bbox_inches="tight")
                    buf.seek(0)
                    pdf_plots["ROC Curve (log-log)"] = buf.getvalue()
                    plt.close(fig)
                except Exception:
                    pass

            # Generate PDF report
            pdf_bytes = generate_min_k_prob_pdf_report(
                evaluation_results=evaluation_results,
                batch_results=batch_results,
                model_choice=effective_model_name,
                provider=provider,
                dataset_name=dataset_name,
                k_percentage=k_percentage,
                plots=pdf_plots or None,
            )
            
            # PDF Preview
            render_pdf_preview_with_blob(
                pdf_bytes,
                title="📋 Audit Report Preview",
                iframe_height=450,
                download_filename=f"min_k_prob_analysis_{dataset_name.lower()}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.pdf"
            )
        except Exception as e:
            st.warning(f"⚠️ Could not generate PDF report: {str(e)}")


def _get_completion_logprobs(prompt: str, api_key: str, model_name: str, base_url: Optional[str] = None, progress_message: Optional[str] = None) -> Tuple[List[float], Optional[str]]:
    """Get logprobs using Completion API (echo=True, max_tokens=0) to get prompt logprobs.
    
    This follows the reference implementation which uses Completion API with echo=True
    to get logprobs for the entire prompt, not just generated tokens.
    
    Important: The prompt is used directly as-is, without any special wrapping for Instruct/Chat models.
    This ensures that Base and Instruct models are treated identically - we compute the natural
    likelihood of the text sequence, not the model's response to an instruction about the text.
    
    Args:
        prompt: Input text to analyze (used directly, no special prompt wrapping).
        api_key: API key for the LLM provider.
        model_name: Name of the model to use.
        base_url: Optional base URL for the API (for custom endpoints).
        progress_message: Optional progress message to display.
    
    Returns:
        Tuple of (list of logprobs, error_message). If error_message is not None, the call failed.
    """
    if not OPENAI_AVAILABLE:
        return [], "OpenAI library not available"
    
    # Initialize progress tracking variables
    label_placeholder = None
    bar_placeholder = None
    progress_bar = None
    
    try:
        # Show progress message if provided
        if progress_message:
            from src.direct_recall.comparison import start_llm_progress, update_llm_progress, complete_llm_progress
            label_placeholder, bar_placeholder, progress_bar = start_llm_progress(progress_message)
            update_llm_progress(progress_bar, value=15)
        
        client_kwargs = {"api_key": api_key}
        if base_url:
            client_kwargs["base_url"] = base_url
        
        client = openai.OpenAI(**client_kwargs)
        
        # Use Completion API with echo=True and max_tokens=0 to get prompt logprobs
        # This matches the reference implementation: calculatePerplexity_gpt3
        prompt_clean = prompt.replace('\x00', '')  # Remove null bytes
        
        try:
            response = client.completions.create(
                model=model_name,
                prompt=prompt_clean,
                max_tokens=0,  # Don't generate new tokens
                temperature=1.0,
                logprobs=5,  # Top 5 logprobs
                echo=True,  # Echo the prompt back with logprobs
            )
            
            # Extract logprobs from response
            logprobs_content = response.choices[0].logprobs
            if not logprobs_content or not hasattr(logprobs_content, 'token_logprobs'):
                if progress_message and label_placeholder:
                    from src.direct_recall.comparison import complete_llm_progress
                    complete_llm_progress(
                        label_placeholder,
                        bar_placeholder,
                        progress_bar,
                        final_message="No logprobs in response",
                        success=False,
                    )
                return [], "No logprobs in response"
            
            # Extract token logprobs (filter out None values)
            all_logprobs = [lp for lp in logprobs_content.token_logprobs if lp is not None]
            
            # Complete progress if shown
            if progress_message and label_placeholder:
                from src.direct_recall.comparison import complete_llm_progress
                complete_llm_progress(
                    label_placeholder,
                    bar_placeholder,
                    progress_bar,
                    final_message=f"Completed ({progress_message})",
                    success=True,
                )
            
            return all_logprobs, None
            
        except Exception as e:
            # Completion API might not be available, return error
            if progress_message and label_placeholder:
                from src.direct_recall.comparison import complete_llm_progress
                complete_llm_progress(
                    label_placeholder,
                    bar_placeholder,
                    progress_bar,
                    final_message="Completion API error",
                    success=False,
                )
            return [], f"Completion API error: {str(e)}"
            
    except Exception as e:
        if progress_message and label_placeholder:
            from src.direct_recall.comparison import complete_llm_progress
            complete_llm_progress(
                label_placeholder,
                bar_placeholder,
                progress_bar,
                final_message="Error calling Completion API",
                success=False,
            )
        return [], f"Error calling Completion API: {str(e)}"


def run_min_k_prob_analysis(
    prompt: str,
    api_key: str,
    model_name: str,
    provider: str,
    k_percentage: int = 10,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_tokens: int = 500,
    agent_url: Optional[str] = None,
    agent_key: Optional[str] = None,
) -> Dict[str, Any]:
    """Run MIN-K% PROB analysis on model generation.
    
    This implements the MIN-K% PROB metric as described in unlearning detection papers.
    It calculates the negative mean of the lowest k% log probabilities.
    
    Important: Base Model and Instruct/Chat Model Treatment
    ======================================================
    Whether the model is a Base model or an Instruct/Chat model, the Min-K% Prob calculation
    method is identical:
    
    1. The input text X is treated as a token sequence by the model.
    2. The algorithm directly computes the conditional probability P(x_i | x_{<i}) for each token.
    3. No special prompt is used: For Instruct models, we do NOT wrap the text in a special
       prompt (e.g., "Please tell me if you know this text..."). Instead, we directly compute
       the model's natural likelihood of generating the text sequence.
    4. The calculation is based purely on the token-level conditional probabilities, regardless
       of whether the model was trained as a base language model or an instruction-following model.
    
    Following the reference implementation, this function:
    1. Uses Completion API with echo=True and max_tokens=0 to get prompt logprobs (if supported)
    2. Falls back to ChatCompletion API if Completion API is not available
    3. Calculates Min-k% Prob for multiple k values (5%, 10%, 20%, 30%, 40%, 50%, 60%)
    4. Calculates additional metrics (ppl, ppl/lowercase_ppl, ppl/zlib)
    
    Args:
        prompt: Input text to analyze (treated as a token sequence, no special prompt wrapping).
        api_key: API key for the LLM provider (used if agent_url is not provided).
        model_name: Name of the model to use (works the same for Base and Instruct models).
        provider: LLM provider ("OpenAI", "OpenRouter", etc.) - used if agent_url is not provided.
        k_percentage: Percentage of lowest probability tokens to analyze (e.g., 10 means bottom 10%).
        temperature: Sampling temperature (not used for logprob calculation, only for fallback generation).
        top_p: Top-p sampling parameter (not used for logprob calculation, only for fallback generation).
        max_tokens: Maximum tokens to generate (not used when using Completion API with echo=True).
        agent_url: Optional URL of the deployment agent (if provided, uses this instead of sidebar model).
        agent_key: Optional API key for the deployment agent.
    
    Returns:
        Dictionary containing analysis results with MIN-K% PROB metric and additional metrics.
    """
    # Determine base_url, provider, and model_name if agent_url is provided
    base_url = None
    effective_provider = provider
    effective_api_key = api_key
    effective_model_name = model_name
    custom_progress_message = None
    
    if agent_url and agent_url.strip():
        # Use deployment agent - construct base_url for OpenAI-compatible endpoint
        # The agent_url points to deploy.py server exposed via Cloudflare Tunnel
        # deploy.py provides OpenAI-compatible API at /v1 endpoint
        base_url = f"{agent_url.strip().rstrip('/')}/v1"
        # Use agent_key as api_key if provided
        if agent_key and agent_key.strip():
            effective_api_key = agent_key.strip()
        # Use "Local vLLM" provider for OpenAI-compatible API compatibility
        # But set a custom progress message to reflect it's a server deployment agent
        effective_provider = "Local vLLM"
        # Custom progress message to show it's using server deployment agent
        custom_progress_message = f"Calling Server Deployment Agent · {model_name}"
        # For server deployment, model_name is the model path on the server
        # Use the provided model_name (which should be the server-local path)
        effective_model_name = model_name
    
    # Try to use Completion API first (for getting prompt logprobs with echo=True)
    log_probs, error_msg = _get_completion_logprobs(
        prompt=prompt,
        api_key=effective_api_key,
        model_name=effective_model_name,
        base_url=base_url,
        progress_message=custom_progress_message,  # Use custom message for server deployment
    )
    
    # If Completion API failed, fall back to ChatCompletion API
    if error_msg or not log_probs:
        # Get completion with logprobs using ChatCompletion API
        result = get_llm_completion(
            prompt=prompt,
            api_key=effective_api_key,
            model_name=effective_model_name,
            provider=effective_provider,
            temperature=temperature,
            top_p=top_p,
            max_output_tokens=max_tokens,
            return_logprobs=True,
            base_url=base_url,
            progress_message=custom_progress_message,  # Use custom message for server deployment
        )
        
        if isinstance(result, tuple):
            generated_text, logprobs_data = result
        else:
            # Error case
            if isinstance(result, str) and result.startswith("Error"):
                raise ValueError(result)
            generated_text = result
            logprobs_data = None
        
        if not logprobs_data:
            raise ValueError("The API did not return logprobs. This feature requires logprobs support (OpenAI/OpenRouter).")
        
        # Extract log probabilities from ChatCompletion response
        log_probs = []
        token_details = []
        for token_info in logprobs_data:
            logprob = token_info.get('logprob', 0.0)
            # Filter out None values (some tokens may not have logprobs)
            if logprob is not None:
                log_probs.append(logprob)
                token_details.append({
                    'token': token_info.get('token', ''),
                    'logprob': logprob,
                    'linear_prob': math.exp(logprob) if logprob > -100 else 0.0,
                })
        
        if not log_probs:
            raise ValueError("No token log probabilities found in the response.")
        
        # For ChatCompletion, we only have generated text, not the full prompt
        generated_text = generated_text if 'generated_text' in locals() else ""
    else:
        # Successfully got logprobs from Completion API
        # The logprobs are for the entire prompt (echo=True)
        generated_text = prompt
        token_details = [
            {
                'token': '',  # We don't have token strings from Completion API
                'logprob': lp,
                'linear_prob': math.exp(lp) if lp > -100 else 0.0,
            }
            for lp in log_probs
        ]
    
    if not log_probs:
        raise ValueError("No token log probabilities found in the response.")
    
    # Calculate overall statistics
    overall_avg_logprob = sum(log_probs) / len(log_probs) if log_probs else 0.0
    overall_avg_prob = math.exp(overall_avg_logprob) if overall_avg_logprob > -100 else 0.0
    
    # Calculate perplexity (exp of mean negative log probability)
    perplexity = math.exp(-overall_avg_logprob) if overall_avg_logprob < 0 else float('inf')
    
    # Calculate perplexity for lowercase version (if using Completion API)
    ppl_lowercase = None
    if not error_msg:  # If Completion API worked, try lowercase
        log_probs_lower, _ = _get_completion_logprobs(
            prompt=prompt.lower(),
            api_key=effective_api_key,
            model_name=effective_model_name,
            base_url=base_url,
        )
        if log_probs_lower:
            avg_logprob_lower = sum(log_probs_lower) / len(log_probs_lower) if log_probs_lower else 0.0
            ppl_lower = math.exp(-avg_logprob_lower) if avg_logprob_lower < 0 else float('inf')
            # Ratio of log ppl of lower-case and normal-case
            if perplexity > 0 and ppl_lower > 0:
                ppl_lowercase = -(math.log(ppl_lower) / math.log(perplexity))
    
    # Calculate zlib compression entropy
    zlib_entropy = len(zlib.compress(bytes(prompt, 'utf-8')))
    ppl_zlib = None
    if perplexity > 0 and zlib_entropy > 0:
        ppl_zlib = math.log(perplexity) / zlib_entropy
    
    # Calculate Min-k% Prob for multiple k values (following reference implementation)
    min_k_probs = {}
    k_ratios = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]  # 5%, 10%, 20%, 30%, 40%, 50%, 60%
    
    sorted_log_probs = sorted(log_probs)
    for ratio in k_ratios:
        k_length = max(1, int(len(log_probs) * ratio))
        topk_prob = sorted_log_probs[:k_length]
        if topk_prob:
            min_k_probs[f"Min_{int(ratio*100)}%_Prob"] = -sum(topk_prob) / len(topk_prob)
        else:
            min_k_probs[f"Min_{int(ratio*100)}%_Prob"] = 0.0
    
    # Calculate the requested k_percentage Min-k% Prob (for backward compatibility)
    total_tokens = len(log_probs)
    ratio = k_percentage / 100.0
    k_tokens_count = max(1, int(len(log_probs) * ratio))
    min_k_log_probs = sorted_log_probs[:k_tokens_count]
    min_k_avg_logprob = sum(min_k_log_probs) / len(min_k_log_probs) if min_k_log_probs else 0.0
    min_k_prob = -min_k_avg_logprob  # Negative mean log probability
    
    # Get corresponding token details for the lowest k% tokens
    sorted_token_details = sorted(token_details, key=lambda x: x['logprob'])
    min_k_token_details = sorted_token_details[:k_tokens_count]
    
    return {
        'generated_text': generated_text,
        'min_k_prob': min_k_prob,  # MIN-K% PROB metric (negative mean log prob) for requested k%
        'min_k_avg_logprob': min_k_avg_logprob,  # Average log probability of lowest k%
        'overall_avg_logprob': overall_avg_logprob,  # Overall average log probability
        'overall_avg_prob': overall_avg_prob,  # Overall average linear probability
        'perplexity': perplexity,  # Perplexity (exp of -mean log prob)
        'ppl_lowercase': ppl_lowercase,  # Ratio of log ppl of lower-case and normal-case
        'ppl_zlib': ppl_zlib,  # Ratio of log ppl and zlib entropy
        'min_k_probs': min_k_probs,  # Min-k% Prob for multiple k values (5%, 10%, 20%, 30%, 40%, 50%, 60%)
        'total_tokens': total_tokens,
        'k_tokens_count': k_tokens_count,
        'k_percentage': k_percentage,
        'token_details': min_k_token_details,  # Details of lowest k% tokens
        'all_token_details': token_details,  # All token details
    }


def render_representational_analysis_page(api_key, model_choice, provider):
    """Render the representational analysis experience."""
    
    # Initialize session state for Unlearning Detection
    if 'unlearn_feature_id_index' not in st.session_state:
        st.session_state['unlearn_feature_id_index'] = 0
    if 'unlearn_reference_model' not in st.session_state:
        st.session_state['unlearn_reference_model'] = ""
    if 'unlearn_updated_model' not in st.session_state:
        st.session_state['unlearn_updated_model'] = ""
    if 'unlearn_query_text' not in st.session_state:
        st.session_state['unlearn_query_text'] = ""
    if 'unlearn_batch_size' not in st.session_state:
        st.session_state['unlearn_batch_size'] = 4
    if 'unlearn_num_batches' not in st.session_state:
        st.session_state['unlearn_num_batches'] = 10
    if 'unlearn_max_length' not in st.session_state:
        st.session_state['unlearn_max_length'] = 128
    if 'unlearn_last_result' not in st.session_state:
        st.session_state['unlearn_last_result'] = None
    if 'unlearn_last_request' not in st.session_state:
        st.session_state['unlearn_last_request'] = None
    if 'unlearn_deploy_agent_url' not in st.session_state:
        st.session_state['unlearn_deploy_agent_url'] = ""
    if 'unlearn_deploy_agent_key' not in st.session_state:
        st.session_state['unlearn_deploy_agent_key'] = ""
    
    # Page header with clear cache button
    header_col, button_col = st.columns([4, 1])
    with header_col:
        st.markdown('<h4 class="section-header">🧬 Representational Analysis</h4>', unsafe_allow_html=True)
        st.markdown(
            "Run Fisher Information, PCA shift/sim, and layer-wise CKA probes to quantify how unlearning reshapes the reference versus adapted model across every layer. "
            "(Xu et al., 2025)"
        )
        with st.expander("📚 Reference", expanded=False):
            st.markdown("""
            **Xu, X., Yue, X., Liu, Y., Ye, Q., Zheng, H., Hu, P., Du, M., & Hu, H. (2025).**  
            Unlearning Isn't Deletion: Investigating Reversibility of Machine Unlearning in LLMs.  
            *arXiv preprint arXiv:2505.16831*.  
            [Paper](https://arxiv.org/abs/2505.16831)
            """)
    with button_col:
        register_clear_cache_handler(REPRESENTATIONAL_CLEAR_CACHE_ID, _clear_representational_cache)
        if st.button(
            "🗑️ Clear Cache",
            key="clear_representational_cache",
            help="Reset cached representational analysis results and configuration.",
        ):
            _clear_representational_cache()

    dependencies_available = is_representational_analysis_available()
    if not dependencies_available:
        st.warning(
            "Representational analysis requires optional dependencies (PyTorch, Transformers, scikit-learn, matplotlib). Install the GPU toolkit extras before using this feature."
        )

    features = list_representational_features()
    if not features:
        st.info("No representational analysis features are currently available.")
        return

    feature_lookup = {feature.id: feature for feature in features}

    feature_options = [feature.id for feature in features]
    selected_feature_id = st.selectbox(
        "Select representational probe",
        options=feature_options,
        index=min(st.session_state['unlearn_feature_id_index'], len(feature_options) - 1),
        format_func=lambda feature_id: f"{feature_lookup[feature_id].name} — {feature_lookup[feature_id].description}",
        key="representational_feature_selection",
        help="Maps directly to the `feature` argument of `run_feature_analysis`.",
    )
    # Update index in session state
    if selected_feature_id in feature_options:
        st.session_state['unlearn_feature_id_index'] = feature_options.index(selected_feature_id)

    selected_feature = feature_lookup[selected_feature_id]

    # Server deployment agent configuration
    st.markdown("##### 🚀 Server Deployment Agent Configuration")
    st.caption("Configure the URL and API key of your server deployment agent (e.g., Cloudflare Tunnel URL)")
    col_url, col_key = st.columns([2, 1])
    with col_url:
        agent_url = st.text_input(
            "Deployment Agent URL",
            value=st.session_state.get('unlearn_deploy_agent_url', ''),
            placeholder="https://cool-server-link.trycloudflare.com",
            help="The URL of your server deployment agent (from Cloudflare Tunnel or similar)",
            type="password",
            key="unlearn_deploy_agent_url_input",
        )
    with col_key:
        agent_key = st.text_input(
            "Key",
            value=st.session_state.get('unlearn_deploy_agent_key', ''),
            placeholder="YOUR_API_KEY",
            help="API key set on your server (YOUR_API_KEY environment variable)",
            type="password",
            key="unlearn_deploy_agent_key_input",
        )
    if agent_url:
        st.session_state['unlearn_deploy_agent_url'] = agent_url.strip()
    else:
        st.session_state['unlearn_deploy_agent_url'] = ""
    if agent_key:
        st.session_state['unlearn_deploy_agent_key'] = agent_key.strip()
    else:
        st.session_state['unlearn_deploy_agent_key'] = ""
    
    col_ref, col_upd = st.columns(2)
    with col_ref:
        reference_model_path = st.text_input(
            "Reference model path",
            value=st.session_state['unlearn_reference_model'],
            placeholder="e.g. gpt2, Qwen/Qwen2.5-7B, or /path/to/local/model",
            help="Hugging Face model ID (e.g., 'gpt2') or absolute path to local model directory containing config.json",
            type="password",
            key="representational_reference_model",
        )
        # Update session state
        st.session_state['unlearn_reference_model'] = reference_model_path
    
    with col_upd:
        updated_model_path = st.text_input(
            "Unlearned model path",
            value=st.session_state['unlearn_updated_model'],
            placeholder="Path or HF repo ID for the model under audit",
            help="Hugging Face model ID (e.g., 'microsoft/DialoGPT-medium') or absolute path to local model directory",
            type="password",
            key="representational_updated_model",
        )
        # Update session state
        st.session_state['unlearn_updated_model'] = updated_model_path

    st.markdown("##### Evaluation prompts")
    query_text = st.text_area(
        "Evaluation prompts",
        value=st.session_state['unlearn_query_text'],
        height=180,
        placeholder="Enter one query per line that probes the model's behaviour post-unlearning.\\n\\nExample:\\nThe quick brown fox jumps over the lazy dog.\\nUnlearning LLMs is an active area of research.\\nWhat is the capital of France?",
        help="Each non-empty line is passed as an element of the `query` list. Enter multiple queries (one per line) to test different prompts.",
        key="representational_query_text",
    )
    # Update session state with current query text
    st.session_state['unlearn_query_text'] = query_text
    query_preview = [line.strip() for line in query_text.splitlines() if line.strip()]

    if query_preview:
        with st.expander(f"📝 **{len(query_preview)} query(ies) will be processed**", expanded=False):
            for i, query in enumerate(query_preview, 1):
                st.text(f"{i}. {query}")
    else:
        st.caption("📝 No queries entered yet. Add at least one query above.")

    st.markdown("##### Runtime parameters")
    st.caption("Device is set to `cuda` (GPU enabled).")
    device = "cuda"

    col_batch, col_batches, col_length = st.columns([1, 1, 1])
    with col_batch:
        batch_size = st.number_input(
            "Batch size",
            min_value=1,
            max_value=128,
            value=st.session_state['unlearn_batch_size'],
            step=1,
            help="Mini-batch size for analyses that stream batches (FIM, CKA).",
            key="representational_batch_size",
        )
        # Update session state
        st.session_state['unlearn_batch_size'] = batch_size
    with col_batches:
        num_batches = st.number_input(
            "Batches",
            min_value=1,
            max_value=200,
            value=st.session_state['unlearn_num_batches'],
            step=1,
            help="Number of dataloader batches to use when estimating statistics (FIM, CKA).",
            key="representational_num_batches",
        )
        # Update session state
        st.session_state['unlearn_num_batches'] = num_batches
    with col_length:
        max_length = st.number_input(
            "Max length",
            min_value=16,
            max_value=4096,
            value=st.session_state['unlearn_max_length'],
            step=16,
            help="Maximum sequence length for tokenization.",
            key="representational_max_length",
        )
        # Update session state
        st.session_state['unlearn_max_length'] = max_length

    st.caption("Preview of the backend call that will be executed with your settings:")
    query_list_preview = ", ".join(f'"{q}"' for q in query_preview) or '"<enter at least one query>"'
    call_preview = textwrap.dedent(
        f"""
        run_feature_analysis(
            feature="{selected_feature.id}",
            model_reference_path="{reference_model_path.strip() or '<reference_model>'}",
            model_path="{updated_model_path.strip() or '<updated_model>'}",
            query=[{query_list_preview}],
            device="{device}",
            batch_size={int(batch_size)},
            num_batches={int(num_batches)},
            max_length={int(max_length)},
        )
        """.strip()
    )
    st.code(call_preview, language="python")

    submit_run = render_run_button(
        "Representational Analysis",
        "unlearn_rep_submit_run",
        "🧬 Run: Representational Analysis",
        help="Submit the parameters above and execute the representational probe on the backend.",
    )

    # Load previous result from session state (if exists and button not clicked)
    rep_result = st.session_state.get('unlearn_last_result')
    analysis_request = st.session_state.get('unlearn_last_request')
    
    # Only run analysis when button is clicked
    if submit_run:
            set_active_clear_cache_id(REPRESENTATIONAL_CLEAR_CACHE_ID)
            queries = query_preview
            if not reference_model_path.strip():
                st.warning("⚠️ Provide the reference model path before running representational analysis.")
            elif not updated_model_path.strip():
                st.warning("⚠️ Provide the updated model path before running representational analysis.")
            elif not queries:
                st.warning("⚠️ Enter at least one non-empty query prompt.")
            else:
                # Auto-deploy models if Deployment Agent URL is configured
                agent_url = st.session_state.get('unlearn_deploy_agent_url', '').strip()
                agent_key = st.session_state.get('unlearn_deploy_agent_key', '').strip()
                if agent_url:
                    ref_path = reference_model_path.strip()
                    upd_path = updated_model_path.strip()
                    
                    # Prepare headers with API key if provided
                    deploy_headers = {}
                    if agent_key:
                        deploy_headers["X-API-Key"] = agent_key
                    
                    # Deploy reference model
                    if ref_path:
                        with st.spinner("Sending deployment request for reference model..."):
                            try:
                                response = requests.post(
                                    f"{agent_url}/deploy",
                                    json={"model_path": ref_path},
                                    headers=deploy_headers,
                                    timeout=10
                                )
                                if response.status_code == 200:
                                    res_json = response.json()
                                    if res_json.get("status") == "success":
                                        st.success(f"✅ Reference model deployment initiated: {res_json.get('message', '')}")
                                    else:
                                        st.warning(f"⚠️ Reference model deployment warning: {res_json.get('message', 'Unknown error')}")
                                elif response.status_code == 401:
                                    st.error(f"❌ Reference model deployment failed (401): Authentication failed. Please check your API key in the 'Key' field.")
                                elif response.status_code == 403:
                                    st.error(f"❌ Reference model deployment failed (403): Invalid API key. Please check your API key in the 'Key' field.")
                                elif response.status_code == 530:
                                    st.warning(
                                        f"⚠️ Reference model deployment failed (530): Cloudflare Tunnel cannot reach the server. "
                                        f"Please check:\n"
                                        f"1. Is `deploy_agent.py` running on the server?\n"
                                        f"2. Is Cloudflare Tunnel running and connected?\n"
                                        f"3. Is the Tunnel URL still valid? (Tunnel URLs may expire)\n"
                                        f"Continuing with analysis..."
                                    )
                                else:
                                    st.warning(f"⚠️ Reference model deployment failed with status code: {response.status_code}. Continuing with analysis...")
                            except requests.exceptions.Timeout:
                                st.warning("⏱️ Reference model deployment timeout. The server may be slow or unreachable. Continuing with analysis...")
                            except requests.exceptions.ConnectionError as e:
                                st.warning(f"🔌 Unable to connect to deployment agent for reference model: {str(e)}. Continuing with analysis...")
                            except Exception as e:
                                st.warning(f"⚠️ Reference model deployment error: {str(e)}. Continuing with analysis...")
                    
                    # Deploy updated model
                    if upd_path:
                        with st.spinner("Sending deployment request for updated model..."):
                            try:
                                response = requests.post(
                                    f"{agent_url}/deploy",
                                    json={"model_path": upd_path},
                                    headers=deploy_headers,
                                    timeout=10
                                )
                                if response.status_code == 200:
                                    res_json = response.json()
                                    if res_json.get("status") == "success":
                                        st.success(f"✅ Updated model deployment initiated: {res_json.get('message', '')}")
                                    else:
                                        st.warning(f"⚠️ Updated model deployment warning: {res_json.get('message', 'Unknown error')}")
                                elif response.status_code == 401:
                                    st.error(f"❌ Updated model deployment failed (401): Authentication failed. Please check your API key in the 'Key' field.")
                                elif response.status_code == 403:
                                    st.error(f"❌ Updated model deployment failed (403): Invalid API key. Please check your API key in the 'Key' field.")
                                elif response.status_code == 530:
                                    st.warning(
                                        f"⚠️ Updated model deployment failed (530): Cloudflare Tunnel cannot reach the server. "
                                        f"Please check:\n"
                                        f"1. Is `deploy_agent.py` running on the server?\n"
                                        f"2. Is Cloudflare Tunnel running and connected?\n"
                                        f"3. Is the Tunnel URL still valid? (Tunnel URLs may expire)\n"
                                        f"Continuing with analysis..."
                                    )
                                else:
                                    st.warning(f"⚠️ Updated model deployment failed with status code: {response.status_code}. Continuing with analysis...")
                            except requests.exceptions.Timeout:
                                st.warning("⏱️ Updated model deployment timeout. The server may be slow or unreachable. Continuing with analysis...")
                            except requests.exceptions.ConnectionError as e:
                                st.warning(f"🔌 Unable to connect to deployment agent for updated model: {str(e)}. Continuing with analysis...")
                            except Exception as e:
                                st.warning(f"⚠️ Updated model deployment error: {str(e)}. Continuing with analysis...")
                
                # Validate model paths
                ref_path = reference_model_path.strip()
                upd_path = updated_model_path.strip()
                
                ref_valid = False
                upd_valid = False
                
                # Check if it's a Hugging Face model ID (contains slash or is simple name)
                if '/' in ref_path or ref_path in ['gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl']:
                    ref_valid = True
                # Check if it's a local directory with config.json
                elif os.path.isdir(ref_path) and os.path.exists(os.path.join(ref_path, 'config.json')):
                    ref_valid = True
                else:
                    st.error(f"❌ Reference model path '{ref_path}' is not valid. Use a Hugging Face model ID (e.g., 'gpt2') or a local directory containing config.json")
                
                if '/' in upd_path or upd_path in ['gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl']:
                    upd_valid = True
                elif os.path.isdir(upd_path) and os.path.exists(os.path.join(upd_path, 'config.json')):
                    upd_valid = True
                else:
                    st.error(f"❌ Updated model path '{upd_path}' is not valid. Use a Hugging Face model ID (e.g., 'gpt2') or a local directory containing config.json")
                
                if ref_valid and upd_valid:
                    with detection_job("Representational Analysis"):
                        agent_url = st.session_state.get('unlearn_deploy_agent_url', '').strip()
                        agent_key = st.session_state.get('unlearn_deploy_agent_key', '').strip()

                        analysis_request = {
                            "feature": selected_feature.id,
                            "model_reference_path": ref_path,
                            "model_path": upd_path,
                            "query": queries,
                            "device": device,
                            "batch_size": int(batch_size),
                            "num_batches": int(num_batches),
                            "max_length": int(max_length),
                        }

                        if agent_url:
                            analysis_request["agent_url"] = agent_url
                            if agent_key:
                                analysis_request["agent_key"] = agent_key
                        with st.spinner("🔎 Computing representational differences... this may take several minutes for large models."):
                            try:
                                rep_result = run_representational_analysis(**analysis_request)
                                st.session_state['unlearn_last_result'] = rep_result
                                st.session_state['unlearn_last_request'] = analysis_request
                            except ValueError as exc:
                                st.error(f"❌ {exc}")
                                rep_result = None
                                st.session_state['unlearn_last_result'] = None
                                st.session_state['unlearn_last_request'] = None
                            except RuntimeError as exc:
                                err_text = str(exc)
                                st.error("❌ Representational analysis failed. Expand for full diagnostics below.")
                                sections = []
                                parts = err_text.split("--- Captured stdout ---")
                                if len(parts) == 2:
                                    before_stdout = parts[0]
                                    after_stdout = parts[1]
                                    parts2 = after_stdout.split("--- Captured stderr ---")
                                    if len(parts2) == 2:
                                        stdout_content = parts2[0]
                                        after_stderr = parts2[1]
                                        parts3 = after_stderr.split("--- Traceback ---")
                                        if len(parts3) == 2:
                                            stderr_content = parts3[0]
                                            tb_content = parts3[1]
                                            exception_part = before_stdout.strip()
                                            sections.append(("Exception", exception_part, None))
                                            if stdout_content.strip():
                                                sections.append(("Captured stdout", stdout_content.strip(), None))
                                            if stderr_content.strip():
                                                sections.append(("Captured stderr", stderr_content.strip(), None))
                                            if tb_content.strip():
                                                sections.append(("Traceback", tb_content.strip(), None))
                                        else:
                                            sections.append(("Full Diagnostics", err_text, None))
                                    else:
                                        sections.append(("Full Diagnostics", err_text, None))
                                else:
                                    sections.append(("Full Diagnostics", err_text, None))
                                render_collapsible_panel(
                                    title="Representational Analysis Logs and Traceback",
                                    sections=sections,
                                    expanded=False,
                                    max_height=600,
                                )
                                rep_result = None
                                st.session_state['unlearn_last_result'] = None
                                st.session_state['unlearn_last_request'] = None

    # Display results if available (either from new run or from session state)
    if rep_result:
                st.markdown("---")
                st.success(
                    f"Completed {rep_result.feature_name} analysis. Review the generated artifacts below."
                )

                if analysis_request:
                    st.markdown("##### Parameters sent to the backend")
                    st.json(analysis_request)

                if rep_result.warnings:
                    for warning in rep_result.warnings:
                        st.warning(warning)

                if rep_result.inline_artifacts:
                    st.markdown("##### Visualisations")

                    # Group visualizations in columns for better layout
                    num_artifacts = len(rep_result.inline_artifacts)
                    if num_artifacts <= 3:
                        # For few artifacts, show in a single row
                        cols = st.columns(num_artifacts)
                        for idx, artifact in enumerate(rep_result.inline_artifacts):
                            with cols[idx]:
                                caption = artifact.title or f"Visualisation {idx + 1}"
                                if artifact.mime_type.startswith("image/"):
                                    st.image(artifact.data, caption=caption, width='content')
                                else:
                                    st.download_button(
                                        label=f"⬇️ Download {caption}",
                                        data=artifact.data,
                                        file_name=f"representational_artifact_{idx + 1}",
                                        mime=artifact.mime_type,
                                        key=f"representational_inline_{idx + 1}",
                                    )
                                if artifact.description:
                                    st.caption(artifact.description)
                    else:
                        # For many artifacts, show in a grid
                        cols_per_row = 3
                        for i in range(0, num_artifacts, cols_per_row):
                            row_artifacts = rep_result.inline_artifacts[i:i + cols_per_row]
                            cols = st.columns(len(row_artifacts))
                            for j, artifact in enumerate(row_artifacts):
                                with cols[j]:
                                    caption = artifact.title or f"Visualisation {i + j + 1}"
                                    if artifact.mime_type.startswith("image/"):
                                        st.image(artifact.data, caption=caption, width='content')
                                    else:
                                        st.download_button(
                                            label=f"⬇️ Download {caption}",
                                            data=artifact.data,
                                            file_name=f"representational_artifact_{i + j + 1}",
                                            mime=artifact.mime_type,
                                            key=f"representational_inline_{i + j + 1}",
                                        )
                                    if artifact.description:
                                        st.caption(artifact.description)

                if rep_result.generated_artifacts:
                    st.markdown("##### Generated artifacts")
                    for artifact_path in rep_result.generated_artifacts:
                        st.markdown(f"- `{artifact_path}`")

                    _ResultPath = Path  # local alias to avoid polluting module namespace

                    output_dir_path = _ResultPath(rep_result.output_path)
                    if output_dir_path.is_dir() and len(rep_result.generated_artifacts) > 1:
                        zip_target = output_dir_path / f"{rep_result.feature_id}_artifacts.zip"
                        try:
                            with zipfile.ZipFile(zip_target, "w") as zipf:
                                for artifact in rep_result.generated_artifacts:
                                    artifact_path_obj = _ResultPath(artifact)
                                    if artifact_path_obj.exists():
                                        zipf.write(artifact_path_obj, arcname=artifact_path_obj.name)
                            with open(zip_target, "rb") as zip_bytes:
                                st.download_button(
                                    label="⬇️ Download all artifacts (ZIP)",
                                    data=zip_bytes.read(),
                                    file_name=zip_target.name,
                                    mime="application/zip",
                                    key="representational_zip_download",
                                )
                        except Exception as exc:  # pragma: no cover - file IO errors
                            st.warning(f"Unable to bundle artifacts for download: {exc}")

                    for artifact_path in rep_result.generated_artifacts[:5]:
                        artifact_obj = _ResultPath(artifact_path)
                        if artifact_obj.is_file() and artifact_obj.suffix.lower() == ".pdf":
                            try:
                                with open(artifact_obj, "rb") as pdf_bytes:
                                    st.download_button(
                                        label=f"⬇️ Download {artifact_obj.name}",
                                        data=pdf_bytes.read(),
                                        file_name=artifact_obj.name,
                                        mime="application/pdf",
                                        key=f"representational_pdf_{artifact_obj.name}",
                                    )
                            except Exception as exc:  # pragma: no cover - file IO errors
                                st.warning(f"Could not open {artifact_obj.name} for download: {exc}")
                    if len(rep_result.generated_artifacts) > 5:
                        st.caption(
                            "Additional artifacts are available in the output directory. Download them from the filesystem if needed."
                        )
                if not rep_result.generated_artifacts and not rep_result.inline_artifacts:
                    st.info("No artifacts were detected. Check the logs and ensure the selected feature produces outputs.")

