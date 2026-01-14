"""
Unlearning Detection Module

This module provides the UI for representational analysis to detect unlearning
in language models by comparing reference and updated models.
"""

import os
import textwrap
import zipfile
from pathlib import Path

import requests
import streamlit as st

from src.components import render_collapsible_panel
from src.unlearning_detection import (
    is_representational_analysis_available,
    list_representational_features,
    run_representational_analysis,
)


def render_unlearning_detection_page(api_key, model_choice, provider):
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
    if 'unlearn_deploy_agent_url' not in st.session_state:
        st.session_state['unlearn_deploy_agent_url'] = ""
    if 'unlearn_deploy_agent_key' not in st.session_state:
        st.session_state['unlearn_deploy_agent_key'] = ""
    
    st.markdown('<h4 class="section-header">🧬 Unlearning Detection</h4>', unsafe_allow_html=True)
    st.markdown(
        "Run Fisher Information, PCA shift/sim, and layer-wise CKA probes to quantify how unlearning reshapes the reference versus adapted model across every layer."
    )

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

    st.markdown("##### Model checkpoints")
    st.info("💡 **Model Path Format**: Use Hugging Face model IDs (e.g., 'gpt2', 'microsoft/DialoGPT-medium') or absolute paths to local directories containing `config.json` and model files. Do not use Hugging Face cache paths directly.")
    
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
            key="representational_reference_model",
        )
    
    with col_upd:
        updated_model_path = st.text_input(
            "Unlearned model path",
            value=st.session_state['unlearn_updated_model'],
            placeholder="Path or HF repo ID for the model under audit",
            help="Hugging Face model ID (e.g., 'microsoft/DialoGPT-medium') or absolute path to local model directory",
            key="representational_updated_model",
        )

    st.markdown("##### Evaluation prompts")
    st.text_area(
        "Evaluation prompts",
        value=st.session_state['unlearn_query_text'],
        height=180,
        placeholder="Enter one query per line that probes the model's behaviour post-unlearning.\\n\\nExample:\\nThe quick brown fox jumps over the lazy dog.\\nUnlearning LLMs is an active area of research.\\nWhat is the capital of France?",
        help="Each non-empty line is passed as an element of the `query` list. Enter multiple queries (one per line) to test different prompts.",
        key="representational_query_text",
    )
    query_preview = [line.strip() for line in st.session_state.get('representational_query_text', '').splitlines() if line.strip()]

    if query_preview:
        st.caption(f"📝 **{len(query_preview)} query(ies) will be processed:**")
        for i, query in enumerate(query_preview, 1):
            st.caption(f"{i}. {query}")
    else:
        st.caption("📝 No queries entered yet. Add at least one query above.")

    st.markdown("##### Runtime parameters")
    st.caption("Device is set to `cuda` (GPU enabled).")
    device = "cuda"

    col_batch, col_batches, col_length = st.columns([1, 1, 1])
    with col_batch:
        st.number_input(
            "Batch size",
            min_value=1,
            max_value=128,
            value=st.session_state['unlearn_batch_size'],
            step=1,
            help="Mini-batch size for analyses that stream batches (FIM, CKA).",
            key="representational_batch_size",
        )
    with col_batches:
        st.number_input(
            "Batches",
            min_value=1,
            max_value=200,
            value=st.session_state['unlearn_num_batches'],
            step=1,
            help="Number of dataloader batches to use when estimating statistics (FIM, CKA).",
            key="representational_num_batches",
        )
    with col_length:
        st.number_input(
            "Max length",
            min_value=16,
            max_value=4096,
            value=st.session_state['unlearn_max_length'],
            step=16,
            help="Maximum sequence length for tokenization.",
            key="representational_max_length",
        )

    st.caption("Preview of the backend call that will be executed with your settings:")
    query_list_preview = ", ".join(f'"{q}"' for q in query_preview) or '"<enter at least one query>"'
    reference_model_path = st.session_state.get('representational_reference_model', '')
    updated_model_path = st.session_state.get('representational_updated_model', '')
    batch_size = st.session_state.get('representational_batch_size', 4)
    num_batches = st.session_state.get('representational_num_batches', 10)
    max_length = st.session_state.get('representational_max_length', 128)
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

    submit_run = st.button(
        "🧬 Run Representational Analysis",
        width='stretch',
        help="Submit the parameters above and execute the representational probe on the backend.",
    )

    rep_result = None
    analysis_request = None
    if submit_run:
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
                    # Get agent_url and agent_key from session state if configured
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
                    
                    # Add agent_url and agent_key if configured (for remote execution)
                    if agent_url:
                        analysis_request["agent_url"] = agent_url
                        if agent_key:
                            analysis_request["agent_key"] = agent_key
                    with st.spinner("🔎 Computing representational differences... this may take several minutes for large models."):
                        try:
                            rep_result = run_representational_analysis(**analysis_request)
                            st.session_state["representational_last_run_request"] = analysis_request
                        except ValueError as exc:
                            st.error(f"❌ {exc}")
                            rep_result = None
                        except RuntimeError as exc:
                            # The RuntimeError raised by run_representational_analysis includes
                            # a detailed diagnostic containing captured stdout/stderr and the traceback.
                            err_text = str(exc)
                            st.error("❌ Representational analysis failed. Expand for full diagnostics below.")
                            # Parse the diagnostic into sections for the custom component
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

