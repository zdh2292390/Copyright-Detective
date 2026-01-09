"""Remote execution module for representational analysis on server-side models."""

import base64
import io
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

from .representational_toolkit.types import FeatureAnalysisResult, VisualizationItem


def read_analysis_code_files(feature: str) -> Dict[str, str]:
    """Read analysis code files and return as dictionary."""
    # Correct path: remote_execution.py is in src/unlearning_detection/
    # representational_toolkit is in src/unlearning_detection/representational_toolkit/
    toolkit_dir = Path(__file__).parent / "representational_toolkit"
    
    files_to_read = [
        "types.py",
        "analysis.py",
    ]
    
    feature_file_map = {
        "fim": "fisher_analysis.py",
        "pca_shift": "pca_shift_analysis.py",
        "pca_sim": "pca_sim_analysis.py",
        "cka": "cka_analysis.py",
    }
    
    feature_file = feature_file_map.get(feature.lower(), "")
    if feature_file:
        files_to_read.append(feature_file)
    
    code_files = {}
    missing_files = []
    
    # Debug: print toolkit directory
    print(f"🔍 Looking for representational_toolkit at: {toolkit_dir}")
    print(f"🔍 Toolkit directory exists: {toolkit_dir.exists()}")
    
    for filename in files_to_read:
        if filename:
            file_path = toolkit_dir / filename
            if file_path.exists():
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        code_files[filename] = content
                        print(f"✅ Read {filename} ({len(content)} chars)")
                except Exception as e:
                    missing_files.append(f"{filename} (read error: {str(e)})")
                    print(f"❌ Failed to read {filename}: {str(e)}")
            else:
                missing_files.append(f"{filename} (not found at {file_path})")
                print(f"❌ File not found: {file_path}")
    
    if missing_files:
        # List what files actually exist in the directory
        existing_files = []
        if toolkit_dir.exists():
            existing_files = [f.name for f in toolkit_dir.iterdir() if f.is_file()]
        
        raise RuntimeError(
            f"Failed to read analysis code files. Missing or unreadable files:\n"
            f"  - " + "\n  - ".join(missing_files) + f"\n\n"
            f"Expected toolkit directory: {toolkit_dir}\n"
            f"Directory exists: {toolkit_dir.exists()}\n"
            f"Files in directory: {existing_files}\n"
            f"Please ensure representational_toolkit directory exists at the correct location."
        )
    
    if not code_files:
        raise RuntimeError(f"No code files were read. Toolkit directory: {toolkit_dir}")
    
    print(f"✅ Successfully read {len(code_files)} code file(s)")
    return code_files


def execute_analysis_remotely(
    agent_url: str,
    feature: str,
    model_reference_path: str,
    model_path: str,
    query: List[str],
    device: str = "cuda",
    batch_size: int = 4,
    num_batches: int = 10,
    max_length: int = 128,
    timeout: int = 3600,  # 60 minutes timeout for analysis (FIM analysis can take a very long time)
) -> FeatureAnalysisResult:
    """
    Execute representational analysis on the remote server.
    
    Args:
        agent_url: URL of the deployment agent (e.g., https://xxx.trycloudflare.com)
        feature: Analysis feature type (fim, pca_shift, pca_sim, cka)
        model_reference_path: Path to reference model
        model_path: Path to updated model
        query: List of query strings
        device: Device to use (cuda/cpu)
        batch_size: Batch size for analysis
        num_batches: Number of batches
        max_length: Max sequence length
        timeout: Request timeout in seconds
        
    Returns:
        FeatureAnalysisResult with visualizations and warnings
    """
    
    # Optimize parameters for FIM analysis to avoid Cloudflare timeout
    if feature.lower() == "fim":
        # FIM is computationally intensive and can easily timeout
        # Use very conservative defaults for remote execution
        if batch_size > 1:
            print(f"⚠️ FIM analysis: Reducing batch_size from {batch_size} to 1 to avoid timeout")
            batch_size = 1
        if num_batches > 3:
            print(f"⚠️ FIM analysis: Reducing num_batches from {num_batches} to 3 to avoid timeout")
            num_batches = 3
    
    # Read analysis code from frontend
    try:
        code_files = read_analysis_code_files(feature)
    except Exception as e:
        raise RuntimeError(
            f"Failed to read analysis code files from frontend: {str(e)}\n"
            f"This is a frontend issue. Please check if representational_toolkit files exist."
        ) from e
    
    # Send analysis request to server with code
    # The server should have a /run_analysis endpoint that accepts these parameters
    try:
        response = requests.post(
            f"{agent_url}/run_analysis",
            json={
                "feature": feature,
                "model_reference_path": model_reference_path,
                "model_path": model_path,
                "query": query,
                "device": device,
                "batch_size": batch_size,
                "num_batches": num_batches,
                "max_length": max_length,
                "analysis_code": json.dumps(code_files),  # Send code files as JSON string
            },
            timeout=timeout,
        )
        
        if response.status_code != 200:
            # Handle specific error codes
            if response.status_code == 524:
                # Cloudflare Tunnel timeout
                raise RuntimeError(
                    f"Cloudflare Tunnel timeout (524). The analysis took longer than Cloudflare's timeout limit (usually 100 seconds).\n\n"
                    f"Solutions:\n"
                    f"1. Configure Cloudflare Tunnel with increased timeout (see CLOUDFLARE_TUNNEL_SETUP.md)\n"
                    f"   - Edit ~/.cloudflared/config.yml and set timeout: 1800s\n"
                    f"   - Restart Cloudflare Tunnel\n"
                    f"2. Use ngrok instead of Cloudflare Tunnel (see CLOUDFLARE_TUNNEL_SETUP.md)\n"
                    f"3. Reduce analysis parameters (num_batches, batch_size) to speed up processing\n"
                    f"4. Run analysis locally instead of remotely"
                )
            elif response.status_code == 504:
                # Gateway timeout
                raise RuntimeError(
                    f"Gateway timeout (504). The server took too long to respond.\n\n"
                    f"This may indicate:\n"
                    f"- The analysis is taking longer than expected\n"
                    f"- Server resources are overloaded\n"
                    f"- Network connectivity issues\n\n"
                    f"Try reducing num_batches or batch_size, or check server status."
                )
            else:
                raise RuntimeError(
                    f"Remote execution failed with status code {response.status_code}: {response.text[:500]}"
                )
        
        result = response.json()
        
        if result.get("status") == "error":
            raise RuntimeError(
                f"Remote execution error: {result.get('msg', 'Unknown error')}\n"
                f"Log: {result.get('log', '')}"
            )
        
        # Parse the result
        # The server should return a dict with 'visualizations' and 'warnings'
        data = result.get("data", {})
        
        visualizations = []
        warnings = []
        
        # Parse visualizations (assuming they're returned as base64-encoded images)
        if isinstance(data, dict):
            viz_list = data.get("visualizations", [])
            for viz in viz_list:
                if isinstance(viz, dict):
                    title = viz.get("title", "Visualization")
                    image_data = viz.get("data", "")
                    description = viz.get("description")
                    
                    # Decode base64 image if needed
                    if isinstance(image_data, str):
                        try:
                            image_bytes = base64.b64decode(image_data)
                        except Exception:
                            image_bytes = image_data.encode() if isinstance(image_data, str) else b""
                    else:
                        image_bytes = image_data
                    
                    visualizations.append(
                        VisualizationItem(
                            title=title,
                            data=image_bytes,
                            mime_type=viz.get("mime_type", "image/png"),
                            description=description,
                        )
                    )
            
            warnings = data.get("warnings", [])
        
        return FeatureAnalysisResult(
            visualizations=visualizations,
            warnings=warnings,
        )
        
    except requests.exceptions.Timeout:
        raise RuntimeError(
            f"Remote execution timed out after {timeout} seconds. "
            "The analysis may be taking longer than expected.\n\n"
            "This could be due to:\n"
            "- Cloudflare Tunnel timeout (default 100s) - configure with increased timeout\n"
            "- Long-running analysis (FIM can take 30+ minutes)\n"
            "- Network connectivity issues\n\n"
            "Solutions:\n"
            "- See CLOUDFLARE_TUNNEL_SETUP.md for timeout configuration\n"
            "- Configure Cloudflare Tunnel: edit ~/.cloudflared/config.yml and set timeout: 1800s\n"
            "- Reduce num_batches or batch_size to speed up processing\n"
            "- Check server logs for analysis progress"
        )
    except requests.exceptions.ConnectionError as e:
        raise RuntimeError(
            f"Unable to connect to deployment agent at {agent_url}. "
            f"Please check if the server is running and the URL is correct. Error: {str(e)}"
        )
    except Exception as e:
        raise RuntimeError(f"Remote execution failed: {str(e)}")

