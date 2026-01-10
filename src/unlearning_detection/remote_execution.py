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
    poll_interval: int = 2,  # Poll every 2 seconds
    max_poll_time: int = 3600,  # Maximum time to poll (1 hour)
    api_key: Optional[str] = None,  # API key for authentication
) -> FeatureAnalysisResult:
    """
    Execute representational analysis on the remote server using async task pattern.
    
    The server now uses async task processing to avoid Cloudflare timeout:
    1. Submit analysis request -> get task_id (202 Accepted)
    2. Poll /task_status/{task_id} until completed
    3. Return results
    
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
        timeout: Request timeout in seconds (for individual requests)
        poll_interval: Seconds between status polls
        max_poll_time: Maximum time to poll for results (seconds)
        api_key: API key for authentication (X-API-Key header)
        
    Returns:
        FeatureAnalysisResult with visualizations and warnings
    """
    
    # Optimize parameters for FIM analysis
    if feature.lower() == "fim":
        if batch_size > 1:
            print(f"⚠️ FIM analysis: Reducing batch_size from {batch_size} to 1")
            batch_size = 1
        if num_batches > 3:
            print(f"⚠️ FIM analysis: Reducing num_batches from {num_batches} to 3")
            num_batches = 3
    
    # Read analysis code from frontend
    try:
        code_files = read_analysis_code_files(feature)
    except Exception as e:
        raise RuntimeError(
            f"Failed to read analysis code files from frontend: {str(e)}\n"
            f"This is a frontend issue. Please check if representational_toolkit files exist."
        ) from e
    
    # Step 1: Submit analysis request and get task_id
    # Use a shorter timeout for the initial request (should be fast)
    # Cloudflare has a 100s timeout, so we use 90s to get a clearer error
    submit_timeout = min(90, timeout)
    
    try:
        print(f"📤 Submitting analysis request to {agent_url}/run_analysis...")
        code_size = len(json.dumps(code_files))
        print(f"📦 Request size: {code_size} bytes of code")
        
        # Prepare headers with API key if provided
        headers = {}
        if api_key:
            headers["X-API-Key"] = api_key
        
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
                "analysis_code": json.dumps(code_files),
            },
            headers=headers,
            timeout=submit_timeout,
        )
        
        if response.status_code == 202:
            # Async task created
            result = response.json()
            task_id = result.get("task_id")
            if not task_id:
                raise RuntimeError("Server returned 202 but no task_id in response")
            
            print(f"✅ Task created: {task_id}")
            print(f"⏳ Polling for results (polling every {poll_interval}s, max {max_poll_time}s)...")
        elif response.status_code == 200:
            # Legacy synchronous response (for backward compatibility)
            result = response.json()
            if result.get("status") == "error":
                raise RuntimeError(
                    f"Remote execution error: {result.get('msg', 'Unknown error')}"
                )
            # Process synchronous result
            data = result.get("data", {})
            return _parse_analysis_result(data)
        elif response.status_code == 401:
            raise RuntimeError(
                f"Authentication failed (401): Missing or invalid API key. "
                f"Please check that you've entered the correct API key in the 'Key' field. "
                f"The key should match the YOUR_API_KEY environment variable set on your server."
            )
        elif response.status_code == 403:
            raise RuntimeError(
                f"Access denied (403): Invalid API key. "
                f"Please check that you've entered the correct API key in the 'Key' field. "
                f"The key should match the YOUR_API_KEY environment variable set on your server."
            )
        elif response.status_code == 524:
            # Cloudflare timeout - this shouldn't happen with async, but handle it
            raise RuntimeError(
                f"Cloudflare timeout (524) while submitting request. This usually means:\n"
                f"1. The request body is too large (code files are too big: {code_size} bytes)\n"
                f"2. Network connectivity issues\n"
                f"3. Server is overloaded\n\n"
                f"Solutions:\n"
                f"- Check network connection\n"
                f"- Try again (may be temporary)\n"
                f"- Contact server administrator if problem persists\n"
                f"- Consider reducing the size of analysis code files"
            )
        else:
            raise RuntimeError(
                f"Failed to submit analysis request. Status code: {response.status_code}, "
                f"Response: {response.text[:500]}"
            )
    except requests.exceptions.ConnectionError as e:
        raise RuntimeError(
            f"Unable to connect to deployment agent at {agent_url}. "
            f"Please check if the server is running and the URL is correct. Error: {str(e)}"
        )
    except requests.exceptions.Timeout:
        raise RuntimeError(
            f"Request to server timed out after {submit_timeout} seconds while submitting task. "
            f"This may indicate:\n"
            f"- Request body is too large (code files are very large)\n"
            f"- Network connectivity issues\n"
            f"- Server is not responding quickly enough\n\n"
            f"Note: This timeout is for submitting the task, not for the analysis itself. "
            f"The analysis will run asynchronously once submitted."
        )
    except RuntimeError:
        # Re-raise RuntimeError as-is (already has proper error message)
        raise
    
    # Step 2: Poll for task status
    import time
    start_time = time.time()
    last_status = None
    
    while True:
        elapsed = time.time() - start_time
        if elapsed > max_poll_time:
            raise RuntimeError(
                f"Polling timeout: Analysis did not complete within {max_poll_time} seconds. "
                f"Task {task_id} may still be running on the server."
            )
        
        try:
            # Prepare headers with API key if provided
            headers = {}
            if api_key:
                headers["X-API-Key"] = api_key
            
            status_response = requests.get(
                f"{agent_url}/task_status/{task_id}",
                headers=headers,
                timeout=timeout,
            )
            
            if status_response.status_code == 401:
                raise RuntimeError(
                    f"Authentication failed (401): Missing or invalid API key. "
                    f"Please check that you've entered the correct API key in the 'Key' field."
                )
            elif status_response.status_code == 403:
                raise RuntimeError(
                    f"Access denied (403): Invalid API key. "
                    f"Please check that you've entered the correct API key in the 'Key' field."
                )
            elif status_response.status_code == 404:
                raise RuntimeError(f"Task {task_id} not found on server")
            
            status_data = status_response.json()
            current_status = status_data.get("status")
            
            # Print status updates (only when status changes)
            if current_status != last_status:
                if current_status == "pending":
                    print(f"⏳ Task {task_id}: Queued (waiting to start)...")
                elif current_status == "running":
                    started_at = status_data.get("started_at", "unknown")
                    print(f"🔄 Task {task_id}: Running (started at {started_at})...")
                last_status = current_status
            
            if current_status == "completed":
                print(f"✅ Task {task_id}: Completed!")
                result_data = status_data.get("result", {})
                if result_data.get("status") == "error":
                    raise RuntimeError(
                        f"Analysis failed: {result_data.get('msg', 'Unknown error')}"
                    )
                return _parse_analysis_result(result_data.get("data", {}))
            
            elif current_status == "failed":
                error_info = status_data.get("error", {})
                error_msg = error_info.get("msg", "Unknown error")
                error_traceback = error_info.get("traceback", "")
                raise RuntimeError(
                    f"Analysis task failed: {error_msg}\n"
                    f"{error_traceback[:1000] if error_traceback else ''}"
                )
            
            # Task is still pending or running, wait and poll again
            time.sleep(poll_interval)
            
        except requests.exceptions.Timeout:
            print(f"⚠️ Status poll timed out, retrying...")
            time.sleep(poll_interval)
            continue
        except requests.exceptions.ConnectionError as e:
            raise RuntimeError(
                f"Lost connection to server while polling task {task_id}. "
                f"Error: {str(e)}"
            )


def _parse_analysis_result(data: Dict[str, Any]) -> FeatureAnalysisResult:
    """Parse analysis result data into FeatureAnalysisResult."""
    visualizations = []
    warnings = []
    
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

