"""Remote execution module for representational analysis on server-side models."""

import base64
import io
import json
import math
import time
import xml.etree.ElementTree as ET
from urllib.parse import quote
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
from PIL import Image

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
    
    if not isinstance(agent_url, str) or not agent_url.strip():
        raise ValueError("A deployment agent URL is required.")
    agent_url = agent_url.strip().rstrip("/")
    for name, value in (("timeout", timeout), ("poll_interval", poll_interval), ("max_poll_time", max_poll_time)):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be a positive finite number.")

    # Preserve the existing FIM resource limits.
    if feature.lower() == "fim":
        batch_size = min(batch_size, 1)
        num_batches = min(num_batches, 3)
    code_files = read_analysis_code_files(feature)
    headers = {"X-API-Key": api_key} if api_key else {}
    submit_timeout = min(90, timeout)
    from src.resumable_analysis import checkpoint_call
    submission = {
        "agent_url": agent_url, "feature": feature,
        "model_reference_path": model_reference_path, "model_path": model_path,
        "query": query, "device": device, "batch_size": batch_size,
        "num_batches": num_batches, "max_length": max_length,
        "analysis_code": json.dumps(code_files, sort_keys=True),
    }

    def submit_once():
        try:
            response = requests.post(
                f"{agent_url}/run_analysis",
                json={key: value for key, value in submission.items() if key != "agent_url"},
                headers=headers, timeout=submit_timeout,
            )
        except requests.exceptions.RequestException as exc:
            raise RuntimeError("Unable to submit representational analysis. The server may have accepted the request; check the server before starting another task.") from exc
        try:
            if response.status_code not in (200, 202):
                raise RuntimeError(f"Analysis submission failed with HTTP {response.status_code}. Check deployment agent connectivity and credentials.")
            result = _response_object(response, "Analysis submission")
            if response.status_code == 200:
                _completed_result(result, api_key=api_key)
            else:
                task_id = result.get("task_id")
                if not isinstance(task_id, str) or not task_id.strip() or len(task_id) > 256:
                    raise RuntimeError("Server returned 202 without a valid task_id.")
            return {"http_status": response.status_code, "response": result}
        finally:
            response.close()

    submitted = checkpoint_call("representational.submit", submission, submit_once)
    if submitted["http_status"] == 200:
        return _completed_result(submitted["response"], api_key=api_key)
    task_id = submitted["response"]["task_id"]

    def wait_for_result():
        deadline = time.monotonic() + max_poll_time
        last_error = ""
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                detail = f" Last status error: {last_error}" if last_error else ""
                raise RuntimeError(f"Polling timeout after {max_poll_time} seconds. Task {task_id} may still be running on the server.{detail}")
            try:
                status_response = requests.get(
                    f"{agent_url}/task_status/{quote(task_id, safe='')}",
                    headers=headers, timeout=min(timeout, 30, remaining),
                )
            except (requests.exceptions.Timeout, requests.exceptions.ConnectionError):
                last_error = "The deployment agent could not be reached during the last status check."
            except requests.exceptions.RequestException as exc:
                raise RuntimeError(f"Unable to check analysis task {task_id} status.") from exc
            else:
                try:
                    code = status_response.status_code
                    if code in (408, 429, 500, 502, 503, 504, 524, 530):
                        last_error = f"HTTP {code} while checking task status."
                    elif code not in (200, 202):
                        raise RuntimeError(f"Task {task_id} status check failed with HTTP {code}. Check server availability and credentials.")
                    else:
                        status_data = _response_object(status_response, "Task status")
                        status = status_data.get("status")
                        if status == "completed":
                            _completed_result(status_data.get("result"), api_key=api_key)
                            return status_data["result"]
                        if status == "failed":
                            error = status_data.get("error")
                            message = error.get("msg", "Unknown error") if isinstance(error, dict) else str(error or "Unknown error")
                            if api_key:
                                message = str(message).replace(api_key, "[redacted]")
                            raise RuntimeError(f"Analysis task {task_id} failed: {str(message)[:1000]}")
                        if status not in ("pending", "running"):
                            raise RuntimeError(f"Task {task_id} returned an invalid or missing task status.")
                        last_error = ""
                finally:
                    status_response.close()
            remaining = deadline - time.monotonic()
            if remaining > 0:
                time.sleep(min(poll_interval, remaining))

    result = checkpoint_call(
        "representational.result", {"agent_url": agent_url, "task_id": task_id},
        wait_for_result,
    )
    return _completed_result(result, api_key=api_key)


def _response_object(response, context: str) -> Dict[str, Any]:
    try:
        data = response.json()
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"{context} returned invalid JSON.") from exc
    if not isinstance(data, dict):
        raise RuntimeError(f"{context} returned an invalid response object.")
    return data


def _completed_result(result, api_key=None) -> FeatureAnalysisResult:
    if not isinstance(result, dict):
        raise RuntimeError("Completed analysis did not include a valid result object.")
    if result.get("status") == "error":
        message = str(result.get('msg', 'Unknown error'))
        if api_key:
            message = message.replace(api_key, "[redacted]")
        raise RuntimeError(f"Remote execution error: {message[:1000]}")
    if "data" not in result:
        raise RuntimeError("Completed analysis did not include result data.")
    return _parse_analysis_result(result["data"])


def _parse_analysis_result(data: Dict[str, Any]) -> FeatureAnalysisResult:
    """Validate artifacts rather than presenting malformed bytes as visualizations."""
    if not isinstance(data, dict):
        raise RuntimeError("Analysis result data must be an object.")
    viz_list = data.get("visualizations", [])
    warnings = data.get("warnings", [])
    if not isinstance(viz_list, list) or not isinstance(warnings, list) or any(not isinstance(item, str) for item in warnings):
        raise RuntimeError("Analysis result contains invalid visualizations or warnings.")
    visualizations = []
    warnings = list(warnings)
    for index, viz in enumerate(viz_list, start=1):
        try:
            if not isinstance(viz, dict):
                raise ValueError("artifact is not an object")
            title = viz.get("title", "Visualization")
            mime_type = viz.get("mime_type", "image/png")
            description = viz.get("description")
            if not isinstance(title, str) or not isinstance(mime_type, str) or not mime_type or (description is not None and not isinstance(description, str)):
                raise ValueError("invalid artifact metadata")
            encoded = viz.get("data")
            if isinstance(encoded, str):
                image_bytes = base64.b64decode(encoded, validate=True)
            elif isinstance(encoded, (bytes, bytearray)):
                image_bytes = bytes(encoded)
            else:
                raise ValueError("invalid artifact data")
            if not image_bytes:
                raise ValueError("empty artifact data")
            normalized_mime = mime_type.split(';', 1)[0].strip().lower()
            if normalized_mime.startswith("image/"):
                try:
                    if normalized_mime == "image/svg+xml":
                        root = ET.fromstring(image_bytes)
                        if root.tag not in ('svg', '{http://www.w3.org/2000/svg}svg'):
                            raise ValueError("artifact does not contain an SVG document")
                    else:
                        with Image.open(io.BytesIO(image_bytes)) as decoded:
                            decoded.verify()
                except Exception as exc:
                    raise ValueError("image data could not be decoded") from exc
        except (ValueError, TypeError) as exc:
            warnings.append(f"Visualization {index} was not usable: {exc}.")
            continue
        visualizations.append(VisualizationItem(title=title, data=image_bytes, mime_type=mime_type, description=description))
    if viz_list and not visualizations:
        raise RuntimeError("Analysis returned visualizations, but none contained usable artifact data.")
    if not visualizations:
        warnings.append("Analysis returned no visualization artifacts; completion could not be verified from outputs.")
    return FeatureAnalysisResult(visualizations=visualizations, warnings=warnings)
