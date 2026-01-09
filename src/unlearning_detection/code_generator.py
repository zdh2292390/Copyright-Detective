"""Code generator for serializing representational_toolkit analysis code."""

import inspect
from pathlib import Path
from typing import Dict, List


def get_analysis_code_file_path(feature: str) -> Path:
    """Get the path to the analysis code file for a given feature."""
    toolkit_dir = Path(__file__).parent / "representational_toolkit"
    
    feature_file_map = {
        "fim": "fisher_analysis.py",
        "pca_shift": "pca_shift_analysis.py",
        "pca_sim": "pca_sim_analysis.py",
        "cka": "cka_analysis.py",
    }
    
    filename = feature_file_map.get(feature.lower())
    if not filename:
        raise ValueError(f"Unknown feature: {feature}")
    
    return toolkit_dir / filename


def read_file_content(file_path: Path) -> str:
    """Read file content, handling imports and dependencies."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()


def generate_executable_code(
    feature: str,
    query: List[str],
    device: str,
    batch_size: int,
    num_batches: int,
    max_length: int,
) -> str:
    """
    Generate executable code that can be sent to the server.
    The code will use model_ref and model_upd variables (server's models).
    """
    
    # Read the analysis file
    analysis_file = get_analysis_code_file_path(feature)
    analysis_code = read_file_content(analysis_file)
    
    # Read types file for FeatureAnalysisResult
    toolkit_dir = Path(__file__).parent / "representational_toolkit"
    types_file = toolkit_dir / "types.py"
    types_code = read_file_content(types_file)
    
    # Read analysis.py for run_feature_analysis wrapper
    analysis_wrapper_file = toolkit_dir / "analysis.py"
    analysis_wrapper_code = read_file_content(analysis_wrapper_file)
    
    # Generate the executable code
    # The code should:
    # 1. Import necessary modules
    # 2. Define types and helper functions
    # 3. Execute the analysis using model_ref and model_upd
    # 4. Set result variable with visualizations and warnings
    
    executable_code = f"""
# Generated analysis code for {feature}
import base64
import io
import contextlib
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

# Types definition
{types_code}

# Analysis code
{analysis_code}

# Analysis wrapper
{analysis_wrapper_code}

# Execute analysis
# Use server's models: model_ref and model_upd
try:
    # Call the appropriate analysis function based on feature
    if "{feature}" == "fim":
        analysis_result = run_fim_analysis(
            model_reference_path="",  # Not used, we use model_ref directly
            model_path="",  # Not used, we use model_upd directly
            query={query!r},
            device="{device}",
            batch_size={batch_size},
            num_batches={num_batches},
            max_length={max_length},
        )
    elif "{feature}" == "pca_shift":
        analysis_result = run_pca_shift(
            model_reference_path="",
            model_path="",
            query={query!r},
            device="{device}",
            max_length={max_length},
        )
    elif "{feature}" == "pca_sim":
        analysis_result = run_pca_similarity(
            model_reference_path="",
            model_path="",
            query={query!r},
            device="{device}",
            max_length={max_length},
        )
    elif "{feature}" == "cka":
        analysis_result = run_cka_analysis(
            model_reference_path="",
            model_path="",
            query={query!r},
            device="{device}",
            batch_size={batch_size},
            num_batches={num_batches},
            max_length={max_length},
        )
    else:
        raise ValueError(f"Unknown feature: {{'{feature}'}}")
    
    # Convert result to dict format
    visualizations = []
    for viz in analysis_result.visualizations:
        # Encode image data as base64
        if isinstance(viz.data, bytes):
            image_b64 = base64.b64encode(viz.data).decode('utf-8')
        else:
            image_b64 = ""
        
        visualizations.append({{
            "title": viz.title,
            "data": image_b64,
            "mime_type": getattr(viz, 'mime_type', 'image/png'),
            "description": getattr(viz, 'description', None),
        }})
    
    result = {{
        "visualizations": visualizations,
        "warnings": getattr(analysis_result, 'warnings', []),
    }}
    
except Exception as e:
    import traceback
    result = {{
        "error": str(e),
        "traceback": traceback.format_exc(),
    }}
"""
    
    return executable_code

