# deploy.py (Server-side)
import uvicorn
from fastapi import FastAPI, HTTPException, status, Depends, Security, Request
from starlette.requests import Request
from fastapi.responses import JSONResponse
from fastapi.security import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from pydantic import BaseModel
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
# Import specific tokenizer classes for models that need them
try:
    from transformers import GPTNeoXTokenizer, GPTNeoXTokenizerFast
except ImportError:
    GPTNeoXTokenizer = None
    GPTNeoXTokenizerFast = None
import io
import contextlib
import base64
from pathlib import Path
from typing import List, Dict, Any, Optional
import uuid
import threading
import time
from datetime import datetime
from enum import Enum
import os
import secrets

app = FastAPI(
    title="Model Deployment API",
    description="Secure API for model deployment and analysis",
    version="1.0.0"
)

# Security Middleware Configuration
# CORS Configuration - adjust allowed origins for your use case
ALLOWED_ORIGINS = os.environ.get("ALLOWED_ORIGINS", "*").split(",")
if ALLOWED_ORIGINS == ["*"]:
    # In production, specify exact origins instead of "*"
    print("⚠️  WARNING: CORS is set to allow all origins (*). Set ALLOWED_ORIGINS for production!")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
    max_age=3600,
)

# Trusted Host Middleware - only allow requests from trusted hosts
# Set TRUSTED_HOSTS environment variable (comma-separated) to restrict hosts
# Example: export TRUSTED_HOSTS=localhost,127.0.0.1,yourdomain.com
TRUSTED_HOSTS = os.environ.get("TRUSTED_HOSTS", None)
if TRUSTED_HOSTS:
    trusted_hosts = [h.strip() for h in TRUSTED_HOSTS.split(",")]
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=trusted_hosts
    )
    print(f"✅ Trusted hosts configured: {trusted_hosts}")
else:
    print("⚠️  WARNING: No TRUSTED_HOSTS set. All hosts are allowed. Set for production!")

# Request size limit (in bytes) - default 50MB
MAX_REQUEST_SIZE = int(os.environ.get("MAX_REQUEST_SIZE", 50 * 1024 * 1024))  # 50MB default

# Security Configuration
API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)

# Get API key from environment variable or generate a default one
# In production, set this via environment variable: export YOUR_API_KEY="your-secret-key"
# Supports both forms: export YOUR_API_KEY=key or export YOUR_API_KEY="key"
api_key_raw = os.environ.get("YOUR_API_KEY", None)
API_KEY = api_key_raw.strip('"\'') if api_key_raw else None  # Remove quotes if present

# If no API key is set, generate a random one and print it (for development)
if API_KEY is None:
    API_KEY = secrets.token_urlsafe(32)
    print(f"⚠️  WARNING: No YOUR_API_KEY environment variable set!")
    print(f"⚠️  Generated temporary API key: {API_KEY}")
    print(f"⚠️  Set YOUR_API_KEY environment variable for production use!")
    print(f"⚠️  Example: export YOUR_API_KEY=\"your-secret-key-here\"")

# List of allowed API keys (support multiple keys)
ALLOWED_API_KEYS = set([API_KEY])
if os.environ.get("API_KEYS"):
    # Support comma-separated list of API keys
    # Remove quotes from each key if present
    additional_keys = [
        k.strip().strip('"\'') 
        for k in os.environ.get("API_KEYS", "").split(",") 
        if k.strip()
    ]
    ALLOWED_API_KEYS.update(additional_keys)

def verify_api_key(api_key: str = Security(API_KEY_HEADER)) -> str:
    """
    Verify API key from request header.
    
    Args:
        api_key: API key from X-API-Key header
        
    Returns:
        The verified API key
        
    Raises:
        HTTPException: If API key is missing or invalid
    """
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing API key. Please provide X-API-Key header.",
            headers={"WWW-Authenticate": "ApiKey"},
        )
    
    if api_key not in ALLOWED_API_KEYS:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid API key. Access denied.",
            headers={"WWW-Authenticate": "ApiKey"},
        )
    
    return api_key

def verify_api_key_openai(request: Request) -> str:
    """
    Verify API key for OpenAI-compatible endpoints.
    Supports both X-API-Key header and Authorization: Bearer <api_key> header.
    
    Args:
        request: FastAPI request object
        
    Returns:
        The verified API key
        
    Raises:
        HTTPException: If API key is missing or invalid
    """
    # Try X-API-Key header first (for custom endpoints)
    api_key = request.headers.get("X-API-Key")
    
    # If not found, try Authorization: Bearer <api_key> (for OpenAI-compatible endpoints)
    if not api_key:
        auth_header = request.headers.get("Authorization", "")
        if auth_header.startswith("Bearer "):
            api_key = auth_header[7:]  # Remove "Bearer " prefix
        elif auth_header.startswith("bearer "):
            api_key = auth_header[7:]  # Case-insensitive
    
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing API key. Please provide X-API-Key header or Authorization: Bearer <api_key> header.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    if api_key not in ALLOWED_API_KEYS:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid API key. Access denied.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    return api_key

# Global variables to cache models, avoiding reloading on every request
GLOBAL_REFERENCE_MODEL = None
GLOBAL_UPDATED_MODEL = None
GLOBAL_TOKENIZER = None
CURRENT_REFERENCE_PATH = None
CURRENT_UPDATED_PATH = None

# For OpenAI-compatible API (MIN-K% PROB analysis)
GLOBAL_SINGLE_MODEL = None
GLOBAL_SINGLE_TOKENIZER = None
CURRENT_SINGLE_MODEL_PATH = None

# Task storage for async analysis
class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"

# In-memory task storage (in production, consider using Redis or a database)
TASKS: Dict[str, Dict[str, Any]] = {}
TASKS_LOCK = threading.Lock()

class CodeRequest(BaseModel):
    model_path: str
    code: str  # Receives Python code string from frontend

class DeployRequest(BaseModel):
    """Model deployment request (compatible with frontend)"""
    model_path: str

class AnalysisRequest(BaseModel):
    """Representational analysis request model"""
    feature: str  # "fim", "pca_shift", "pca_sim", "cka"
    model_reference_path: str
    model_path: str
    query: List[str]
    device: str = "cuda"
    batch_size: int = 4
    num_batches: int = 10
    max_length: int = 128
    analysis_code: Optional[str] = None  # Optional: JSON string of code files from frontend

# OpenAI-compatible API request models
class CompletionRequest(BaseModel):
    """OpenAI Completion API request"""
    model: str
    prompt: str
    max_tokens: Optional[int] = None
    temperature: Optional[float] = 1.0
    top_p: Optional[float] = None
    logprobs: Optional[int] = None
    echo: Optional[bool] = False
    stop: Optional[List[str]] = None

class ChatCompletionMessage(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    """OpenAI ChatCompletion API request"""
    model: str
    messages: List[ChatCompletionMessage]
    max_tokens: Optional[int] = None
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.9
    logprobs: Optional[bool] = False
    top_logprobs: Optional[int] = None
    stop: Optional[List[str]] = None

def execute_analysis_code(
    code_files: Dict[str, str],
    feature: str,
    query: List[str],
    device: str,
    batch_size: int,
    num_batches: int,
    max_length: int,
):
    """
    Execute analysis code sent from frontend.
    Load code files, modify them to use server's models, and execute.
    """
    global GLOBAL_REFERENCE_MODEL, GLOBAL_UPDATED_MODEL, GLOBAL_TOKENIZER
    global CURRENT_REFERENCE_PATH, CURRENT_UPDATED_PATH
    
    import os
    import sys
    
    # Prepare execution context with server models
    # Add built-in variables that Python modules expect
    import builtins
    exec_globals = {
        "__name__": "__main__",
        "__package__": None,
        "__file__": None,
        "__builtins__": builtins,
        "model_ref": GLOBAL_REFERENCE_MODEL,
        "model_upd": GLOBAL_UPDATED_MODEL,
        "model_reference": GLOBAL_REFERENCE_MODEL,
        "model_updated": GLOBAL_UPDATED_MODEL,
        "tokenizer": GLOBAL_TOKENIZER,
        "torch": torch,
        "base64": base64,
        "io": io,
        "contextlib": contextlib,
        "Path": Path,
    }
    
    # Add common imports
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
    from torch.utils.data import DataLoader, Dataset
    import sklearn
    from sklearn.decomposition import PCA
    import pandas as pd
    import gc
    
    exec_globals.update({
        "np": np,
        "plt": plt,
        "matplotlib": matplotlib,
        "mpl": matplotlib,
        "AutoModelForCausalLM": AutoModelForCausalLM,
        "AutoTokenizer": AutoTokenizer,
        "AutoConfig": AutoConfig,
        "DataLoader": DataLoader,
        "Dataset": Dataset,
        "sklearn": sklearn,
        "PCA": PCA,
        "pd": pd,
        "gc": gc,
    })
    
    # Monkey patch transformers to use server models
    import transformers
    _original_model_from_pretrained = transformers.AutoModelForCausalLM.from_pretrained
    _original_tokenizer_from_pretrained = transformers.AutoTokenizer.from_pretrained
    _original_config_from_pretrained = transformers.AutoConfig.from_pretrained
    
    # Track model loading context to determine which model to return
    _model_loading_context = {"current_path": None, "call_count": 0, "reference_called": False}
    
    def _patched_model_from_pretrained(path, **kwargs):
        _model_loading_context["current_path"] = path
        _model_loading_context["call_count"] += 1
        
        # Check if path matches our model paths
        if path == CURRENT_REFERENCE_PATH or (path == "" and not _model_loading_context["reference_called"]):
            _model_loading_context["reference_called"] = True
            return GLOBAL_REFERENCE_MODEL
        elif path == CURRENT_UPDATED_PATH or (path == "" and _model_loading_context["reference_called"]):
            return GLOBAL_UPDATED_MODEL
        else:
            return _original_model_from_pretrained(path, **kwargs)
    
    def _patched_tokenizer_from_pretrained(path, **kwargs):
        if path == "" or path is None or path == CURRENT_REFERENCE_PATH:
            return GLOBAL_TOKENIZER
        return _original_tokenizer_from_pretrained(path, **kwargs)
    
    def _patched_config_from_pretrained(path, **kwargs):
        if path == "" or path is None or path == CURRENT_REFERENCE_PATH:
            return GLOBAL_REFERENCE_MODEL.config
        elif path == CURRENT_UPDATED_PATH:
            return GLOBAL_UPDATED_MODEL.config
        return _original_config_from_pretrained(path, **kwargs)
    
    transformers.AutoModelForCausalLM.from_pretrained = _patched_model_from_pretrained
    transformers.AutoTokenizer.from_pretrained = _patched_tokenizer_from_pretrained
    transformers.AutoConfig.from_pretrained = _patched_config_from_pretrained
    
    def preprocess_code(code: str, filename: str, current_feature: str = None) -> str:
        """Preprocess code to fix relative imports and other issues."""
        # Replace relative imports with direct imports
        # from .types import ... -> from types import ...
        # from .xxx import ... -> from xxx import ...
        lines = code.split('\n')
        processed_lines = []
        
        # For analysis.py, we need to handle imports differently
        # since it imports from all feature files, but we only execute one
        if filename == "analysis.py" and current_feature:
            # For analysis.py, replace imports with direct assignments from exec_globals
            # This avoids Python's import system which won't find our fake modules
            for line in lines:
                stripped = line.strip()
                if stripped.startswith('from .types import') or stripped.startswith('from types import'):
                    # Remove types import - classes are in exec_globals
                    pass  # Skip
                elif stripped.startswith('from .') and 'import' in stripped:
                    # Replace: from .pca_shift_analysis import run_pca_shift
                    # With: run_pca_shift = pca_shift_analysis.run_pca_shift
                    # The modules are in exec_globals, so we can access them directly
                    import_part = stripped.split('import')[0].replace('from .', '').strip()
                    func_part = stripped.split('import')[1].strip()
                    # Create direct assignment
                    if func_part:
                        processed_lines.append(f"# {line}  # Replaced with direct assignment")
                        # Extract function name(s) - handle multiple imports
                        funcs = [f.strip() for f in func_part.split(',')]
                        for func in funcs:
                            processed_lines.append(f"{func} = {import_part}.{func}")
                    else:
                        processed_lines.append(line)
                else:
                    processed_lines.append(line)
        else:
            # For other files, handle imports
            for line in lines:
                stripped = line.strip()
                if stripped.startswith('from .types import') or stripped.startswith('from types import'):
                    # Remove 'from .types import X, Y' or 'from types import X, Y'
                    # because 'types' conflicts with Python's standard library.
                    # The classes (FeatureAnalysisResult, VisualizationItem) will be
                    # directly available in exec_globals after types.py is executed.
                    # No import needed - just skip this line.
                    pass  # Skip the import line entirely
                elif stripped.startswith('from .'):
                    # Replace other relative imports with direct imports
                    processed_line = line.replace('from .', 'from ')
                    processed_lines.append(processed_line)
                else:
                    processed_lines.append(line)
        
        return '\n'.join(processed_lines)
    
    try:
        # Import and execute analysis files in correct order
        # 1. Types first
        if "types.py" in code_files:
            types_code = preprocess_code(code_files["types.py"], "types.py")
            exec(compile(types_code, "types.py", "exec"), exec_globals)
            # After executing types.py, FeatureAnalysisResult and VisualizationItem
            # are now directly available in exec_globals.
            # We don't create a 'types' module to avoid conflict with stdlib.
            # The preprocessing step has already removed 'from types import' statements.
        
        # 2. Feature-specific analysis file
        feature_file_map = {
            "fim": "fisher_analysis.py",
            "pca_shift": "pca_shift_analysis.py",
            "pca_sim": "pca_sim_analysis.py",
            "cka": "cka_analysis.py",
        }
        feature_file = feature_file_map.get(feature)
        if not feature_file:
            raise RuntimeError(f"Unknown feature: {feature}. Expected one of: {list(feature_file_map.keys())}")
        
        if feature_file not in code_files:
            available_files = list(code_files.keys())
            raise RuntimeError(
                f"Feature file '{feature_file}' not found in code_files.\n"
                f"Feature: {feature}\n"
                f"Available files: {available_files}\n"
                f"Expected files: types.py, analysis.py, {feature_file}\n"
                f"Please ensure the frontend correctly reads and sends all required files."
            )
        
        # Preprocess and execute feature file
        feature_code = preprocess_code(code_files[feature_file], feature_file)
        exec(compile(feature_code, feature_file, "exec"), exec_globals)
        
        # Create fake module for the feature file so analysis.py can import from it
        feature_module_name = feature_file.replace('.py', '')
        class FakeFeatureModule:
            pass
        feature_module = FakeFeatureModule()
        # Extract the main function from exec_globals (e.g., run_pca_shift, run_fim_analysis, etc.)
        # The function name depends on the feature file
        function_name_map = {
            "fisher_analysis.py": "run_fim_analysis",
            "pca_shift_analysis.py": "run_pca_shift",
            "pca_sim_analysis.py": "run_pca_similarity",
            "cka_analysis.py": "run_cka_analysis",
        }
        func_name = function_name_map.get(feature_file)
        if func_name and func_name in exec_globals:
            setattr(feature_module, func_name, exec_globals[func_name])
            print(f"✅ Created module '{feature_module_name}' with function '{func_name}'")
        else:
            print(f"⚠️ Warning: Function '{func_name}' not found in exec_globals for module '{feature_module_name}'")
        exec_globals[feature_module_name] = feature_module
        
        # Create stub modules for other feature files that analysis.py might import
        # but we haven't executed (only if they don't already exist)
        all_feature_modules = {
            "fisher_analysis": "run_fim_analysis",
            "pca_shift_analysis": "run_pca_shift",
            "pca_sim_analysis": "run_pca_similarity",
            "cka_analysis": "run_cka_analysis",
        }
        for module_name, func_name in all_feature_modules.items():
            if module_name not in exec_globals:
                # Create a stub module with a dummy function
                class StubModule:
                    pass
                stub_module = StubModule()
                # Create a stub function that raises an error if called
                # (This should never be called since analysis.py only calls the current feature's function)
                def stub_func(*args, **kwargs):
                    raise RuntimeError(
                        f"Function {func_name} from {module_name} was not loaded. "
                        f"Only the current feature's analysis file is executed."
                    )
                setattr(stub_module, func_name, stub_func)
                exec_globals[module_name] = stub_module
                print(f"✅ Created stub module '{module_name}' with stub function '{func_name}'")
            else:
                # Module already exists (current feature), make sure it has the function
                existing_module = exec_globals[module_name]
                if not hasattr(existing_module, func_name) and func_name in exec_globals:
                    setattr(existing_module, func_name, exec_globals[func_name])
                    print(f"✅ Added function '{func_name}' to existing module '{module_name}'")
        
        # 3. Analysis wrapper (must be last as it imports from feature files)
        if "analysis.py" in code_files:
            analysis_code = preprocess_code(code_files["analysis.py"], "analysis.py", feature)
            exec(compile(analysis_code, "analysis.py", "exec"), exec_globals)
        
        # Call run_feature_analysis
        run_feature_analysis = exec_globals.get("run_feature_analysis")
        if not run_feature_analysis:
            raise RuntimeError("run_feature_analysis function not found in code")
        
        # Reset context for model loading tracking
        _model_loading_context["call_count"] = 0
        _model_loading_context["reference_called"] = False
        
        # Use actual model paths so monkey patch can match them
        # Create a temporary output path (not used in remote execution, but required by function signature)
        import tempfile
        temp_output_path = tempfile.mkdtemp(prefix="analysis_output_")
        
        print(f"🚀 Calling run_feature_analysis for feature={feature}...")
        result = run_feature_analysis(
            feature=feature,
            model_reference_path=CURRENT_REFERENCE_PATH or "",  # Use actual path for matching
            model_path=CURRENT_UPDATED_PATH or "",  # Use actual path for matching
            query=query,
            output_path=temp_output_path,  # Temporary path (not actually used for remote execution)
            device=device,
            batch_size=batch_size,
            num_batches=num_batches,
            max_length=max_length,
        )
        
        print(f"📊 run_feature_analysis returned: type={type(result)}")
        if result is None:
            raise RuntimeError("run_feature_analysis returned None")
        if not hasattr(result, 'visualizations'):
            raise RuntimeError(f"Result missing 'visualizations' attribute. Has: {dir(result)}")
        
        viz_count = len(result.visualizations) if hasattr(result, 'visualizations') else 0
        print(f"📊 Result has {viz_count} visualizations")
        
        # Convert result to response format
        visualizations = []
        warnings = getattr(result, 'warnings', [])
        
        # Check if result has visualizations
        if not hasattr(result, 'visualizations'):
            raise RuntimeError("FeatureAnalysisResult missing 'visualizations' attribute")
        
        print(f"📊 Processing {len(result.visualizations)} visualizations...")
        
        for idx, viz in enumerate(result.visualizations):
            try:
                image_data = getattr(viz, 'data', b'')
                if isinstance(image_data, bytes):
                    image_b64 = base64.b64encode(image_data).decode('utf-8')
                    print(f"  ✓ Visualization {idx+1}: {len(image_data)} bytes -> {len(image_b64)} base64 chars")
                else:
                    print(f"  ⚠️ Visualization {idx+1}: image_data is not bytes (type: {type(image_data)})")
                    image_b64 = ""
                
                visualizations.append({
                    "title": getattr(viz, 'title', f'Visualization {idx+1}'),
                    "data": image_b64,
                    "mime_type": getattr(viz, 'mime_type', 'image/png'),
                    "description": getattr(viz, 'description', None),
                })
            except Exception as e:
                print(f"  ❌ Error processing visualization {idx+1}: {str(e)}")
                import traceback
                print(f"  Traceback: {traceback.format_exc()}")
                raise RuntimeError(f"Failed to process visualization {idx+1}: {str(e)}")
        
        print(f"✅ Analysis code executed: {len(visualizations)} visualizations, {len(warnings)} warnings")
        
        return {
            "status": "success",
            "data": {
                "visualizations": visualizations,
                "warnings": warnings,
            }
        }
    except Exception as e:
        import traceback
        error_traceback = traceback.format_exc()
        print(f"❌ Analysis code execution failed: {str(e)}")
        print(f"Traceback:\n{error_traceback}")
        return {
            "status": "error",
            "msg": f"Analysis code execution error: {str(e)}",
            "traceback": error_traceback,
        }
    finally:
        # Restore original functions
        transformers.AutoModelForCausalLM.from_pretrained = _original_model_from_pretrained
        transformers.AutoTokenizer.from_pretrained = _original_tokenizer_from_pretrained
        transformers.AutoConfig.from_pretrained = _original_config_from_pretrained



def _fix_config_json_if_needed(model_path: str):
    """Fix config.json if it's missing model_type field"""
    import json
    import shutil
    from pathlib import Path
    
    config_path = os.path.join(model_path, "config.json")
    
    if not os.path.exists(config_path):
        raise RuntimeError(f"config.json not found at {config_path}")
    
    # Read config
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    # If model_type is missing, try to infer from path or config
    if "model_type" not in config:
        model_type = None
        path_lower = model_path.lower()
        
        # Try to infer from path (more specific patterns first)
        if "llama-2" in path_lower or "llama2" in path_lower:
            model_type = "llama"
        elif "llama" in path_lower:
            model_type = "llama"
        elif "gpt" in path_lower:
            model_type = "gpt2"  # Default to gpt2
        elif "bert" in path_lower:
            model_type = "bert"
        elif "roberta" in path_lower:
            model_type = "roberta"
        elif "t5" in path_lower:
            model_type = "t5"
        elif "mistral" in path_lower:
            model_type = "mistral"
        elif "mixtral" in path_lower:
            model_type = "mixtral"
        elif "qwen" in path_lower:
            model_type = "qwen2"
        elif "gemma" in path_lower:
            model_type = "gemma"
        
        # If still not found, check config for hints
        if not model_type:
            if "architectures" in config and len(config["architectures"]) > 0:
                arch = config["architectures"][0].lower()
                if "llama" in arch:
                    model_type = "llama"
                elif "gpt" in arch:
                    model_type = "gpt2"
                elif "bert" in arch:
                    model_type = "bert"
                elif "mistral" in arch:
                    model_type = "mistral"
                elif "mixtral" in arch:
                    model_type = "mixtral"
                elif "qwen" in arch:
                    model_type = "qwen2"
        
        if model_type:
            # Create backup first
            backup_path = config_path + ".backup"
            try:
                shutil.copy2(config_path, backup_path)
                print(f"📋 Created backup: {backup_path}")
            except Exception as e:
                print(f"⚠️ Could not create backup: {str(e)}")
            
            # Update config
            config["model_type"] = model_type
            
            # Save updated config with proper encoding and atomic write
            try:
                # Write to temp file first, then rename (atomic operation)
                temp_path = config_path + ".tmp"
                with open(temp_path, 'w', encoding='utf-8') as f:
                    json.dump(config, f, indent=2, ensure_ascii=False)
                
                # Atomic rename
                os.replace(temp_path, config_path)
                print(f"✅ Added model_type='{model_type}' to config.json based on path/config analysis")
                
                # Verify the write
                with open(config_path, 'r', encoding='utf-8') as f:
                    verify_config = json.load(f)
                    if verify_config.get("model_type") == model_type:
                        print(f"✅ Verified: model_type='{model_type}' is now in config.json")
                        return True
                    else:
                        print(f"⚠️ Warning: model_type was not saved correctly")
                        return False
            except Exception as e:
                print(f"❌ Error writing config.json: {str(e)}")
                # Try to restore backup if it exists
                if os.path.exists(backup_path):
                    try:
                        shutil.copy2(backup_path, config_path)
                        print(f"✅ Restored backup")
                    except:
                        pass
                raise RuntimeError(f"Failed to write config.json: {str(e)}")
        else:
            print(f"⚠️ Could not infer model_type from path: {model_path}")
            print(f"   Config keys: {list(config.keys())}")
            if "architectures" in config:
                print(f"   Architectures: {config['architectures']}")
            return False
    
    return False

def load_single_model_if_needed(model_path: str):
    """Load a single model for OpenAI-compatible API (MIN-K% PROB analysis)"""
    global GLOBAL_SINGLE_MODEL, GLOBAL_SINGLE_TOKENIZER, CURRENT_SINGLE_MODEL_PATH
    
    # Check if this is a davinci model (OpenAI API model, no local loading needed)
    if "davinci" in model_path.lower():
        # For davinci models, we don't load tokenizer/model locally
        # They are accessed via OpenAI API
        return
    
    # Load tokenizer
    if GLOBAL_SINGLE_TOKENIZER is None or CURRENT_SINGLE_MODEL_PATH != model_path:
        print(f"🔄 Loading tokenizer from {model_path}...")
        
        # Detect model type from path or config
        model_name_lower = model_path.lower()
        is_pythia = "pythia" in model_name_lower
        
        try:
            # First, try to read config to check tokenizer type
            config = None
            tokenizer_class_name = None
            try:
                config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
                tokenizer_class_name = getattr(config, 'tokenizer_class', None)
                print(f"   Detected tokenizer class: {tokenizer_class_name}")
            except Exception as e:
                print(f"   Could not read config: {e}, proceeding with default method")
            
            # For Pythia models or if tokenizer_class is GPTNeoXTokenizer
            if is_pythia or (tokenizer_class_name and "GPTNeoX" in str(tokenizer_class_name)):
                # Try GPTNeoXTokenizer if available
                if GPTNeoXTokenizer is not None:
                    try:
                        print("   Trying GPTNeoXTokenizer...")
                        GLOBAL_SINGLE_TOKENIZER = GPTNeoXTokenizer.from_pretrained(
                            model_path,
                            trust_remote_code=True
                        )
                    except Exception as e1:
                        print(f"   GPTNeoXTokenizer failed: {e1}")
                        # Fall through to try other methods
                        pass
                
                # If GPTNeoXTokenizer failed or not available, try loading with local_files_only
                # and manually construct tokenizer from files
                if GLOBAL_SINGLE_TOKENIZER is None:
                    try:
                        print("   Trying AutoTokenizer with local_files_only=True...")
                        # Try loading with local_files_only to bypass class detection
                        GLOBAL_SINGLE_TOKENIZER = AutoTokenizer.from_pretrained(
                            model_path,
                            trust_remote_code=True,
                            use_fast=False,
                            local_files_only=True
                        )
                    except Exception as e2:
                        print(f"   local_files_only failed: {e2}")
                        # Try without local_files_only but with use_fast=False
                        try:
                            print("   Trying AutoTokenizer with use_fast=False...")
                            GLOBAL_SINGLE_TOKENIZER = AutoTokenizer.from_pretrained(
                                model_path,
                                trust_remote_code=True,
                                use_fast=False
                            )
                        except Exception as e3:
                            print(f"   AutoTokenizer failed: {e3}")
                            # Last resort: try to load tokenizer files directly
                            try:
                                print("   Trying to load tokenizer from tokenizer.json directly...")
                                from transformers import PreTrainedTokenizer
                                tokenizer_path = Path(model_path)
                                if (tokenizer_path / "tokenizer.json").exists():
                                    # Try loading with PreTrainedTokenizer base class
                                    GLOBAL_SINGLE_TOKENIZER = AutoTokenizer.from_pretrained(
                                        str(tokenizer_path),
                                        trust_remote_code=True,
                                        use_fast=True
                                    )
                                else:
                                    raise RuntimeError("tokenizer.json not found")
                            except Exception as e4:
                                raise RuntimeError(
                                    f"All tokenizer loading methods failed.\n"
                                    f"GPTNeoXTokenizer not available. Last error: {e4}\n"
                                    f"Try installing: pip install transformers[torch] or ensure GPTNeoXTokenizer is available."
                                )
            else:
                # For non-Pythia models, use standard AutoTokenizer
                try:
                    # First try with trust_remote_code and use_fast=False
                    GLOBAL_SINGLE_TOKENIZER = AutoTokenizer.from_pretrained(
                        model_path,
                        trust_remote_code=True,
                        use_fast=False
                    )
                except Exception as e1:
                    print(f"   AutoTokenizer with use_fast=False failed: {e1}")
                    # Try with use_fast=True
                    try:
                        GLOBAL_SINGLE_TOKENIZER = AutoTokenizer.from_pretrained(
                            model_path,
                            trust_remote_code=True,
                            use_fast=True
                        )
                    except Exception as e2:
                        raise RuntimeError(f"Failed to load tokenizer: {e2}")
            
            # Ensure tokenizer has pad token
            if GLOBAL_SINGLE_TOKENIZER.pad_token is None:
                if GLOBAL_SINGLE_TOKENIZER.eos_token is not None:
                    GLOBAL_SINGLE_TOKENIZER.pad_token = GLOBAL_SINGLE_TOKENIZER.eos_token
                elif GLOBAL_SINGLE_TOKENIZER.bos_token is not None:
                    GLOBAL_SINGLE_TOKENIZER.pad_token = GLOBAL_SINGLE_TOKENIZER.bos_token
                else:
                    GLOBAL_SINGLE_TOKENIZER.add_special_tokens({"pad_token": "[PAD]"})
            print("✅ Tokenizer loaded")
        except Exception as e:
            raise RuntimeError(f"Failed to load tokenizer: {str(e)}")
    
    # Determine device
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    
    # Load model
    if GLOBAL_SINGLE_MODEL is None or CURRENT_SINGLE_MODEL_PATH != model_path:
        print(f"🔄 Loading model from {model_path}...")
        try:
            GLOBAL_SINGLE_MODEL = AutoModelForCausalLM.from_pretrained(
                model_path,
                trust_remote_code=True,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                low_cpu_mem_usage=True
            )
            
            GLOBAL_SINGLE_MODEL = GLOBAL_SINGLE_MODEL.to(device)
            GLOBAL_SINGLE_MODEL.eval()
            CURRENT_SINGLE_MODEL_PATH = model_path
            print(f"✅ Model loaded on {device}")
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {str(e)}")

def load_models_if_needed(reference_path: str, updated_path: str):
    """Load two models on demand (reference and updated)"""
    global GLOBAL_REFERENCE_MODEL, GLOBAL_UPDATED_MODEL, GLOBAL_TOKENIZER
    global CURRENT_REFERENCE_PATH, CURRENT_UPDATED_PATH
    
    # Check if reference model is davinci (OpenAI API model)
    if "davinci" in reference_path.lower():
        # For davinci models, we don't load tokenizer/model locally
        return
    
    # Load tokenizer (usually both models use the same tokenizer, loaded from reference model)
    if GLOBAL_TOKENIZER is None or CURRENT_REFERENCE_PATH != reference_path:
        print(f"🔄 Loading tokenizer from {reference_path}...")
        
        # Detect model type from path or config
        model_name_lower = reference_path.lower()
        is_pythia = "pythia" in model_name_lower
        
        try:
            # First, try to read config to check tokenizer type
            config = None
            tokenizer_class_name = None
            try:
                config = AutoConfig.from_pretrained(reference_path, trust_remote_code=True)
                tokenizer_class_name = getattr(config, 'tokenizer_class', None)
                print(f"   Detected tokenizer class: {tokenizer_class_name}")
            except Exception as e:
                print(f"   Could not read config: {e}, proceeding with default method")
            
            # For Pythia models or if tokenizer_class is GPTNeoXTokenizer
            if is_pythia or (tokenizer_class_name and "GPTNeoX" in str(tokenizer_class_name)):
                # Try GPTNeoXTokenizer if available
                if GPTNeoXTokenizer is not None:
                    try:
                        print("   Trying GPTNeoXTokenizer...")
                        GLOBAL_TOKENIZER = GPTNeoXTokenizer.from_pretrained(
                            reference_path,
                            trust_remote_code=True
                        )
                    except Exception as e1:
                        print(f"   GPTNeoXTokenizer failed: {e1}")
                        # Fall through to try other methods
                        pass
                
                # If GPTNeoXTokenizer failed or not available, try loading with local_files_only
                # and manually construct tokenizer from files
                if GLOBAL_TOKENIZER is None:
                    try:
                        print("   Trying AutoTokenizer with local_files_only=True...")
                        # Try loading with local_files_only to bypass class detection
                        GLOBAL_TOKENIZER = AutoTokenizer.from_pretrained(
                            reference_path,
                            trust_remote_code=True,
                            use_fast=False,
                            local_files_only=True
                        )
                    except Exception as e2:
                        print(f"   local_files_only failed: {e2}")
                        # Try without local_files_only but with use_fast=False
                        try:
                            print("   Trying AutoTokenizer with use_fast=False...")
                            GLOBAL_TOKENIZER = AutoTokenizer.from_pretrained(
                                reference_path,
                                trust_remote_code=True,
                                use_fast=False
                            )
                        except Exception as e3:
                            print(f"   AutoTokenizer failed: {e3}")
                            # Last resort: try to load tokenizer files directly
                            try:
                                print("   Trying to load tokenizer from tokenizer.json directly...")
                                from transformers import PreTrainedTokenizer
                                tokenizer_path = Path(reference_path)
                                if (tokenizer_path / "tokenizer.json").exists():
                                    # Try loading with PreTrainedTokenizer base class
                                    GLOBAL_TOKENIZER = AutoTokenizer.from_pretrained(
                                        str(tokenizer_path),
                                        trust_remote_code=True,
                                        use_fast=True
                                    )
                                else:
                                    raise RuntimeError("tokenizer.json not found")
                            except Exception as e4:
                                raise RuntimeError(
                                    f"All tokenizer loading methods failed.\n"
                                    f"GPTNeoXTokenizer not available. Last error: {e4}\n"
                                    f"Try installing: pip install transformers[torch] or ensure GPTNeoXTokenizer is available."
                                )
            else:
                # For non-Pythia models, use standard AutoTokenizer
                try:
                    # First try with trust_remote_code and use_fast=False
                    GLOBAL_TOKENIZER = AutoTokenizer.from_pretrained(
                        reference_path,
                        trust_remote_code=True,
                        use_fast=False
                    )
                except Exception as e1:
                    print(f"   AutoTokenizer with use_fast=False failed: {e1}")
                    # Try with use_fast=True
                    try:
                        GLOBAL_TOKENIZER = AutoTokenizer.from_pretrained(
                            reference_path,
                            trust_remote_code=True,
                            use_fast=True
                        )
                    except Exception as e2:
                        raise RuntimeError(f"Failed to load tokenizer: {e2}")
            
            # Ensure tokenizer has pad token
            if GLOBAL_TOKENIZER.pad_token is None:
                if GLOBAL_TOKENIZER.eos_token is not None:
                    GLOBAL_TOKENIZER.pad_token = GLOBAL_TOKENIZER.eos_token
                elif GLOBAL_TOKENIZER.bos_token is not None:
                    GLOBAL_TOKENIZER.pad_token = GLOBAL_TOKENIZER.bos_token
                else:
                    GLOBAL_TOKENIZER.add_special_tokens({"pad_token": "[PAD]"})
            print("✅ Tokenizer loaded")
        except Exception as e:
            raise RuntimeError(f"Failed to load tokenizer: {str(e)}")
    
    # Determine device to use (ensure both models use the same device)
    if torch.cuda.is_available():
        # Use the first available GPU to ensure both models are on the same device
        device = torch.device("cuda:0")
        print(f"🖥️ Using device: {device}")
    else:
        device = torch.device("cpu")
        print(f"🖥️ Using device: {device}")
    
    # Load reference model
    if GLOBAL_REFERENCE_MODEL is None or CURRENT_REFERENCE_PATH != reference_path:
        print(f"🔄 Loading reference model from {reference_path}...")
        try:
            # Load model to CPU first, then move to target device
            # This ensures the entire model is on one device
            GLOBAL_REFERENCE_MODEL = AutoModelForCausalLM.from_pretrained(
                reference_path,
                trust_remote_code=True,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                low_cpu_mem_usage=True
            )
            # Move entire model to the target device
            GLOBAL_REFERENCE_MODEL = GLOBAL_REFERENCE_MODEL.to(device)
            GLOBAL_REFERENCE_MODEL.eval()
            CURRENT_REFERENCE_PATH = reference_path
            print(f"✅ Reference model loaded on {device}")
        except Exception as e:
            raise RuntimeError(f"Failed to load reference model: {str(e)}")
    
    # Load updated model
    if GLOBAL_UPDATED_MODEL is None or CURRENT_UPDATED_PATH != updated_path:
        print(f"🔄 Loading updated model from {updated_path}...")
        try:
            # Load model to CPU first, then move to target device
            # This ensures the entire model is on one device
            GLOBAL_UPDATED_MODEL = AutoModelForCausalLM.from_pretrained(
                updated_path,
                trust_remote_code=True,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                low_cpu_mem_usage=True
            )
            # Move entire model to the same device as reference model
            GLOBAL_UPDATED_MODEL = GLOBAL_UPDATED_MODEL.to(device)
            GLOBAL_UPDATED_MODEL.eval()
            CURRENT_UPDATED_PATH = updated_path
            print(f"✅ Updated model loaded on {device}")
        except Exception as e:
            raise RuntimeError(f"Failed to load updated model: {str(e)}")

@app.post("/deploy")
def deploy(req: DeployRequest, api_key: str = Depends(verify_api_key)):
    """
    Model deployment endpoint (compatible with frontend deployment requests).
    This endpoint is mainly used to notify the server to prepare models. 
    Actual analysis is performed in /run_analysis.
    """
    global GLOBAL_REFERENCE_MODEL, GLOBAL_TOKENIZER, CURRENT_REFERENCE_PATH
    
    model_path = req.model_path.strip()
    
    # Check if path exists (if it's a local path)
    import os
    if os.path.exists(model_path) or '/' in model_path or model_path in ['gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl']:
        # If it's a Hugging Face model ID or valid path, return success
        # Note: We don't actually load the model here, as analysis requires two models
        return {
            "status": "success",
            "message": f"Server has received the instruction and is deploying in the background: {model_path}"
        }
    else:
        return {
            "status": "error",
            "message": f"Path not found on server: {model_path}"
        }

@app.post("/run_dynamic_code")
def run_dynamic_code(req: CodeRequest, request: Request, api_key: str = Depends(verify_api_key)):
    """
    Execute dynamic code asynchronously.
    
    Returns 202 Accepted with task_id immediately, then executes code in background.
    Use /task_status/{task_id} to poll for results.
    """
    import time
    request_start = time.time()
    
    # Check request size
    try:
        if request:
            content_length = request.headers.get("content-length")
            if content_length and int(content_length) > MAX_REQUEST_SIZE:
                raise HTTPException(
                    status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                    detail=f"Request too large. Maximum size: {MAX_REQUEST_SIZE / 1024 / 1024:.1f}MB"
                )
    except (AttributeError, ValueError, TypeError):
        pass  # Ignore if headers are not available or invalid
    
    # Check code size
    if len(req.code) > MAX_REQUEST_SIZE:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Code too large. Maximum size: {MAX_REQUEST_SIZE / 1024 / 1024:.1f}MB"
        )
    
    # Generate unique task ID
    task_id = str(uuid.uuid4())
    
    # Create task entry
    with TASKS_LOCK:
        TASKS[task_id] = {
            "status": TaskStatus.PENDING,
            "created_at": datetime.now().isoformat(),
            "model_path": req.model_path,
            "code_length": len(req.code),
        }
    
    # Start background thread to execute code
    thread = threading.Thread(
        target=_execute_dynamic_code_task,
        args=(task_id, req),
        daemon=True
    )
    thread.start()
    
    request_elapsed = time.time() - request_start
    print(f"📋 Created task {task_id} for dynamic code execution (took {request_elapsed*1000:.1f}ms)")
    
    # Return 202 Accepted with task_id immediately
    return JSONResponse(
        status_code=status.HTTP_202_ACCEPTED,
        content={
            "status": "accepted",
            "task_id": task_id,
            "message": "Dynamic code execution task created. Use /task_status/{task_id} to check status."
        }
    )

def _execute_dynamic_code_task(task_id: str, req: CodeRequest):
    """
    Execute dynamic code task in background thread.
    Updates task status in TASKS dictionary.
    """
    with TASKS_LOCK:
        TASKS[task_id]["status"] = TaskStatus.RUNNING
        TASKS[task_id]["started_at"] = datetime.now().isoformat()
    
    try:
        print(f"🔎 [Task {task_id}] Starting dynamic code execution for model: {req.model_path}")
        
        global GLOBAL_REFERENCE_MODEL, GLOBAL_TOKENIZER, CURRENT_REFERENCE_PATH
        
        # If the path has changed or not loaded yet, load it
        if CURRENT_REFERENCE_PATH != req.model_path:
            print(f"🔄 [Task {task_id}] Loading model: {req.model_path} ...")
            try:
                GLOBAL_TOKENIZER = AutoTokenizer.from_pretrained(req.model_path, trust_remote_code=True)
                GLOBAL_REFERENCE_MODEL = AutoModelForCausalLM.from_pretrained(
                    req.model_path, 
                    device_map="auto", 
                    trust_remote_code=True,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
                )
                CURRENT_REFERENCE_PATH = req.model_path
                print(f"✅ [Task {task_id}] Model loaded successfully")
            except Exception as e:
                raise RuntimeError(f"Model loading failed: {str(e)}")

        # Prepare execution environment
        exec_globals = {
            "model": GLOBAL_REFERENCE_MODEL,
            "tokenizer": GLOBAL_TOKENIZER,
            "torch": torch,
            "result": None  # Frontend code must assign the result to this variable
        }

        # Execute code sent from frontend
        print(f"🚀 [Task {task_id}] Executing dynamic code...")
        try:
            # Capture stdout output (print content)
            f = io.StringIO()
            with contextlib.redirect_stdout(f):
                exec(req.code, exec_globals)
            
            output_log = f.getvalue()
            
            # Get the 'result' calculated in the code
            calc_result = exec_globals.get("result")
            
            result = {
                "status": "success",
                "log": output_log,     # print content
                "data": calc_result    # Final calculation result (e.g., loss, accuracy, etc.)
            }
            
            print(f"✅ [Task {task_id}] Dynamic code execution completed")
            
        except Exception as e:
            import traceback
            error_traceback = traceback.format_exc()
            raise RuntimeError(
                f"Code execution error: {str(e)}\n"
                f"Traceback: {error_traceback}"
            )
        
        # Update task with result
        with TASKS_LOCK:
            TASKS[task_id]["status"] = TaskStatus.COMPLETED
            TASKS[task_id]["completed_at"] = datetime.now().isoformat()
            TASKS[task_id]["result"] = result
        
        print(f"✅ [Task {task_id}] Task completed successfully and status updated")
        
    except Exception as e:
        import traceback
        error_traceback = traceback.format_exc()
        error_msg = f"Dynamic code execution failed: {str(e)}"
        
        print(f"❌ [Task {task_id}] {error_msg}")
        print(f"Traceback:\n{error_traceback}")
        
        with TASKS_LOCK:
            TASKS[task_id]["status"] = TaskStatus.FAILED
            TASKS[task_id]["completed_at"] = datetime.now().isoformat()
            TASKS[task_id]["error"] = {
                "msg": error_msg,
                "traceback": error_traceback
            }

def _execute_analysis_task(task_id: str, req: AnalysisRequest):
    """
    Execute analysis task in background thread.
    Updates task status in TASKS dictionary.
    """
    with TASKS_LOCK:
        TASKS[task_id]["status"] = TaskStatus.RUNNING
        TASKS[task_id]["started_at"] = datetime.now().isoformat()
    
    try:
        print(f"🔎 [Task {task_id}] Starting analysis: feature={req.feature}, "
              f"reference={req.model_reference_path}, updated={req.model_path}")
        
        # Optimize parameters for FIM analysis
        batch_size = req.batch_size
        num_batches = req.num_batches
        if req.feature.lower() == "fim":
            if batch_size > 1:
                print(f"⚠️ [Task {task_id}] FIM analysis: Reducing batch_size from {batch_size} to 1")
                batch_size = 1
            if num_batches > 3:
                print(f"⚠️ [Task {task_id}] FIM analysis: Reducing num_batches from {num_batches} to 3")
                num_batches = 3
        
        # 1. Load models (if needed)
        load_models_if_needed(req.model_reference_path, req.model_path)
        
        # 2. Execute analysis
        result = None
        if req.analysis_code:
            print(f"📦 [Task {task_id}] Using analysis code from frontend...")
            import json
            code_files = json.loads(req.analysis_code)
            
            if not code_files:
                raise ValueError("No code files received from frontend")
            
            print(f"📦 [Task {task_id}] Received {len(code_files)} code file(s): {list(code_files.keys())}")
            print(f"🚀 [Task {task_id}] Executing analysis code...")
            result = execute_analysis_code(
                code_files=code_files,
                feature=req.feature,
                query=req.query,
                device=req.device,
                batch_size=batch_size,
                num_batches=num_batches,
                max_length=req.max_length,
            )
            print(f"📊 [Task {task_id}] Analysis code execution returned: status={result.get('status') if result else 'None'}")
            
            # Check if execute_analysis_code returned an error
            if not result:
                raise RuntimeError("execute_analysis_code returned None")
            if result.get("status") == "error":
                raise RuntimeError(
                    f"Analysis code execution failed: {result.get('msg', 'Unknown error')}\n"
                    f"Traceback: {result.get('traceback', '')}"
                )
            if result.get("status") != "success":
                raise RuntimeError(
                    f"Unexpected status from execute_analysis_code: {result.get('status')}"
                )
            
            # Verify result has data
            if "data" not in result:
                raise RuntimeError("execute_analysis_code result missing 'data' field")
            
            data = result.get("data", {})
            viz_count = len(data.get("visualizations", []))
            print(f"✅ [Task {task_id}] Analysis completed: {viz_count} visualizations generated")
        else:
            # Import from server
            import sys
            import os
            
            current_dir = os.path.dirname(os.path.abspath(__file__))
            parent_dir = os.path.dirname(current_dir)
            
            toolkit_env_path = os.environ.get("REPRESENTATIONAL_TOOLKIT_PATH")
            if toolkit_env_path and os.path.exists(toolkit_env_path):
                possible_paths = [toolkit_env_path]
            else:
                possible_paths = [
                    os.path.join(current_dir, "representational_toolkit"),
                    os.path.join(parent_dir, "representational_toolkit"),
                    os.path.join(current_dir, "src", "unlearning_detection", "representational_toolkit"),
                    os.path.join(parent_dir, "src", "unlearning_detection", "representational_toolkit"),
                    os.path.join(current_dir, "..", "src", "unlearning_detection", "representational_toolkit"),
                ]
            
            toolkit_path = None
            for path in possible_paths:
                abs_path = os.path.abspath(path)
                if os.path.exists(abs_path) and os.path.isdir(abs_path):
                    analysis_file = os.path.join(abs_path, "analysis.py")
                    if os.path.exists(analysis_file):
                        toolkit_path = abs_path
                        break
            
            if toolkit_path:
                toolkit_parent = os.path.dirname(toolkit_path)
                if toolkit_parent not in sys.path:
                    sys.path.insert(0, toolkit_parent)
                
                try:
                    if "src" in toolkit_path and "unlearning_detection" in toolkit_path:
                        from src.unlearning_detection.representational_toolkit.analysis import run_feature_analysis
                    else:
                        from representational_toolkit.analysis import run_feature_analysis
                except ImportError:
                    if toolkit_path not in sys.path:
                        sys.path.insert(0, toolkit_path)
                    import importlib.util
                    spec = importlib.util.spec_from_file_location(
                        "analysis", 
                        os.path.join(toolkit_path, "analysis.py")
                    )
                    if spec and spec.loader:
                        analysis_module = importlib.util.module_from_spec(spec)
                        spec.loader.exec_module(analysis_module)
                        run_feature_analysis = analysis_module.run_feature_analysis
            else:
                try:
                    from representational_toolkit.analysis import run_feature_analysis
                except ImportError:
                    try:
                        from src.unlearning_detection.representational_toolkit.analysis import run_feature_analysis
                    except ImportError:
                        raise ImportError(f"Cannot find representational_toolkit directory")
            
            if run_feature_analysis is None:
                raise RuntimeError("Failed to import run_feature_analysis function")
            
            print(f"🚀 [Task {task_id}] Running {req.feature} analysis...")
            analysis_result = run_feature_analysis(
                feature=req.feature,
                model_reference_path=req.model_reference_path,
                model_path=req.model_path,
                query=req.query,
                device=req.device,
                batch_size=batch_size,
                num_batches=num_batches,
                max_length=req.max_length,
            )
            
            # Process results
            visualizations = []
            warnings = []
            
            if hasattr(analysis_result, 'warnings'):
                warnings = analysis_result.warnings if analysis_result.warnings else []
            
            if hasattr(analysis_result, 'visualizations'):
                for viz in analysis_result.visualizations:
                    image_data = getattr(viz, 'data', b'')
                    if isinstance(image_data, bytes):
                        image_b64 = base64.b64encode(image_data).decode('utf-8')
                    elif isinstance(image_data, str):
                        image_b64 = image_data
                    else:
                        image_b64 = ""
                    
                    visualizations.append({
                        "title": getattr(viz, 'title', 'Visualization'),
                        "data": image_b64,
                        "mime_type": getattr(viz, 'mime_type', 'image/png'),
                        "description": getattr(viz, 'description', None),
                    })
            
            result = {
                "status": "success",
                "data": {
                    "visualizations": visualizations,
                    "warnings": warnings,
                }
            }
        
        # Verify result before updating task
        if result is None:
            raise RuntimeError("Result is None, cannot update task")
        if not isinstance(result, dict):
            raise RuntimeError(f"Result is not a dict: {type(result)}")
        if "status" not in result:
            raise RuntimeError("Result missing 'status' field")
        if result.get("status") != "success":
            raise RuntimeError(f"Result status is not 'success': {result.get('status')}")
        
        print(f"📝 [Task {task_id}] Updating task status to COMPLETED...")
        
        # Update task with result
        with TASKS_LOCK:
            TASKS[task_id]["status"] = TaskStatus.COMPLETED
            TASKS[task_id]["completed_at"] = datetime.now().isoformat()
            TASKS[task_id]["result"] = result
        
        print(f"✅ [Task {task_id}] Analysis completed successfully and task status updated")
        
    except Exception as e:
        import traceback
        error_traceback = traceback.format_exc()
        error_msg = f"Analysis failed: {str(e)}"
        
        print(f"❌ [Task {task_id}] {error_msg}")
        print(f"Traceback:\n{error_traceback}")
        
        with TASKS_LOCK:
            TASKS[task_id]["status"] = TaskStatus.FAILED
            TASKS[task_id]["completed_at"] = datetime.now().isoformat()
            TASKS[task_id]["error"] = {
                "msg": error_msg,
                "traceback": error_traceback
            }


@app.post("/run_analysis")
def run_analysis(req: AnalysisRequest, request: Request, api_key: str = Depends(verify_api_key)):
    """
    Execute representational analysis asynchronously.
    
    Returns 202 Accepted with task_id immediately, then executes analysis in background.
    Use /task_status/{task_id} to poll for results.
    """
    import time
    request_start = time.time()
    
    # Check request size
    try:
        if request:
            content_length = request.headers.get("content-length")
            if content_length and int(content_length) > MAX_REQUEST_SIZE:
                raise HTTPException(
                    status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                    detail=f"Request too large. Maximum size: {MAX_REQUEST_SIZE / 1024 / 1024:.1f}MB"
                )
    except (AttributeError, ValueError, TypeError):
        pass  # Ignore if headers are not available or invalid
    
    # Check analysis code size
    if req.analysis_code:
        code_size = len(req.analysis_code)
        if code_size > MAX_REQUEST_SIZE:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"Analysis code too large. Maximum size: {MAX_REQUEST_SIZE / 1024 / 1024:.1f}MB"
            )
        elif code_size > 1_000_000:  # 1MB
            print(f"⚠️ Large request detected: {code_size / 1024 / 1024:.2f} MB of code")
        else:
            print(f"📦 Request size: {code_size / 1024:.2f} KB of code")
    
    # Generate unique task ID
    task_id = str(uuid.uuid4())
    
    # Create task entry
    with TASKS_LOCK:
        TASKS[task_id] = {
            "status": TaskStatus.PENDING,
            "created_at": datetime.now().isoformat(),
            "feature": req.feature,
            "model_reference_path": req.model_reference_path,
            "model_path": req.model_path,
        }
    
    # Start background thread to execute analysis
    thread = threading.Thread(
        target=_execute_analysis_task,
        args=(task_id, req),
        daemon=True
    )
    thread.start()
    
    request_elapsed = time.time() - request_start
    print(f"📋 Created task {task_id} for {req.feature} analysis (took {request_elapsed*1000:.1f}ms)")
    
    # Return 202 Accepted with task_id immediately
    return JSONResponse(
        status_code=status.HTTP_202_ACCEPTED,
        content={
            "status": "accepted",
            "task_id": task_id,
            "message": "Analysis task created. Use /task_status/{task_id} to check status."
        }
    )


@app.get("/task_status/{task_id}")
def get_task_status(task_id: str, api_key: str = Depends(verify_api_key)):
    """
    Get status of an analysis task.
    
    Returns:
    - 200: Task completed or failed (includes result/error)
    - 202: Task still running
    - 404: Task not found
    """
    with TASKS_LOCK:
        task = TASKS.get(task_id)
    
    if task is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Task {task_id} not found"
        )
    
    task_status = task["status"]
    
    if task_status == TaskStatus.COMPLETED:
        return {
            "status": "completed",
            "task_id": task_id,
            "result": task.get("result"),
            "created_at": task.get("created_at"),
            "started_at": task.get("started_at"),
            "completed_at": task.get("completed_at"),
        }
    elif task_status == TaskStatus.FAILED:
        return {
            "status": "failed",
            "task_id": task_id,
            "error": task.get("error"),
            "created_at": task.get("created_at"),
            "started_at": task.get("started_at"),
            "completed_at": task.get("completed_at"),
        }
    elif task_status == TaskStatus.RUNNING:
        return JSONResponse(
            status_code=status.HTTP_202_ACCEPTED,
            content={
                "status": "running",
                "task_id": task_id,
                "created_at": task.get("created_at"),
                "started_at": task.get("started_at"),
                "message": "Analysis is still running. Please poll again later."
            }
        )
    else:  # PENDING
        return JSONResponse(
            status_code=status.HTTP_202_ACCEPTED,
            content={
                "status": "pending",
                "task_id": task_id,
                "created_at": task.get("created_at"),
                "message": "Task is queued and will start soon."
            }
        )

@app.post("/v1/completions")
def completions(req: CompletionRequest, request: Request, api_key: str = Depends(verify_api_key_openai)):
    """
    OpenAI-compatible Completion API endpoint.
    Supports echo=True, max_tokens=0 for getting prompt logprobs (MIN-K% PROB analysis).
    """
    try:
        # Load model if needed
        load_single_model_if_needed(req.model)
        
        if GLOBAL_SINGLE_MODEL is None or GLOBAL_SINGLE_TOKENIZER is None:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Model not loaded"
            )
        
        # Tokenize prompt
        prompt_clean = req.prompt.replace('\x00', '')  # Remove null bytes
        input_ids = GLOBAL_SINGLE_TOKENIZER.encode(prompt_clean, return_tensors="pt", add_special_tokens=False)
        device = next(GLOBAL_SINGLE_MODEL.parameters()).device
        input_ids = input_ids.to(device)
        
        token_logprobs = []
        tokens = []
        generated_text = prompt_clean if req.echo else ""
        
        with torch.no_grad():
            # Forward pass to get logits
            outputs = GLOBAL_SINGLE_MODEL(input_ids)
            logits = outputs.logits
            
            # Get log probabilities
            log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
            
            # Extract token logprobs for prompt tokens (for echo=True)
            token_ids_list = input_ids[0].tolist()
            for i in range(len(token_ids_list)):
                token_id = token_ids_list[i]
                tokens.append(GLOBAL_SINGLE_TOKENIZER.decode([token_id]))
                
                # Get logprob for this token (from previous position's prediction)
                if i > 0 and i - 1 < log_probs.shape[1]:
                    logprob = log_probs[0, i - 1, token_id].item()
                    token_logprobs.append(logprob)
                else:
                    # First token: use a default or compute differently
                    token_logprobs.append(0.0)
            
            # If max_tokens=0, we're done (just return prompt logprobs)
            if req.max_tokens == 0:
                generated_text = prompt_clean if req.echo else ""
            else:
                # Generate new tokens
                max_new_tokens = req.max_tokens or 500
                generated_ids = GLOBAL_SINGLE_MODEL.generate(
                    input_ids,
                    max_new_tokens=max_new_tokens,
                    temperature=req.temperature or 1.0,
                    top_p=req.top_p,
                    do_sample=req.temperature is not None and req.temperature > 0,
                    pad_token_id=GLOBAL_SINGLE_TOKENIZER.pad_token_id or GLOBAL_SINGLE_TOKENIZER.eos_token_id,
                )
                
                # Decode generated text
                generated_token_ids = generated_ids[0][len(input_ids[0]):]
                generated_text = GLOBAL_SINGLE_TOKENIZER.decode(generated_token_ids, skip_special_tokens=True)
                
                # For generated tokens, we can't easily get logprobs without re-running the model
                # This is a limitation - in production, you'd want to track logprobs during generation
                if req.logprobs:
                    for token_id in generated_token_ids:
                        tokens.append(GLOBAL_SINGLE_TOKENIZER.decode([token_id.item()]))
                        # Placeholder logprob (in production, track during generation)
                        token_logprobs.append(0.0)
        
        # Build response
        response_data = {
            "id": f"cmpl-{uuid.uuid4().hex[:24]}",
            "object": "text_completion",
            "created": int(time.time()),
            "model": req.model,
            "choices": [{
                "text": generated_text,
                "index": 0,
                "logprobs": None,
                "finish_reason": "length" if req.max_tokens and req.max_tokens > 0 else "stop"
            }]
        }
        
        # Add logprobs if requested
        if req.logprobs and token_logprobs:
            top_logprobs = req.logprobs if isinstance(req.logprobs, int) else 5
            # Build top_logprobs structure
            top_logprobs_list = []
            for i, (token, logprob) in enumerate(zip(tokens, token_logprobs)):
                top_logprobs_list.append({
                    token: logprob
                })
            
            response_data["choices"][0]["logprobs"] = {
                "tokens": tokens,
                "token_logprobs": token_logprobs,
                "top_logprobs": top_logprobs_list,
                "text_offset": list(range(len(prompt_clean))) if req.echo else []
            }
        
        return response_data
        
    except Exception as e:
        import traceback
        error_traceback = traceback.format_exc()
        print(f"❌ Error in /v1/completions: {str(e)}")
        print(f"Traceback:\n{error_traceback}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error generating completion: {str(e)}"
        )

@app.post("/v1/chat/completions")
def chat_completions(req: ChatCompletionRequest, request: Request, api_key: str = Depends(verify_api_key_openai)):
    """
    OpenAI-compatible ChatCompletion API endpoint.
    Supports logprobs for MIN-K% PROB analysis.
    """
    try:
        # Load model if needed
        load_single_model_if_needed(req.model)
        
        if GLOBAL_SINGLE_MODEL is None or GLOBAL_SINGLE_TOKENIZER is None:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Model not loaded"
            )
        
        # Convert messages to prompt
        prompt_parts = []
        for msg in req.messages:
            role = msg.role
            content = msg.content
            if role == "system":
                prompt_parts.append(f"System: {content}")
            elif role == "user":
                prompt_parts.append(f"User: {content}")
            elif role == "assistant":
                prompt_parts.append(f"Assistant: {content}")
        
        prompt = "\n".join(prompt_parts) + "\nAssistant:"
        
        # Tokenize
        input_ids = GLOBAL_SINGLE_TOKENIZER.encode(prompt, return_tensors="pt")
        device = next(GLOBAL_SINGLE_MODEL.parameters()).device
        input_ids = input_ids.to(device)
        
        # Generate
        max_new_tokens = req.max_tokens or 500
        generated_ids = GLOBAL_SINGLE_MODEL.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=req.temperature or 0.7,
            top_p=req.top_p or 0.9,
            do_sample=True,
            pad_token_id=GLOBAL_SINGLE_TOKENIZER.pad_token_id or GLOBAL_SINGLE_TOKENIZER.eos_token_id,
        )
        
        # Decode generated text (only the new tokens)
        generated_token_ids = generated_ids[0][len(input_ids[0]):]
        generated_text = GLOBAL_SINGLE_TOKENIZER.decode(generated_token_ids, skip_special_tokens=True)
        
        # Get logprobs if requested
        logprobs_data = None
        if req.logprobs:
            with torch.no_grad():
                outputs = GLOBAL_SINGLE_MODEL(input_ids, labels=input_ids)
                logits = outputs.logits
                log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
                
                # Extract logprobs for generated tokens
                token_logprobs_list = []
                top_logprobs_list = []
                tokens_list = []
                
                # For generated tokens, we need to compute logprobs
                # This is simplified - in production, you'd want to track logprobs during generation
                for token_id in generated_token_ids:
                    token_id_item = token_id.item()
                    tokens_list.append(GLOBAL_SINGLE_TOKENIZER.decode([token_id_item]))
                    # Simplified: use average logprob (in production, track during generation)
                    token_logprobs_list.append(0.0)
                    top_logprobs_list.append({})
                
                logprobs_data = {
                    "content": [
                        {
                            "token": token,
                            "logprob": logprob,
                            "bytes": None,
                            "top_logprobs": top_logprob
                        }
                        for token, logprob, top_logprob in zip(tokens_list, token_logprobs_list, top_logprobs_list)
                    ]
                }
        
        # Build response
        response_data = {
            "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": req.model,
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": generated_text
                },
                "logprobs": logprobs_data,
                "finish_reason": "stop"
            }]
        }
        
        return response_data
        
    except Exception as e:
        import traceback
        error_traceback = traceback.format_exc()
        print(f"❌ Error in /v1/chat/completions: {str(e)}")
        print(f"Traceback:\n{error_traceback}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error generating chat completion: {str(e)}"
        )

@app.get("/health")
def health_check(api_key: str = Depends(verify_api_key)):
    """Health check endpoint"""
    with TASKS_LOCK:
        task_count = len(TASKS)
        running_tasks = sum(1 for t in TASKS.values() if t.get("status") == TaskStatus.RUNNING)
        pending_tasks = sum(1 for t in TASKS.values() if t.get("status") == TaskStatus.PENDING)
        completed_tasks = sum(1 for t in TASKS.values() if t.get("status") == TaskStatus.COMPLETED)
        failed_tasks = sum(1 for t in TASKS.values() if t.get("status") == TaskStatus.FAILED)
    
    return {
        "status": "ok",
        "models_loaded": {
            "reference": CURRENT_REFERENCE_PATH is not None,
            "updated": CURRENT_UPDATED_PATH is not None,
            "tokenizer": GLOBAL_TOKENIZER is not None,
        },
        "tasks": {
            "total": task_count,
            "running": running_tasks,
            "pending": pending_tasks,
            "completed": completed_tasks,
            "failed": failed_tasks,
        }
    }

if __name__ == "__main__":
    # Remember to kill previous process first: fuser -k 1234/tcp or lsof -ti:1234 | xargs kill
    import os
    port = int(os.environ.get("PORT", 1234))
    # Increase timeout for long-running analyses (FIM can take several minutes)
    # Note: This is for uvicorn keep-alive, but Cloudflare Tunnel has its own timeout (default 100s)
    # To increase Cloudflare timeout, configure it in ~/.cloudflared/config.yml
    timeout_keep_alive = int(os.environ.get("TIMEOUT_KEEP_ALIVE", 1800))  # 30 minutes default (increased from 600)
    
    print(f"🚀 Starting deployment agent on port {port}...")
    print("\n🔐 Security Configuration:")
    # Check if API key was set via environment variable (not auto-generated)
    api_key_from_env = os.environ.get("YOUR_API_KEY", None)
    if api_key_from_env:
        print(f"   ✅ API Key authentication: ENABLED")
        print(f"   ✅ API Key configured via environment variable")
    else:
        print(f"   ⚠️  API Key: {API_KEY[:20]}... (temporary, set YOUR_API_KEY env var)")
    print(f"   ✅ CORS: {'Enabled' if ALLOWED_ORIGINS else 'Disabled'}")
    print(f"   ✅ Max request size: {MAX_REQUEST_SIZE / 1024 / 1024:.1f}MB")
    print("\n📝 Available endpoints (all require X-API-Key header):")
    print("   - POST /deploy - Deploy model (compatibility endpoint)")
    print("   - POST /run_dynamic_code - Execute dynamic Python code (async, returns 202 + task_id)")
    print("   - POST /run_analysis - Run representational analysis (async, returns 202 + task_id)")
    print("   - POST /v1/completions - OpenAI-compatible Completion API (supports logprobs, echo=True)")
    print("   - POST /v1/chat/completions - OpenAI-compatible ChatCompletion API (supports logprobs)")
    print("   - GET /task_status/{task_id} - Check status of any task (analysis or dynamic code)")
    print("   - GET /health - Health check")
    print(f"\n💡 Environment Variables:")
    print(f"   - PORT: Server port (default: 1234)")
    print(f"   - YOUR_API_KEY: Required API key for authentication")
    print(f"   - API_KEYS: Comma-separated list of additional API keys")
    print(f"   - ALLOWED_ORIGINS: Comma-separated CORS origins (default: *)")
    print(f"   - TRUSTED_HOSTS: Comma-separated trusted hosts (optional)")
    print(f"   - MAX_REQUEST_SIZE: Max request size in bytes (default: 50MB)")
    print(f"   - TIMEOUT_KEEP_ALIVE: Keep-alive timeout in seconds (default: 1800s / 30min)")
    print(f"\n✅ Async task processing enabled for all long-running operations!")
    print(f"   - No more Cloudflare timeout issues!")
    print(f"   - All tasks use the same /task_status/{{task_id}} endpoint for polling")
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=port,
        timeout_keep_alive=timeout_keep_alive,
        timeout_graceful_shutdown=30,
    )

