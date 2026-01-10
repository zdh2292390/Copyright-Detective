# deploy.py (Server-side)
import uvicorn
from fastapi import FastAPI, HTTPException, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
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

app = FastAPI()

# Global variables to cache models, avoiding reloading on every request
GLOBAL_REFERENCE_MODEL = None
GLOBAL_UPDATED_MODEL = None
GLOBAL_TOKENIZER = None
CURRENT_REFERENCE_PATH = None
CURRENT_UPDATED_PATH = None

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



def load_models_if_needed(reference_path: str, updated_path: str):
    """Load two models on demand (reference and updated)"""
    global GLOBAL_REFERENCE_MODEL, GLOBAL_UPDATED_MODEL, GLOBAL_TOKENIZER
    global CURRENT_REFERENCE_PATH, CURRENT_UPDATED_PATH
    
    # Load tokenizer (usually both models use the same tokenizer, loaded from reference model)
    if GLOBAL_TOKENIZER is None or CURRENT_REFERENCE_PATH != reference_path:
        print(f"🔄 Loading tokenizer from {reference_path}...")
        try:
            GLOBAL_TOKENIZER = AutoTokenizer.from_pretrained(
                reference_path, 
                trust_remote_code=True
            )
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
def deploy(req: DeployRequest):
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
def run_dynamic_code(req: CodeRequest):
    """
    Execute dynamic code asynchronously.
    
    Returns 202 Accepted with task_id immediately, then executes code in background.
    Use /task_status/{task_id} to poll for results.
    """
    import time
    request_start = time.time()
    
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
def run_analysis(req: AnalysisRequest):
    """
    Execute representational analysis asynchronously.
    
    Returns 202 Accepted with task_id immediately, then executes analysis in background.
    Use /task_status/{task_id} to poll for results.
    """
    import time
    request_start = time.time()
    
    # Check request size (warn if too large, but don't block)
    if req.analysis_code:
        code_size = len(req.analysis_code)
        if code_size > 1_000_000:  # 1MB
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
def get_task_status(task_id: str):
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

@app.get("/health")
def health_check():
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
    print("📝 Available endpoints:")
    print("   - POST /deploy - Deploy model (compatibility endpoint)")
    print("   - POST /run_dynamic_code - Execute dynamic Python code (async, returns 202 + task_id)")
    print("   - POST /run_analysis - Run representational analysis (async, returns 202 + task_id)")
    print("   - GET /task_status/{task_id} - Check status of any task (analysis or dynamic code)")
    print("   - GET /health - Health check")
    print(f"\n💡 Tip: Set PORT environment variable to change port (default: 1234)")
    print(f"💡 Tip: Set TIMEOUT_KEEP_ALIVE environment variable to change timeout (default: 1800s / 30min)")
    print(f"✅ Async task processing enabled for all long-running operations!")
    print(f"   - No more Cloudflare timeout issues!")
    print(f"   - All tasks use the same /task_status/{{task_id}} endpoint for polling")
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=port,
        timeout_keep_alive=timeout_keep_alive,
        timeout_graceful_shutdown=30,
    )

