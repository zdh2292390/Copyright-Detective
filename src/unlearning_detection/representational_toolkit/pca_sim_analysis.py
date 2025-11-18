# src/cka_analyzer/pca_similarity.py

import contextlib
import io
from pathlib import Path
from typing import List

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA
from transformers import AutoModelForCausalLM, AutoTokenizer

from .types import FeatureAnalysisResult, VisualizationItem


def _load_tokenizer(path: str) -> AutoTokenizer:
    local_dir = Path(path)
    try:
        if local_dir.exists() and local_dir.is_dir():
            return AutoTokenizer.from_pretrained(str(local_dir), trust_remote_code=True)
        return AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    except Exception as exc:  # pragma: no cover - HF hub issues
        print(f"[!] Online tokenizer loading failed ({path}): {exc}")
        print("[!] Retrying with local_files_only=True...")
        try:
            return AutoTokenizer.from_pretrained(path, trust_remote_code=True, local_files_only=True)
        except Exception as exc2:
            raise RuntimeError(
                f"Failed to load tokenizer for '{path}'.\n"
                "Tried online access and offline cache lookup but both failed.\n"
                "Possible causes:\n"
                " - The model id is incorrect or points to a private/gated repo (requires HF authentication).\n"
                " - You are offline and the model isn't cached locally.\n"
                " - The path is a Hugging Face cache directory (do not use cache paths directly).\n\n"
                f"Online attempt error: {exc}\n"
                f"Offline attempt error: {exc2}\n\n"
                "Suggested fixes:\n"
                " - Use a Hugging Face model ID (e.g., 'gpt2', 'microsoft/DialoGPT-medium').\n"
                " - Provide a local path to a directory containing the model/tokenizer files.\n"
                " - If using a private model, authenticate with `huggingface-cli login`.\n"
            )


def _ensure_tokenizer_has_pad(tokenizer: AutoTokenizer) -> bool:
    if tokenizer.pad_token is not None:
        return False
    if tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        return False
    if tokenizer.bos_token is not None:
        tokenizer.pad_token = tokenizer.bos_token
        tokenizer.pad_token_id = tokenizer.bos_token_id
        return False
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    return True


def _load_model(
    path: str,
    *,
    tokenizer: AutoTokenizer,
    tokenizer_added_pad: bool,
    device: torch.device,
    torch_dtype: torch.dtype,
) -> AutoModelForCausalLM:
    base_kwargs = {
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,
    }
    dtype_keys = ("dtype", "torch_dtype")
    last_error: Exception | None = None

    for dtype_key in dtype_keys:
        kwargs = dict(base_kwargs)
        kwargs[dtype_key] = torch_dtype
        for local_only in (False, True):
            try:
                try:
                    import accelerate  # type: ignore  # noqa: F401

                    if device.type == "cuda":
                        kwargs["device_map"] = "auto"
                    model = AutoModelForCausalLM.from_pretrained(path, **kwargs)
                except (ImportError, ValueError):
                    kwargs.pop("device_map", None)
                    model = AutoModelForCausalLM.from_pretrained(path, **kwargs).to(device)
                if tokenizer_added_pad:
                    with contextlib.suppress(Exception):
                        model.resize_token_embeddings(len(tokenizer))
                return model.eval()
            except TypeError as exc:
                last_error = exc
                if dtype_key == "dtype" and "unexpected keyword" in str(exc):
                    break
                raise
            except Exception as exc:  # pragma: no cover - IO / HF hub issues
                last_error = exc
                if local_only:
                    break
                print(f"[!] Online model loading failed ({path}): {exc}")
                print("[!] Retrying with local_files_only=True...")
                kwargs["local_files_only"] = True
        if last_error and dtype_key == "dtype" and isinstance(last_error, TypeError):
            continue
        if last_error and not isinstance(last_error, TypeError):
            continue
    assert last_error is not None
    raise last_error


def run_pca_similarity(
    model_reference_path: str,
    model_path: str,
    query: List[str],
    device: str = "cuda",
    max_length: int = 128,
) -> FeatureAnalysisResult:
    """Plot cosine similarity between the top principal components of two models."""

    if isinstance(device, str) and device.startswith("cuda") and not torch.cuda.is_available():
        print("[!] CUDA requested but not available; falling back to CPU")
        device = "cpu"
    compute_device = torch.device(device)
    torch_dtype = torch.float16 if compute_device.type == "cuda" else torch.float32

    sanitized_query = [item.strip() for item in query if item and item.strip()]
    if not sanitized_query:
        raise ValueError("At least one non-empty query string is required for PCA similarity analysis.")
    
    if len(sanitized_query) < 2:
        print("[!] Warning: Only 1 query provided. PCA similarity analysis works best with 2+ queries for better statistical representation.")

    tokenizer = _load_tokenizer(model_reference_path)
    tokenizer_added_pad = _ensure_tokenizer_has_pad(tokenizer)

    model_ref = _load_model(
        model_reference_path,
        tokenizer=tokenizer,
        tokenizer_added_pad=tokenizer_added_pad,
        device=compute_device,
        torch_dtype=torch_dtype,
    )
    model_upd = _load_model(
        model_path,
        tokenizer=tokenizer,
        tokenizer_added_pad=tokenizer_added_pad,
        device=compute_device,
        torch_dtype=torch_dtype,
    )

    def extract_mean(model: AutoModelForCausalLM, layer_idx: int) -> np.ndarray:
        enc = tokenizer(
            sanitized_query,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )
        enc = {key: value.to(compute_device) for key, value in enc.items()}
        with torch.no_grad():
            outputs = model(**enc, output_hidden_states=True)
        hidden_states = outputs.hidden_states
        if hidden_states is None:
            raise RuntimeError("Model did not return hidden states; enable output_hidden_states support.")
        layer_hidden_tensor = hidden_states[layer_idx].float().detach().cpu()
        try:
            layer_hidden = layer_hidden_tensor.numpy()
        except RuntimeError as e:
            if "Numpy is not available" in str(e):
                layer_hidden = np.array(layer_hidden_tensor.tolist())
            else:
                raise
        return layer_hidden.mean(axis=1)

    cfg = model_ref.config
    num_layers = getattr(cfg, "num_hidden_layers", None) or getattr(cfg, "n_layer", None)
    if num_layers is None:
        raise ValueError("Unable to determine the number of decoder layers from the reference model configuration.")
    layers = list(range(num_layers + 1))

    pcs_ref = {}
    for layer_idx in layers:
        feats_ref = extract_mean(model_ref, layer_idx)
        n_samples = feats_ref.shape[0]
        n_features = feats_ref.shape[1]
        n_components = min(1, n_samples, n_features)
        
        if n_components < 1:
            raise ValueError(f"Insufficient data for PCA at layer {layer_idx}: shape={feats_ref.shape}")
        
        pcs_ref[layer_idx] = PCA(n_components=1).fit(feats_ref).components_[0]

    sims = []
    for layer_idx in layers:
        ref_component = pcs_ref[layer_idx]
        feats_upd = extract_mean(model_upd, layer_idx)
        n_samples = feats_upd.shape[0]
        n_features = feats_upd.shape[1]
        n_components = min(1, n_samples, n_features)
        
        if n_components < 1:
            raise ValueError(f"Insufficient data for PCA at layer {layer_idx}: shape={feats_upd.shape}")
        
        upd_component = PCA(n_components=1).fit(feats_upd).components_[0]
        numerator = float(ref_component @ upd_component)
        denominator = float(np.linalg.norm(ref_component) * np.linalg.norm(upd_component))
        sims.append(numerator / denominator if denominator else 0.0)

    del model_ref, model_upd
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["DejaVu Serif"],
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "lines.linewidth": 2,
        "lines.markersize": 6,
        "axes.linewidth": 1.2,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.linestyle": "--",
        "grid.linewidth": 0.6,
        "grid.alpha": 0.6,
        "legend.frameon": True,
        "legend.fontsize": 8,
        "legend.title_fontsize": 8,
        "axes.prop_cycle": mpl.cycler("color", ["#0072B2", "#D55E00", "#009E73"]),
    })

    fig, ax = plt.subplots(figsize=(6, 4))

    marker_freq = 5
    marker_indices = [idx for idx in range(len(layers)) if idx % marker_freq == 0]

    ax.plot(layers, sims, linestyle="-", label="Updated vs Reference")
    ax.plot(
        [layers[idx] for idx in marker_indices],
        [sims[idx] for idx in marker_indices],
        linestyle="None",
        marker="o",
    )

    ax.set_xlabel("Layer index")
    ax.set_ylabel("Cosine similarity of PC1")
    ax.set_title("Layer-wise PCA Similarity", pad=12)
    ax.set_ylim(-1, 1)
    ax.grid(True)
    ax.legend(loc="best")

    fig.tight_layout()

    image_buffer = io.BytesIO()
    fig.savefig(image_buffer, dpi=300, bbox_inches="tight", format="png")
    plt.close(fig)

    visualization = VisualizationItem(
        title="Layer-wise PCA Similarity",
        data=image_buffer.getvalue(),
        description="Cosine similarity of the first principal component across decoder layers.",
    )

    return FeatureAnalysisResult(visualizations=[visualization])
