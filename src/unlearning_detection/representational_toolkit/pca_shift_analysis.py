# src/cka_analyzer/pca_analysis.py

import contextlib
import io
from pathlib import Path
from typing import List

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

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
            # Fix for "loss_type=None" warning
            try:
                config = AutoConfig.from_pretrained(path, trust_remote_code=True, local_files_only=local_only)
                if getattr(config, "loss_type", None) is None:
                    config.loss_type = "ForCausalLMLoss"
                kwargs["config"] = config
            except Exception:
                pass

            try:
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


def run_pca_shift(
    model_reference_path: str,
    model_path: str,
    query: List[str],
    device: str = "cuda",
    max_length: int = 128,
) -> FeatureAnalysisResult:
    """Compute PCA shift diagnostics between a reference and an updated model."""

    if isinstance(device, str) and device.startswith("cuda") and not torch.cuda.is_available():
        print("[!] CUDA requested but not available; falling back to CPU")
        device = "cpu"
    compute_device = torch.device(device)
    torch_dtype = torch.float16 if compute_device.type == "cuda" else torch.float32

    sanitized_query = [item.strip() for item in query if item and item.strip()]
    if not sanitized_query:
        raise ValueError("At least one non-empty query string is required for PCA shift analysis.")
    
    if len(sanitized_query) < 2:
        print("[!] Warning: Only 1 query provided. PCA shift analysis works best with 2+ queries for better statistical representation.")

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

    def extract_mean_hidden(model: AutoModelForCausalLM, layer_idx: int) -> np.ndarray:
        encodings = tokenizer(
            sanitized_query,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )
        encodings = {key: value.to(compute_device) for key, value in encodings.items()}
        with torch.no_grad():
            outputs = model(**encodings, output_hidden_states=True)
        hidden_states = outputs.hidden_states
        if hidden_states is None:
            raise RuntimeError("Model did not return hidden states; enable output_hidden_states support.")
        selected = hidden_states[layer_idx].float().detach().cpu()
        try:
            selected_np = selected.numpy()
        except RuntimeError as e:
            if "Numpy is not available" in str(e):
                selected_np = np.array(selected.tolist())
            else:
                raise
        return selected_np.mean(axis=1)

    cfg = model_ref.config
    num_layers = getattr(cfg, "num_hidden_layers", None) or getattr(cfg, "n_layer", None)
    if num_layers is None:
        raise ValueError("Unable to determine the number of decoder layers from the reference model configuration.")
    layers = list(range(num_layers + 1))

    records = []
    for layer_idx in layers:
        ref_features = extract_mean_hidden(model_ref, layer_idx)
        upd_features = extract_mean_hidden(model_upd, layer_idx)

        # Determine number of PCA components based on available samples
        n_samples = ref_features.shape[0]
        n_features = ref_features.shape[1]
        n_components = min(2, n_samples, n_features)
        
        if n_components < 2:
            # Fall back to 1 component if insufficient samples
            pca = PCA(n_components=1).fit(ref_features)
            comp1 = pca.components_[0]
            comp2 = np.zeros_like(comp1)  # Dummy second component
        else:
            pca = PCA(n_components=2).fit(ref_features)
            comp1, comp2 = pca.components_

        pc1_ref = ref_features.dot(comp1).mean()
        pc2_ref = ref_features.dot(comp2).mean()
        pc1_upd = upd_features.dot(comp1).mean()
        pc2_upd = upd_features.dot(comp2).mean()

        records.append({
            "layer": layer_idx,
            "state": "Reference",
            "shift": 0.0,
            "principal": pc2_ref,
        })
        records.append({
            "layer": layer_idx,
            "state": "Updated",
            "shift": pc1_upd - pc1_ref,
            "principal": pc2_upd,
        })

    del model_ref, model_upd
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    df = pd.DataFrame(records)

    # Modern, professional styling
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans", "Helvetica"],
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "lines.linewidth": 1.5,
        "lines.markersize": 8,
        "axes.linewidth": 1.0,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": "#333333",
        "axes.grid": True,
        "grid.linestyle": "-",
        "grid.linewidth": 0.5,
        "grid.alpha": 0.3,
        "legend.frameon": True,
        "legend.fontsize": 10,
        "legend.title_fontsize": 10,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    })

    # Larger figure for better readability
    fig, ax = plt.subplots(figsize=(6, 5), facecolor="white")

    # Modern color palette
    state_colors = {"Reference": "#E74C3C", "Updated": "#3498DB"}  # Red and Blue
    
    for layer in df["layer"].unique():
        subset = df[df["layer"] == layer].sort_values("state")
        ax.plot(
            subset["shift"],
            subset["principal"],
            color="#BDC3C7",  # Light gray for connections
            linewidth=1.5,
            alpha=0.4,
            zorder=1,
        )

    markers = {"Reference": "o", "Updated": "^"}
    marker_sizes = {"Reference": 120, "Updated": 120}
    
    for state in df["state"].unique():
        subset = df[df["state"] == state]
        ax.scatter(
            subset["shift"],
            subset["principal"],
            marker=markers[state],
            c=state_colors[state],
            edgecolors="white",
            linewidths=2,
            s=marker_sizes[state],
            label=state,
            zorder=3,
            alpha=0.8,
        )

    ax.set_xlabel("Δ PC1", fontweight="medium")
    ax.set_ylabel("PC2", fontweight="medium")
    ax.set_title("PCA Shift Analysis", fontweight="bold", pad=15)

    x_min, x_max = df["shift"].min(), df["shift"].max()
    y_min, y_max = df["principal"].min(), df["principal"].max()
    x_pad = 0.05 * (x_max - x_min)
    y_pad = 0.1 * (y_max - y_min)
    ax.set_xlim(x_min - x_pad, x_max + x_pad)
    ax.set_ylim(y_min - y_pad, y_max + y_pad)

    # Improved legend
    ax.legend(
        loc="upper right",
        frameon=True,
        fancybox=True,
        shadow=True,
        framealpha=0.9,
        edgecolor="#CCCCCC",
        facecolor="white",
    )
    
    # Subtle background
    ax.set_facecolor("#FAFAFA")
    fig.patch.set_facecolor("white")

    plt.tight_layout()

    image_buffer = io.BytesIO()
    plt.savefig(image_buffer, bbox_inches="tight", dpi=200, format="png", facecolor="white")
    plt.close(fig)

    visualization = VisualizationItem(
        title="PCA Shift",
        data=image_buffer.getvalue(),
        description="Scatter plot showing per-layer PCA shift between reference and updated models.",
    )

    return FeatureAnalysisResult(visualizations=[visualization])

