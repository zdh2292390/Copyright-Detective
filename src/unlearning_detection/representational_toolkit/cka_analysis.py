# src/cka_analyzer/analysis.py

import contextlib
import io
from pathlib import Path
from typing import Dict, List

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from .types import FeatureAnalysisResult, VisualizationItem


def _load_tokenizer(path: str) -> AutoTokenizer:
    local_dir = Path(path)
    try:
        if local_dir.exists() and local_dir.is_dir():
            return AutoTokenizer.from_pretrained(str(local_dir), use_fast=True)
        return AutoTokenizer.from_pretrained(path, use_fast=True)
    except Exception as exc:  # pragma: no cover - HF hub issues
        print(f"[!] Online tokenizer loading failed ({path}): {exc}")
        print("[!] Retrying with local_files_only=True...")
        try:
            return AutoTokenizer.from_pretrained(path, use_fast=True, local_files_only=True)
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


def _get_layer_module_pattern(model: AutoModelForCausalLM) -> str:
    """Detect the layer naming pattern in the model modules."""
    module_names = list(model.named_modules())
    
    # Common patterns
    patterns = [
        "model.layers.{layer_idx}",
        "transformer.h.{layer_idx}",
        "encoder.layer.{layer_idx}",
        "bert.encoder.layer.{layer_idx}",
    ]
    
    for pattern in patterns:
        # Check if any module matches the pattern with layer_idx=0
        test_key = pattern.format(layer_idx=0)
        if any(test_key == name for name, _ in module_names):
            return pattern
    
    # Fallback: try to find any pattern with numbers
    import re
    for name, _ in module_names:
        # Look for patterns like layers.0, h.0, layer.0
        match = re.search(r'\b(layers?|h|layer)\.(\d+)', name)
        if match:
            prefix = match.group(1)
            layer_num = int(match.group(2))
            if layer_num == 0:  # Assume layer 0 exists
                pattern = f"{prefix}.{{layer_idx}}"
                # Find the full path
                parts = name.split('.')[:-2]  # Remove the last two parts (prefix.num)
                if parts:
                    full_pattern = '.'.join(parts) + '.' + pattern
                else:
                    full_pattern = pattern
                return full_pattern
    
    raise ValueError("Unable to detect layer naming pattern in model modules.")


def run_cka_analysis(
    model_reference_path: str,
    model_path: str,
    query: List[str],
    device: str = "cuda",
    batch_size: int = 4,
    num_batches: int = 10,
    max_length: int = 128,
) -> FeatureAnalysisResult:
    """Compute layer-wise linear CKA between two language models."""

    if isinstance(device, str) and device.startswith("cuda") and not torch.cuda.is_available():
        print("[!] CUDA requested but not available; falling back to CPU")
        device = "cpu"
    compute_device = torch.device(device)
    torch_dtype = torch.float16 if compute_device.type == "cuda" else torch.float32

    sanitized_query = [item.strip() for item in query if item and item.strip()]
    if not sanitized_query:
        raise ValueError("At least one non-empty query string is required for CKA analysis.")

    tokenizer = _load_tokenizer(model_reference_path)
    tokenizer_added_pad = _ensure_tokenizer_has_pad(tokenizer)

    class TextDataset(Dataset):
        def __init__(self, prompts: List[str]):
            encodings = tokenizer(
                prompts,
                return_tensors="pt",
                truncation=True,
                padding="max_length",
                max_length=max_length,
            )
            self.input_ids = encodings["input_ids"].to(compute_device)
            self.attention_mask = encodings["attention_mask"].to(compute_device)

        def __len__(self) -> int:
            return self.input_ids.size(0)

        def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
            return {
                "input_ids": self.input_ids[index],
                "attention_mask": self.attention_mask[index],
            }

    dataset = TextDataset(sanitized_query)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

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

    def center_gram(kernel: np.ndarray) -> np.ndarray:
        n = kernel.shape[0]
        u = np.ones((n, n), dtype=kernel.dtype) / n
        return kernel - u @ kernel - kernel @ u + u @ kernel @ u

    def linear_cka(features_x: np.ndarray, features_y: np.ndarray) -> float:
        x_centered = features_x - features_x.mean(0, keepdims=True)
        y_centered = features_y - features_y.mean(0, keepdims=True)
        gram_x = x_centered @ x_centered.T
        gram_y = y_centered @ y_centered.T
        hsic = np.trace(center_gram(gram_x) @ center_gram(gram_y))
        denom = np.sqrt(
            np.trace(center_gram(gram_x) @ center_gram(gram_x))
            * np.trace(center_gram(gram_y) @ center_gram(gram_y))
            + 1e-12
        )
        return float(hsic / denom)

    def extract_activations(model: AutoModelForCausalLM) -> Dict[int, np.ndarray]:
        activations: Dict[int, np.ndarray] = {}
        module_lookup = dict(model.named_modules())
        layer_module_pattern = _get_layer_module_pattern(model)
        layer_names = [name for name in module_lookup if layer_module_pattern.format(layer_idx=0) in name]
        # Filter to only numeric layer indices
        layer_indices = []
        for name in layer_names:
            try:
                idx = int(name.rsplit(".", 1)[-1])
                layer_indices.append(idx)
            except ValueError:
                continue  # Skip non-numeric layer names like 'ln_1'
        layer_indices = sorted(set(layer_indices))  # Remove duplicates and sort

        for layer_idx in layer_indices:
            buffer: List[np.ndarray] = []

            def hook(_, __, output):
                tensor = output[0] if isinstance(output, tuple) else output
                tensor_cpu = tensor[:, 0, :].float().detach().cpu()
                try:
                    arr = tensor_cpu.numpy()
                except RuntimeError as e:
                    if "Numpy is not available" in str(e):
                        arr = np.array(tensor_cpu.tolist())
                    else:
                        raise
                buffer.append(arr)

            layer_key = layer_module_pattern.format(layer_idx=layer_idx)
            hook_handle = module_lookup[layer_key].register_forward_hook(hook)
            try:
                with torch.no_grad():
                    for batch_index, batch in enumerate(loader):
                        if batch_index >= num_batches:
                            break
                        model(**batch)
                activations[layer_idx] = np.concatenate(buffer, axis=0)
            finally:
                hook_handle.remove()
        return activations

    ref_acts = extract_activations(model_ref)
    upd_acts = extract_activations(model_upd)

    del model_ref, model_upd
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    cka_scores = {layer: linear_cka(ref_acts[layer], upd_acts[layer]) for layer in ref_acts}

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

    layers = list(cka_scores.keys())
    values = [cka_scores[layer] for layer in layers]

    fig, ax = plt.subplots(figsize=(5, 3))

    marker_freq = 5
    marker_indices = [idx for idx in range(len(layers)) if idx % marker_freq == 0]

    plot_color = "#1b9e77"
    ax.plot(layers, values, linestyle="--", color=plot_color, label="Updated vs Reference")
    ax.plot(
        [layers[idx] for idx in marker_indices],
        [values[idx] for idx in marker_indices],
        "o",
        color=plot_color,
        markersize=8,
        linestyle="None",
    )

    ax.set_xticks([layers[idx] for idx in marker_indices])
    ax.set_xticklabels([str(layers[idx]) for idx in marker_indices])
    ax.set_xlabel("Layer index")
    ax.set_ylabel("Linear CKA")
    ax.set_title("Layerwise CKA", pad=12)
    ax.set_ylim(-1, 3)
    ax.legend(loc="best", frameon=False, fancybox=True)

    fig.tight_layout()

    image_buffer = io.BytesIO()
    fig.savefig(image_buffer, dpi=300, bbox_inches="tight", format="png")
    plt.close(fig)

    visualization = VisualizationItem(
        title="Layer-wise Linear CKA",
        data=image_buffer.getvalue(),
        description="Linear CKA similarity across transformer layers between updated and reference models.",
    )

    return FeatureAnalysisResult(visualizations=[visualization])

