import contextlib
import gc
import io
from pathlib import Path
from typing import Dict, Iterable, List

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from .types import FeatureAnalysisResult, VisualizationItem


def _warn_if_low_memory() -> None:
    with contextlib.suppress(ImportError):
        import psutil  # type: ignore

        available_gb = psutil.virtual_memory().available / (1024 ** 3)
        if available_gb < 2.0:
            print(
                f"[!] WARNING: Low memory detected (only {available_gb:.1f} GB available).\n"
                "[!] Recommendation: close other applications or choose a smaller model before running FIM analysis."
            )


def _load_tokenizer(path: str) -> AutoTokenizer:
    # Prefer a local directory if the path points to one (useful for offline workflows)
    local_dir = Path(path)
    try:
        if local_dir.exists() and local_dir.is_dir():
            return AutoTokenizer.from_pretrained(str(local_dir), use_fast=True)

        # First attempt: normal (may download from the Hub)
        return AutoTokenizer.from_pretrained(path, use_fast=True)
    except Exception as exc:  # pragma: no cover - network / HF hub issues
        print(f"[!] Online tokenizer loading failed ({path}): {exc}")
        print("[!] Retrying with local_files_only=True...")
        try:
            return AutoTokenizer.from_pretrained(path, use_fast=True, local_files_only=True)
        except Exception as exc2:
            # Provide an actionable error message so users can resolve auth/offline issues quickly.
            raise RuntimeError(
                f"Failed to load tokenizer for '{path}'.\n"
                "Tried online access and offline cache lookup but both failed.\n"
                "Possible causes:\n"
                " - The model id is incorrect or points to a private/gated repo (requires HF authentication).\n"
                " - You are offline and the model isn't cached locally.\n\n"
                f"Online attempt error: {exc}\n"
                f"Offline attempt error: {exc2}\n\n"
                "Suggested fixes:\n"
                " - If the model is on Hugging Face, authenticate with `huggingface-cli login` or `hf auth login` and retry.\n"
                " - Provide a local path to a directory containing the model/tokenizer files (e.g. './models/Qwen2-0.5B').\n"
                " - If you intentionally want to allow downloads, ensure network access and that `local_files_only` is not set in calling code.\n"
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


def _normalise_layers(config: AutoConfig, layers: Iterable[int] | None) -> List[int]:
    num_layers = getattr(config, "num_hidden_layers", None) or getattr(config, "n_layer", None)
    if num_layers is None:
        raise ValueError("Unable to determine the number of transformer layers from the reference model configuration.")
    if layers is None:
        return list(range(num_layers))
    resolved: List[int] = []
    for layer_index in layers:
        if not isinstance(layer_index, int):
            continue
        if 0 <= layer_index < num_layers:
            resolved.append(layer_index)
    if not resolved:
        raise ValueError("No valid layer indices matched the reference model configuration.")
    return resolved


def _get_layer_key_pattern(model: AutoModelForCausalLM) -> str:
    """Detect the layer naming pattern in the model parameters."""
    param_names = [name for name, param in model.named_parameters() if param.requires_grad]
    
    # Common patterns
    patterns = [
        "model.layers.{layer_idx}",
        "transformer.h.{layer_idx}",
        "encoder.layer.{layer_idx}",
        "bert.encoder.layer.{layer_idx}",
    ]
    
    for pattern in patterns:
        # Check if any param matches the pattern with layer_idx=0
        test_key = pattern.format(layer_idx=0)
        if any(test_key in name for name in param_names):
            return pattern
    
    # Fallback: try to find any pattern with numbers
    import re
    layer_matches = []
    for name in param_names:
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
    
    raise ValueError("Unable to detect layer naming pattern in model parameters.")


def run_fim_analysis(
    model_reference_path: str,
    model_path: str,
    query: List[str],
    device: str = "cuda",
    batch_size: int = 4,
    num_batches: int = 10,
    max_length: int = 128,
    layers_to_analyze: List[int] | None = None,
) -> FeatureAnalysisResult:
    """Compute Fisher Information Matrix histograms comparing two language models."""

    _warn_if_low_memory()

    if isinstance(device, str) and device.startswith("cuda") and not torch.cuda.is_available():
        print("[!] CUDA requested but not available; falling back to CPU")
        device = "cpu"
    compute_device = torch.device(device)
    target_dtype = torch.float16 if compute_device.type == "cuda" else torch.float32

    tokenizer = _load_tokenizer(model_reference_path)
    tokenizer_added_pad = _ensure_tokenizer_has_pad(tokenizer)

    class TextDataset(Dataset):
        def __init__(self, items: List[str]):
            encodings = tokenizer(
                items,
                return_tensors="pt",
                truncation=True,
                padding="max_length",
                max_length=max_length,
            )
            self.input_ids = encodings["input_ids"]
            self.attention_mask = encodings["attention_mask"]
            self.labels = encodings["input_ids"]

        def __len__(self) -> int:
            return self.input_ids.size(0)

        def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
            return {
                "input_ids": self.input_ids[index],
                "attention_mask": self.attention_mask[index],
                "labels": self.labels[index],
            }

    dataset = TextDataset([item.strip() for item in query if item and item.strip()])
    if len(dataset) == 0:
        raise ValueError("At least one non-empty query string is required for FIM analysis.")

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=(compute_device.type == "cuda"),
    )

    def _load_model(weights_path: str) -> AutoModelForCausalLM:
        base_kwargs = {
            "trust_remote_code": True,
            "low_cpu_mem_usage": True,
            "dtype": target_dtype,
        }
        
        for local_only in (False, True):
            kwargs = dict(base_kwargs)
            if local_only:
                kwargs["local_files_only"] = True
            
            # Fix for "loss_type=None" warning
            try:
                config = AutoConfig.from_pretrained(weights_path, trust_remote_code=True, local_files_only=local_only)
                if getattr(config, "loss_type", None) is None:
                    config.loss_type = "ForCausalLMLoss"
                kwargs["config"] = config
            except Exception:
                pass

            try:
                model = AutoModelForCausalLM.from_pretrained(weights_path, **kwargs).to(compute_device)
                if tokenizer_added_pad:
                    with contextlib.suppress(Exception):
                        model.resize_token_embeddings(len(tokenizer))
                return model.eval()
            except TypeError as exc:
                # Fallback: try with torch_dtype if dtype is not supported
                if "dtype" in str(exc) and "unexpected keyword" not in str(exc):
                    raise
                kwargs_fallback = dict(base_kwargs)
                kwargs_fallback.pop("dtype", None)
                kwargs_fallback["torch_dtype"] = target_dtype
                if local_only:
                    kwargs_fallback["local_files_only"] = True
                try:
                    try:
                        import accelerate  # type: ignore  # noqa: F401

                        if compute_device.type == "cuda":
                            kwargs_fallback["device_map"] = "auto"
                        model = AutoModelForCausalLM.from_pretrained(weights_path, **kwargs_fallback)
                    except (ImportError, ValueError):
                        kwargs_fallback.pop("device_map", None)
                        model = AutoModelForCausalLM.from_pretrained(weights_path, **kwargs_fallback).to(compute_device)
                    if tokenizer_added_pad:
                        with contextlib.suppress(Exception):
                            model.resize_token_embeddings(len(tokenizer))
                    return model.eval()
                except Exception:
                    if local_only:
                        raise
            except Exception as exc:  # pragma: no cover - IO / HF hub issues
                if local_only:
                    raise
                print(f"[!] Online model loading failed ({weights_path}): {exc}")
                print("[!] Retrying with local_files_only=True...")
        
        raise RuntimeError(f"Failed to load model from {weights_path} after trying all loading strategies.")

    def _compute_fim_diagonal(model: AutoModelForCausalLM, layer_key: str) -> np.ndarray:
        params = [(name, param) for name, param in model.named_parameters() if param.requires_grad and layer_key in name]
        if not params:
            raise ValueError(f"Layer key '{layer_key}' did not match any trainable parameters.")

        accumulators = [torch.zeros(param.numel(), dtype=torch.float32) for _, param in params]
        effective_batches = 0

        for effective_batches, batch in enumerate(loader, start=1):
            if effective_batches > num_batches:
                break
            batch = {key: tensor.to(compute_device) for key, tensor in batch.items()}
            model.zero_grad(set_to_none=True)
            loss = model(**batch).loss
            loss.backward()

            for idx, (_, param) in enumerate(params):
                grad = param.grad
                if grad is None:
                    continue
                grad_cpu = grad.detach().float().cpu().view(-1)
                accumulators[idx] += grad_cpu.pow_(2)

            model.zero_grad(set_to_none=True)

        if effective_batches == 0:
            raise RuntimeError("Failed to execute any dataloader batches for FIM computation.")

        denom = max(1, min(len(loader), num_batches, effective_batches))
        fim_values = torch.cat([accumulator / denom for accumulator in accumulators])
        # Convert to numpy array safely
        try:
            return fim_values.detach().cpu().numpy()
        except RuntimeError as e:
            if "Numpy is not available" in str(e):
                # Fallback: convert to list and then to numpy array
                return np.array(fim_values.detach().cpu().tolist())
            else:
                raise

    with contextlib.suppress(Exception):
        config = AutoConfig.from_pretrained(model_reference_path)
    if 'config' not in locals() or config is None:
        with contextlib.suppress(Exception):
            config = AutoConfig.from_pretrained(model_reference_path, local_files_only=True)
    if 'config' not in locals() or config is None:
        raise RuntimeError(
            f"Unable to load the reference model configuration from '{model_reference_path}'. "
            "Please ensure this is a valid Hugging Face model ID (e.g., 'gpt2') or a local directory containing config.json. "
            "Do not use Hugging Face cache paths directly."
        )

    target_layers = _normalise_layers(config, layers_to_analyze)

    def _collect_fim_by_layer(weights_path: str) -> Dict[int, np.ndarray]:
        model = _load_model(weights_path)
        try:
            layer_key_pattern = _get_layer_key_pattern(model)
            results: Dict[int, np.ndarray] = {}
            for layer_idx in target_layers:
                layer_key = layer_key_pattern.format(layer_idx=layer_idx)
                results[layer_idx] = _compute_fim_diagonal(model, layer_key)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            return results
        finally:
            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    fim_reference = _collect_fim_by_layer(model_reference_path)
    fim_updated = _collect_fim_by_layer(model_path)

    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Serif"],
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.dpi": 150,
        "axes.grid": True,
        "grid.linestyle": "--",
        "grid.linewidth": 0.4,
        "grid.alpha": 0.5,
    })

    plot_colors = {"Reference": "#d62728", "Updated": "#1f77b4"}
    linestyles = {"Reference": "-", "Updated": "--"}

    visualizations: List[VisualizationItem] = []

    for layer_idx in target_layers:
        fim_ref = fim_reference[layer_idx]
        fim_upd = fim_updated[layer_idx]

        fig, ax = plt.subplots(figsize=(3.5, 2.5))
        for tag, values in (("Reference", fim_ref), ("Updated", fim_upd)):
            ax.hist(
                values,
                bins=40,
                histtype="step",
                linewidth=2,
                linestyle=linestyles[tag],
                color=plot_colors[tag],
                label=tag,
            )

        ax.set_xscale("log")
        ax.set_xlabel("FIM diagonal values (log scale)")
        ax.set_ylabel("Frequency")
        ax.set_title(f"FIM Histogram @ Layer {layer_idx}", pad=8)

        ymax = 0
        for values in (fim_ref, fim_upd):
            counts, _ = np.histogram(values, bins=40)
            ymax = max(ymax, counts.max() if len(counts) else 0)
        ax.set_ylim(0, max(1, ymax * 1.2))

        combined = np.concatenate([fim_ref, fim_upd])
        combined = combined[combined > 0]
        if combined.size:
            x_min, x_max = combined.min(), combined.max()
            if x_min == x_max:
                pad_decades = 0.1
            else:
                pad_decades = 0.1 * (np.log10(x_max) - np.log10(x_min))
            ax.set_xlim(
                10 ** (np.log10(x_min) - pad_decades),
                10 ** (np.log10(x_max) + pad_decades),
            )

        ax.legend(loc="upper right", frameon=False, fancybox=True)

        fig.tight_layout()
        image_buffer = io.BytesIO()
        fig.savefig(image_buffer, dpi=100, bbox_inches="tight", format="png")
        plt.close(fig)

        visualizations.append(
            VisualizationItem(
                title=f"Layer {layer_idx}",
                data=image_buffer.getvalue(),
                description="Fisher Information diagonals: Reference vs Updated model" if layer_idx == target_layers[0] else None,
            )
        )

    return FeatureAnalysisResult(visualizations=visualizations)



