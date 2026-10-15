"""Unit tests for layer selection in the representational analysis toolkit.

Pure name matching, so the tests run without torch, transformers or a GPU:

    pytest tests/test_layer_matching.py
"""

import importlib.util
from pathlib import Path

import pytest

# Loaded by path: importing the package would pull in the Streamlit/OpenAI
# runtime, which these tests do not need.
_MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "unlearning_detection"
    / "representational_toolkit"
    / "layer_matching.py"
)
_spec = importlib.util.spec_from_file_location("layer_matching", _MODULE_PATH)
layer_matching = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(layer_matching)

matches_layer = layer_matching.matches_layer


def _named_parameters(num_layers: int = 24):
    """Parameter names in the layout ``_get_layer_key_pattern`` detects."""
    names = []
    for layer in range(num_layers):
        names.append(f"model.layers.{layer}.self_attn.q_proj.weight")
        names.append(f"model.layers.{layer}.mlp.down_proj.weight")
    return names


@pytest.mark.parametrize("layer_idx", [0, 1, 2, 3, 11, 23])
def test_layer_key_selects_exactly_one_layer(layer_idx):
    names = _named_parameters()
    layer_key = f"model.layers.{layer_idx}"
    selected = [name for name in names if matches_layer(name, layer_key)]

    assert len(selected) == 2
    assert {name.split(".")[2] for name in selected} == {str(layer_idx)}


def test_single_digit_key_does_not_absorb_double_digit_layers():
    """Regression: ``"model.layers.1" in name`` also matched layers 10-19."""
    names = _named_parameters()
    selected = [name for name in names if matches_layer(name, "model.layers.1")]

    assert not any(name.startswith("model.layers.10.") for name in selected)
    assert len(selected) == 2


def test_matches_layer_handles_other_naming_schemes():
    assert matches_layer("transformer.h.3.attn.c_attn.weight", "transformer.h.3")
    assert not matches_layer("transformer.h.31.attn.c_attn.weight", "transformer.h.3")
    assert matches_layer("bert.encoder.layer.0.output.dense.weight", "bert.encoder.layer.0")
    assert matches_layer("model.layers.5", "model.layers.5")
    assert not matches_layer("model.layers.5.mlp.weight", "")
