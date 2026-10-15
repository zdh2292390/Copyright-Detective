"""Parameter-name helpers for the layer-wise representational analyses.

Kept free of ``torch``/``transformers`` imports so the matching behaviour can be
unit-tested without a model runtime.
"""

from __future__ import annotations


def matches_layer(param_name: str, layer_key: str) -> bool:
    """Return ``True`` when ``param_name`` belongs to the layer ``layer_key``.

    ``layer_key`` is a prefix such as ``"model.layers.1"``. A plain substring
    test would also select ``"model.layers.10.*" .. "model.layers.19.*"``, which
    silently merges several layers into a single layer's diagnostic. Matching on
    the dot-delimited prefix keeps each layer separate.
    """

    if not layer_key:
        return False
    return param_name == layer_key or param_name.startswith(layer_key + ".")
