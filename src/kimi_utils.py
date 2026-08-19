"""Moonshot/Kimi API parameter helpers."""

from typing import Tuple

KIMI_K2_FIXED_TEMPERATURE = 1.0
KIMI_K2_FIXED_TOP_P = 0.95


def kimi_requires_fixed_sampling(model_name: str) -> bool:
    """Return True for Kimi K2 models with fixed temperature/top_p constraints."""
    name = (model_name or "").lower().strip()
    return name.startswith("kimi-k2")


def kimi_requires_fixed_temperature(model_name: str) -> bool:
    """Return True for Kimi K2 models that only accept temperature=1."""
    return kimi_requires_fixed_sampling(model_name)


def normalize_kimi_sampling_params(
    model_name: str,
    temperature: float,
    top_p: float,
) -> Tuple[float, float]:
    """Clamp sampling params to values accepted by the target Kimi model."""
    if kimi_requires_fixed_sampling(model_name):
        return KIMI_K2_FIXED_TEMPERATURE, KIMI_K2_FIXED_TOP_P
    return temperature, top_p
