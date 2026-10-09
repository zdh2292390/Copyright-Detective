"""Moonshot/Kimi API sampling rules from the official model parameter reference.

https://platform.kimi.ai/docs/api/models-overview
"""

from typing import Tuple

KIMI_K2_FIXED_TEMPERATURE = 1.0
KIMI_K2_FIXED_TOP_P = 0.95
KIMI_K2_NON_THINKING_TEMPERATURE = 0.6


def kimi_requires_fixed_sampling(model_name: str) -> bool:
    """Return whether the Kimi model fixes temperature and top-p."""
    name = (model_name or "").lower().strip()
    return name.startswith(("kimi-k2", "kimi-k3"))


def kimi_requires_fixed_temperature(model_name: str) -> bool:
    """Return whether temperature is fixed for the chosen model and mode."""
    return kimi_requires_fixed_sampling(model_name)


def normalize_kimi_sampling_params(
    model_name: str,
    temperature: float,
    top_p: float,
    *,
    thinking_enabled: bool = False,
) -> Tuple[float, float]:
    """Return accepted sampling values without changing other providers.

    K2.6 accepts 0.6 only when the request explicitly disables thinking.
    The application uses non-thinking K2.6; its thinking mode, K2.7 Code,
    and K3 all require 1.0.
    """
    if kimi_requires_fixed_sampling(model_name):
        name = (model_name or "").lower().strip()
        if name == "kimi-k2.6" and thinking_enabled is False:
            return KIMI_K2_NON_THINKING_TEMPERATURE, KIMI_K2_FIXED_TOP_P
        return KIMI_K2_FIXED_TEMPERATURE, KIMI_K2_FIXED_TOP_P
    return temperature, top_p


def kimi_request_extra_body(model_name: str) -> dict:
    """Use non-thinking K2.6; omit unsupported thinking fields for K3/Code."""
    if (model_name or "").lower().strip() == "kimi-k2.6":
        return {"thinking": {"type": "disabled"}}
    return {}
