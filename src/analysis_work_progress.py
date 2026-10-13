"""Validate and display domain work progress without counting API checkpoints."""
from __future__ import annotations

from collections.abc import Mapping
import re
from typing import Any


_MAX_TOTAL = 1_000_000_000
_UNIT_RE = re.compile(r"[A-Za-z]+(?:[ -][A-Za-z]+)*")
_STAGE_RE = re.compile(r"[a-z][a-z0-9]*(?:[_.:-][a-z0-9]+)*")
_RESUME_PROGRESS = "Progress will update on Resume"


def normalize_work_progress(value: Any) -> dict[str, int | str] | None:
    """Return only valid scalar progress fields, without modifying the input.

    ``current`` counts finished domain units such as inference runs, not saved
    calls. Unit labels are short plain English text; stages are stable lowercase
    identifiers. Additional metadata is never copied into the display contract.
    """
    if not isinstance(value, Mapping):
        return None
    current = value.get("current")
    total = value.get("total")
    unit = value.get("unit")
    stage = value.get("stage")
    if (type(current) is not int or type(total) is not int
            or not 0 <= current <= total or not 1 <= total <= _MAX_TOTAL):
        return None
    if (not isinstance(unit, str) or not 1 <= len(unit) <= 32
            or _UNIT_RE.fullmatch(unit) is None):
        return None
    if (not isinstance(stage, str) or not 1 <= len(stage) <= 64
            or _STAGE_RE.fullmatch(stage) is None):
        return None
    return {"current": current, "total": total, "unit": unit, "stage": stage}


def format_work_progress(task: Any) -> str:
    """Prefer recorded work progress; older tasks never infer a call fraction.

    Summary queries expose ``work_progress`` directly, while a full task keeps
    it under ``metadata``. A complete legacy task can show its known final state
    even when its original domain count was never recorded.
    """
    if not isinstance(task, Mapping):
        return _RESUME_PROGRESS
    progress = normalize_work_progress(task.get("work_progress"))
    if progress is None:
        metadata = task.get("metadata")
        if isinstance(metadata, Mapping):
            progress = normalize_work_progress(metadata.get("work_progress"))
    if progress is not None:
        return f"**Progress: {progress['current']} / {progress['total']}** {progress['unit']}"
    return "Complete" if task.get("status") == "complete" else _RESUME_PROGRESS
