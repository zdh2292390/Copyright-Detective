"""Persist uploaded files across Streamlit reruns when widgets are temporarily disabled."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import streamlit as st


@dataclass(frozen=True)
class CachedUpload:
    data: bytes
    name: str
    mime_type: str

    @property
    def type(self) -> str:
        return self.mime_type

    def getvalue(self) -> bytes:
        return self.data

    def read(self) -> bytes:
        return self.data


def cache_uploaded_file(cache_key: str, uploaded_file: Any) -> None:
    if uploaded_file is None:
        return
    st.session_state[cache_key] = {
        "data": uploaded_file.getvalue(),
        "name": uploaded_file.name,
        "mime_type": getattr(uploaded_file, "type", "") or "",
    }


def resolve_uploaded_file(cache_key: str, uploaded_file: Any) -> Optional[Any]:
    if uploaded_file is not None:
        cache_uploaded_file(cache_key, uploaded_file)
        return uploaded_file
    cached = st.session_state.get(cache_key)
    if not cached:
        return None
    return CachedUpload(
        data=cached["data"],
        name=cached["name"],
        mime_type=cached.get("mime_type", ""),
    )


def clear_upload_cache(cache_key: str) -> None:
    st.session_state.pop(cache_key, None)
