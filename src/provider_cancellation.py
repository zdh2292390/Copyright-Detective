"""Prevent SDK HTTP retries after a run is cancelled without changing retries."""
from __future__ import annotations

from contextlib import contextmanager
from threading import Lock
from weakref import WeakKeyDictionary

import httpx

from src.run_cancellation import (
    cancellable_call, check_cancellation, current_cancellation_token,
)

_HOOK_LOCK = Lock()
_HOOKS = WeakKeyDictionary()


def _sdk_http_client(invoke):
    """Find synchronous transports used by OpenAI, Anthropic and Google SDKs."""
    resource = getattr(invoke, '__self__', None)
    if resource is None:
        return None
    pending = [(resource, 0)]
    seen = set()
    while pending:
        current, depth = pending.pop(0)
        if id(current) in seen:
            continue
        seen.add(id(current))
        if isinstance(current, httpx.Client):
            return current
        if depth >= 2:
            continue
        for name in ('_client', '_api_client', '_httpx_client'):
            child = getattr(current, name, None)
            if child is not None:
                pending.append((child, depth + 1))
    return None


def _check_transport_cancellation(_):
    # A shared client can serve different owners concurrently. The thread that
    # sends/receives HTTP must check its own token, never an installer's token.
    check_cancellation()


@contextmanager
def _cancel_sdk_retries(invoke):
    client = _sdk_http_client(invoke)
    if client is None:
        yield
        return

    with _HOOK_LOCK:
        record = _HOOKS.get(client)
        if record is None:
            callbacks = [client.event_hooks['request'], client.event_hooks['response']]
            for hooks in callbacks:
                hooks.append(_check_transport_cancellation)
            record = {'users': 0, 'callbacks': callbacks}
            _HOOKS[client] = record
        record['users'] += 1
    try:
        yield
    finally:
        with _HOOK_LOCK:
            record['users'] -= 1
            if record['users'] == 0:
                # No scoped request can still be iterating this client's hooks.
                # Keep one check per HTTP event, even when hundreds share it.
                for hooks in record['callbacks']:
                    try:
                        hooks.remove(_check_transport_cancellation)
                    except ValueError:
                        pass
                _HOOKS.pop(client, None)


def provider_call(invoke, *args, **kwargs):
    """Interrupt network waits and stop retries inside synchronous provider SDKs."""
    if current_cancellation_token() is None:
        return invoke(*args, **kwargs)

    def call():
        check_cancellation()
        with _cancel_sdk_retries(invoke):
            return invoke(*args, **kwargs)

    # Hooks live in the network worker, including after its waiter is cancelled.
    # The original SDK retry count, timings and HTTP configuration stay intact.
    return cancellable_call(call)