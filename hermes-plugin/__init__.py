# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Inject Hermes session IDs as Dynamo trajectory headers."""

HEADER = "x-dynamo-trajectory-id"
_PATCHED_ATTR = "_dynamo_trajectory_headers_patched"


def register(ctx) -> None:
    ctx.register_hook("pre_api_request", ensure_dynamo_headers)


def ensure_dynamo_headers(**_kwargs) -> None:
    patch_hermes_openai_client()


def patch_hermes_openai_client() -> None:
    from run_agent import AIAgent

    original = AIAgent._create_openai_client
    if getattr(original, _PATCHED_ATTR, False):
        return

    def wrapped(self, client_kwargs, *, reason, shared):
        session_id = str(getattr(self, "session_id", "") or "").strip()
        if session_id:
            client_kwargs = dict(client_kwargs)
            headers = dict(client_kwargs.get("default_headers") or {})
            headers.setdefault(HEADER, session_id)
            client_kwargs["default_headers"] = headers
        return original(self, client_kwargs, reason=reason, shared=shared)

    setattr(wrapped, _PATCHED_ATTR, True)
    AIAgent._create_openai_client = wrapped
