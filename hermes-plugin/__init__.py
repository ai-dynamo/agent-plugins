# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Inject Hermes session IDs as Dynamo trajectory headers."""

HEADER = "x-dynamo-trajectory-id"


def register(ctx) -> None:
    ctx.register_middleware("llm_request", add_dynamo_trajectory_header)


def add_dynamo_trajectory_header(**kwargs):
    session_id = str(kwargs.get("session_id") or "").strip()
    if not session_id:
        return None

    request = dict(kwargs.get("request") or {})
    raw_headers = request.get("extra_headers")
    headers = dict(raw_headers) if isinstance(raw_headers, dict) else {}
    headers.setdefault(HEADER, session_id)
    request["extra_headers"] = headers
    return {"request": request}
