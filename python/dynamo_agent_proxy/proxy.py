# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generic OpenAI/Anthropic proxy that injects Dynamo ``nvext`` metadata."""

from __future__ import annotations

import argparse
import gzip
import json
import os
import sys
import uuid
import zlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from http.client import HTTPConnection, HTTPSConnection, HTTPResponse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import urlsplit

DEFAULT_BASE_URL = "http://127.0.0.1:8000/v1"
DEFAULT_API_KEY = "dynamo-local"
DEFAULT_LISTEN_HOST = "127.0.0.1"
DEFAULT_LISTEN_PORT = 18080
DEFAULT_SESSION_TYPE_ID = "generic_agent"
DEFAULT_MODEL = "default"
STREAM_CHUNK_SIZE = 64 * 1024

HOP_BY_HOP_HEADERS = {
    "connection",
    "content-length",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
}

REQUEST_BODY_METHODS = {"POST", "PUT", "PATCH"}
OPENAI_INJECTION_PATHS = {
    "/chat/completions",
    "/responses",
    "/v1/chat/completions",
    "/v1/responses",
}
ANTHROPIC_MESSAGES_PATHS = {"/messages", "/v1/messages"}


@dataclass(frozen=True)
class AgentAnnotation:
    session_type_id: str
    session_id: str
    trajectory_id: str
    parent_trajectory_id: str | None = None
    agent_hints: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ProxyConfig:
    upstream_scheme: str
    upstream_host: str
    upstream_port: int
    upstream_prefix: str
    api_key: str
    model: str
    annotation: AgentAnnotation


@dataclass(frozen=True)
class ListenConfig:
    host: str
    port: int


@dataclass(frozen=True)
class RuntimeConfig:
    proxy: ProxyConfig
    listen: ListenConfig
    upstream_display: str


class UnsupportedContentEncoding(ValueError):
    pass


def _env_value(env: Mapping[str, str], key: str) -> str | None:
    value = env.get(key)
    if value is None:
        return None
    stripped = value.strip()
    return stripped or None


def _normalize_base_url(raw_base_url: str | None) -> str:
    raw = (raw_base_url or DEFAULT_BASE_URL).strip().rstrip("/")
    try:
        parsed = urlsplit(raw)
    except ValueError:
        return raw
    if parsed.scheme and parsed.netloc and parsed.path in {"", "/"}:
        return raw + "/v1"
    return raw


def _parse_json_object(raw: str | None) -> dict[str, Any]:
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _parse_hint(value: str) -> tuple[str, Any]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("agent hints must use key=value")
    key, raw_value = value.split("=", 1)
    if not key:
        raise argparse.ArgumentTypeError("agent hint key must not be empty")
    try:
        return key, json.loads(raw_value)
    except json.JSONDecodeError:
        return key, raw_value


def read_proxy_config(
    argv: Sequence[str] | None = None,
    env: Mapping[str, str] | None = None,
) -> RuntimeConfig:
    env = env or os.environ
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--listen-host", default=_env_value(env, "DYNAMO_PROXY_LISTEN_HOST") or DEFAULT_LISTEN_HOST)
    parser.add_argument(
        "--listen-port",
        type=int,
        default=int(_env_value(env, "DYNAMO_PROXY_LISTEN_PORT") or DEFAULT_LISTEN_PORT),
    )
    parser.add_argument(
        "--upstream",
        "--base-url",
        dest="base_url",
        default=_env_value(env, "DYNAMO_BASE_URL") or _env_value(env, "OPENAI_BASE_URL") or DEFAULT_BASE_URL,
    )
    parser.add_argument("--api-key", default=_env_value(env, "DYNAMO_API_KEY") or DEFAULT_API_KEY)
    parser.add_argument("--model", default=_env_value(env, "DYNAMO_MODEL") or DEFAULT_MODEL)
    parser.add_argument(
        "--session-type-id",
        default=_env_value(env, "DYN_AGENT_SESSION_TYPE_ID") or DEFAULT_SESSION_TYPE_ID,
    )
    parser.add_argument("--session-id", default=_env_value(env, "DYN_AGENT_SESSION_ID"))
    parser.add_argument("--trajectory-id", default=_env_value(env, "DYN_AGENT_TRAJECTORY_ID"))
    parser.add_argument(
        "--parent-trajectory-id",
        default=_env_value(env, "DYN_AGENT_PARENT_TRAJECTORY_ID"),
    )
    parser.add_argument("--priority", type=int)
    parser.add_argument("--osl", type=int)
    parser.add_argument(
        "--agent-hint",
        action="append",
        default=[],
        type=_parse_hint,
        help="Additional nvext.agent_hints entry as key=value; value may be JSON.",
    )
    args = parser.parse_args(argv)

    agent_hints = _parse_json_object(_env_value(env, "DYN_AGENT_HINTS"))
    if args.priority is not None:
        agent_hints["priority"] = args.priority
    if args.osl is not None:
        agent_hints["osl"] = args.osl
    for key, value in args.agent_hint:
        agent_hints[key] = value

    normalized = _normalize_base_url(args.base_url)
    upstream = urlsplit(normalized)
    if upstream.scheme not in {"http", "https"}:
        raise SystemExit("--upstream must start with http:// or https://")
    if not upstream.hostname:
        raise SystemExit("--upstream must include a host")

    session_id = args.session_id or f"proxy-{uuid.uuid4().hex}"
    proxy = ProxyConfig(
        upstream_scheme=upstream.scheme,
        upstream_host=upstream.hostname,
        upstream_port=upstream.port or (443 if upstream.scheme == "https" else 80),
        upstream_prefix=upstream.path.rstrip("/"),
        api_key=args.api_key,
        model=args.model,
        annotation=AgentAnnotation(
            session_type_id=args.session_type_id,
            session_id=session_id,
            trajectory_id=args.trajectory_id or f"{session_id}:main",
            parent_trajectory_id=args.parent_trajectory_id,
            agent_hints=agent_hints,
        ),
    )
    return RuntimeConfig(
        proxy=proxy,
        listen=ListenConfig(host=args.listen_host, port=args.listen_port),
        upstream_display=normalized,
    )


def merge_dynamo_metadata(payload: Any, annotation: AgentAnnotation) -> dict[str, Any]:
    payload_record = dict(payload) if isinstance(payload, dict) else {}
    existing_nvext = payload_record.get("nvext") if isinstance(payload_record.get("nvext"), dict) else {}
    existing_agent_context = (
        existing_nvext.get("agent_context")
        if isinstance(existing_nvext.get("agent_context"), dict)
        else {}
    )
    existing_agent_hints = (
        existing_nvext.get("agent_hints")
        if isinstance(existing_nvext.get("agent_hints"), dict)
        else {}
    )

    agent_context: dict[str, Any] = dict(existing_agent_context)
    agent_context.update(
        {
            "session_type_id": annotation.session_type_id,
            "session_id": annotation.session_id,
            "trajectory_id": annotation.trajectory_id,
        }
    )
    if annotation.parent_trajectory_id:
        agent_context["parent_trajectory_id"] = annotation.parent_trajectory_id

    nvext = dict(existing_nvext)
    nvext["agent_context"] = agent_context
    if annotation.agent_hints or existing_agent_hints:
        agent_hints = dict(existing_agent_hints)
        agent_hints.update(annotation.agent_hints)
        nvext["agent_hints"] = agent_hints

    payload_record["nvext"] = nvext
    return payload_record


def _is_json_content_type(value: str | None) -> bool:
    if not value:
        return False
    media_type = value.split(";", 1)[0].strip().lower()
    return media_type == "application/json" or media_type.endswith("+json")


def _decode_request_body(body: bytes, content_encoding: str | None) -> bytes:
    encodings = [
        item.strip().lower()
        for item in (content_encoding or "identity").split(",")
        if item.strip()
    ]
    for encoding in reversed(encodings):
        if encoding == "identity":
            continue
        if encoding in {"gzip", "x-gzip"}:
            try:
                body = gzip.decompress(body)
            except (EOFError, OSError) as exc:
                raise UnsupportedContentEncoding(f"invalid gzip request body: {exc}") from exc
            continue
        if encoding == "deflate":
            try:
                body = zlib.decompress(body)
            except zlib.error as exc:
                raise UnsupportedContentEncoding(f"invalid deflate request body: {exc}") from exc
            continue
        raise UnsupportedContentEncoding(f"unsupported request Content-Encoding: {encoding}")
    return body


def annotate_json_request_body(
    headers: Mapping[str, str],
    body: bytes,
    annotation: AgentAnnotation,
) -> tuple[bytes, bool]:
    if not body or not _is_json_content_type(headers.get("content-type")):
        return body, False

    decoded = _decode_request_body(body, headers.get("content-encoding"))
    try:
        payload = json.loads(decoded.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid JSON request body: {exc}") from exc

    if not isinstance(payload, dict):
        return body, False

    annotated = json.dumps(
        merge_dynamo_metadata(payload, annotation),
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return annotated, True


def _normalized_path(request_path: str) -> str:
    path = urlsplit(request_path).path.rstrip("/")
    return path or "/"


def _make_upstream_path(prefix: str, request_path: str, forced_path: str | None = None) -> str:
    split = urlsplit(request_path)
    suffix = forced_path or split.path or "/"
    if suffix == "/v1":
        suffix = "/"
    elif suffix.startswith("/v1/"):
        suffix = suffix[3:]
    if not suffix.startswith("/"):
        suffix = f"/{suffix}"
    base = prefix.rstrip("/")
    path = f"{base}{suffix}" if base else suffix
    if forced_path is None and split.query:
        path = f"{path}?{split.query}"
    return path


def _connection(config: ProxyConfig) -> HTTPConnection:
    if config.upstream_scheme == "https":
        return HTTPSConnection(config.upstream_host, config.upstream_port, timeout=300)
    return HTTPConnection(config.upstream_host, config.upstream_port, timeout=300)


def _anthropic_block_to_text(block: Any) -> str:
    if isinstance(block, str):
        return block
    if not isinstance(block, dict):
        return ""
    if block.get("type") == "text" and isinstance(block.get("text"), str):
        return block["text"]
    if block.get("type") == "tool_result":
        content = block.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return "\n".join(filter(None, (_anthropic_block_to_text(item) for item in content)))
    return ""


def _anthropic_system_to_openai(system: Any) -> str | None:
    if isinstance(system, str):
        return system
    if isinstance(system, list):
        text = "\n".join(filter(None, (_anthropic_block_to_text(block) for block in system)))
        return text or None
    return None


def _anthropic_message_to_openai_messages(message: Mapping[str, Any]) -> list[dict[str, Any]]:
    role = "assistant" if message.get("role") == "assistant" else "user"
    content = message.get("content")
    if isinstance(content, str):
        return [{"role": role, "content": content}]
    if not isinstance(content, list):
        return [{"role": role, "content": ""}]

    messages: list[dict[str, Any]] = []
    text_parts: list[str] = []
    tool_calls: list[dict[str, Any]] = []

    for block in content:
        if not isinstance(block, dict):
            continue
        if block.get("type") == "tool_result":
            if text_parts:
                messages.append({"role": "user", "content": "\n".join(text_parts)})
                text_parts = []
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": str(block.get("tool_use_id", "")),
                    "content": _anthropic_block_to_text(block),
                }
            )
            continue
        if role == "assistant" and block.get("type") == "tool_use":
            tool_calls.append(
                {
                    "id": str(block.get("id") or f"toolu_{uuid.uuid4().hex}"),
                    "type": "function",
                    "function": {
                        "name": str(block.get("name") or "tool"),
                        "arguments": json.dumps(block.get("input") or {}),
                    },
                }
            )
            continue
        text = _anthropic_block_to_text(block)
        if text:
            text_parts.append(text)

    message_body: dict[str, Any] = {"role": role, "content": "\n".join(text_parts)}
    if role == "assistant":
        message_body["content"] = message_body["content"] or None
        if tool_calls:
            message_body["tool_calls"] = tool_calls
    messages.append(message_body)
    return messages


def _anthropic_tools_to_openai(tools: Any) -> Any | None:
    if not isinstance(tools, list):
        return None
    converted = []
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        function: dict[str, Any] = {
            "name": str(tool.get("name") or "tool"),
            "parameters": tool.get("input_schema") if isinstance(tool.get("input_schema"), dict) else {"type": "object", "properties": {}},
        }
        if isinstance(tool.get("description"), str):
            function["description"] = tool["description"]
        converted.append({"type": "function", "function": function})
    return converted


def _anthropic_tool_choice_to_openai(tool_choice: Any) -> Any | None:
    if not isinstance(tool_choice, dict):
        return None
    choice_type = tool_choice.get("type")
    if choice_type == "auto":
        return "auto"
    if choice_type == "any":
        return "required"
    if choice_type == "none":
        return "none"
    if choice_type == "tool" and isinstance(tool_choice.get("name"), str):
        return {"type": "function", "function": {"name": tool_choice["name"]}}
    return None


def translate_anthropic_messages_request(payload: Any, config: ProxyConfig) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("Anthropic messages request body must be a JSON object")

    messages: list[dict[str, Any]] = []
    system = _anthropic_system_to_openai(payload.get("system"))
    if system is not None:
        messages.append({"role": "system", "content": system})
    if isinstance(payload.get("messages"), list):
        for message in payload["messages"]:
            if isinstance(message, dict):
                messages.extend(_anthropic_message_to_openai_messages(message))

    openai_request: dict[str, Any] = {
        "model": payload.get("model") if isinstance(payload.get("model"), str) else config.model,
        "messages": messages,
        "stream": payload.get("stream") is True,
    }
    for anthropic_key, openai_key in (
        ("max_tokens", "max_tokens"),
        ("temperature", "temperature"),
        ("top_p", "top_p"),
    ):
        if isinstance(payload.get(anthropic_key), (int, float)):
            openai_request[openai_key] = payload[anthropic_key]
    if isinstance(payload.get("stop_sequences"), list):
        openai_request["stop"] = payload["stop_sequences"]
    tools = _anthropic_tools_to_openai(payload.get("tools"))
    if tools is not None:
        openai_request["tools"] = tools
    tool_choice = _anthropic_tool_choice_to_openai(payload.get("tool_choice"))
    if tool_choice is not None:
        openai_request["tool_choice"] = tool_choice

    return merge_dynamo_metadata(openai_request, config.annotation)


def _map_finish_reason(finish_reason: Any) -> str | None:
    if finish_reason == "length":
        return "max_tokens"
    if finish_reason == "tool_calls":
        return "tool_use"
    if finish_reason == "stop":
        return "end_turn"
    return None


def _text_content_from_openai_message(message: Mapping[str, Any]) -> str:
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            part.get("text", "")
            for part in content
            if isinstance(part, dict) and part.get("type") == "text" and isinstance(part.get("text"), str)
        )
    return ""


def translate_openai_chat_response_to_anthropic(payload: Any, fallback_model: str) -> dict[str, Any]:
    body = payload if isinstance(payload, dict) else {}
    choices = body.get("choices") if isinstance(body.get("choices"), list) else []
    first_choice = choices[0] if choices and isinstance(choices[0], dict) else {}
    message = first_choice.get("message") if isinstance(first_choice.get("message"), dict) else {}
    content: list[dict[str, Any]] = []
    text = _text_content_from_openai_message(message)
    if text:
        content.append({"type": "text", "text": text})

    tool_calls = message.get("tool_calls") if isinstance(message, dict) else None
    if isinstance(tool_calls, list):
        for tool_call in tool_calls:
            if not isinstance(tool_call, dict) or not isinstance(tool_call.get("function"), dict):
                continue
            function = tool_call["function"]
            tool_input: Any = {}
            arguments = function.get("arguments")
            if isinstance(arguments, str) and arguments.strip():
                try:
                    tool_input = json.loads(arguments)
                except json.JSONDecodeError:
                    tool_input = {"raw_arguments": arguments}
            content.append(
                {
                    "type": "tool_use",
                    "id": str(tool_call.get("id") or f"toolu_{uuid.uuid4().hex}"),
                    "name": str(function.get("name") or "tool"),
                    "input": tool_input,
                }
            )

    usage = body.get("usage") if isinstance(body.get("usage"), dict) else {}
    return {
        "id": body.get("id") if isinstance(body.get("id"), str) else f"msg_{uuid.uuid4().hex}",
        "type": "message",
        "role": "assistant",
        "model": body.get("model") if isinstance(body.get("model"), str) else fallback_model,
        "content": content,
        "stop_reason": _map_finish_reason(first_choice.get("finish_reason")),
        "stop_sequence": None,
        "usage": {
            "input_tokens": usage.get("prompt_tokens") if isinstance(usage.get("prompt_tokens"), int) else 0,
            "output_tokens": usage.get("completion_tokens") if isinstance(usage.get("completion_tokens"), int) else 0,
        },
    }


def _write_sse(handler: BaseHTTPRequestHandler, event: str, data: Any) -> None:
    handler.wfile.write(f"event: {event}\n".encode("utf-8"))
    handler.wfile.write(
        b"data: " + json.dumps(data, separators=(",", ":")).encode("utf-8") + b"\n\n"
    )
    handler.wfile.flush()


def _iter_sse_data(response: HTTPResponse):
    data_lines: list[str] = []
    while line := response.readline():
        decoded = line.decode("utf-8", errors="replace").rstrip("\r\n")
        if decoded == "":
            if data_lines:
                yield "\n".join(data_lines)
                data_lines = []
            continue
        if decoded.startswith("data:"):
            data_lines.append(decoded[5:].lstrip())
    if data_lines:
        yield "\n".join(data_lines)


def _send_anthropic_stream(handler: BaseHTTPRequestHandler, response: HTTPResponse, model: str) -> None:
    handler.send_response(response.status, response.reason)
    handler.send_header("Content-Type", "text/event-stream; charset=utf-8")
    handler.send_header("Cache-Control", "no-cache")
    handler.send_header("Connection", "close")
    handler.end_headers()

    message_id = f"msg_{uuid.uuid4().hex}"
    _write_sse(
        handler,
        "message_start",
        {
            "type": "message_start",
            "message": {
                "id": message_id,
                "type": "message",
                "role": "assistant",
                "model": model,
                "content": [],
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {"input_tokens": 0, "output_tokens": 0},
            },
        },
    )

    text_block_open = False
    next_block_index = 0
    stop_reason: str | None = None
    output_tokens = 0
    tool_blocks: dict[int, int] = {}

    for data in _iter_sse_data(response):
        if data == "[DONE]":
            continue
        try:
            chunk = json.loads(data)
        except json.JSONDecodeError:
            continue
        if isinstance(chunk.get("usage"), dict) and isinstance(chunk["usage"].get("completion_tokens"), int):
            output_tokens = chunk["usage"]["completion_tokens"]
        choices = chunk.get("choices") if isinstance(chunk.get("choices"), list) else []
        for choice in choices:
            if not isinstance(choice, dict):
                continue
            if choice.get("finish_reason") is not None:
                stop_reason = _map_finish_reason(choice.get("finish_reason"))
            delta = choice.get("delta") if isinstance(choice.get("delta"), dict) else {}
            if isinstance(delta.get("content"), str) and delta["content"]:
                if not text_block_open:
                    _write_sse(
                        handler,
                        "content_block_start",
                        {
                            "type": "content_block_start",
                            "index": next_block_index,
                            "content_block": {"type": "text", "text": ""},
                        },
                    )
                    text_block_open = True
                    next_block_index += 1
                _write_sse(
                    handler,
                    "content_block_delta",
                    {
                        "type": "content_block_delta",
                        "index": next_block_index - 1,
                        "delta": {"type": "text_delta", "text": delta["content"]},
                    },
                )
            if isinstance(delta.get("tool_calls"), list):
                if text_block_open:
                    _write_sse(
                        handler,
                        "content_block_stop",
                        {"type": "content_block_stop", "index": next_block_index - 1},
                    )
                    text_block_open = False
                for tool_call in delta["tool_calls"]:
                    if not isinstance(tool_call, dict):
                        continue
                    index = tool_call.get("index") if isinstance(tool_call.get("index"), int) else 0
                    block_index = tool_blocks.get(index)
                    function = tool_call.get("function") if isinstance(tool_call.get("function"), dict) else {}
                    if block_index is None:
                        block_index = next_block_index
                        tool_blocks[index] = block_index
                        next_block_index += 1
                        _write_sse(
                            handler,
                            "content_block_start",
                            {
                                "type": "content_block_start",
                                "index": block_index,
                                "content_block": {
                                    "type": "tool_use",
                                    "id": str(tool_call.get("id") or f"toolu_{uuid.uuid4().hex}"),
                                    "name": str(function.get("name") or "tool"),
                                    "input": {},
                                },
                            },
                        )
                    arguments = function.get("arguments")
                    if isinstance(arguments, str) and arguments:
                        _write_sse(
                            handler,
                            "content_block_delta",
                            {
                                "type": "content_block_delta",
                                "index": block_index,
                                "delta": {"type": "input_json_delta", "partial_json": arguments},
                            },
                        )

    if text_block_open:
        _write_sse(
            handler,
            "content_block_stop",
            {"type": "content_block_stop", "index": next_block_index - 1},
        )
    for block_index in tool_blocks.values():
        _write_sse(handler, "content_block_stop", {"type": "content_block_stop", "index": block_index})
    _write_sse(
        handler,
        "message_delta",
        {
            "type": "message_delta",
            "delta": {"stop_reason": stop_reason or "end_turn", "stop_sequence": None},
            "usage": {"output_tokens": output_tokens},
        },
    )
    _write_sse(handler, "message_stop", {"type": "message_stop"})


def make_handler(config: ProxyConfig) -> type[BaseHTTPRequestHandler]:
    class DynamoAgentProxyHandler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_GET(self) -> None:
            self._proxy_openai()

        def do_HEAD(self) -> None:
            self._proxy_openai()

        def do_POST(self) -> None:
            if _normalized_path(self.path) in ANTHROPIC_MESSAGES_PATHS:
                self._proxy_anthropic_messages()
            else:
                self._proxy_openai()

        def do_PUT(self) -> None:
            self._proxy_openai()

        def do_PATCH(self) -> None:
            self._proxy_openai()

        def do_DELETE(self) -> None:
            self._proxy_openai()

        def _read_body(self) -> bytes:
            if self.command not in REQUEST_BODY_METHODS:
                return b""
            length_header = self.headers.get("content-length")
            if not length_header:
                return b""
            try:
                length = int(length_header)
            except ValueError as exc:
                raise ValueError(f"invalid Content-Length: {length_header}") from exc
            return self.rfile.read(length)

        def _outgoing_headers(self, annotated: bool, body_length: int) -> dict[str, str]:
            outgoing: dict[str, str] = {}
            for key, value in self.headers.items():
                lower = key.lower()
                if lower in HOP_BY_HOP_HEADERS or lower == "host":
                    continue
                if annotated and lower in {"content-encoding", "content-type"}:
                    continue
                if lower in {"authorization", "x-api-key", "anthropic-version", "accept-encoding"}:
                    continue
                outgoing[key] = value

            outgoing["Host"] = f"{config.upstream_host}:{config.upstream_port}"
            outgoing["Authorization"] = f"Bearer {config.api_key}"
            if not any(key.lower() == "x-request-id" for key in outgoing):
                outgoing["x-request-id"] = str(uuid.uuid4())
            if self.command in REQUEST_BODY_METHODS:
                outgoing["Content-Length"] = str(body_length)
            if annotated:
                outgoing["Content-Type"] = "application/json"
            return outgoing

        def _proxy_openai(self) -> None:
            try:
                request_headers = {key.lower(): value for key, value in self.headers.items()}
                body = self._read_body()
                annotated = False
                if self.command in REQUEST_BODY_METHODS and _normalized_path(self.path) in OPENAI_INJECTION_PATHS:
                    body, annotated = annotate_json_request_body(
                        request_headers,
                        body,
                        config.annotation,
                    )
                headers = self._outgoing_headers(annotated, len(body))
            except UnsupportedContentEncoding as exc:
                self.send_error(415, str(exc))
                return
            except ValueError as exc:
                self.send_error(400, str(exc))
                return

            upstream_path = _make_upstream_path(config.upstream_prefix, self.path)
            try:
                conn = _connection(config)
                conn.request(
                    self.command,
                    upstream_path,
                    body=body if self.command in REQUEST_BODY_METHODS else None,
                    headers=headers,
                )
                response = conn.getresponse()
                self._send_upstream_response(response)
            except OSError as exc:
                self.send_error(502, f"upstream request failed: {exc}")
            finally:
                try:
                    conn.close()
                except UnboundLocalError:
                    pass

        def _proxy_anthropic_messages(self) -> None:
            try:
                body = self._read_body()
                decoded = _decode_request_body(body, self.headers.get("content-encoding"))
                payload = json.loads(decoded.decode("utf-8"))
                translated = translate_anthropic_messages_request(payload, config)
                outgoing_body = json.dumps(translated, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
                headers = self._outgoing_headers(True, len(outgoing_body))
                headers["Content-Type"] = "application/json"
            except UnsupportedContentEncoding as exc:
                self.send_error(415, str(exc))
                return
            except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
                self.send_error(400, f"invalid Anthropic request body: {exc}")
                return

            try:
                conn = _connection(config)
                conn.request(
                    "POST",
                    _make_upstream_path(config.upstream_prefix, self.path, forced_path="/chat/completions"),
                    body=outgoing_body,
                    headers=headers,
                )
                response = conn.getresponse()
                content_type = response.getheader("content-type", "")
                if response.status >= 400:
                    self._send_upstream_response(response)
                    return
                if translated.get("stream") is True and "text/event-stream" in content_type.lower():
                    _send_anthropic_stream(self, response, str(translated.get("model") or config.model))
                    self.close_connection = True
                    return
                response_body = response.read()
                try:
                    upstream_payload = json.loads(response_body.decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                    self.send_error(502, f"invalid upstream JSON response: {exc}")
                    return
                anthropic_body = json.dumps(
                    translate_openai_chat_response_to_anthropic(
                        upstream_payload,
                        str(translated.get("model") or config.model),
                    ),
                    separators=(",", ":"),
                    ensure_ascii=False,
                ).encode("utf-8")
                self.send_response(response.status, response.reason)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(anthropic_body)))
                self.send_header("Connection", "close")
                self.end_headers()
                self.wfile.write(anthropic_body)
                self.close_connection = True
            except OSError as exc:
                self.send_error(502, f"upstream request failed: {exc}")
            finally:
                try:
                    conn.close()
                except UnboundLocalError:
                    pass

        def _send_upstream_response(self, response: HTTPResponse) -> None:
            self.send_response(response.status, response.reason)
            content_type = None
            for key, value in response.getheaders():
                lower = key.lower()
                if lower in HOP_BY_HOP_HEADERS:
                    continue
                if lower == "content-type":
                    content_type = value
                self.send_header(key, value)
            self.send_header("Connection", "close")
            self.end_headers()

            if self.command == "HEAD":
                self.close_connection = True
                return

            media_type = (content_type or "").split(";", 1)[0].strip().lower()
            if media_type == "text/event-stream":
                while line := response.readline():
                    self.wfile.write(line)
                    self.wfile.flush()
            else:
                while chunk := response.read(STREAM_CHUNK_SIZE):
                    self.wfile.write(chunk)
                    self.wfile.flush()
            self.close_connection = True

    return DynamoAgentProxyHandler


def serve(config: RuntimeConfig) -> None:
    handler = make_handler(config.proxy)
    server = ThreadingHTTPServer((config.listen.host, config.listen.port), handler)
    print(
        "dynamo-agent-proxy listening on "
        f"http://{config.listen.host}:{config.listen.port} -> {config.upstream_display}; "
        f"session_id={config.proxy.annotation.session_id}; "
        f"trajectory_id={config.proxy.annotation.trajectory_id}",
        file=sys.stderr,
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def main(argv: Sequence[str] | None = None) -> None:
    serve(read_proxy_config(argv))


if __name__ == "__main__":
    main()
