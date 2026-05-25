# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import http.client
import json
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from dynamo_agent_proxy.proxy import (
    AgentAnnotation,
    ProxyConfig,
    annotate_json_request_body,
    make_handler,
    merge_dynamo_metadata,
    read_proxy_config,
    translate_anthropic_messages_request,
    translate_openai_chat_response_to_anthropic,
)


def annotation() -> AgentAnnotation:
    return AgentAnnotation(
        session_type_id="generic_agent",
        session_id="session-1",
        trajectory_id="session-1:main",
        agent_hints={"priority": 7, "osl": 512},
    )


def proxy_config(upstream_port: int = 8000) -> ProxyConfig:
    return ProxyConfig(
        upstream_scheme="http",
        upstream_host="127.0.0.1",
        upstream_port=upstream_port,
        upstream_prefix="/v1",
        api_key="test-key",
        model="fallback-model",
        annotation=annotation(),
    )


def start_server(server: ThreadingHTTPServer) -> None:
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()


def close_server(server: ThreadingHTTPServer) -> None:
    server.shutdown()
    server.server_close()


class RecordingUpstreamHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def do_POST(self) -> None:
        length = int(self.headers.get("content-length", "0"))
        body = self.rfile.read(length)
        self.server.recorded.append(
            {
                "path": self.path,
                "headers": dict(self.headers.items()),
                "body": body,
            }
        )
        if self.path == "/v1/chat/completions":
            response = json.dumps(
                {
                    "id": "chatcmpl-1",
                    "model": "demo",
                    "choices": [
                        {"finish_reason": "stop", "message": {"role": "assistant", "content": "ok"}}
                    ],
                    "usage": {"prompt_tokens": 3, "completion_tokens": 1},
                },
                separators=(",", ":"),
            ).encode("utf-8")
        else:
            response = b'{"ok":true}'
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(response)))
        self.end_headers()
        self.wfile.write(response)

    def log_message(self, format: str, *args: object) -> None:
        return


class StreamingUpstreamHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def do_POST(self) -> None:
        length = int(self.headers.get("content-length", "0"))
        self.server.recorded_body = self.rfile.read(length)
        chunks = [
            b'data: {"choices":[{"delta":{"content":"he"}}]}\n\n',
            b'data: {"choices":[{"delta":{"content":"llo"},"finish_reason":"stop"}],"usage":{"completion_tokens":1}}\n\n',
            b"data: [DONE]\n\n",
        ]
        body = b"".join(chunks)
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:
        return


class AgentProxyTests(unittest.TestCase):
    def test_reads_env_and_cli_overrides(self) -> None:
        runtime = read_proxy_config(
            ["--listen-port", "19090", "--session-id", "session-cli", "--agent-hint", "osl=1024"],
            env={
                "DYNAMO_BASE_URL": "http://dynamo.test",
                "DYNAMO_API_KEY": "dyn-key",
                "DYN_AGENT_SESSION_TYPE_ID": "agent-kind",
                "DYN_AGENT_HINTS": '{"priority":3}',
            },
        )

        self.assertEqual(runtime.listen.port, 19090)
        self.assertEqual(runtime.proxy.upstream_prefix, "/v1")
        self.assertEqual(runtime.proxy.upstream_host, "dynamo.test")
        self.assertEqual(runtime.proxy.api_key, "dyn-key")
        self.assertEqual(runtime.proxy.annotation.session_type_id, "agent-kind")
        self.assertEqual(runtime.proxy.annotation.session_id, "session-cli")
        self.assertEqual(runtime.proxy.annotation.trajectory_id, "session-cli:main")
        self.assertEqual(runtime.proxy.annotation.agent_hints, {"priority": 3, "osl": 1024})

    def test_merge_dynamo_metadata_preserves_request_values(self) -> None:
        payload = merge_dynamo_metadata(
            {
                "model": "demo",
                "nvext": {
                    "extra_fields": ["worker_id"],
                    "agent_context": {"trajectory_id": "client-traj", "custom": "kept"},
                    "agent_hints": {"priority": 1, "custom_hint": True},
                },
            },
            annotation(),
        )

        self.assertEqual(payload["nvext"]["agent_context"]["session_id"], "session-1")
        self.assertEqual(payload["nvext"]["agent_context"]["trajectory_id"], "session-1:main")
        self.assertEqual(payload["nvext"]["agent_context"]["custom"], "kept")
        self.assertEqual(payload["nvext"]["agent_hints"], {"priority": 7, "osl": 512, "custom_hint": True})

    def test_annotates_json_object_request_body(self) -> None:
        body, annotated = annotate_json_request_body(
            {"content-type": "application/json"},
            json.dumps({"model": "demo", "input": "hello"}).encode("utf-8"),
            annotation(),
        )

        self.assertTrue(annotated)
        forwarded = json.loads(body.decode("utf-8"))
        self.assertEqual(forwarded["nvext"]["agent_context"]["trajectory_id"], "session-1:main")
        self.assertEqual(forwarded["nvext"]["agent_hints"], {"priority": 7, "osl": 512})

    def test_translates_anthropic_messages_request(self) -> None:
        translated = translate_anthropic_messages_request(
            {
                "model": "claude-style-model",
                "system": "You are concise.",
                "max_tokens": 64,
                "messages": [{"role": "user", "content": [{"type": "text", "text": "hello"}]}],
                "tools": [
                    {
                        "name": "search",
                        "description": "Search docs",
                        "input_schema": {"type": "object", "properties": {"query": {"type": "string"}}},
                    }
                ],
                "tool_choice": {"type": "tool", "name": "search"},
            },
            proxy_config(),
        )

        self.assertEqual(translated["model"], "claude-style-model")
        self.assertEqual(
            translated["messages"],
            [
                {"role": "system", "content": "You are concise."},
                {"role": "user", "content": "hello"},
            ],
        )
        self.assertEqual(translated["tool_choice"], {"type": "function", "function": {"name": "search"}})
        self.assertEqual(translated["nvext"]["agent_context"]["session_id"], "session-1")

    def test_translates_openai_response_to_anthropic(self) -> None:
        translated = translate_openai_chat_response_to_anthropic(
            {
                "id": "chatcmpl-1",
                "model": "demo",
                "choices": [
                    {
                        "finish_reason": "tool_calls",
                        "message": {
                            "content": "checking",
                            "tool_calls": [
                                {
                                    "id": "call-1",
                                    "type": "function",
                                    "function": {"name": "search", "arguments": '{"query":"dynamo"}'},
                                }
                            ],
                        },
                    }
                ],
                "usage": {"prompt_tokens": 11, "completion_tokens": 7},
            },
            "fallback",
        )

        self.assertEqual(translated["stop_reason"], "tool_use")
        self.assertEqual(translated["usage"], {"input_tokens": 11, "output_tokens": 7})
        self.assertEqual(
            translated["content"],
            [
                {"type": "text", "text": "checking"},
                {"type": "tool_use", "id": "call-1", "name": "search", "input": {"query": "dynamo"}},
            ],
        )

    def test_openai_chat_and_responses_requests_are_forwarded_with_nvext(self) -> None:
        upstream = ThreadingHTTPServer(("127.0.0.1", 0), RecordingUpstreamHandler)
        upstream.recorded = []
        start_server(upstream)
        proxy = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(proxy_config(upstream.server_address[1])))
        start_server(proxy)
        conn = http.client.HTTPConnection("127.0.0.1", proxy.server_address[1], timeout=5)
        try:
            for path in ["/v1/chat/completions", "/v1/responses"]:
                conn.request(
                    "POST",
                    path,
                    body=json.dumps({"model": "demo", "input": "hello"}),
                    headers={"Content-Type": "application/json", "Authorization": "Bearer client-key"},
                )
                response = conn.getresponse()
                self.assertEqual(response.status, 200)
                response.read()
        finally:
            conn.close()
            close_server(proxy)
            close_server(upstream)

        self.assertEqual([item["path"] for item in upstream.recorded], ["/v1/chat/completions", "/v1/responses"])
        self.assertEqual([item["headers"]["Authorization"] for item in upstream.recorded], ["Bearer test-key", "Bearer test-key"])
        first_body = json.loads(upstream.recorded[0]["body"].decode("utf-8"))
        second_body = json.loads(upstream.recorded[1]["body"].decode("utf-8"))
        self.assertEqual(first_body["nvext"]["agent_context"]["trajectory_id"], "session-1:main")
        self.assertEqual(second_body["nvext"]["agent_hints"], {"priority": 7, "osl": 512})

    def test_anthropic_messages_request_forwards_chat_completion_to_dynamo(self) -> None:
        upstream = ThreadingHTTPServer(("127.0.0.1", 0), RecordingUpstreamHandler)
        upstream.recorded = []
        start_server(upstream)
        proxy = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(proxy_config(upstream.server_address[1])))
        start_server(proxy)
        conn = http.client.HTTPConnection("127.0.0.1", proxy.server_address[1], timeout=5)
        try:
            conn.request(
                "POST",
                "/v1/messages",
                body=json.dumps({"model": "demo", "max_tokens": 8, "messages": [{"role": "user", "content": "hello"}]}),
                headers={"Content-Type": "application/json", "x-api-key": "client-key"},
            )
            response = conn.getresponse()
            body = json.loads(response.read().decode("utf-8"))
            self.assertEqual(response.status, 200)
        finally:
            conn.close()
            close_server(proxy)
            close_server(upstream)

        self.assertEqual(upstream.recorded[0]["path"], "/v1/chat/completions")
        forwarded = json.loads(upstream.recorded[0]["body"].decode("utf-8"))
        self.assertEqual(forwarded["nvext"]["agent_context"]["session_type_id"], "generic_agent")
        self.assertEqual(body["type"], "message")
        self.assertEqual(body["content"], [{"type": "text", "text": "ok"}])
        self.assertEqual(body["stop_reason"], "end_turn")

    def test_anthropic_streaming_response_is_translated_to_anthropic_sse(self) -> None:
        upstream = ThreadingHTTPServer(("127.0.0.1", 0), StreamingUpstreamHandler)
        start_server(upstream)
        proxy = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(proxy_config(upstream.server_address[1])))
        start_server(proxy)
        conn = http.client.HTTPConnection("127.0.0.1", proxy.server_address[1], timeout=5)
        try:
            conn.request(
                "POST",
                "/v1/messages",
                body=json.dumps(
                    {
                        "model": "demo",
                        "max_tokens": 8,
                        "stream": True,
                        "messages": [{"role": "user", "content": "hello"}],
                    }
                ),
                headers={"Content-Type": "application/json", "x-api-key": "client-key"},
            )
            response = conn.getresponse()
            body = response.read().decode("utf-8")
            self.assertEqual(response.status, 200)
        finally:
            conn.close()
            close_server(proxy)
            close_server(upstream)

        forwarded = json.loads(upstream.recorded_body.decode("utf-8"))
        self.assertTrue(forwarded["stream"])
        self.assertEqual(forwarded["nvext"]["agent_context"]["trajectory_id"], "session-1:main")
        self.assertIn("event: message_start", body)
        self.assertIn('"type":"text_delta","text":"he"', body)
        self.assertIn('"type":"text_delta","text":"llo"', body)
        self.assertIn('"stop_reason":"end_turn"', body)
        self.assertIn("event: message_stop", body)


if __name__ == "__main__":
    unittest.main()
