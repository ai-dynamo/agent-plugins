// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { decode } from "@msgpack/msgpack";
import type { ExtensionContext } from "@mariozechner/pi-coding-agent";
import { describe, expect, it } from "vitest";
import { DEFAULT_DYNAMO_BASE_URL, DEFAULT_SESSION_TYPE_ID } from "../src/dynamo-provider.js";
import {
	buildDynamoRequestTraceAgentContext,
	DEFAULT_TOOL_EVENT_QUEUE_CAPACITY,
	DynamoToolEventPublisher,
	DynamoToolEventRelay,
	getToolClass,
	readDynamoToolRelayConfig,
	type DynamoRequestTraceRecord,
	type ToolEventSocket,
} from "../src/tool-relay.js";

const config = {
	baseUrl: DEFAULT_DYNAMO_BASE_URL,
	apiKey: "test-key",
	traceEnabled: true,
	sessionTypeId: DEFAULT_SESSION_TYPE_ID,
};

class FakeToolEventSocket implements ToolEventSocket {
	connectedEndpoint: string | undefined;
	closed = false;
	readonly sent: [Buffer, Buffer, Buffer][] = [];

	async connect(endpoint: string): Promise<void> {
		this.connectedEndpoint = endpoint;
	}

	async send(frames: [Buffer, Buffer, Buffer]): Promise<void> {
		this.sent.push(frames);
	}

	close(): void {
		this.closed = true;
	}
}

function createContext(sessionId: string): ExtensionContext {
	return {
		sessionManager: {
			getSessionId: () => sessionId,
		},
	} as unknown as ExtensionContext;
}

function decodeTraceRecord(frame: Buffer): DynamoRequestTraceRecord {
	return decode(frame) as DynamoRequestTraceRecord;
}

describe("tool relay config", () => {
	it("reads Dynamo tool relay env aliases", () => {
		expect(
			readDynamoToolRelayConfig({
				DYN_REQUEST_TRACE_TOOL_EVENTS_ZMQ_ENDPOINT: "tcp://127.0.0.1:20390",
				DYN_REQUEST_TRACE_TOOL_EVENTS_ZMQ_TOPIC: "tools",
			}),
		).toEqual({
			endpoint: "tcp://127.0.0.1:20390",
			topic: "tools",
			queueCapacity: DEFAULT_TOOL_EVENT_QUEUE_CAPACITY,
		});

		expect(
			readDynamoToolRelayConfig({
				DYN_REQUEST_TRACE_TOOL_EVENTS_ZMQ_ENDPOINT: "ipc:///tmp/pi-tools",
				DYN_REQUEST_TRACE_TOOL_EVENTS_ZMQ_TOPIC: "pi-tools",
				DYN_REQUEST_TRACE_TOOL_EVENTS_QUEUE_CAPACITY: "7",
			}),
		).toEqual({
			endpoint: "ipc:///tmp/pi-tools",
			topic: "pi-tools",
			queueCapacity: 7,
		});
	});
});

describe("tool relay agent context", () => {
	it("uses the Pi session ID as default trajectory ID", () => {
		expect(buildDynamoRequestTraceAgentContext(config, "pi-session")).toEqual({
			session_type_id: DEFAULT_SESSION_TYPE_ID,
			trajectory_id: "pi-session",
		});
	});

	it("uses configured trajectory IDs when provided", () => {
		expect(
			buildDynamoRequestTraceAgentContext(
				{
					...config,
					trajectoryId: "trajectory-1",
					parentTrajectoryId: "parent-1",
				},
				"pi-session",
			),
		).toEqual({
			session_type_id: DEFAULT_SESSION_TYPE_ID,
			trajectory_id: "trajectory-1",
			parent_trajectory_id: "parent-1",
		});
	});
});

describe("tool relay records", () => {
	it("publishes msgpack-framed tool_start and tool_end records", async () => {
		const socket = new FakeToolEventSocket();
		const publisher = new DynamoToolEventPublisher(
			{ endpoint: "tcp://127.0.0.1:20390", topic: "tools", queueCapacity: 10 },
			() => socket,
		);
		await publisher.start();

		let unixMs = 1000;
		let perfMs = 10;
		const relay = new DynamoToolEventRelay(
			config,
			publisher,
			() => unixMs,
			() => perfMs,
		);

		relay.handleToolExecutionStart(
			{ toolCallId: "call-1", toolName: "agent_tools---search", args: { query: "hello" } },
			createContext("pi-session"),
		);
		await publisher.flush();

		expect(socket.connectedEndpoint).toBe("tcp://127.0.0.1:20390");
		expect(socket.sent).toHaveLength(1);
		expect(socket.sent[0]?.[0].toString("utf8")).toBe("tools");
		expect(socket.sent[0]?.[1].readBigUInt64BE()).toBe(0n);
		expect(decodeTraceRecord(socket.sent[0]?.[2] ?? Buffer.alloc(0))).toEqual({
			schema: "dynamo.request.trace.v1",
			event_type: "tool_start",
			event_time_unix_ms: 1000,
			event_source: "harness",
			agent_context: {
				session_type_id: DEFAULT_SESSION_TYPE_ID,
				trajectory_id: "pi-session",
			},
			tool: {
				tool_call_id: "call-1",
				tool_class: "agent_tools",
				started_at_unix_ms: 1000,
				status: "running",
			},
		});

		unixMs = 1500;
		perfMs = 15.25;
		relay.handleToolExecutionEnd(
			{
				toolCallId: "call-1",
				toolName: "agent_tools---search",
				result: { content: [{ type: "text", text: "done" }] },
				isError: false,
			},
			createContext("pi-session"),
		);
		await publisher.flush();

		expect(socket.sent).toHaveLength(2);
		expect(socket.sent[1]?.[1].readBigUInt64BE()).toBe(1n);
		expect(decodeTraceRecord(socket.sent[1]?.[2] ?? Buffer.alloc(0))).toEqual({
			schema: "dynamo.request.trace.v1",
			event_type: "tool_end",
			event_time_unix_ms: 1500,
			event_source: "harness",
			agent_context: {
				session_type_id: DEFAULT_SESSION_TYPE_ID,
				trajectory_id: "pi-session",
			},
			tool: {
				tool_call_id: "call-1",
				tool_class: "agent_tools",
				started_at_unix_ms: 1000,
				ended_at_unix_ms: 1500,
				duration_ms: 5.25,
				status: "succeeded",
				output_bytes: 4,
			},
		});
	});

	it("publishes self-contained terminal errors even without a start event", async () => {
		const socket = new FakeToolEventSocket();
		const publisher = new DynamoToolEventPublisher(
			{ endpoint: "tcp://127.0.0.1:20390", topic: "tools", queueCapacity: 10 },
			() => socket,
		);
		await publisher.start();

		const relay = new DynamoToolEventRelay(config, publisher, () => 2000, () => 20);
		relay.handleToolExecutionEnd(
			{
				toolCallId: "call-2",
				toolName: "bash",
				result: { content: [{ type: "text", text: "failed" }] },
				isError: true,
			},
			createContext("pi-session"),
		);
		await publisher.flush();

		expect(decodeTraceRecord(socket.sent[0]?.[2] ?? Buffer.alloc(0))).toEqual({
			schema: "dynamo.request.trace.v1",
			event_type: "tool_error",
			event_time_unix_ms: 2000,
			event_source: "harness",
			agent_context: {
				session_type_id: DEFAULT_SESSION_TYPE_ID,
				trajectory_id: "pi-session",
			},
			tool: {
				tool_call_id: "call-2",
				tool_class: "bash",
				started_at_unix_ms: 2000,
				ended_at_unix_ms: 2000,
				duration_ms: 0,
				status: "error",
				error_type: "pi_tool_error",
				output_bytes: 6,
			},
		});
	});

	it("normalizes Pi and MCP-style tool names to tool classes", () => {
		expect(getToolClass("agent_tools---search")).toBe("agent_tools");
		expect(getToolClass("mcp/server.tool")).toBe("mcp");
		expect(getToolClass("bash")).toBe("bash");
	});
});
