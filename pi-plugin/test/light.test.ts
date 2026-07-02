// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { decode } from "@msgpack/msgpack";
import { createAssistantMessageEventStream, type Context, type Model, type SimpleStreamOptions } from "@mariozechner/pi-ai";
import type { ExtensionContext } from "@mariozechner/pi-coding-agent";
import { describe, expect, it } from "vitest";
import {
	applySubagentSessionBridge,
	buildToolAgentContext,
	createDynamoStreamSimple,
	DEFAULT_DYNAMO_BASE_URL,
	DEFAULT_DYNAMO_MODEL_ID,
	DynamoToolEventPublisher,
	DynamoToolEventRelay,
	DYNAMO_API,
	readDynamoConfig,
	sendDynamoSessionFinal,
	type DynamoConfig,
	type DynamoRequestTraceRecord,
	type ToolEventSocket,
} from "../src/index.js";

const model = {
	id: DEFAULT_DYNAMO_MODEL_ID,
	name: "Default",
	api: DYNAMO_API,
	provider: "dynamo",
	baseUrl: DEFAULT_DYNAMO_BASE_URL,
	reasoning: false,
	input: ["text"],
	cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
	contextWindow: 128000,
	maxTokens: 8192,
} satisfies Model<typeof DYNAMO_API>;

const context: Context = { messages: [] };
const config: DynamoConfig = {
	baseUrl: DEFAULT_DYNAMO_BASE_URL,
	apiKey: "test-key",
	traceEnabled: true,
};

class FakeToolEventSocket implements ToolEventSocket {
	readonly sent: [Buffer, Buffer, Buffer][] = [];
	async connect(_endpoint: string): Promise<void> {}
	async send(frames: [Buffer, Buffer, Buffer]): Promise<void> {
		this.sent.push(frames);
	}
	close(): void {}
}

function createContext(sessionId: string): ExtensionContext {
	return { sessionManager: { getSessionId: () => sessionId } } as unknown as ExtensionContext;
}

describe("light provider", () => {
	it("keeps Pi sessionId and adds Dynamo session headers", () => {
		let capturedModel: Model<"openai-completions"> | undefined;
		let capturedOptions: SimpleStreamOptions | undefined;
		const streamSimple = createDynamoStreamSimple(
			config,
			(model, _context, options) => {
				capturedModel = model;
				capturedOptions = options;
				return createAssistantMessageEventStream();
			},
			() => "request-1",
		);

		streamSimple(model, context, { sessionId: "pi-session" });

		expect(capturedOptions?.sessionId).toBe("pi-session");
		expect(capturedModel?.compat?.sendSessionAffinityHeaders).toBeUndefined();
		expect(capturedOptions?.headers).toEqual({
			"x-request-id": "request-1",
			"x-dynamo-session-id": "pi-session",
		});
	});

	it("adds session headers when Dynamo request tracing is disabled", () => {
		let capturedOptions: SimpleStreamOptions | undefined;
		createDynamoStreamSimple(
			{ ...config, traceEnabled: false },
			(_model, _context, options) => {
				capturedOptions = options;
				return createAssistantMessageEventStream();
			},
			() => "request-1",
		)(model, context, { sessionId: "pi-session" });

		expect(capturedOptions?.headers).toEqual({
			"x-request-id": "request-1",
			"x-dynamo-session-id": "pi-session",
		});
	});

	it("sends a best-effort terminal session header", async () => {
		let url: string | URL | Request | undefined;
		let init: RequestInit | undefined;
		const sent = await sendDynamoSessionFinal(
			config,
			"test-model",
			"pi-session",
			() => "request-final",
			async (input, options) => {
				url = input;
				init = options;
				return { ok: true } as Response;
			},
		);

		expect(sent).toBe(true);
		expect(url?.toString()).toBe(`${DEFAULT_DYNAMO_BASE_URL}/chat/completions`);
		expect(init?.headers).toMatchObject({
			"x-request-id": "request-final",
			"x-dynamo-session-id": "pi-session",
			"x-dynamo-session-final": "true",
		});
		expect(JSON.parse(init?.body as string)).toMatchObject({ model: "test-model", max_tokens: 1, stream: false });
	});

	it("bridges pi-subagents through Dynamo session headers", () => {
		const env: NodeJS.ProcessEnv = {
			DYN_REQUEST_TRACE: "0",
			DYN_AGENT_SESSION_ID: "parent",
			PI_SUBAGENT_CHILD: "1",
			PI_SUBAGENT_RUN_ID: "run",
			PI_SUBAGENT_CHILD_AGENT: "researcher",
		};
		expect(applySubagentSessionBridge(env)).toBe(true);
		const cfg = readDynamoConfig(env);
		expect(cfg.traceEnabled).toBe(false);

		let capturedOptions: SimpleStreamOptions | undefined;
		createDynamoStreamSimple(
			cfg,
			(_model, _context, options) => {
				capturedOptions = options;
				return createAssistantMessageEventStream();
			},
			() => "request-1",
		)(model, context, { sessionId: "pi-session" });

		expect(capturedOptions?.sessionId).toBe("pi-session");
		expect(capturedOptions?.headers).toMatchObject({
			"x-dynamo-session-id": "run:researcher:0",
			"x-dynamo-parent-session-id": "parent",
		});
	});

	it("does not self-parent child sessions", () => {
		const env: NodeJS.ProcessEnv = {
			DYN_REQUEST_TRACE: "1",
			DYN_AGENT_SESSION_ID: "run:researcher:0",
			DYN_AGENT_PARENT_SESSION_ID: "run:researcher:0",
			PI_SUBAGENT_CHILD: "1",
			PI_SUBAGENT_RUN_ID: "run",
			PI_SUBAGENT_CHILD_AGENT: "researcher",
		};

		expect(applySubagentSessionBridge(env)).toBe(true);
		const cfg = readDynamoConfig(env);

		expect(cfg.sessionId).toBe("run:researcher:0");
		expect(cfg.parentSessionId).toBeUndefined();
		expect(env.DYN_AGENT_PARENT_SESSION_ID).toBeUndefined();
	});

	it("emits top-level session fields for ZMQ tool events", async () => {
		const socket = new FakeToolEventSocket();
		const publisher = new DynamoToolEventPublisher(
			{ endpoint: "tcp://127.0.0.1:20390", topic: "tools", queueCapacity: 10 },
			() => socket,
		);
		await publisher.start();
		const tracedConfig = { ...config, sessionId: "child", parentSessionId: "parent" };
		const relay = new DynamoToolEventRelay(tracedConfig, publisher, () => 1000, () => 10);

		relay.handleToolExecutionStart({ toolCallId: "call-1", toolName: "bash", args: {} }, createContext("pi-session"));
		await publisher.flush();

		const record = decode(socket.sent[0]?.[2] ?? Buffer.alloc(0)) as DynamoRequestTraceRecord;
		expect(buildToolAgentContext(tracedConfig, "pi-session")).toEqual({
			session_id: "child",
			parent_session_id: "parent",
		});
		expect(record.session_id).toBe("child");
		expect(record.parent_session_id).toBe("parent");
		expect(record).not.toHaveProperty("agent_context");
	});
});
