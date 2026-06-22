// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { decode } from "@msgpack/msgpack";
import { createAssistantMessageEventStream, type Context, type Model, type SimpleStreamOptions } from "@mariozechner/pi-ai";
import type { ExtensionContext } from "@mariozechner/pi-coding-agent";
import { describe, expect, it } from "vitest";
import {
	applySubagentTrajectoryBridge,
	buildToolAgentContext,
	createDynamoStreamSimple,
	DEFAULT_DYNAMO_BASE_URL,
	DEFAULT_DYNAMO_MODEL_ID,
	DynamoToolEventPublisher,
	DynamoToolEventRelay,
	DYNAMO_API,
	readDynamoConfig,
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
	it("keeps Pi sessionId and adds Dynamo trajectory headers", () => {
		let capturedOptions: SimpleStreamOptions | undefined;
		const streamSimple = createDynamoStreamSimple(
			config,
			(_model, _context, options) => {
				capturedOptions = options;
				return createAssistantMessageEventStream();
			},
			() => "request-1",
		);

		streamSimple(model, context, { sessionId: "pi-session" });

		expect(capturedOptions?.sessionId).toBe("pi-session");
		expect(capturedOptions?.headers).toEqual({
			"x-request-id": "request-1",
			"x-dynamo-trajectory-id": "pi-session",
		});
	});

	it("bridges pi-subagents through Dynamo trajectory headers", () => {
		const env: NodeJS.ProcessEnv = {
			DYN_REQUEST_TRACE: "1",
			DYN_AGENT_TRAJECTORY_ID: "parent",
			PI_SUBAGENT_CHILD: "1",
			PI_SUBAGENT_RUN_ID: "run",
			PI_SUBAGENT_CHILD_AGENT: "researcher",
		};
		expect(applySubagentTrajectoryBridge(env)).toBe(true);
		const cfg = readDynamoConfig(env);

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
			"x-dynamo-trajectory-id": "run:researcher:0",
			"x-dynamo-parent-trajectory-id": "parent",
		});
	});

	it("emits trajectory-only ZMQ tool context", async () => {
		const socket = new FakeToolEventSocket();
		const publisher = new DynamoToolEventPublisher(
			{ endpoint: "tcp://127.0.0.1:20390", topic: "tools", queueCapacity: 10 },
			() => socket,
		);
		await publisher.start();
		const relay = new DynamoToolEventRelay(config, publisher, () => 1000, () => 10);

		relay.handleToolExecutionStart({ toolCallId: "call-1", toolName: "bash", args: {} }, createContext("pi-session"));
		await publisher.flush();

		const record = decode(socket.sent[0]?.[2] ?? Buffer.alloc(0)) as DynamoRequestTraceRecord;
		expect(buildToolAgentContext(config, "pi-session")).toEqual({ trajectory_id: "pi-session" });
		expect(record.agent_context).toEqual({ trajectory_id: "pi-session" });
	});
});
