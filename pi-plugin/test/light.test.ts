// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { decode } from "@msgpack/msgpack";
import { createAssistantMessageEventStream, type Context, type Model, type SimpleStreamOptions } from "@mariozechner/pi-ai";
import type { ExtensionContext } from "@mariozechner/pi-coding-agent";
import { describe, expect, it } from "vitest";
import {
	applySubagentSessionBridge,
	buildToolAgentContext,
	createDynamoModels,
	createDynamoStreamSimple,
	DEFAULT_DYNAMO_BASE_URL,
	DEFAULT_DYNAMO_CONTEXT_WINDOW,
	DEFAULT_DYNAMO_MAX_TOKENS,
	DEFAULT_DYNAMO_MODEL_ID,
	describeDynamoModelLimits,
	discoverDynamoModels,
	DynamoToolEventPublisher,
	DynamoToolEventRelay,
	DYNAMO_API,
	readDynamoConfig,
	resolveDynamoModelLimits,
	sendDynamoSessionFinal,
	type DynamoConfig,
	type DynamoModelLimits,
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

	it("sends a best-effort terminal session header when request tracing is disabled", async () => {
		let url: string | URL | Request | undefined;
		let init: RequestInit | undefined;
		const sent = await sendDynamoSessionFinal(
			{ ...config, traceEnabled: false },
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

	it("skips the terminal request when session finalization is disabled", async () => {
		let calls = 0;
		const sent = await sendDynamoSessionFinal(
			{ ...config, sessionFinalEnabled: false },
			"test-model",
			"pi-session",
			undefined,
			async () => {
				calls += 1;
				return { ok: true } as Response;
			},
		);

		expect(sent).toBe(false);
		expect(calls).toBe(0);
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

describe("model limits", () => {
	const noEnv = {};

	function modelsResponse(data: unknown): typeof fetch {
		return async () => ({ ok: true, json: async () => ({ data }) }) as Response;
	}

	it("falls back to the defaults when nothing is configured", () => {
		const [created] = createDynamoModels(["served"], DEFAULT_DYNAMO_BASE_URL, { env: noEnv });

		expect(created).toMatchObject({
			id: "served",
			contextWindow: DEFAULT_DYNAMO_CONTEXT_WINDOW,
			maxTokens: DEFAULT_DYNAMO_MAX_TOKENS,
		});
	});

	it("uses the advertised context window before the default", () => {
		const servedContextWindows = new Map([["served", 40960]]);
		const [created] = createDynamoModels(["served"], DEFAULT_DYNAMO_BASE_URL, { servedContextWindows, env: noEnv });

		expect(created?.contextWindow).toBe(40960);
		expect(created?.maxTokens).toBe(DEFAULT_DYNAMO_MAX_TOKENS);
	});

	it("uses the explicit overrides before the advertised values", () => {
		const servedContextWindows = new Map([["served", 40960]]);
		const env = { DYNAMO_CONTEXT_WINDOW: " 65536 ", DYNAMO_MAX_TOKENS: "4096" };
		const [created] = createDynamoModels(["served"], DEFAULT_DYNAMO_BASE_URL, { servedContextWindows, env });

		expect(created?.contextWindow).toBe(65536);
		expect(created?.maxTokens).toBe(4096);
	});

	it("treats a blank override as unset", () => {
		const env = { DYNAMO_CONTEXT_WINDOW: "", DYNAMO_MAX_TOKENS: "   " };
		const [created] = createDynamoModels([], DEFAULT_DYNAMO_BASE_URL, { env });

		expect(created).toMatchObject({
			id: DEFAULT_DYNAMO_MODEL_ID,
			contextWindow: DEFAULT_DYNAMO_CONTEXT_WINDOW,
			maxTokens: DEFAULT_DYNAMO_MAX_TOKENS,
		});
	});

	it("rejects overrides that are not positive integers", () => {
		for (const value of ["lots", "0", "-1", "1.5"]) {
			expect(() => createDynamoModels([], DEFAULT_DYNAMO_BASE_URL, { env: { DYNAMO_MAX_TOKENS: value } })).toThrow(
				/DYNAMO_MAX_TOKENS must be a positive integer/,
			);
		}
		expect(() => createDynamoModels([], DEFAULT_DYNAMO_BASE_URL, { env: { DYNAMO_CONTEXT_WINDOW: "wide" } })).toThrow(
			/DYNAMO_CONTEXT_WINDOW must be a positive integer/,
		);
	});

	it("rejects an output cap at or above the context window", () => {
		const servedContextWindows = new Map([["served", 8192]]);

		expect(() => createDynamoModels(["served"], DEFAULT_DYNAMO_BASE_URL, { servedContextWindows, env: noEnv })).toThrow(
			/max tokens 8192 must be below the context window 8192/,
		);
		expect(() =>
			createDynamoModels(["served"], DEFAULT_DYNAMO_BASE_URL, { env: { DYNAMO_MAX_TOKENS: "200000" } }),
		).toThrow(/max tokens 200000 must be below the context window 128000/);
	});

	it("reports where each limit came from", () => {
		const servedContextWindows = new Map([["served", 40960]]);

		expect(resolveDynamoModelLimits("served", { servedContextWindows, env: noEnv })).toEqual({
			contextWindow: 40960,
			contextWindowSource: "advertised",
			maxTokens: DEFAULT_DYNAMO_MAX_TOKENS,
			maxTokensSource: "default",
		});
		expect(
			resolveDynamoModelLimits("served", {
				servedContextWindows,
				env: { DYNAMO_CONTEXT_WINDOW: "32768", DYNAMO_MAX_TOKENS: "4096" },
			}),
		).toEqual({
			contextWindow: 32768,
			contextWindowSource: "DYNAMO_CONTEXT_WINDOW",
			maxTokens: 4096,
			maxTokensSource: "DYNAMO_MAX_TOKENS",
		});
		expect(resolveDynamoModelLimits("other", { servedContextWindows, env: noEnv })).toMatchObject({
			contextWindow: DEFAULT_DYNAMO_CONTEXT_WINDOW,
			contextWindowSource: "default",
		});
	});

	it("describes resolved limits in one line per model", () => {
		const limits: DynamoModelLimits = {
			contextWindow: 40960,
			contextWindowSource: "advertised",
			maxTokens: 8192,
			maxTokensSource: "default",
		};

		expect(describeDynamoModelLimits("qwen", limits)).toBe(
			"pi-dynamo-provider: qwen contextWindow 40960 (advertised), maxTokens 8192 (default)",
		);
	});

	it("reads advertised context windows from the models list", async () => {
		const fetchImpl = modelsResponse([
			{ id: "qwen", context_window: 40960 },
			{ id: "plain" },
			{ id: "qwen", context_window: 40960 },
			{ id: "", context_window: 1 },
			{ id: "odd", context_window: "40960" },
		]);

		const models = await discoverDynamoModels(config, 1000, { fetchImpl, env: noEnv });

		expect(models.map((model) => [model.id, model.contextWindow])).toEqual([
			["qwen", 40960],
			["plain", DEFAULT_DYNAMO_CONTEXT_WINDOW],
			["odd", DEFAULT_DYNAMO_CONTEXT_WINDOW],
		]);
		expect(models.every((model) => model.maxTokens === DEFAULT_DYNAMO_MAX_TOKENS)).toBe(true);
	});

	it("applies the explicit override to every discovered model and reports it", async () => {
		const fetchImpl = modelsResponse([{ id: "qwen", context_window: 40960 }, { id: "plain" }]);
		const reported: string[] = [];

		const models = await discoverDynamoModels(config, 1000, {
			fetchImpl,
			env: { DYNAMO_CONTEXT_WINDOW: "32768" },
			onLimits: (modelId, limits) => reported.push(describeDynamoModelLimits(modelId, limits)),
		});

		expect(models.map((model) => model.contextWindow)).toEqual([32768, 32768]);
		expect(reported).toEqual([
			"pi-dynamo-provider: qwen contextWindow 32768 (DYNAMO_CONTEXT_WINDOW), maxTokens 8192 (default)",
			"pi-dynamo-provider: plain contextWindow 32768 (DYNAMO_CONTEXT_WINDOW), maxTokens 8192 (default)",
		]);
	});

	it("returns no models when discovery fails but still surfaces bad overrides", async () => {
		const failing: typeof fetch = async () => {
			throw new Error("connection refused");
		};
		const rejected: typeof fetch = async () => ({ ok: false }) as Response;

		expect(await discoverDynamoModels(config, 1000, { fetchImpl: failing, env: noEnv })).toEqual([]);
		expect(await discoverDynamoModels(config, 1000, { fetchImpl: rejected, env: noEnv })).toEqual([]);
		await expect(
			discoverDynamoModels(config, 1000, {
				fetchImpl: modelsResponse([{ id: "qwen" }]),
				env: { DYNAMO_MAX_TOKENS: "many" },
			}),
		).rejects.toThrow(/DYNAMO_MAX_TOKENS/);
	});
});
