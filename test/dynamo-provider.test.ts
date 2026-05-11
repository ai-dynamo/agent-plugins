// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { createAssistantMessageEventStream, type Context, type Model, type SimpleStreamOptions } from "@mariozechner/pi-ai";
import { describe, expect, it } from "vitest";
import {
	buildDynamoAgentContext,
	buildDynamoHeaders,
	createDynamoStreamSimple,
	DEFAULT_DYNAMO_BASE_URL,
	DEFAULT_DYNAMO_MODEL_ID,
	DEFAULT_SESSION_TYPE_ID,
	DYNAMO_API,
	mergeDynamoAgentContext,
	normalizeDynamoBaseUrl,
	readDynamoConfig,
} from "../src/dynamo-provider.js";

const config = {
	baseUrl: DEFAULT_DYNAMO_BASE_URL,
	apiKey: "test-key",
	sessionTypeId: DEFAULT_SESSION_TYPE_ID,
};

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

const context: Context = {
	messages: [],
};

describe("dynamo provider config", () => {
	it("normalizes bare endpoint roots to /v1", () => {
		expect(normalizeDynamoBaseUrl("http://127.0.0.1:8000")).toBe("http://127.0.0.1:8000/v1");
		expect(normalizeDynamoBaseUrl("http://127.0.0.1:8000/v1/")).toBe("http://127.0.0.1:8000/v1");
	});

	it("reads env values with Dynamo precedence", () => {
		expect(
			readDynamoConfig({
				OPENAI_BASE_URL: "http://ignored.test/v1",
				DYNAMO_BASE_URL: "http://dynamo.test",
				DYNAMO_API_KEY: "dyn-key",
				DYN_AGENT_SESSION_TYPE_ID: "session-kind",
				DYN_AGENT_SESSION_ID: "session-id",
				DYN_AGENT_TRAJECTORY_ID: "trajectory-id",
				DYN_AGENT_PARENT_TRAJECTORY_ID: "parent-id",
			}),
		).toEqual({
			baseUrl: "http://dynamo.test/v1",
			apiKey: "dyn-key",
			sessionTypeId: "session-kind",
			sessionId: "session-id",
			trajectoryId: "trajectory-id",
			parentTrajectoryId: "parent-id",
		});
	});
});

describe("pi-subagents trajectory bridge", () => {
	it("rewrites trajectory id and parent id when running as a pi-subagents child", () => {
		const config = readDynamoConfig({
			DYNAMO_BASE_URL: "http://dynamo.test",
			DYN_AGENT_SESSION_ID: "run-42",
			DYN_AGENT_TRAJECTORY_ID: "run-42:orchestrator",
			PI_SUBAGENT_CHILD: "1",
			PI_SUBAGENT_RUN_ID: "run-42",
			PI_SUBAGENT_CHILD_AGENT: "researcher",
			PI_SUBAGENT_CHILD_INDEX: "2",
		});
		expect(config.trajectoryId).toBe("run-42:researcher:2");
		expect(config.parentTrajectoryId).toBe("run-42:orchestrator");
	});

	it("defaults child index to 0 when pi-subagents does not set it", () => {
		const config = readDynamoConfig({
			DYN_AGENT_TRAJECTORY_ID: "parent",
			PI_SUBAGENT_CHILD: "1",
			PI_SUBAGENT_RUN_ID: "r",
			PI_SUBAGENT_CHILD_AGENT: "child",
		});
		expect(config.trajectoryId).toBe("r:child:0");
		expect(config.parentTrajectoryId).toBe("parent");
	});

	it("respects an explicit DYN_AGENT_PARENT_TRAJECTORY_ID over the bridge", () => {
		const config = readDynamoConfig({
			DYN_AGENT_TRAJECTORY_ID: "explicit-traj",
			DYN_AGENT_PARENT_TRAJECTORY_ID: "explicit-parent",
			PI_SUBAGENT_CHILD: "1",
			PI_SUBAGENT_RUN_ID: "run",
			PI_SUBAGENT_CHILD_AGENT: "agent",
		});
		expect(config.trajectoryId).toBe("explicit-traj");
		expect(config.parentTrajectoryId).toBe("explicit-parent");
	});

	it("does nothing when PI_SUBAGENT_CHILD is unset", () => {
		const config = readDynamoConfig({
			DYN_AGENT_TRAJECTORY_ID: "top-level",
			PI_SUBAGENT_RUN_ID: "run",
			PI_SUBAGENT_CHILD_AGENT: "agent",
		});
		expect(config.trajectoryId).toBe("top-level");
		expect(config.parentTrajectoryId).toBeUndefined();
	});

	it("does nothing when the inherited DYN_AGENT_TRAJECTORY_ID is missing", () => {
		const config = readDynamoConfig({
			PI_SUBAGENT_CHILD: "1",
			PI_SUBAGENT_RUN_ID: "run",
			PI_SUBAGENT_CHILD_AGENT: "agent",
		});
		expect(config.trajectoryId).toBeUndefined();
		expect(config.parentTrajectoryId).toBeUndefined();
	});

	it("does nothing when pi-subagents bookkeeping vars are partial", () => {
		const config = readDynamoConfig({
			DYN_AGENT_TRAJECTORY_ID: "top-level",
			PI_SUBAGENT_CHILD: "1",
			PI_SUBAGENT_RUN_ID: "run",
			// PI_SUBAGENT_CHILD_AGENT missing — bridge should not fire.
		});
		expect(config.trajectoryId).toBe("top-level");
		expect(config.parentTrajectoryId).toBeUndefined();
	});
});

describe("agent context injection", () => {
	it("uses the Pi session ID as the default trajectory ID", () => {
		expect(buildDynamoAgentContext(config, { sessionId: "pi-session" })).toEqual({
			trajectory_id: "pi-session",
			session_type_id: DEFAULT_SESSION_TYPE_ID,
			phase: "reasoning",
		});
	});

	it("lets DYN_AGENT_TRAJECTORY_ID override the session ID default", () => {
		expect(buildDynamoAgentContext({ ...config, trajectoryId: "trajectory-from-env" }, { sessionId: "pi-session" })).toEqual({
			trajectory_id: "trajectory-from-env",
			session_type_id: DEFAULT_SESSION_TYPE_ID,
			phase: "reasoning",
		});
	});

	it("merges nvext.agent_context without dropping existing nvext fields", () => {
		const payload = mergeDynamoAgentContext(
			{
				model: "demo",
				nvext: {
					extra_fields: ["worker_id", "timing"],
					agent_context: {
						session_id: "existing-session",
						custom_field: "kept",
					},
				},
			},
			{
				trajectory_id: "trajectory",
				session_id: "default-session",
				session_type_id: DEFAULT_SESSION_TYPE_ID,
				phase: "reasoning",
			},
		);

		expect(payload).toEqual({
			model: "demo",
			nvext: {
				extra_fields: ["worker_id", "timing"],
				agent_context: {
					trajectory_id: "trajectory",
					session_id: "existing-session",
					session_type_id: DEFAULT_SESSION_TYPE_ID,
					phase: "reasoning",
					custom_field: "kept",
				},
			},
		});
	});
});

describe("request headers", () => {
	it("sets x-request-id when absent", () => {
		expect(buildDynamoHeaders(undefined, () => "request-1")).toEqual({ "x-request-id": "request-1" });
	});

	it("preserves an existing x-request-id header regardless of casing", () => {
		expect(buildDynamoHeaders({ "X-Request-Id": "provided" }, () => "request-1")).toEqual({
			"X-Request-Id": "provided",
		});
	});
});

describe("streamSimple wrapper", () => {
	it("delegates through openai-completions with injected payload and headers", async () => {
		let capturedModel: Model<"openai-completions"> | undefined;
		let capturedOptions: SimpleStreamOptions | undefined;

		const streamSimple = createDynamoStreamSimple(
			config,
			(openAIModel, _context, options) => {
				capturedModel = openAIModel;
				capturedOptions = options;
				return createAssistantMessageEventStream();
			},
			() => "request-1",
		);

		streamSimple(model, context, {
			sessionId: "pi-session",
			onPayload: (payload) => payload,
		});

		const onPayload = capturedOptions?.onPayload;
		if (!onPayload) {
			throw new Error("expected wrapped onPayload");
		}
		const injectedPayload = await onPayload({ model: "default" }, model);

		expect(capturedModel?.api).toBe("openai-completions");
		expect(capturedModel?.provider).toBe("dynamo");
		expect(capturedOptions?.apiKey).toBe("test-key");
		expect(capturedOptions?.headers).toEqual({ "x-request-id": "request-1" });
		expect(injectedPayload).toEqual({
			model: "default",
			nvext: {
				agent_context: {
					trajectory_id: "pi-session",
					session_type_id: DEFAULT_SESSION_TYPE_ID,
					phase: "reasoning",
				},
			},
		});
	});
});
