// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { createAssistantMessageEventStream, type Context, type Model, type SimpleStreamOptions } from "@mariozechner/pi-ai";
import { describe, expect, it } from "vitest";
import {
	applySubagentBridge,
	buildDynamoAgentContext,
	buildDynamoHeaders,
	computeSubagentTrajectoryRewrite,
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
	const childEnv = {
		DYN_AGENT_TRAJECTORY_ID: "parent-traj",
		PI_SUBAGENT_CHILD: "1",
		PI_SUBAGENT_RUN_ID: "run-1",
		PI_SUBAGENT_CHILD_AGENT: "researcher",
		PI_SUBAGENT_CHILD_INDEX: "2",
	} as const;

	it("reinterprets inherited DYN_AGENT_TRAJECTORY_ID as parent when in a subagent child", () => {
		expect(computeSubagentTrajectoryRewrite(childEnv)).toEqual({
			parentTrajectoryId: "parent-traj",
			trajectoryId: "run-1:researcher:2",
		});
	});

	it("defaults PI_SUBAGENT_CHILD_INDEX to 0 when absent", () => {
		const { PI_SUBAGENT_CHILD_INDEX: _omit, ...envWithoutIndex } = childEnv;
		expect(computeSubagentTrajectoryRewrite(envWithoutIndex)).toEqual({
			parentTrajectoryId: "parent-traj",
			trajectoryId: "run-1:researcher:0",
		});
	});

	it("skips the bridge when PI_SUBAGENT_CHILD is not 1", () => {
		expect(computeSubagentTrajectoryRewrite({ ...childEnv, PI_SUBAGENT_CHILD: undefined })).toBeNull();
	});

	it("does NOT override an explicit DYN_AGENT_PARENT_TRAJECTORY_ID (manual wins)", () => {
		expect(
			computeSubagentTrajectoryRewrite({ ...childEnv, DYN_AGENT_PARENT_TRAJECTORY_ID: "manual-parent" }),
		).toBeNull();
	});

	it("skips when inherited DYN_AGENT_TRAJECTORY_ID is absent", () => {
		expect(computeSubagentTrajectoryRewrite({ ...childEnv, DYN_AGENT_TRAJECTORY_ID: undefined })).toBeNull();
	});

	it("skips when PI_SUBAGENT_RUN_ID or PI_SUBAGENT_CHILD_AGENT is missing", () => {
		expect(computeSubagentTrajectoryRewrite({ ...childEnv, PI_SUBAGENT_RUN_ID: undefined })).toBeNull();
		expect(computeSubagentTrajectoryRewrite({ ...childEnv, PI_SUBAGENT_CHILD_AGENT: undefined })).toBeNull();
	});

	it("readDynamoConfig surfaces the synthesized ids", () => {
		const cfg = readDynamoConfig(childEnv);
		expect(cfg.trajectoryId).toBe("run-1:researcher:2");
		expect(cfg.parentTrajectoryId).toBe("parent-traj");
	});

	it("applySubagentBridge mutates process.env so nested spawns chain correctly", () => {
		const env: NodeJS.ProcessEnv = { ...childEnv };
		expect(applySubagentBridge(env)).toBe(true);
		expect(env.DYN_AGENT_TRAJECTORY_ID).toBe("run-1:researcher:2");
		expect(env.DYN_AGENT_PARENT_TRAJECTORY_ID).toBe("parent-traj");

		// Idempotent: a second call sees the now-set parent and short-circuits.
		expect(applySubagentBridge(env)).toBe(false);
		expect(env.DYN_AGENT_TRAJECTORY_ID).toBe("run-1:researcher:2");

		// Chaining: when this grandchild spawns its own subagent, pi-subagents
		// passes { ...process.env, ...subagentEnv }. The grandchild then sees
		// its own synthesized id as inherited DYN_AGENT_TRAJECTORY_ID, so the
		// next rewrite treats THIS generation as the parent.
		const grandchildEnv = {
			...env,
			DYN_AGENT_PARENT_TRAJECTORY_ID: undefined,
			PI_SUBAGENT_CHILD_AGENT: "subworker",
			PI_SUBAGENT_CHILD_INDEX: "0",
		};
		expect(computeSubagentTrajectoryRewrite(grandchildEnv)).toEqual({
			parentTrajectoryId: "run-1:researcher:2",
			trajectoryId: "run-1:subworker:0",
		});
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
