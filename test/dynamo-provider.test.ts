// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { createAssistantMessageEventStream, type Context, type Model, type SimpleStreamOptions } from "@mariozechner/pi-ai";
import { describe, expect, it } from "vitest";
import {
	applySubagentBridge,
	buildDynamoAgentContext,
	buildDynamoHeaders,
	computeSubagentTrajectoryId,
	computeSubagentTrajectoryRewrite,
	createDynamoStreamSimple,
	DEFAULT_DYNAMO_BASE_URL,
	DEFAULT_DYNAMO_MODEL_ID,
	DEFAULT_SESSION_TYPE_ID,
	DYNAMO_API,
	mergeDynamoAgentContext,
	normalizeDynamoBaseUrl,
	readDynamoConfig,
	seedRootTrajectory,
	sendTrajectoryFinal,
} from "../src/dynamo-provider.js";

// Spread `base` with the given keys dropped (env-absent). Avoids the
// exactOptionalPropertyTypes friction of `{ ...base, KEY: undefined }`, which TS
// rejects because an explicit `undefined` is not assignable to an optional
// `string` property.
function envWithout<T extends Record<string, unknown>>(base: T, ...keys: (keyof T)[]): Partial<T> {
	const copy: Partial<T> = { ...base };
	for (const key of keys) delete copy[key];
	return copy;
}

const config = {
	baseUrl: DEFAULT_DYNAMO_BASE_URL,
	apiKey: "test-key",
	traceEnabled: true,
	sessionTypeId: DEFAULT_SESSION_TYPE_ID,
	isSubagent: false,
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
				DYN_REQUEST_TRACE: "1",
				DYN_AGENT_SESSION_TYPE_ID: "session-kind",
				DYN_AGENT_SESSION_ID: "session-id",
				DYN_AGENT_TRAJECTORY_ID: "trajectory-id",
				DYN_AGENT_PARENT_TRAJECTORY_ID: "parent-id",
			}),
		).toEqual({
			baseUrl: "http://dynamo.test/v1",
			apiKey: "dyn-key",
			traceEnabled: true,
			sessionTypeId: "session-kind",
			sessionId: "session-id",
			trajectoryId: "trajectory-id",
			parentTrajectoryId: "parent-id",
			isSubagent: false,
		});
	});

	it("treats DYN_REQUEST_TRACE as a truthy master switch, default off", () => {
		expect(readDynamoConfig({}).traceEnabled).toBe(false);
		for (const v of ["1", "true", "TRUE", "yes", "on"]) {
			expect(readDynamoConfig({ DYN_REQUEST_TRACE: v }).traceEnabled).toBe(true);
		}
		for (const v of ["0", "false", "no", ""]) {
			expect(readDynamoConfig({ DYN_REQUEST_TRACE: v }).traceEnabled).toBe(false);
		}
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
		expect(computeSubagentTrajectoryRewrite(envWithout(childEnv, "PI_SUBAGENT_CHILD"))).toBeNull();
	});

	it("uses an explicit DYN_AGENT_PARENT_TRAJECTORY_ID when present (manual wins)", () => {
		expect(
			computeSubagentTrajectoryRewrite({ ...childEnv, DYN_AGENT_PARENT_TRAJECTORY_ID: "manual-parent" }),
		).toEqual({
			parentTrajectoryId: "manual-parent",
			trajectoryId: "run-1:researcher:2",
		});
	});

	it("still creates a child trajectory when inherited DYN_AGENT_TRAJECTORY_ID is absent", () => {
		expect(computeSubagentTrajectoryRewrite(envWithout(childEnv, "DYN_AGENT_TRAJECTORY_ID"))).toEqual({
			trajectoryId: "run-1:researcher:2",
		});
	});

	it("skips when PI_SUBAGENT_RUN_ID or PI_SUBAGENT_CHILD_AGENT is missing", () => {
		expect(computeSubagentTrajectoryRewrite(envWithout(childEnv, "PI_SUBAGENT_RUN_ID"))).toBeNull();
		expect(computeSubagentTrajectoryRewrite(envWithout(childEnv, "PI_SUBAGENT_CHILD_AGENT"))).toBeNull();
	});

	it("readDynamoConfig surfaces the synthesized ids", () => {
		const cfg = readDynamoConfig(childEnv);
		expect(cfg.trajectoryId).toBe("run-1:researcher:2");
		expect(cfg.parentTrajectoryId).toBe("parent-traj");
		expect(cfg.isSubagent).toBe(true);
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
			...envWithout(env, "DYN_AGENT_PARENT_TRAJECTORY_ID"),
			PI_SUBAGENT_CHILD_AGENT: "subworker",
			PI_SUBAGENT_CHILD_INDEX: "0",
		};
		expect(computeSubagentTrajectoryRewrite(grandchildEnv)).toEqual({
			parentTrajectoryId: "run-1:researcher:2",
			trajectoryId: "run-1:subworker:0",
		});
	});
});

describe("root trajectory seed", () => {
	it("seeds DYN_AGENT_TRAJECTORY_ID at the root so subagents inherit a parent", () => {
		const env: NodeJS.ProcessEnv = { DYN_REQUEST_TRACE: "1" };
		expect(seedRootTrajectory(env, () => "root-traj")).toBe(true);
		expect(env.DYN_AGENT_TRAJECTORY_ID).toBe("root-traj");
		// The bug fix: a subagent spawned from this env now resolves a parent.
		const childEnv = {
			...env,
			PI_SUBAGENT_CHILD: "1",
			PI_SUBAGENT_RUN_ID: "run-1",
			PI_SUBAGENT_CHILD_AGENT: "researcher",
		};
		expect(computeSubagentTrajectoryRewrite(childEnv)).toEqual({
			parentTrajectoryId: "root-traj",
			trajectoryId: "run-1:researcher:0",
		});
	});

	it("uses DYN_AGENT_SESSION_ID as the root trajectory when present", () => {
		const env: NodeJS.ProcessEnv = { DYN_REQUEST_TRACE: "1", DYN_AGENT_SESSION_ID: "sess-7" };
		expect(seedRootTrajectory(env, () => "unused")).toBe(true);
		expect(env.DYN_AGENT_TRAJECTORY_ID).toBe("sess-7");
	});

	it("no-ops when trace is off, in a subagent child, or trajectory already set", () => {
		expect(seedRootTrajectory({}, () => "x")).toBe(false);
		expect(seedRootTrajectory({ DYN_REQUEST_TRACE: "1", PI_SUBAGENT_CHILD: "1" }, () => "x")).toBe(false);
		const preset: NodeJS.ProcessEnv = { DYN_REQUEST_TRACE: "1", DYN_AGENT_TRAJECTORY_ID: "caller" };
		expect(seedRootTrajectory(preset, () => "x")).toBe(false);
		expect(preset.DYN_AGENT_TRAJECTORY_ID).toBe("caller");
	});
});

describe("agent context injection", () => {
	it("defaults both trajectory_id and session_id to the Pi session ID", () => {
		expect(buildDynamoAgentContext(config, { sessionId: "pi-session" })).toEqual({
			trajectory_id: "pi-session",
			session_id: "pi-session",
			session_type_id: DEFAULT_SESSION_TYPE_ID,
			phase: "reasoning",
		});
	});

	it("lets DYN_AGENT_* override the Pi-session defaults", () => {
		expect(
			buildDynamoAgentContext(
				{ ...config, trajectoryId: "trajectory-from-env", sessionId: "session-from-env" },
				{ sessionId: "pi-session" },
			),
		).toEqual({
			trajectory_id: "trajectory-from-env",
			session_id: "session-from-env",
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
					session_id: "pi-session",
					session_type_id: DEFAULT_SESSION_TYPE_ID,
					phase: "reasoning",
				},
			},
		});
	});

	it("injects nothing when DYN_REQUEST_TRACE is off (plain provider), but still sets x-request-id", async () => {
		let capturedOptions: SimpleStreamOptions | undefined;
		const streamSimple = createDynamoStreamSimple(
			{ ...config, traceEnabled: false },
			(_model, _context, options) => {
				capturedOptions = options;
				return createAssistantMessageEventStream();
			},
			() => "request-1",
		);

		streamSimple(model, context, { sessionId: "pi-session" });
		expect(capturedOptions?.headers).toEqual({ "x-request-id": "request-1" });
		// No onPayload wrapper means no nvext injection.
		const payload = { model: "default" };
		expect((await capturedOptions?.onPayload?.(payload, model)) ?? payload).toEqual({ model: "default" });
	});
});

describe("subagent trajectory context", () => {
	const subagentEnv = {
		DYNAMO_BASE_URL: "http://dynamo.test",
		DYN_AGENT_TRAJECTORY_ID: "orchestrator",
		PI_SUBAGENT_CHILD: "1",
		PI_SUBAGENT_RUN_ID: "run-1",
		PI_SUBAGENT_CHILD_AGENT: "scout",
		PI_SUBAGENT_CHILD_INDEX: "3",
	} as const;

	it("sets child trajectory only for a pi-subagents child", () => {
		expect(computeSubagentTrajectoryId(subagentEnv)).toBe("run-1:scout:3");
		const { PI_SUBAGENT_CHILD: _omit, ...leadEnv } = subagentEnv;
		expect(computeSubagentTrajectoryId(leadEnv)).toBeUndefined();
	});

	it("derives child trajectory from PI_SUBAGENT_* alone", () => {
		const noTrajectory = envWithout(subagentEnv, "DYN_AGENT_TRAJECTORY_ID");
		const cfg = readDynamoConfig(noTrajectory);
		expect(cfg.trajectoryId).toBe("run-1:scout:3");
		expect(cfg.parentTrajectoryId).toBeUndefined();
		expect(cfg.isSubagent).toBe(true);
	});

	it("requires a complete subagent identity (run id + agent name)", () => {
		expect(computeSubagentTrajectoryId(envWithout(subagentEnv, "PI_SUBAGENT_RUN_ID"))).toBeUndefined();
		expect(computeSubagentTrajectoryId(envWithout(subagentEnv, "PI_SUBAGENT_CHILD_AGENT"))).toBeUndefined();
		// Index defaults to 0 when absent.
		expect(computeSubagentTrajectoryId(envWithout(subagentEnv, "PI_SUBAGENT_CHILD_INDEX"))).toBe("run-1:scout:0");
	});

	it("trajectory_final sends agent_context only", async () => {
		const calls: Array<{ url: string; body: unknown; headers: unknown }> = [];
		const fakeFetch = async (url: string, init: RequestInit) => {
			calls.push({
				url,
				body: JSON.parse(String(init.body)),
				headers: init.headers,
			});
			return { ok: true, status: 200 };
		};

		const cfg = readDynamoConfig({ ...subagentEnv, DYN_REQUEST_TRACE: "1", DYN_AGENT_SESSION_ID: "run-1" });
		expect(await sendTrajectoryFinal(cfg, "zai-org/GLM-4.7-Flash", () => "close-req-1", fakeFetch)).toBe(true);
		expect(calls).toHaveLength(1);
		expect(calls[0]?.url).toBe("http://dynamo.test/v1/chat/completions");
		expect(calls[0]?.body).toEqual({
			model: "zai-org/GLM-4.7-Flash",
			messages: [{ role: "user", content: "." }],
			max_tokens: 1,
			stream: false,
			nvext: {
				agent_context: {
					trajectory_id: "run-1:scout:3",
					parent_trajectory_id: "orchestrator",
					session_id: "run-1",
					session_type_id: DEFAULT_SESSION_TYPE_ID,
					phase: "reasoning",
					trajectory_final: true,
				},
			},
		});
		expect((calls[0]?.headers as Record<string, string>)["x-request-id"]).toBe("close-req-1");
	});

	it("streamSimple injects subagent agent_context without session_control", async () => {
		let capturedOptions: SimpleStreamOptions | undefined;
		const subagentConfig = readDynamoConfig({ ...subagentEnv, DYN_REQUEST_TRACE: "1", DYN_AGENT_SESSION_ID: "run-1" });
		const streamSimple = createDynamoStreamSimple(
			subagentConfig,
			(_model, _context, options) => {
				capturedOptions = options;
				return createAssistantMessageEventStream();
			},
			() => "request-1",
		);

		streamSimple(model, context, { sessionId: "pi-session" });
		const onPayload = capturedOptions?.onPayload;
		if (!onPayload) throw new Error("expected wrapped onPayload");
		const injected = (await onPayload({ model: DEFAULT_DYNAMO_MODEL_ID }, model)) as {
			nvext: { agent_context: unknown; session_control?: unknown };
		};

		expect(injected.nvext.agent_context).toEqual({
			trajectory_id: "run-1:scout:3",
			parent_trajectory_id: "orchestrator",
			session_id: "run-1",
			session_type_id: DEFAULT_SESSION_TYPE_ID,
			phase: "reasoning",
		});
		expect(injected.nvext.session_control).toBeUndefined();
	});
});
