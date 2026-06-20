// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { createAssistantMessageEventStream, type Context, type Model, type SimpleStreamOptions } from "@mariozechner/pi-ai";
import { describe, expect, it } from "vitest";
import {
	applySubagentBridge,
	buildDynamoHeaders,
	computeSubagentTrajectoryId,
	computeSubagentTrajectoryRewrite,
	createDynamoStreamSimple,
	DEFAULT_DYNAMO_BASE_URL,
	DEFAULT_DYNAMO_MODEL_ID,
	DEFAULT_SESSION_TYPE_ID,
	type DynamoProviderRuntimeConfig,
	DYNAMO_API,
	normalizeDynamoBaseUrl,
	readDynamoConfig,
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
				DYN_AGENT_TRAJECTORY_ID: "trajectory-id",
				DYN_AGENT_PARENT_TRAJECTORY_ID: "parent-id",
			}),
		).toEqual({
			baseUrl: "http://dynamo.test/v1",
			apiKey: "dyn-key",
			traceEnabled: true,
			sessionTypeId: "session-kind",
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
	it("delegates through openai-completions with Pi session affinity enabled", () => {
		let capturedModel: Model<"openai-completions"> | undefined;
		let capturedOptions: SimpleStreamOptions | undefined;
		const runtimeConfig: DynamoProviderRuntimeConfig = { ...config };
		const onPayload = (payload: unknown) => payload;

		const streamSimple = createDynamoStreamSimple(
			runtimeConfig,
			(openAIModel, _context, options) => {
				capturedModel = openAIModel;
				capturedOptions = options;
				return createAssistantMessageEventStream();
			},
			() => "request-1",
		);

		streamSimple(model, context, {
			sessionId: "pi-session",
			onPayload,
		});

		expect(capturedModel?.api).toBe("openai-completions");
		expect(capturedModel?.provider).toBe("dynamo");
		expect(capturedModel?.compat?.sendSessionAffinityHeaders).toBe(true);
		expect(capturedOptions?.apiKey).toBe("test-key");
		expect(capturedOptions?.sessionId).toBe("pi-session");
		expect(capturedOptions?.onPayload).toBe(onPayload);
		expect(capturedOptions?.headers).toEqual({ "x-request-id": "request-1" });
	});

	it("injects no payload wrapper when DYN_REQUEST_TRACE is off, but still sets x-request-id", async () => {
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
		expect(capturedOptions?.sessionId).toBe("pi-session");
		const payload = { model: "default" };
		expect((await capturedOptions?.onPayload?.(payload, model)) ?? payload).toEqual({ model: "default" });
	});

	it("preserves Pi's runtime session id for root requests even when a trace trajectory is configured", () => {
		let capturedOptions: SimpleStreamOptions | undefined;
		const streamSimple = createDynamoStreamSimple(
			{ ...config, trajectoryId: "manual-root-traj", isSubagent: false },
			(_model, _context, options) => {
				capturedOptions = options;
				return createAssistantMessageEventStream();
			},
			() => "request-1",
		);

		streamSimple(model, context, { sessionId: "pi-session" });

		expect(capturedOptions?.sessionId).toBe("pi-session");
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

	it("uses the subagent trajectory as the provider session id", () => {
		let capturedOptions: SimpleStreamOptions | undefined;
		const subagentConfig = readDynamoConfig({ ...subagentEnv, DYN_REQUEST_TRACE: "1" });
		const streamSimple = createDynamoStreamSimple(
			subagentConfig,
			(_model, _context, options) => {
				capturedOptions = options;
				return createAssistantMessageEventStream();
			},
			() => "request-1",
		);

		streamSimple(model, context, { sessionId: "pi-session" });

		expect(capturedOptions?.sessionId).toBe("run-1:scout:3");
	});
});
