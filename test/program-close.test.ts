// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import dynamoProviderExtension from "../src/index.js";

// Multiturn program-close contract: the trajectory_final release must fire ONCE
// at true session teardown (session_shutdown reason "quit") — never per user turn
// (agent_end) and never on continuation reasons (reload/fork/new/resume), which
// keep the same trajectory_id alive.

type Handler = (event: any, ctx?: any) => unknown | Promise<unknown>;

function makePi(onRegisterProvider?: (providerConfig: any) => void) {
	const handlers: Record<string, Handler> = {};
	const pi = {
		registerProvider: vi.fn((_id: string, providerConfig: any) => {
			onRegisterProvider?.(providerConfig);
		}),
		on: (event: string, handler: Handler) => {
			handlers[event] = handler;
		},
	};
	return { pi, handlers };
}

// Capture POST /chat/completions close pings; answer GET /models for discovery.
function installFetch() {
	const closeBodies: any[] = [];
	const fetchMock = vi.fn(async (url: any, init: any = {}) => {
		const u = String(url);
		if (init.method === "POST" && u.includes("/chat/completions")) {
			closeBodies.push(JSON.parse(init.body));
			return { ok: true, json: async () => ({ choices: [] }) } as any;
		}
		// model discovery
		return { ok: true, json: async () => ({ data: [{ id: "nvidia/MiniMax-M2.7-NVFP4" }] }) } as any;
	});
	vi.stubGlobal("fetch", fetchMock);
	return closeBodies;
}

describe("program close (trajectory_final) — multiturn", () => {
	const savedEnv = process.env;
	beforeEach(() => {
		process.env = {
			...savedEnv,
			DYN_REQUEST_TRACE: "1",
			DYN_AGENT_SESSION_ID: "t-1",
			DYN_AGENT_TRAJECTORY_ID: "t-1",
			DYNAMO_BASE_URL: "http://frontend:8000/v1",
		};
		delete process.env.PI_SUBAGENT_CHILD; // ensure lead-agent path, not subagent
	});
	afterEach(() => {
		process.env = savedEnv;
		vi.unstubAllGlobals();
		vi.restoreAllMocks();
	});

	it("does NOT hook agent_end (no per-turn close)", async () => {
		const closeBodies = installFetch();
		const { pi, handlers } = makePi();
		await dynamoProviderExtension(pi as any);
		expect(handlers.agent_end).toBeUndefined();
		expect(closeBodies).toHaveLength(0);
	});

	it("does NOT close on continuation reasons (reload/fork)", async () => {
		const closeBodies = installFetch();
		const { pi, handlers } = makePi();
		await dynamoProviderExtension(pi as any);
		await handlers.session_shutdown!({ type: "session_shutdown", reason: "reload" });
		await handlers.session_shutdown!({ type: "session_shutdown", reason: "fork" });
		expect(closeBodies).toHaveLength(0);
	});

	it("closes exactly once on session_shutdown reason 'quit', carrying trajectory_final", async () => {
		const closeBodies = installFetch();
		const { pi, handlers } = makePi();
		await dynamoProviderExtension(pi as any);
		await handlers.session_shutdown!({ type: "session_shutdown", reason: "quit" });
		// idempotent: a second quit (or any later event) must not re-close
		await handlers.session_shutdown!({ type: "session_shutdown", reason: "quit" });
		expect(closeBodies).toHaveLength(1);
		const ctx = closeBodies[0].nvext.agent_context;
		expect(ctx.trajectory_final).toBe(true);
		expect(ctx.trajectory_id).toBe("t-1");
		expect(closeBodies[0].max_tokens).toBe(1);
	});

	it("uses a stable discovered model id for the shutdown close ping", async () => {
		const closeBodies = installFetch();
		const { pi, handlers } = makePi((providerConfig) => {
			providerConfig.models[0].id = "";
		});
		await dynamoProviderExtension(pi as any);
		await handlers.session_shutdown!({ type: "session_shutdown", reason: "quit" });

		expect(closeBodies).toHaveLength(1);
		expect(closeBodies[0].model).toBe("nvidia/MiniMax-M2.7-NVFP4");
	});
});

describe("subagent trajectory close", () => {
	const savedEnv = process.env;
	beforeEach(() => {
		process.env = {
			...savedEnv,
			DYN_REQUEST_TRACE: "1",
			DYN_AGENT_SESSION_ID: "root-session",
			DYN_AGENT_TRAJECTORY_ID: "root-trajectory",
			DYNAMO_BASE_URL: "http://frontend:8000/v1",
			PI_SUBAGENT_CHILD: "1",
			PI_SUBAGENT_RUN_ID: "root-session",
			PI_SUBAGENT_CHILD_AGENT: "researcher",
			PI_SUBAGENT_CHILD_INDEX: "0",
		};
	});
	afterEach(() => {
		process.env = savedEnv;
		vi.unstubAllGlobals();
		vi.restoreAllMocks();
	});

	it("closes the child trajectory on agent_end", async () => {
		const closeBodies = installFetch();
		const { pi, handlers } = makePi();
		await dynamoProviderExtension(pi as any);
		await handlers.agent_end!({ type: "agent_end", messages: [] });
		await handlers.session_shutdown!({ type: "session_shutdown", reason: "quit" });

		expect(closeBodies).toHaveLength(1);
		const ctx = closeBodies[0].nvext.agent_context;
		expect(ctx.trajectory_final).toBe(true);
		expect(ctx.trajectory_id).toBe("root-session:researcher:0");
		expect(ctx.parent_trajectory_id).toBe("root-trajectory");
	});
});
