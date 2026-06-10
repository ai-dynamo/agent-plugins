// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { randomUUID } from "node:crypto";
import type { ExtensionAPI } from "@mariozechner/pi-coding-agent";
import {
	applySubagentBridge,
	buildDynamoAgentContext,
	createDynamoModels,
	createDynamoProviderConfig,
	DEFAULT_DYNAMO_MODEL_ID,
	DYNAMO_PROVIDER_ID,
	DynamoSubagentSession,
	discoverDynamoModels,
	readDynamoConfig,
	seedRootTrajectory,
} from "./dynamo-provider.js";
import { registerDynamoToolEventRelay } from "./tool-relay.js";

export default async function dynamoProviderExtension(pi: ExtensionAPI): Promise<void> {
	// Seed a root trajectory id (root only) BEFORE anything spawns subagents, so
	// the first generation of pi-subagents has a parent to inherit; without it the
	// bridge no-ops and the whole chain stays flat (no parent_trajectory_id).
	seedRootTrajectory();
	// Mutate process.env BEFORE readDynamoConfig so the rewrite also reaches
	// any pi-subagents this process later spawns. readDynamoConfig itself
	// recomputes the rewrite independently, so omitting this call still
	// yields a correct config for THIS process — but nested subagent chains
	// collapse to the root parent without the env mutation.
	applySubagentBridge();
	const config = readDynamoConfig();
	const discoveredModels = await discoverDynamoModels(config);
	const models =
		discoveredModels.length > 0 ? discoveredModels : createDynamoModels([DEFAULT_DYNAMO_MODEL_ID], config.baseUrl);

	// DYN_AGENT_TRACE gates the agentic emissions. A pi-subagents child gets a
	// streaming session keyed on its own identity; the lead agent stays unpinned.
	const session =
		config.traceEnabled && config.sessionControlId
			? new DynamoSubagentSession({
					baseUrl: config.baseUrl,
					apiKey: config.apiKey,
					sessionControlId: config.sessionControlId,
					...(config.sessionTimeoutSecs !== undefined ? { sessionTimeoutSecs: config.sessionTimeoutSecs } : {}),
				})
			: undefined;

	pi.registerProvider(DYNAMO_PROVIDER_ID, createDynamoProviderConfig(config, models, session));
	if (config.traceEnabled) {
		await registerDynamoToolEventRelay(pi, config);
	}

	if (session) {
		// agent_end fires when this subagent's loop finishes, while the event loop
		// is still alive — so we can await the close request and free KV
		// deterministically. session_shutdown is the teardown-time backstop; both
		// are idempotent. If neither lands (e.g. SIGKILL), Dynamo's idle timeout
		// reaps the session.
		pi.on("agent_end", async () => {
			await session.close();
		});
		pi.on("session_shutdown", async () => {
			await session.close();
		});
	}

	// Program close (thunderagent_router): whenever agent_context is being injected
	// (trace enabled + a trajectory id), release the program from the router's table
	// when the whole session ends. A throwaway max_tokens=1 request carries
	// agent_context.trajectory_final; the thunderagent_router short-circuits it
	// (deletes the program, never forwards to the engine). Best-effort — Dynamo's
	// idle reaper is the backstop if the process dies before it lands. Separate from
	// session_control above: that frees SGLang KV; this frees scheduler bookkeeping.
	const programTrajectoryId = config.trajectoryId ?? config.sessionId;
	if (config.traceEnabled && programTrajectoryId) {
		const closeModelId = models[0]?.id ?? DEFAULT_DYNAMO_MODEL_ID;
		let programClosed = false;
		const closeProgram = async (): Promise<void> => {
			if (programClosed) return;
			programClosed = true;
			const agentContext = { ...buildDynamoAgentContext(config), trajectory_final: true };
			try {
				await fetch(`${config.baseUrl}/chat/completions`, {
					method: "POST",
					headers: {
						"content-type": "application/json",
						authorization: `Bearer ${config.apiKey}`,
						"x-request-id": randomUUID(),
					},
					body: JSON.stringify({
						model: closeModelId,
						messages: [{ role: "user", content: "." }],
						max_tokens: 1,
						stream: false,
						nvext: { agent_context: agentContext },
					}),
					signal: AbortSignal.timeout(5000),
				});
			} catch {
				// best-effort: the router's idle reaper is the safety net
			}
		};
		// Agents here are multiturn: the whole interactive session is ONE
		// trajectory/program (same trajectory_id across every prompt). Release it
		// once at true teardown — NOT on agent_end, which fires per user prompt and
		// would close after the first turn, dropping the program's worker/KV affinity
		// mid-session (later prompts re-create an unreleased program that only decay
		// reaps). Only reason "quit" means the trajectory is done in this process;
		// "reload"/"fork"/"new"/"resume" keep the same trajectory_id, so the program
		// continues. print-mode (batch) also emits "quit" on dispose and awaits the
		// handler, so one-shot runs still close exactly once.
		pi.on("session_shutdown", async (event) => {
			if (event.reason === "quit") await closeProgram();
		});
	}
}

export * from "./dynamo-provider.js";
export * from "./tool-relay.js";
