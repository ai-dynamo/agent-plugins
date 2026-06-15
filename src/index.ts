// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { randomUUID } from "node:crypto";
import type { ExtensionAPI } from "@mariozechner/pi-coding-agent";
import {
	applySubagentBridge,
	createDynamoModels,
	createDynamoProviderConfig,
	DEFAULT_DYNAMO_MODEL_ID,
	DYNAMO_PROVIDER_ID,
	discoverDynamoModels,
	readDynamoConfig,
	seedRootTrajectory,
	sendTrajectoryFinal,
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
	const closeModelId = models.map((model) => model.id.trim()).find((id) => id.length > 0) ?? DEFAULT_DYNAMO_MODEL_ID;
	const providerModels = models.map((model) => ({ ...model }));

	pi.registerProvider(DYNAMO_PROVIDER_ID, createDynamoProviderConfig(config, providerModels));
	if (config.traceEnabled) {
		await registerDynamoToolEventRelay(pi, config);
	}

	// trajectory_final closes the current trajectory: every subagent on agent_end,
	// the root only on true quit. Other session_shutdown reasons keep the same
	// trajectory alive across reload/fork/new/resume flows.
	const programTrajectoryId = config.trajectoryId ?? config.sessionId;
	if (config.traceEnabled && programTrajectoryId) {
		let programClosed = false;
		const closeProgram = async (): Promise<void> => {
			if (programClosed) return;
			programClosed = true;
			await sendTrajectoryFinal(config, closeModelId, randomUUID);
		};
		if (config.isSubagent) {
			pi.on("agent_end", closeProgram);
			pi.on("session_shutdown", closeProgram);
		} else {
			pi.on("session_shutdown", async (event) => {
				if (event.reason === "quit") await closeProgram();
			});
		}
	}
}

export * from "./dynamo-provider.js";
export * from "./tool-relay.js";
