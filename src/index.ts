// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { ExtensionAPI } from "@mariozechner/pi-coding-agent";
import {
	applySubagentBridge,
	createDynamoModels,
	createDynamoProviderConfig,
	DEFAULT_DYNAMO_MODEL_ID,
	DYNAMO_PROVIDER_ID,
	DynamoSubagentSession,
	discoverDynamoModels,
	readDynamoConfig,
} from "./dynamo-provider.js";
import { registerDynamoToolEventRelay } from "./tool-relay.js";

export default async function dynamoProviderExtension(pi: ExtensionAPI): Promise<void> {
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
}

export * from "./dynamo-provider.js";
export * from "./tool-relay.js";
