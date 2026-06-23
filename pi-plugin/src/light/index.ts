// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { ExtensionAPI } from "@mariozechner/pi-coding-agent";
import {
	createDynamoModels,
	createDynamoProviderConfig,
	DEFAULT_DYNAMO_MODEL_ID,
	DYNAMO_PROVIDER_ID,
	discoverDynamoModels,
	readDynamoConfig,
} from "./provider.js";
import { registerDynamoToolEventRelay } from "./tool-relay.js";
import { applySubagentSessionBridge } from "./session.js";

export default async function dynamoProviderExtension(pi: ExtensionAPI): Promise<void> {
	applySubagentSessionBridge();
	const config = readDynamoConfig();
	const discoveredModels = await discoverDynamoModels(config);
	const models =
		discoveredModels.length > 0 ? discoveredModels : createDynamoModels([DEFAULT_DYNAMO_MODEL_ID], config.baseUrl);
	pi.registerProvider(DYNAMO_PROVIDER_ID, createDynamoProviderConfig(config, models.map((model) => ({ ...model }))));
	await registerDynamoToolEventRelay(pi, config);
}

export * from "./provider.js";
export * from "./tool-relay.js";
export * from "./session.js";
