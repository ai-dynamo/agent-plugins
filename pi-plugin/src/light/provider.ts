// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { randomUUID } from "node:crypto";
import { streamSimpleOpenAICompletions } from "@mariozechner/pi-ai";
import type {
	Api,
	AssistantMessageEventStream,
	Context,
	Model,
	OpenAICompletionsCompat,
	SimpleStreamOptions,
} from "@mariozechner/pi-ai";
import type { ProviderConfig, ProviderModelConfig } from "@mariozechner/pi-coding-agent";
import {
	envValue,
	isTruthyEnv,
	resolveSessionContext,
	type DynamoSessionEnvironment,
} from "./session.js";

export const DYNAMO_PROVIDER_ID = "dynamo";
export const DYNAMO_API = "dynamo-openai-completions" satisfies Api;
export const DEFAULT_DYNAMO_BASE_URL = "http://127.0.0.1:8000/v1";
export const DEFAULT_DYNAMO_API_KEY = "dynamo-local";
export const DEFAULT_DYNAMO_MODEL_ID = "default";

export interface DynamoEnvironment extends DynamoSessionEnvironment {
	DYNAMO_BASE_URL?: string;
	OPENAI_BASE_URL?: string;
	DYNAMO_API_KEY?: string;
}

export interface DynamoConfig {
	baseUrl: string;
	apiKey: string;
	traceEnabled: boolean;
	sessionId?: string;
	parentSessionId?: string;
}

interface OpenAIModelsResponse {
	data?: Array<{ id?: unknown }>;
}

type OpenAICompletionsStreamSimple = (
	model: Model<"openai-completions">,
	context: Context,
	options?: SimpleStreamOptions,
) => AssistantMessageEventStream;

type ProviderStreamSimple = NonNullable<ProviderConfig["streamSimple"]>;

export function normalizeDynamoBaseUrl(rawBaseUrl: string | undefined): string {
	const raw = rawBaseUrl?.trim() || DEFAULT_DYNAMO_BASE_URL;
	const withoutTrailingSlash = raw.replace(/\/+$/, "");
	try {
		const url = new URL(withoutTrailingSlash);
		if (url.pathname === "" || url.pathname === "/") url.pathname = "/v1";
		return url.toString().replace(/\/+$/, "");
	} catch {
		return withoutTrailingSlash;
	}
}

export function readDynamoConfig(env: DynamoEnvironment = process.env): DynamoConfig {
	const session = resolveSessionContext(env);
	return {
		baseUrl: normalizeDynamoBaseUrl(envValue(env, "DYNAMO_BASE_URL") ?? envValue(env, "OPENAI_BASE_URL")),
		apiKey: envValue(env, "DYNAMO_API_KEY") ?? DEFAULT_DYNAMO_API_KEY,
		traceEnabled: isTruthyEnv(envValue(env, "DYN_REQUEST_TRACE")),
		...(session.sessionId ? { sessionId: session.sessionId } : {}),
		...(session.parentSessionId ? { parentSessionId: session.parentSessionId } : {}),
	};
}

function hasHeader(headers: Record<string, string>, target: string): boolean {
	const normalizedTarget = target.toLowerCase();
	return Object.keys(headers).some((key) => key.toLowerCase() === normalizedTarget);
}

export function buildDynamoHeaders(
	headers: Record<string, string> | undefined,
	config: Pick<DynamoConfig, "traceEnabled" | "sessionId" | "parentSessionId">,
	runtimeSessionId: string | undefined,
	createRequestId: () => string = randomUUID,
): Record<string, string> {
	const nextHeaders = { ...headers };
	if (!hasHeader(nextHeaders, "x-request-id")) nextHeaders["x-request-id"] = createRequestId();
	if (!config.traceEnabled) return nextHeaders;

	const sessionId = config.sessionId ?? runtimeSessionId;
	if (sessionId && !hasHeader(nextHeaders, "x-dynamo-session-id")) {
		nextHeaders["x-dynamo-session-id"] = sessionId;
	}
	if (config.parentSessionId && !hasHeader(nextHeaders, "x-dynamo-parent-session-id")) {
		nextHeaders["x-dynamo-parent-session-id"] = config.parentSessionId;
	}
	return nextHeaders;
}

const dynamoOpenAICompat = {
	supportsStore: false,
	supportsDeveloperRole: false,
	supportsReasoningEffort: false,
	supportsUsageInStreaming: true,
	maxTokensField: "max_tokens",
	supportsStrictMode: false,
	supportsLongCacheRetention: false,
} satisfies OpenAICompletionsCompat;

export function createDynamoModels(modelIds: string[], baseUrl: string): ProviderModelConfig[] {
	const ids = modelIds.length > 0 ? modelIds : [DEFAULT_DYNAMO_MODEL_ID];
	return ids.map((id) => ({
		id,
		name: id,
		api: DYNAMO_API,
		baseUrl,
		reasoning: false,
		input: ["text"],
		cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
		contextWindow: 128000,
		maxTokens: 8192,
		compat: dynamoOpenAICompat,
	}));
}

export async function discoverDynamoModels(config: DynamoConfig, timeoutMs = 2000): Promise<ProviderModelConfig[]> {
	const controller = new AbortController();
	const timeout = setTimeout(() => controller.abort(), timeoutMs);
	try {
		const response = await fetch(`${config.baseUrl}/models`, {
			headers: { Authorization: `Bearer ${config.apiKey}` },
			signal: controller.signal,
		});
		if (!response.ok) return [];
		const body = (await response.json()) as OpenAIModelsResponse;
		const ids =
			body.data?.map((model) => model.id).filter((id): id is string => typeof id === "string" && id.length > 0) ??
			[];
		return createDynamoModels([...new Set(ids)], config.baseUrl);
	} catch {
		return [];
	} finally {
		clearTimeout(timeout);
	}
}

function toOpenAICompletionsModel(model: Model<Api>): Model<"openai-completions"> {
	const { api: _api, compat, ...rest } = model;
	return {
		...rest,
		api: "openai-completions",
		compat: (compat as OpenAICompletionsCompat | undefined) ?? dynamoOpenAICompat,
	};
}

export function createDynamoStreamSimple(
	config: DynamoConfig,
	delegate: OpenAICompletionsStreamSimple = streamSimpleOpenAICompletions,
	createRequestId: () => string = randomUUID,
): ProviderStreamSimple {
	return (model: Model<Api>, context: Context, options?: SimpleStreamOptions): AssistantMessageEventStream => {
		const runtimeSessionId = options?.sessionId?.trim();
		return delegate(toOpenAICompletionsModel(model), context, {
			...options,
			apiKey: options?.apiKey ?? config.apiKey,
			headers: buildDynamoHeaders(options?.headers, config, runtimeSessionId, createRequestId),
		});
	};
}

export function createDynamoProviderConfig(config: DynamoConfig, models: ProviderModelConfig[]): ProviderConfig {
	return {
		name: "Dynamo",
		baseUrl: config.baseUrl,
		apiKey: config.apiKey,
		api: DYNAMO_API,
		models,
		streamSimple: createDynamoStreamSimple(config),
	};
}
