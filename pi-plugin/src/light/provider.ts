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
export const DEFAULT_DYNAMO_CONTEXT_WINDOW = 128000;
export const DEFAULT_DYNAMO_MAX_TOKENS = 8192;

export interface DynamoEnvironment extends DynamoSessionEnvironment {
	DYNAMO_BASE_URL?: string;
	OPENAI_BASE_URL?: string;
	DYNAMO_API_KEY?: string;
	/** Overrides the context_window Dynamo advertises for each model. Without either, 128000. */
	DYNAMO_CONTEXT_WINDOW?: string;
	/** Output cap sent as max_tokens on every request. Without it, 8192. */
	DYNAMO_MAX_TOKENS?: string;
	DYN_AGENT_SESSION_FINAL?: string;
}

export type DynamoLimitsEnvironment = Pick<DynamoEnvironment, "DYNAMO_CONTEXT_WINDOW" | "DYNAMO_MAX_TOKENS">;

export type DynamoContextWindowSource = "DYNAMO_CONTEXT_WINDOW" | "advertised" | "default";
export type DynamoMaxTokensSource = "DYNAMO_MAX_TOKENS" | "default";

export interface DynamoModelLimits {
	contextWindow: number;
	contextWindowSource: DynamoContextWindowSource;
	maxTokens: number;
	maxTokensSource: DynamoMaxTokensSource;
}

export interface DynamoModelOptions {
	/** context_window per model id, as listed by GET /v1/models. */
	servedContextWindows?: ReadonlyMap<string, number>;
	env?: DynamoLimitsEnvironment;
	/** Receives each model's resolved limits and where each value came from. */
	onLimits?: (modelId: string, limits: DynamoModelLimits) => void;
}

export interface DynamoDiscoveryOptions extends Pick<DynamoModelOptions, "env" | "onLimits"> {
	fetchImpl?: typeof fetch;
}

export interface DynamoConfig {
	baseUrl: string;
	apiKey: string;
	traceEnabled: boolean;
	sessionFinalEnabled?: boolean;
	sessionId?: string;
	parentSessionId?: string;
}

interface OpenAIModelsResponse {
	data?: Array<{ id?: unknown; context_window?: unknown }>;
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
		sessionFinalEnabled:
			envValue(env, "DYN_AGENT_SESSION_FINAL") === undefined ||
			isTruthyEnv(envValue(env, "DYN_AGENT_SESSION_FINAL")),
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
	config: Pick<DynamoConfig, "sessionId" | "parentSessionId">,
	runtimeSessionId: string | undefined,
	createRequestId: () => string = randomUUID,
): Record<string, string> {
	const nextHeaders = { ...headers };
	if (!hasHeader(nextHeaders, "x-request-id")) nextHeaders["x-request-id"] = createRequestId();

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

// A blank value means "not configured" so job templates can leave the override
// empty. Anything else is a misconfiguration worth failing loudly on: a silent
// fallback would let Pi run with limits the operator believes they changed.
function limitFromEnv(env: DynamoLimitsEnvironment, key: keyof DynamoLimitsEnvironment): number | undefined {
	const value = envValue(env, key);
	if (value === undefined) return undefined;
	const parsed = Number(value);
	if (!Number.isSafeInteger(parsed) || parsed <= 0) {
		throw new Error(`${key} must be a positive integer, got ${JSON.stringify(value)}`);
	}
	return parsed;
}

export function resolveDynamoModelLimits(
	modelId: string,
	{ servedContextWindows, env = process.env }: Pick<DynamoModelOptions, "servedContextWindows" | "env"> = {},
): DynamoModelLimits {
	// Pi compacts the conversation against contextWindow, so a window larger
	// than the engine serves lets a prompt grow until Dynamo rejects it. The
	// explicit override comes first for endpoints that advertise nothing or
	// that an operator wants to run below the engine limit.
	const envContextWindow = limitFromEnv(env, "DYNAMO_CONTEXT_WINDOW");
	const envMaxTokens = limitFromEnv(env, "DYNAMO_MAX_TOKENS");
	const served = servedContextWindows?.get(modelId);

	let contextWindow: number;
	let contextWindowSource: DynamoContextWindowSource;
	if (envContextWindow !== undefined) {
		contextWindow = envContextWindow;
		contextWindowSource = "DYNAMO_CONTEXT_WINDOW";
	} else if (served !== undefined) {
		contextWindow = served;
		contextWindowSource = "advertised";
	} else {
		contextWindow = DEFAULT_DYNAMO_CONTEXT_WINDOW;
		contextWindowSource = "default";
	}

	const maxTokens = envMaxTokens ?? DEFAULT_DYNAMO_MAX_TOKENS;
	const maxTokensSource: DynamoMaxTokensSource = envMaxTokens !== undefined ? "DYNAMO_MAX_TOKENS" : "default";

	if (maxTokens >= contextWindow) {
		throw new Error(
			`Dynamo model ${modelId}: max tokens ${maxTokens} must be below the context window ${contextWindow}`,
		);
	}
	return { contextWindow, contextWindowSource, maxTokens, maxTokensSource };
}

export function describeDynamoModelLimits(modelId: string, limits: DynamoModelLimits): string {
	return (
		`pi-dynamo-provider: ${modelId} contextWindow ${limits.contextWindow} (${limits.contextWindowSource}), ` +
		`maxTokens ${limits.maxTokens} (${limits.maxTokensSource})`
	);
}

export function createDynamoModels(
	modelIds: string[],
	baseUrl: string,
	options: DynamoModelOptions = {},
): ProviderModelConfig[] {
	const ids = modelIds.length > 0 ? modelIds : [DEFAULT_DYNAMO_MODEL_ID];
	return ids.map((id) => {
		const limits = resolveDynamoModelLimits(id, options);
		options.onLimits?.(id, limits);
		return {
			id,
			name: id,
			api: DYNAMO_API,
			baseUrl,
			reasoning: false,
			input: ["text"],
			cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
			contextWindow: limits.contextWindow,
			maxTokens: limits.maxTokens,
			compat: dynamoOpenAICompat,
		};
	});
}

export async function discoverDynamoModels(
	config: DynamoConfig,
	timeoutMs = 2000,
	options: DynamoDiscoveryOptions = {},
): Promise<ProviderModelConfig[]> {
	const { fetchImpl = fetch, ...modelOptions } = options;
	const controller = new AbortController();
	const timeout = setTimeout(() => controller.abort(), timeoutMs);
	let body: OpenAIModelsResponse;
	try {
		const response = await fetchImpl(`${config.baseUrl}/models`, {
			headers: { Authorization: `Bearer ${config.apiKey}` },
			signal: controller.signal,
		});
		if (!response.ok) return [];
		body = (await response.json()) as OpenAIModelsResponse;
	} catch {
		return [];
	} finally {
		clearTimeout(timeout);
	}
	const ids: string[] = [];
	const servedContextWindows = new Map<string, number>();
	for (const entry of body.data ?? []) {
		if (typeof entry.id !== "string" || entry.id.length === 0) continue;
		ids.push(entry.id);
		// Dynamo lists the engine's serving limit as context_window; the OpenAI
		// schema has no such field, so other endpoints simply fall through.
		if (typeof entry.context_window === "number" && Number.isSafeInteger(entry.context_window) && entry.context_window > 1) {
			servedContextWindows.set(entry.id, entry.context_window);
		}
	}
	// Limits are validated outside the try block on purpose: a bad override is
	// an operator error, not a discovery failure to paper over with defaults.
	return createDynamoModels([...new Set(ids)], config.baseUrl, { ...modelOptions, servedContextWindows });
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

export async function sendDynamoSessionFinal(
	config: DynamoConfig,
	modelId: string,
	runtimeSessionId: string | undefined,
	createRequestId: () => string = randomUUID,
	fetchImpl: typeof fetch = fetch,
): Promise<boolean> {
	const sessionId = config.sessionId ?? runtimeSessionId?.trim();
	if (config.sessionFinalEnabled === false || !sessionId) return false;

	try {
		const response = await fetchImpl(`${config.baseUrl}/chat/completions`, {
			method: "POST",
			headers: buildDynamoHeaders(
				{
					"content-type": "application/json",
					authorization: `Bearer ${config.apiKey}`,
					"x-dynamo-session-final": "true",
				},
				config,
				sessionId,
				createRequestId,
			),
			body: JSON.stringify({
				model: modelId.trim() || DEFAULT_DYNAMO_MODEL_ID,
				messages: [{ role: "user", content: "." }],
				max_tokens: 1,
				stream: false,
			}),
			signal: AbortSignal.timeout(5000),
		});
		return response.ok;
	} catch {
		return false;
	}
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
