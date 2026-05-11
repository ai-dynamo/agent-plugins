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

export const DYNAMO_PROVIDER_ID = "dynamo";
export const DYNAMO_API = "dynamo-openai-completions" satisfies Api;
export const DEFAULT_DYNAMO_BASE_URL = "http://127.0.0.1:8000/v1";
export const DEFAULT_DYNAMO_API_KEY = "dynamo-local";
export const DEFAULT_SESSION_TYPE_ID = "pi_coding_agent";
export const DEFAULT_DYNAMO_MODEL_ID = "default";

export interface DynamoEnvironment {
	DYNAMO_BASE_URL?: string;
	OPENAI_BASE_URL?: string;
	DYNAMO_API_KEY?: string;
	DYN_AGENT_SESSION_TYPE_ID?: string;
	DYN_AGENT_SESSION_ID?: string;
	DYN_AGENT_TRAJECTORY_ID?: string;
	DYN_AGENT_PARENT_TRAJECTORY_ID?: string;
	// pi-subagents bookkeeping vars. Set by pi-subagents when it spawns a
	// child pi process; read here only to bridge them into the dynamo agent
	// context. See readDynamoConfig for the rewrite rule.
	PI_SUBAGENT_CHILD?: string;
	PI_SUBAGENT_RUN_ID?: string;
	PI_SUBAGENT_CHILD_AGENT?: string;
	PI_SUBAGENT_CHILD_INDEX?: string;
}

export interface DynamoProviderRuntimeConfig {
	baseUrl: string;
	apiKey: string;
	sessionTypeId: string;
	sessionId?: string;
	trajectoryId?: string;
	parentTrajectoryId?: string;
}

export interface DynamoAgentContext {
	trajectory_id?: string;
	parent_trajectory_id?: string;
	session_id?: string;
	session_type_id: string;
	phase: "reasoning";
}

interface OpenAIModelsResponse {
	data?: Array<{
		id?: unknown;
	}>;
}

type OpenAICompletionsStreamSimple = (
	model: Model<"openai-completions">,
	context: Context,
	options?: SimpleStreamOptions,
) => AssistantMessageEventStream;

type ProviderStreamSimple = NonNullable<ProviderConfig["streamSimple"]>;

function getEnvValue(env: DynamoEnvironment, key: keyof DynamoEnvironment): string | undefined {
	const value = env[key];
	const trimmed = value?.trim();
	return trimmed ? trimmed : undefined;
}

export function normalizeDynamoBaseUrl(rawBaseUrl: string | undefined): string {
	const raw = rawBaseUrl?.trim() || DEFAULT_DYNAMO_BASE_URL;
	const withoutTrailingSlash = raw.replace(/\/+$/, "");

	try {
		const url = new URL(withoutTrailingSlash);
		if (url.pathname === "" || url.pathname === "/") {
			url.pathname = "/v1";
		}
		return url.toString().replace(/\/+$/, "");
	} catch {
		return withoutTrailingSlash;
	}
}

export function readDynamoConfig(env: DynamoEnvironment = process.env): DynamoProviderRuntimeConfig {
	const sessionId = getEnvValue(env, "DYN_AGENT_SESSION_ID");
	let trajectoryId = getEnvValue(env, "DYN_AGENT_TRAJECTORY_ID");
	let parentTrajectoryId = getEnvValue(env, "DYN_AGENT_PARENT_TRAJECTORY_ID");

	// pi-subagents bridge. pi-subagents spawns each child agent as a child
	// node process with `{ ...process.env, ...subagentEnv }`, so the parent's
	// DYN_AGENT_TRAJECTORY_ID arrives in the child's env unchanged — under
	// the wrong name. When PI_SUBAGENT_CHILD=1, treat the inherited
	// trajectory id as the parent's, and synthesize a deterministic child
	// trajectory id from pi-subagents' (run_id, child_agent, child_index)
	// triple. Skipped if the caller already set DYN_AGENT_PARENT_TRAJECTORY_ID
	// explicitly so manual overrides win. See README "Subagent trajectory
	// linking" for the data flow and a worked example.
	if (
		!parentTrajectoryId &&
		getEnvValue(env, "PI_SUBAGENT_CHILD") === "1" &&
		trajectoryId !== undefined
	) {
		const piRunId = getEnvValue(env, "PI_SUBAGENT_RUN_ID");
		const piChildAgent = getEnvValue(env, "PI_SUBAGENT_CHILD_AGENT");
		const piChildIndex = getEnvValue(env, "PI_SUBAGENT_CHILD_INDEX") ?? "0";
		if (piRunId && piChildAgent) {
			parentTrajectoryId = trajectoryId;
			trajectoryId = `${piRunId}:${piChildAgent}:${piChildIndex}`;
		}
	}

	return {
		baseUrl: normalizeDynamoBaseUrl(getEnvValue(env, "DYNAMO_BASE_URL") ?? getEnvValue(env, "OPENAI_BASE_URL")),
		apiKey: getEnvValue(env, "DYNAMO_API_KEY") ?? DEFAULT_DYNAMO_API_KEY,
		sessionTypeId: getEnvValue(env, "DYN_AGENT_SESSION_TYPE_ID") ?? DEFAULT_SESSION_TYPE_ID,
		...(sessionId ? { sessionId } : {}),
		...(trajectoryId ? { trajectoryId } : {}),
		...(parentTrajectoryId ? { parentTrajectoryId } : {}),
	};
}

export function buildDynamoAgentContext(
	config: DynamoProviderRuntimeConfig,
	options?: Pick<SimpleStreamOptions, "sessionId">,
): DynamoAgentContext {
	const trajectoryId = config.trajectoryId ?? options?.sessionId;
	return {
		...(trajectoryId ? { trajectory_id: trajectoryId } : {}),
		...(config.parentTrajectoryId ? { parent_trajectory_id: config.parentTrajectoryId } : {}),
		...(config.sessionId ? { session_id: config.sessionId } : {}),
		session_type_id: config.sessionTypeId,
		phase: "reasoning",
	};
}

function isRecord(value: unknown): value is Record<string, unknown> {
	return typeof value === "object" && value !== null && !Array.isArray(value);
}

export function mergeDynamoAgentContext(payload: unknown, agentContext: DynamoAgentContext): unknown {
	const payloadRecord = isRecord(payload) ? payload : {};
	const existingNvext = isRecord(payloadRecord.nvext) ? payloadRecord.nvext : {};
	const existingAgentContext = isRecord(existingNvext.agent_context) ? existingNvext.agent_context : {};

	return {
		...payloadRecord,
		nvext: {
			...existingNvext,
			agent_context: {
				...agentContext,
				...existingAgentContext,
			},
		},
	};
}

function hasHeader(headers: Record<string, string>, target: string): boolean {
	const normalizedTarget = target.toLowerCase();
	return Object.keys(headers).some((key) => key.toLowerCase() === normalizedTarget);
}

export function buildDynamoHeaders(
	headers: Record<string, string> | undefined,
	createRequestId: () => string = randomUUID,
): Record<string, string> {
	const nextHeaders = { ...headers };
	if (!hasHeader(nextHeaders, "x-request-id")) {
		nextHeaders["x-request-id"] = createRequestId();
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

export async function discoverDynamoModels(
	config: DynamoProviderRuntimeConfig,
	options: { timeoutMs?: number } = {},
): Promise<ProviderModelConfig[]> {
	const controller = new AbortController();
	const timeout = setTimeout(() => controller.abort(), options.timeoutMs ?? 2000);
	try {
		const response = await fetch(`${config.baseUrl}/models`, {
			headers: {
				Authorization: `Bearer ${config.apiKey}`,
			},
			signal: controller.signal,
		});
		if (!response.ok) {
			return [];
		}

		const body = (await response.json()) as OpenAIModelsResponse;
		const modelIds =
			body.data
				?.map((model) => model.id)
				.filter((id): id is string => typeof id === "string" && id.length > 0) ?? [];
		return createDynamoModels([...new Set(modelIds)], config.baseUrl);
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
	config: DynamoProviderRuntimeConfig,
	delegate: OpenAICompletionsStreamSimple = streamSimpleOpenAICompletions,
	createRequestId: () => string = randomUUID,
): ProviderStreamSimple {
	return (model: Model<Api>, context: Context, options?: SimpleStreamOptions): AssistantMessageEventStream => {
		const agentContext = buildDynamoAgentContext(config, options);
		const openAIModel = toOpenAICompletionsModel(model);
		const previousOnPayload = options?.onPayload;
		const headers = buildDynamoHeaders(options?.headers, createRequestId);

		return delegate(openAIModel, context, {
			...options,
			apiKey: options?.apiKey ?? config.apiKey,
			headers,
			onPayload: async (payload) => {
				const injectedPayload = mergeDynamoAgentContext(payload, agentContext);
				return (await previousOnPayload?.(injectedPayload, model)) ?? injectedPayload;
			},
		});
	};
}

export function createDynamoProviderConfig(
	config: DynamoProviderRuntimeConfig,
	models: ProviderModelConfig[],
): ProviderConfig {
	return {
		name: "Dynamo",
		baseUrl: config.baseUrl,
		apiKey: config.apiKey,
		api: DYNAMO_API,
		models,
		streamSimple: createDynamoStreamSimple(config),
	};
}
