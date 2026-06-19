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
	// Master switch for provider request-trace emissions. When truthy the
	// provider uses Pi's session_id affinity header for LLM requests and enables
	// the tool-event relay when an endpoint is configured. When unset/falsy the
	// provider is just a plain `dynamo/<model>` provider.
	DYN_REQUEST_TRACE?: string;
	DYN_AGENT_SESSION_TYPE_ID?: string;
	DYN_AGENT_SESSION_ID?: string;
	DYN_AGENT_TRAJECTORY_ID?: string;
	DYN_AGENT_PARENT_TRAJECTORY_ID?: string;
	// pi-subagents bookkeeping. pi-subagents spawns each child agent as a
	// node child_process with `{ ...process.env, ...subagentEnv }`, so the
	// parent's DYN_AGENT_TRAJECTORY_ID arrives in the child unchanged —
	// under the wrong name. The bridge below reinterprets it. See
	// `applySubagentBridge` and the README "Subagent trajectory linking".
	PI_SUBAGENT_CHILD?: string;
	PI_SUBAGENT_RUN_ID?: string;
	PI_SUBAGENT_CHILD_AGENT?: string;
	PI_SUBAGENT_CHILD_INDEX?: string;
}

export interface DynamoProviderRuntimeConfig {
	baseUrl: string;
	apiKey: string;
	// DYN_REQUEST_TRACE master switch. Gates session affinity headers and the
	// tool relay; the model provider itself is registered regardless.
	traceEnabled: boolean;
	sessionTypeId: string;
	sessionId?: string;
	trajectoryId?: string;
	parentTrajectoryId?: string;
	isSubagent?: boolean;
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

function isTruthyEnv(value: string | undefined): boolean {
	if (!value) return false;
	return ["1", "true", "yes", "on"].includes(value.toLowerCase());
}

/**
 * The subagent's stable trajectory id, derived purely from the pi-subagents
 * `PI_SUBAGENT_*` bookkeeping. Returns `undefined` outside a pi-subagents child
 * or when the identity is incomplete (no run id / agent name).
 *
 * `PI_SUBAGENT_CHILD_INDEX` defaults to `"0"` when absent.
 */
export function computeSubagentTrajectoryId(env: DynamoEnvironment): string | undefined {
	if (getEnvValue(env, "PI_SUBAGENT_CHILD") !== "1") return undefined;
	const runId = getEnvValue(env, "PI_SUBAGENT_RUN_ID");
	const childAgent = getEnvValue(env, "PI_SUBAGENT_CHILD_AGENT");
	if (!runId || !childAgent) return undefined;
	const childIndex = getEnvValue(env, "PI_SUBAGENT_CHILD_INDEX") ?? "0";
	return `${runId}:${childAgent}:${childIndex}`;
}

/**
 * Compute the trajectory rewrite that pi-subagents inheritance implies, without
 * mutating any caller-visible state. Pure: takes the raw env, returns either
 * `null` (not a pi-subagents child) or the child's trajectory id plus optional
 * parent id.
 *
 * `PI_SUBAGENT_CHILD === "1"` switches the current process identity from root
 * to child: `trajectory_id` becomes the child id, and the inherited
 * `DYN_AGENT_TRAJECTORY_ID` becomes `parent_trajectory_id` unless an explicit
 * parent override is already present.
 */
export function computeSubagentTrajectoryRewrite(
	env: DynamoEnvironment,
): { trajectoryId: string; parentTrajectoryId?: string } | null {
	const trajectoryId = computeSubagentTrajectoryId(env);
	if (!trajectoryId) return null;
	const parentTrajectoryId =
		getEnvValue(env, "DYN_AGENT_PARENT_TRAJECTORY_ID") ?? getEnvValue(env, "DYN_AGENT_TRAJECTORY_ID");
	return {
		trajectoryId,
		...(parentTrajectoryId ? { parentTrajectoryId } : {}),
	};
}

/**
 * Apply the pi-subagents trajectory rewrite to `process.env` so subsequent
 * pi-subagents spawns inherit this generation's synthesized trajectory_id as
 * their parent. Without this, nested subagent chains collapse — every
 * generation would observe the original grandparent as its parent and the
 * middle generations would be invisible in the dynamo trace.
 *
 * Idempotent: a second call has no effect once the env already contains the
 * computed child trajectory and parent link. Safe to invoke from extension init.
 *
 * Mutates the supplied env object in place (defaults to `process.env`); also
 * returns whether a rewrite was applied so callers can log/test.
 */
export function applySubagentBridge(env: NodeJS.ProcessEnv = process.env): boolean {
	const rewrite = computeSubagentTrajectoryRewrite(env);
	if (!rewrite) return false;
	if (
		getEnvValue(env, "DYN_AGENT_TRAJECTORY_ID") === rewrite.trajectoryId &&
		getEnvValue(env, "DYN_AGENT_PARENT_TRAJECTORY_ID") === rewrite.parentTrajectoryId
	) {
		return false;
	}
	if (rewrite.parentTrajectoryId) env.DYN_AGENT_PARENT_TRAJECTORY_ID = rewrite.parentTrajectoryId;
	env.DYN_AGENT_TRAJECTORY_ID = rewrite.trajectoryId;
	return true;
}

export function readDynamoConfig(env: DynamoEnvironment = process.env): DynamoProviderRuntimeConfig {
	const rewrite = computeSubagentTrajectoryRewrite(env);
	const sessionId = getEnvValue(env, "DYN_AGENT_SESSION_ID");
	const trajectoryId = rewrite?.trajectoryId ?? getEnvValue(env, "DYN_AGENT_TRAJECTORY_ID");
	const parentTrajectoryId =
		rewrite?.parentTrajectoryId ?? getEnvValue(env, "DYN_AGENT_PARENT_TRAJECTORY_ID");

	return {
		baseUrl: normalizeDynamoBaseUrl(getEnvValue(env, "DYNAMO_BASE_URL") ?? getEnvValue(env, "OPENAI_BASE_URL")),
		apiKey: getEnvValue(env, "DYNAMO_API_KEY") ?? DEFAULT_DYNAMO_API_KEY,
		traceEnabled: isTruthyEnv(getEnvValue(env, "DYN_REQUEST_TRACE")),
		sessionTypeId: getEnvValue(env, "DYN_AGENT_SESSION_TYPE_ID") ?? DEFAULT_SESSION_TYPE_ID,
		...(sessionId ? { sessionId } : {}),
		...(trajectoryId ? { trajectoryId } : {}),
		...(parentTrajectoryId ? { parentTrajectoryId } : {}),
		isSubagent: rewrite !== null,
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
	sendSessionAffinityHeaders: true,
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
		const runtimeSessionId = options?.sessionId?.trim();
		const openAIModel = toOpenAICompletionsModel(model);
		const sessionId =
			config.traceEnabled && config.isSubagent ? (config.trajectoryId ?? runtimeSessionId) : runtimeSessionId;
		if (!config.sessionId && sessionId) {
			config.sessionId = sessionId;
		}
		const headers = buildDynamoHeaders(options?.headers, createRequestId);
		const baseOptions: SimpleStreamOptions = {
			...options,
			...(sessionId ? { sessionId } : {}),
			apiKey: options?.apiKey ?? config.apiKey,
			headers,
		};
		return delegate(openAIModel, context, baseOptions);
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
