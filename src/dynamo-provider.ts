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
	// Master switch for the provider's agentic emissions. When truthy the
	// provider injects nvext.agent_context, drives subagent session_control, and
	// (if an endpoint is set) the tool-event relay — all with sensible defaults
	// that the more specific DYN_AGENT_* / DYNAMO_* vars below override. When
	// unset/falsy the provider is just a plain `dynamo/<model>` provider.
	DYN_AGENT_TRACE?: string;
	DYN_AGENT_SESSION_TYPE_ID?: string;
	DYN_AGENT_SESSION_ID?: string;
	DYN_AGENT_TRAJECTORY_ID?: string;
	DYN_AGENT_PARENT_TRAJECTORY_ID?: string;
	// Per-session inactivity timeout (seconds) sent on the streaming-session
	// open. Omitted when unset so Dynamo applies its own 300s default.
	DYN_AGENT_SESSION_TIMEOUT?: string;
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
	// DYN_AGENT_TRACE master switch. Gates agent_context, session_control, and
	// the tool relay; the model provider itself is registered regardless.
	traceEnabled: boolean;
	sessionTypeId: string;
	sessionId?: string;
	trajectoryId?: string;
	parentTrajectoryId?: string;
	// Streaming-session id for subagent KV isolation. Set whenever this process
	// is a pi-subagents child (derived from PI_SUBAGENT_* alone — independent of
	// the trajectory bridge). One child process == one Dynamo streaming session.
	sessionControlId?: string;
	// Inactivity timeout (seconds) for that session, when DYN_AGENT_SESSION_TIMEOUT
	// is set. Omitted otherwise so Dynamo's 300s default applies.
	sessionTimeoutSecs?: number;
}

// nvext.session_control: sticky routing + SGLang streaming-session KV isolation
// for subagents. Sibling of agent_context on the request payload. action is
// omitted on intermediate turns; timeout only matters on the open.
export interface DynamoSessionControl {
	session_id: string;
	action?: "open" | "close";
	timeout?: number;
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

function isTruthyEnv(value: string | undefined): boolean {
	if (!value) return false;
	return ["1", "true", "yes", "on"].includes(value.toLowerCase());
}

/**
 * The subagent's stable, unique streaming-session id, derived purely from the
 * pi-subagents `PI_SUBAGENT_*` bookkeeping. This is the natural key for KV
 * isolation and is deliberately independent of `DYN_AGENT_TRAJECTORY_ID` /
 * tracing — a subagent gets its own session whether or not anyone set up a
 * trajectory lineage. Returns `undefined` outside a pi-subagents child or when
 * the identity is incomplete (no run id / agent name).
 *
 * `PI_SUBAGENT_CHILD_INDEX` defaults to `"0"` when absent.
 */
export function computeSubagentSessionId(env: DynamoEnvironment): string | undefined {
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
 * `null` (no rewrite applies) or `{ trajectoryId, parentTrajectoryId }` where
 * `trajectoryId` is the synthesized child id and `parentTrajectoryId` is the
 * inherited parent id.
 *
 * This is the TRACING bridge — distinct from `computeSubagentSessionId`. It
 * fires only when ALL of these hold:
 *   - `PI_SUBAGENT_CHILD === "1"` (this process was spawned by pi-subagents)
 *   - inherited `DYN_AGENT_TRAJECTORY_ID` is set (the parent's id we want to
 *     reinterpret as this child's parent)
 *   - `DYN_AGENT_PARENT_TRAJECTORY_ID` is NOT already set (manual override wins)
 *   - the subagent identity is complete (`computeSubagentSessionId` resolves)
 */
export function computeSubagentTrajectoryRewrite(
	env: DynamoEnvironment,
): { trajectoryId: string; parentTrajectoryId: string } | null {
	if (getEnvValue(env, "DYN_AGENT_PARENT_TRAJECTORY_ID")) return null;
	const inherited = getEnvValue(env, "DYN_AGENT_TRAJECTORY_ID");
	if (!inherited) return null;
	const trajectoryId = computeSubagentSessionId(env);
	if (!trajectoryId) return null;
	return { parentTrajectoryId: inherited, trajectoryId };
}

/**
 * Apply the pi-subagents trajectory rewrite to `process.env` so subsequent
 * pi-subagents spawns inherit this generation's synthesized trajectory_id as
 * their parent. Without this, nested subagent chains collapse — every
 * generation would observe the original grandparent as its parent and the
 * middle generations would be invisible in the dynamo trace.
 *
 * Idempotent: a second call has no effect because the rewrite condition
 * requires `DYN_AGENT_PARENT_TRAJECTORY_ID` to be absent, which it isn't after
 * the first call. Safe to invoke from extension init.
 *
 * Mutates the supplied env object in place (defaults to `process.env`); also
 * returns whether a rewrite was applied so callers can log/test.
 */
export function applySubagentBridge(env: NodeJS.ProcessEnv = process.env): boolean {
	const rewrite = computeSubagentTrajectoryRewrite(env);
	if (!rewrite) return false;
	env.DYN_AGENT_PARENT_TRAJECTORY_ID = rewrite.parentTrajectoryId;
	env.DYN_AGENT_TRAJECTORY_ID = rewrite.trajectoryId;
	return true;
}

function parsePositiveIntOrUndefined(value: string | undefined): number | undefined {
	if (value === undefined) return undefined;
	const parsed = Number.parseInt(value, 10);
	return Number.isFinite(parsed) && parsed > 0 ? parsed : undefined;
}

export function readDynamoConfig(env: DynamoEnvironment = process.env): DynamoProviderRuntimeConfig {
	const rewrite = computeSubagentTrajectoryRewrite(env);
	const sessionId = getEnvValue(env, "DYN_AGENT_SESSION_ID");
	const trajectoryId = rewrite?.trajectoryId ?? getEnvValue(env, "DYN_AGENT_TRAJECTORY_ID");
	const parentTrajectoryId =
		rewrite?.parentTrajectoryId ?? getEnvValue(env, "DYN_AGENT_PARENT_TRAJECTORY_ID");
	// Only a pi-subagents child gets a streaming session, keyed on the subagent's
	// own identity — independent of the trajectory bridge, so the lead agent
	// stays load-balanced while each subagent pins regardless of DYN_AGENT_*.
	const sessionControlId = computeSubagentSessionId(env);
	const sessionTimeoutSecs = parsePositiveIntOrUndefined(getEnvValue(env, "DYN_AGENT_SESSION_TIMEOUT"));

	return {
		baseUrl: normalizeDynamoBaseUrl(getEnvValue(env, "DYNAMO_BASE_URL") ?? getEnvValue(env, "OPENAI_BASE_URL")),
		apiKey: getEnvValue(env, "DYNAMO_API_KEY") ?? DEFAULT_DYNAMO_API_KEY,
		traceEnabled: isTruthyEnv(getEnvValue(env, "DYN_AGENT_TRACE")),
		sessionTypeId: getEnvValue(env, "DYN_AGENT_SESSION_TYPE_ID") ?? DEFAULT_SESSION_TYPE_ID,
		...(sessionId ? { sessionId } : {}),
		...(trajectoryId ? { trajectoryId } : {}),
		...(parentTrajectoryId ? { parentTrajectoryId } : {}),
		...(sessionControlId ? { sessionControlId } : {}),
		...(sessionTimeoutSecs !== undefined ? { sessionTimeoutSecs } : {}),
	};
}

export function buildDynamoAgentContext(
	config: DynamoProviderRuntimeConfig,
	options?: Pick<SimpleStreamOptions, "sessionId">,
): DynamoAgentContext {
	// session_id and trajectory_id both default to Pi's own session id when not
	// pinned via DYN_AGENT_*. Dynamo's AgentContext requires session_id, so a
	// default keeps the payload valid with zero operator env beyond DYN_AGENT_TRACE.
	const trajectoryId = config.trajectoryId ?? options?.sessionId;
	const sessionId = config.sessionId ?? options?.sessionId;
	return {
		...(trajectoryId ? { trajectory_id: trajectoryId } : {}),
		...(config.parentTrajectoryId ? { parent_trajectory_id: config.parentTrajectoryId } : {}),
		...(sessionId ? { session_id: sessionId } : {}),
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

export function mergeDynamoSessionControl(payload: unknown, sessionControl: DynamoSessionControl): unknown {
	const payloadRecord = isRecord(payload) ? payload : {};
	const existingNvext = isRecord(payloadRecord.nvext) ? payloadRecord.nvext : {};
	const existingSessionControl = isRecord(existingNvext.session_control) ? existingNvext.session_control : {};

	return {
		...payloadRecord,
		nvext: {
			...existingNvext,
			session_control: {
				...sessionControl,
				...existingSessionControl,
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

type FetchLike = (input: string, init: RequestInit) => Promise<{ ok: boolean; status: number }>;

/**
 * One streaming session for one pi-subagents child process. Dynamo has no
 * standalone close RPC — close must ride a routed request — so the lifecycle is:
 *   - first turn  -> action "open" (+ timeout): the worker holds this subagent's
 *     KV in a dedicated slot outside the radix tree, invisible to eviction.
 *   - later turns -> bare session_id: the router pins them to the same worker
 *     (O(1) KV restore).
 *   - on agent_end -> a throwaway max_tokens=1 request carries action "close",
 *     freeing the KV deterministically instead of waiting out the idle timeout.
 * The idle timeout is the safety net if the process dies before close fires.
 */
export class DynamoSubagentSession {
	readonly sessionId: string;
	modelId = "";
	private readonly baseUrl: string;
	private readonly apiKey: string;
	private readonly timeoutSecs: number | undefined;
	private readonly createRequestId: () => string;
	private opened = false;

	constructor(
		config: Pick<DynamoProviderRuntimeConfig, "baseUrl" | "apiKey"> & {
			sessionControlId: string;
			sessionTimeoutSecs?: number;
		},
		createRequestId: () => string = randomUUID,
	) {
		this.sessionId = config.sessionControlId;
		this.baseUrl = config.baseUrl;
		this.apiKey = config.apiKey;
		this.timeoutSecs = config.sessionTimeoutSecs;
		this.createRequestId = createRequestId;
	}

	/** Build session_control for the current turn and advance lifecycle state. */
	controlForTurn(): DynamoSessionControl {
		const action = this.opened ? undefined : ("open" as const);
		this.opened = true;
		return {
			session_id: this.sessionId,
			...(action ? { action } : {}),
			...(this.timeoutSecs !== undefined ? { timeout: this.timeoutSecs } : {}),
		};
	}

	/**
	 * Fire-and-forget close. Best-effort: skipped if no turn opened the session,
	 * and any transport error is swallowed (KV cleanup is best-effort — never
	 * block Pi's shutdown on it). The 5s timeout bounds a hung frontend; the idle
	 * reaper covers the case where this never lands.
	 *
	 * Re-openable: clearing `opened` synchronously both guards a double-fire
	 * (agent_end then session_shutdown) and lets a later turn re-emit `action:
	 * "open"`. agent_end fires once per prompt, so a multi-prompt subagent frees
	 * its KV between prompts and re-warms a fresh session on the next one.
	 */
	async close(fetchImpl: FetchLike = fetch): Promise<boolean> {
		if (!this.opened) return false;
		this.opened = false;
		const sessionControl: DynamoSessionControl = { session_id: this.sessionId, action: "close" };
		const body = {
			model: this.modelId,
			messages: [{ role: "user", content: "." }],
			max_tokens: 1,
			stream: false,
			nvext: { session_control: sessionControl },
		};
		try {
			const response = await fetchImpl(`${this.baseUrl}/chat/completions`, {
				method: "POST",
				headers: {
					"content-type": "application/json",
					authorization: `Bearer ${this.apiKey}`,
					"x-request-id": this.createRequestId(),
				},
				body: JSON.stringify(body),
				signal: AbortSignal.timeout(5000),
			});
			return response.ok;
		} catch {
			return false;
		}
	}
}

export function createDynamoStreamSimple(
	config: DynamoProviderRuntimeConfig,
	delegate: OpenAICompletionsStreamSimple = streamSimpleOpenAICompletions,
	createRequestId: () => string = randomUUID,
	session?: DynamoSubagentSession,
): ProviderStreamSimple {
	return (model: Model<Api>, context: Context, options?: SimpleStreamOptions): AssistantMessageEventStream => {
		const openAIModel = toOpenAICompletionsModel(model);
		const headers = buildDynamoHeaders(options?.headers, createRequestId);
		const baseOptions: SimpleStreamOptions = {
			...options,
			apiKey: options?.apiKey ?? config.apiKey,
			headers,
		};

		// DYN_AGENT_TRACE off: behave as a plain dynamo/<model> provider — still
		// add x-request-id for correlation, but inject no agentic nvext.
		if (!config.traceEnabled) {
			return delegate(openAIModel, context, baseOptions);
		}

		const agentContext = buildDynamoAgentContext(config, options);
		const previousOnPayload = options?.onPayload;
		// Capture the live model id so the deferred close request targets a real
		// model, and advance the open->sticky transition once per turn.
		if (session) session.modelId = model.id;
		const sessionControl = session?.controlForTurn();

		return delegate(openAIModel, context, {
			...baseOptions,
			onPayload: async (payload) => {
				let injectedPayload = mergeDynamoAgentContext(payload, agentContext);
				if (sessionControl) injectedPayload = mergeDynamoSessionControl(injectedPayload, sessionControl);
				return (await previousOnPayload?.(injectedPayload, model)) ?? injectedPayload;
			},
		});
	};
}

export function createDynamoProviderConfig(
	config: DynamoProviderRuntimeConfig,
	models: ProviderModelConfig[],
	session?: DynamoSubagentSession,
): ProviderConfig {
	return {
		name: "Dynamo",
		baseUrl: config.baseUrl,
		apiKey: config.apiKey,
		api: DYNAMO_API,
		models,
		streamSimple: createDynamoStreamSimple(config, streamSimpleOpenAICompletions, randomUUID, session),
	};
}
