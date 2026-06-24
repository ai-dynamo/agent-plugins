// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

export interface DynamoSessionEnvironment {
	DYN_REQUEST_TRACE?: string;
	DYN_AGENT_SESSION_ID?: string;
	DYN_AGENT_PARENT_SESSION_ID?: string;
	PI_SUBAGENT_CHILD?: string;
	PI_SUBAGENT_RUN_ID?: string;
	PI_SUBAGENT_CHILD_AGENT?: string;
	PI_SUBAGENT_CHILD_INDEX?: string;
}

export interface DynamoSessionContext {
	sessionId?: string;
	parentSessionId?: string;
}

export function envValue<T extends object, K extends keyof T>(env: T, key: K): string | undefined {
	const value = env[key];
	const trimmed = typeof value === "string" ? value.trim() : undefined;
	return trimmed ? trimmed : undefined;
}

export function isTruthyEnv(value: string | undefined): boolean {
	return value ? ["1", "true", "yes", "on"].includes(value.toLowerCase()) : false;
}

export function subagentSessionId(env: DynamoSessionEnvironment): string | undefined {
	if (envValue(env, "PI_SUBAGENT_CHILD") !== "1") return undefined;
	const runId = envValue(env, "PI_SUBAGENT_RUN_ID");
	const childAgent = envValue(env, "PI_SUBAGENT_CHILD_AGENT");
	if (!runId || !childAgent) return undefined;
	return `${runId}:${childAgent}:${envValue(env, "PI_SUBAGENT_CHILD_INDEX") ?? "0"}`;
}

export function resolveSessionContext(env: DynamoSessionEnvironment): DynamoSessionContext {
	const childSessionId = subagentSessionId(env);
	if (childSessionId) {
		const explicitParentSessionId = envValue(env, "DYN_AGENT_PARENT_SESSION_ID");
		const inheritedSessionId = envValue(env, "DYN_AGENT_SESSION_ID");
		const parentCandidate = explicitParentSessionId ?? inheritedSessionId;
		const parentSessionId = parentCandidate !== childSessionId ? parentCandidate : undefined;
		return {
			sessionId: childSessionId,
			...(parentSessionId ? { parentSessionId } : {}),
		};
	}

	const sessionId = envValue(env, "DYN_AGENT_SESSION_ID");
	const parentSessionId = envValue(env, "DYN_AGENT_PARENT_SESSION_ID");
	return {
		...(sessionId ? { sessionId } : {}),
		...(parentSessionId ? { parentSessionId } : {}),
	};
}

export function applySubagentSessionBridge(env: NodeJS.ProcessEnv = process.env): boolean {
	const context = resolveSessionContext(env);
	if (!subagentSessionId(env) || !context.sessionId) return false;
	if (
		envValue(env, "DYN_AGENT_SESSION_ID") === context.sessionId &&
		envValue(env, "DYN_AGENT_PARENT_SESSION_ID") === context.parentSessionId
	) {
		return false;
	}
	if (context.parentSessionId) {
		env.DYN_AGENT_PARENT_SESSION_ID = context.parentSessionId;
	} else {
		delete env.DYN_AGENT_PARENT_SESSION_ID;
	}
	env.DYN_AGENT_SESSION_ID = context.sessionId;
	return true;
}
