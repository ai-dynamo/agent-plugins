// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

export interface DynamoTrajectoryEnvironment {
	DYN_REQUEST_TRACE?: string;
	DYN_AGENT_TRAJECTORY_ID?: string;
	DYN_AGENT_PARENT_TRAJECTORY_ID?: string;
	PI_SUBAGENT_CHILD?: string;
	PI_SUBAGENT_RUN_ID?: string;
	PI_SUBAGENT_CHILD_AGENT?: string;
	PI_SUBAGENT_CHILD_INDEX?: string;
}

export interface DynamoTrajectoryContext {
	trajectoryId?: string;
	parentTrajectoryId?: string;
}

export function envValue<T extends object, K extends keyof T>(env: T, key: K): string | undefined {
	const value = env[key];
	const trimmed = typeof value === "string" ? value.trim() : undefined;
	return trimmed ? trimmed : undefined;
}

export function isTruthyEnv(value: string | undefined): boolean {
	return value ? ["1", "true", "yes", "on"].includes(value.toLowerCase()) : false;
}

export function subagentTrajectoryId(env: DynamoTrajectoryEnvironment): string | undefined {
	if (envValue(env, "PI_SUBAGENT_CHILD") !== "1") return undefined;
	const runId = envValue(env, "PI_SUBAGENT_RUN_ID");
	const childAgent = envValue(env, "PI_SUBAGENT_CHILD_AGENT");
	if (!runId || !childAgent) return undefined;
	return `${runId}:${childAgent}:${envValue(env, "PI_SUBAGENT_CHILD_INDEX") ?? "0"}`;
}

export function resolveTrajectoryContext(env: DynamoTrajectoryEnvironment): DynamoTrajectoryContext {
	const childTrajectoryId = subagentTrajectoryId(env);
	if (childTrajectoryId) {
		const parentTrajectoryId =
			envValue(env, "DYN_AGENT_PARENT_TRAJECTORY_ID") ?? envValue(env, "DYN_AGENT_TRAJECTORY_ID");
		return {
			trajectoryId: childTrajectoryId,
			...(parentTrajectoryId ? { parentTrajectoryId } : {}),
		};
	}

	const trajectoryId = envValue(env, "DYN_AGENT_TRAJECTORY_ID");
	const parentTrajectoryId = envValue(env, "DYN_AGENT_PARENT_TRAJECTORY_ID");
	return {
		...(trajectoryId ? { trajectoryId } : {}),
		...(parentTrajectoryId ? { parentTrajectoryId } : {}),
	};
}

export function applySubagentTrajectoryBridge(env: NodeJS.ProcessEnv = process.env): boolean {
	const context = resolveTrajectoryContext(env);
	if (!subagentTrajectoryId(env) || !context.trajectoryId) return false;
	if (
		envValue(env, "DYN_AGENT_TRAJECTORY_ID") === context.trajectoryId &&
		envValue(env, "DYN_AGENT_PARENT_TRAJECTORY_ID") === context.parentTrajectoryId
	) {
		return false;
	}
	if (context.parentTrajectoryId) env.DYN_AGENT_PARENT_TRAJECTORY_ID = context.parentTrajectoryId;
	env.DYN_AGENT_TRAJECTORY_ID = context.trajectoryId;
	return true;
}
