// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { Buffer } from "node:buffer";
import { encode } from "@msgpack/msgpack";
import type { ExtensionAPI, ExtensionContext } from "@mariozechner/pi-coding-agent";
import { Push } from "zeromq";
import type { DynamoEnvironment, DynamoProviderRuntimeConfig } from "./dynamo-provider.js";

export const DEFAULT_TOOL_EVENTS_TOPIC = "agent-tool-events";
export const DEFAULT_TOOL_EVENT_QUEUE_CAPACITY = 100000;

export interface DynamoToolRelayEnvironment extends DynamoEnvironment {
	DYN_REQUEST_TRACE_TOOL_EVENTS_ZMQ_ENDPOINT?: string;
	DYN_REQUEST_TRACE_TOOL_EVENTS_ZMQ_TOPIC?: string;
	DYN_REQUEST_TRACE_TOOL_EVENTS_QUEUE_CAPACITY?: string;
}

export interface DynamoToolRelayConfig {
	endpoint?: string;
	topic: string;
	queueCapacity: number;
}

export interface DynamoRequestTraceAgentContext {
	session_type_id: string;
	trajectory_id: string;
	parent_trajectory_id?: string;
}

export type DynamoToolStatus = "running" | "succeeded" | "error" | "cancelled";
export type DynamoToolTraceEventType = "tool_start" | "tool_end" | "tool_error";

export interface DynamoRequestTraceToolEvent {
	tool_call_id: string;
	tool_class: string;
	started_at_unix_ms?: number;
	ended_at_unix_ms?: number;
	status?: DynamoToolStatus;
	duration_ms?: number;
	output_bytes?: number;
	error_type?: string;
}

export interface DynamoRequestTraceRecord {
	schema: "dynamo.request.trace.v1";
	event_type: DynamoToolTraceEventType;
	event_time_unix_ms: number;
	event_source: "harness";
	agent_context: DynamoRequestTraceAgentContext;
	tool: DynamoRequestTraceToolEvent;
}

export interface ToolEventSocket {
	connect(endpoint: string): Promise<void> | void;
	send(frames: [Buffer, Buffer, Buffer]): Promise<void>;
	close(): void;
}

export type ToolEventSocketFactory = () => ToolEventSocket;

export interface PiToolExecutionStartEvent {
	toolCallId: string;
	toolName: string;
	args: unknown;
}

export interface PiToolExecutionEndEvent {
	toolCallId: string;
	toolName: string;
	result: unknown;
	isError: boolean;
}

interface ToolCallStart {
	agentContext: DynamoRequestTraceAgentContext;
	toolName: string;
	toolClass: string;
	startedAtUnixMs: number;
	startedAtPerfMs: number;
}

function isRecord(value: unknown): value is Record<string, unknown> {
	return typeof value === "object" && value !== null && !Array.isArray(value);
}

function getEnvValue(env: DynamoToolRelayEnvironment, key: keyof DynamoToolRelayEnvironment): string | undefined {
	const value = env[key];
	const trimmed = value?.trim();
	return trimmed ? trimmed : undefined;
}

function parsePositiveInteger(value: string | undefined, fallback: number): number {
	if (!value) return fallback;
	const parsed = Number.parseInt(value, 10);
	return Number.isFinite(parsed) && parsed > 0 ? parsed : fallback;
}

export function readDynamoToolRelayConfig(env: DynamoToolRelayEnvironment = process.env): DynamoToolRelayConfig {
	const endpoint = getEnvValue(env, "DYN_REQUEST_TRACE_TOOL_EVENTS_ZMQ_ENDPOINT");

	return {
		...(endpoint ? { endpoint } : {}),
		topic: getEnvValue(env, "DYN_REQUEST_TRACE_TOOL_EVENTS_ZMQ_TOPIC") ?? DEFAULT_TOOL_EVENTS_TOPIC,
		queueCapacity: parsePositiveInteger(
			getEnvValue(env, "DYN_REQUEST_TRACE_TOOL_EVENTS_QUEUE_CAPACITY"),
			DEFAULT_TOOL_EVENT_QUEUE_CAPACITY,
		),
	};
}

export function buildDynamoRequestTraceAgentContext(
	config: DynamoProviderRuntimeConfig,
	sessionId: string | undefined,
): DynamoRequestTraceAgentContext | undefined {
	const trajectoryId = config.trajectoryId ?? sessionId;
	if (!trajectoryId) return undefined;

	return {
		session_type_id: config.sessionTypeId,
		trajectory_id: trajectoryId,
		...(config.parentTrajectoryId ? { parent_trajectory_id: config.parentTrajectoryId } : {}),
	};
}

export function getToolClass(toolName: string | undefined): string {
	const name = toolName?.trim();
	if (!name) return "unknown";
	return name.split("---", 1)[0]?.split("/", 1)[0] || "unknown";
}

export function getToolResultOutputBytes(result: unknown): number | undefined {
	if (!isRecord(result) || !Array.isArray(result.content)) {
		return undefined;
	}

	const output = result.content
		.map((item) => {
			if (isRecord(item) && typeof item.text === "string") {
				return item.text;
			}
			return JSON.stringify(item);
		})
		.join("\n");
	return Buffer.byteLength(output, "utf8");
}

function createSequenceFrame(sequence: bigint): Buffer {
	const frame = Buffer.alloc(8);
	frame.writeBigUInt64BE(sequence);
	return frame;
}

export function createZeroMqPushSocket(): ToolEventSocket {
	const socket = new Push({ sendHighWaterMark: DEFAULT_TOOL_EVENT_QUEUE_CAPACITY, linger: 0 });
	return {
		connect: (endpoint) => socket.connect(endpoint),
		send: (frames) => socket.send(frames),
		close: () => socket.close(),
	};
}

export class DynamoToolEventPublisher {
	private readonly topicFrame: Buffer;
	private readonly socket: ToolEventSocket;
	private sequence = 0n;
	private queued = 0;
	private closed = false;
	private sendChain: Promise<void> = Promise.resolve();

	constructor(
		private readonly config: DynamoToolRelayConfig,
		socketFactory: ToolEventSocketFactory = createZeroMqPushSocket,
	) {
		this.topicFrame = Buffer.from(config.topic, "utf8");
		this.socket = socketFactory();
	}

	async start(): Promise<void> {
		if (!this.config.endpoint) return;
		await this.socket.connect(this.config.endpoint);
	}

	publish(record: DynamoRequestTraceRecord): boolean {
		if (this.closed || !this.config.endpoint) return false;
		if (this.queued >= this.config.queueCapacity) return false;

		const frames: [Buffer, Buffer, Buffer] = [
			this.topicFrame,
			createSequenceFrame(this.sequence),
			Buffer.from(encode(record)),
		];
		this.sequence += 1n;
		this.queued += 1;

		this.sendChain = this.sendChain
			.catch(() => undefined)
			.then(() => this.socket.send(frames))
			.catch(() => undefined)
			.finally(() => {
				this.queued -= 1;
			});
		return true;
	}

	async flush(): Promise<void> {
		await this.sendChain;
	}

	close(): void {
		if (this.closed) return;
		this.closed = true;
		this.socket.close();
	}
}

export class DynamoToolEventRelay {
	private readonly starts = new Map<string, ToolCallStart>();

	constructor(
		private readonly config: DynamoProviderRuntimeConfig,
		private readonly publisher: DynamoToolEventPublisher,
		private readonly nowUnixMs: () => number = () => Date.now(),
		private readonly nowPerfMs: () => number = () => performance.now(),
	) {}

	handleToolExecutionStart(event: PiToolExecutionStartEvent, ctx: ExtensionContext): void {
		const agentContext = buildDynamoRequestTraceAgentContext(this.config, ctx.sessionManager.getSessionId());
		if (!agentContext) return;

		const startedAtUnixMs = this.nowUnixMs();
		const toolClass = getToolClass(event.toolName);
		this.starts.set(event.toolCallId, {
			agentContext,
			toolName: event.toolName,
			toolClass,
			startedAtUnixMs,
			startedAtPerfMs: this.nowPerfMs(),
		});

		this.publisher.publish({
			schema: "dynamo.request.trace.v1",
			event_type: "tool_start",
			event_time_unix_ms: startedAtUnixMs,
			event_source: "harness",
			agent_context: agentContext,
			tool: {
				tool_call_id: event.toolCallId,
				tool_class: toolClass,
				started_at_unix_ms: startedAtUnixMs,
				status: "running",
			},
		});
	}

	handleToolExecutionEnd(event: PiToolExecutionEndEvent, ctx: ExtensionContext): void {
		const endedAtUnixMs = this.nowUnixMs();
		const endedAtPerfMs = this.nowPerfMs();
		const start = this.starts.get(event.toolCallId);
		this.starts.delete(event.toolCallId);

		const agentContext =
			start?.agentContext ?? buildDynamoRequestTraceAgentContext(this.config, ctx.sessionManager.getSessionId());
		if (!agentContext) return;

		const startedAtUnixMs = start?.startedAtUnixMs ?? endedAtUnixMs;
		const durationMs =
			start === undefined ? 0 : Math.max(0, Math.round((endedAtPerfMs - start.startedAtPerfMs) * 1000) / 1000);
		const status: DynamoToolStatus = event.isError ? "error" : "succeeded";
		const toolClass = start?.toolClass ?? getToolClass(event.toolName);
		const outputBytes = getToolResultOutputBytes(event.result);

		this.publisher.publish({
			schema: "dynamo.request.trace.v1",
			event_type: event.isError ? "tool_error" : "tool_end",
			event_time_unix_ms: endedAtUnixMs,
			event_source: "harness",
			agent_context: agentContext,
			tool: {
				tool_call_id: event.toolCallId,
				tool_class: toolClass,
				started_at_unix_ms: startedAtUnixMs,
				ended_at_unix_ms: endedAtUnixMs,
				duration_ms: durationMs,
				status,
				...(event.isError ? { error_type: "pi_tool_error" } : {}),
				...(outputBytes === undefined ? {} : { output_bytes: outputBytes }),
			},
		});
	}
}

export async function registerDynamoToolEventRelay(
	pi: ExtensionAPI,
	config: DynamoProviderRuntimeConfig,
	relayConfig: DynamoToolRelayConfig = readDynamoToolRelayConfig(),
	socketFactory: ToolEventSocketFactory = createZeroMqPushSocket,
): Promise<DynamoToolEventRelay | undefined> {
	if (!relayConfig.endpoint) return undefined;

	const publisher = new DynamoToolEventPublisher(relayConfig, socketFactory);
	await publisher.start();
	const relay = new DynamoToolEventRelay(config, publisher);

	pi.on("tool_execution_start", (event, ctx) => {
		relay.handleToolExecutionStart(event, ctx);
	});
	pi.on("tool_execution_end", (event, ctx) => {
		relay.handleToolExecutionEnd(event, ctx);
	});
	pi.on("session_shutdown", () => {
		publisher.close();
	});

	return relay;
}
