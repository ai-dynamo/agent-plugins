// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { Buffer } from "node:buffer";
import { encode } from "@msgpack/msgpack";
import type { ExtensionAPI, ExtensionContext } from "@mariozechner/pi-coding-agent";
import { Push } from "zeromq";
import type { DynamoConfig, DynamoEnvironment } from "./provider.js";
import { envValue } from "./session.js";

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
	session_id: string;
	parent_session_id?: string;
}

type ToolTraceEventType = "tool_start" | "tool_end" | "tool_error";
type ToolStatus = "running" | "succeeded" | "error";

export interface DynamoRequestTraceRecord {
	schema: "dynamo.request.trace.v1";
	event_type: ToolTraceEventType;
	event_time_unix_ms: number;
	event_source: "harness";
	agent_context: DynamoRequestTraceAgentContext;
	tool: {
		tool_call_id: string;
		tool_class: string;
		started_at_unix_ms?: number;
		ended_at_unix_ms?: number;
		status?: ToolStatus;
		duration_ms?: number;
		output_bytes?: number;
		error_type?: string;
	};
}

export interface ToolEventSocket {
	connect(endpoint: string): Promise<void> | void;
	send(frames: [Buffer, Buffer, Buffer]): Promise<void>;
	close(): void;
}

export type ToolEventSocketFactory = () => ToolEventSocket;

interface ToolStart {
	agentContext: DynamoRequestTraceAgentContext;
	toolClass: string;
	startedAtUnixMs: number;
	startedAtPerfMs: number;
}

function parsePositiveInteger(value: string | undefined, fallback: number): number {
	const parsed = value ? Number.parseInt(value, 10) : Number.NaN;
	return Number.isFinite(parsed) && parsed > 0 ? parsed : fallback;
}

export function readDynamoToolRelayConfig(env: DynamoToolRelayEnvironment = process.env): DynamoToolRelayConfig {
	const endpoint = envValue(env, "DYN_REQUEST_TRACE_TOOL_EVENTS_ZMQ_ENDPOINT");
	return {
		...(endpoint ? { endpoint } : {}),
		topic: envValue(env, "DYN_REQUEST_TRACE_TOOL_EVENTS_ZMQ_TOPIC") ?? DEFAULT_TOOL_EVENTS_TOPIC,
		queueCapacity: parsePositiveInteger(
			envValue(env, "DYN_REQUEST_TRACE_TOOL_EVENTS_QUEUE_CAPACITY"),
			DEFAULT_TOOL_EVENT_QUEUE_CAPACITY,
		),
	};
}

export function buildToolAgentContext(config: DynamoConfig, runtimeSessionId: string | undefined) {
	const sessionId = config.sessionId ?? runtimeSessionId;
	if (!sessionId) return undefined;
	return {
		session_id: sessionId,
		...(config.parentSessionId ? { parent_session_id: config.parentSessionId } : {}),
	};
}

export function getToolClass(toolName: string | undefined): string {
	const name = toolName?.trim();
	return name ? name.split("---", 1)[0]?.split("/", 1)[0] || "unknown" : "unknown";
}

function outputBytes(result: unknown): number | undefined {
	if (typeof result !== "object" || result === null || !("content" in result) || !Array.isArray(result.content)) {
		return undefined;
	}
	const text = result.content
		.map((item: unknown) =>
			typeof item === "object" && item !== null && "text" in item && typeof item.text === "string"
				? item.text
				: JSON.stringify(item),
		)
		.join("\n");
	return Buffer.byteLength(text, "utf8");
}

function sequenceFrame(sequence: bigint): Buffer {
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
		if (this.config.endpoint) await this.socket.connect(this.config.endpoint);
	}

	publish(record: DynamoRequestTraceRecord): boolean {
		if (this.closed || !this.config.endpoint || this.queued >= this.config.queueCapacity) return false;
		const frames: [Buffer, Buffer, Buffer] = [
			this.topicFrame,
			sequenceFrame(this.sequence),
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
	private readonly starts = new Map<string, ToolStart>();

	constructor(
		private readonly config: DynamoConfig,
		private readonly publisher: DynamoToolEventPublisher,
		private readonly nowUnixMs: () => number = () => Date.now(),
		private readonly nowPerfMs: () => number = () => performance.now(),
	) {}

	handleToolExecutionStart(
		event: { toolCallId: string; toolName: string; args: unknown },
		ctx: ExtensionContext,
	): void {
		const agentContext = buildToolAgentContext(this.config, ctx.sessionManager.getSessionId());
		if (!agentContext) return;
		const startedAtUnixMs = this.nowUnixMs();
		const toolClass = getToolClass(event.toolName);
		this.starts.set(event.toolCallId, {
			agentContext,
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

	handleToolExecutionEnd(
		event: { toolCallId: string; toolName: string; result: unknown; isError: boolean },
		ctx: ExtensionContext,
	): void {
		const endedAtUnixMs = this.nowUnixMs();
		const endedAtPerfMs = this.nowPerfMs();
		const start = this.starts.get(event.toolCallId);
		this.starts.delete(event.toolCallId);
		const agentContext = start?.agentContext ?? buildToolAgentContext(this.config, ctx.sessionManager.getSessionId());
		if (!agentContext) return;
		const startedAtUnixMs = start?.startedAtUnixMs ?? endedAtUnixMs;
		const bytes = outputBytes(event.result);
		this.publisher.publish({
			schema: "dynamo.request.trace.v1",
			event_type: event.isError ? "tool_error" : "tool_end",
			event_time_unix_ms: endedAtUnixMs,
			event_source: "harness",
			agent_context: agentContext,
			tool: {
				tool_call_id: event.toolCallId,
				tool_class: start?.toolClass ?? getToolClass(event.toolName),
				started_at_unix_ms: startedAtUnixMs,
				ended_at_unix_ms: endedAtUnixMs,
				duration_ms: start ? Math.max(0, Math.round((endedAtPerfMs - start.startedAtPerfMs) * 1000) / 1000) : 0,
				status: event.isError ? "error" : "succeeded",
				...(event.isError ? { error_type: "pi_tool_error" } : {}),
				...(bytes === undefined ? {} : { output_bytes: bytes }),
			},
		});
	}
}

export async function registerDynamoToolEventRelay(
	pi: ExtensionAPI,
	config: DynamoConfig,
	relayConfig: DynamoToolRelayConfig = readDynamoToolRelayConfig(),
	socketFactory: ToolEventSocketFactory = createZeroMqPushSocket,
): Promise<DynamoToolEventRelay | undefined> {
	if (!config.traceEnabled || !relayConfig.endpoint) return undefined;
	const publisher = new DynamoToolEventPublisher(relayConfig, socketFactory);
	await publisher.start();
	const relay = new DynamoToolEventRelay(config, publisher);
	pi.on("tool_execution_start", (event, ctx) => relay.handleToolExecutionStart(event, ctx));
	pi.on("tool_execution_end", (event, ctx) => relay.handleToolExecutionEnd(event, ctx));
	pi.on("session_shutdown", () => publisher.close());
	return relay;
}
