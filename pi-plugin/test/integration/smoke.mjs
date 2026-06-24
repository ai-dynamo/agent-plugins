// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Integration smoke test: spins up a Dynamo frontend + mocker, sends one chat
// completion through Dynamo and asserts that x-dynamo-session-id becomes
// session identity in the JSONL request trace.
//
// Not a unit test — runs out-of-band of vitest. Driven by
// scripts/integration-smoke.sh which boots Dynamo, exports the trace sink env
// vars, and invokes this file. Exits 0 on pass, non-zero on any assertion or
// transport failure.
//
// Assertions, in order:
//   1. x-dynamo-session-id becomes Dynamo agent_context session_id
//   2. subagent bridge derives a child session id when
//      PI_SUBAGENT_CHILD=1 + bookkeeping vars are exported
//
// Mocker output text is intentionally garbage; we never assert on response
// content, only on the trace envelope.

import { readFileSync, existsSync } from "node:fs";
import { setTimeout as delay } from "node:timers/promises";

import { readDynamoConfig } from "../../dist/light/provider.js";

const TRACE_PATH = mustEnv("DYN_REQUEST_TRACE_OUTPUT_PATH");
const BASE_URL = mustEnv("DYNAMO_BASE_URL");
const MODEL_ID = mustEnv("DYNAMO_TEST_MODEL_ID");

function mustEnv(name) {
	const value = process.env[name];
	if (!value) {
		console.error(`smoke: ${name} must be set`);
		process.exit(2);
	}
	return value;
}

function readTraceEvents() {
	if (!existsSync(TRACE_PATH)) return [];
	const text = readFileSync(TRACE_PATH, "utf-8");
	const events = [];
	for (const line of text.split("\n")) {
		if (!line.trim()) continue;
		try {
			const envelope = JSON.parse(line);
			const event = envelope.event ?? envelope;
			if (event && typeof event === "object") events.push(event);
		} catch {
			// best-effort: dynamo writes one JSON object per line, ignore garbage
		}
	}
	return events;
}

async function waitForTraceMatching(predicate, label, timeoutMs = 15000) {
	const startMs = Date.now();
	while (Date.now() - startMs < timeoutMs) {
		const events = readTraceEvents();
		const found = events.find(predicate);
		if (found) return found;
		await delay(200);
	}
	throw new Error(`smoke: timed out waiting for trace event: ${label}`);
}

async function postChat({ sessionId, parentSessionId, xRequestId }) {
	const body = {
		model: MODEL_ID,
		messages: [{ role: "user", content: "smoke" }],
		max_tokens: 4,
		stream: false,
	};
	const response = await fetch(`${BASE_URL}/chat/completions`, {
		method: "POST",
		headers: {
			"content-type": "application/json",
			"x-dynamo-session-id": sessionId,
			...(parentSessionId ? { "x-dynamo-parent-session-id": parentSessionId } : {}),
			"x-request-id": xRequestId,
			authorization: `Bearer ${process.env.DYNAMO_API_KEY ?? "dynamo-local"}`,
		},
		body: JSON.stringify(body),
	});
	if (!response.ok) {
		const text = await response.text();
		throw new Error(`smoke: POST /chat/completions ${response.status}: ${text}`);
	}
	// Drain the body; mocker output is garbage but the request must complete
	// before request_end is written to the trace.
	await response.text();
}

function assert(condition, message) {
	if (!condition) throw new Error(`smoke: assertion failed: ${message}`);
}

async function caseTopLevelSessionHeader() {
	const xRequestId = "smoke-toplevel-" + Date.now();
	const sessionId = "smoke-session-toplevel";
	await postChat({ sessionId, xRequestId });

	const event = await waitForTraceMatching(
		(e) => e.event_type === "request_end" && e.request?.x_request_id === xRequestId,
		`request_end with x_request_id=${xRequestId}`,
	);

	assert(event.agent_context, "trace event missing agent_context");
	assert(
		event.agent_context.session_id === sessionId,
		`session_id mismatch: got ${event.agent_context.session_id}`,
	);
	assert(
		event.agent_context.parent_session_id === undefined ||
			event.agent_context.parent_session_id === null,
		`parent_session_id should be unset for top-level case`,
	);
	console.log("  PASS top-level session_id from x-dynamo-session-id");
}

async function caseSubagentBridge() {
	// Simulate the env shape pi-subagents would set on a spawned child:
	// inherited DYN_AGENT_SESSION_ID (parent's id) plus PI_SUBAGENT_* bookkeeping.
	// readDynamoConfig should rewrite both ids before the request is sent.
	const env = {
		DYNAMO_BASE_URL: BASE_URL,
		DYN_REQUEST_TRACE: "1",
		DYN_AGENT_SESSION_ID: "smoke-orchestrator",
		PI_SUBAGENT_CHILD: "1",
		PI_SUBAGENT_RUN_ID: "smoke-run",
		PI_SUBAGENT_CHILD_AGENT: "researcher",
		PI_SUBAGENT_CHILD_INDEX: "0",
	};
	const config = readDynamoConfig(env);
	assert(
		config.sessionId === "smoke-run:researcher:0",
		`bridge did not rewrite session_id: got ${config.sessionId}`,
	);
	assert(
		config.parentSessionId === "smoke-orchestrator",
		`bridge did not set parent_session_id: got ${config.parentSessionId}`,
	);

	const xRequestId = "smoke-subagent-" + Date.now();
	await postChat({
		sessionId: config.sessionId,
		parentSessionId: config.parentSessionId,
		xRequestId,
	});

	const event = await waitForTraceMatching(
		(e) => e.event_type === "request_end" && e.request?.x_request_id === xRequestId,
		`request_end with x_request_id=${xRequestId}`,
	);

	assert(event.agent_context, "trace event missing agent_context");
	assert(
		event.agent_context.session_id === "smoke-run:researcher:0",
		`subagent session_id mismatch: got ${event.agent_context.session_id}`,
	);
	console.log("  PASS pi-subagents session_id header");
}

async function main() {
	console.log(`smoke: trace path = ${TRACE_PATH}`);
	console.log(`smoke: dynamo base = ${BASE_URL}`);
	console.log(`smoke: model = ${MODEL_ID}`);

	await caseTopLevelSessionHeader();
	await caseSubagentBridge();

	console.log("smoke: all assertions passed");
}

main().catch((err) => {
	console.error(err.message ?? err);
	process.exit(1);
});
