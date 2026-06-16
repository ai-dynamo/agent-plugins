#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Start a Dynamo frontend + mocker worker against a known-good Dynamo version,
# then run test/integration/smoke.mjs which asserts that nvext.agent_context
# round-trips end-to-end through the trace sink. Tears down processes on exit.
#
# Required env:
#   DYNAMO_TEST_MODEL_ID  HuggingFace model id for the mocker tokenizer
#                         (default: Qwen/Qwen3-0.6B — tokenizer is small, no weights)
#
# Inputs:
#   $1  optional path to an existing Dynamo Python install (skips pip install)
#
# Designed to run on GitHub-hosted ubuntu-latest runners. Mocker has no GPU
# requirement; tokenizer downloads ~70MB to ~/.cache/huggingface.

set -euo pipefail

readonly SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
readonly REPO_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)

: "${DYNAMO_TEST_MODEL_ID:=Qwen/Qwen3-0.6B}"
: "${DYNAMO_FRONTEND_PORT:=18083}"
: "${TRACE_DIR:=$(mktemp -d -t pi-dynamo-smoke-XXXXXX)}"
: "${DYNAMO_TIMEOUT_SECS:=120}"

readonly TRACE_PATH="${TRACE_DIR}/dynamo-request-trace.jsonl"
readonly FRONTEND_LOG="${TRACE_DIR}/frontend.log"
readonly MOCKER_LOG="${TRACE_DIR}/mocker.log"

FRONTEND_PID=""
MOCKER_PID=""

cleanup() {
	set +e
	if [[ -n "${MOCKER_PID}" ]] && kill -0 "${MOCKER_PID}" 2>/dev/null; then
		kill -TERM "${MOCKER_PID}" 2>/dev/null
		wait "${MOCKER_PID}" 2>/dev/null
	fi
	if [[ -n "${FRONTEND_PID}" ]] && kill -0 "${FRONTEND_PID}" 2>/dev/null; then
		kill -TERM "${FRONTEND_PID}" 2>/dev/null
		wait "${FRONTEND_PID}" 2>/dev/null
	fi
	if [[ "${SMOKE_KEEP_LOGS:-0}" != "1" ]]; then
		rm -rf "${TRACE_DIR}"
	else
		echo "smoke: logs preserved in ${TRACE_DIR}" >&2
	fi
}
trap cleanup EXIT

wait_for_http() {
	local url="$1"
	local deadline=$(( $(date +%s) + DYNAMO_TIMEOUT_SECS ))
	while [[ $(date +%s) -lt ${deadline} ]]; do
		if curl -sf -o /dev/null --max-time 2 "${url}"; then
			return 0
		fi
		sleep 1
	done
	echo "smoke: timed out waiting for ${url}" >&2
	echo "--- frontend.log ---" >&2; tail -n 80 "${FRONTEND_LOG}" >&2 || true
	echo "--- mocker.log ---" >&2;   tail -n 80 "${MOCKER_LOG}" >&2 || true
	return 1
}

echo "smoke: trace dir = ${TRACE_DIR}"

# Build the provider so the smoke test can import dist/dynamo-provider.js.
echo "smoke: building pi-dynamo-provider"
(cd "${REPO_ROOT}" && npm run build >/dev/null)

# Trace sink config. flush interval kept short so the smoke test doesn't race
# the writer when reading the JSONL between requests.
export DYN_REQUEST_TRACE=1
export DYN_REQUEST_TRACE_SINKS=jsonl
export DYN_REQUEST_TRACE_OUTPUT_PATH="${TRACE_PATH}"
export DYN_REQUEST_TRACE_JSONL_FLUSH_INTERVAL_MS=100

# Local transport plane — no NATS, no etcd. file-backed discovery + tcp + zmq.
export DYN_DISCOVERY_BACKEND=file
export DYN_REQUEST_PLANE=tcp
export DYN_EVENT_PLANE=zmq
export DYN_FILE_KV="${TRACE_DIR}/file-kv"
export DYN_HTTP_PORT="${DYNAMO_FRONTEND_PORT}"

echo "smoke: starting dynamo.frontend on port ${DYNAMO_FRONTEND_PORT}"
python -m dynamo.frontend \
	--discovery-backend file \
	--request-plane tcp \
	--event-plane zmq \
	--router-mode round-robin \
	>"${FRONTEND_LOG}" 2>&1 &
FRONTEND_PID=$!

echo "smoke: starting dynamo.mocker with --model-path ${DYNAMO_TEST_MODEL_ID}"
DYN_SYSTEM_PORT=$((DYNAMO_FRONTEND_PORT + 1)) \
python -m dynamo.mocker \
	--model-path "${DYNAMO_TEST_MODEL_ID}" \
	--discovery-backend file \
	--request-plane tcp \
	--event-plane zmq \
	--num-workers 1 \
	--speedup-ratio 10.0 \
	>"${MOCKER_LOG}" 2>&1 &
MOCKER_PID=$!

wait_for_http "http://127.0.0.1:${DYNAMO_FRONTEND_PORT}/v1/models"

# Confirm the mocker registered. Frontend serves /v1/models even before workers
# attach, so we additionally check the model id appears.
if ! curl -sf "http://127.0.0.1:${DYNAMO_FRONTEND_PORT}/v1/models" | grep -q "${DYNAMO_TEST_MODEL_ID}"; then
	echo "smoke: ${DYNAMO_TEST_MODEL_ID} not yet visible to /v1/models; retrying briefly"
	for _ in 1 2 3 4 5 6 7 8 9 10; do
		sleep 2
		curl -sf "http://127.0.0.1:${DYNAMO_FRONTEND_PORT}/v1/models" | grep -q "${DYNAMO_TEST_MODEL_ID}" && break
	done
fi

export DYNAMO_BASE_URL="http://127.0.0.1:${DYNAMO_FRONTEND_PORT}/v1"
export DYNAMO_TEST_MODEL_ID
export DYN_REQUEST_TRACE_OUTPUT_PATH

echo "smoke: running assertions"
node "${REPO_ROOT}/test/integration/smoke.mjs"
