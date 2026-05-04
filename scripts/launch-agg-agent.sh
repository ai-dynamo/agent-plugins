#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: scripts/launch-agg-agent.sh [OPTIONS] [-- SGLANG_ARGS...]

Launch Dynamo's OpenAI-compatible frontend plus one SGLang worker for
GLM-4.7-Flash with agent tracing and Pi tool-event ingest enabled.

This launcher uses file discovery, TCP request plane, and ZMQ event plane.
It does not require NATS or etcd.

Options:
  --workdir PATH              Cache/work directory.
                              Default: ${XDG_CACHE_HOME:-$HOME/.cache}/pi-dynamo-provider
  --dynamo-dir PATH           Dynamo checkout path. Default: <workdir>/dynamo
  --model MODEL               Served model. Default: zai-org/GLM-4.7-Flash
  --gpu GPUS                  CUDA_VISIBLE_DEVICES for one worker. Default: 0
  --tp N                      Tensor parallelism for the worker. Default: 1
  --http-port PORT            Dynamo HTTP port. Default: 18083
  --system-port PORT          Worker system metrics/control port. Default: 18084
  --tool-events-endpoint END  ZMQ endpoint for Pi tool events. Default: tcp://127.0.0.1:20390
  -h, --help                  Show this help.

Environment overrides:
  PI_DYNAMO_WORKDIR
  DYNAMO_DIR
  MODEL
  CUDA_VISIBLE_DEVICES
  TP
  DYN_HTTP_PORT
  DYN_SYSTEM_PORT
  DYN_AGENT_TRACE_TOOL_EVENTS_ZMQ_ENDPOINT
  DYN_AGENT_TRACE_OUTPUT_PATH

Examples:
  scripts/launch-agg-agent.sh
  scripts/launch-agg-agent.sh --gpu 1
  scripts/launch-agg-agent.sh --gpu 0,1 --tp 2
  scripts/launch-agg-agent.sh -- --disable-cuda-graph
EOF
}

log() {
    printf '[pi-dynamo-launch] %s\n' "$*"
}

die() {
    printf '[pi-dynamo-launch] ERROR: %s\n' "$*" >&2
    exit 1
}

require_cmd() {
    command -v "$1" >/dev/null 2>&1 || die "missing required command: $1"
}

cleanup() {
    local rc=$?
    if ((${#CHILD_PIDS[@]} > 0)); then
        log "Cleaning up launched processes..."
        for pid in "${CHILD_PIDS[@]}"; do
            kill "$pid" >/dev/null 2>&1 || true
        done
        wait >/dev/null 2>&1 || true
    fi
    exit "$rc"
}

DEFAULT_WORKDIR="${XDG_CACHE_HOME:-$HOME/.cache}/pi-dynamo-provider"
WORKDIR="${PI_DYNAMO_WORKDIR:-$DEFAULT_WORKDIR}"
DYNAMO_DIR="${DYNAMO_DIR:-}"
MODEL="${MODEL:-zai-org/GLM-4.7-Flash}"
GPUS="${CUDA_VISIBLE_DEVICES:-0}"
TP="${TP:-1}"
HTTP_PORT="${DYN_HTTP_PORT:-18083}"
SYSTEM_PORT="${DYN_SYSTEM_PORT:-18084}"
TOOL_EVENTS_ENDPOINT="${DYN_AGENT_TRACE_TOOL_EVENTS_ZMQ_ENDPOINT:-tcp://127.0.0.1:20390}"
SGLANG_ARGS=()

while (($# > 0)); do
    case "$1" in
        --workdir)
            WORKDIR="$2"
            shift 2
            ;;
        --dynamo-dir)
            DYNAMO_DIR="$2"
            shift 2
            ;;
        --model|--model-path)
            MODEL="$2"
            shift 2
            ;;
        --gpu|--gpus)
            GPUS="$2"
            shift 2
            ;;
        --tp)
            TP="$2"
            shift 2
            ;;
        --http-port)
            HTTP_PORT="$2"
            shift 2
            ;;
        --system-port)
            SYSTEM_PORT="$2"
            shift 2
            ;;
        --tool-events-endpoint)
            TOOL_EVENTS_ENDPOINT="$2"
            shift 2
            ;;
        --)
            shift
            SGLANG_ARGS+=("$@")
            break
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            die "unknown option: $1"
            ;;
    esac
done

DYNAMO_DIR="${DYNAMO_DIR:-$WORKDIR/dynamo}"
[[ -x "$DYNAMO_DIR/.venv/bin/python" ]] || die "missing Dynamo venv at $DYNAMO_DIR/.venv; run scripts/install-dynamo.sh first"

RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
RUN_DIR="$WORKDIR/runs/$RUN_ID"
LOG_DIR="$RUN_DIR/logs"
TRACE_PATH="${DYN_AGENT_TRACE_OUTPUT_PATH:-$RUN_DIR/dynamo-agent-trace.jsonl}"
FILE_KV="${DYN_FILE_KV:-$RUN_DIR/file-kv}"
CHILD_PIDS=()

trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

require_cmd curl

mkdir -p "$RUN_DIR" "$LOG_DIR" "$(dirname "$TRACE_PATH")" "$FILE_KV"

print_ready_block() {
    cat <<EOF

Dynamo is ready.

Pi environment for another shell:

  export DYNAMO_BASE_URL=http://127.0.0.1:${HTTP_PORT}/v1
  export DYNAMO_API_KEY=dummy
  export DYN_AGENT_WORKFLOW_TYPE_ID=pi_coding_agent
  export DYN_AGENT_WORKFLOW_ID=pi-demo-${RUN_ID}
  export DYN_AGENT_TOOL_EVENTS_ZMQ_ENDPOINT=${TOOL_EVENTS_ENDPOINT}

Example Pi command:

  pi --model dynamo/${MODEL} -p "Reply exactly ok."

Trace output:

  ${TRACE_PATH}

Perfetto conversion:

  cd ${DYNAMO_DIR}
  source .venv/bin/activate
  python benchmarks/agent_trace/convert_to_perfetto.py \\
    ${TRACE_PATH} \\
    --include-markers \\
    --separate-stage-tracks \\
    --output ${RUN_DIR}/dynamo-agent-trace.perfetto.json

EOF
}

wait_for_model() {
    local url="http://127.0.0.1:${HTTP_PORT}/v1/models"
    local deadline=$((SECONDS + 900))
    log "Waiting for $url to expose $MODEL..."

    while ((SECONDS < deadline)); do
        if models_json="$(curl -sf --max-time 2 "$url" 2>/dev/null)"; then
            if grep -Fq "\"id\":\"${MODEL}\"" <<<"$models_json" || grep -Fq "\"id\": \"${MODEL}\"" <<<"$models_json"; then
                print_ready_block
                return
            fi
        fi

        for pid in "${CHILD_PIDS[@]}"; do
            kill -0 "$pid" >/dev/null 2>&1 || die "a launched process exited before readiness; see $LOG_DIR"
        done
        sleep 5
    done

    die "timed out waiting for model readiness at $url"
}

cd "$DYNAMO_DIR"
# shellcheck disable=SC1091
source .venv/bin/activate

export CUDA_VISIBLE_DEVICES="$GPUS"
export DYN_HTTP_PORT="$HTTP_PORT"
export DYN_DISCOVERY_BACKEND=file
export DYN_REQUEST_PLANE=tcp
export DYN_EVENT_PLANE=zmq
export DYN_FILE_KV="$FILE_KV"
export DYN_AGENT_TRACE_SINKS="${DYN_AGENT_TRACE_SINKS:-jsonl}"
export DYN_AGENT_TRACE_OUTPUT_PATH="$TRACE_PATH"
export DYN_AGENT_TRACE_JSONL_FLUSH_INTERVAL_MS="${DYN_AGENT_TRACE_JSONL_FLUSH_INTERVAL_MS:-100}"
export DYN_AGENT_TRACE_TOOL_EVENTS_ZMQ_ENDPOINT="$TOOL_EVENTS_ENDPOINT"
export DYN_LOG="${DYN_LOG:-info}"

log "Run directory: $RUN_DIR"
log "Dynamo checkout: $DYNAMO_DIR"
log "Model: $MODEL"
log "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
log "TP=$TP"
log "Discovery: file ($DYN_FILE_KV)"
log "Request plane: tcp"
log "Event plane: zmq"
log "HTTP: http://127.0.0.1:$HTTP_PORT/v1"
log "Trace JSONL: $TRACE_PATH"
log "Tool relay endpoint: $TOOL_EVENTS_ENDPOINT"

log "Starting Dynamo frontend"
python3 -m dynamo.frontend \
    --discovery-backend file \
    --request-plane tcp \
    --event-plane zmq \
    --router-mode round-robin \
    >"$LOG_DIR/frontend.log" 2>&1 &
CHILD_PIDS+=("$!")

log "Starting Dynamo SGLang worker"
DYN_SYSTEM_PORT="$SYSTEM_PORT" \
python3 -m dynamo.sglang \
    --discovery-backend file \
    --request-plane tcp \
    --event-plane zmq \
    --model-path "$MODEL" \
    --served-model-name "$MODEL" \
    --page-size 16 \
    --tp "$TP" \
    --trust-remote-code \
    --enable-streaming-session \
    --skip-tokenizer-init \
    --dyn-reasoning-parser glm45 \
    --dyn-tool-call-parser glm47 \
    --enable-metrics \
    "${SGLANG_ARGS[@]}" \
    >"$LOG_DIR/worker.log" 2>&1 &
CHILD_PIDS+=("$!")

wait_for_model

log "Dynamo is running. Press Ctrl+C to stop."
wait -n "${CHILD_PIDS[@]}"
