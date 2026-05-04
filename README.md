# pi-dynamo-provider

Pi extension package that registers a `dynamo` provider backed by Dynamo's OpenAI-compatible chat-completions endpoint.

It lets Pi use Dynamo as a normal model provider:

```bash
pi --model dynamo/<model-id>
```

The extension also injects Dynamo agent-context metadata on every LLM request and can relay Pi tool events into Dynamo's agent trace sink, so one Dynamo trace can show both LLM requests and Pi tool trajectories.

## What This Package Does

- Registers a Pi provider named `dynamo`.
- Discovers models from Dynamo's `/v1/models` endpoint.
- Delegates chat-completion streaming to Pi's existing OpenAI-compatible provider.
- Adds `nvext.agent_context` to each request payload.
- Adds `x-request-id` when the request does not already have one.
- Optionally publishes Pi tool events to Dynamo over ZMQ:
  - `tool_start`
  - `tool_end`
  - `tool_error`
- Keeps this integration outside `pi-mono`; no Pi core patch is required.

Request and tool data flow:

```text
Pi
  |
  |  OpenAI-compatible /v1/chat/completions
  |  + nvext.agent_context
  |  + x-request-id
  v
Dynamo frontend
  |
  v
Dynamo agent trace sink

Pi tool events
  |
  |  ZMQ PUB: [topic, seq_be_u64, msgpack(AgentTraceRecord)]
  v
Dynamo tool-event ingest
  |
  v
Dynamo agent trace sink
```

## Install

From npm, after the package is published:

```bash
pi install npm:pi-dynamo-provider
```

From this GitHub repo:

```bash
pi install git:git@github.com:NVIDIA-dev/pi-dynamo-provider.git
```

For a project-local install that can be checked into `.pi/settings.json`, add `-l`:

```bash
pi install -l git:git@github.com:NVIDIA-dev/pi-dynamo-provider.git
```

For local development from a checkout:

```bash
git clone git@github.com:NVIDIA-dev/pi-dynamo-provider.git
cd pi-dynamo-provider
npm install
npm run build

pi install /absolute/path/to/pi-dynamo-provider
```

You can also try the extension for one run without installing it:

```bash
cd /absolute/path/to/pi-dynamo-provider
pi -e ./src/index.ts --model dynamo/<model-id>
```

## Quick Start

Start Dynamo with an OpenAI-compatible endpoint, then point Pi at it:

```bash
export DYNAMO_BASE_URL=http://127.0.0.1:8000/v1
export DYNAMO_API_KEY=dummy

pi --model dynamo/<model-id> -p "Reply exactly ok."
```

For local Dynamo, the API key is usually not checked. This package defaults to `dynamo-local` if `DYNAMO_API_KEY` is unset.

## Dynamo Requirements

Minimum:

- Dynamo serves an OpenAI-compatible `/v1/chat/completions` endpoint.
- Dynamo serves `/v1/models`, or you are willing to use the fallback model id `dynamo/default`.
- Pi can reach `DYNAMO_BASE_URL` from the machine running Pi.

For request tracing:

- Dynamo agent tracing must be enabled.
- Pi requests must include `nvext.agent_context`, which this package injects.

For tool tracing:

- Dynamo must include the tool-event ingest/relay support.
- Dynamo and Pi must use the same ZMQ endpoint.
- Pi must run with tools enabled and execute at least one tool.

Example Dynamo launch shape for `examples/backends/sglang/launch/agg_agent.sh`:

```bash
cd /home/nvidia/dynamo

export CUDA_VISIBLE_DEVICES=0,1
export DYN_HTTP_PORT=18083
export DYN_DISCOVERY_BACKEND=file
export DYN_FILE_KV=/tmp/dynamo-file-kv

# Required by the KV-routed agg_agent.sh path.
export DYN_EVENT_PLANE=nats
export NATS_SERVER=nats://127.0.0.1:4222

# Request trace sink.
export DYN_AGENT_TRACE_SINKS=jsonl
export DYN_AGENT_TRACE_OUTPUT_PATH=/tmp/dynamo-agent-trace.jsonl

# Tool-event subscriber endpoint. Pi publishes to this same endpoint.
export DYN_AGENT_TRACE_TOOL_EVENTS_ZMQ_ENDPOINT=tcp://127.0.0.1:20390

bash examples/backends/sglang/launch/agg_agent.sh
```

Then run Pi:

```bash
export DYNAMO_BASE_URL=http://127.0.0.1:18083/v1
export DYNAMO_API_KEY=dummy

export DYN_AGENT_WORKFLOW_TYPE_ID=pi_coding_agent
export DYN_AGENT_WORKFLOW_ID=pi-demo-001
export DYN_AGENT_TOOL_EVENTS_ZMQ_ENDPOINT=tcp://127.0.0.1:20390

pi --model dynamo/zai-org/GLM-4.7-Flash \
  -p "Run the tests in this folder, fix the smallest bug, and rerun the tests."
```

## Model Names

Use Pi model names in this form:

```text
dynamo/<model-id-from-dynamo>
```

Examples:

```bash
pi --model dynamo/zai-org/GLM-4.7-Flash
pi --model dynamo/Qwen/Qwen3-32B
```

On startup, the extension calls:

```text
<DYNAMO_BASE_URL>/models
```

If model discovery fails, the extension still registers:

```text
dynamo/default
```

That fallback is useful for smoke tests against minimal OpenAI-compatible endpoints, but normal Dynamo runs should use the actual model id returned by `/v1/models`.

<details>
<summary>Pi-side environment variables</summary>

| Variable | Default | Purpose |
| --- | --- | --- |
| `DYNAMO_BASE_URL` | `http://127.0.0.1:8000/v1` | Dynamo OpenAI-compatible endpoint root. |
| `OPENAI_BASE_URL` | unset | Fallback endpoint root when `DYNAMO_BASE_URL` is unset. |
| `DYNAMO_API_KEY` | `dynamo-local` | Bearer token sent to Dynamo. Local Dynamo usually accepts a dummy value. |
| `DYN_AGENT_WORKFLOW_TYPE_ID` | `pi_coding_agent` | Stable workflow type used by Dynamo traces/profilers. |
| `DYN_AGENT_WORKFLOW_ID` | unset | Workflow/run id. If unset for tool events, the Pi session id is used. |
| `DYN_AGENT_PROGRAM_ID` | unset | Program id override. If unset, Pi's session id is used per request. |
| `DYN_AGENT_PARENT_PROGRAM_ID` | unset | Optional parent program id for nested/subagent workflows. |
| `DYN_AGENT_TOOL_EVENTS_ZMQ_ENDPOINT` | unset | ZMQ endpoint Pi binds and publishes tool events on. |
| `DYN_AGENT_TRACE_TOOL_ZMQ_ENDPOINT` | unset | Alias for the tool-event endpoint. |
| `DYN_AGENT_TRACE_TOOL_EVENTS_ZMQ_ENDPOINT` | unset | Alias for the tool-event endpoint. |
| `DYN_AGENT_TOOL_EVENTS_ZMQ_TOPIC` | `agent-tool-events` | ZMQ topic frame. |
| `DYN_AGENT_TRACE_TOOL_ZMQ_TOPIC` | unset | Alias for the tool-event topic. |
| `DYN_AGENT_TRACE_TOOL_EVENTS_ZMQ_TOPIC` | unset | Alias for the tool-event topic. |
| `DYN_AGENT_TOOL_EVENTS_QUEUE_CAPACITY` | `100000` | Local publish queue capacity before tool events are dropped. |

</details>

<details>
<summary>Dynamo-side environment variables</summary>

These are the Dynamo variables most commonly needed for Pi traces. Exact launch details can differ by Dynamo backend and branch.

| Variable | Example | Purpose |
| --- | --- | --- |
| `DYN_HTTP_PORT` | `18083` | Dynamo HTTP port for `/v1/models` and `/v1/chat/completions`. |
| `DYN_AGENT_TRACE_SINKS` | `jsonl` | Enables Dynamo agent trace sinks. |
| `DYN_AGENT_TRACE_OUTPUT_PATH` | `/tmp/dynamo-agent-trace.jsonl` | JSONL trace output path. |
| `DYN_AGENT_TRACE_JSONL_FLUSH_INTERVAL_MS` | `100` | Optional flush interval for faster interactive validation. |
| `DYN_AGENT_TRACE_TOOL_EVENTS_ZMQ_ENDPOINT` | `tcp://127.0.0.1:20390` | ZMQ subscriber endpoint for Pi tool events. |
| `DYN_EVENT_PLANE` | `nats` | Required by some KV-routed Dynamo launches, including the tested `agg_agent.sh` path. |
| `NATS_SERVER` | `nats://127.0.0.1:4222` | NATS endpoint for Dynamo's event plane. |

</details>

<details>
<summary>Injected request metadata</summary>

Every LLM request payload gets:

```json
{
  "nvext": {
    "agent_context": {
      "workflow_type_id": "pi_coding_agent",
      "workflow_id": "pi-demo-001",
      "program_id": "<pi-session-id>",
      "parent_program_id": "<optional-parent>",
      "phase": "reasoning"
    }
  }
}
```

The extension preserves existing `nvext` fields and existing `nvext.agent_context` fields. It also adds `x-request-id` if the caller did not already set one.

</details>

<details>
<summary>Tool-event wire format</summary>

When `DYN_AGENT_TOOL_EVENTS_ZMQ_ENDPOINT` or one of its aliases is configured, Pi publishes one multipart ZMQ message per tool event:

```text
[topic, seq_be_u64, msgpack(AgentTraceRecord)]
```

The decoded `AgentTraceRecord` uses Dynamo's agent trace schema:

```json
{
  "schema": "dynamo.agent.trace.v1",
  "event_type": "tool_end",
  "event_time_unix_ms": 1777915663000,
  "event_source": "harness",
  "agent_context": {
    "workflow_type_id": "pi_coding_agent",
    "workflow_id": "pi-demo-001",
    "program_id": "<pi-session-id>"
  },
  "tool": {
    "tool_call_id": "<pi-tool-call-id>",
    "tool_class": "bash",
    "started_at_unix_ms": 1777915662000,
    "ended_at_unix_ms": 1777915663000,
    "duration_ms": 1000,
    "status": "succeeded",
    "output_bytes": 1234
  }
}
```

Terminal `tool_end` and `tool_error` records include enough timing information to render a span even if a consumer missed the corresponding `tool_start`.

</details>

## Generate Perfetto

After a run, convert the Dynamo JSONL trace:

```bash
cd /home/nvidia/dynamo
source .venv/bin/activate

python benchmarks/agent_trace/convert_to_perfetto.py \
  /tmp/dynamo-agent-trace.jsonl \
  --include-markers \
  --separate-stage-tracks \
  --output /tmp/dynamo-agent-trace.perfetto.json
```

Open the generated JSON in Perfetto UI:

```text
https://ui.perfetto.dev
```

Expected trace shape:

- `dynamo.llm` spans for LLM requests.
- `dynamo.llm.stage` spans for prefill/decode stages when Dynamo records them.
- `dynamo.agent.tool` spans for Pi tools when ZMQ tool relay is enabled.

## Development

```bash
npm install
npm run check
npm run test
npm run build
```

Run from source without installing:

```bash
export DYNAMO_BASE_URL=http://127.0.0.1:8000/v1
pi -e ./src/index.ts --model dynamo/<model-id>
```

## Troubleshooting

`/v1/models` is empty:

- Wait for the Dynamo backend to finish loading.
- Check Dynamo logs for worker registration failures.
- For `agg_agent.sh`, make sure NATS is running and set `DYN_EVENT_PLANE=nats` plus `NATS_SERVER=nats://127.0.0.1:4222`.

Pi says the model is unknown:

- Run `curl -s "$DYNAMO_BASE_URL/models"` and use the returned id as `dynamo/<id>`.
- If discovery failed during Pi startup, restart Pi after Dynamo is ready.

LLM traces exist but tool spans are missing:

- Set `DYN_AGENT_TRACE_TOOL_EVENTS_ZMQ_ENDPOINT` on Dynamo.
- Set `DYN_AGENT_TOOL_EVENTS_ZMQ_ENDPOINT` to the same endpoint on Pi.
- Confirm the Pi run actually used tools.

Tool spans exist but request spans do not:

- Enable Dynamo tracing with `DYN_AGENT_TRACE_SINKS=jsonl`.
- Set `DYN_AGENT_TRACE_OUTPUT_PATH`.
- Confirm the request reached Dynamo's `/v1/chat/completions` endpoint.

Authentication fails:

- Set `DYNAMO_API_KEY` to the token expected by your Dynamo deployment.
- For local Dynamo, `DYNAMO_API_KEY=dummy` is usually sufficient.

## Current Scope

Included:

- OpenAI-compatible chat-completions path.
- Model discovery from `/v1/models`.
- Dynamo request metadata injection.
- Pi session id as default `program_id`.
- Optional ZMQ tool-event relay into Dynamo traces.
- Perfetto-compatible trace output through Dynamo's converter.

Not included:

- No `pi-mono` core changes.
- No native Rust Pi extension ABI. A Rust implementation would still need a TypeScript/JavaScript package entrypoint, for example through N-API.
- No automatic Dynamo launch management.
- No automatic Perfetto upload or viewer hosting.
