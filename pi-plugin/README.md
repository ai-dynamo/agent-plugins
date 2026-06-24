# pi-dynamo-provider

A Pi extension that registers a `dynamo` provider backed by [Dynamo](https://github.com/ai-dynamo/dynamo)'s OpenAI-compatible endpoint, so Pi can use Dynamo as a normal model:

```bash
pi --model dynamo/<model-id>
```

With one switch (`DYN_REQUEST_TRACE=1`) it also stamps Dynamo session headers, gives each pi-subagent its own session id, and can relay Pi tool events into the trace — all without patching `pi-mono`.

Tested with Pi `0.72.1`. CI also type-checks, tests, and builds against the
latest published Pi packages.

## What it does

- **Model provider** — registers `dynamo`, discovers models from `/v1/models` (falls back to `dynamo/default`), and streams via Pi's OpenAI-compatible path.
- **Session headers** — adds `x-dynamo-session-id` and optional parent headers so Dynamo can attribute each LLM request as a session in its trace.
- **Subagent session ids** — gives each [pi-subagents](https://github.com/nicobailon/pi-subagents) child its own session id. See [Subagent session ids](#subagent-session-ids).
- **Tool-event relay** — optionally pushes Pi `tool_start` / `tool_end` / `tool_error` events to Dynamo over ZMQ so one trace shows LLM spans and tool spans together.

Everything but the bare model provider is gated by the `DYN_REQUEST_TRACE` master switch and is off by default.
Session headers carry identity only; they do not activate sticky or session-aware routing.

## Install

```bash
# From a local checkout, after `npm install && npm run build`
pi install /absolute/path/to/pi-dynamo-provider/pi-plugin

# Or try it for a single run, no install
cd pi-plugin
pi -e ./src/index.ts --model dynamo/<model-id>
```

## Quick start

Point Pi at a running Dynamo endpoint:

```bash
export DYNAMO_BASE_URL=http://127.0.0.1:8000/v1
export DYNAMO_API_KEY=dummy        # local Dynamo usually ignores this; defaults to dynamo-local
export DYN_REQUEST_TRACE=1         # opt into session tracing + optional tool relay

pi --model dynamo/<model-id> -p "Reply exactly ok."
```

That's the whole required setup. Everything else is only set when you want to override it — see [Configuration](#configuration).

## Subagent session ids

When `DYN_REQUEST_TRACE=1`, the provider preserves Pi's normal `sessionId` and adds explicit Dynamo session headers.

```mermaid
sequenceDiagram
    participant Root as Root pi process
    participant Child as Subagent pi process
    participant Dynamo
    Root->>Dynamo: x-dynamo-session-id = S_root
    Child->>Dynamo: x-dynamo-session-id = S_child, parent = S_root
```

- The root `session_id` is Pi's own `sessionId`.
- The child `session_id` is the subagent's own identity (`PI_SUBAGENT_RUN_ID:PI_SUBAGENT_CHILD_AGENT:PI_SUBAGENT_CHILD_INDEX`), so it needs no extra operator setup.
- The provider sends those values as `x-dynamo-session-id` and `x-dynamo-parent-session-id`.

> ZMQ tool records can include parent/child **session ids** when `DYN_AGENT_SESSION_ID` is set on the root. See [Session linking](#session-linking).

## Configuration

The only thing you must set is the connection (`DYNAMO_BASE_URL`) and, to enable the agentic features, `DYN_REQUEST_TRACE`. Everything below is an optional override.

| Variable | Default | Purpose |
| --- | --- | --- |
| `DYNAMO_BASE_URL` | `http://127.0.0.1:8000/v1` | Dynamo endpoint root (falls back to `OPENAI_BASE_URL`). |
| `DYNAMO_API_KEY` | `dynamo-local` | Bearer token. |
| `DYN_REQUEST_TRACE` | off | **Master switch.** When truthy (`1`/`true`/`yes`/`on`), enables Dynamo session headers and the tool relay. |
| `DYN_AGENT_SESSION_ID` | unset | Optional parent session seed for [session linking](#session-linking) in subagents. |
| `DYN_AGENT_PARENT_SESSION_ID` | unset | Parent session; set manually to override the bridge. |
| `DYN_REQUEST_TRACE_TOOL_EVENTS_ZMQ_ENDPOINT` | unset | Dynamo-bound ZMQ PULL endpoint for the tool relay. |

`PI_SUBAGENT_CHILD` / `PI_SUBAGENT_RUN_ID` / `PI_SUBAGENT_CHILD_AGENT` / `PI_SUBAGENT_CHILD_INDEX` are **read, never set** — pi-subagents populates them and the provider uses them to derive the child `session_id` and parent link.

With `DYN_REQUEST_TRACE` on, the provider does not mutate request payloads. It adds Dynamo session headers and `x-request-id` when absent.

<details>
<summary>Tool-event wire format</summary>

When a tool-event endpoint is set, Pi connects a ZMQ PUSH socket and sends one multipart message per event:

```text
[topic, seq_be_u64, msgpack(RequestTraceRecord)]
```

The record uses Dynamo's `dynamo.request.trace.v1` schema (`event_type`, `event_source`, `agent_context`, and a `tool` object with timing/status). Dynamo owns the PULL bind side, so multiple Pi processes and subagents can all connect as producers. Terminal `tool_end` / `tool_error` records are self-contained.
</details>

## Session linking

The provider keeps parent and child session ids distinct for ZMQ tool records. When a pi-subagents child inherits the parent's `DYN_AGENT_SESSION_ID`, the provider reinterprets it as the child's `parent_session_id` and synthesizes a fresh child `session_id` (`runId:childAgent:childIndex`), mutating `process.env` so nested chains stay attributable. Setting `DYN_AGENT_PARENT_SESSION_ID` manually overrides the parent link. If you don't set `DYN_AGENT_SESSION_ID` at all, every subagent still gets its own child session id — only the explicit parent-to-child link is absent.

## Local Dynamo

Two helper scripts onboard a local Dynamo for testing:

```bash
cd pi-plugin
./scripts/install-dynamo.sh    # clone + build Dynamo into a cache dir via uv + maturin
./scripts/launch-agg-agent.sh  # serve GLM-4.7-Flash: one frontend + one SGLang worker
```

`launch-agg-agent.sh` uses file discovery + TCP + ZMQ (no NATS/etcd), enables JSONL tracing, and prints the exact Pi env to use. Common overrides:

```bash
cd pi-plugin
./scripts/launch-agg-agent.sh --gpu 1            # different single GPU
./scripts/launch-agg-agent.sh --gpu 0,1 --tp 2   # one worker across two GPUs
./scripts/launch-agg-agent.sh -- --disable-cuda-graph   # forward flags to dynamo.sglang
```

## Development

```bash
npm install
npm run check   # tsc --noEmit (strict)
npm run test    # vitest
npm run build   # -> dist/
```

`scripts/integration-smoke.sh` boots Dynamo's frontend + mocker and asserts `x-dynamo-session-id` becomes `session_id` in the trace; it is the out-of-band end-to-end check.

## Troubleshooting

- **`/v1/models` empty** — wait for the backend to load; confirm frontend and worker share the same discovery/request/event planes and `DYN_FILE_KV`.
- **Model unknown** — `curl "$DYNAMO_BASE_URL/models"` and use the returned id as `dynamo/<id>`; restart Pi if discovery failed before Dynamo was ready.
- **No agent_context in trace rows** — make sure `DYN_REQUEST_TRACE` is set and Dynamo is new enough to map `x-dynamo-session-id`.
- **Tool spans missing** — set a tool-event endpoint on both sides and confirm the run actually used tools.

## Scope

No `pi-mono` core changes, no native Rust ABI, no Dynamo launch management beyond the helper scripts. The request trace schema is owned upstream by Dynamo.
