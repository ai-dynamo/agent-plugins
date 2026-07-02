---
name: pi-headless-dynamo
description: Drive the real Pi CLI headlessly against a Dynamo or OpenAI-compatible endpoint for pi-dynamo-provider validation. Use when testing Pi provider installs, session header tracing, Pi subagent runs, saved traces, or parent/child session behavior without manually faking Pi or pi-subagents internals.
---

# Pi Headless Dynamo

## Purpose

Drive Pi the way a human would: launch the real interactive Pi CLI in a
pseudoterminal, type normal prompts or public slash commands, let Pi and
pi-subagents create child sessions, then verify behavior from Pi artifacts and
Dynamo/SGLang logs.

Do not synthesize `PI_SUBAGENT_*`, edit Pi session files, call the provider
directly to stand in for Pi, or patch pi-subagents while validating this repo.

## Preconditions

Use a running Dynamo endpoint or start one with the repo launcher:

```bash
pi-plugin/scripts/launch-agg-agent.sh --dynamo-dir /ephemeral/dynamo-radix-native --gpu 0,1 --tp 2 --http-port 18083 --system-port 18084
```

Before launching Pi, verify the endpoint and model:

```bash
curl -sf http://127.0.0.1:18083/v1/models
```

For session-native release evidence, the endpoint must use Dynamo
`--router-mode kv` and an SGLang worker with `--enable-session-radix-cache`.
The local launcher prints the exact Pi environment and trace path; prefer that
block over hand-rolled env.

## Launch Pi

Use a fresh artifact root and run the real TUI under `script(1)` so the
terminal transcript is saved:

```bash
RUN_ROOT=/ephemeral/pi-headless-$(date -u +%Y%m%dT%H%M%SZ)
WORKSPACE=$RUN_ROOT/workspace
MODEL=zai-org/GLM-4.7-Flash
mkdir -p "$WORKSPACE"

export DYNAMO_BASE_URL=http://127.0.0.1:18083/v1
export DYNAMO_API_KEY=dummy
export DYN_REQUEST_TRACE=1
export DYN_REQUEST_TRACE_TOOL_EVENTS_ZMQ_ENDPOINT=tcp://127.0.0.1:20390

cd "$WORKSPACE"
script -qefc "pi --model dynamo/${MODEL} --tools subagent,bash,write,read,ls,grep,find" "${RUN_ROOT}/pi-terminal.typescript"
```

Control that process through its PTY like a user:

- wait for the Pi prompt before sending the first prompt;
- paste a full prompt or slash command as text;
- send Enter to submit;
- wait for Pi to finish before sending the next prompt;
- type `/quit` and wait for process exit so Pi shuts down cleanly.

Do not kill Pi to end a lifecycle run unless it is hung and the failure is the
thing being tested.

The provider stamps `x-dynamo-session-id` on every LLM request with a Pi session. Normal root turns use Pi's own session id; pi-subagents children derive theirs from `PI_SUBAGENT_*`. `DYN_REQUEST_TRACE=1` above enables the optional tool relay.

## Drive A Lifecycle Run

Start with a normal parent turn:

```text
Create a short project brief in this workspace, then tell me when you are ready for delegated follow-up work.
```

Launch children through pi-subagents' public command surface. For Dynamo tests,
pin the child model in every step; otherwise builtins may inherit a non-Dynamo
default model and fail before reaching the endpoint.

```text
/parallel delegate[model=dynamo/zai-org/GLM-4.7-Flash,output=child-a.md,outputMode=file-only] "Work only in the current workspace. Create logs/a.md with a concise result. Read it back. End with CHILD_A_DONE." -> delegate[model=dynamo/zai-org/GLM-4.7-Flash,output=child-b.md,outputMode=file-only] "Work only in the current workspace. Create logs/b.md with a concise result. Read it back. End with CHILD_B_DONE." -> delegate[model=dynamo/zai-org/GLM-4.7-Flash,output=child-c.md,outputMode=file-only] "Work only in the current workspace. Create logs/c.md with a concise result. Read it back. End with CHILD_C_DONE."
```

Prefer `delegate` for fresh-context lifecycle plumbing tests.

## Forked Context

Forked context is a first-class scenario: it is how agents such as `worker`,
`planner`, and `oracle` inherit the parent conversation. Start with one forked
child after the parent has completed at least one normal turn, and do not pass a
custom `--session-dir` while validating fork behavior. Let Pi persist sessions
where it normally does and collect the session paths from Pi/pi-subagents output
afterward.

```text
/run worker[model=dynamo/zai-org/GLM-4.7-Flash,output=fork-worker.md,outputMode=file-only] "Use the inherited parent context. Work only in the current workspace. Create logs/fork-worker.md with one follow-up note, read it back, and end with CHILD_FORK_DONE." --fork
```

Once the single fork passes, scale to parallel forked children:

```text
/parallel worker[model=dynamo/zai-org/GLM-4.7-Flash,output=fork-a.md,outputMode=file-only] "Use inherited context. Create logs/fork-a.md, read it back, end CHILD_FORK_A_DONE." -> worker[model=dynamo/zai-org/GLM-4.7-Flash,output=fork-b.md,outputMode=file-only] "Use inherited context. Create logs/fork-b.md, read it back, end CHILD_FORK_B_DONE." --fork
```

For Dynamo/SGLang, the important expectation is that forked children may share
the parent's prefix KV. The SGLang session-radix cache supports multiple
session ids on the same radix node; closing one child must remove only that
child's holder and must not free a node still held by the parent or another
forked child.

After Pi reports the children complete, keep talking to the parent without
subagents:

```text
Do not call subagent. Inspect the child artifacts in this workspace and summarize them. End with PARENT_AFTER_CHILDREN_OK.
```

Then send one final parent-only turn:

```text
One final parent-only turn. Do not call subagent. Rank the top two artifacts for follow-up and give one reason each. End with PARENT_FINAL_OK.
```

Exit the Pi session with `/quit`, then wait for the `script(1)` process to exit
with code 0.

## Verify Evidence

Collect the artifact paths in the final report:

- Pi transcript: `${RUN_ROOT}/pi-terminal.typescript`
- Pi sessions: paths printed by Pi/pi-subagents in the transcript or child result
- Dynamo trace: the `dynamo-request-trace.jsonl` path printed by the launcher
- frontend and worker logs from the launcher run directory

Useful checks:

```bash
TRACE_PATH=/path/from/launcher/dynamo-request-trace.jsonl
FRONTEND_LOG=/path/from/launcher/logs/frontend.log
WORKER_LOG=/path/from/launcher/logs/worker.log

rg -n "CHILD_.*_DONE|PARENT_AFTER_CHILDREN_OK|PARENT_FINAL_OK" "$RUN_ROOT/pi-terminal.typescript"

jq -s '{
  events: length,
  agent_context_rows: (map(select(.event.agent_context? != null)) | length),
  output_tokens_total: (map(.event.request.output_tokens // 0) | add),
  input_lengths: {
    min: (map(.event.request.replay.input_length // 0) | min),
    max: (map(.event.request.replay.input_length // 0) | max)
  },
  first_ms: .[0].event.event_time_unix_ms,
  last_ms: .[-1].event.event_time_unix_ms
}' "$TRACE_PATH"

rg -n "Removing session affinity|close_session response|release_session" "$FRONTEND_LOG" "$WORKER_LOG"

ps -u "$USER" -o pid,args | rg 'launch-agg-agent|dynamo\.frontend|dynamo\.sglang|sglang::|aiperf|pi --model' | rg -v 'rg|bash -lc ps' || true
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader
```

The lifecycle ordering to prove:

1. Child LLM requests carry child session ids.
2. Parent-only turns still carry the parent session id.
3. The server is stopped and GPUs return to baseline.

With Dynamo request-trace unification (#10701 and later), session identity
lives on the same `dynamo.request.trace.v1` rows as request metrics. If trace
rows are present but `agent_context_rows` is zero, check Dynamo tracing and the
provider package install.

## Troubleshooting

- `401 "Invalid username or password."` in child sessions means a child did not
  use the Dynamo model. Add `model=dynamo/<served-model>` to every subagent
  step or configure `subagents.agentOverrides`.
- `Failed to create forked subagent session` means a forked-context child could
  not branch from the parent session. First retry with Pi's normal session
  storage and no custom `--session-dir`; if it still fails, run
  `/subagents-doctor` inside Pi and capture the exact parent/child session paths.
- No trace rows means Dynamo was not launched with `DYN_REQUEST_TRACE=1` or the
  trace path points at the wrong run.
- Trace rows without `agent_context` usually mean Dynamo tracing is disabled or
  the provider install is stale.
