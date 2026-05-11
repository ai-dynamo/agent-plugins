<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# pi-dynamo-provider

Pi extension registering a `dynamo` provider for Dynamo's OpenAI-compatible chat-completions endpoint. Three source files in `src/` (~650 lines total):

- `index.ts` — extension entrypoint; calls `readDynamoConfig`, discovers models via `/v1/models`, registers the provider, wires the tool-event relay.
- `dynamo-provider.ts` — config + agent_context construction + streamSimple wrapper. Reads `DYN_AGENT_*` and `PI_SUBAGENT_*` env vars; emits `nvext.agent_context` on every LLM request.
- `tool-relay.ts` — ZMQ PUSH publisher for Pi tool events. Connects to a Dynamo-bound PULL endpoint. Wire format: `[topic, seq_be_u64, msgpack(AgentTraceRecord)]`.

## Build, test, check

```bash
npm install
npm run check     # tsc --noEmit (strict + exactOptionalPropertyTypes + noUncheckedIndexedAccess)
npm test          # vitest run
npm run build     # tsc -p tsconfig.build.json → dist/
```

Tests live in `test/` as siblings of `src/`. Use vitest's `describe`/`it`/`expect`. Mirror the existing structure: one test file per source file, fixture data inline rather than separate fixture files.

`test/integration/smoke.mjs` is the out-of-band end-to-end check — driven by `scripts/integration-smoke.sh`, not vitest. It boots Dynamo's frontend + mocker, sends one real chat completion, and asserts `nvext.agent_context` round-trips into the trace JSONL. Two cases: top-level agent_context and the pi-subagents bridge. Mocker output is garbage; assertions only target the trace envelope. CI runs `pip install --pre ai-dynamo` to fetch the latest NVIDIA-validated wheel (~30s install) rather than building from source — broken-build commits on upstream main are filtered out by the wheel-publish gate, so PR CI stays green when upstream is mid-refactor. `workflow_dispatch` accepts a `dynamo_spec` input for ad-hoc validation against a specific pip spec, including `git+https://github.com/ai-dynamo/dynamo@<sha>` for source builds.

## Coding standards

- TypeScript strict mode. Don't add `any`; prefer `unknown` + narrow.
- SPDX header on every source/test file: `// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.` + `// SPDX-License-Identifier: Apache-2.0`.
- ESM only (`"type": "module"`); import `.js` extensions in source even though the file is `.ts`.
- `Pick<SimpleStreamOptions, "sessionId">` style — narrow type at the use site rather than introducing fresh interfaces.
- No emojis anywhere in code or comments.
- Mermaid diagrams in markdown, not ASCII art.
- Comments explain WHY, not WHAT. Read the bridge block in `readDynamoConfig` for the tone — it covers the non-obvious env-var inheritance behavior in a few lines.
- No new top-level exports unless they're part of the public surface; the package re-exports `dynamo-provider` and `tool-relay` from `index.ts`, that's the entire API.

## Architecture invariants

- **One-way knowledge flow**: pi-dynamo-provider knows about pi-subagents' env contract (`PI_SUBAGENT_*` vars). pi-subagents never knows about us. Keep it that way — don't propose changes to pi-subagents to fix problems we can solve here.
- **No `pi-mono` core patches**. Everything we want must be expressible through the public `ExtensionAPI` (`registerProvider`, `streamSimple` wrapper, tool-event hooks). If you find yourself wanting a Pi core change, the answer is almost always "find a different angle in this repo first."
- **Dynamo owns the ZMQ bind side** for tool events. We're a PUSH connect-side producer. Don't try to bind.
- **Trace data is best-effort, not durable**. Don't add retry loops, persistent queues, or back-pressure that would block Pi. The `DynamoToolEventPublisher` drops events when its bounded queue is full; that's correct.

## Env-var naming contract

| Prefix | Direction | Examples |
|---|---|---|
| `DYNAMO_*` | client config (we read) | `DYNAMO_BASE_URL`, `DYNAMO_API_KEY` |
| `DYN_AGENT_*` | dynamo agent context (we read + emit) | `DYN_AGENT_SESSION_ID`, `DYN_AGENT_TRAJECTORY_ID`, `DYN_AGENT_TOOL_EVENTS_ZMQ_ENDPOINT` |
| `PI_SUBAGENT_*` | pi-subagents bookkeeping (we read only) | `PI_SUBAGENT_CHILD`, `PI_SUBAGENT_RUN_ID`, `PI_SUBAGENT_CHILD_AGENT`, `PI_SUBAGENT_CHILD_INDEX` |
| `OPENAI_BASE_URL` | OpenAI-compatibility fallback (we read) | only consulted when `DYNAMO_BASE_URL` is unset |

Don't introduce new prefixes. If you need a new var, justify which existing namespace it belongs in.

## Git workflow

- Feature branches as `<username>/<short-name>`, forked from `main`.
- DCO sign-off required: `git commit -s` (the `Signed-off-by:` trailer).
- Use HEREDOCs for multi-line commit messages so formatting survives.
- Co-authored-by Claude trailers are fine and welcomed when applicable.
- Don't bypass hooks (`--no-verify`) unless explicitly asked.
- Push to a remote branch, open a PR with a `Test plan` checklist (see PR #1 for the template).

## CONTRIBUTING.md context

External contributions are not currently accepted. This is an NVIDIA-internal coordination layer between Pi and Dynamo; both upstreams move and we sync here. The repo's audience for now is NVIDIA-dev members doing first-party work.

## What to leave alone

- The `nvext.agent_context` schema field names match ATIF (`session_type_id`, `session_id`, `trajectory_id`, `parent_trajectory_id`). Don't rename them — downstream tooling in Dynamo's converter and benchmark stack joins on these.
- The `phase: "reasoning"` field is deliberately hardcoded; it tags the LLM call as an agent reasoning step (vs. e.g. a synthesis or grading step). Adding other phase values requires Dynamo-side coordination.
- The `agent_trace.v1` schema is owned upstream by Dynamo (`dynamo/lib/llm/src/agents/trace/`). Don't change record shapes here without an upstream PR landing first.
- `package-lock.json` churn from npm version differences should be reverted before committing (`git checkout -- package-lock.json` if a no-op edit appears).
