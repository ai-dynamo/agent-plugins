<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Dynamo agent plugins

Repo layout:

- `pi-plugin/` — Pi extension registering a `dynamo` provider for Dynamo's OpenAI-compatible chat-completions endpoint.
- `hermes-plugin/` — Hermes middleware plugin that injects Dynamo session headers from Hermes `session_id`.

The Pi plugin has three source files under `pi-plugin/src/`:

- `index.ts` — thin re-export of the light implementation.
- `src/light/provider.ts` — config + streamSimple wrapper. Reads `DYN_REQUEST_TRACE`, `DYN_AGENT_*`, and `PI_SUBAGENT_*` env vars. Always stamps `x-dynamo-session-id` / parent headers when Pi provides a session and leaves Pi `sessionId` untouched.
- `src/light/tool-relay.ts` — ZMQ PUSH publisher for Pi tool events. Connects to a Dynamo-bound PULL endpoint. Wire format: `[topic, seq_be_u64, msgpack(RequestTraceRecord)]`.

## Build, test, check

```bash
cd pi-plugin
npm install
npm run check     # tsc --noEmit (strict + exactOptionalPropertyTypes + noUncheckedIndexedAccess)
npm test          # vitest run
npm run build     # tsc -p tsconfig.build.json → dist/
```

Pi tests live in `pi-plugin/test/` as siblings of `pi-plugin/src/`. Use vitest's `describe`/`it`/`expect`. Mirror the existing structure: one test file per source file, fixture data inline rather than separate fixture files.

`pi-plugin/test/integration/smoke.mjs` is the out-of-band end-to-end check — driven by `pi-plugin/scripts/integration-smoke.sh`, not vitest. It boots Dynamo's frontend + mocker, sends one real chat completion, and asserts `x-dynamo-session-id` becomes `session_id` in the request trace JSONL. Two cases: top-level session id and the pi-subagents bridge. Mocker output is garbage; assertions only target the trace envelope. CI clones `ai-dynamo/dynamo@main` and builds from source. Cargo cache keeps warm runs ~60-90s, cold ~10 min. `workflow_dispatch` accepts a `dynamo_ref` input for ad-hoc validation against a specific branch, tag, or SHA.

For real Pi CLI lifecycle validation against a Dynamo endpoint, read `pi-plugin/skills/pi-headless-dynamo/SKILL.md` first and drive the actual interactive Pi TUI instead of faking provider requests or pi-subagents env.

Hermes plugin validation:

```bash
python3 -m unittest discover -s hermes-plugin/tests
```

## Coding standards

- TypeScript strict mode. Don't add `any`; prefer `unknown` + narrow.
- SPDX header on every source/test file: `// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.` + `// SPDX-License-Identifier: Apache-2.0`.
- ESM only (`"type": "module"`); import `.js` extensions in source even though the file is `.ts`.
- `Pick<SimpleStreamOptions, "sessionId">` style — narrow type at the use site rather than introducing fresh interfaces.
- No emojis anywhere in code or comments.
- Mermaid diagrams in markdown, not ASCII art.
- Comments explain WHY, not WHAT. Read the bridge block in `readDynamoConfig` for the tone — it covers the non-obvious env-var inheritance behavior in a few lines.
- No new Pi top-level exports unless they're part of the public surface; `pi-plugin/src/index.ts` is the package API.

## Architecture invariants

- **One-way knowledge flow**: `pi-plugin` knows about pi-subagents' env contract (`PI_SUBAGENT_*` vars). pi-subagents never knows about us. Keep it that way — don't propose changes to pi-subagents to fix problems we can solve here.
- **No `pi-mono` core patches**. Everything we want must be expressible through the public `ExtensionAPI` (`registerProvider`, `streamSimple` wrapper, tool-event hooks). If you find yourself wanting a Pi core change, the answer is almost always "find a different angle in this repo first."
- **Dynamo owns the ZMQ bind side** for tool events. We're a PUSH connect-side producer. Don't try to bind.
- **Trace data is best-effort, not durable**. Don't add retry loops, persistent queues, or back-pressure that would block Pi/Hermes. The Pi `DynamoToolEventPublisher` drops events when its bounded queue is full; that's correct.

## Env-var naming contract

| Prefix | Direction | Examples |
|---|---|---|
| `DYNAMO_*` | client config (we read) | `DYNAMO_BASE_URL`, `DYNAMO_API_KEY` |
| `DYN_AGENT_*` | optional session override / subagent parent link | `DYN_AGENT_SESSION_ID`, `DYN_AGENT_PARENT_SESSION_ID` |
| `DYN_REQUEST_TRACE*` | request trace switch and tool bridge | `DYN_REQUEST_TRACE`, `DYN_REQUEST_TRACE_TOOL_EVENTS_ZMQ_ENDPOINT` |
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

- Dynamo owns the request trace schema. The Pi provider stamps session headers for LLM requests and keeps explicit tool calls on the ZMQ trace path. The Hermes plugin only stamps request headers.
- The `request.trace.v1` schema is owned upstream by Dynamo (`dynamo/lib/llm/src/request_trace/`). Don't change record shapes here without an upstream PR landing first.
- `pi-plugin/package-lock.json` churn from npm version differences should be reverted before committing (`git checkout -- pi-plugin/package-lock.json` if a no-op edit appears).
