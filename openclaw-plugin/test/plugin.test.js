// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import assert from "node:assert/strict";
import test from "node:test";
import plugin from "../index.js";

function registerPlugin(entries = {}) {
  let provider;
  const hooks = {};
  plugin.register({
    on(name, handler) {
      hooks[name] = handler;
    },
    registerProvider(value) {
      provider = value;
    },
    runtime: {
      agent: {
        session: {
          getSessionEntry({ sessionKey }) {
            return entries[sessionKey];
          },
          listSessionEntries() {
            return Object.entries(entries).map(([sessionKey, entry]) => ({ sessionKey, entry }));
          },
        },
      },
    },
  });
  return { hooks, provider };
}

function registeredProvider() {
  return registerPlugin().provider;
}

test("registers a Dynamo provider that adds the OpenClaw session ID", () => {
  const provider = registeredProvider();
  const stream = provider.wrapStreamFn({ streamFn: (_model, _context, options) => options });

  assert.equal(provider.id, "dynamo");
  assert.deepEqual(stream({}, {}, { sessionId: " openclaw-session ", headers: { "x-test": "1" } }), {
    sessionId: " openclaw-session ",
    headers: {
      "x-test": "1",
      "x-dynamo-session-id": "openclaw-session",
    },
  });
});

test("preserves an explicit Dynamo session header case-insensitively", () => {
  const provider = registeredProvider();
  const stream = provider.wrapSimpleCompletionStreamFn({
    streamFn: (_model, _context, options) => options,
  });

  assert.deepEqual(
    stream({}, {}, {
      sessionId: "runtime-session",
      headers: { "X-Dynamo-Session-ID": "explicit-session" },
    }),
    {
      sessionId: "runtime-session",
      headers: { "X-Dynamo-Session-ID": "explicit-session" },
    },
  );
});

test("leaves requests without a session ID unchanged", () => {
  const provider = registeredProvider();
  const options = { headers: { "x-test": "1" } };
  const stream = provider.wrapStreamFn({ streamFn: (_model, _context, value) => value });

  assert.equal(stream({}, {}, options), options);
});

test("adds immediate parent identity for OpenClaw subagents", () => {
  const { hooks, provider } = registerPlugin({
    "agent:main:main": { sessionId: "parent", updatedAt: 1 },
    "agent:main:subagent:child": {
      sessionId: "child",
      spawnedBy: "agent:main:main",
      updatedAt: 2,
    },
  });
  hooks.before_model_resolve({}, {
    sessionId: "child",
    sessionKey: "agent:main:subagent:child",
  });
  const stream = provider.wrapStreamFn({
    agentId: "main",
    streamFn: (_model, _context, options) => options,
  });

  assert.deepEqual(stream({}, {}, { sessionId: "child" }).headers, {
    "x-dynamo-session-id": "child",
    "x-dynamo-parent-session-id": "parent",
  });

  hooks.session_end({ sessionId: "child", nextSessionId: "child-after-compaction" });
  assert.equal(
    stream({}, {}, { sessionId: "child-after-compaction" }).headers["x-dynamo-parent-session-id"],
    "parent",
  );
});
