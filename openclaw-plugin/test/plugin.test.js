// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import assert from "node:assert/strict";
import test from "node:test";
import plugin from "../index.js";

function registeredProvider() {
  let provider;
  plugin.register({
    registerProvider(value) {
      provider = value;
    },
  });
  return provider;
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
