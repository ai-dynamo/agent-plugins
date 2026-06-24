// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import assert from "node:assert/strict";
import { mkdtemp, mkdir, readFile, rm, writeFile } from "node:fs/promises";
import { createServer } from "node:http";
import { tmpdir } from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { spawn } from "node:child_process";

const pluginRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const runRoot = await mkdtemp(path.join(tmpdir(), "openclaw-dynamo-smoke-"));
const stateDir = path.join(runRoot, "state");
const workspace = path.join(runRoot, "workspace");
const rootSessionId = "openclaw-ci-root";
const requests = [];

function responseCompleted(output) {
  return {
    type: "response.completed",
    response: {
      id: "resp_openclaw_ci",
      status: "completed",
      output,
      usage: { input_tokens: 1, output_tokens: 1, total_tokens: 2 },
    },
  };
}

function textEvents(text) {
  const item = {
    type: "message",
    id: "msg_openclaw_ci",
    role: "assistant",
    status: "completed",
    content: [{ type: "output_text", text, annotations: [] }],
  };
  return [
    {
      type: "response.output_item.added",
      item: { ...item, status: "in_progress", content: [] },
    },
    {
      type: "response.output_text.delta",
      item_id: item.id,
      output_index: 0,
      content_index: 0,
      delta: text,
    },
    {
      type: "response.output_text.done",
      item_id: item.id,
      output_index: 0,
      content_index: 0,
      text,
    },
    { type: "response.output_item.done", item },
    responseCompleted([item]),
  ];
}

function toolEvents(name, argumentsValue) {
  const argumentsJson = JSON.stringify(argumentsValue);
  const item = {
    type: "function_call",
    id: `fc_${name}`,
    call_id: `call_${name}`,
    name,
    arguments: argumentsJson,
  };
  return [
    {
      type: "response.output_item.added",
      item: { ...item, arguments: "" },
    },
    {
      type: "response.function_call_arguments.delta",
      item_id: item.id,
      output_index: 0,
      delta: argumentsJson,
    },
    { type: "response.output_item.done", item },
    responseCompleted([item]),
  ];
}

function writeJson(response, body) {
  response.writeHead(200, { "content-type": "application/json" });
  response.end(JSON.stringify(body));
}

function writeSse(response, events) {
  const body = `${events.map((event) => `data: ${JSON.stringify(event)}\n\n`).join("")}data: [DONE]\n\n`;
  response.writeHead(200, {
    "content-type": "text/event-stream",
    "content-length": Buffer.byteLength(body),
  });
  response.end(body);
}

let rootRequestCount = 0;
const server = createServer((request, response) => {
  void (async () => {
    const url = new URL(request.url ?? "/", "http://127.0.0.1");
    if (request.method === "GET" && url.pathname === "/v1/models") {
      writeJson(response, { object: "list", data: [{ id: "openclaw-ci-model" }] });
      return;
    }
    if (request.method !== "POST" || url.pathname !== "/v1/responses") {
      response.writeHead(404).end();
      return;
    }

    for await (const _chunk of request) {
      // Drain the request before replying so OpenClaw can reuse the connection.
    }
    const sessionId = String(request.headers["x-dynamo-session-id"] ?? "");
    const parentSessionId = String(request.headers["x-dynamo-parent-session-id"] ?? "");
    requests.push({ sessionId, parentSessionId, path: url.pathname });

    if (parentSessionId) {
      writeSse(response, textEvents("CHILD_OK"));
      return;
    }

    rootRequestCount += 1;
    if (rootRequestCount === 1) {
      writeSse(
        response,
        toolEvents("sessions_spawn", {
          runtime: "subagent",
          mode: "run",
          taskName: "ci_child",
          task: "Reply exactly CHILD_OK",
        }),
      );
    } else if (rootRequestCount === 2) {
      writeSse(response, toolEvents("sessions_yield", {}));
    } else {
      writeSse(response, textEvents("PARENT_OK"));
    }
  })().catch((error) => {
    response.writeHead(500).end(String(error));
  });
});

async function runOpenClaw(args, env, timeoutMs = 90_000) {
  const child = spawn("openclaw", args, { env, cwd: workspace });
  let stdout = "";
  let stderr = "";
  child.stdout.on("data", (chunk) => (stdout += chunk));
  child.stderr.on("data", (chunk) => (stderr += chunk));
  const timer = setTimeout(() => child.kill("SIGKILL"), timeoutMs);
  const code = await new Promise((resolve, reject) => {
    child.once("error", reject);
    child.once("exit", resolve);
  });
  clearTimeout(timer);
  assert.equal(code, 0, `openclaw ${args.join(" ")} failed\n${stdout}\n${stderr}`);
  return { stdout, stderr };
}

await mkdir(stateDir, { recursive: true });
await mkdir(workspace, { recursive: true });
await new Promise((resolve, reject) => {
  server.once("error", reject);
  server.listen(0, "127.0.0.1", resolve);
});

try {
  const address = server.address();
  assert(address && typeof address !== "string");
  const env = { ...process.env, OPENCLAW_STATE_DIR: stateDir };

  await runOpenClaw(["plugins", "install", "--link", pluginRoot], env);
  await runOpenClaw(["plugins", "enable", "dynamo"], env);

  const configPath = path.join(stateDir, "openclaw.json");
  const config = JSON.parse(await readFile(configPath, "utf8"));
  config.gateway = { auth: { mode: "token", token: "openclaw-ci-token" } };
  config.tools = { profile: "coding" };
  config.models = {
    providers: {
      dynamo: {
        baseUrl: `http://127.0.0.1:${address.port}/v1`,
        apiKey: "openclaw-ci",
        api: "openai-responses",
        models: [
          {
            id: "openclaw-ci-model",
            name: "OpenClaw CI Model",
            contextWindow: 128000,
            maxTokens: 4096,
          },
        ],
      },
    },
  };
  config.agents = {
    defaults: {
      workspace,
      model: { primary: "dynamo/openclaw-ci-model" },
    },
  };
  await writeFile(configPath, `${JSON.stringify(config, null, 2)}\n`);

  await runOpenClaw(
    [
      "agent",
      "--local",
      "--session-id",
      rootSessionId,
      "--thinking",
      "off",
      "--timeout",
      "60",
      "--json",
      "--message",
      "Spawn one child and wait for it.",
    ],
    env,
  );

  const rootRequests = requests.filter(({ parentSessionId }) => !parentSessionId);
  const childRequests = requests.filter(({ parentSessionId }) => parentSessionId);
  assert(rootRequests.length >= 2, `expected at least 2 root requests, got ${requests.length}`);
  assert(childRequests.length >= 1, `expected a child request, got ${JSON.stringify(requests)}`);
  assert.deepEqual(new Set(rootRequests.map(({ sessionId }) => sessionId)), new Set([rootSessionId]));
  for (const request of childRequests) {
    assert(request.sessionId && request.sessionId !== rootSessionId);
    assert.equal(request.parentSessionId, rootSessionId);
    assert.equal(request.path, "/v1/responses");
  }
  console.log(
    JSON.stringify({
      openclaw: process.env.OPENCLAW_TEST_VERSION ?? "unknown",
      rootRequests: rootRequests.length,
      childRequests: childRequests.length,
      childSessionIds: [...new Set(childRequests.map(({ sessionId }) => sessionId))],
    }),
  );
} finally {
  await new Promise((resolve) => server.close(resolve));
  await rm(runRoot, { recursive: true, force: true });
}
