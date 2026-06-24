# OpenClaw Dynamo Provider

OpenClaw provider plugin that copies the current OpenClaw `sessionId` into
`x-dynamo-session-id` on each request sent through the `dynamo` provider. Native
subagents also send their immediate parent's ID in `x-dynamo-parent-session-id`.

## Install

```bash
git clone https://github.com/ai-dynamo/agent-plugins.git ~/agent-plugins
openclaw plugins install --link ~/agent-plugins/openclaw-plugin
openclaw plugins enable dynamo
```

## Configure

Add a Dynamo-backed model to `~/.openclaw/openclaw.json`:

```json5
{
  models: {
    providers: {
      dynamo: {
        baseUrl: "http://127.0.0.1:8000/v1",
        apiKey: "dynamo-local",
        api: "openai-responses",
        models: [
          {
            id: "zai-org/GLM-4.7-Flash",
            name: "Dynamo GLM 4.7 Flash",
            reasoning: true,
            contextWindow: 128000,
            maxTokens: 8192,
          },
        ],
      },
    },
  },
  agents: {
    defaults: {
      model: { primary: "dynamo/zai-org/GLM-4.7-Flash" },
    },
  },
}
```

The plugin preserves an explicitly supplied `x-dynamo-session-id`. Session
headers carry identity only; they do not enable sticky routing.

## Validate

```bash
cd openclaw-plugin
npm test
```
