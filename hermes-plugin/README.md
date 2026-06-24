# Hermes Dynamo Session Plugin

Hermes plugin that copies the current Hermes `session_id` into Dynamo's `x-dynamo-session-id` request header.

## Install

```bash
git clone https://github.com/ai-dynamo/agent-plugins.git ~/agent-plugins
mkdir -p ~/.hermes/plugins
ln -sfnT ~/agent-plugins/hermes-plugin ~/.hermes/plugins/dynamo_session
hermes plugins enable dynamo_session
```

## Validate

```bash
python3 -m unittest discover -s hermes-plugin/tests
```
