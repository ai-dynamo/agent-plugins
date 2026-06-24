# Hermes Dynamo Session Plugin

Hermes plugin that copies the current Hermes `session_id` into Dynamo's `x-dynamo-session-id` request header.

Tested with Hermes `0.17.0` at `a7983d5`. CI also validates the plugin against
the latest Hermes `main` using the real plugin loader and `AIAgent` class.

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
