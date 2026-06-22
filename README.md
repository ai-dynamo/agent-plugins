# Dynamo Agent Plugins

Small agent integrations for Dynamo request tracing.

## Layout

- `pi-plugin/` - Pi provider plugin for Dynamo's OpenAI-compatible endpoint.
- `hermes-plugin/` - Hermes middleware plugin that maps Hermes `session_id` to `x-dynamo-trajectory-id`.

Each plugin owns its own tests and install instructions.
