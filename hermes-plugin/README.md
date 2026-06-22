# Hermes Dynamo Trajectory Plugin

Hermes middleware plugin that copies the current Hermes `session_id` into Dynamo's `x-dynamo-trajectory-id` request header.

## Install

```bash
hermes plugins install /absolute/path/to/repo/hermes-plugin
hermes plugins enable dynamo_trajectory
```

## Validate

```bash
python3 -m unittest discover -s hermes-plugin/tests
```
