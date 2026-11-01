# Harbor Pi Adapter

Harbor adapter for running Pi against Dynamo with one stable Dynamo session per Harbor trial.

## Use

Upstream Harbor's built-in Pi adapter does not install external Pi providers and runs Pi with `--no-session`. `dynamo_pi:DynamoPi` installs the provider from a read-only mount and maps Harbor's per-trial `agent.session_id` to `DYN_AGENT_SESSION_ID`.

```bash
export PYTHONPATH=/absolute/path/to/agent-plugins/pi-plugin/harbor${PYTHONPATH:+:$PYTHONPATH}

harbor run \
  --agent dynamo_pi:DynamoPi \
  --model dynamo/<model-id> \
  --agent-env DYNAMO_BASE_URL=http://<dynamo-host>:8000/v1 \
  --mounts '[{"type":"bind","source":"/absolute/path/to/agent-plugins/pi-plugin","target":"/opt/pi-dynamo-provider","read_only":true}]'
```

## Host networking

Large local-Docker runs should pass [`host-network.yml`](host-network.yml) with Harbor's `--extra-docker-compose` option to avoid creating one bridge network per trial:

```bash
harbor run \
  --extra-docker-compose /absolute/path/to/agent-plugins/pi-plugin/harbor/host-network.yml \
  ...
```

The full Dynamo and SWE-bench launch sequence is documented in Dynamo's ThunderAgent guide.
