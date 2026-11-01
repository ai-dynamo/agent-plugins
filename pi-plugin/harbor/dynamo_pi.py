# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Upstream Harbor adapter for Pi through Dynamo."""

import json
import shlex
from typing import override

from harbor.agents.installed.base import with_prompt_template
from harbor.agents.installed.pi import Pi
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext

# Written by a prepared task image. Records the baked Pi version so install()
# can skip the runtime installation, which needs registry egress that sandboxed
# environments often do not have.
PREPARED_RUNTIME_MARKER = "/opt/pi-runtime.json"


class DynamoPi(Pi):
    """Pi with the local Dynamo provider and Harbor's per-trial session ID."""

    def __init__(
        self,
        *args,
        provider_path: str = "/opt/pi-dynamo-provider",
        **kwargs,
    ):
        self._provider_path = provider_path
        self._prepared = False
        super().__init__(*args, **kwargs)

    @override
    async def install(self, environment: BaseEnvironment) -> None:
        marker = await self._read_prepared_runtime(environment)
        if marker is not None:
            baked = marker.get("pi")
            if not isinstance(baked, str) or not baked:
                raise RuntimeError(
                    f"{PREPARED_RUNTIME_MARKER} does not record a Pi version"
                )
            if self._version and self._version != baked:
                raise RuntimeError(
                    f"Prepared image has Pi {baked} but the job pins {self._version}"
                )
            self._version = baked
            self._prepared = True
            self.logger.info("Using the prepared Pi %s runtime", baked)
            await self._start_tool_event_tunnel(environment, install_package=False)
            return

        self.logger.warning(
            "%s not found; installing Pi and the provider. This needs egress to "
            "GitHub, nodejs.org, the npm registry, and the image's apt mirrors.",
            PREPARED_RUNTIME_MARKER,
        )
        await super().install(environment)
        await self.exec_as_agent(
            environment,
            command=(
                "set -euo pipefail; . ~/.nvm/nvm.sh; "
                f"provider={shlex.quote(self._provider_path)}; "
                'test -f "$provider/package.json"; '
                'work="$HOME/pi-dynamo-provider"; rm -rf "$work"; '
                'mkdir -p "$work"; cp -a "$provider"/. "$work"; '
                'cd "$work"; npm ci; npm run build; pi install "$work"'
            ),
        )
        await self._start_tool_event_tunnel(environment, install_package=True)

    async def _read_prepared_runtime(
        self, environment: BaseEnvironment
    ) -> dict | None:
        """Return the prepared-runtime marker, or None when the image has none."""
        result = await self.exec_as_agent(
            environment,
            command=f"cat {PREPARED_RUNTIME_MARKER} 2>/dev/null || true",
        )
        text = (result.stdout or "").strip()
        if not text:
            return None
        try:
            marker = json.loads(text)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"{PREPARED_RUNTIME_MARKER} is not valid JSON") from exc
        if not isinstance(marker, dict):
            raise RuntimeError(f"{PREPARED_RUNTIME_MARKER} must hold a JSON object")
        return marker

    async def _start_tool_event_tunnel(
        self, environment: BaseEnvironment, install_package: bool
    ) -> None:
        """Start the TLS client for tool events when the job configures one.

        The provider publishes tool events over plain ZMQ. Sandboxes whose
        egress forwards only HTTP and TLS need stunnel to carry that stream to
        Dynamo, so the job points this variable at a stunnel client config.
        """
        tunnel_config = self._get_env("DYN_REQUEST_TRACE_TOOL_EVENTS_STUNNEL_CONFIG")
        if not tunnel_config:
            return
        command = f"stunnel {shlex.quote(tunnel_config)}"
        if install_package:
            command = (
                "apt-get update -qq && "
                "DEBIAN_FRONTEND=noninteractive apt-get install -y -qq stunnel4 && "
                + command
            )
        await self.exec_as_root(
            environment,
            command=command,
            timeout_sec=180 if install_package else None,
        )

    def _dynamo_env(self) -> dict[str, str]:
        base_url = self._get_env("DYNAMO_BASE_URL")
        if not base_url:
            raise ValueError("DYNAMO_BASE_URL is required for DynamoPi")
        if not self.session_id:
            raise RuntimeError("Harbor did not assign an agent session ID")
        session_final = self._get_env("DYN_AGENT_SESSION_FINAL") or "1"
        return {
            "DYNAMO_BASE_URL": base_url,
            "DYNAMO_API_KEY": self._get_env("DYNAMO_API_KEY") or "dynamo-local",
            "DYN_AGENT_SESSION_ID": self.session_id,
            "DYN_AGENT_SESSION_FINAL": (
                "1" if session_final.lower() in {"1", "true", "yes", "on"} else "0"
            ),
        }

    @override
    @with_prompt_template
    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        if not self.model_name or not self.model_name.startswith("dynamo/"):
            raise ValueError("DynamoPi requires a model named dynamo/<model-name>")

        model_name = self.model_name.removeprefix("dynamo/")
        if not model_name:
            raise ValueError("DynamoPi requires a model named dynamo/<model-name>")

        cli_flags = self.build_cli_flags()
        if cli_flags:
            cli_flags += " "
        final_body = shlex.quote(
            json.dumps(
                {
                    "model": model_name,
                    "messages": [{"role": "user", "content": "."}],
                    "max_tokens": 1,
                    "stream": False,
                }
            )
        )
        env = self._dynamo_env()
        final_command = ""
        if env["DYN_AGENT_SESSION_FINAL"] == "1":
            final_command = (
                "rc=$?; "
                "curl --fail --silent --show-error --retry 3 --retry-all-errors "
                '"$DYNAMO_BASE_URL/chat/completions" '
                "-H 'Content-Type: application/json' "
                '-H "Authorization: Bearer $DYNAMO_API_KEY" '
                '-H "x-dynamo-session-id: $DYN_AGENT_SESSION_ID" '
                "-H 'x-dynamo-session-final: true' "
                f"--data {final_body} >/dev/null || exit 70; "
                'exit "$rc"'
            )
        # A prepared image puts node and pi on PATH; the installed path needs nvm.
        shell_prefix = "" if self._prepared else ". ~/.nvm/nvm.sh; "
        await self.exec_as_agent(
            environment,
            command=(
                f"{shell_prefix}"
                "pi --print --mode json --no-session "
                f"--provider dynamo --model {shlex.quote(model_name)} {cli_flags}"
                f"{shlex.quote(instruction)} "
                "2>&1 </dev/null | grep -v '\"type\":\"message_update\"' "
                "| stdbuf -oL tee /logs/agent/pi.txt; "
                f"{final_command}"
            ),
            env=env,
        )
