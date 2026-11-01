# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Upstream Harbor adapter for Pi through Dynamo."""

import shlex
from typing import override

from harbor.agents.installed.base import with_prompt_template
from harbor.agents.installed.pi import Pi
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext


class DynamoPi(Pi):
    """Pi with the local Dynamo provider and Harbor's per-trial session ID."""

    def __init__(
        self,
        *args,
        provider_path: str = "/opt/pi-dynamo-provider",
        **kwargs,
    ):
        self._provider_path = provider_path
        super().__init__(*args, **kwargs)

    @override
    async def install(self, environment: BaseEnvironment) -> None:
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

    def _dynamo_env(self) -> dict[str, str]:
        base_url = self._get_env("DYNAMO_BASE_URL")
        if not base_url:
            raise ValueError("DYNAMO_BASE_URL is required for DynamoPi")
        if not self.session_id:
            raise RuntimeError("Harbor did not assign an agent session ID")
        return {
            "DYNAMO_BASE_URL": base_url,
            "DYNAMO_API_KEY": self._get_env("DYNAMO_API_KEY") or "dynamo-local",
            "DYN_AGENT_SESSION_ID": self.session_id,
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
        env = self._dynamo_env()
        await self.exec_as_agent(
            environment,
            command=(
                ". ~/.nvm/nvm.sh; "
                "pi --print --mode json --no-session "
                f"--provider dynamo --model {shlex.quote(model_name)} {cli_flags}"
                f"{shlex.quote(instruction)} "
                "2>&1 </dev/null | grep -v '\"type\":\"message_update\"' "
                "| stdbuf -oL tee /logs/agent/pi.txt"
            ),
            env=env,
        )
