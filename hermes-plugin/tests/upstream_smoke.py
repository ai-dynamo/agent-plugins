# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise the plugin against Hermes' real AIAgent class."""

import importlib.util
import pathlib

from run_agent import AIAgent


PLUGIN_PATH = pathlib.Path(__file__).resolve().parents[1] / "__init__.py"


def main() -> None:
    original = AIAgent._create_openai_client

    def sentinel(self, client_kwargs, *, reason, shared):
        return client_kwargs

    try:
        AIAgent._create_openai_client = sentinel
        spec = importlib.util.spec_from_file_location("dynamo_session_plugin", PLUGIN_PATH)
        assert spec and spec.loader
        plugin = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(plugin)

        hooks = {}
        plugin.register(type("Context", (), {"register_hook": lambda _, name, fn: hooks.setdefault(name, fn)})())
        hooks["pre_api_request"]()

        agent = object.__new__(AIAgent)
        agent.session_id = "hermes-upstream-smoke"
        result = agent._create_openai_client(
            {"default_headers": {"x-test": "1"}},
            reason="ci",
            shared=False,
        )
        assert result["default_headers"] == {
            "x-test": "1",
            "x-dynamo-session-id": "hermes-upstream-smoke",
        }
    finally:
        AIAgent._create_openai_client = original


if __name__ == "__main__":
    main()
