# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import pathlib
import sys
import types
import unittest


PLUGIN_PATH = pathlib.Path(__file__).resolve().parents[1] / "__init__.py"


def load_plugin():
    spec = importlib.util.spec_from_file_location("dynamo_session_plugin", PLUGIN_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class DynamoSessionPluginTest(unittest.TestCase):
    def test_pre_api_request_patches_openai_client_creation(self):
        plugin = load_plugin()
        calls = []

        class AIAgent:
            session_id = "hermes-session"

            def _create_openai_client(self, client_kwargs, *, reason, shared):
                return client_kwargs

        previous = sys.modules.get("run_agent")
        sys.modules["run_agent"] = types.SimpleNamespace(AIAgent=AIAgent)
        try:
            ctx = types.SimpleNamespace(
                register_hook=lambda name, callback: calls.append((name, callback))
            )
            plugin.register(ctx)
            calls[0][1]()
            result = AIAgent()._create_openai_client(
                {"default_headers": {"x-test": "1"}},
                reason="test",
                shared=False,
            )
        finally:
            if previous is None:
                sys.modules.pop("run_agent", None)
            else:
                sys.modules["run_agent"] = previous

        self.assertEqual(calls[0][0], "pre_api_request")
        self.assertEqual(result["default_headers"]["x-test"], "1")
        self.assertEqual(
            result["default_headers"]["x-dynamo-session-id"],
            "hermes-session",
        )


if __name__ == "__main__":
    unittest.main()
