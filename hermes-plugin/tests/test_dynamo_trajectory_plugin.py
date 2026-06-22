# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import pathlib
import types
import unittest


PLUGIN_PATH = pathlib.Path(__file__).resolve().parents[1] / "__init__.py"


def load_plugin():
    spec = importlib.util.spec_from_file_location("dynamo_trajectory_plugin", PLUGIN_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class DynamoTrajectoryPluginTest(unittest.TestCase):
    def test_adds_session_id_as_dynamo_trajectory_header(self):
        plugin = load_plugin()

        result = plugin.add_dynamo_trajectory_header(
            session_id="hermes-session",
            request={"model": "qwen", "extra_headers": {"x-test": "1"}},
        )

        self.assertEqual(
            result,
            {
                "request": {
                    "model": "qwen",
                    "extra_headers": {
                        "x-test": "1",
                        "x-dynamo-trajectory-id": "hermes-session",
                    },
                }
            },
        )

    def test_preserves_explicit_dynamo_trajectory_header(self):
        plugin = load_plugin()

        result = plugin.add_dynamo_trajectory_header(
            session_id="hermes-session",
            request={"extra_headers": {"x-dynamo-trajectory-id": "explicit"}},
        )

        self.assertEqual(result["request"]["extra_headers"]["x-dynamo-trajectory-id"], "explicit")

    def test_skips_without_session_id(self):
        plugin = load_plugin()

        self.assertIsNone(plugin.add_dynamo_trajectory_header(request={"model": "qwen"}))

    def test_registers_llm_request_middleware(self):
        plugin = load_plugin()
        calls = []
        ctx = types.SimpleNamespace(
            register_middleware=lambda kind, callback: calls.append((kind, callback))
        )

        plugin.register(ctx)

        self.assertEqual(calls, [("llm_request", plugin.add_dynamo_trajectory_header)])


if __name__ == "__main__":
    unittest.main()
