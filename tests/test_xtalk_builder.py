"""Tests for staged Xtalk runtime configuration."""

from __future__ import annotations

import unittest
from collections.abc import Iterable, Sequence
from typing import Any, cast
from unittest.mock import patch, sentinel

from langchain_core.language_models.fake_chat_models import FakeListChatModel

from xtalk import Xtalk, XtalkBuilder, model
from xtalk.models import Agent
from xtalk.models.agents import AgentContext, AgentOutput
from xtalk.models.agents.default import DefaultAgent
from xtalk.models.agents.tools import (
    SyncTool,
    Tool,
    ToolEngineState,
    ToolInput,
    ToolOutput,
)


class _ToolBindableFakeChatModel(FakeListChatModel):
    """Fake chat model that accepts tool binding."""

    def bind_tools(
        self,
        tools: Sequence[Any],
        **kwargs: Any,
    ) -> _ToolBindableFakeChatModel:
        """Accept tool binding and remain directly runnable."""

        del tools, kwargs
        return self


class _BuilderToolInput(ToolInput):
    """Input for the Builder integration-test tool."""

    value: str


class _BuilderToolOutput(ToolOutput):
    """Output from the Builder integration-test tool."""

    value: str


class _BuilderTool(SyncTool):
    """Native tool mounted through XtalkBuilder in the integration test."""

    name = "builder_test_tool"

    @classmethod
    def invoke(
        cls,
        tool_input: _BuilderToolInput,
        global_state: ToolEngineState,
    ) -> _BuilderToolOutput:
        """Echo the validated test input."""

        del cls, global_state
        return _BuilderToolOutput(value=tool_input.value)


@model(name="BuilderRegisteredAgent")
class _BuilderRegisteredAgent(Agent):
    """Stateless registered Agent used by Builder tests."""

    def accept(self, context: AgentContext) -> Iterable[AgentOutput]:
        """Return no output for the test context."""

        del context
        return ()

    def clone(self) -> _BuilderRegisteredAgent:
        """Return a fresh test Agent."""

        return _BuilderRegisteredAgent()

    def restore_history(self, messages: list[dict[str, Any]]) -> None:
        """Ignore restored history in the stateless test Agent."""

        del messages


class XtalkBuilderTests(unittest.TestCase):
    """Verify runtime Agent tools are attached before model construction."""

    def _capture_effective_config(
        self,
        source_config: dict,
        tools: list[Tool],
    ) -> dict:
        """Build with a mocked constructor and return its effective config."""

        builder = Xtalk.configure(source_config).add_agent_tools(tools)
        with patch.object(
            Xtalk,
            "_build_from_config_dict",
            return_value=sentinel.xtalk,
        ) as build_mock:
            result = builder.build()

        self.assertIs(result, sentinel.xtalk)
        return build_mock.call_args.args[0]

    def test_configure_returns_fluent_builder(self) -> None:
        """Return the same builder from every staged operation."""

        builder = Xtalk.configure({"llm_agent": "DefaultAgent"})
        tool = cast(Tool, object())

        self.assertIsInstance(builder, XtalkBuilder)
        self.assertIs(builder.transform_config(lambda config: config), builder)
        self.assertIs(builder.set_model(_BuilderRegisteredAgent), builder)
        self.assertIs(builder.add_agent_tools([tool]), builder)

    def test_config_transforms_run_in_order_without_mutating_source(self) -> None:
        """Apply arbitrary transforms in order to a structural config copy."""

        source_config = {
            "service_config": {
                "labels": ["source"],
            }
        }
        call_order: list[str] = []

        def append_label(config: dict[str, Any]) -> dict[str, Any]:
            call_order.append("append")
            config["service_config"]["labels"].append("first")
            return config

        def add_marker(config: dict[str, Any]) -> dict[str, Any]:
            call_order.append("mark")
            return {**config, "marker": "second"}

        builder = (
            Xtalk.configure(source_config)
            .transform_config(append_label)
            .transform_config(add_marker)
        )
        with patch.object(
            Xtalk,
            "_build_from_config_dict",
            return_value=sentinel.xtalk,
        ) as build_mock:
            result = builder.build()

        effective_config = build_mock.call_args.args[0]
        self.assertIs(result, sentinel.xtalk)
        self.assertEqual(call_order, ["append", "mark"])
        self.assertEqual(
            effective_config["service_config"]["labels"],
            ["source", "first"],
        )
        self.assertEqual(effective_config["marker"], "second")
        self.assertEqual(source_config["service_config"]["labels"], ["source"])

    def test_transform_config_validates_callable_and_return_type(self) -> None:
        """Reject invalid transform registrations and return values."""

        builder = Xtalk.configure({})
        with self.assertRaises(TypeError):
            builder.transform_config(cast(Any, None))

        with self.assertRaises(TypeError):
            builder.transform_config(cast(Any, lambda config: None)).build()

    def test_set_model_uses_registered_name_and_preserves_config(self) -> None:
        """Replace only the model type using registry and slot metadata."""

        source_config = {
            "llm_agent": {
                "type": "DefaultAgent",
                "params": {"system_prompt": "Keep this prompt."},
                "sample_option": True,
            },
            "service_config": {"enable_persistence": False},
        }
        with patch.object(
            Xtalk,
            "_build_from_config_dict",
            return_value=sentinel.xtalk,
        ) as build_mock:
            result = (
                Xtalk.configure(source_config)
                .set_model(_BuilderRegisteredAgent)
                .build()
            )

        effective_config = build_mock.call_args.args[0]
        self.assertIs(result, sentinel.xtalk)
        self.assertEqual(
            effective_config["llm_agent"],
            {
                "type": "BuilderRegisteredAgent",
                "params": {"system_prompt": "Keep this prompt."},
                "sample_option": True,
            },
        )
        self.assertEqual(source_config["llm_agent"]["type"], "DefaultAgent")
        self.assertIsNot(
            effective_config["llm_agent"]["params"],
            source_config["llm_agent"]["params"],
        )

    def test_set_model_supports_canonical_and_missing_slots(self) -> None:
        """Resolve canonical slots and create a missing registered model slot."""

        canonical_source = {"agents": "DefaultAgent"}
        with patch.object(
            Xtalk,
            "_build_from_config_dict",
            return_value=sentinel.xtalk,
        ) as build_mock:
            Xtalk.configure(canonical_source).set_model(_BuilderRegisteredAgent).build()
        self.assertEqual(
            build_mock.call_args.args[0]["agents"],
            {
                "type": "BuilderRegisteredAgent",
                "params": {},
            },
        )

        with patch.object(
            Xtalk,
            "_build_from_config_dict",
            return_value=sentinel.xtalk,
        ) as build_mock:
            Xtalk.configure({}).set_model(_BuilderRegisteredAgent).build()
        self.assertEqual(
            build_mock.call_args.args[0]["agents"],
            {
                "type": "BuilderRegisteredAgent",
                "params": {},
            },
        )

    def test_build_replaces_configured_model_before_construction(self) -> None:
        """Instantiate the registered replacement through the real loader."""

        source_config = {
            "llm_agent": "DefaultAgent",
            "service_config": {"enable_persistence": False},
        }

        application = (
            Xtalk.configure(source_config).set_model(_BuilderRegisteredAgent).build()
        )

        agent = application._models.require(Agent)
        self.assertIsInstance(agent, _BuilderRegisteredAgent)
        self.assertEqual(source_config["llm_agent"], "DefaultAgent")

    def test_set_model_rejects_unregistered_class_and_invalid_params(self) -> None:
        """Reject unregistered implementations and malformed model params."""

        class UnregisteredModel:
            pass

        with self.assertRaises(TypeError):
            Xtalk.configure({}).set_model(UnregisteredModel)

        with self.assertRaises(ValueError):
            (
                Xtalk.configure(
                    {
                        "llm_agent": {
                            "type": "DefaultAgent",
                            "params": [],
                        }
                    }
                )
                .set_model(_BuilderRegisteredAgent)
                .build()
            )

    def test_model_replacement_composes_with_agent_tools(self) -> None:
        """Preserve replacement parameters when attaching runtime tools."""

        tool = cast(Tool, object())
        source_config = {
            "llm_agent": {
                "type": "DefaultAgent",
                "params": {"system_prompt": "Keep this prompt."},
            }
        }
        with patch.object(
            Xtalk,
            "_build_from_config_dict",
            return_value=sentinel.xtalk,
        ) as build_mock:
            (
                Xtalk.configure(source_config)
                .set_model(_BuilderRegisteredAgent)
                .add_agent_tools([tool])
                .build()
            )

        effective_agent = build_mock.call_args.args[0]["llm_agent"]
        self.assertEqual(effective_agent["type"], "BuilderRegisteredAgent")
        self.assertEqual(
            effective_agent["params"],
            {
                "system_prompt": "Keep this prompt.",
                "tools": [tool],
            },
        )

    def test_runtime_tools_append_without_mutating_source_config(self) -> None:
        """Append tools in registration order using structural config copies."""

        existing_tool = cast(Tool, object())
        first_runtime_tool = cast(Tool, object())
        second_runtime_tool = cast(Tool, object())
        source_config = {
            "llm_agent": {
                "type": "DefaultAgent",
                "params": {
                    "system_prompt": "Keep this prompt.",
                    "tools": [existing_tool],
                },
            },
            "service_config": {"enable_persistence": False},
        }

        builder = Xtalk.configure(source_config)
        builder.add_agent_tools([first_runtime_tool])
        builder.add_agent_tools([second_runtime_tool])
        with patch.object(
            Xtalk,
            "_build_from_config_dict",
            return_value=sentinel.xtalk,
        ) as build_mock:
            builder.build()
        effective_config = build_mock.call_args.args[0]

        self.assertEqual(
            effective_config["llm_agent"]["params"]["tools"],
            [existing_tool, first_runtime_tool, second_runtime_tool],
        )
        self.assertEqual(
            effective_config["llm_agent"]["params"]["system_prompt"],
            "Keep this prompt.",
        )
        self.assertEqual(
            source_config["llm_agent"]["params"]["tools"],
            [existing_tool],
        )
        self.assertIsNot(effective_config, source_config)
        self.assertIsNot(
            effective_config["llm_agent"],
            source_config["llm_agent"],
        )
        self.assertIsNot(
            effective_config["llm_agent"]["params"],
            source_config["llm_agent"]["params"],
        )

    def test_tools_attach_to_shorthand_and_canonical_agent_slots(self) -> None:
        """Support string model shorthand and the canonical agents config key."""

        shorthand_tool = cast(Tool, object())
        shorthand_config = self._capture_effective_config(
            {"llm_agent": "DefaultAgent"},
            [shorthand_tool],
        )
        self.assertEqual(
            shorthand_config["llm_agent"],
            {
                "type": "DefaultAgent",
                "params": {"tools": [shorthand_tool]},
            },
        )

        canonical_tool = cast(Tool, object())
        canonical_config = self._capture_effective_config(
            {"agents": {"type": "DefaultAgent"}},
            [canonical_tool],
        )
        self.assertEqual(
            canonical_config["agents"]["params"]["tools"],
            [canonical_tool],
        )

    def test_build_mounts_native_tool_before_agent_construction(self) -> None:
        """Pass a native tool class through the real DefaultAgent loader."""

        source_config = {
            "llm_agent": {
                "type": "DefaultAgent",
                "params": {
                    "model": _ToolBindableFakeChatModel(responses=["ready"]),
                },
            },
            "service_config": {"enable_persistence": False},
        }

        application = (
            Xtalk.configure(source_config).add_agent_tools([_BuilderTool]).build()
        )

        agent = application._models.require(Agent)
        self.assertIsInstance(agent, DefaultAgent)
        assert isinstance(agent, DefaultAgent)
        self.assertIn(_BuilderTool, agent.tools)
        self.assertNotIn("tools", source_config["llm_agent"]["params"])

    def test_invalid_agent_tool_targets_fail_before_model_construction(self) -> None:
        """Reject missing Agent config and malformed params or tool lists."""

        tool = cast(Tool, object())
        invalid_configs = (
            {},
            {"llm_agent": 1},
            {"llm_agent": {"type": "DefaultAgent", "params": []}},
            {
                "llm_agent": {
                    "type": "DefaultAgent",
                    "params": {"tools": ()},
                }
            },
        )

        for source_config in invalid_configs:
            with self.subTest(source_config=source_config):
                with self.assertRaises(ValueError):
                    Xtalk.configure(source_config).add_agent_tools([tool]).build()

    def test_from_config_delegates_to_builder(self) -> None:
        """Keep from_config as the configure-and-build convenience path."""

        source_config = {"service_config": {"enable_persistence": False}}
        with patch.object(Xtalk, "configure") as configure_mock:
            configure_mock.return_value.build.return_value = sentinel.xtalk

            result = Xtalk.from_config(source_config)

        self.assertIs(result, sentinel.xtalk)
        configure_mock.assert_called_once_with(source_config)
        configure_mock.return_value.build.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
