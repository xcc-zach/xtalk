"""Regression tests for AgenticASR Refiner model discovery."""

from __future__ import annotations

import unittest
from unittest.mock import patch

from xtalk.models.asr.agentic_asr import (
    _OpenAICompatibleRefiner,
    _extract_model_id,
    _resolve_models_url,
)


class _FakeResponse:
    """Provide the asynchronous context-manager subset of an HTTP response."""

    def __init__(self, body: dict, *, status: int = 200) -> None:
        self.body = body
        self.status = status

    async def __aenter__(self) -> "_FakeResponse":
        return self

    async def __aexit__(self, exc_type, exc, traceback) -> None:
        del exc_type, exc, traceback

    async def json(self) -> dict:
        """Return the configured JSON body."""

        return self.body

    async def text(self) -> str:
        """Return a readable representation of the configured body."""

        return str(self.body)


class _FakeSession:
    """Record model discovery and chat-completions requests."""

    def __init__(self) -> None:
        self.get_urls: list[str] = []
        self.posts: list[tuple[str, dict]] = []

    async def __aenter__(self) -> "_FakeSession":
        return self

    async def __aexit__(self, exc_type, exc, traceback) -> None:
        del exc_type, exc, traceback

    def get(self, url: str) -> _FakeResponse:
        """Return one advertised local model."""

        self.get_urls.append(url)
        return _FakeResponse({"object": "list", "data": [{"id": "/models/refiner"}]})

    def post(self, url: str, *, json: dict) -> _FakeResponse:
        """Record a completion request and return refined text."""

        self.posts.append((url, json))
        return _FakeResponse(
            {
                "choices": [
                    {"message": {"role": "assistant", "content": "纠错结果"}}
                ]
            }
        )


class RefinerModelDiscoveryTests(unittest.IsolatedAsyncioTestCase):
    """Verify model discovery and reuse for an OpenAI-compatible Refiner."""

    def test_resolve_models_url_from_api_base(self) -> None:
        """Append the models route to a conventional API base URL."""

        self.assertEqual(
            _resolve_models_url("http://127.0.0.1:18080/v1"),
            "http://127.0.0.1:18080/v1/models",
        )

    def test_resolve_models_url_from_chat_endpoint(self) -> None:
        """Replace a full chat endpoint with its sibling models route."""

        self.assertEqual(
            _resolve_models_url(
                "http://127.0.0.1:18080/v1/chat/completions"
            ),
            "http://127.0.0.1:18080/v1/models",
        )

    def test_extract_model_id_rejects_empty_model_list(self) -> None:
        """Raise a useful error when the server advertises no model."""

        with self.assertRaisesRegex(RuntimeError, "did not advertise any models"):
            _extract_model_id({"object": "list", "data": []})

    async def test_discover_model_once_and_use_it_for_completions(self) -> None:
        """Cache the first advertised model ID across refinement requests."""

        session = _FakeSession()
        refiner = _OpenAICompatibleRefiner("http://127.0.0.1:18080/v1")

        with patch("aiohttp.ClientSession", return_value=session):
            self.assertEqual(await refiner.async_refine("原始文本"), "纠错结果")
            self.assertEqual(
                await refiner.async_refine("另一段文本"),
                "纠错结果",
            )

        self.assertEqual(session.get_urls, ["http://127.0.0.1:18080/v1/models"])
        self.assertEqual(len(session.posts), 2)
        self.assertEqual(
            [payload["model"] for _, payload in session.posts],
            ["/models/refiner", "/models/refiner"],
        )


if __name__ == "__main__":
    unittest.main()
