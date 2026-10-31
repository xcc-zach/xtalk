"""Failures end one turn; the next turn works normally."""

from __future__ import annotations

import pytest

from tests.support.mock_models.mock_agent import GREETING_REPLY

pytestmark = pytest.mark.anyio


async def _greeting_still_works(client) -> None:
    await client.send_text("你好")
    reply = await client.wait_reply()
    assert reply.status == "completed"
    assert reply.transcript == GREETING_REPLY


async def test_recognition_failure_is_reported_and_not_answered(connect):
    client = await connect()
    await client.say(7)
    failed = await client.wait_for("conversation.item.input_audio_transcription.failed")
    assert failed["item_id"]
    await _greeting_still_works(client)
    greeting = client.replies[-1]
    assert client.find("response.created", before=greeting.created_index) == []


async def test_generation_failure_ends_the_reply_as_failed(connect):
    client = await connect()
    await client.send_text("现在报错")
    reply = await client.wait_reply()
    assert reply.status == "failed"
    await _greeting_still_works(client)


async def test_synthesis_failure_ends_the_reply_without_completion(connect):
    client = await connect()
    await client.send_text("让合成失败")
    reply = await client.wait_reply()
    assert reply.status != "completed"
    await _greeting_still_works(client)


async def test_invalid_session_update_does_not_break_the_conversation(connect):
    client = await connect()
    await client.send_raw(
        {
            "type": "session.update",
            "event_id": "bad-threshold",
            "session": {"audio": {"input": {"turn_detection": {"threshold": 5}}}},
        }
    )
    error = await client.wait_for("error", where=lambda e: e["error"]["event_id"] == "bad-threshold")
    assert error["error"]["type"] == "invalid_request_error"
    await _greeting_still_works(client)


async def test_unsupported_requests_are_rejected_without_a_reply(connect):
    client = await connect()
    await client.send_raw({"type": "nonsense.event", "event_id": "nonsense"})
    await client.wait_for("error", where=lambda e: e["error"]["event_id"] == "nonsense")
    await client.send_raw({"type": "response.create", "event_id": "create"})
    await client.wait_for("error", where=lambda e: e["error"]["event_id"] == "create")
    await _greeting_still_works(client)
    greeting = client.replies[-1]
    assert client.find("response.created", before=greeting.created_index) == []
