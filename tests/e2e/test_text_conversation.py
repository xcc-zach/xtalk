"""Typed questions get spoken answers and the conversation remembers them."""

from __future__ import annotations

import pytest

from tests.support.mock_models.mock_agent import GREETING_REPLY, summary_reply

pytestmark = pytest.mark.anyio


async def test_greeting_gets_a_spoken_reply(connect):
    client = await connect()
    await client.send_text("你好")
    reply = await client.wait_reply()
    assert reply.status == "completed"
    assert reply.transcript == GREETING_REPLY
    assert reply.audio_ms > 0


async def test_user_item_is_echoed_without_speech_transcription(connect):
    client = await connect()
    await client.send_text("你好", item_id="client-item-1")
    done = await client.wait_for(
        "conversation.item.done", where=lambda event: event["item"]["id"] == "client-item-1"
    )
    assert done["item"]["role"] == "user"
    await client.wait_reply()
    assert client.find("conversation.item.input_audio_transcription.delta") == []
    assert client.find("conversation.item.input_audio_transcription.completed") == []


async def test_second_turn_remembers_the_first(connect):
    client = await connect()
    await client.send_text("你好")
    first = await client.wait_reply()
    await client.wait_heard(first)
    await client.send_text("你刚才说了什么")
    second = await client.wait_reply()
    assert second.transcript == summary_reply(GREETING_REPLY, 2)


async def test_invalid_text_is_rejected_and_the_conversation_continues(connect):
    client = await connect()
    await client.send_text("   ", item_id="blank")
    blank = await client.wait_for("error", where=lambda event: event["error"]["event_id"] == "blank")
    assert blank["error"]["type"] == "invalid_request_error"
    await client.send_text("长" * 3000, item_id="too-long")
    await client.wait_for("error", where=lambda event: event["error"]["event_id"] == "too-long")
    await client.send_text("你好")
    reply = await client.wait_reply()
    assert reply.transcript == GREETING_REPLY
    assert client.find("response.created", before=reply.created_index) == []
