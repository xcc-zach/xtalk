"""Spoken turns: the assistant answers what was said, after it was said."""

from __future__ import annotations

import pytest

from tests.support.mock_models.mock_agent import GREETING_REPLY, summary_reply

pytestmark = pytest.mark.anyio

MANUAL_TURNS = {"audio": {"input": {"turn_detection": None}}}
TRANSCRIBE_ONLY = {"audio": {"input": {"turn_detection": {"create_response": False}}}}


async def test_spoken_greeting_is_understood_and_answered(connect):
    client = await connect()
    await client.say(0)
    assert await client.wait_transcript() == "你好"
    reply = await client.wait_reply()
    assert reply.status == "completed"
    assert reply.transcript == GREETING_REPLY
    assert reply.audio_ms > 0


async def test_reply_starts_only_after_the_user_finished_speaking(connect):
    client = await connect()
    mark = client.mark
    await client.say(0)
    reply = await client.wait_reply()
    started = client.find("input_audio_buffer.speech_started", after=mark)[0]
    stopped = client.find("input_audio_buffer.speech_stopped", after=mark)[0]
    assert started["_i"] < stopped["_i"] < reply.created_index
    assert client.find("response.created", after=started["_i"], before=stopped["_i"]) == []


async def test_silence_gets_no_reply(connect):
    client = await connect()
    await client.silence(1500)
    await client.say(0)
    reply = await client.wait_reply()
    assert reply.transcript == GREETING_REPLY
    assert client.find("response.created", before=reply.created_index) == []


async def test_two_spoken_turns_share_memory(connect):
    client = await connect()
    await client.say(0)
    first = await client.wait_reply()
    await client.wait_heard(first)
    await client.say(4)
    assert await client.wait_transcript() == "你刚才说了什么"
    second = await client.wait_reply()
    assert second.transcript == summary_reply(GREETING_REPLY, 2)


async def test_binary_and_base64_audio_are_understood_the_same_way(connect):
    client = await connect()
    await client.say(0, binary=True)
    assert await client.wait_transcript() == "你好"
    binary_reply = await client.wait_reply()
    await client.wait_heard(binary_reply)
    await client.say(0)
    assert await client.wait_transcript() == "你好"
    base64_reply = await client.wait_reply()
    assert binary_reply.transcript == base64_reply.transcript == GREETING_REPLY


async def test_manual_turns_reply_only_after_commit(connect):
    client = await connect(MANUAL_TURNS)
    await client.say(0, realtime=False)
    assert client.find("response.created") == []
    await client.commit()
    committed = await client.wait_for("input_audio_buffer.committed")
    reply = await client.wait_reply()
    assert reply.transcript == GREETING_REPLY
    assert client.find("response.created", before=committed["_i"]) == []


async def test_empty_commit_is_rejected_and_the_turn_still_works(connect):
    client = await connect(MANUAL_TURNS)
    await client.send_raw({"type": "input_audio_buffer.commit", "event_id": "empty"})
    error = await client.wait_for("error", where=lambda event: event["error"]["event_id"] == "empty")
    assert error["error"]["type"] == "invalid_request_error"
    await client.say(0, realtime=False)
    await client.commit()
    reply = await client.wait_reply()
    assert reply.transcript == GREETING_REPLY


async def test_transcribe_only_turns_enter_memory_without_a_reply(connect):
    client = await connect(TRANSCRIBE_ONLY)
    await client.say(0)
    assert await client.wait_transcript() == "你好"
    await client.send_text("你刚才说了什么")
    reply = await client.wait_reply()
    assert reply.transcript == summary_reply(None, 1)
    assert client.find("response.created", before=reply.created_index) == []


async def test_voice_and_text_turns_share_memory(connect):
    client = await connect()
    await client.say(0)
    spoken = await client.wait_reply()
    await client.wait_heard(spoken)
    await client.send_text("你刚才说了什么")
    typed = await client.wait_reply()
    assert typed.transcript == summary_reply(GREETING_REPLY, 2)
