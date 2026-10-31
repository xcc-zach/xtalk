"""Speaking over the assistant stops it; it only remembers what was heard."""

from __future__ import annotations

import pytest

from tests.support.mock_models.mock_agent import (
    GREETING_REPLY,
    STORY_SENTENCES,
    summary_reply,
)

pytestmark = pytest.mark.anyio

STORY = "".join(STORY_SENTENCES)
NO_INTERRUPT = {"audio": {"input": {"turn_detection": {"interrupt_response": False}}}}


async def _start_story(client, heard_ms: float = 1550.0) -> str:
    """Ask for the long story and hear part of it."""
    client.hear_up_to(heard_ms)
    await client.say(1)
    created = await client.wait_for("response.created")
    response_id = created["response"]["id"]
    await client.wait_until_heard(response_id, min(heard_ms, 1500.0))
    return response_id


async def test_barge_in_stops_the_reply_and_keeps_only_the_heard_part(connect):
    client = await connect()
    story_id = await _start_story(client)
    await client.say(0)
    story = await client.wait_reply()
    assert story.response_id == story_id
    assert story.status in {"cancelled", "completed"}
    heard = await client.wait_heard(story)
    assert story.stop_requested
    assert heard and STORY.startswith(heard) and heard != STORY
    stop = client.find(
        "response.output_audio.playback.stop", where=lambda e: e["response_id"] == story_id
    )[0]
    assert client.find(
        "response.output_audio.delta", where=lambda e: e["response_id"] == story_id, after=stop["_i"]
    ) == []
    new_reply = await client.wait_reply()
    assert new_reply.transcript == GREETING_REPLY


async def test_memory_after_barge_in_holds_the_heard_prefix(connect):
    client = await connect()
    await _start_story(client)
    await client.say(4)
    story = await client.wait_reply()
    heard = await client.wait_heard(story)
    assert STORY.startswith(heard) and heard != STORY
    summary = await client.wait_reply()
    assert summary.transcript == summary_reply(heard, 2)


async def test_speech_does_not_stop_the_reply_when_interruption_is_off(connect):
    client = await connect(NO_INTERRUPT)
    story_id = await _start_story(client)
    mark = client.mark
    await client.say(0)
    started = client.find("input_audio_buffer.speech_started", after=mark)[0]
    stopped = client.find("input_audio_buffer.speech_stopped", after=mark)[0]
    assert client.find(
        "response.output_audio.playback.stop",
        where=lambda e: e["response_id"] == story_id,
        after=started["_i"],
        before=stopped["_i"],
    ) == []
    assert client.find("response.done", after=started["_i"], before=stopped["_i"]) == []


async def test_cancel_ends_the_reply_without_late_output(connect):
    client = await connect()
    await client.send_text("慢慢说")
    created = await client.wait_for("response.created")
    await client.cancel()
    reply = await client.wait_reply()
    assert reply.response_id == created["response"]["id"]
    assert reply.status == "cancelled"
    assert client.find(
        "response.output_audio.delta",
        where=lambda e: e["response_id"] == reply.response_id,
        after=reply.done_index,
    ) == []
    assert client.find(
        "response.output_audio_transcript.delta",
        where=lambda e: e["response_id"] == reply.response_id,
        after=reply.done_index,
    ) == []


async def test_cancel_by_id_behaves_the_same(connect):
    client = await connect()
    await client.send_text("慢慢说")
    created = await client.wait_for("response.created")
    await client.cancel(created["response"]["id"])
    reply = await client.wait_reply()
    assert reply.status == "cancelled"


async def test_cancel_without_an_active_reply_is_an_error(connect):
    client = await connect()
    await client.send_raw({"type": "response.cancel", "event_id": "idle-cancel"})
    error = await client.wait_for("error", where=lambda e: e["error"]["event_id"] == "idle-cancel")
    assert error["error"]["type"] == "invalid_request_error"


async def test_server_settles_even_when_the_client_never_reports_the_stop(connect):
    client = await connect(respond_to_stop=False)
    await _start_story(client)
    await client.say(0)
    story = await client.wait_reply()
    heard = await client.wait_heard(story)
    assert STORY.startswith(heard)
    new_reply = await client.wait_reply()
    assert new_reply.transcript == GREETING_REPLY
