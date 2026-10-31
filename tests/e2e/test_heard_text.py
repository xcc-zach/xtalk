"""The displayed assistant text never runs ahead of what was actually played."""

from __future__ import annotations

import pytest

from tests.support.mock_models.mock_agent import STORY_SENTENCES

pytestmark = pytest.mark.anyio

STORY = "".join(STORY_SENTENCES)


async def test_subtitles_follow_playback_and_end_with_the_full_text(connect):
    client = await connect()
    client.hear_up_to(600.0)
    await client.send_text("请讲一个很长的故事")
    created = await client.wait_for("response.created")
    response_id = created["response"]["id"]
    await client.wait_until_heard(response_id, 600.0)
    client.hear_up_to(1800.0)
    await client.wait_until_heard(response_id, 1800.0)
    await client.resume()
    reply = await client.wait_reply()
    heard = await client.wait_heard(reply)
    assert heard == STORY == reply.transcript

    updates = client.find(
        "response.output_audio.playback.updated", where=lambda e: e["response_id"] == response_id
    )
    assert updates
    previous = ""
    for update in updates:
        text = update["text"]
        assert STORY.startswith(text)
        assert len(text) >= len(previous)
        if update["_heard_ms"] + 1e-6 < reply.audio_ms:
            assert len(text) < len(STORY)
        previous = text
