"""Many simultaneous conversations stay correct and isolated."""

from __future__ import annotations

import asyncio
import os

import pytest

from tests.support.mock_models.mock_agent import GREETING_REPLY, STORY_SENTENCES, echo_reply

pytestmark = pytest.mark.anyio

SESSIONS = int(os.environ.get("XTALK_STRESS_SESSIONS", "32"))
TIMEOUT = float(os.environ.get("XTALK_STRESS_TIMEOUT", "60"))
STORY = "".join(STORY_SENTENCES)


async def _open_all(connect, count: int):
    return [await connect() for _ in range(count)]


async def _text_turn(client, index: int):
    await client.send_text(f"第{8 + index}句话")
    return await client.wait_reply()


async def _voice_turn(client, index: int):
    await client.say(8 + index)
    transcript = await client.wait_transcript()
    return transcript, await client.wait_reply()


async def _interrupted_story(client):
    client.hear_up_to(1550.0)
    await client.say(1)
    created = await client.wait_for("response.created")
    await client.wait_until_heard(created["response"]["id"], 1500.0)
    await client.say(0)
    story = await client.wait_reply()
    heard = await client.wait_heard(story)
    greeting = await client.wait_reply()
    return heard, greeting


async def test_text_storm(connect):
    async with asyncio.timeout(TIMEOUT):
        clients = await _open_all(connect, SESSIONS)
        replies = await asyncio.gather(
            *(_text_turn(client, index) for index, client in enumerate(clients))
        )
    for index, reply in enumerate(replies):
        assert reply.status == "completed"
        assert reply.transcript == echo_reply(f"第{8 + index}句话")
    assert len({reply.response_id for reply in replies}) == SESSIONS
    assert all(client.errors == [] for client in clients)


async def test_voice_storm(connect):
    async with asyncio.timeout(TIMEOUT):
        clients = await _open_all(connect, SESSIONS)
        results = await asyncio.gather(
            *(_voice_turn(client, index) for index, client in enumerate(clients))
        )
    for index, (transcript, reply) in enumerate(results):
        assert transcript == f"第{8 + index}句话"
        assert reply.transcript == echo_reply(transcript)
    assert all(client.errors == [] for client in clients)


async def test_mixed_storm(connect):
    async with asyncio.timeout(TIMEOUT):
        clients = await _open_all(connect, SESSIONS)
        half = SESSIONS // 2
        interrupted, plain = await asyncio.gather(
            asyncio.gather(*(_interrupted_story(client) for client in clients[:half])),
            asyncio.gather(*(_text_turn(client, 0) for client in clients[half:])),
        )
    for heard, greeting in interrupted:
        assert heard and STORY.startswith(heard) and heard != STORY
        assert greeting.transcript == GREETING_REPLY
    for reply in plain:
        assert reply.status == "completed"
        assert reply.transcript == echo_reply("第8句话")


async def test_service_is_healthy_after_a_storm(connect):
    async with asyncio.timeout(TIMEOUT):
        clients = await _open_all(connect, SESSIONS)
        await asyncio.gather(*(_text_turn(client, index) for index, client in enumerate(clients)))
        for client in clients:
            await client.close()
        fresh = await connect()
        await fresh.send_text("你好")
        reply = await fresh.wait_reply()
    assert reply.transcript == GREETING_REPLY
