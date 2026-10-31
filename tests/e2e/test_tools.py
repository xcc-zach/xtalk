"""Tools run at the right moment and background tools report back on their own."""

from __future__ import annotations

import pytest

from tests.support.mock_models.mock_agent import (
    GREETING_REPLY,
    LOOKUP_PREAMBLE,
    LOOKUP_RESULT_REPLY,
    echo_reply,
)
from tests.support.mock_models.scripted_chat_model import (
    TIMER_FINISHED_PREFIX,
    TIMER_PROGRESS_PREFIX,
    TIMER_STARTED_REPLY,
    TIMER_STATUS_PREFIX,
    TIMER_STOPPED_REPLY,
    TIMER_SUBSCRIBED_REPLY,
)

pytestmark = [pytest.mark.anyio, pytest.mark.mock_config(tools=True)]


def _is_lookup(event):
    return event["name"] == "mock_lookup"


async def test_lookup_runs_after_the_preamble_and_its_result_is_spoken(connect):
    client = await connect()
    await client.send_text("帮我查一下天气")
    preamble = await client.wait_reply()
    assert preamble.transcript == LOOKUP_PREAMBLE
    await client.wait_heard(preamble)
    tool = await client.wait_for("response.tool.called", where=_is_lookup)
    assert tool["_i"] > preamble.heard_index
    result = await client.wait_reply()
    assert result.transcript == LOOKUP_RESULT_REPLY


async def test_interrupting_the_preamble_skips_the_lookup(connect):
    client = await connect()
    client.hear_up_to(200.0)
    await client.send_text("帮我查一下天气")
    created = await client.wait_for("response.created")
    await client.wait_until_heard(created["response"]["id"], 200.0)
    await client.say(0)
    await client.wait_reply()
    greeting = await client.wait_reply()
    assert greeting.transcript == GREETING_REPLY
    assert client.find("response.tool.called", where=_is_lookup) == []
    assert all(reply.transcript != LOOKUP_RESULT_REPLY for reply in client.replies)


async def test_timer_is_acknowledged_immediately_and_reported_when_done(connect):
    client = await connect()
    await client.send_text("定一个计时器")
    started = await client.wait_reply()
    assert started.transcript == TIMER_STARTED_REPLY
    assert started.status == "completed"
    report = await client.wait_reply(timeout=10.0)
    assert report.transcript.startswith(TIMER_FINISHED_PREFIX)
    assert report.status == "completed"
    assert report.audio_ms > 0
    assert started.done_index < report.created_index


async def test_subscribed_timer_reports_progress_before_completion(connect):
    client = await connect()
    await client.send_text("定一个带提醒的计时器")
    started = await client.wait_reply()
    assert started.transcript == TIMER_SUBSCRIBED_REPLY
    reports = []
    while not reports or not reports[-1].transcript.startswith(TIMER_FINISHED_PREFIX):
        reports.append(await client.wait_reply(timeout=10.0))
    progress = [reply for reply in reports if reply.transcript.startswith(TIMER_PROGRESS_PREFIX)]
    assert progress
    assert all(reply.created_index < reports[-1].created_index for reply in progress)


async def _timer_finishes_while_the_user_speaks(client):
    await client.send_text("定一个计时器")
    started = await client.wait_reply()
    assert started.transcript == TIMER_STARTED_REPLY
    mark = client.mark
    await client.say(9, voiced_ms=1200.0)
    speech_started = client.find("input_audio_buffer.speech_started", after=mark)[0]
    speech_stopped = client.find("input_audio_buffer.speech_stopped", after=mark)[0]
    return speech_started, speech_stopped


async def test_timer_report_is_not_spoken_while_the_user_speaks(connect):
    client = await connect()
    speech_started, speech_stopped = await _timer_finishes_while_the_user_speaks(client)
    assert client.find("response.created", after=speech_started["_i"], before=speech_stopped["_i"]) == []
    answer = await client.wait_reply(timeout=10.0)
    assert answer.created_index > speech_stopped["_i"]
    assert answer.transcript.startswith(echo_reply("第9句话")[:3])
    report = await client.wait_reply(timeout=10.0)
    assert report.transcript.startswith(TIMER_FINISHED_PREFIX)


@pytest.mark.xfail(
    strict=True,
    reason="The proactive tool report preempts the answer that is still playing "
    "(TTSResponseCoordinator stops the delivering response), so the answer ends "
    "as cancelled instead of completed.",
)
async def test_timer_report_waits_until_the_answer_finished(connect):
    client = await connect()
    await _timer_finishes_while_the_user_speaks(client)
    answer = await client.wait_reply()
    assert answer.transcript == echo_reply("第9句话")
    assert answer.status == "completed"
    report = await client.wait_reply(timeout=10.0)
    assert report.transcript.startswith(TIMER_FINISHED_PREFIX)
    assert report.created_index > answer.done_index


async def test_stopped_timer_never_reports(connect):
    client = await connect()
    await client.send_text("定一个长计时器")
    started = await client.wait_reply()
    assert started.transcript == TIMER_STARTED_REPLY
    await client.send_text("停止计时器")
    stopped = await client.wait_reply()
    assert stopped.transcript == TIMER_STOPPED_REPLY
    await client.say(0, voiced_ms=3500.0)
    greeting = await client.wait_reply()
    assert greeting.transcript == GREETING_REPLY
    assert all(not reply.transcript.startswith(TIMER_FINISHED_PREFIX) for reply in client.replies)
    assert len(client.find("response.created")) == 3


async def test_timer_status_can_be_asked_while_running(connect):
    client = await connect()
    await client.send_text("定一个长计时器")
    started = await client.wait_reply()
    assert started.transcript == TIMER_STARTED_REPLY
    await client.send_text("计时器进度如何")
    status = await client.wait_reply()
    assert status.transcript.startswith(TIMER_STATUS_PREFIX)
    assert all(not reply.transcript.startswith(TIMER_FINISHED_PREFIX) for reply in client.replies)
