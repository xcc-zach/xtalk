"""Self-checks for the test harness: speech codec and mock model rules.

These tests never touch the service pipeline. A failure here means the
harness is broken, not the product.
"""

from __future__ import annotations

import json

import pytest
import soxr
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from tests.support import speech
from tests.support.mock_models import (
    MockASR,
    MockAgent,
    MockLookupTool,
    MockTTS,
    MockTimerTool,
    MockVAD,
    ScriptedChatModel,
    echo_reply,
    summary_reply,
)
from tests.support.mock_models.mock_agent import (
    GREETING_REPLY,
    LOOKUP_PREAMBLE,
    LOOKUP_RESULT_REPLY,
    LOOKUP_TOOL_CALL,
    STORY_SENTENCES,
)
from tests.support.mock_models.scripted_chat_model import (
    TIMER_FINISHED_PREFIX,
    TIMER_PROGRESS_PREFIX,
    TIMER_STARTED_REPLY,
    TIMER_STATUS_PREFIX,
    TIMER_STOPPED_REPLY,
    TIMER_SUBSCRIBED_REPLY,
    plan_for,
)
from xtalk.models.agents import AgentTurnBoundary
from xtalk.models.agents.tools import Finished, Running, ToolEngine

pytestmark = pytest.mark.anyio


# ---------------------------------------------------------------------------
# speech codec
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("sentence_id", range(speech.MIN_ID, speech.MAX_ID + 1))
def test_every_sentence_survives_encoding_and_resampling(sentence_id: int):
    pcm = speech.encode(sentence_id, voiced_ms=400, trailing_silence_ms=200, sample_rate=24000)
    assert speech.recognize(pcm, 24000) == sentence_id
    samples = speech._samples(pcm)
    resampled = (soxr.resample(samples, 24000, 16000) * 32768.0).astype("<i2").tobytes()
    assert speech.recognize(resampled, 16000) == sentence_id


def test_silence_and_voiced_frames_are_told_apart():
    assert not speech.is_voiced(speech.silence(32, sample_rate=16000))
    assert speech.is_voiced(speech.encode(0, voiced_ms=32, sample_rate=16000))
    assert speech.recognize(speech.silence(500, sample_rate=16000), 16000) is None


# ---------------------------------------------------------------------------
# MockASR
# ---------------------------------------------------------------------------
async def test_asr_partials_grow_into_the_full_text():
    asr = MockASR()
    pcm = speech.encode(1, voiced_ms=1000, sample_rate=16000)
    step = 16000 * 2 // 10  # 100 ms
    previous = ""
    for start in range(0, len(pcm), step):
        partial = await asr.async_recognize_stream(pcm[start : start + step])
        assert speech.text_for(1).startswith(partial)
        assert len(partial) >= len(previous)
        previous = partial
    assert await asr.async_recognize_stream(b"", is_final=True) == speech.text_for(1)


async def test_asr_returns_nothing_for_silence_and_resets_between_turns():
    asr = MockASR()
    assert await asr.async_recognize_stream(speech.silence(300, sample_rate=16000)) == ""
    await asr.async_recognize_stream(speech.encode(0, voiced_ms=600, sample_rate=16000))
    asr.reset()
    assert await asr.async_recognize_stream(b"", is_final=True) == ""


async def test_asr_fails_on_the_failure_sentence():
    asr = MockASR()
    await asr.async_recognize_stream(speech.encode(7, voiced_ms=600, sample_rate=16000))
    with pytest.raises(RuntimeError):
        await asr.async_recognize_stream(b"", is_final=True)


# ---------------------------------------------------------------------------
# MockAgent
# ---------------------------------------------------------------------------
async def _outputs(agent: MockAgent, text: str) -> list:
    items = []
    async for item in agent.async_accept({"type": "asr_final", "data": {"text": text}}):
        items.append(item)
    return items


async def test_agent_rule_table():
    agent = MockAgent(sentence_interval_ms=0)
    assert await _outputs(agent, "你好") == [GREETING_REPLY]
    assert await _outputs(agent, "请讲一个很长的故事") == list(STORY_SENTENCES)
    lookup = await _outputs(agent, "帮我查一下天气")
    assert lookup[0] == LOOKUP_PREAMBLE
    assert lookup[1] == LOOKUP_TOOL_CALL
    assert isinstance(lookup[2], AgentTurnBoundary)
    assert lookup[3] == LOOKUP_RESULT_REPLY
    assert len(await _outputs(agent, "慢慢说")) == 5
    assert await _outputs(agent, "让合成失败") == ["合成失败。"]
    assert await _outputs(agent, "第12句话") == [echo_reply("第12句话")]
    with pytest.raises(RuntimeError):
        await _outputs(agent, "现在报错")


async def test_agent_summary_reflects_playback_confirmed_memory():
    agent = MockAgent()
    assert await _outputs(agent, "你刚才说了什么") == [summary_reply(None, 0)]
    await _outputs(agent, "你好")
    async for _ in agent.async_accept(
        {"type": "response_update", "data": {"response_id": "r1", "text": "你好，我"}}
    ):
        pass
    async for _ in agent.async_accept(
        {"type": "response_finish", "data": {"response_id": "r1", "text": "你好，我是"}}
    ):
        pass
    assert await _outputs(agent, "你刚才说了什么") == [summary_reply("你好，我是", 3)]
    assert agent.clone().history == []


async def test_agent_ignores_other_contexts():
    agent = MockAgent()
    items = [item async for item in agent.async_accept({"type": "asr_partial", "data": {"text": "你"}})]
    assert items == []


# ---------------------------------------------------------------------------
# MockTTS and MockVAD
# ---------------------------------------------------------------------------
def test_tts_duration_follows_text_and_tone_is_stable():
    tts = MockTTS()
    short = tts.synthesize("你好")
    long = tts.synthesize("你好，我是测试助手。")
    assert len(short) == 24000 * 2 * 200 // 1000
    assert len(long) > len(short)
    assert tts.sample_rate == 24000
    assert MockTTS.frequency_for("你好") == MockTTS.frequency_for("你好")
    assert MockTTS.frequency_for("你好") != MockTTS.frequency_for("故事第一句。")
    with pytest.raises(RuntimeError):
        tts.synthesize("合成失败。")


def test_vad_threshold_boundary():
    vad = MockVAD()
    loud = speech.encode(0, voiced_ms=32, sample_rate=16000)
    quiet = speech.silence(32, sample_rate=16000)
    assert vad.detect_frames([loud, quiet]) == [True, False]
    vad.set_threshold(threshold=5.0)
    assert not vad.is_speech(loud)


# ---------------------------------------------------------------------------
# ScriptedChatModel and MockTimerTool
# ---------------------------------------------------------------------------
def _history(*messages):
    return [SystemMessage(content="system"), *messages]


def test_scripted_model_user_rules():
    assert plan_for(_history(HumanMessage(content="你好"))).text == GREETING_REPLY
    assert plan_for(_history(HumanMessage(content="第9句话"))).text == echo_reply("第9句话")
    lookup = plan_for(_history(HumanMessage(content="帮我查一下天气")))
    assert lookup.text == LOOKUP_PREAMBLE and lookup.tool_calls[0]["name"] == "mock_lookup"
    timer = plan_for(_history(HumanMessage(content="定一个计时器")))
    assert timer.tool_calls[0]["name"] == "mock_timer"
    assert plan_for(_history(HumanMessage(content="随便说"))).text == echo_reply("随便说")


def test_scripted_model_tool_result_rules():
    call = {"name": "mock_timer", "args": {"duration_seconds": 3.0}, "id": "call_1"}
    timer_started = _history(
        HumanMessage(content="定一个长计时器"),
        AIMessage(content="", tool_calls=[call]),
        ToolMessage(content="计时开始", tool_call_id="call_1", name="mock_timer"),
    )
    assert plan_for(timer_started).text == TIMER_STARTED_REPLY
    stop = plan_for([*timer_started, HumanMessage(content="停止计时器")])
    assert stop.tool_calls[0]["name"] == "stop_async_tool"
    assert stop.tool_calls[0]["args"] == {"source_call_id": "call_1"}
    status = plan_for([*timer_started, HumanMessage(content="计时器进度如何")])
    assert status.tool_calls[0]["name"] == "id_to_async_tool_status"
    assert plan_for(_history(ToolMessage(content="x", tool_call_id="c", name="stop_async_tool"))).text == TIMER_STOPPED_REPLY
    assert plan_for(_history(ToolMessage(content="{}", tool_call_id="c", name="id_to_async_tool_status"))).text.startswith(TIMER_STATUS_PREFIX)
    assert plan_for(_history(ToolMessage(content="x", tool_call_id="c", name="subscribe_async_tool"))).text == TIMER_SUBSCRIBED_REPLY
    assert plan_for(_history(ToolMessage(content="今天晴", tool_call_id="c", name="mock_lookup"))).text == LOOKUP_RESULT_REPLY
    reminder_call = {"name": "mock_timer", "args": {"duration_seconds": 1.2, "reminder_interval_seconds": 0.3}, "id": "call_2"}
    subscribe = plan_for(_history(AIMessage(content="", tool_calls=[reminder_call]), ToolMessage(content="开始", tool_call_id="call_2", name="mock_timer")))
    assert subscribe.tool_calls[0]["name"] == "subscribe_async_tool"
    for running, prefix in ((True, TIMER_PROGRESS_PREFIX), (False, TIMER_FINISHED_PREFIX)):
        report = plan_for(
            _history(
                ToolMessage(content=json.dumps({"running": running, "tool_output": "x"}), tool_call_id="u", name="async_tool_updated"),
                SystemMessage(content="report"),
            )
        )
        assert report.text.startswith(prefix)


async def test_scripted_model_streams_text_and_tool_calls():
    model = ScriptedChatModel()
    chunks = [chunk async for chunk in model.astream(_history(HumanMessage(content="帮我查一下天气")))]
    gathered = chunks[0]
    for chunk in chunks[1:]:
        gathered = gathered + chunk
    assert gathered.content == LOOKUP_PREAMBLE
    assert gathered.tool_calls[0]["name"] == "mock_lookup"


async def test_timer_tool_lifecycle_through_the_engine():
    updates: list[tuple[str, dict]] = []
    engine = ToolEngine(tools=[MockLookupTool, MockTimerTool], state={})
    engine.on_async_tool_update(lambda call, message: updates.append((call["name"], json.loads(str(message.content)))))

    lookup = await engine.ainvoke({"name": "mock_lookup", "args": {"query": "天气"}, "id": "l1"})
    assert lookup.content == "今天晴"

    plain = await engine.ainvoke({"name": "mock_timer", "args": {"duration_seconds": 0.2}, "id": "t1"})
    assert "t1" in str(plain.content)
    await _wait_for_updates(updates, 1)
    assert [payload["running"] for _, payload in updates] == [False]

    updates.clear()
    await engine.ainvoke({"name": "mock_timer", "args": {"duration_seconds": 0.5, "reminder_interval_seconds": 0.15}, "id": "t2"})
    await engine.ainvoke({"name": "subscribe_async_tool", "args": {"source_call_id": "t2"}, "id": "s2"})
    await _wait_for_finished(updates)
    assert updates[-1][1]["running"] is False
    assert len(updates) >= 2
    assert all(payload["running"] for _, payload in updates[:-1])

    updates.clear()
    await engine.ainvoke({"name": "mock_timer", "args": {"duration_seconds": 0.3}, "id": "t3"})
    await engine.ainvoke({"name": "stop_async_tool", "args": {"source_call_id": "t3"}, "id": "x3"})
    import asyncio

    await asyncio.sleep(0.5)
    assert updates == []
    await engine.shutdown()


async def _wait_for_finished(updates: list, timeout: float = 3.0) -> None:
    import asyncio

    deadline = asyncio.get_running_loop().time() + timeout
    while not any(payload["running"] is False for _, payload in updates):
        if asyncio.get_running_loop().time() > deadline:
            raise AssertionError(f"expected a finished update, got {updates}")
        await asyncio.sleep(0.01)


async def _wait_for_updates(updates: list, count: int, timeout: float = 3.0) -> None:
    import asyncio

    deadline = asyncio.get_running_loop().time() + timeout
    while len(updates) < count:
        if asyncio.get_running_loop().time() > deadline:
            raise AssertionError(f"expected {count} updates, got {updates}")
        await asyncio.sleep(0.01)


def test_timer_tool_results_are_typed():
    state = MockTimerTool.state_type(call_id="c")
    initial = MockTimerTool.emit_initial("c", MockTimerTool.input_type(duration_seconds=1.0), state, {})
    assert isinstance(initial, Running) and "c" in initial.content
    assert MockTimerTool.output_type is not None
    assert isinstance(Finished(MockTimerTool.output_type(content="done")), Finished)
