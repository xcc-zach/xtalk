"""Tests for final-only MTD speaker exemplar selection."""

import numpy as np

from xtalk.serving.mtd.audio_layout import float32_to_pcm16_bytes
from xtalk.serving.mtd.exemplar_pool import SpeakerExemplarPool


def _pool() -> SpeakerExemplarPool:
    return SpeakerExemplarPool(
        {
            "max_speakers": 16,
            "min_register_duration_s": 0.70,
            "min_update_duration_s": 0.45,
            "preferred_min_duration_s": 1.0,
            "preferred_max_duration_s": 7.0,
            "min_rms_dbfs": -42.0,
            "max_clipping_ratio": 0.01,
            "min_boundary_margin_s": 0.08,
            "replace_score_margin": 0.08,
            "score_weights": {
                "duration": 0.35,
                "rms": 0.25,
                "non_overlap": 0.25,
                "boundary": 0.10,
                "unclipped": 0.05,
            },
        }
    )


def test_pool_keeps_complete_partially_overlapped_mtd_segment() -> None:
    """Overlap must not cut audio or guess a matching text subrange."""

    pool = _pool()
    t = np.arange(4 * 16000, dtype=np.float32) / 16000
    audio = float32_to_pcm16_bytes(0.2 * np.sin(2 * np.pi * 220 * t))
    decisions = pool.update_from_final(
        audio,
        [
            {"start_s": 0.2, "end_s": 3.2, "speaker_id": "S01", "text": "甲说了一段话"},
            {"start_s": 1.2, "end_s": 2.2, "speaker_id": "S02", "text": "乙插话"},
        ],
        source_segment_id=1,
    )
    first = decisions[0]
    assert first["action"] == "register"
    assert first["overlap_class"] == "partial_overlap"
    assert first["quality"]["used_non_overlap"] is False
    assert (first["candidate_start_s"], first["candidate_end_s"]) == (0.2, 3.2)
    assert first["text"] == "甲说了一段话"
    assert pool.items["S01"].duration_s == 3.0
    assert pool.items["S01"].text == "甲说了一段话"


def test_pool_prefers_complete_non_overlapped_segment_before_better_overlap() -> None:
    """Overlap class has priority; score only ranks candidates in one class."""

    pool = _pool()
    t = np.arange(4 * 16000, dtype=np.float32) / 16000
    audio = float32_to_pcm16_bytes(0.2 * np.sin(2 * np.pi * 220 * t))
    decisions = pool.update_from_final(
        audio,
        [
            {"start_s": 0.0, "end_s": 1.0, "speaker_id": "S01", "text": "纯净片段"},
            {"start_s": 1.0, "end_s": 4.0, "speaker_id": "S01", "text": "较长但重叠片段"},
            {"start_s": 1.5, "end_s": 2.5, "speaker_id": "S02", "text": "插话"},
        ],
        source_segment_id=1,
    )
    s01_actions = [item for item in decisions if item["speaker_id"] == "S01"]
    selected = next(item for item in s01_actions if item["action"] == "register")
    assert selected["overlap_class"] == "non_overlap"
    assert (selected["candidate_start_s"], selected["candidate_end_s"]) == (0.0, 1.0)
    assert pool.items["S01"].text == "纯净片段"


def test_pool_prefers_partial_before_full_overlap() -> None:
    """A partially overlapped complete segment outranks a fully overlapped one."""

    pool = _pool()
    t = np.arange(4 * 16000, dtype=np.float32) / 16000
    audio = float32_to_pcm16_bytes(0.2 * np.sin(2 * np.pi * 220 * t))
    decisions = pool.update_from_final(
        audio,
        [
            {"start_s": 0.0, "end_s": 1.0, "speaker_id": "S01", "text": "部分重叠"},
            {"start_s": 2.0, "end_s": 3.0, "speaker_id": "S01", "text": "完全重叠"},
            {"start_s": 0.5, "end_s": 1.5, "speaker_id": "S02", "text": "插话甲"},
            {"start_s": 2.0, "end_s": 3.0, "speaker_id": "S02", "text": "插话乙"},
        ],
        source_segment_id=1,
    )
    s01_actions = [item for item in decisions if item["speaker_id"] == "S01"]
    selected = next(item for item in s01_actions if item["action"] == "register")
    assert selected["overlap_class"] == "partial_overlap"
    assert selected["text"] == "部分重叠"
    assert pool.items["S01"].text == "部分重叠"


def test_pool_falls_back_to_complete_fully_overlapped_segment() -> None:
    """A fully surrounded speaker keeps its whole MTD segment and text."""

    pool = _pool()
    t = np.arange(3 * 16000, dtype=np.float32) / 16000
    audio = float32_to_pcm16_bytes(0.2 * np.sin(2 * np.pi * 180 * t))
    decisions = pool.update_from_final(
        audio,
        [
            {"start_s": 0.0, "end_s": 3.0, "speaker_id": "S01", "text": "外层说话"},
            {"start_s": 1.0, "end_s": 2.0, "speaker_id": "S02", "text": "中间插话"},
        ],
        source_segment_id=1,
    )
    second = decisions[1]
    assert second["action"] == "register"
    assert second["overlap_class"] == "full_overlap"
    assert second["quality"]["used_non_overlap"] is False
    assert (second["candidate_start_s"], second["candidate_end_s"]) == (1.0, 2.0)
    assert second["text"] == "中间插话"
    assert pool.items["S02"].text == "中间插话"


def test_clean_candidate_replaces_higher_scoring_overlapped_exemplar() -> None:
    """Overlap priority applies across finals, before score comparison."""

    pool = _pool()
    long_t = np.arange(3 * 16000, dtype=np.float32) / 16000
    loud = float32_to_pcm16_bytes(0.3 * np.sin(2 * np.pi * 220 * long_t))
    first = pool.update_from_final(
        loud,
        [
            {"start_s": 0.0, "end_s": 2.5, "speaker_id": "S01", "text": "重叠旧样本"},
            {"start_s": 0.5, "end_s": 2.0, "speaker_id": "S02", "text": "另一个人"},
        ],
        source_segment_id=1,
    )
    assert next(item for item in first if item["speaker_id"] == "S01")["action"] == "register"
    assert pool.items["S01"].quality["overlap_class"] == "partial_overlap"
    old_score = pool.items["S01"].score

    short_t = np.arange(0.8 * 16000, dtype=np.float32) / 16000
    quiet = float32_to_pcm16_bytes(0.02 * np.sin(2 * np.pi * 220 * short_t))
    second = pool.update_from_final(
        quiet,
        [{"start_s": 0.0, "end_s": 0.8, "speaker_id": "S01", "text": "纯净新样本"}],
        source_segment_id=2,
    )
    assert second[0]["action"] == "replace"
    assert second[0]["reason"] == "better_overlap_class"
    assert pool.items["S01"].score < old_score
    assert pool.items["S01"].text == "纯净新样本"


def test_overlapped_candidate_never_replaces_clean_exemplar_by_score() -> None:
    """A higher score cannot demote a speaker to a worse overlap class."""

    pool = _pool()
    short_t = np.arange(0.8 * 16000, dtype=np.float32) / 16000
    quiet = float32_to_pcm16_bytes(0.02 * np.sin(2 * np.pi * 220 * short_t))
    first = pool.update_from_final(
        quiet,
        [{"start_s": 0.0, "end_s": 0.8, "speaker_id": "S01", "text": "纯净旧样本"}],
        source_segment_id=1,
    )
    assert first[0]["action"] == "register"
    old_score = pool.items["S01"].score

    long_t = np.arange(3 * 16000, dtype=np.float32) / 16000
    loud = float32_to_pcm16_bytes(0.3 * np.sin(2 * np.pi * 220 * long_t))
    second = pool.update_from_final(
        loud,
        [
            {"start_s": 0.0, "end_s": 2.5, "speaker_id": "S01", "text": "重叠新样本"},
            {"start_s": 0.5, "end_s": 2.0, "speaker_id": "S02", "text": "另一个人"},
        ],
        source_segment_id=2,
    )
    s01 = next(item for item in second if item["speaker_id"] == "S01")
    assert s01["quality"]["score"] > old_score
    assert s01["action"] == "reject"
    assert s01["reason"] == "worse_overlap_class"
    assert pool.items["S01"].text == "纯净旧样本"
