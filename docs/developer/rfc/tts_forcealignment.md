# RFC：TTS逐字播报

* **作者**：刘展勋
* **状态**：Draft
* **日期**：2026.7.14

## 1. 背景

目前XTalk支持非流式输入的TTS输出内容逐字跟踪，以实现打断系统数数后正确回答数到哪里这样的问题。但是在流式输入的情况下，由于无法定位TTS片段和输入文本的对应关系，所以目前基于时间比例的方法失效。

## 2. 目标

引入forcealignment模型对齐TTS播放进度。


## 3. 方案

引入新模型类型ForceAligner，改进[TTSPlaybackMananger](https://github.com/xcc-zach/xtalk/blob/dev/src/xtalk/serving/modules/tts_playback_manager.py)，首先对全量TTS音频调用ForceAligner打出时间戳，并根据TTSChunkPlayback时间按时间戳推进已播放文本。

## 4. 验收标准

* [ ] 输入流式的TTS也能做到进度跟踪，正确回答“打断一下，你现在数到哪里”
