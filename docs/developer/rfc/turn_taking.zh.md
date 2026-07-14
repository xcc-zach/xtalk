# RFC：轮次控制

* **作者**：刘展勋
* **状态**：Draft
* **日期**：2026.7.14

## 1. 背景

目前对话系统是你一轮我一轮，同时会有误打断和误触发的情况。

## 2. 目标

本方案希望实现：

* 对话系统在合适的时候开始说话，合适的时候结束说话。覆盖同传这种两个人一起说话的情景。

## 3. 方案

### 基础方案：回合检测+LLM微调

需要微调两个LLM：

- 回合检测LLM
    - 输入对话历史和流式ASR内容，输出start/stop/do_nothing
        - 示例1 普通对话（以下内容apply chat template后完整输入）
            - 轮次1
            ```
            system: 当前状态listening
            user: 你好。
            assistant: 你好
            user: 今天天气
            assistant: <think>语义未完整</think>do_nothing
            ```
            - 轮次2
            ```
            system: 当前状态listening
            user: 你好。
            assistant: 你好
            user: 今天天气真不错。
            assistant: <think>语义完整</think>start
            ```
            - 轮次3
            ```
            system: 当前状态speaking
            user: 你好。
            assistant: 你好
            user: 今天天气真不错。
            assistant: 是呀，今天天气真不错，小鸟在唱
            user: 对了，关于昨天我和你说的方案
            assistant: <think>用户抛出新话题，应停下倾听</think>stop
            ```
        - 示例2 同传（以下内容apply chat template后完整输入）
            - 轮次1
            ```
            system: 当前状态listening
            user: 接下来将我说的内容同声传译为英语
            assistant: 好的
            user: 今天天气
            assistant: <think>同声传译情景，目前部分可翻译</think>start
            ```
            - 轮次2
            ```
            system: 当前状态speaking
            user: 接下来将我说的内容同声传译为英语
            assistant: 好的
            user: 今天天气
            assistant: The weather <-- 按照流式ASR进度注入用户消息，按照TTS进度注入AI消息，TTS目前还没说到today
            user: 真不错
            assistant:<think>保持同传</think>do_nothing
            ```
    - 第一版先不加think，构造数据可直接从双人对话的说话人转换获取训练信号，无需上游LLM合成推理
- 对话LLM
    微调目标是user assistant说话片段交替；回合检测模型给出start时以TTS语速保持慢生成；stop停止生成；仅将TTS播放的内容插回LLM历史；仅将用户ASR内容插入LLM历史
    - 示例
        ```
        <|user|>今天天气<\|user|><|ai|>The weather<\|ai|><|user|>真不错<\|user|><|ai|>today is<\|ai|>
        ```
    - 对于工具调用与工具信息，生成后直接插入对话历史

### 替代方案：两个模型做成一个

微调的LLM认为不需要生成时直接输出一个end，代码检测到end时停止生成，但是end从不进对话历史；end表示listening时继续listen，或者结束speaking转到listen；listen到speak由LLM生成end以外内容触发

### 进阶1：输入编码声学信息

取最新x秒（例如x=5）音频编码tokens，通过扩充交叉注意力的KV来插入模型

### 进阶2: 输出TTS辅助tokens

参考minicpm4.5o等端到端模型，每输出一个文本token输出对应的声学token，用于给到流式TTS合成带声学上下文的语音

### 数据

- 大量双人对话数据

### 接入X-Talk

- LLM Agent与回合检测接口基本兼容；回合检测模型训练好后可直接接入X-Talk，使用未经微调的LLM主观判断表现

- 适配回合检测的微调LLM编写LLM Agent接口的实现接入


## 4. 风险

- 对话数据如何清洗打标需进一步明确
- 使用声学信息的数据需明确用什么
- 创新范式，有可能不work

## 5. 验收标准

* [ ] 回合检测模型
* [ ] 适配回合检测模型的微调LLM
* [ ] 接入X-Talk可运行demo
* [ ] 可选进阶：回合检测与对话一体模型
* [ ] 进阶：带声学辅助信息的微调LLM
* [ ] 进阶：带声学辅助信息的TTS
