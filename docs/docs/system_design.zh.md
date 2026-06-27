# 整体架构

## 前后端分离

前端API在[`frontend`](https://github.com/xcc-zach/xtalk/tree/main/frontend)下，后端在[`src/xtalk`](https://github.com/xcc-zach/xtalk/tree/main/src/xtalk)。

前端负责音频与控制信号的收发、以及轻量的语音处理（VAD、语音增强）；后端承载X-Talk的主体逻辑。

## 后端架构

后端组件分为模型与`Manager`两级；模型为外部模型的适配器，例如如果想在X-Talk中使用IndexTTS，就要实现TTS接口定义的适配器，然后外部启动IndexTTS后，通过配置的方式接入；`Manager`为模型调度器，例如ASRManager控制ASR模型开始、暂停、继续、结束识别；`Manager`之间依赖观察者模式通信，力求解耦各个模型的能力，将模型间交互落到事件通信中。

模型接口定义于[`src/xtalk/models/agents/interfaces.py`](https://github.com/xcc-zach/xtalk/blob/main/src/xtalk/models/agents/interfaces.py)和[`src/xtalk/models/*/interfaces.py`](https://github.com/xcc-zach/xtalk/tree/main/src/xtalk/models)；前者为LLM的适配接口：LLM是X-Talk的智能核心，负责综合各种语音模型的信息进行输出；后者包含各类语音模块的适配接口，例如ASR、TTS、VAD，以及一些实验性的适配接口，例如TurnDetector、SpeakerEncoder。
