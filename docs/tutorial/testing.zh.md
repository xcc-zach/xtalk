*测试中的功能*

X-Talk 提供了统一的 [`scripts/test.py`](https://github.com/xcc-zach/xtalk/blob/main/scripts/test.py) 脚本，用于测试集生成和后端自动化评测。

它有两种模式：

- `--create`：从文本模板生成可运行的音频测试集
- `--input`：启动内嵌的 X-Talk 服务并自动运行测试集

## 准备测试模板

先创建一个测试集根目录，包含：

- 根目录下一个 TTS 配置 JSON，例如 `tts_config.json`
- 根目录下一个可选的 `test_config.json`
- 一个或多个 case 子目录

例如：

```plaintext
logs/test_templates/smoke/
├── tts_config.json
├── test_config.json
└── basic_turn/
    └── timestamp.txt
```

## 编写 `tts_config.json`

在 `--create` 模式下，脚本会在根目录寻找一个 JSON 文件，例如 `tts_config.json`、`config.json` 或 `sample_local.json`。

这个文件可以使用以下两种形式之一：

1. 独立的 TTS 模型配置
2. 含有顶层 `tts` 字段的完整 X-Talk 服务配置

为了避免歧义，建议使用专门用于生成测试集的独立 `tts_config.json`。

`IndexTTS` 的最小示例：

```json
{
  "type": "IndexTTS",
  "params": {
    "base_url": "http://127.0.0.1:11996",
    "voices": [
      {
        "name": "man",
        "path": "/path/to/reference_voice.wav"
      }
    ]
  }
}
```

等价的完整服务配置写法：

```json
{
  "tts": {
    "type": "IndexTTS",
    "params": {
      "base_url": "http://127.0.0.1:11996",
      "voices": [
        {
          "name": "man",
          "path": "/path/to/reference_voice.wav"
        }
      ]
    }
  }
}
```

其中 `type` 应与 TTS 模型的 Python 类名一致，`params` 应与该类的初始化参数一致。可用的 TTS 后端及其可选依赖请参阅 [Supported Models](../technical_reference/supported_models.zh.md)。

对于 `IndexTTS`，`voices` 中的每一项都需要：

- `name`：暴露给 X-Talk 的音色标识
- `path`：参考音频 WAV 文件，或包含参考音频文件的目录

如果您已经有一个可用的服务端配置文件，通常直接复用其中的 `tts` 段最方便。

## 可选配置

您可以在测试集根目录放一个可选的 `test_config.json`，它支持：

- `concurrency`
- `with_vad`
- `vad_redemption_ms`
- `judge_llm`

示例：

```json
{
  "concurrency": 1,
  "with_vad": false,
  "vad_redemption_ms": 500,
  "judge_llm": {
    "model": "Qwen/Qwen2.5-7B-Instruct",
    "base_url": "http://127.0.0.1:8000/v1",
    "api_key": "YOUR_API_KEY"
  }
}
```

每个 case 子目录中还可以放一个可选的 `criteria.yaml`：

```yaml
judge_asr: true
```

当启用 `judge_asr: true` 时，可运行测试集中的每条 `timestamp.txt` 记录都必须在第三列提供期望转录文本。由 `--create` 生成的数据集会自动满足这个格式。

## 创建`timestamp.txt`

每个 case 子目录都必须包含一个 `timestamp.txt`。在 `--create` 模式下，每一行使用 `<time_spec>:<text>` 格式：

```plaintext
# basic_turn/timestamp.txt
0:Hello, how are you today?
ai_end:Tell me more about your plan.
ai_end+2.5:I also want to ask about pricing.
```

`<time_spec>` 可以是：

- 绝对秒数，例如 `0`、`5.0`、`10.5`
- `ai_start`
- `ai_end`
- `user_start`
- `user_end`
- 上述任一锚点再加偏移量，例如 `ai_end+2.5`

相对时间按文件顺序解析。`ai_*` 锚点表示由上一条用户音频触发的下一轮 AI 响应，`user_*` 锚点表示上一条用户音频本身。

## 生成可运行测试集

先安装脚本依赖：

```bash
pip install numpy requests soundfile websockets pyyaml uvicorn fastapi
```

可选：

```bash
pip install soxr
```

此外，还需要安装所选 TTS 后端对应的 X-Talk 可选依赖。例如，`IndexTTS` 需要安装 `index-tts` extra。

然后生成测试集：

```bash
python scripts/test.py --create logs/test_templates/smoke --out logs/tests
```

脚本会从根目录 JSON 配置中加载 TTS 模型，为每一行合成一个 WAV 文件，并输出可直接运行的 case 目录。生成后的 `timestamp.txt` 会使用 `<time_spec>:<audio_file>:<expected_text>` 格式。

例如：

```plaintext
logs/tests/smoke/
├── tts_config.json
├── test_config.json
└── basic_turn/
    ├── audio_000.wav
    ├── audio_001.wav
    ├── audio_002.wav
    └── timestamp.txt
```

## 运行自动化测试

在 `--input` 模式下，脚本会自行启动一个内嵌 uvicorn 服务，因此不需要手动先启动 X-Talk。

使用后端服务配置运行生成好的测试集：

```bash
python scripts/test.py --config server_configs/sample_local.json --input logs/tests/smoke --out logs/test_results/smoke
```

您也可以通过命令行覆盖运行参数：

```bash
python scripts/test.py --config server_configs/sample_local.json --input logs/tests/smoke --out logs/test_results/smoke --concurrency 2 --with-vad
```

## 输出结果

测试结果目录中会包含：

- `<case_name>.mp3`：该 case 的最终单声道录音。脚本会先基于双声道会话音频完成分析，再用高质量 MP3 编码压缩以节省空间
- `eval.json`：整体延迟和每个 case 的通过/失败汇总
- `logs/<case_name>.asr.json`：期望文本、实际 ASR 事件以及可选的评判结果
- `service_config.json`：本次运行使用的实际后端配置
- `test_config.json`：本次运行使用的实际测试配置

## 注意事项

- `--with-vad` 启用客户端 VAD。此时应从服务端配置中移除后端 `vad`，以避免重复 turn 事件。
- `--without-vad` 需要服务端配置中存在后端 `vad` 模型。
- 如果任何 case 启用了 `judge_asr`，则必须在 `test_config.json` 或命令行覆盖参数中提供 `judge_llm`。
