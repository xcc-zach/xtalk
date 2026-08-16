*Experimental functionality*

X-Talk provides a single script, [`scripts/test.py`](https://github.com/xcc-zach/xtalk/blob/main/scripts/test.py), for both test-set generation and automated backend evaluation.

It has two modes:

- `--create`: generate a runnable audio dataset from text templates
- `--input`: start an embedded X-Talk server and run the dataset automatically

## Preparing a Test Template

Create a dataset root with:

- one TTS config JSON at the root, such as `tts_config.json`
- an optional `test_config.json` at the root
- one or more case subdirectories

For example:

```plaintext
logs/test_templates/smoke/
├── tts_config.json
├── test_config.json
└── basic_turn/
    └── timestamp.txt
```

## Writing `tts_config.json`

In `--create` mode, the script looks for one root JSON file such as `tts_config.json`, `config.json`, or `sample_local.json`.

That file can use either of these shapes:

1. a standalone TTS model config
2. a full X-Talk service config that contains a top-level `tts` field

For clarity, we recommend a standalone `tts_config.json` dedicated to dataset generation.

Minimal example with `IndexTTS`:

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

Equivalent full-service form:

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

`type` should match the Python class name of the TTS model, and `params` should match that class's initialization arguments. See [Supported Models](../technical_reference/supported_models.md) for available TTS backends and their optional dependencies.

For `IndexTTS`, each `voices` entry needs:

- `name`: the voice identifier exposed to X-Talk
- `path`: a reference WAV file, or a directory containing reference audio files

If you already have a working server config, reusing its `tts` section is usually the easiest option.

## Optional Configuration

You may place an optional `test_config.json` at the dataset root. It supports:

- `concurrency`
- `with_vad`
- `vad_redemption_ms`
- `judge_llm`

Example:

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

Each case directory may also contain an optional `criteria.yaml`:

```yaml
judge_asr: true
```

When `judge_asr: true` is enabled, every runnable `timestamp.txt` entry must include the expected transcript text as the third column. Datasets produced by `--create` already satisfy this format automatically.

## Creating `timestamp.txt`

Each case directory must contain a `timestamp.txt`. In `--create` mode, each line uses the format `<time_spec>:<text>`:

```plaintext
# basic_turn/timestamp.txt
0:Hello, how are you today?
ai_end:Tell me more about your plan.
ai_end+2.5:I also want to ask about pricing.
```

`<time_spec>` can be:

- an absolute second value such as `0`, `5.0`, or `10.5`
- `ai_start`
- `ai_end`
- `user_start`
- `user_end`
- one of the anchors above plus an offset such as `ai_end+2.5`

Relative timestamps are resolved in file order. `ai_*` anchors refer to the next AI response triggered by the previous user clip, while `user_*` anchors refer to the previous user clip itself.

## Generating a Runnable Dataset

Install the script dependencies first:

```bash
pip install numpy requests soundfile websockets pyyaml uvicorn fastapi
```

Optional:

```bash
pip install soxr
```

Install the optional X-Talk dependency required by your chosen TTS backend as well. For example, `IndexTTS` needs the `index-tts` extra.

Then generate the dataset:

```bash
python scripts/test.py --create logs/test_templates/smoke --out logs/tests
```

The script loads the TTS model from the root JSON config, synthesizes one WAV file per line, and writes runnable case folders. The generated `timestamp.txt` will use the format `<time_spec>:<audio_file>:<expected_text>`.

For example:

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

## Running Automated Tests

In `--input` mode, the script starts an embedded uvicorn server by itself. You do not need to manually start X-Talk first.

Run the generated dataset against a backend service config:

```bash
python scripts/test.py --config server_configs/sample_local.json --input logs/tests/smoke --out logs/test_results/smoke
```

You can also override runtime options from the command line:

```bash
python scripts/test.py --config server_configs/sample_local.json --input logs/tests/smoke --out logs/test_results/smoke --concurrency 2 --with-vad
```

## Outputs

The test result folder contains:

- `<case_name>.mp3`: the final mono recording for that case, compressed from the analyzed stereo session audio with high-quality MP3 encoding to save space
- `eval.json`: overall latency and per-case pass/fail summary
- `logs/<case_name>.asr.json`: expected transcripts, observed ASR events, and optional judge results
- `service_config.json`: the effective backend config used for the run
- `test_config.json`: the effective dataset runtime config used for the run

## Notes

- `--with-vad` enables client-side VAD. In that mode, remove backend `vad` from the server config to avoid duplicate turn events.
- `--without-vad` requires a backend `vad` model in the server config.
- If `judge_asr` is enabled for any case, configure `judge_llm` either in `test_config.json` or via CLI overrides.
