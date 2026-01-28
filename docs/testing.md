You can input audios to test X-Talk with `scripts/offline_client.py`.

## Creating Test Cases from Text

If you don't have audio files ready, you can use `scripts/create_test_case.py` to generate them from a transcription file using DashScope TTS API.

First, install the dependency and set your API key:

```bash
pip install requests
export DASHSCOPE_API_KEY=your_api_key
```

Create a transcription file with the format `<timestamp>:<text>` per line:

```plaintext
# transcription.txt
0:Hello, how are you today?
on_response_finish:I have another question for you.
5.0:This message will be sent at 5 seconds.
```

Where `<timestamp>` is:
- A float number: seconds from start (e.g., `0`, `5.0`, `10.5`)
- `on_response_finish`: wait for previous response to finish before sending

Then run the script to generate audio files:

```bash
# Voice: Cherry (default), Language: Auto (default)
python scripts/create_test_case.py --input transcription.txt --output /path/to/audio_dir

# Optional: specify voice and language
python scripts/create_test_case.py --input transcription.txt --output /path/to/audio_dir --voice Cherry --language Chinese
```

This will create:
- Audio files: `audio_000.wav`, `audio_001.wav`, etc.
- `timestamp.txt` in the format expected by `offline_client.py`

## Running Tests with Offline Client

First start an X-Talk server, remember the port, like 7634; then install dependencies for the offline client and prepare an audio directory with audio files and a `timestamp.txt` file for testing:

```bash
pip install websockets soundfile numpy soxr
```
```plaintext
/path/to/audio_dir/
├── audio1.wav
├── audio2.wav
└── timestamp.txt

timestamp.txt content:
audio1.wav:0
audio2.wav:on_response_finish
```

Then run the offline client with the WebSocket URL of the server and the input audio directory:

```bash
    python scripts/offline_client.py --ws ws://127.0.0.1:8000/ws --with-vad --input /path/to/audio_dir --output /path/to/recording.wav
```

You will see result in `/path/to/recording.wav`. See `scripts/offline_client.py` for more details.