import asyncio
import numpy as np
import soundfile as sf
import librosa
import grpc
from grpc import aio
import logging
from typing import Optional, Generator, AsyncGenerator
from contextlib import asynccontextmanager
from .interfaces import TTS
from ..registry import model
import queue
import threading

# Append dependency helper
import importlib

# Default output sample rate (Hz) from the TTS service
DEFAULT_SERVER_SAMPLE_RATE = 24000

# Lazily import soxr to avoid raising errors when unused
_soxr_spec = importlib.util.find_spec("soxr")
if _soxr_spec is not None:
    import soxr  # type: ignore
else:
    soxr = None  # type: ignore

# Import generated protobuf files
try:
    from .grpc_pb import cosyvoice_pb2
    from .grpc_pb import cosyvoice_pb2_grpc
except ImportError as e:
    raise ImportError(
        "Failed to import CosyVoice protobuf files. Please ensure:\n"
        "1. cosyvoice_pb2.py and cosyvoice_pb2_grpc.py have been generated.\n"
        "2. The files are located in the expected path.\n"
        f"Error details: {e}"
    )

# CosyVoice logger uses xtalk log_utils so no second import is required


@model
class CosyVoiceLocal(TTS):
    """
    Local CosyVoice TTS implementation that talks to a gRPC service.
    Supports synthesis modes including sft, zero_shot, cross_lingual, instruct2, etc.
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 50000,
        mode: str = "zero_shot_by_spk_id",
        spk_id: str = "001",
        sample_rate: int = 48000,  # default 48 kHz
        stream: bool = False,
        speed: float = 1.0,
        format: str = "pcm",  # default PCM 16-bit
        text_frontend: bool = True,
    ):
        """
        Initialize the CosyVoice local client.

        Args:
            host: Server host.
            port: Server port (default 50000).
            mode: Inference mode, supports sft, zero_shot, cross_lingual, instruct2,
                instruct2_by_spk_id, zero_shot_by_spk_id.
            spk_id: Speaker ID.
            sample_rate: Target output sample rate (default 48000).
            stream: Whether to use streaming responses.
            speed: Speech speed control.
            format: Audio format ("pcm" for PCM 16-bit, "" for raw float32).
            text_frontend: Whether to enable the text frontend.
        """
        self.host = host
        self.port = port
        self.mode = mode
        self.spk_id = spk_id
        self._sample_rate = sample_rate
        self.stream = stream
        self.speed = speed
        self.format = format
        self.text_frontend = text_frontend

        # Validate mode
        valid_modes = [
            "sft",
            "zero_shot",
            "cross_lingual",
            "instruct2",
            "instruct2_by_spk_id",
            "zero_shot_by_spk_id",
            "register_spk",
        ]
        if mode not in valid_modes:
            raise ValueError(f"Unsupported mode {mode}, supported modes: {valid_modes}")

        # gRPC channel address
        self.channel_address = f"{host}:{port}"

        self._loop = asyncio.new_event_loop()
        self._loop_thread = threading.Thread(target=self._loop.run_forever, daemon=True)
        self._loop_thread.start()

        self._channel = None
        self._stub = None

        async def _init_grpc():
            # Create long-lived channel/stub once
            self._channel = aio.insecure_channel(self.channel_address)
            # Wait for readiness to avoid racing the first RPC
            await self._channel.channel_ready()
            self._stub = cosyvoice_pb2_grpc.CosyVoiceStub(self._channel)

        # Wait synchronously in the foreground thread for initialization
        asyncio.run_coroutine_threadsafe(_init_grpc(), self._loop).result()

    def clone(self):
        return CosyVoiceLocal(
            host=self.host,
            port=self.port,
            mode=self.mode,
            spk_id=self.spk_id,
            sample_rate=self.sample_rate,
            stream=self.stream,
            speed=self.speed,
            format=self.format,
            text_frontend=self.text_frontend,
        )

    def close(self):
        if not hasattr(self, "_loop"):
            return

        async def _shutdown():
            try:
                if self._channel is not None:
                    await self._channel.close(grace=2.0)
            finally:
                self._loop.stop()

        asyncio.run_coroutine_threadsafe(_shutdown(), self._loop).result()
        self._loop_thread.join(timeout=5.0)

    @staticmethod
    def _pcm_bytes_to_float32(pcm_bytes: bytes) -> np.ndarray:
        """PCM int16 bytes -> float32 ndarray in range [-1, 1]."""
        audio_int16 = np.frombuffer(pcm_bytes, dtype=np.int16)
        audio_float = audio_int16.astype(np.float32) / 32768.0
        return audio_float

    @staticmethod
    def _float32_to_pcm_bytes(audio_float: np.ndarray) -> bytes:
        """float32 ndarray [-1, 1] -> PCM int16 bytes."""
        audio_int16 = np.clip(audio_float * 32768.0, -32768, 32767).astype(np.int16)
        return audio_int16.tobytes()

    def _resample_pcm_block(
        self, pcm_bytes: bytes, orig_sr: int, target_sr: int
    ) -> bytes:
        """Resample PCM16 audio with soxr."""
        if soxr is None:
            raise RuntimeError("soxr is required for resampling, install via 'pip install soxr'")

        audio_float = self._pcm_bytes_to_float32(pcm_bytes)
        # soxr.resample supports 1D/2D np.float32 input
        resampled_float = soxr.resample(audio_float, orig_sr, target_sr)
        return self._float32_to_pcm_bytes(resampled_float)

    def synthesize(
        self,
        text: str,
        mode: Optional[str] = None,
        spk_id: Optional[str] = None,
        prompt_text: Optional[str] = None,
        prompt_wav: Optional[str] = None,
        instruct_text: Optional[str] = None,
        stream: Optional[bool] = None,
        speed: Optional[float] = None,
    ) -> bytes:
        """
        Synthesize speech.

        Args:
            text: Text to synthesize.
            mode: Optional mode overriding the default.
            spk_id: Optional speaker ID overriding the default.
            prompt_text: Prompt text (used in zero_shot mode).
            prompt_wav: Prompt audio path (used in zero_shot/cross_lingual/instruct2).
            instruct_text: Instruction text (used in instruct2 mode).
            stream: Override the streaming flag if provided.
            speed: Override speech speed if provided.

        Returns:
            bytes: Synthesized audio payload.
        """
        # Apply overrides if provided
        selected_mode = mode or self.mode
        selected_spk_id = spk_id or self.spk_id
        selected_stream = stream if stream is not None else self.stream
        selected_speed = speed if speed is not None else self.speed

        try:
            return asyncio.run_coroutine_threadsafe(
                self._async_synthesize(
                    text=text,
                    mode=selected_mode,
                    spk_id=selected_spk_id,
                    prompt_text=prompt_text,
                    prompt_wav=prompt_wav,
                    instruct_text=instruct_text,
                    stream=selected_stream,
                    speed=selected_speed,
                ),
                self._loop,
            ).result()
        except Exception as e:
            logging.error(f"TTS synthesis failed: {e}")
            raise RuntimeError(f"TTS synthesis failed: {e}")

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    def synthesize_stream(self, text: str) -> Generator[bytes, None, None]:
        """
        Streaming speech synthesis.

        Args:
            text: Text to synthesize.

        Yields:
            bytes: Audio chunks.
        """
        q: "queue.Queue[Optional[bytes]]" = queue.Queue(maxsize=1000)
        stop_evt = threading.Event()
        done_evt = threading.Event()
        exc_holder = []

        async def _worker():
            """
            Run the real async streaming logic in the background loop while
            keeping the original soxr.ResampleStream resampling.
            """
            # Create the streaming resampler on demand (same as original logic)
            rs = None
            if self.format == "pcm" and self._sample_rate != DEFAULT_SERVER_SAMPLE_RATE:
                if soxr is None:
                    exc_holder.append(
                        RuntimeError("soxr is required, install via 'pip install soxr'")
                    )
                    return
                rs = soxr.ResampleStream(
                    in_rate=DEFAULT_SERVER_SAMPLE_RATE,
                    out_rate=self._sample_rate,
                    num_channels=1,
                    dtype="float32",
                )

            call = None
            try:
                # Build the request (same structure as construct_request)
                req = self.construct_request(
                    text=text,
                    mode=self.mode,
                    spk_id=self.spk_id,
                    prompt_text=None,
                    prompt_wav=None,
                    instruct_text=None,
                    stream=True,
                    speed=self.speed,
                )

                # Issue streaming RPC on the long-lived stub
                call = self._stub.Inference(req)

                async for resp in call:
                    if stop_evt.is_set():
                        break
                    if not resp.tts_audio:
                        continue

                    chunk = resp.tts_audio

                    # Keep the original resampling logic
                    if rs is not None:
                        # Utility: pcm16 -> float32 [-1, 1]
                        audio_float = self._pcm_bytes_to_float32(chunk)
                        y = rs.resample_chunk(audio_float, last=False)
                        chunk = self._float32_to_pcm_bytes(y)

                    # Move potentially blocking q.put into a thread pool to avoid blocking the loop
                    await asyncio.to_thread(q.put, chunk, True)

                # Flush the resampler on normal exit
                if rs is not None:
                    y_last = rs.resample_chunk(
                        np.empty((0,), dtype=np.float32), last=True
                    )
                    if y_last.size > 0:
                        await asyncio.to_thread(
                            q.put, self._float32_to_pcm_bytes(y_last), True
                        )

            except Exception as e:
                exc_holder.append(e)
            finally:
                # Cancel the underlying stream to clean up on interruption/error
                try:
                    if call is not None:
                        call.cancel()  # proactively cancel the streaming call
                except Exception:
                    pass
                # Signal completion
                await asyncio.to_thread(q.put, None, True)
                done_evt.set()

        # Start worker on the background loop
        fut = asyncio.run_coroutine_threadsafe(_worker(), self._loop)

        try:
            # Consume synchronously on the foreground thread
            while True:
                # Surface async-side exceptions first
                if exc_holder:
                    raise exc_holder[0]
                item = q.get()
                if item is None:
                    break
                yield item
        except GeneratorExit:
            # User stopped early (e.g., playback stopped fetching)
            stop_evt.set()
            # Cancel background task and wait for cleanup
            fut.cancel()
            done_evt.wait(timeout=3.0)
            raise
        except Exception:
            stop_evt.set()
            fut.cancel()
            done_evt.wait(timeout=3.0)
            raise

    @asynccontextmanager
    async def _get_grpc_channel(self):
        """Async context manager that yields a gRPC channel."""
        channel = None
        try:
            channel = aio.insecure_channel(self.channel_address)
            yield channel
        except Exception as e:
            logging.error(f"gRPC channel error: {e}")
            raise
        finally:
            if channel:
                try:
                    await channel.close()
                except Exception as e:
                    # Ignore shutdown errors (common when loops are closing)
                    pass

    async def _async_synthesize_stream(self, text: str) -> AsyncGenerator[bytes, None]:
        """Async streaming synthesis."""
        import time

        try:
            async with self._get_grpc_channel() as channel:
                stub = cosyvoice_pb2_grpc.CosyVoiceStub(channel)

                try:
                    start_time = time.time()
                    last_time = start_time
                    chunk_index = 0

                    # Build streaming request
                    request = self.construct_request(
                        text=text,
                        mode=self.mode,
                        spk_id=self.spk_id,
                        prompt_text=None,
                        prompt_wav=None,
                        instruct_text=None,
                        stream=True,  # force streaming mode
                        speed=self.speed,
                    )

                    # Send request and receive streaming responses
                    response_stream = stub.Inference(request)

                    async for response in response_stream:
                        if response.tts_audio:
                            last_time = time.time()
                            chunk_index += 1
                            yield response.tts_audio
                except grpc.RpcError as e:
                    logging.error(f"Streaming RPC error: {e.code()}: {e.details()}")
                    raise RuntimeError(f"Streaming RPC error: {e.code()}: {e.details()}")
                except Exception as e:
                    logging.error(f"Streaming processing failed: {str(e)}", exc_info=True)
                    raise RuntimeError(f"Streaming processing failed: {str(e)}")

        except Exception as e:
            logging.error(f"Async streaming synthesis failed: {str(e)}")
            raise

    def load_wav(self, wav_path: str, target_sr: int = 16000) -> np.ndarray:
        """Load an audio file and resample to the target rate."""
        try:
            data, sample_rate = sf.read(wav_path, dtype="float32")
            # Convert multi-channel audio to mono
            if len(data.shape) > 1:
                data = np.mean(data, axis=1)
            # Resample if needed
            if sample_rate != target_sr:
                data = librosa.resample(data, orig_sr=sample_rate, target_sr=target_sr)
            # Add batch dimension (1, samples)
            return data.reshape(1, -1)
        except Exception as e:
            logging.error(f"Failed to load audio {wav_path}: {str(e)}")
            raise RuntimeError(f"Failed to load audio {wav_path}: {str(e)}")

    def convert_audio_bytes_to_ndarray(
        self, raw_audio: bytes, format: str = None
    ) -> np.ndarray:
        """Convert raw bytes to numpy array."""
        if not format or format == "":
            return np.frombuffer(raw_audio, dtype=np.float32).reshape(1, -1)
        elif format == "pcm":
            return np.frombuffer(raw_audio, dtype=np.int16).reshape(1, -1)
        else:
            raise ValueError(f"Unsupported format: {format}")

    def convert_audio_ndarray_to_bytes(self, array: np.ndarray) -> bytes:
        """Convert numpy array to raw bytes."""
        return array.astype(np.float32).tobytes()

    def construct_request(
        self,
        text: str,
        mode: str,
        spk_id: str,
        prompt_text: Optional[str] = None,
        prompt_wav: Optional[str] = None,
        instruct_text: Optional[str] = None,
        stream: bool = False,
        speed: float = 1.0,
    ) -> cosyvoice_pb2.Request:
        """Build a gRPC request object."""
        request = cosyvoice_pb2.Request()
        request.tts_text = text
        request.stream = stream
        request.speed = speed
        request.text_frontend = self.text_frontend
        request.format = self.format

        # Populate request by mode
        if mode == "sft":
            logging.info("Building sft request")
            sft_request = request.sft_request
            sft_request.spk_id = spk_id
        elif mode == "zero_shot":
            logging.info("Building zero_shot request")
            if not prompt_text or not prompt_wav:
                raise ValueError("zero_shot mode requires prompt_text and prompt_wav")
            zero_shot_request = request.zero_shot_request
            zero_shot_request.prompt_text = prompt_text
            prompt_speech = self.load_wav(prompt_wav, 16000)
            zero_shot_request.prompt_audio = self.convert_audio_ndarray_to_bytes(
                prompt_speech
            )
        elif mode == "cross_lingual":
            logging.info("Building cross_lingual request")
            if not prompt_wav:
                raise ValueError("cross_lingual mode requires prompt_wav")
            cross_lingual_request = request.cross_lingual_request
            prompt_speech = self.load_wav(prompt_wav, 16000)
            cross_lingual_request.prompt_audio = self.convert_audio_ndarray_to_bytes(
                prompt_speech
            )
        elif mode == "instruct2":
            logging.info("Building instruct2 request")
            if not instruct_text or not prompt_wav:
                raise ValueError("instruct2 mode requires instruct_text and prompt_wav")
            instruct2_request = request.instruct2_request
            instruct2_request.instruct_text = instruct_text
            prompt_speech = self.load_wav(prompt_wav, 16000)
            instruct2_request.prompt_audio = self.convert_audio_ndarray_to_bytes(
                prompt_speech
            )
        elif mode == "instruct2_by_spk_id":
            logging.info("Building instruct2_by_spk_id request")
            if not instruct_text:
                raise ValueError("instruct2_by_spk_id mode requires instruct_text")
            instruct2_by_spk_id_request = request.instruct2_by_spk_id_request
            instruct2_by_spk_id_request.instruct_text = instruct_text
            instruct2_by_spk_id_request.spk_id = spk_id
        else:  # zero_shot_by_spk_id
            logging.info("Building zero_shot_by_spk_id request")
            zero_shot_by_spk_id_request = request.zero_shot_by_spk_id_request
            zero_shot_by_spk_id_request.spk_id = spk_id

        return request

    async def _async_synthesize(
        self,
        text: str,
        mode: str,
        spk_id: str,
        prompt_text: Optional[str] = None,
        prompt_wav: Optional[str] = None,
        instruct_text: Optional[str] = None,
        stream: bool = False,
        speed: float = 1.0,
    ) -> bytes:
        """Asynchronous speech synthesis."""
        import time

        try:
            async with self._get_grpc_channel() as channel:
                stub = cosyvoice_pb2_grpc.CosyVoiceStub(channel)

                try:
                    start_time = time.time()
                    last_time = start_time
                    chunk_index = 0
                    tts_audio = b""

                    # Build request
                    request = self.construct_request(
                        text,
                        mode,
                        spk_id,
                        prompt_text,
                        prompt_wav,
                        instruct_text,
                        stream,
                        speed,
                    )

                    # Send request and aggregate responses
                    response_stream = stub.Inference(request)

                    async for response in response_stream:
                        tts_audio += response.tts_audio
                        chunk_index += 1

                    # Resample all audio chunks to the target rate if required
                    if (
                        self.format == "pcm"
                        and self._sample_rate != DEFAULT_SERVER_SAMPLE_RATE
                        and len(tts_audio) > 0
                    ):
                        try:
                            tts_audio = self._resample_pcm_block(
                                tts_audio,
                                orig_sr=DEFAULT_SERVER_SAMPLE_RATE,
                                target_sr=self._sample_rate,
                            )
                        except Exception as e:
                            logging.error(f"Offline resampling failed: {e}")
                            raise RuntimeError(f"Offline resampling failed: {e}")

                    return tts_audio

                except grpc.RpcError as e:
                    logging.error(f"RPC error: {e.code()}: {e.details()}")
                    raise RuntimeError(f"RPC error: {e.code()}: {e.details()}")
                except Exception as e:
                    logging.error(f"Processing failed: {str(e)}", exc_info=True)
                    raise RuntimeError(f"Processing failed: {str(e)}")

        except Exception as e:
            logging.error(f"Async synthesis failed: {str(e)}")
            raise

    def save_audio(self, audio_data: bytes, output_path: str):
        """Persist synthesized audio to a file."""
        try:
            if self.format == "" or self.format is None:
                # Raw float32 format
                audio_array = self.convert_audio_bytes_to_ndarray(audio_data)
                # Transpose to (samples, channels) for soundfile
                sf.write(output_path, audio_array.T, self.sample_rate)
            elif self.format == "pcm":
                # PCM int16 format
                audio_array = self.convert_audio_bytes_to_ndarray(audio_data, "pcm")
                # Transpose to (samples, channels) for soundfile
                sf.write(output_path, audio_array.T, self.sample_rate)
            else:
                # Write raw bytes directly
                with open(output_path, "wb") as f:
                    f.write(audio_data)

            # Compute duration
            if self.format in ["", None, "pcm"]:
                audio_array = self.convert_audio_bytes_to_ndarray(
                    audio_data, self.format
                )
                duration = audio_array.shape[1] / self.sample_rate
                logging.info(f"Audio saved to {output_path} (duration: {duration:.2f}s)")
        except Exception as e:
            logging.error(f"Failed to save audio: {str(e)}")
            raise RuntimeError(f"Failed to save audio: {str(e)}")

    def register_speaker(self, spk_id: str, prompt_text: str, prompt_wav: str) -> bool:
        """
        Register a custom speaker.

        Args:
            spk_id: Speaker identifier.
            prompt_text: Prompt text.
            prompt_wav: Prompt audio file path.

        Returns:
            bool: Whether registration succeeded.
        """
        try:
            return asyncio.run(
                self._async_register_speaker(spk_id, prompt_text, prompt_wav)
            )
        except Exception as e:
            logging.error(f"Speaker registration failed: {str(e)}")
            raise RuntimeError(f"Speaker registration failed: {str(e)}")

    async def _async_register_speaker(
        self, spk_id: str, prompt_text: str, prompt_wav: str
    ) -> bool:
        """Asynchronously register a speaker."""
        try:
            async with self._get_grpc_channel() as channel:
                stub = cosyvoice_pb2_grpc.CosyVoiceStub(channel)

                try:
                    # Load audio data
                    audio_data = self.load_wav(prompt_wav, 16000)
                    audio_bytes = self.convert_audio_ndarray_to_bytes(audio_data)

                    # Build registration request
                    request = cosyvoice_pb2.RegisterSpkRequest(
                        spk_id=spk_id,
                        prompt_text=prompt_text,
                        prompt_audio_bytes=audio_bytes,
                        ori_sample_rate=16000,
                    )

                    # Send registration request
                    response = await stub.RegisterSpk(request)
                    logging.info(f"Speaker registration response: {response}")

                    if response.status == cosyvoice_pb2.RegisterSpkResponse.Status.OK:
                        logging.info(f"Speaker {spk_id} registered successfully")
                        return True
                    else:
                        logging.error(f"Speaker {spk_id} registration failed")
                        return False

                except grpc.RpcError as e:
                    logging.error(f"Registration RPC error: {e.code()}: {e.details()}")
                    raise RuntimeError(
                        f"Registration RPC error: {e.code()}: {e.details()}"
                    )
                except Exception as e:
                    logging.error(f"Registration failed: {str(e)}", exc_info=True)
                    raise RuntimeError(f"Registration failed: {str(e)}")

        except Exception as e:
            logging.error(f"Async registration failed: {str(e)}")
            raise

    def set_mode(self, mode: str):
        """Set the default inference mode."""
        valid_modes = [
            "sft",
            "zero_shot",
            "cross_lingual",
            "instruct2",
            "instruct2_by_spk_id",
            "zero_shot_by_spk_id",
        ]
        if mode not in valid_modes:
            raise ValueError(f"Unsupported mode {mode}, supported modes: {valid_modes}")
        self.mode = mode

    def set_spk_id(self, spk_id: str):
        """Set the default speaker ID."""
        self.spk_id = spk_id

    def set_speed(self, speed: float):
        """Set the default speech speed."""
        self.speed = speed

    def set_format(self, format: str):
        """Set the default audio format."""
        if format not in ["", "pcm"]:
            raise ValueError(f"Unsupported audio format {format}, allowed: '', 'pcm'")
        self.format = format
