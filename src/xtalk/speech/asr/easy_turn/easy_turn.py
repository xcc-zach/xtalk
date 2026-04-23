import grpc
import numpy as np
import re
from ...interfaces import ASR

from .audio_service_pb2 import AudioRequest, HealthRequest
from .audio_service_pb2_grpc import AudioInferenceServiceStub


class EasyTurnASR(ASR):
    """
    Easy Turn ASR model using gRPC service for speech recognition.
    
    This model connects to a gRPC speech recognition service and performs
    speech-to-text conversion with turn detection capabilities.
    """

    def __init__(self, 
                 host: str = 'localhost', 
                 port: int = 50051,
                 language: str = 'zh',
                 task: str = '<TRANSCRIBE> <BACKCHANNEL> <COMPLETE>',
                 timeout: float = 30.0):
        """
        Initialize the EasyTurnASR model.

        Args:
            host (str): gRPC server host address
            port (int): gRPC server port
            language (str): Language code for recognition (default: 'zh' for Chinese)
            task (str): Recognition task type (default: '<TRANSCRIBE>')
            timeout (float): Request timeout in seconds
        """
        self.host = host
        self.port = port
        self.language = language
        self.task = task
        self.timeout = timeout
        self._channel = None
        self._stub = None
        # Streaming recognition buffer and latest transcript
        self._stream_buffer = bytearray()
        self._latest_stream_text = ""

    def _get_connection(self):
        """
        Get or create gRPC connection.
        
        Returns:
            AudioInferenceServiceStub: gRPC stub for audio inference
        """
        if self._channel is None:
            self._channel = grpc.insecure_channel(f'{self.host}:{self.port}')
            self._stub = AudioInferenceServiceStub(self._channel)
        return self._stub

    def recognize(self, audio) -> str:
        """
        Convert audio to text using gRPC service.

        Args:
            audio: PCM bytes or float32 numpy array.

        Returns:
            str: The recognized text.
        """
        def _do_call() -> str:
            stub = self._get_connection()

            # Support PCM bytes or numpy arrays by converting to float32 [-1, 1]
            if isinstance(audio, (bytes, bytearray, memoryview)):
                pcm = bytes(audio)
                audio_np = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
            elif isinstance(audio, np.ndarray):
                # For numpy input, treat ints as PCM and cast floats to float32
                if np.issubdtype(audio.dtype, np.integer):
                    audio_np = audio.astype(np.float32) / 32768.0
                elif np.issubdtype(audio.dtype, np.floating):
                    audio_np = audio.astype(np.float32)
                else:
                    raise TypeError("Unsupported numpy dtype for audio input")
            else:
                raise TypeError("Unsupported audio input type")

            request = AudioRequest(
                audio_data=audio_np.tobytes(),
                audio_data_shape=list(audio_np.shape),
                sample_rate=16000,  # Default sample rate
                task=self.task,
                lang=self.language,
            )

            response = stub.RecognizeAudio(request, timeout=self.timeout)
            if response.success:
                return response.result_text
            raise RuntimeError(f"Recognition failed: {response.error_message}")

        try:
            return _do_call()
        except Exception:
            # Reset the connection and retry once when the channel is closed/failed
            self.close()
            try:
                return _do_call()
            except Exception as inner:
                raise RuntimeError(f"Recognition error after retry: {inner}") from inner

    def recognize_stream(self, audio: bytes, *, is_final: bool = False) -> str:
        """
        Streaming recognition: buffer incremental PCM data and return accumulated text.
        """
        if isinstance(audio, (bytes, bytearray, memoryview)):
            chunk = bytes(audio)
        else:
            raise TypeError("EasyTurnASR recognize_stream expects PCM bytes")

        if chunk:
            self._stream_buffer.extend(chunk)

        should_update = bool(chunk) or (is_final and self._stream_buffer)
        if should_update:
            buffered = bytes(self._stream_buffer)
            if buffered:
                self._latest_stream_text = self.recognize(buffered)

        result = self._latest_stream_text
        if not is_final:
            result = self._strip_trailing_tags(result)

        return result

    @staticmethod
    def _strip_trailing_tags(text: str) -> str:
        """Strip trailing <...> tags during streaming to hide control markers."""
        if not text:
            return text
        return re.sub(r"(?:\s*<[^>]+>)+\s*$", "", text)

    def stream_chunk_bytes_hint(self) -> int | None:
        """Return the recommended streaming trigger size (~0.6 s @ 16k16bit)."""
        return 19200


    def health_check(self) -> dict:
        """
        Check the health of the gRPC service.

        Returns:
            dict: Dictionary containing health status and message
        """
        try:
            # Get gRPC connection
            stub = self._get_connection()
            
            # Create health check request
            request = HealthRequest()
            
            # Make gRPC call
            response = stub.HealthCheck(request, timeout=self.timeout)
            
            return {
                'status': response.status,
                'message': response.message
            }
                
        except grpc.RpcError as e:
            raise RuntimeError(f"gRPC health check error: {str(e)}") from e
        except Exception as e:
            raise RuntimeError(f"Health check error: {str(e)}") from e

    def reset(self) -> None:
        """Reset internal connection state."""
        self._stream_buffer.clear()
        self._latest_stream_text = ""
        self.close()

    def clone(self) -> "EasyTurnASR":
        """Create an independent clone sharing config but not connections."""
        return EasyTurnASR(
            host=self.host,
            port=self.port,
            language=self.language,
            task=self.task,
            timeout=self.timeout,
        )

    def close(self):
        """
        Close the gRPC connection.
        """
        if self._channel is not None:
            self._channel.close()
            self._channel = None
            self._stub = None

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()

    def __del__(self):
        """Destructor to ensure connection is closed."""
        self.close()


# Example usage and testing function
def test_easy_turn_asr():
    """
    Test function for EasyTurnASR.
    """
    
    # Generate test audio data
    sample_rate = 16000
    duration = 2  # 2 seconds
    samples = int(sample_rate * duration)
    audio_data = np.random.randn(samples).astype(np.float32)
    
    # Test the ASR
    try:
        with EasyTurnASR() as asr:
            print("Testing EasyTurnASR...")
            
            # Test health check first
            health = asr.health_check()
            print(f"Health check: {health}")
            
            # Test basic recognition
            result = asr.recognize(audio_data)
            print(f"Recognition result: {result}")
            
            # Test with metrics
            metrics = asr.recognize_with_metrics(audio_data)
            print(f"Detailed metrics: {metrics}")
            
    except (RuntimeError, ConnectionError) as e:
        print(f"Test failed: {e}")


if __name__ == "__main__":
    test_easy_turn_asr()
