#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gRPC audio inference service that accepts numpy arrays or file paths.
"""

import grpc
from concurrent import futures
import numpy as np
import os
import sys
import time
import logging
import tempfile
import traceback

# Add wenet binary path
current_dir = os.path.dirname(os.path.abspath(__file__))
wenet_bin_dir = os.path.join(current_dir, 'wenet', 'bin')
sys.path.insert(0, wenet_bin_dir)

from recognize_single_audio import SingleAudioRecognizer

# Import generated protobuf stubs
try:
    import audio_service_pb2
    import audio_service_pb2_grpc
except ImportError:
    print(
        "Please run 'python -m grpc_tools.protoc --python_out=. "
        "--grpc_python_out=. audio_service.proto' to generate protobuf stubs."
    )
    sys.exit(1)


class AudioInferenceService(audio_service_pb2_grpc.AudioInferenceServiceServicer):
    """Audio inference service implementation."""
    
    def __init__(self, config_path, checkpoint_path, device="cpu", dtype="fp32", gpu=-1):
        """
        Initialize the service.

        Args:
            config_path(str): Path to the model config file.
            checkpoint_path(str): Path to the model checkpoint.
            device(str): Device type.
            dtype(str): Data type.
            gpu(int): GPU ID.
        """
        self.config_path = config_path
        self.checkpoint_path = checkpoint_path
        self.device = device
        self.dtype = dtype
        self.gpu = gpu
        
        # Initialize recognizer
        self.recognizer = None
        self._initialize_recognizer()
    
    def _initialize_recognizer(self):
        """Initialize the underlying recognizer."""
        try:
            logging.info("Initializing audio recognizer...")
            logging.info("Config path: %s", self.config_path)
            logging.info("Checkpoint path: %s", self.checkpoint_path)
            logging.info(
                "Device: %s, dtype: %s, GPU: %d", self.device, self.dtype, self.gpu
            )
            
            self.recognizer = SingleAudioRecognizer(
                config_path=self.config_path,
                checkpoint_path=self.checkpoint_path,
                device=self.device,
                dtype=self.dtype,
                gpu=self.gpu,
                verbose=False
            )
            logging.info("Audio recognizer initialized successfully")
        except Exception as e:
            logging.error("Failed to initialize audio recognizer: %s", e)
            logging.error("Exception type: %s", type(e).__name__)
            logging.error("Full traceback:")
            logging.error(traceback.format_exc())
            raise
    
    def _save_audio_from_numpy(self, audio_data: np.ndarray, sample_rate: int = 16000) -> str:
        """
        Save a numpy array as a temporary audio file.
        
        Args:
            audio_data(np.ndarray): Audio samples.
            sample_rate(int): Sample rate.
            
        Returns:
            str: Path to the temp file.
        """
        import soundfile as sf
        
        logging.info("_save_audio_from_numpy started")
        logging.info("Input audio shape: %s", audio_data.shape)
        logging.info("Input audio dtype: %s", audio_data.dtype)
        logging.info("Input sample rate: %d", sample_rate)
        
        # Create a temp file
        temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        temp_path = temp_file.name
        temp_file.close()
        logging.info("Created temp file: %s", temp_path)
        
        # Ensure audio has the correct shape
        if audio_data.ndim > 1:
            logging.info("Detected multi-dimensional audio: %d dims", audio_data.ndim)
            # For multi-channel audio, average or squeeze
            if audio_data.shape[0] > 1:
                logging.info("Multi-channel audio detected; averaging channels")
                audio_data = np.mean(audio_data, axis=0)
            else:
                logging.info("Single-channel audio; squeezing dimension")
                audio_data = audio_data.squeeze(0)
            logging.info("Audio shape after processing: %s", audio_data.shape)
        
        # Ensure dtype is correct
        if audio_data.dtype != np.float32:
            logging.info("Converting dtype from %s to float32", audio_data.dtype)
            audio_data = audio_data.astype(np.float32)
        
        # Save the audio data
        try:
            logging.info("Saving audio file to: %s", temp_path)
            logging.info("Final audio shape: %s, dtype: %s", audio_data.shape, audio_data.dtype)
            sf.write(temp_path, audio_data, sample_rate)
            logging.info("Audio file saved")
        except Exception as e:
            logging.error("Failed to save audio file: %s", e)
            logging.error("Exception type: %s", type(e).__name__)
            logging.error(
                "Audio data shape: %s, dtype: %s", audio_data.shape, audio_data.dtype
            )
            logging.error("Full traceback:")
            logging.error(traceback.format_exc())
            raise
        
        return temp_path
    
    def RecognizeAudio(self, request, context):
        """
        Handle audio inference requests.

        Args:
            request: Incoming request containing audio payload.
            context: gRPC context.

        Returns:
            AudioResponse: Inference result.
        """
        try:
            start_time = time.time()
            
            # Handle audio payload
            if request.audio_data:
                # Decode numpy-style payload
                try:
                    logging.info(
                        "Processing numpy audio payload, size: %d bytes",
                        len(request.audio_data),
                    )
                    logging.info(
                        "Audio shape metadata: %s",
                        list(request.audio_data_shape)
                        if request.audio_data_shape
                        else "None",
                    )
                    logging.info("Sample rate: %d", request.sample_rate)
                    
                    audio_array = np.frombuffer(request.audio_data, dtype=np.float32)
                    logging.info("Array shape from buffer: %s", audio_array.shape)
                    logging.info("Array dtype from buffer: %s", audio_array.dtype)
                    
                    if request.audio_data_shape:
                        # Reshape when shape metadata is provided
                        shape = list(request.audio_data_shape)
                        logging.info("Reshaping array to: %s", shape)
                        audio_array = audio_array.reshape(shape)
                        logging.info("Array shape after reshape: %s", audio_array.shape)
                    
                    # Validate payload
                    if audio_array.size == 0:
                        logging.error("Audio payload is empty")
                        context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                        context.set_details("Audio payload is empty")
                        return audio_service_pb2.AudioResponse(
                            success=False,
                            error_message="Audio payload is empty",
                        )
                    
                    # Save to a temp file
                    sample_rate = request.sample_rate if request.sample_rate > 0 else 16000
                    logging.info("Saving audio file with sample rate: %d", sample_rate)
                    temp_audio_path = self._save_audio_from_numpy(audio_array, sample_rate)
                    logging.info("Audio file stored at: %s", temp_audio_path)
                    
                    # Track whether we should delete the temp file
                    cleanup_temp_file = True
                except Exception as e:
                    logging.error("Failed to process audio payload: %s", e)
                    logging.error("Exception type: %s", type(e).__name__)
                    logging.error("Full traceback:")
                    logging.error(traceback.format_exc())
                    context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                    context.set_details(f"Failed to process audio payload: {str(e)}")
                    return audio_service_pb2.AudioResponse(
                        success=False,
                        error_message=f"Failed to process audio payload: {str(e)}",
                    )
            else:
                # Use the provided file path
                temp_audio_path = request.audio_path
                cleanup_temp_file = False
            
            # Run inference
            logging.info("Starting inference, audio path: %s", temp_audio_path)
            logging.info(
                "Inference params - task: %s, lang: %s, speaker: %s",
                request.task if request.task else "<TRANSCRIBE>",
                request.lang if request.lang else "<CN>",
                request.speaker if request.speaker else "unknown",
            )
            
            try:
                result = self.recognizer.recognize(
                    audio_path=temp_audio_path,
                    task=request.task if request.task else "<TRANSCRIBE>",
                    lang=request.lang if request.lang else "<CN>",
                    speaker=request.speaker if request.speaker else "unknown",
                    text=request.text if request.text else "",
                    key=request.key if request.key else f"grpc_{int(time.time())}",
                    duration=request.duration if request.duration > 0 else None,
                    return_metadata=True
                )
                logging.info("Inference finished, result: %s", result)
            except Exception as e:
                logging.error("Exception during inference: %s", str(e))
                logging.error("Exception type: %s", type(e).__name__)
                logging.error("Full traceback:")
                logging.error(traceback.format_exc())
                raise
            
            # Clean up temp file
            if cleanup_temp_file and os.path.exists(temp_audio_path):
                os.unlink(temp_audio_path)
            
            if result is None:
                logging.error("Inference result is empty")
                context.set_code(grpc.StatusCode.INTERNAL)
                context.set_details("Inference failed")
                return audio_service_pb2.AudioResponse(
                    success=False,
                    error_message="Inference failed",
                )
            
            # Build response
            logging.info("Building response, result type: %s", type(result))
            logging.info("Result payload: %s", result)
            
            # Extract and cast response fields safely
            result_text = str(result.get("result", ""))
            inference_time = float(result.get("inference_time", 0.0))
            audio_path = str(result.get("audio_path", ""))
            key = str(result.get("key", ""))
            task = str(result.get("task", ""))
            lang = str(result.get("lang", ""))
            speaker = str(result.get("speaker", ""))
            prompt = str(result.get("prompt", ""))
            total_time = float(time.time() - start_time)
            
            logging.info("Response - result_text: %s (type: %s)", result_text, type(result_text))
            logging.info("Response - inference_time: %s (type: %s)", inference_time, type(inference_time))
            logging.info("Response - audio_path: %s (type: %s)", audio_path, type(audio_path))
            logging.info("Response - key: %s (type: %s)", key, type(key))
            logging.info("Response - task: %s (type: %s)", task, type(task))
            logging.info("Response - lang: %s (type: %s)", lang, type(lang))
            logging.info("Response - speaker: %s (type: %s)", speaker, type(speaker))
            logging.info("Response - prompt: %s (type: %s)", prompt, type(prompt))
            logging.info("Response - total_time: %s (type: %s)", total_time, type(total_time))
            
            response = audio_service_pb2.AudioResponse(
                success=True,
                result_text=result_text,
                inference_time=inference_time,
                audio_path=audio_path,
                key=key,
                task=task,
                lang=lang,
                speaker=speaker,
                prompt=prompt,
                total_time=total_time
            )
            
            return response
            
        except Exception as e:
            logging.error("Unexpected error during inference: %s", e)
            logging.error("Exception type: %s", type(e).__name__)
            logging.error("Exception details: %s", str(e))
            logging.error("Full traceback:")
            logging.error(traceback.format_exc())
            
            # Safely build an error response
            try:
                error_message = f"Unexpected error during inference: {str(e)}"
                context.set_code(grpc.StatusCode.INTERNAL)
                context.set_details(error_message)
                return audio_service_pb2.AudioResponse(
                    success=False,
                    error_message=error_message
                )
            except Exception as response_error:
                logging.error("Failed to construct error response: %s", response_error)
                logging.error("Response construction traceback:")
                logging.error(traceback.format_exc())
                # Return a minimal error response
                return audio_service_pb2.AudioResponse(
                    success=False,
                    error_message="Internal service error"
                )
    
    def HealthCheck(self, request, context):
        """
        Health check endpoint.

        Args:
            request: Health check request.
            context: gRPC context.

        Returns:
            HealthResponse: Health status payload.
        """
        try:
            logging.info("Performing health check")
            # Ensure recognizer is ready
            if self.recognizer is None:
                logging.warning("Recognizer is not initialized")
                return audio_service_pb2.HealthResponse(
                    status="UNHEALTHY",
                    message="Recognizer is not initialized",
                )
            
            logging.info("Health check passed")
            return audio_service_pb2.HealthResponse(
                status="HEALTHY",
                message="Service is running normally",
            )
        except Exception as e:
            logging.error("Health check failed: %s", e)
            logging.error("Health check exception type: %s", type(e).__name__)
            logging.error("Health check traceback:")
            logging.error(traceback.format_exc())
            return audio_service_pb2.HealthResponse(
                status="UNHEALTHY",
                message=f"Health check failed: {str(e)}",
            )


def serve(config_path, checkpoint_path, device="cpu", dtype="fp32", gpu=-1, 
          host="0.0.0.0", port=50051, max_workers=10):
    """
    Start the gRPC service.

    Args:
        config_path(str): Path to the config file.
        checkpoint_path(str): Path to the checkpoint.
        device(str): Device type.
        dtype(str): Data type.
        gpu(int): GPU ID.
        host(str): Host to bind to.
        port(int): Service port.
        max_workers(int): Max worker threads.
    """
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create server
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=max_workers))
    
    # Register service
    audio_service = AudioInferenceService(
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        device=device,
        dtype=dtype,
        gpu=gpu
    )
    
    audio_service_pb2_grpc.add_AudioInferenceServiceServicer_to_server(
        audio_service, server
    )
    
    # Start server
    server.add_insecure_port(f'{host}:{port}')
    server.start()
    
    logging.info("gRPC audio inference service is listening at %s:%s", host, port)
    logging.info("Config file: %s", config_path)
    logging.info("Checkpoint: %s", checkpoint_path)
    logging.info("Device: %s, dtype: %s", device, dtype)
    
    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        logging.info("Shutting down service...")
        server.stop(0)


def main():
    """CLI entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="gRPC audio inference service")
    
    # Required arguments
    parser.add_argument('--config', required=True, help='Path to config file')
    parser.add_argument('--checkpoint', required=True, help='Path to checkpoint')
    
    # Optional arguments
    parser.add_argument('--device', default='cpu', choices=['cpu', 'cuda', 'npu'], help='Device type')
    parser.add_argument('--dtype', default='fp32', choices=['fp16', 'fp32', 'bf16'], help='Data type')
    parser.add_argument('--gpu', type=int, default=-1, help='GPU ID')
    parser.add_argument('--host', default='0.0.0.0', help='Host address')
    parser.add_argument('--port', type=int, default=50051, help='Service port')
    parser.add_argument('--max_workers', type=int, default=10, help='Max worker threads')
    
    args = parser.parse_args()
    
    # Launch service
    serve(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        device=args.device,
        dtype=args.dtype,
        gpu=args.gpu,
        host=args.host,
        port=args.port,
        max_workers=args.max_workers
    )


if __name__ == '__main__':
    main()
