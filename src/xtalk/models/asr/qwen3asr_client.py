import base64
import io
import logging
import os
import re
import wave
from typing import Any, Dict, List, Optional

import numpy as np
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from ..registry import model
from .interfaces import ASR

logger = logging.getLogger(__name__)

@model
class Qwen3ASRClient(ASR):
    TARGET_SAMPLE_RATE = 16000

    def __init__(
        self,
        base_url: str = "http://localhost:8001/v1/recognize",
        timeout: float = 10.0,  # 缩短超时时间，配合重试机制提高流式响应性
        **kwargs: Dict[str, Any],
    ):
        """
        Qwen3 ASR 客户端优化版
        """
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.kwargs = kwargs
        
        # 属性对齐：保持 0.6s 以降低首字延迟
        self.chunk_secs = 0.6
        
        self.session = requests.Session()
        self.dup_punc_pattern = re.compile(r'([。，！？,.!?])\1+')
        # 配置重试策略：仅针对网络抖动进行快速重试，不阻塞主流程
        retry_strategy = Retry(
            total=2,
            backoff_factor=0.1,
            status_forcelist=[500, 502, 503, 504],
        )
        
        # 配置连接池：增加池大小以应对高并发的流式请求，避免连接等待
        adapter = HTTPAdapter(
            pool_connections=20,
            pool_maxsize=50,
            max_retries=retry_strategy,
            pool_block=False 
        )
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        
        self.punc_pattern = re.compile(r'[。，！？,.!?]$')

    def __del__(self):
        self.close()

    def close(self):
        """显式关闭资源"""
        if hasattr(self, 'session'):
            self.session.close()

    def _post(self, payload: dict) -> dict:
        """支持 Keep-Alive 并快速处理异常"""
        try:
            headers = {"Connection": "keep-alive"}
            resp = self.session.post(
                self.base_url, 
                json=payload, 
                headers=headers,
                timeout=self.timeout
            )
            if resp.status_code == 200:
                return resp.json()
            logger.error(f"Qwen3 Server 返回异常: {resp.status_code}")
            return {}
        except Exception as e:
            logger.error(f"Qwen3 请求失败: {e}")
            return {}

    def _pcm_s16le_to_wav(self, pcm: np.ndarray, sample_rate: int) -> bytes:
        if pcm.dtype != np.int16:
            pcm = (pcm * 32768).clip(-32768, 32767).astype(np.int16)
        
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(pcm.tobytes())
        return buf.getvalue()

    def recognize(self, audio: np.ndarray) -> str:
        return self.recognize_stream(audio, {}, is_final=True)

    def recognize_stream(self, audio: np.ndarray, cache: dict, is_final: bool) -> str:
        try:
            if audio is None or len(audio) == 0:
                return ""
            if "session_id" not in cache:
                # 生成带时间戳或固定前缀的 ID，方便排查
                cache["session_id"] = f"qwen_{os.urandom(8).hex()}"
                cache["confirmed_len"] = 0 

            # 音频编码
            wav_bytes = self._pcm_s16le_to_wav(audio, self.TARGET_SAMPLE_RATE)
            data_url = f"data:audio/wav;base64,{base64.b64encode(wav_bytes).decode('ascii')}"
            
            payload = {
                "audio": data_url,
                "session_id": cache["session_id"],
                "is_final": is_final,
                **self.kwargs
            }
            
            resp_json = self._post(payload)
            full_text = resp_json.get("text", "").strip()
            full_text = self.dup_punc_pattern.sub(r'\1', full_text)
            if not full_text:
                return ""

            confirmed_len = cache.get("confirmed_len", 0)
            
            # 过滤中间态标点，防止固化
            processed_text = full_text
            if not is_final:
                processed_text = self.punc_pattern.sub('', full_text)

            if len(processed_text) >= confirmed_len:
                new_increment = processed_text[confirmed_len:]
                cache["confirmed_len"] = len(processed_text)
                return new_increment
            else:
                cache["confirmed_len"] = len(processed_text)
                return processed_text
            
        except Exception as e:
            logger.error(f"Qwen3 recognize_stream 异常: {e}")
            return ""

    def is_streaming(self) -> bool:
        return True

    def resample(self, audio: np.ndarray, ori_sampling_rate: int) -> np.ndarray:
        if ori_sampling_rate == self.TARGET_SAMPLE_RATE:
            return audio
        import soxr
        return soxr.resample(audio, ori_sampling_rate, self.TARGET_SAMPLE_RATE)

    def get_chunks(self, audio: np.ndarray, ori_sampling_rate: int) -> List[np.ndarray]:
        audio = self.resample(audio, ori_sampling_rate)
        chunk_stride = int(self.chunk_secs * self.TARGET_SAMPLE_RATE)
        
        if len(audio) == 0:
            return []
            
        chunks = []
        for i in range(0, len(audio), chunk_stride):
            chunk = audio[i : i + chunk_stride]
            if len(chunk) > 0:
                chunks.append(chunk)
        return chunks
