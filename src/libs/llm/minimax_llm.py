"""MiniMax Anthropic-compatible LLM implementation."""

from __future__ import annotations

import os
from typing import Any, Dict, Generator, List, Optional

import httpx

from src.libs.llm.base_llm import BaseLLM, ChatResponse, Message


class MiniMaxLLMError(RuntimeError):
    """Raised when MiniMax API call fails."""


class StreamChunk:
    """Represents a chunk from streaming response."""
    def __init__(self, content: str):
        self.content = content


class MiniMaxLLM(BaseLLM):
    """MiniMax provider using the Anthropic-compatible messages endpoint."""

    DEFAULT_BASE_URL = "https://api.minimaxi.com/anthropic/v1"
    DEFAULT_TIMEOUT_SECONDS = 180.0

    @staticmethod
    def _normalize_base_url(raw_url: str) -> str:
        url = str(raw_url or MiniMaxLLM.DEFAULT_BASE_URL).rstrip("/")
        if url.endswith("/v1"):
            return url
        if url.endswith("/anthropic"):
            return f"{url}/v1"
        return url

    def __init__(
        self,
        settings: Any,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        self.model = settings.llm.model
        self.default_temperature = settings.llm.temperature
        self.default_max_tokens = settings.llm.max_tokens
        self.request_timeout = float(
            kwargs.pop("timeout", os.environ.get("MINIMAX_LLM_TIMEOUT_SEC", self.DEFAULT_TIMEOUT_SECONDS))
        )

        self.api_key = (
            api_key
            or os.environ.get("MINIMAX_API_KEY")
            or getattr(settings.llm, "api_key", None)
            or os.environ.get("LLM_API_KEY")
        )
        if not self.api_key:
            raise ValueError(
                "MiniMax API key not provided. Set MINIMAX_API_KEY / LLM_API_KEY or settings.llm.api_key."
            )

        self.base_url = self._normalize_base_url(
            (
            base_url
            or getattr(settings.llm, "base_url", None)
            or os.environ.get("MINIMAX_API_URL")
            or os.environ.get("MINIMAX_AI_URL")
            or self.DEFAULT_BASE_URL
            )
        )
        self.api_version = kwargs.pop("anthropic_version", os.environ.get("MINIMAX_ANTHROPIC_VERSION", "2023-06-01"))

    def chat(
        self,
        messages: List[Message],
        trace: Optional[Any] = None,
        **kwargs: Any,
    ) -> ChatResponse:
        self.validate_messages(messages)

        temperature = kwargs.get("temperature", self.default_temperature)
        max_tokens = kwargs.get("max_tokens", self.default_max_tokens)
        model = kwargs.get("model", self.model)

        system_chunks: List[str] = []
        anthropic_messages: List[Dict[str, Any]] = []
        for msg in messages:
            if msg.role == "system":
                system_chunks.append(msg.content.strip())
                continue
            anthropic_messages.append(
                {
                    "role": "assistant" if msg.role == "assistant" else "user",
                    "content": [{"type": "text", "text": msg.content}],
                }
            )

        if not anthropic_messages:
            anthropic_messages.append(
                {"role": "user", "content": [{"type": "text", "text": "请继续。"}]}
            )

        payload: Dict[str, Any] = {
            "model": model,
            "messages": anthropic_messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if system_chunks:
            payload["system"] = "\n\n".join(system_chunks)

        response_data = self._call_api(payload)

        content_blocks = response_data.get("content", [])
        text_parts = [block.get("text", "") for block in content_blocks if block.get("type") == "text" and block.get("text")]
        thinking_parts = [block.get("thinking", "") for block in content_blocks if block.get("type") == "thinking" and block.get("thinking")]
        content = "\n".join(text_parts).strip() or "\n".join(thinking_parts).strip()

        usage_raw = response_data.get("usage") or {}
        usage = {
            "prompt_tokens": int(usage_raw.get("input_tokens", 0) or 0),
            "completion_tokens": int(usage_raw.get("output_tokens", 0) or 0),
            "total_tokens": int((usage_raw.get("input_tokens", 0) or 0) + (usage_raw.get("output_tokens", 0) or 0)),
        }

        return ChatResponse(
            content=content,
            model=response_data.get("model", model),
            usage=usage,
            raw_response=response_data,
        )

    def chat_stream(
        self,
        messages: List[Message],
        trace: Optional[Any] = None,
        **kwargs: Any,
    ) -> Generator[StreamChunk, None, None]:
        """
        Generate a streaming chat completion response.
        
        Args:
            messages: List of conversation messages (role + content).
            trace: Optional TraceContext for observability.
            **kwargs: Provider-specific parameters.
        
        Yields:
            StreamChunk objects containing incremental content.
        """
        self.validate_messages(messages)

        temperature = kwargs.get("temperature", self.default_temperature)
        max_tokens = kwargs.get("max_tokens", self.default_max_tokens)
        model = kwargs.get("model", self.model)

        system_chunks: List[str] = []
        anthropic_messages: List[Dict[str, Any]] = []
        for msg in messages:
            if msg.role == "system":
                system_chunks.append(msg.content.strip())
                continue
            anthropic_messages.append(
                {
                    "role": "assistant" if msg.role == "assistant" else "user",
                    "content": [{"type": "text", "text": msg.content}],
                }
            )

        if not anthropic_messages:
            anthropic_messages.append(
                {"role": "user", "content": [{"type": "text", "text": "请继续。"}]}
            )

        payload: Dict[str, Any] = {
            "model": model,
            "messages": anthropic_messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": True,  # 启用流式输出
        }
        if system_chunks:
            payload["system"] = "\n\n".join(system_chunks)

        # 调用流式API
        yield from self._call_api_stream(payload)

    def _call_api(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        url = f"{self.base_url}/messages"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "anthropic-version": self.api_version,
        }

        try:
            with httpx.Client(timeout=self.request_timeout) as client:
                response = client.post(url, json=payload, headers=headers)
            if response.status_code != 200:
                raise MiniMaxLLMError(
                    f"[MiniMax] API error (HTTP {response.status_code}): {response.text[:500]}"
                )
            return response.json()
        except httpx.TimeoutException as exc:
            raise MiniMaxLLMError("[MiniMax] Request timed out") from exc
        except httpx.HTTPError as exc:
            raise MiniMaxLLMError(f"[MiniMax] HTTP error: {exc}") from exc

    def _call_api_stream(self, payload: Dict[str, Any]) -> Generator[StreamChunk, None, None]:
        """Call API with streaming enabled."""
        url = f"{self.base_url}/messages"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "anthropic-version": self.api_version,
        }

        try:
            with httpx.Client(timeout=self.request_timeout) as client:
                with client.stream("POST", url, json=payload, headers=headers) as response:
                    if response.status_code != 200:
                        error_text = response.read().decode('utf-8')
                        raise MiniMaxLLMError(
                            f"[MiniMax] API error (HTTP {response.status_code}): {error_text[:500]}"
                        )
                    
                    # 解析SSE流
                    for line in response.iter_lines():
                        if not line:
                            continue
                        
                        # SSE格式: "data: {...}"
                        if line.startswith("data: "):
                            data_str = line[6:]  # 去掉 "data: " 前缀
                            
                            # 跳过 [DONE] 标记
                            if data_str.strip() == "[DONE]":
                                break
                            
                            try:
                                import json
                                data = json.loads(data_str)
                                
                                # 提取内容
                                if data.get("type") == "content_block_delta":
                                    delta = data.get("delta", {})
                                    if delta.get("type") == "text_delta":
                                        text = delta.get("text", "")
                                        if text:
                                            yield StreamChunk(content=text)
                                
                            except json.JSONDecodeError:
                                # 忽略无法解析的行
                                continue
                                
        except httpx.TimeoutException as exc:
            raise MiniMaxLLMError("[MiniMax] Request timed out") from exc
        except httpx.HTTPError as exc:
            raise MiniMaxLLMError(f"[MiniMax] HTTP error: {exc}") from exc
