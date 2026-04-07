"""
LLM客户端模块 - 支持 OpenAI-compatible API (含 Anthropic 代理) 与 ZhiPu API
提供统一的调用接口、重试、缓存与日志记录。
"""
import json
import hashlib
import os
import time
from pathlib import Path
from typing import Optional, Dict, Any, List

from loguru import logger

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

try:
    from zhipuai import ZhipuAI
except ImportError:
    ZhipuAI = None

try:
    import requests
except ImportError:
    requests = None


class LLMClient:
    """统一的LLM调用客户端"""
    _global_last_request_at = 0.0

    def __init__(self, config: Dict[str, Any]):
        """
        初始化LLM客户端

        Args:
            config: llm配置字典，包含 provider, base_url, api_key, model 等
        """
        self.provider = config.get("provider", "openai_compatible")
        self.model = config.get("model", "claude-sonnet-4-20250514")
        self.temperature = config.get("temperature", 0.1)
        self.max_tokens = config.get("max_tokens", 4096)
        self.timeout = config.get("timeout", 600)
        self.retry_count = config.get("retry_count", 3)
        self.retry_delay = config.get("retry_delay", 2)
        self.min_interval_seconds = config.get("min_interval_seconds", 0)
        self.base_url = config.get("base_url", "")
        self.api_key = config.get("api_key", "")

        # 缓存目录
        self.cache_dir = Path(config.get("cache_dir", "artifacts/v20260225/llm_cache"))
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # 日志目录
        self.log_dir = self.cache_dir.parent / "llm_logs"
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # 初始化客户端
        if self.provider == "openai_compatible":
            if OpenAI is None:
                raise ImportError("需要安装 openai 包: pip install openai")
            self.client = OpenAI(
                base_url=self.base_url or "https://api.mmw.ink/v1",
                api_key=self.api_key,
                timeout=self.timeout,
            )
        elif self.provider == "anthropic":
            if requests is None:
                raise ImportError("需要安装 requests 包: pip install requests")
            if not self.base_url:
                raise ValueError("anthropic provider 需要 base_url")
            if not self.api_key:
                raise ValueError("anthropic provider 需要 api_key")
            self.client = None
        elif self.provider == "zhipuai":
            if ZhipuAI is None:
                raise ImportError("需要安装 zhipuai 包: pip install zhipuai")
            self.client = ZhipuAI(api_key=self.api_key)
        else:
            raise ValueError(f"不支持的 provider: {self.provider}")

        # 统计
        self.total_calls = 0
        self.cache_hits = 0
        self.total_tokens = 0

        logger.info(f"LLM客户端初始化完成: provider={self.provider}, model={self.model}")

    def _normalize_messages_for_anthropic(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """将 OpenAI 风格消息转换为 Anthropic Messages API 格式。"""
        system_parts: List[str] = []
        anthropic_messages: List[Dict[str, Any]] = []

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if isinstance(content, list):
                content = "\n".join(
                    part.get("text", "") if isinstance(part, dict) else str(part)
                    for part in content
                )
            else:
                content = str(content)

            if role == "system":
                if content.strip():
                    system_parts.append(content.strip())
                continue

            anthropic_messages.append({
                "role": "assistant" if role == "assistant" else "user",
                "content": content,
            })

        if not anthropic_messages:
            anthropic_messages.append({"role": "user", "content": ""})

        return {
            "system": "\n\n".join(system_parts).strip(),
            "messages": anthropic_messages,
        }

    def _cache_key(self, messages: List[Dict], **kwargs) -> str:
        """生成缓存键"""
        content = json.dumps(messages, ensure_ascii=False, sort_keys=True)
        content += json.dumps(kwargs, sort_keys=True)
        return hashlib.md5(content.encode()).hexdigest()

    def _load_cache(self, key: str) -> Optional[str]:
        """从缓存加载结果"""
        cache_file = self.cache_dir / f"{key}.json"
        if cache_file.exists():
            with open(cache_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data.get("response")
        return None

    def _save_cache(self, key: str, messages: List[Dict], response: str, tokens: int = 0):
        """保存结果到缓存"""
        cache_file = self.cache_dir / f"{key}.json"
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump({
                "messages": messages,
                "response": response,
                "tokens": tokens,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "model": self.model,
            }, f, ensure_ascii=False, indent=2)

    def _log_call(self, messages: List[Dict], response: str, tokens: int, duration: float):
        """记录LLM调用日志"""
        log_file = self.log_dir / f"calls_{time.strftime('%Y%m%d')}.jsonl"
        log_entry = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "model": self.model,
            "input_preview": str(messages[-1].get("content", ""))[:200],
            "output_preview": response[:200],
            "tokens": tokens,
            "duration_s": round(duration, 2),
        }
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

    def chat(
        self,
        messages,
        system: Optional[str] = None,
        use_cache: bool = True,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """
        发送聊天请求

        支持两种调用方式:
            1. chat([{"role":"system","content":"..."},{"role":"user","content":"..."}])
            2. chat("用户prompt字符串", system="系统提示")

        Args:
            messages: 消息列表或用户prompt字符串
            system: 系统提示(仅当messages为字符串时生效)
            use_cache: 是否使用缓存
            temperature: 覆盖默认温度
            max_tokens: 覆盖默认最大token数

        Returns:
            LLM响应文本
        """
        # 兼容字符串调用: chat("prompt", system="...")
        if isinstance(messages, str):
            prompt_text = messages
            messages = []
            if system:
                messages.append({"role": "system", "content": system})
            messages.append({"role": "user", "content": prompt_text})
        self.total_calls += 1
        temp = temperature if temperature is not None else self.temperature
        max_tok = max_tokens if max_tokens is not None else self.max_tokens

        # 检查缓存
        cache_key = self._cache_key(messages, temperature=temp)
        if use_cache:
            cached = self._load_cache(cache_key)
            if cached is not None:
                self.cache_hits += 1
                logger.debug(f"缓存命中 (总调用: {self.total_calls}, 命中: {self.cache_hits})")
                return cached

        # 重试调用
        last_error = None
        for attempt in range(self.retry_count):
            try:
                if self.provider == "anthropic" and self.min_interval_seconds > 0:
                    elapsed = time.time() - LLMClient._global_last_request_at
                    if elapsed < self.min_interval_seconds:
                        wait_gap = self.min_interval_seconds - elapsed
                        logger.debug(f"Anthropic 节流等待: {wait_gap:.1f}s")
                        time.sleep(wait_gap)

                start_time = time.time()
                if self.provider == "anthropic":
                    LLMClient._global_last_request_at = start_time
                    normalized = self._normalize_messages_for_anthropic(messages)
                    payload = {
                        "model": self.model,
                        "max_tokens": max_tok,
                        "temperature": temp,
                        "messages": normalized["messages"],
                    }
                    if normalized["system"]:
                        payload["system"] = normalized["system"]

                    response = requests.post(
                        self.base_url.rstrip("/") + "/v1/messages",
                        headers={
                            "x-api-key": self.api_key,
                            "anthropic-version": "2023-06-01",
                            "content-type": "application/json",
                        },
                        json=payload,
                        timeout=self.timeout,
                    )
                    response.raise_for_status()
                    response_json = response.json()
                    duration = time.time() - start_time
                    content = "\n".join(
                        block.get("text", "")
                        for block in response_json.get("content", [])
                        if block.get("type") == "text"
                    ).strip()
                    usage = response_json.get("usage", {})
                    tokens = (
                        usage.get("input_tokens", 0)
                        + usage.get("output_tokens", 0)
                        + usage.get("cache_creation_input_tokens", 0)
                        + usage.get("cache_read_input_tokens", 0)
                    )
                else:
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        temperature=temp,
                        max_tokens=max_tok,
                    )
                    duration = time.time() - start_time

                    content = response.choices[0].message.content.strip()
                    tokens = getattr(response.usage, "total_tokens", 0) if response.usage else 0
                self.total_tokens += tokens

                # 保存缓存和日志
                if use_cache:
                    self._save_cache(cache_key, messages, content, tokens)
                self._log_call(messages, content, tokens, duration)

                logger.debug(f"LLM调用成功: {tokens} tokens, {duration:.1f}s")
                return content

            except Exception as e:
                last_error = e
                logger.warning(f"LLM调用失败 (尝试 {attempt + 1}/{self.retry_count}): {e}")
                if attempt < self.retry_count - 1:
                    wait_seconds = self.retry_delay * (attempt + 1)
                    error_text = str(e)
                    if "429" in error_text or "Too Many Requests" in error_text:
                        wait_seconds = max(wait_seconds, 15 * (attempt + 1))
                    time.sleep(wait_seconds)

        raise RuntimeError(f"LLM调用失败，已重试{self.retry_count}次: {last_error}")

    def extract_json(
        self,
        messages_or_text,
        use_cache: bool = True,
    ) -> Optional[Dict]:
        """
        解析JSON响应

        支持两种调用方式:
            1. extract_json(response_text_str)  — 直接解析已有的LLM响应文本
            2. extract_json(messages_list)       — 先调用chat()再解析

        Args:
            messages_or_text: LLM响应文本字符串，或消息列表
            use_cache: 是否使用缓存(仅消息列表模式)

        Returns:
            解析后的JSON字典，失败返回None
        """
        if isinstance(messages_or_text, str):
            response = messages_or_text
        else:
            response = self.chat(messages_or_text, use_cache=use_cache)

        # 清理可能的markdown代码块
        cleaned = response.strip()
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            # 去掉首尾的 ```
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            cleaned = "\n".join(lines)

        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            # 尝试找到JSON部分
            import re
            json_match = re.search(r'\{[\s\S]*\}', cleaned)
            if json_match:
                try:
                    return json.loads(json_match.group())
                except json.JSONDecodeError:
                    pass
            logger.warning(f"JSON解析失败: {cleaned[:200]}")
            return None

    def get_stats(self) -> Dict[str, Any]:
        """获取调用统计"""
        return {
            "total_calls": self.total_calls,
            "cache_hits": self.cache_hits,
            "cache_hit_rate": f"{self.cache_hits / max(1, self.total_calls) * 100:.1f}%",
            "total_tokens": self.total_tokens,
        }
