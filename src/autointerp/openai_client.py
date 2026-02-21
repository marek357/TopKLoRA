from __future__ import annotations

import asyncio
import json
import os
import time
from pathlib import Path
from typing import Union
import logging

from openai import AsyncOpenAI
from delphi.clients.client import Client, Response


class OpenAIClient(Client):
    provider = "openai"

    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        base_url: str | None = None,
        max_tokens: int = 3000,
        timeout: float = 60.0,
        tokenizer=None,
        cost_monitor_enabled: bool = False,
        cost_monitor_every_n_requests: int = 10,
        input_cost_per_1m: float | None = None,
        output_cost_per_1m: float | None = None,
        log_dir: str | None = None,
    ):
        super().__init__(model)
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("openai").setLevel(logging.WARNING)
        self.client = AsyncOpenAI(api_key=api_key, base_url=base_url, timeout=timeout)
        self.max_tokens = max_tokens
        self.tokenizer = tokenizer
        self.monitor_enabled = cost_monitor_enabled
        self.monitor_every_n_requests = cost_monitor_every_n_requests
        self.input_cost_per_1m = input_cost_per_1m
        self.output_cost_per_1m = output_cost_per_1m
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_cost = 0.0
        self.request_count = 0
        self._stats_lock = asyncio.Lock()

        # Request/response logging
        if log_dir is None:
            log_dir = os.path.join("autointerp", "openai_logs")
        self._log_dir = Path(log_dir)
        self._log_dir.mkdir(parents=True, exist_ok=True)
        self._log_path = self._log_dir / f"requests_{int(time.time())}.jsonl"
        self._log_lock = asyncio.Lock()
        logging.info(f"OpenAI request/response log: {self._log_path}")

    async def generate(
        self, prompt: Union[str, list[dict[str, str]]], **kwargs
    ) -> Response:  # type: ignore
        kwargs.pop("schema", None)
        max_tokens = kwargs.pop("max_tokens", self.max_tokens)
        prompt_logprobs = kwargs.pop("prompt_logprobs", None)
        top_logprobs = kwargs.pop("top_logprobs", None)
        logprobs = kwargs.pop("logprobs", None)

        if isinstance(prompt, str):
            messages = [{"role": "user", "content": prompt}]
        else:
            messages = prompt

        logprobs_request = bool(logprobs or top_logprobs)

        response = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_completion_tokens=max_tokens,
            logprobs=logprobs_request if logprobs_request else None,
            top_logprobs=top_logprobs if logprobs_request else None,
        )

        if prompt_logprobs is not None:
            logging.warning(
                "prompt_logprobs requested but OpenAI chat completions API does not support prompt logprobs"
            )

        usage = getattr(response, "usage", None)
        if usage is not None:
            prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
            completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)

            async with self._stats_lock:
                self.total_prompt_tokens += prompt_tokens
                self.total_completion_tokens += completion_tokens
                self.request_count += 1

                if self.input_cost_per_1m is not None:
                    self.total_cost += (
                        prompt_tokens / 1_000_000.0
                    ) * self.input_cost_per_1m
                if self.output_cost_per_1m is not None:
                    self.total_cost += (
                        completion_tokens / 1_000_000.0
                    ) * self.output_cost_per_1m

                if self.monitor_enabled and self.monitor_every_n_requests > 0:
                    if self.request_count % self.monitor_every_n_requests == 0:
                        total_tokens = (
                            self.total_prompt_tokens + self.total_completion_tokens
                        )
                        cost_str = (
                            f", est. cost=${self.total_cost:.6f}"
                            if self.input_cost_per_1m is not None
                            or self.output_cost_per_1m is not None
                            else ""
                        )
                        logging.info(
                            "OpenAI usage: %d requests, %d tokens (prompt=%d, completion=%d)%s",
                            self.request_count,
                            total_tokens,
                            self.total_prompt_tokens,
                            self.total_completion_tokens,
                            cost_str,
                        )

        message = response.choices[0].message.content or ""

        # Log full request and response to JSONL
        log_entry = {
            "timestamp": time.time(),
            "model": self.model,
            "request": {
                "messages": messages,
                "max_tokens": max_tokens,
            },
            "response": {
                "content": message,
            },
            "usage": {
                "prompt_tokens": int(getattr(usage, "prompt_tokens", 0) or 0)
                if usage
                else None,
                "completion_tokens": int(getattr(usage, "completion_tokens", 0) or 0)
                if usage
                else None,
            },
        }
        async with self._log_lock:
            with open(self._log_path, "a") as f:
                f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

        return Response(text=message, logprobs=None, prompt_logprobs=None)
