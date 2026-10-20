"""OpenAI-compatible /v1/chat/completions with streamed tool calls (vLLM, SGLang, llama.cpp, OpenAI, ...)."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import ClassVar

from ..meter import Meter, Turn
from ..task import Setup, Task
from .base import Harness, Transcript

REASONING_KEYS = ("reasoning_content", "reasoning")  # SGLang / DeepSeek, vLLM


class OpenAIChatHarness(Harness):
    name = "openai-chat"
    options: ClassVar[dict[str, str]] = {
        "base_url": "endpoint up to /v1 (default $OPENAI_BASE_URL or http://localhost:8000/v1)",
        "model": "model name (default: the first one /v1/models lists)",
        "api_key_env": "environment variable holding the API key (default OPENAI_API_KEY; unset = no key)",
        "temperature": "sampling temperature (default 0)",
        "max_tokens": "output cap per request (default 8192)",
        "extra_body": 'JSON merged into every request, e.g. \'{"chat_template_kwargs": {"thinking": false}}\'',
        "timeout": "seconds per request (default 600)",
    }

    def __init__(self, **options: str):
        super().__init__(**options)
        o = self.opts
        self.base = (o.get("base_url") or os.environ.get("OPENAI_BASE_URL") or "http://localhost:8000/v1").rstrip("/")
        self.key = os.environ.get(o.get("api_key_env") or "OPENAI_API_KEY")
        self.temperature = float(o.get("temperature", 0))
        self.max_tokens = int(o.get("max_tokens", 8192))
        self.extra = json.loads(o.get("extra_body", "{}"))
        self.timeout = float(o.get("timeout", 600))
        self.model = o.get("model") or self._first_model()

    def _request(self, path: str, body: dict | None = None):
        headers = {"Content-Type": "application/json"}
        if self.key:
            headers["Authorization"] = f"Bearer {self.key}"
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(self.base + path, data=data, headers=headers)
        return urllib.request.urlopen(req, timeout=self.timeout)

    def _first_model(self) -> str:
        with self._request("/models") as r:
            return json.load(r)["data"][0]["id"]

    def run(self, task: Task, setup: Setup, meter: Meter, *, max_turns: int) -> Transcript:
        tools = setup.tools
        messages = [{"role": "system", "content": setup.system}, {"role": "user", "content": setup.prompt}]
        tr = Transcript(messages=messages)
        for _ in range(max_turns):
            try:
                msg = self._turn(messages, tools.schemas(), meter)
            except (urllib.error.URLError, TimeoutError, ValueError) as e:
                tr.stop = f"error: {e}"
                return tr
            messages.append(msg)
            calls = msg.get("tool_calls") or []
            if not calls:
                tr.stop = "no_tool_calls"
                return tr
            for c in calls:
                result = tools.call(c["function"]["name"], c["function"]["arguments"] or "{}")
                messages.append({"role": "tool", "tool_call_id": c["id"], "content": result})
            if tools.finished:
                tr.stop = "finish"
                return tr
        tr.stop = "max_turns"
        return tr

    def _turn(self, messages: list[dict], schemas: list[dict], meter: Meter) -> dict:
        body = {
            "model": self.model,
            "messages": messages,
            "tools": schemas,
            "tool_choice": "auto",
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": True,
            "stream_options": {"include_usage": True},
            **self.extra,
        }
        turn = meter.begin()
        content, calls, usage = [], {}, None
        with self._request("/chat/completions", body) as r:
            for raw in r:
                line = raw.decode().strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                chunk = json.loads(data)
                usage = chunk.get("usage") or usage
                for choice in chunk.get("choices") or []:
                    self._delta(choice.get("delta") or {}, turn, meter, content, calls)
        for c in calls.values():
            meter.tool_call(turn, c["function"]["name"], c["function"]["arguments"])
        meter.content(turn, "".join(content))
        meter.end(turn, usage)
        msg = {"role": "assistant", "content": "".join(content) or None}
        if calls:
            msg["tool_calls"] = [calls[i] for i in sorted(calls)]
        return msg

    @staticmethod
    def _delta(delta: dict, turn: Turn, meter: Meter, content: list[str], calls: dict) -> None:
        text = delta.get("content")
        if text:
            meter.first_token(turn)
            content.append(text)
        for key in REASONING_KEYS:
            if delta.get(key):
                meter.first_token(turn)
                meter.text(turn, "reasoning", delta[key])
        for tc in delta.get("tool_calls") or []:
            meter.first_token(turn)
            c = calls.setdefault(
                tc.get("index", 0), {"id": "", "type": "function", "function": {"name": "", "arguments": ""}}
            )
            c["id"] = tc.get("id") or c["id"]
            fn = tc.get("function") or {}
            c["function"]["name"] += fn.get("name") or ""
            c["function"]["arguments"] += fn.get("arguments") or ""
