"""Gemini function-calling adapter for the agent loop.

Calls the Gemini REST API directly (no SDK) so raw response parts can be
replayed verbatim. Gemini 3.x attaches a thought_signature to function-call
parts and rejects later turns that replay those calls without it, so the
adapter keeps the raw JSON part of every model tool call and sends it back
untouched. Kept separate from core.py so the loop itself stays
model-agnostic (unit-tested with a fake model, no API key needed).
"""
from __future__ import annotations

import json
import os
import urllib.request

from .core import ModelResponse
from .tools import ToolRegistry

_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={key}"


class GeminiAgentModel:
    def __init__(self, tools: ToolRegistry, api_key: str | None = None, model: str = "gemini-3.6-flash"):
        self._key = api_key or os.environ.get("GEMINI_API_KEY")
        if not self._key:
            raise RuntimeError("GEMINI_API_KEY is not set. Copy .env.example to .env and add your key.")
        self._tools = tools
        self.model_name = model
        self._raw_model_parts: list[dict] = []  # raw JSON parts of each model turn, in order

    def step(self, history: list[dict]) -> ModelResponse:
        body = {
            "contents": self._to_contents(history),
            "tools": [{"functionDeclarations": self._tools.declarations()}],
        }
        req = urllib.request.Request(
            _API_URL.format(model=self.model_name, key=self._key),
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                payload = json.loads(resp.read())
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode(errors="replace")[:500]
            raise RuntimeError(f"Gemini API error {exc.code}: {detail}") from exc

        parts = payload["candidates"][0]["content"]["parts"]
        part = parts[0]
        call = part.get("functionCall")
        if call and call.get("name"):
            # Keep the raw part (with thought_signature) so the replay in
            # _to_contents satisfies Gemini 3.x signature validation.
            self._raw_model_parts.append(part)
            return ModelResponse(tool_name=call["name"], tool_args=dict(call.get("args") or {}))
        return ModelResponse(text=part.get("text") or "")

    def _to_contents(self, history: list[dict]) -> list[dict]:
        contents: list[dict] = []
        raw_idx = 0
        for msg in history:
            if msg["role"] == "user":
                contents.append({"role": "user", "parts": [{"text": msg["content"]}]})
            elif msg["role"] == "model":
                # Replay the raw model part verbatim: Gemini 3.x requires the
                # thought_signature that came with the function call.
                contents.append({"role": "model", "parts": [self._raw_model_parts[raw_idx]]})
                raw_idx += 1
            elif msg["role"] == "tool":
                # Gemini 3.x dropped the "function" role: tool results go back
                # as function_response parts under the "user" role.
                contents.append({"role": "user", "parts": [
                    {"functionResponse": {"name": msg["name"], "response": {"result": msg["content"]}}}
                ]})
        return contents
