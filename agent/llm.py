"""Gemini function-calling adapter for the agent loop.

Translates between the loop's ModelResponse and Gemini's function-calling
API. Kept separate from core.py so the loop itself is model-agnostic (and
unit-testable with a fake model, no API key needed).
"""
from __future__ import annotations

import os

from .core import ModelResponse
from .tools import ToolRegistry


class GeminiAgentModel:
    def __init__(self, tools: ToolRegistry, api_key: str | None = None, model: str = "gemini-2.0-flash"):
        import google.generativeai as genai  # lazy import

        key = api_key or os.environ.get("GEMINI_API_KEY")
        if not key:
            raise RuntimeError("GEMINI_API_KEY is not set. Copy .env.example to .env and add your key.")
        genai.configure(api_key=key)
        self._model = genai.GenerativeModel(
            model,
            tools=[{"function_declarations": tools.declarations()}],
        )
        self.model_name = model

    def step(self, history: list[dict]) -> ModelResponse:
        # Flatten the loop's history into a simple transcript; Gemini's
        # function-calling responses carry either text or a function_call part.
        contents = self._to_contents(history)
        resp = self._model.generate_content(contents)
        part = resp.candidates[0].content.parts[0]
        if getattr(part, "function_call", None) and part.function_call.name:
            return ModelResponse(
                tool_name=part.function_call.name,
                tool_args=dict(part.function_call.args),
            )
        return ModelResponse(text=getattr(part, "text", "") or "")

    @staticmethod
    def _to_contents(history: list[dict]) -> list[dict]:
        contents = []
        for msg in history:
            if msg["role"] == "user":
                contents.append({"role": "user", "parts": [msg["content"]]})
            elif msg["role"] == "model":
                contents.append({"role": "model", "parts": [
                    {"function_call": {"name": msg["tool_call"], "args": msg["args"]}}
                ]})
            elif msg["role"] == "tool":
                contents.append({"role": "function", "parts": [
                    {"function_response": {"name": msg["name"], "response": {"result": msg["content"]}}}
                ]})
        return contents
