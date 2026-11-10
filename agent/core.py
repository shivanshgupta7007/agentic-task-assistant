"""The agent loop: model proposes, tools dispose, repeat until done.

An 'agent' here is not magic: it is a loop. The model receives the task
plus tool declarations, and on each step either (a) asks to call a tool,
which we execute and feed back, or (b) produces a final answer, which
ends the loop. Every step is recorded in a trace so behaviour can be
scored by the eval harness and debugged by a human.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from .tools import ToolRegistry


@dataclass
class ModelResponse:
    """One step from the model: either a tool call or final text."""
    text: str | None = None
    tool_name: str | None = None
    tool_args: dict | None = None


class AgentModel(Protocol):
    """Anything the loop can talk to - the real Gemini client, or a fake
    in tests. One method: given the conversation so far, take one step."""
    def step(self, history: list[dict]) -> ModelResponse: ...


@dataclass
class AgentResult:
    answer: str
    trace: list[dict] = field(default_factory=list)
    steps: int = 0


class Agent:
    def __init__(self, model: AgentModel, tools: ToolRegistry | None = None, max_steps: int = 8):
        self.model = model
        self.tools = tools or ToolRegistry()
        self.max_steps = max_steps

    def run(self, task: str) -> AgentResult:
        history: list[dict] = [{"role": "user", "content": task}]
        trace: list[dict] = []
        for step_no in range(1, self.max_steps + 1):
            resp = self.model.step(history)
            if resp.tool_name:
                result = self.tools.execute(resp.tool_name, resp.tool_args or {})
                trace.append({
                    "step": step_no,
                    "tool": resp.tool_name,
                    "args": resp.tool_args or {},
                    "result": result,
                })
                history.append({"role": "model", "tool_call": resp.tool_name, "args": resp.tool_args or {}})
                history.append({"role": "tool", "name": resp.tool_name, "content": result})
                continue
            final = resp.text or ""
            trace.append({"step": step_no, "final": final})
            return AgentResult(answer=final, trace=trace, steps=step_no)
        return AgentResult(
            answer="Stopped: reached the step limit before finishing.",
            trace=trace,
            steps=self.max_steps,
        )
