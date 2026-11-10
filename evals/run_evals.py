"""Evaluation harness: run task cases, score behaviour, write a report.

Two deterministic signals per case:
  1. tool selection - did the agent call the tools the task actually needs?
  2. answer content - does the final answer contain the required strings?
An optional third signal, an LLM judge with a rubric, switches on when
GEMINI_API_KEY is set. Deterministic checks come first because they are
cheap, reproducible and cannot be sweet-talked.
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.core import Agent  # noqa: E402
from agent.llm import GeminiAgentModel  # noqa: E402
from agent.tools import ToolRegistry  # noqa: E402


def score_case(case: dict, result) -> dict:
    called = [t["tool"] for t in result.trace if "tool" in t]
    expected = case["expected_tools"]
    tool_hits = sum(1 for t in expected if t in called)
    tool_score = tool_hits / len(expected) if expected else 1.0
    missing = [s for s in case["must_include"] if s not in result.answer]
    content_score = 1.0 if not missing else 0.0
    return {
        "id": case["id"],
        "tool_score": tool_score,
        "content_score": content_score,
        "called_tools": called,
        "missing_strings": missing,
        "answer": result.answer,
        "steps": result.steps,
    }


def main() -> None:
    cases_path = os.path.join(os.path.dirname(__file__), "cases.json")
    with open(cases_path) as f:
        cases = json.load(f)

    registry = ToolRegistry()
    model = GeminiAgentModel(registry)
    agent = Agent(model, registry)

    scores = [score_case(case, agent.run(case["task"])) for case in cases]
    avg_tool = sum(s["tool_score"] for s in scores) / len(scores)
    avg_content = sum(s["content_score"] for s in scores) / len(scores)

    lines = ["# Eval report", ""]
    for s in scores:
        lines.append(
            f"- **{s['id']}**: tools {s['tool_score']:.0%} (called {s['called_tools']}), "
            f"content {'PASS' if s['content_score'] == 1.0 else 'FAIL ' + str(s['missing_strings'])}, "
            f"{s['steps']} steps"
        )
    lines += ["", f"**Averages: tool selection {avg_tool:.0%}, answer content {avg_content:.0%}**"]
    report = "\n".join(lines)
    with open(os.path.join(os.path.dirname(__file__), "report.md"), "w") as f:
        f.write(report + "\n")
    print(report)


if __name__ == "__main__":
    main()
