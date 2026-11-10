# Agentic Task Assistant + Evaluation Harness

A small but honest AI agent: it takes a task in plain English, decides which **tools** to call (calculator, clock, word counter, notes), executes them step by step, and composes the final answer. Paired with an **evaluation harness** that scores the agent's tool selection and answer quality across a suite of task cases - because an agent you can't measure is a demo, not a system.

## What it demonstrates

- **Function calling** with the Gemini API: the model proposes tool calls, the runtime validates and executes them, results feed back into the loop.
- **An auditable agent loop**: every step lands in a trace (`AgentResult.trace`), with a hard step limit so a confused model can't spin forever.
- **Safe tool design**: the calculator parses arithmetic with Python's `ast` and evaluates only numeric operations - no `eval()`, no code execution.
- **Model-agnostic core**: the loop talks to a `step(history)` protocol, so tests run against a scripted fake model with no API key and no network.
- **Evaluation as a first-class feature**: `evals/run_evals.py` scores tool-selection accuracy and answer content deterministically, then writes `evals/report.md`.

## Stack

Python · Gemini API (function calling) · pytest

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # add your GEMINI_API_KEY
```

## Run the agent

```bash
python - <<'PY'
from agent.core import Agent
from agent.llm import GeminiAgentModel
from agent.tools import ToolRegistry

registry = ToolRegistry()
agent = Agent(GeminiAgentModel(registry), registry)
result = agent.run("Count the words in 'hello wide world of agents' and multiply by 12")
print(result.answer)
for step in result.trace:
    print(step)
PY
```

## Run the evals

```bash
python evals/run_evals.py   # writes evals/report.md
```

## Tests (no API key needed)

```bash
pytest
```
