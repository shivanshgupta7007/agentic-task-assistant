# Agentic Task Assistant - How to Explain It (Study Guide)

## The 30-second version

"It's an AI agent with an evaluation harness. The agent is a loop: the model gets a task and a list of tools - calculator, clock, word counter, notes - and on each step it either calls a tool or gives the final answer. Tool execution happens on my side, never in the model, so every action is validated, logged in a trace, and capped by a step limit. The eval harness then runs a suite of task cases and scores two things deterministically: did it pick the right tools, and did the final answer contain the right content. That's the difference between an agent demo and an agent you can actually trust."

## End-to-end flow

1. User gives a task ("count the words in X and multiply by 12").
2. The model receives the task plus JSON-schema declarations of the five tools.
3. It replies with a **function call**, e.g. `word_counter(text=...)` - just a request, no execution.
4. `ToolRegistry.execute()` validates the name and arguments and runs the actual Python function.
5. The result goes back into the conversation as a tool message.
6. The model sees the result and either calls another tool or answers.
7. Every step is appended to `trace`; the loop stops at a final answer or the 8-step limit.

## Key code to know cold

- `agent/core.py` -> `Agent.run()`: the whole idea in one loop. Expect to be asked to write this on a whiteboard.
- `agent/tools.py` -> `calculator()` + `_eval_node()`: why `ast` instead of `eval()` (security - `eval("...")` would execute anything the model emits).
- `evals/run_evals.py` -> `score_case()`: the two deterministic signals.

## Why these choices

- **Why does the model not execute code itself?** Control and auditability. The model proposes, the runtime disposes. We can validate arguments, log every call, and sandbox execution.
- **Why a step limit?** Agents can loop (call a tool, misunderstand the result, call it again). A hard cap turns an infinite loop into a clean failure message.
- **Why a trace?** Debugging and evaluation. When an agent misbehaves you replay the trace instead of guessing.
- **Why deterministic evals first?** They're cheap, reproducible, and can't be sweet-talked. An LLM judge is a useful second layer for open-ended quality, but the foundation is "did it call the calculator and did the answer contain 60."
- **Why inject the model as a protocol?** Testability. The fake model in the tests replays scripted steps, so the loop's logic is verified with zero API calls and zero flakiness.

## Likely interview questions

**Q: How is this different from just prompting an LLM?**
A prompt returns text. An agent acts: it calls tools, observes results, and adapts over multiple steps. The loop + tools + trace are what make it agentic.

**Q: How do you keep an agent safe?**
Tool allowlisting (the model can only ask for declared tools), argument validation, no `eval`, step limits, and full trace logging. For higher-stakes tools you'd add human-in-the-loop approval before execution.

**Q: How do you evaluate an agent?**
Three levels: deterministic checks (tool selection, required answer content - what I built), rubric-based LLM judging for open-ended answers, and trajectory analysis (was the path efficient, not just the destination). Start deterministic; add judges where determinism can't reach.

**Q: What's function calling?**
The API lets you declare tools as JSON schemas. The model's response can then be a structured `function_call` (name + arguments) instead of free text. The application executes it and returns the result as a `function_response` message.

**Q: Where does this break in production?**
Ambiguous tasks (the model picks the wrong tool confidently), tool errors it can't recover from, compounding mistakes across steps, and cost/latency of multi-step loops. The mitigations: better tool descriptions, validation retries, step limits, and caching.

**Q: How would you extend this?**
Add retrieval (my DocChat project is exactly that - RAG as a tool), add planning for long tasks, add memory beyond the notes tool, and parallel tool calls for independent steps.
