"""Tools the agent can call.

Each tool is a plain function plus a declaration (name, description,
JSON-schema parameters) that is handed to the model. The model never runs
code - it *asks* for a tool call, the loop validates and executes it, and
the result goes back into the conversation. Keeping execution on our side
is what makes tool use auditable and safe.
"""
from __future__ import annotations

import ast
import operator
from dataclasses import dataclass
from datetime import datetime, timezone


# ---------- calculator: safe arithmetic via AST, never eval() ----------
_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _eval_node(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _eval_node(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval_node(node.left), _eval_node(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval_node(node.operand))
    raise ValueError(f"Unsupported expression element: {ast.dump(node)}")


def calculator(expression: str) -> str:
    """Evaluate a basic arithmetic expression, e.g. '2 * (3 + 4)'."""
    try:
        result = _eval_node(ast.parse(expression.strip(), mode="eval"))
    except Exception as exc:
        return f"Error: {exc}"
    return str(result)


def get_current_time() -> str:
    """Return the current UTC time in ISO 8601."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def word_counter(text: str) -> str:
    """Count the words in a piece of text."""
    return str(len(text.split()))


_NOTES: dict[str, str] = {}


def save_note(key: str, content: str) -> str:
    """Save a short note under a key for later recall."""
    _NOTES[key] = content
    return f"Saved note '{key}'."


def read_note(key: str) -> str:
    """Read back a previously saved note."""
    return _NOTES.get(key, f"No note saved under '{key}'.")


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict
    fn: object  # callable


class ToolRegistry:
    """Name -> Tool map. The declarations go to the model; the callables
    stay with us."""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}
        for tool in _default_tools():
            self._tools[tool.name] = tool

    def declarations(self) -> list[dict]:
        return [
            {"name": t.name, "description": t.description, "parameters": t.parameters}
            for t in self._tools.values()
        ]

    def execute(self, name: str, args: dict) -> str:
        tool = self._tools.get(name)
        if tool is None:
            return f"Error: unknown tool '{name}'"
        try:
            return str(tool.fn(**args))
        except TypeError as exc:
            return f"Error: bad arguments for '{name}': {exc}"


def _default_tools() -> list[Tool]:
    str_param = lambda desc: {"type": "STRING", "description": desc}  # noqa: E731
    return [
        Tool("calculator", "Evaluate a basic arithmetic expression.",
             {"type": "OBJECT", "properties": {"expression": str_param("e.g. '2 * (3 + 4)'")}, "required": ["expression"]},
             calculator),
        Tool("get_current_time", "Get the current UTC time in ISO 8601.",
             {"type": "OBJECT", "properties": {}}, get_current_time),
        Tool("word_counter", "Count words in a text.",
             {"type": "OBJECT", "properties": {"text": str_param("text to count words in")}, "required": ["text"]},
             word_counter),
        Tool("save_note", "Save a short note under a key.",
             {"type": "OBJECT", "properties": {"key": str_param("note key"), "content": str_param("note content")}, "required": ["key", "content"]},
             save_note),
        Tool("read_note", "Read a previously saved note.",
             {"type": "OBJECT", "properties": {"key": str_param("note key")}, "required": ["key"]},
             read_note),
    ]
