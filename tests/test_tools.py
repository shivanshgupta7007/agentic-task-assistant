import pytest

from agent.tools import ToolRegistry, calculator, read_note, save_note, word_counter


def test_calculator_basic_arithmetic():
    assert calculator("2 * (3 + 4)") == "14"
    assert calculator("10 / 4") == "2.5"


def test_calculator_rejects_non_arithmetic():
    assert calculator("__import__('os').system('ls')").startswith("Error")
    assert calculator("open('file')").startswith("Error")


def test_word_counter():
    assert word_counter("the quick brown fox") == "4"


def test_notes_roundtrip():
    save_note("editor", "VS Code")
    assert read_note("editor") == "VS Code"
    assert "No note" in read_note("missing")


def test_registry_executes_by_name():
    registry = ToolRegistry()
    assert registry.execute("calculator", {"expression": "1 + 1"}) == "2"
    assert "unknown tool" in registry.execute("nope", {})


def test_registry_declarations_match_functions():
    registry = ToolRegistry()
    names = {d["name"] for d in registry.declarations()}
    assert {"calculator", "get_current_time", "word_counter", "save_note", "read_note"} <= names
