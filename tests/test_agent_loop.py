"""The loop is tested with a scripted fake model: no API key, no network,
full control over the step sequence."""
from agent.core import Agent, ModelResponse


class FakeModel:
    def __init__(self, script):
        self.script = list(script)

    def step(self, history):
        return self.script.pop(0)


def test_agent_calls_tool_then_answers():
    model = FakeModel([
        ModelResponse(tool_name="calculator", tool_args={"expression": "2 * (3 + 4)"}),
        ModelResponse(text="The total is 14."),
    ])
    result = Agent(model).run("What is 2 * (3 + 4)?")
    assert result.answer == "The total is 14."
    assert result.trace[0]["tool"] == "calculator"
    assert result.trace[0]["result"] == "14"
    assert result.steps == 2


def test_tool_result_is_fed_back_into_history():
    seen_histories = []

    class RecordingModel:
        def step(self, history):
            seen_histories.append([m["role"] for m in history])
            if len(seen_histories) == 1:
                return ModelResponse(tool_name="word_counter", tool_args={"text": "one two three"})
            return ModelResponse(text="3 words.")

    result = Agent(RecordingModel()).run("count words")
    assert seen_histories[1] == ["user", "model", "tool"]
    assert result.answer == "3 words."


def test_step_limit_stops_runaway_loop():
    class EndlessModel:
        def step(self, history):
            return ModelResponse(tool_name="get_current_time", tool_args={})

    result = Agent(EndlessModel(), max_steps=3).run("loop forever")
    assert "step limit" in result.answer
    assert result.steps == 3
