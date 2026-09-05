from masar.nodes.context import context_node


class FakeResponse:
    content = "Rain and an incident likely explain the slowdown."


class FakeLLM:
    def __init__(self):
        self.prompts = []

    def invoke(self, prompt):
        self.prompts.append(prompt)
        return FakeResponse()


def test_context_uses_available_signals_without_network(make_state):
    llm = FakeLLM()
    result = context_node(
        make_state(incidents=["Minor collision"], weather_condition="Rain", is_raining=True),
        llm=llm,
    )
    assert result["context_notes"] == FakeResponse.content
    assert result["nearby_events"] == []
    assert "Minor collision" in llm.prompts[0]
    assert "Rain" in llm.prompts[0]