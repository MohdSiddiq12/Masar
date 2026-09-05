from masar.nodes.synthesis import SynthesisOutput, synthesis_node


class FakeLLM:
    def invoke(self, prompt):
        assert "Marina -> Business Bay" in prompt
        return SynthesisOutput(message_en="Take the metro.", message_ar="استقل المترو.")


def test_synthesis_returns_bilingual_messages(make_state):
    result = synthesis_node(
        make_state(
            recommended_mode="metro",
            recommended_route=["Marina", "Business Bay"],
            context_notes="Heavy congestion detected.",
        ),
        llm=FakeLLM(),
    )
    assert result == {"message_en": "Take the metro.", "message_ar": "استقل المترو."}