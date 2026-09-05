import masar.nodes.predictor as predictor_module

from masar.graph import build_graph
from masar.nodes.synthesis import SynthesisOutput


class FakeContextLLM:
    def invoke(self, prompt):
        return type("Response", (), {"content": "The route needs extra context."})()


class FakeSynthesisLLM:
    def invoke(self, prompt):
        return SynthesisOutput(message_en="Use the recommended route.", message_ar="استخدم المسار الموصى به.")


def test_graph_runs_deep_path_end_to_end(make_state, monkeypatch):
    monkeypatch.setattr(predictor_module, "MODEL_PATH", "missing-model.pkl")
    monkeypatch.setattr(predictor_module, "_model", None)
    monkeypatch.setattr(predictor_module, "_model_checked", False)
    result = build_graph(FakeContextLLM(), FakeSynthesisLLM()).invoke(make_state())
    assert result["route_path"] == "deep"
    assert result["context_notes"] == "The route needs extra context."
    assert result["message_en"] == "Use the recommended route."


def test_graph_runs_fast_path_without_context(make_state, monkeypatch):
    class ConfidentModel:
        def predict_proba(self, features):
            return [[0.05, 0.95]]

    monkeypatch.setattr(predictor_module, "_model", ConfidentModel())
    monkeypatch.setattr(predictor_module, "_model_checked", True)
    result = build_graph(None, FakeSynthesisLLM()).invoke(make_state())
    assert result["route_path"] == "fast"
    assert result["context_notes"] is None
    assert result["message_ar"] == "استخدم المسار الموصى به."