import pandas as pd

from train_model import combine_training_data, fetch_real_rows, generate_synthetic_rows


class FakeResponse:
    data = [
        {
            "created_at": "2026-09-06T10:00:00+00:00",
            "location_name": "Dubai_Marina",
            "current_speed": 40,
            "free_flow_speed": 90,
            "speed_ratio": 0.444,
            "weather_main": "Clear",
            "rain_mm": 0,
            "raw_data": {"weather": {"main": {"temp": 35}}},
        }
    ]


class ReadOnlySupabaseClient:
    def __init__(self):
        self.operations = []

    def table(self, name):
        self.operations.append(("table", name))
        return self

    def select(self, columns):
        self.operations.append(("select", columns))
        return self

    def order(self, column):
        self.operations.append(("order", column))
        return self

    def execute(self):
        self.operations.append(("execute",))
        return FakeResponse()

    def insert(self, *_args, **_kwargs):
        raise AssertionError("Training must never insert into Supabase")

    def upsert(self, *_args, **_kwargs):
        raise AssertionError("Training must never upsert into Supabase")


def test_synthetic_rows_are_explicitly_local():
    rows = generate_synthetic_rows(3)
    assert set(rows["source"]) == {"synthetic"}


def test_real_fetch_is_read_only_and_marks_rows_real():
    client = ReadOnlySupabaseClient()
    rows = fetch_real_rows(client)
    assert set(rows["source"]) == {"real"}
    assert [operation[0] for operation in client.operations] == [
        "table", "select", "order", "execute"
    ]


def test_combining_training_data_does_not_change_source_frames():
    real = pd.DataFrame({"source": ["real"], "value": [1]})
    synthetic = pd.DataFrame({"source": ["synthetic"], "value": [2]})
    combined = combine_training_data(real, synthetic)
    assert real.to_dict("records") == [{"source": "real", "value": 1}]
    assert synthetic.to_dict("records") == [{"source": "synthetic", "value": 2}]
    assert set(combined["source"]) == {"real", "synthetic"}


def test_combining_training_data_rejects_unknown_sources():
    frame = pd.DataFrame({"source": ["supabase_write"]})
    try:
        combine_training_data(frame)
    except ValueError as error:
        assert "Unexpected training data source" in str(error)
    else:
        raise AssertionError("Unknown data sources must be rejected")