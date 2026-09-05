from datetime import datetime

import masar.live_traffic as live_traffic


class _FakeResponse:
    def __init__(self, data):
        self.data = data


class _FakeQuery:
    def __init__(self, rows):
        self._rows = rows

    def select(self, *_args, **_kwargs):
        return self

    def eq(self, *_args, **_kwargs):
        return self

    def order(self, *_args, **_kwargs):
        return self

    def limit(self, *_args, **_kwargs):
        return self

    def execute(self):
        return _FakeResponse(self._rows)


class FakeSupabaseClient:
    def __init__(self, rows):
        self._rows = rows

    def table(self, _name):
        return _FakeQuery(self._rows)


def _sample_row(**overrides):
    row = {
        "current_speed": 22.0,
        "free_flow_speed": 90.0,
        "speed_ratio": 0.24,
        "weather_main": "Clear",
        "rain_mm": 0,
        "raw_data": {"weather": {"main": {"temp": 41.2}}},
        "created_at": "2026-09-06T10:00:00+00:00",
    }
    row.update(overrides)
    return row


def test_fetch_latest_row_returns_the_row_for_a_recognised_location():
    client = FakeSupabaseClient([_sample_row()])
    row = live_traffic.fetch_latest_row(client, "Sheikh Zayed Road")
    assert row["current_speed"] == 22.0


def test_fetch_latest_row_returns_none_for_unmapped_location():
    client = FakeSupabaseClient([_sample_row()])
    assert live_traffic.fetch_latest_row(client, "Somewhere Else") is None


def test_fetch_latest_row_returns_none_when_no_rows_exist():
    client = FakeSupabaseClient([])
    assert live_traffic.fetch_latest_row(client, "Marina") is None


def test_row_to_state_fields_maps_all_expected_keys():
    fields = live_traffic.row_to_state_fields(_sample_row())
    assert fields == {
        "current_speed": 22.0,
        "free_flow_speed": 90.0,
        "congestion_ratio": 0.24,
        "weather_condition": "Clear",
        "is_raining": False,
        "temperature": 41.2,
    }


def test_row_to_state_fields_detects_rain_from_rain_mm():
    fields = live_traffic.row_to_state_fields(_sample_row(rain_mm=1.5, weather_main="Clouds"))
    assert fields["is_raining"] is True


def test_row_to_state_fields_detects_rain_from_weather_main_when_rain_mm_missing():
    fields = live_traffic.row_to_state_fields(_sample_row(rain_mm=0, weather_main="Rain"))
    assert fields["is_raining"] is True


def test_row_to_state_fields_omits_temperature_when_raw_data_lacks_it():
    fields = live_traffic.row_to_state_fields(_sample_row(raw_data={}))
    assert "temperature" not in fields


def test_row_age_seconds_computes_elapsed_time():
    row = _sample_row(created_at="2026-09-06T09:59:00+00:00")
    now = datetime.fromisoformat("2026-09-06T10:00:00+00:00")
    assert live_traffic.row_age_seconds(row, now=now) == 60.0


def test_row_age_seconds_returns_none_without_a_timestamp():
    row = _sample_row(created_at=None)
    assert live_traffic.row_age_seconds(row) is None


def test_get_live_fields_returns_empty_when_no_row_is_found():
    client = FakeSupabaseClient([])
    fields, row = live_traffic.get_live_fields(client, "Marina")
    assert fields == {}
    assert row is None


def test_get_live_fields_returns_fields_and_the_raw_row_together():
    client = FakeSupabaseClient([_sample_row()])
    fields, row = live_traffic.get_live_fields(client, "Marina")
    assert fields["current_speed"] == 22.0
    assert row["current_speed"] == 22.0