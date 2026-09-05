# Masar (مسار) — Dubai Road-Congestion & Multi-Modal Commute Agent

A LangGraph multi-agent system that predicts road congestion for key Dubai
corridors and recommends drive / metro / drive-to-metro, with bilingual
(English/Arabic) output. Built as a Dubai-specific reframe of the classic
"predict transit delay" project: Dubai Metro is famously punctual, so the
real local pain point is road congestion and mode choice, not transit
reliability.

## Architecture

```
Predictor  →  Router  →  (fast: straight to Optimizer)
                       →  (deep: Context agent → Optimizer)
                                                    ↓
                                              Synthesis (EN/AR)
```

- **Predictor** — XGBoost classifier if a trained model exists, otherwise
  a heuristic that deliberately reports low confidence so the Router
  correctly sends traffic through Context until a real model is trained.
- **Router** — reads confidence, uses LangGraph's `Command` to route
  dynamically. Missing confidence fails safe toward the deep path.
- **Context** — LLM reasoning over available signal (incidents, weather).
  Social/events lookups are honest stubs pending Reddit (paused) and an
  events API (not yet wired).
- **Route Optimizer** — NetworkX Dijkstra over an interim hand-built graph
  of the 5 monitored corridors. Real RTA network topology was never
  confirmed as integrated — this is a known, documented placeholder.
- **Synthesis** — structured bilingual output via Pydantic + LLM.

Every LLM-calling node accepts an optional `llm` parameter — production
omits it and gets a real Groq client; tests inject a fake and verify
logic with zero network calls.

## Data pipeline

```
TomTom + OpenWeatherMap → scripts/collect_traffic.py → Supabase (traffic_logs)
   (continuous, every 10 min via GitHub Actions)

Supabase → scripts/train_model.py → models/congestion_xgb.pkl
   (periodic — run manually until there's a real retraining cadence)
```

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Environment variables needed: `SUPABASE_URL`, `SUPABASE_KEY`,
`GROQ_API_KEY`, `TOMTOM_API_KEY`, `OPENWEATHER_API_KEY`.
`MASAR_MODEL_PATH` optionally overrides the default `models/congestion_xgb.pkl`.

## Manual Supabase Check

Run this read-only snippet from the repository root after creating `.env`:

```python
import os
from pprint import pprint

from dotenv import load_dotenv
from supabase import create_client

load_dotenv()
client = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_KEY"])

traffic = (
  client.table("traffic_logs")
  .select("location_name,current_speed,free_flow_speed,speed_ratio,weather_main,rain_mm")
  .limit(10)
  .execute()
)
chatter = (
  client.table("reddit_chatter")
  .select("reddit_id,title,score,post_url,published_at")
  .order("published_at", desc=True)
  .limit(10)
  .execute()
)

print("Latest traffic rows:")
pprint(traffic.data)
print("\nLatest Reddit chatter:")
pprint(chatter.data)
```

Save it as `view_data.py`, then run:

```powershell
.\VE\Scripts\python.exe view_data.py
```

This only performs `select` queries; it does not insert or update data.

## Manual Component Check

This snippet exercises the predictor, router, context, route optimizer, and
bilingual synthesis without network calls:

```python
import masar.nodes.predictor as predictor_module
from masar.graph import build_graph
from masar.nodes.synthesis import SynthesisOutput


class ContextLLM:
  def invoke(self, prompt):
    return type("Response", (), {"content": "Traffic context was reviewed."})()


class SynthesisLLM:
  def invoke(self, prompt):
    return SynthesisOutput(
      message_en="Use the recommended route.",
      message_ar="استخدم المسار الموصى به.",
    )


predictor_module.MODEL_PATH = "missing-model.pkl"
predictor_module._model = None
predictor_module._model_checked = False

state = {
  "location": "Sheikh Zayed Road",
  "timestamp": "2026-09-05T18:00:00Z",
  "current_speed": 40.0,
  "free_flow_speed": 100.0,
  "congestion_ratio": 0.4,
  "incidents": ["Minor collision"],
  "temperature": 38.0,
  "weather_condition": "Clear",
  "is_raining": False,
  "predicted_congestion": None,
  "prediction_confidence": None,
  "is_anomaly": False,
  "route_path": "fast",
  "origin": "Marina",
  "destination": "Business Bay",
  "context_notes": None,
  "nearby_events": [],
  "social_signal": None,
  "recommended_mode": None,
  "recommended_route": [],
  "message_en": None,
  "message_ar": None,
}

result = build_graph(ContextLLM(), SynthesisLLM()).invoke(state)
for key in (
  "predicted_congestion",
  "prediction_confidence",
  "route_path",
  "context_notes",
  "recommended_mode",
  "recommended_route",
  "message_en",
  "message_ar",
):
  print(f"{key}: {result.get(key)}")
```

Save it as `manual_components.py`, then run:

```powershell
.\VE\Scripts\python.exe manual_components.py
```

## Run GitHub Actions From the CLI

After installing and authenticating GitHub CLI, dispatch the traffic workflow
on the `testing` branch:

```powershell
gh auth login
gh workflow run collect_traffic.yml --ref testing
gh run list --workflow collect_traffic.yml --branch testing --limit 1
gh run watch RUN_ID --exit-status
```

Replace `RUN_ID` with the run ID printed by `gh run list`. The workflow needs
the repository secrets `TOMTOM_API_KEY`, `OPENWEATHER_API_KEY`,
`SUPABASE_URL`, and `SUPABASE_KEY`.

## Running tests

```bash
python -m pytest -v
```

15 tests across the component and graph test files, all isolated — no
network access is required to run the full suite. Each node is tested
standalone before the graph tests exercise both the fast and deep paths.

## Training the model

```bash
python -m scripts.train_model --synthetic-rows 2000   # synthetic only
python -m scripts.train_model --include-real          # + your Supabase data
python -m scripts.train_model --real-only             # Supabase data only
```

**Known limitation:** as of this build, real Supabase coverage is ~81 rows
(roughly a day and a half) — not enough for the per-location/hour baseline
label to be meaningful yet. `--include-real` and `--real-only` are wired
correctly but haven't been exercised against live data in every
environment. Realistic target before retraining on real data alone:
2–3 weeks of coverage.

## Known gaps (tracked deliberately, not accidentally)

- Reddit social signal: paused (public JSON endpoint blocked) — PRAW +
  OAuth is the known fix
- Route Optimizer topology: hand-built placeholder graph, not real RTA
  network data
- No backend/frontend/deployment yet — this is the agent core only
