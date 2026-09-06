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
  (continuous, every hour at minute 0 UTC via GitHub Actions)

Supabase → scripts/train_model.py → models/congestion_xgb.pkl
  (local training command; synthetic data is never written to Supabase)

The local web app reads the trained model and serves recommendations:

```
Browser → web_app.py → Masar graph → predictor → router → optimizer → synthesis
                    ↓
              live traffic lookup (optional)
```
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

The `models/` directory is ignored by Git. Train the model locally before
starting the UI if `models/congestion_xgb.pkl` does not exist.

## Local Web App

Train from synthetic data and start the local recommendation site:

```powershell
.\VE\Scripts\python.exe train_model.py --synthetic-rows 6000
.\VE\Scripts\python.exe web_app.py
```

Open <http://127.0.0.1:8000>. Live traffic conditions are selected by default;
manual speed/weather fields become available only when live traffic is turned
off. The UI displays the selected data source and the age of live traffic.
Use `demo` mode for local fake LLM responses, or enable live Groq reasoning
from the UI when `GROQ_API_KEY` is configured.

The API endpoint is `POST /api/recommend`. To force manual/local values from
PowerShell:

```powershell
$body = @{ demo=$true; use_live_traffic=$false; origin="Marina"; destination="Business Bay"; current_speed=25; free_flow_speed=90; temperature=36; weather_condition="Clear"; is_raining=$false } | ConvertTo-Json
Invoke-RestMethod http://127.0.0.1:8000/api/recommend -Method Post -ContentType "application/json" -Body $body
```

The read-only traffic display is available at `GET /api/traffic`; the latest
row for one location is available at `GET /api/traffic/latest?location=Marina`.

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

Run tests through pytest rather than executing a test file directly. Pytest
adds the repository root to Python's import path, while `python
tests/test_api_report.py` starts with only `tests/` on the path and cannot
resolve the `masar` package.

The current suite contains 33 isolated tests. No network access is required
for the default suite; external calls are replaced with fakes where needed.
Each node is tested standalone before graph tests exercise both fast and deep
paths.

## Component and API Reports

Run isolated checks without calling external services:

```powershell
.\VE\Scripts\python.exe -m scripts.check_components --report-path reports/api_calls.jsonl
```

The command prints PASS/FAIL results for every component and records the same
run as JSONL. Each record includes a UTC timestamp, service, operation,
sanitized request, returned response, duration, status, and error. Groq
prompts/responses, offline LLM calls, and component function checks are
included. API keys, tokens, passwords, and authorization values are redacted;
long values are truncated.

For an explicit Supabase read-only check, add `--include-network`:

```powershell
.\VE\Scripts\python.exe -m scripts.check_components --include-network --report-path reports/api_calls.jsonl
```

Traffic collection, Reddit collection, training reads, live-traffic reads,
and web-app traffic snapshots/inserts also use `MASAR_API_REPORT_PATH` when
they run. Reports can contain prompts and returned API content, so keep
`reports/` local even though credentials are redacted.

## Individual Test Files

Run any test from the repository root with pytest. Do not run test files
directly with `python tests/test_predictor.py`, because that bypasses pytest
and can cause `ModuleNotFoundError: No module named 'masar'`.

| Test file | What it verifies |
| --- | --- |
| `tests/test_predictor.py` | Predictor outputs, probability bounds, fallback heuristic, congestion ordering, and state immutability |
| `tests/test_router.py` | Low, high, and missing confidence routing |
| `tests/test_context.py` | Context component output using an injected fake LLM |
| `tests/test_optimizer.py` | Route calculation, mode thresholds, and unknown endpoints |
| `tests/test_synthesis.py` | English and Arabic synthesis output |
| `tests/test_graph.py` | Full LangGraph deep and fast paths |
| `tests/test_live_traffic.py` | Supabase row mapping, rain/temperature handling, and row age |
| `tests/test_training_data.py` | Real/synthetic provenance and read-only training boundary |
| `tests/test_export_synthetic.py` | Excel export row count and synthetic-only contents |
| `tests/test_api_report.py` | API report fields, response capture, and secret redaction |

Examples:

```powershell
.\VE\Scripts\python.exe -m pytest tests\test_predictor.py -q
.\VE\Scripts\python.exe -m pytest tests\test_graph.py -q
.\VE\Scripts\python.exe -m pytest tests\test_training_data.py -q
```

Run every test file separately and see which file fails:

```powershell
Get-ChildItem tests\test_*.py | ForEach-Object {
  Write-Host "===== $($_.Name) ====="
  .\VE\Scripts\python.exe -m pytest $_.FullName -q
  if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}
```

Run the complete suite:

```powershell
.\VE\Scripts\python.exe -m pytest tests\ -q
```

For input/output explanations rather than only pass counts, use:

```powershell
.\VE\Scripts\python.exe -m scripts.check_components
```

That report prints the test input, observed output, and reason each component
passed. Add `--report-path reports/api_calls.jsonl` to also save call details.

## Training the model

```bash
python -m scripts.train_model --synthetic-rows 2000   # synthetic only
python -m scripts.train_model --include-real          # + your Supabase data
python -m scripts.train_model --real-only             # Supabase data only
```

Generate a local Excel copy of synthetic data for inspection or import:

```powershell
.\VE\Scripts\python.exe -m scripts.export_synthetic --rows 6000
```

This creates `data/synthetic_traffic.xlsx`. It contains synthetic rows only,
with `source=synthetic`, and never writes to Supabase.

Benchmark the saved model against a majority baseline and the existing
`1 - congestion_ratio` heuristic:

```powershell
.\VE\Scripts\python.exe -m scripts.benchmark_model `
  --rows 6000 `
  --model models\congestion_xgb.pkl `
  --output reports\model_benchmark.json
```

This writes both `reports/model_benchmark.json` and a readable
`reports/model_benchmark.md`. The current synthetic test result is:

| Benchmark | Accuracy | Balanced accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Majority baseline | 0.925 | 0.500 | 0.000 | 0.000 | 0.000 | 0.500 | 0.075 |
| Heuristic | 0.807 | 0.656 | 0.189 | 0.478 | 0.270 | 0.858 | 0.386 |
| XGBoost | 0.928 | 0.925 | 0.509 | 0.922 | 0.656 | 0.982 | 0.871 |

The test contains 6,000 synthetic rows, with 4,800 used for training and
1,200 held out for testing. There are 90 anomalous rows in the test split.
Because labels are generated by the synthetic z-score rule, these metrics
measure how well the model reproduces that rule. They are not evidence of
future real-world traffic accuracy. Real-only metrics should be reported
after independently labeled traffic outcomes are available.

The current benchmark is based on 6,000 synthetic rows and compares three
strategies on 1,200 held-out rows:

- Majority baseline: accuracy 0.925, balanced accuracy 0.500, recall 0.000,
  PR-AUC 0.075
- Existing heuristic: accuracy 0.807, balanced accuracy 0.656, recall 0.478,
  PR-AUC 0.386
- XGBoost: accuracy 0.928, balanced accuracy 0.925, recall 0.922, PR-AUC
  0.871

Regenerate the detailed JSON and readable Markdown reports with
`scripts.benchmark_model`. These metrics measure reproduction of synthetic
labels, not validated future real-world traffic forecasting.

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
- Real-only model evaluation needs independently labeled future traffic
  outcomes; current evaluation uses synthetic labels
- Reddit collection currently uses the public JSON endpoint despite OAuth
  secrets being present in the workflow
