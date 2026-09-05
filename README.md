# Masar

Masar is a Dubai traffic intelligence prototype. It collects road-flow and weather data, stores the observations in Supabase, gathers relevant public Reddit chatter, and predicts congestion with a scikit-learn model or a lightweight heuristic fallback.

## Features

- Collects traffic flow from TomTom for selected Dubai locations.
- Collects weather conditions from OpenWeather.
- Stores traffic observations in Supabase table `traffic_logs`.
- Collects recent traffic-related posts from `r/dubai` and stores them in `reddit_chatter`.
- Converts traffic rows into a shared feature vector for model training and prediction.
- Uses a trained model when available and falls back to a low-confidence heuristic when it is not.
- Defines a shared `MasarState` contract for downstream graph nodes and bilingual recommendations.

## Project Layout

```text
.
├── scripts/
│   ├── collect_traffic.py   # TomTom and OpenWeather collection
│   ├── collect_chatter.py   # Reddit collection and Supabase upserts
│   ├── train_model.py       # Train and save the congestion classifier
│   ├── check_env.py         # Check whether environment variables are loaded
│   ├── test_database_insert.py # Manual Supabase write check
│   └── test_groq.py         # Manual Groq API check
├── predictor.py             # Compatibility export for predictor_node
├── state.py                 # Compatibility export for MasarState
├── sql/
│   └── supabase_reddit_chatter.sql
├── masar/
│   ├── features.py          # Feature order and row transformations
│   ├── state.py             # Shared TypedDict state contract
│   └── nodes/predictor.py   # Model-backed predictor node
└── tests/
    └── test_predictor.py
```

## Requirements

- Python 3.12 or newer recommended
- A Supabase project
- TomTom Traffic API key
- OpenWeather API key
- Groq API key only if you run `test_groq.py`

Install the dependencies in the active virtual environment:

```powershell
python -m pip install -r requirements.txt
```

## Configuration

Create a `.env` file in the project root. Do not commit this file.

```dotenv
TOMTOM_API_KEY=your_tomtom_api_key
OPENWEATHER_API_KEY=your_openweather_api_key
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_KEY=your_supabase_key
REDDIT_USER_AGENT=python:masar:v1.0 (by /u/your_reddit_username)
GROQ_API_KEY=your_groq_api_key
```

Run the environment check before making API requests:

```powershell
python -m scripts.check_env
```

The check prints only a shortened representation of secret values. Treat the output as sensitive nonetheless.

## Supabase Setup

Run [sql/supabase_reddit_chatter.sql](sql/supabase_reddit_chatter.sql) in the Supabase SQL editor to create the `reddit_chatter` table.

The traffic collector expects a `traffic_logs` table with at least these columns:

| Column | Description |
| --- | --- |
| `location_name` | Named Dubai location |
| `lat`, `lon` | Coordinates |
| `current_speed` | Current road speed in km/h |
| `free_flow_speed` | Expected free-flow speed in km/h |
| `speed_ratio` | Current speed divided by free-flow speed |
| `delay_seconds` | Current travel-time delay |
| `weather_main` | OpenWeather condition category |
| `rain_mm` | Rainfall amount |
| `raw_data` | Original TomTom and OpenWeather payloads as JSON |

## Collect Data

Collect traffic and weather observations for the configured Dubai locations:

```powershell
python -m scripts.collect_traffic
```

Collect matching Reddit posts from the previous 24 hours:

```powershell
python -m scripts.collect_chatter
```

Both collectors require network access and valid Supabase credentials. The Reddit collector skips storage when credentials are missing; the traffic collector raises an error if no rows are stored.

## View Stored Data

Use this read-only snippet to inspect the latest rows from both Supabase tables:

```python
import os
from pprint import pprint

from dotenv import load_dotenv
from supabase import create_client

load_dotenv()

supabase = create_client(
    os.environ["SUPABASE_URL"],
    os.environ["SUPABASE_KEY"],
)

traffic = (
    supabase.table("traffic_logs")
    .select("*")
    .order("created_at", desc=True)
    .limit(10)
    .execute()
)

chatter = (
    supabase.table("reddit_chatter")
    .select("reddit_id,title,score,post_url,published_at,snippet")
    .order("published_at", desc=True)
    .limit(10)
    .execute()
)

print("Latest traffic rows:")
pprint(traffic.data)

print("\nLatest Reddit chatter:")
pprint(chatter.data)
```

Save it as `view_data.py` or paste it into a Python session, then run:

```powershell
python view_data.py
```

If `traffic_logs` does not have a `created_at` column, order by an available timestamp column or remove the `.order(...)` call.

## Train the Model

Train on deterministic synthetic history without requiring Supabase traffic rows:

```powershell
python -m scripts.train_model
```

By default, the model is saved to `models/congestion_xgb.pkl` (the filename is retained for compatibility even though the current implementation uses `RandomForestClassifier`).

Include collected traffic rows alongside synthetic rows:

```powershell
python -m scripts.train_model --include-real --synthetic-rows 200
```

Train only on real Supabase rows:

```powershell
python -m scripts.train_model --real-only
```

Use a custom model path:

```powershell
python -m scripts.train_model --output models/my_model.pkl
```

Set `MASAR_MODEL_PATH` when loading a custom model:

```powershell
$env:MASAR_MODEL_PATH = "models/my_model.pkl"
```

The classifier labels a row as congested when its speed ratio is below `0.7`. The shared feature order is:

```text
current_speed, free_flow_speed, congestion_ratio, temperature, is_raining
```

## Tests

Run the predictor tests:

```powershell
python -m pytest
```

The predictor tests force the no-model path, so they do not need a trained model or API credentials.

## Notes

- `scripts/test_database_insert.py` performs a real insert into `traffic_logs`; use it only when you intentionally want to test database writes.
- `scripts/test_groq.py` makes a real Groq request and requires `GROQ_API_KEY`.
- API responses are retained under `raw_data`; review Supabase retention and access policies before using production data.