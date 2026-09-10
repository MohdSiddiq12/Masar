# Masar (مسار) — Dubai Congestion Prediction and Commute Recommendation System

Masar is a Dubai-focused traffic and multimodal commute decision system built around a LangGraph workflow. It predicts congestion, routes low-confidence cases through a context-enrichment step, recommends a travel mode, and returns bilingual English/Arabic guidance. The project is intentionally designed for local experimentation, demo use, and extension into a production-ready travel intelligence platform.

> Facts: This repository uses Python 3.11, `langgraph`, `langchain-groq`, `xgboost`, `networkx`, and a standard-library HTTP server. The code is organized around a stateful graph rather than a full FastAPI service.
>
> Assumptions: Real RTA routing data, a live events API, and full production authentication were not implemented in this repo. Where the project required external data or production-grade services, the code either uses a stub or a documented placeholder.
>
> Recommendations: Treat the current system as a working local prototype for urban mobility reasoning. Use it as a foundation for production integration with verified traffic feeds, persistent storage, and proper auth.

Read the [Documentation](docs/Masar_Technical_Documentation.pdf) for in-depth information.
## 1. Executive Overview

### Project purpose
Masar addresses a specific urban mobility problem in Dubai: road congestion is often more important than mass-transit reliability, and mode choice between driving, metro, or drive-to-metro is not always obvious when traffic and weather change rapidly.

### Business problem
A commuter needs actionable guidance that accounts for real-time traffic conditions, route congestion, weather, key corridors, and likely delays. The repo models that decision as a structured AI workflow instead of a simple static map or rule engine.

### Objectives
- Predict whether a route or location is anomalously congested.
- Route low-confidence predictions through a deeper reasoning path.
- Recommend a travel mode and route.
- Return bilingual English/Arabic experience messages.
- Work with local demo data or live Supabase/traffic feed data.
- Keep the system testable without network calls by supporting fake LLMs and synthetic data.

### Target users
- Dubai commuters and local mobility planners.
- Engineers validating AI-driven route recommendations.
- Demonstrators and product teams preparing a local urban mobility prototype.

### Stakeholders
- System owners: project maintainers and researchers.
- Data and model providers: TomTom, OpenWeatherMap, Supabase, Groq.
- End users: route recommendation consumers and local demo users.

### Core features
- Congestion prediction using XGBoost.
- Dynamic graph routing with LangGraph and a fallback deep path.
- Context agent for low-confidence forecast scenarios.
- NetworkX-based route optimization across Dubai hubs and districts.
- Bilingual synthesis of recommendations.
- Local web app with map, manual inputs, and live traffic integration.
- Synthetic data generation for model training without needing production data.

### Success metrics
The project is considered successful when it can:
- produce a valid congestion probability and confidence score,
- execute fast and deep graph paths correctly,
- produce bilingual recommendations from stateful inputs,
- serve a demo route recommendation without network dependencies,
- train a model on synthetic or real traffic data and persist it.

### Project scope
- Dubai corridor modeling,
- local route recommendations,
- congestion scoring,
- synthetic training data,
- web demo and API surface,
- LLM-driven reasoning hooks.

### Out of scope
- Full turn-by-turn road navigation,
- official Dubai Roads and Transport Authority (RTA) integration,
- production authentication and RBAC,
- event-stream ingestion pipelines beyond the current traffic collector,
- production-grade multi-tenant deployment and observability.

---

## 2. Complete Workspace Analysis

### Root-level repository structure

```text
Masar/
├── .env.example (assumed if present outside repo snapshot)
├── Dockerfile
├── README.md
├── context.py
├── docker-compose.yaml
├── features.py
├── graph.py
├── LICENSE
├── map.py
├── optimizer.py
├── predictor.py
├── pytest.ini
├── requirements.txt
├── router.py
├── state.py
├── synthesis.py
├── tester.py
├── tomtom_route_sample.json
├── train_model.py
├── web_app.py
├── data/
├── docs/
│   └── archive/
├── frontend/
├── masar/
│   ├── __init__.py
│   ├── api_report.py
│   ├── features.py
│   ├── graph.py
│   ├── live_data.py
│   ├── live_traffic.py
│   ├── locations.py
│   ├── state.py
│   └── nodes/
│       ├── __init__.py
│       ├── context.py
│       ├── optimizer.py
│       ├── predictor.py
│       ├── router.py
│       └── synthesis.py
├── models/
├── reports/
├── scripts/
│   ├── benchmark_model.py
│   ├── check_components.py
│   ├── check_env.py
│   ├── collect_chatter.py
│   ├── collect_traffic.py
│   ├── export_synthetic.py
│   ├── test_database_insert.py
│   ├── test_groq.py
│   └── train_model.py
├── sql/
│   └── supabase_reddit_chatter.sql
├── tests/
│   ├── conftest.py
│   ├── test_api_report.py
│   ├── test_context.py
│   ├── test_export_synthetic.py
│   ├── test_graph.py
│   ├── test_live_traffic.py
│   ├── test_optimizer.py
│   ├── test_predictor.py
│   ├── test_router.py
│   ├── test_synthesis.py
│   ├── test_training_data.py
│   └── test_web_app.py
├── VE/
└── other local artifacts
```

### Folder responsibilities

| Folder / path | Purpose | Responsibility | Main interaction |
| --- | --- | --- | --- |
| `masar/` | Core application package | Contains the model, graph, state, live data, and routing logic | Imported by the app and training scripts |
| `masar/nodes/` | Workflow nodes | Implements predictor, router, context, optimizer, and synthesis nodes | Executed by `masar.graph.build_graph()` |
| `scripts/` | Operational automation | Data collection, env checks, benchmark, training, export | Invoked manually or by CI |
| `tests/` | Quality gate | Unit and integration tests for each major component | Run via `pytest` |
| `frontend/` | Browser UI | Static HTML and client-side UI logic | Served by `web_app.py` |
| `data/` | Local datasets | Stores raw or derived data snapshots | Used by scripts and analysis |
| `models/` | Model artifacts | Stores trained XGBoost models and local checkpoints | Loaded by `masar/nodes/predictor.py` |
| `reports/` | Benchmark reports | Stores generated benchmark outputs | Used for model evaluation and documentation |
| `docs/` | Historical reference docs | Archive of old implementations and diffs | Provides migration context |
| `sql/` | Data-contract definitions | Database schema and examples for Supabase | Supports traffic and chatter setups |
| `VE/` | Local virtual environment | Local runtime environment for this workspace | Used during local development |

### Key files and purpose inventory

| File | Purpose | Imported by | Depends on |
| --- | --- | --- | --- |
| `web_app.py` | Local recommendation HTTP server and demo app | Run directly | `masar.graph`, `masar.live_traffic`, `masar.locations` |
| `masar/graph.py` | Compiles the LangGraph workflow | `web_app.py`, tests | all node modules and `MasarState` |
| `masar/state.py` | Shared state contract | all nodes and graph | `TypedDict`, `operator`, `Literal` |
| `train_model.py` | Trains the anomaly classifier | CLI entrypoint | `masar.features`, `masar.locations`, `xgboost` |
| `masar/nodes/predictor.py` | Congestion prediction | graph | `joblib`, `numpy`, `masar.features` |
| `masar/nodes/router.py` | Chooses deep vs fast path | graph | `langgraph.types.Command` |
| `masar/nodes/context.py` | Context enrichment and reasoning | graph | `langchain_groq`, `masar.api_report` |
| `masar/nodes/optimizer.py` | Route decision and mode selection | graph | `networkx`, `masar.locations` |
| `masar/nodes/synthesis.py` | Bilingual final output | graph | `pydantic`, LLM output |
| `masar/live_data.py` | Live TomTom/OpenWeather fetching | `masar/live_traffic.py` | `httpx`, `masar.api_report` |
| `masar/live_traffic.py` | Maps live data to state fields | `web_app.py` | `masar.locations` |
| `masar/locations.py` | Canonical Dubai catalog | optimizer, live_data, web_app | static list |
| `masar/features.py` | Feature transforms | `train_model.py`, predictor | `numpy` |
| `masar/api_report.py` | Redacted API call reporting | all external call sites | `json`, `os`, `time` |
| `scripts/collect_traffic.py` | Background traffic collection | CI or manual run | `httpx`, `supabase`, `dotenv` |
| `scripts/collect_chatter.py` | Social signal collection stub | manual run | `praw`, `dotenv` |
| `scripts/check_components.py` | End-to-end component checks | manual run | local modules |
| `map.py` | TomTom route comparison helper | ad hoc use | `httpx`, local JSON |
| `tester.py` | Environment and service smoke tests | manual run | external environment |

---

## 3. Dependency & Environment Specification

### Python and runtime environment

| Item | Value | Notes |
| --- | --- | --- |
| Python version | 3.11 | Confirmed by `Dockerfile: FROM python:3.11-slim` |
| Virtual environment | Local `VE/` folder in repo | Used for developer environment |
| Package manager | `pip` via `requirements.txt` | No `pyproject.toml` was found in the repo snapshot |
| OS | Windows support via local env; Linux container support via Docker | Dockerfile is Linux-based |

### Dependency inventory

| Package | Version | Why used |
| --- | --- | --- |
| `langgraph` | Unpinned | State-machine orchestration and graph routing |
| `langchain-groq` | Unpinned | Groq LLM client integration |
| `langchain-core` | Unpinned | Core LLM abstractions and message types |
| `pydantic` | Unpinned | Structured output validation for synthesis |
| `joblib` | Unpinned | Model serialization/deserialization |
| `numpy` | Unpinned | Numeric array generation and feature engineering |
| `pandas` | Unpinned | Data processing and training frame assembly |
| `openpyxl` | Unpinned | Excel export support for synthetic data |
| `scikit-learn` | Unpinned | Metrics and model pipeline utilities |
| `xgboost` | Unpinned | Primary congestion anomaly classifier |
| `networkx` | Unpinned | Route graph construction and shortest-path logic |
| `pytest` | Unpinned | Automated tests |
| `python-dotenv` | Unpinned | Local environment variable loading |
| `supabase` | Unpinned | Traffic data access from Supabase |
| `httpx` | Unpinned | Async HTTP client for TomTom and OpenWeatherMap |
| `praw` | Unpinned | Reddit chatter collection support |

### Runtime requirements

| Requirement | Value | Notes |
| --- | --- | --- |
| CPU | Standard x86_64 CPU acceptable | XGBoost runs on CPU by default |
| GPU | Optional, not required | No CUDA configuration or cuDNN path is defined |
| CUDA / cuDNN | Not configured | This project does not declare a CUDA environment |
| RAM | Approximately 2-8 GB typical for local use | Enough for model and request processing |
| Storage | Low; mostly source and a few model artifacts | `models/` may hold trained joblib files |
| Network | Required for live traffic, Groq, and Supabase calls | Demo mode avoids some external dependencies |

### Environment variables

| Variable | Example / default | Required | Purpose |
| --- | --- | --- | --- |
| `SUPABASE_URL` | `https://...supabase.co` | Yes for live data / real training | Supabase traffic database |
| `SUPABASE_KEY` | secret key | Yes for live data / real training | Supabase auth |
| `GROQ_API_KEY` | secret key | Yes for live LLM reasoning | LLM access |
| `GROQ_MODEL` | `openai/gpt-oss-120b` | No | Overrides LLM model |
| `TOMTOM_API_KEY` | secret key | Yes for live traffic | TomTom flow API |
| `OPENWEATHER_API_KEY` | secret key | Yes for live weather | OpenWeatherMap weather API |
| `MASAR_MODEL_PATH` | `models/congestion_xgb.pkl` | No | Path override for trained model |
| `MASAR_API_REPORT_PATH` | file path | No | Persists redacted API call data |
| `PORT` | `8000` | No | HTTP server port |
| `MASAR_PORT` | `8000` | No | Alternate port variable |
| `MASAR_HOST` | `127.0.0.1` or `0.0.0.0` | No | Bind address |

### Secrets and feature flags
- Secrets are expected in `.env` or environment variables and are never committed to source control.
- Feature flags in the code are mostly implicit: `demo`, `use_live_traffic`, and model availability gates.
- Live traffic fetches are optional; if missing credentials, the app falls back to manual inputs or demo behavior.

---

## 4. Configuration Management

### Configuration sources
- `.env` (expected by `python-dotenv.load_dotenv()`)
- OS environment variables
- Local defaults in code, such as `MODEL_PATH`, `CONFIDENCE_THRESHOLD`, and `GROQ_MODEL`
- JSON payloads in HTTP requests to the web app
- Pydantic models for synthesis output

### Variable catalog

| Variable | Description | Default | Required |
| --- | --- | --- | --- |
| `MODEL_PATH` | Path used by the predictor | `models/congestion_xgb.pkl` | No |
| `CONFIDENCE_THRESHOLD` | Router threshold for fast vs deep path | `0.7` | No |
| `location` | Current place label used for routeing | Provided at runtime | Yes in request state |
| `origin` | Route origin | `Marina` | No |
| `destination` | Route destination | `Business Bay` | No |
| `congestion_ratio` | Ratio of current to free flow speed | Derived from input or live data | No |
| `weather_condition` | Weather label | `Clear` | No |
| `is_raining` | Rain flag | `False` | No |
| `demo` | Whether to fake LLM calls in web app | `True` | No |
| `use_live_traffic` | Whether to fetch live data for a recommendation | `True` | No |

### Secrets management
The project follows a typical local-development pattern: keep secrets in `.env` and read them through `python-dotenv`. The app does not expose secrets in logs because `masar/api_report.py` redacts any key-like field names before storing them locally.

### Feature flags
- `demo` mode disables real Groq traffic for the local recommendation workflow.
- `use_live_traffic` toggles live provider calls on or off.
- `MASAR_MODEL_PATH` allow a non-default model path.
- `MASAR_API_REPORT_PATH` enables call logging for API and model requests.

---

## 5. End-to-End System Architecture

### High-level runtime flow

```text
User / Browser
      │
      ▼
web_app.py
      │
      ├── parse request payload
      ├── optionally refresh live traffic
      ├── build state dict
      └── invoke LangGraph workflow
                      │
                      ▼
            masar.graph.build_graph()
                      │
      ┌───────────────┴───────────────┐
      ▼                               ▼
predictor node                  router node
  │                                  │
  └── if model exists → XGB        └── confidence < 0.7 → deep path
                                   else → fast path
                                           │
                                           ▼
                                     context_agent (optional)
                                           │
                                           ▼
                                      route_optimizer
                                           │
                                           ▼
                                       synthesis node
                                           │
                                           ▼
                                 English + Arabic output
```

### Traffic and model lifecycle

```text
TomTom + OpenWeatherMap
          │
          ▼
    fetch_live_row()
          │
          ▼
    Supabase traffic_logs
          │
          ▼
  live_traffic.refresh_live_fields()
          │
          ▼
  recommendation state
          │
          ▼
  predictor → router → optimizer → synthesis
```

### Training architecture

```text
Synthetic rows and/or Supabase rows
                 │
                 ▼
      train_model.py
                 │
                 ▼
      label_anomalies()
                 │
                 ▼
          XGBClassifier fit
                 │
                 ▼
     models/congestion_xgb.pkl
```

### Data and signal flow

```text
Location + Speed + Weather + Incidents
            │
            ▼
         state dict
            │
            ▼
 predictor (XGB / heuristic)
            │
            ▼
      router decision
            │
       ┌────┴────┐
       ▼         ▼
  fast path  deep path
       │         │
       ▼         ▼
 route_optimizer  context_agent
       │         │
       └────┬────┘
            ▼
       synthesis
```

---

## 6. Component-Wise Technical Breakdown

### Module: `masar/nodes/predictor.py`

| Item | Details |
| --- | --- |
| Purpose | Predict whether a location/time pattern is anomalous and estimate a confidence value |
| Input | `MasarState` with current speed, free-flow speed, congestion ratio, weather, rain, temperature |
| Output | `predicted_congestion`, `prediction_confidence` |
| Dependencies | `joblib`, `numpy`, `masar.features`, `masar.state` |
| Key functions | `_load_model()`, `_build_feature_vector()`, `_heuristic_predict()`, `predictor_node()` |
| Error handling | Falls back to a heuristic model if the artifact is missing |
| Complexity | O(1) per prediction, dominated by model inference |
| Limitation | Model confidence is only as good as the data and feature design |

### Module: `masar/nodes/router.py`

| Item | Details |
| --- | --- |
| Purpose | Choose between fast and deep reasoning paths |
| Input | `prediction_confidence` and related state |
| Output | `Command` with update and route destination |
| Dependencies | `langgraph.types.Command`, `masar.nodes.predictor` |
| Key functions | `router_node()` |
| Error handling | Missing confidence triggers deep path safely |
| Complexity | O(1) |
| Limitation | Threshold is static and not learned dynamically |

### Module: `masar/nodes/context.py`

| Item | Details |
| --- | --- |
| Purpose | Add human-readable context for low-confidence conditions |
| Input | State and optional LLM client |
| Output | `context_notes`, `nearby_events`, `social_signal` |
| Dependencies | `langchain_groq` or injected LLM stub |
| Key functions | `_get_social_signal()`, `_get_nearby_events()`, `_build_prompt()`, `context_node()` |
| Error handling | Graceful fallback to stub text when real integrations are absent |
| Complexity | O(1) plus LLM latency |
| Limitation | Social layer is not fully implemented |

### Module: `masar/nodes/optimizer.py`

| Item | Details |
| --- | --- |
| Purpose | Choose a route and mode, using weighted graph shortest path logic |
| Input | Predicted congestion, origin, destination, current route state |
| Output | `recommended_mode`, `recommended_route`, `route_options` |
| Dependencies | `networkx`, `masar.locations` |
| Key functions | `_build_base_graph()`, `_build_location_graph()`, `_distance_minutes()`, `_recommend_mode()`, `route_optimizer_node()`, `_path_weight()`, `_path_congestion()` |
| Error handling | If path is unavailable, it returns empty route data |
| Complexity | Shortest path is roughly O(E log V) for the graph |
| Limitation | The graph is a corridor approximation, not a full transport network |

### Module: `masar/nodes/synthesis.py`

| Item | Details |
| --- | --- |
| Purpose | Convert route guidance into bilingual output |
| Input | Final route state and context notes |
| Output | `message_en`, `message_ar` |
| Dependencies | `pydantic`, LLM invocation |
| Key functions | `SynthesisOutput`, `_build_prompt()`, `synthesis_node()` |
| Error handling | Validates structured output using Pydantic |
| Complexity | O(1) plus LLM latency |
| Limitation | Very short prompts and no broad multi-turn memory |

### Module: `masar/live_data.py`

| Item | Details |
| --- | --- |
| Purpose | Fetch real-time TomTom and weather data and shape it into a traffic row |
| Input | location label and configured API keys |
| Output | row dict with current speed, free-flow speed, weather, and raw provider values |
| Dependencies | `httpx`, `asyncio`, `masar.locations` |
| Key functions | `_fetch_tomtom()`, `_fetch_weather()`, `fetch_live_row()`, `fetch_and_store()` |
| Error handling | Raises exceptions on missing API credentials or provider failures |
| Complexity | O(1) network calls per location |
| Limitation | Requires valid API keys; no retry/backoff implemented |

### Module: `masar/live_traffic.py`

| Item | Details |
| --- | --- |
| Purpose | Read and normalize the latest traffic row into state fields |
| Input | Supabase client and location label |
| Output | `fields`, `row`, `age_seconds` |
| Dependencies | `masar.locations`, `masar.live_data` |
| Key functions | `fetch_latest_row()`, `_row_temperature()`, `row_to_state_fields()`, `row_age_seconds()`, `get_live_fields()`, `refresh_live_fields()` |
| Error handling | Returns empty fields when no row is available |
| Complexity | O(1) |
| Limitation | Depends on the table schema and raw data quality |

---

## 7. Function-by-Function Documentation

### `train_model.py`

#### `generate_synthetic_rows(n_rows: int, seed: int = 42) -> pd.DataFrame`
Purpose: Generate synthetic Dubai congestion data for local training and validation.

Parameters:
| Name | Type | Description |
| --- | --- | --- |
| `n_rows` | `int` | Number of synthetic rows |
| `seed` | `int` | Random seed for deterministic generation |

Returns:
| Type | Description |
| --- | --- |
| `pd.DataFrame` | Synthetic rows with location, weather, and congestion indicators |

Called by: `main()`
Calls: none directly in the module
Exceptions: None expected
Time complexity: O(n_rows)

#### `fetch_real_rows(client=None) -> pd.DataFrame`
Purpose: Load traffic rows from Supabase without writing data.

Parameters:
| Name | Type | Description |
| --- | --- | --- |
| `client` | optional client object | Supabase client override |

Returns:
| Type | Description |
| --- | --- |
| `pd.DataFrame` | Real traffic rows normalized for training |

Called by: `main()`
Calls: `masar.api_report.measured_call()`, `row_to_features()`
Exceptions: Supabase access errors
Time complexity: O(n)

#### `combine_training_data(*frames: pd.DataFrame) -> pd.DataFrame`
Purpose: Concatenate training frames while validating provenance.

Parameters:
| Name | Type | Description |
| --- | --- | --- |
| `frames` | `pd.DataFrame` | Training data frames |

Returns:
| Type | Description |
| --- | --- |
| `pd.DataFrame` | Combined local training data |

Called by: `main()`
Calls: none outside pandas
Exceptions: `ValueError` if sources are not in `{real, synthetic}`
Time complexity: O(n)

#### `label_anomalies(df: pd.DataFrame) -> pd.DataFrame`
Purpose: Compute a per-location time-of-day baseline and label anomalies.

Parameters:
| Name | Type | Description |
| --- | --- | --- |
| `df` | `pd.DataFrame` | Training dataset |

Returns:
| Type | Description |
| --- | --- |
| `pd.DataFrame` | Dataset with `is_anomaly` column |

Called by: `main()`
Calls: pandas groupby operations
Exceptions: None expected
Time complexity: O(n)

#### `train(df: pd.DataFrame) -> XGBClassifier`
Purpose: Train a gradient-boosted anomaly classifier.

Parameters:
| Name | Type | Description |
| --- | --- | --- |
| `df` | `pd.DataFrame` | Labeled training data |

Returns:
| Type | Description |
| --- | --- |
| `XGBClassifier` | Trained model |

Called by: `main()`
Calls: `model.fit()`
Exceptions: None beyond model training errors
Time complexity: O(n * trees)

#### `main()`
Purpose: CLI entrypoint for training and saving the model artifact.

Parameters: None; uses CLI args.
Returns: None
Called by: direct Python execution
Calls: `generate_synthetic_rows()`, `fetch_real_rows()`, `combine_training_data()`, `label_anomalies()`, `train()`
Exceptions: `ValueError` if the dataset lacks both classes
Time complexity: O(n * trees)

### `masar/nodes/predictor.py`

#### `_load_model()`
Purpose: Cache a model object from disk if available.

Parameters: None
Returns: model or `None`
Called by: `predictor_node()`
Calls: `joblib.load()`
Exceptions: file-not-found is handled by existence check
Time complexity: O(1) after first load

#### `_build_feature_vector(state: MasarState) -> np.ndarray`
Purpose: Build a 1xN feature vector matching the training columns.

Parameters:
| Name | Type | Description |
| --- | --- | --- |
| `state` | `MasarState` | Runtime forecasting input |

Returns:
| Type | Description |
| --- | --- |
| `np.ndarray` | Features in the expected order |

Called by: `predictor_node()`
Calls: none outside NumPy
Exceptions: None expected
Time complexity: O(F), where F is feature count

#### `_heuristic_predict(state: MasarState) -> tuple[float, float]`
Purpose: Provide a fallback rule-based congestion estimate when no trained model exists.

Parameters:
| Name | Type | Description |
| --- | --- | --- |
| `state` | `MasarState` | Runtime data |

Returns:
| Type | Description |
| --- | --- |
| `(float, float)` | Predicted congestion score and low confidence |

Called by: `predictor_node()`
Calls: none
Exceptions: None
Time complexity: O(1)

#### `predictor_node(state: MasarState) -> dict`
Purpose: top-level prediction node attached to the graph.

Parameters:
| Name | Type | Description |
| --- | --- | --- |
| `state` | `MasarState` | Graph state |

Returns:
| Type | Description |
| --- | --- |
| `dict` | `predicted_congestion` and `prediction_confidence` |

Called by: graph runtime
Calls: `_load_model()`, model prediction or `_heuristic_predict()`
Exceptions: file and model loading errors are guarded
Time complexity: O(1) or model inference cost

### `masar/nodes/router.py`

#### `router_node(state: MasarState) -> Command[Literal["context_agent", "route_optimizer"]]`
Purpose: Determine whether the request should route to context or optimizer.

Parameters:
| Name | Type | Description |
| --- | --- | --- |
| `state` | `MasarState` | Graph state |

Returns:
| Type | Description |
| --- | --- |
| `Command` | LangGraph route command |

Called by: graph runtime
Calls: none beyond `Command` update
Exceptions: none
Time complexity: O(1)

### `masar/nodes/context.py`

#### `_get_social_signal(location: str) -> str`
Purpose: Return a placeholder social signal when Reddit integration is not active.

Parameters:
| Name | Type | Description |
| --- | --- | --- |
| `location` | `str` | Location label |

Returns:
| Type | Description |
| --- | --- |
| `str` | Human-readable message |

Called by: `context_node()`
Calls: none
Exceptions: none
Time complexity: O(1)

#### `_get_nearby_events(location: str) -> list[str]`
Purpose: Return an empty list because no events service is wired.

Parameters:
| Name | Type | Description |
| --- | --- | --- |
| `location` | `str` | Location label |

Returns:
| Type | Description |
| --- | --- |
| `list[str]` | Empty list |

Called by: `context_node()`
Calls: none
Exceptions: none
Time complexity: O(1)

#### `_build_prompt(state: MasarState, social_signal: str, events: list[str]) -> str`
Purpose: Format prompt for a low-confidence context explanation.

Parameters: described by function name
Returns: string prompt
Called by: `context_node()`
Calls: none
Exceptions: none
Time complexity: O(1)

#### `context_node(state: MasarState, llm=None) -> dict`
Purpose: Query an LLM to explain unusual congestion and enrich the graph state.

Parameters:
| Name | Type | Description |
| --- | --- | --- |
| `state` | `MasarState` | Current graph state |
| `llm` | optional object | Fake or real LLM client |

Returns:
| Type | Description |
| --- | --- |
| `dict` | context notes and social/event metadata |

Called by: graph runtime
Calls: `measured_call()`, `llm.invoke()`
Exceptions: provider exceptions propagate
Time complexity: O(1) plus model latency

### `masar/nodes/optimizer.py`

#### `_build_base_graph()`
Purpose: Create a minimal corridor graph linking key Dubai hubs.

Returns: `networkx.Graph`
Called by: `route_optimizer_node()`
Calls: `graph.add_weighted_edges_from()`
Time complexity: O(E)

#### `_build_location_graph()`
Purpose: Expand the graph to include additional Dubai districts by nearest-hub approximation.

Returns: `networkx.Graph`
Called by: `route_optimizer_node()`
Calls: `_distance_minutes()`
Time complexity: O(V log V)

#### `_distance_minutes(source: str, target: str) -> float`
Purpose: Approximate geographic distance between two coordinates.

Returns: float minutes-like value
Called by: `_build_location_graph()`
Exceptions: none
Time complexity: O(1)

#### `_recommend_mode(congestion: float) -> str`
Purpose: Turn predicted congestion into one of `drive`, `drive_to_metro`, or `metro`.

Returns: string mode
Called by: `route_optimizer_node()`
Time complexity: O(1)

#### `route_optimizer_node(state: MasarState) -> dict`
Purpose: compute the best route and route options under congestion stress.

Returns: `recommended_mode`, `recommended_route`, `route_options`
Called by: graph runtime
Calls: `networkx.dijkstra_path()`, `_path_weight()`, `_path_congestion()`
Exceptions: NodeNotFound and no-path conditions are handled gracefully
Time complexity: O(E log V)

### `masar/nodes/synthesis.py`

#### `SynthesisOutput` model
Purpose: Structured output schema for English + Arabic messages.

Returns: Pydantic schema with required strings.
Used by: `synthesis_node()`

#### `_build_prompt(state: MasarState) -> str`
Purpose: Generate a short message prompt for bilingual synthesis.

Calls: `synthesis_node()`
Time complexity: O(1)

#### `synthesis_node(state: MasarState, llm=None) -> dict`
Purpose: Produce final bilingual output.

Returns: `message_en`, `message_ar`
Calls: `llm.invoke()` or Groq structured output
Exceptions: invalid output is validated by Pydantic
Time complexity: O(1) plus LLM latency

### `web_app.py`

#### `make_state(payload)`
Purpose: Convert HTTP JSON into a valid `MasarState`.

#### `_get_supabase_client()`
Purpose: Build a Supabase client from environment variables.

#### `run_recommendation(payload)`
Purpose: Build the graph, optionally refresh live traffic, and return a result payload.

#### `traffic_snapshot()`
Purpose: Read recent traffic rows from Supabase for the UI dashboard.

#### `latest_traffic_for_location(location)`
Purpose: Return the most recent row for a particular location.

#### `MasarHandler.do_GET()`
Purpose: Serve health, root page, traffic, and latest traffic endpoints.

#### `MasarHandler.do_POST()`
Purpose: Accept `POST /api/recommend` and run the recommendation workflow.

### `masar/live_data.py`

#### `_fetch_tomtom(client, lat, lon, api_key)`
Purpose: Fetch and normalize TomTom flow data.

#### `_fetch_weather(client, lat, lon, api_key)`
Purpose: Fetch and normalize OpenWeatherMap data.

#### `fetch_live_row(location)`
Purpose: Retrieve a fully assembled live traffic row from provider APIs.

#### `fetch_and_store(location, supabase_client)`
Purpose: Persist a row into Supabase after fetching live traffic.

---

## 8. AI Agent Documentation

| Agent | Responsibility | Goal | Tooling |
| --- | --- | --- | --- |
| Predictor | Estimate congestion | Output a predicted score + confidence | XGBoost model or heuristic fallback |
| Router | Decide reasoning path | Send low-confidence cases to context route | LangGraph Command |
| Context Agent | Diagnose unusual congestion | Explain likely cause using available signals | Groq or test fake LLM |
| Route Optimizer | Find route and mode | Compute best route under current congestion | NetworkX Dijkstra |
| Synthesis Agent | Produce final user-facing response | Return short bilingual recommendation | Pydantic + Groq or fake LLM |

### Decision logic

```text
Predictor emits score p and confidence c
     │
     ▼
if c < 0.7 or missing => deep path
else => fast path
     │
     ├──── fast: route_optimizer → synthesis
     └──── deep: context_agent → route_optimizer → synthesis
```

### Failure behavior
- Missing model: predictor falls back to heuristic scoring.
- Missing confidence: router treats it as an anomaly and uses the deep path.
- Missing provider credentials: live traffic lookup fails quietly or reverts to manual input.
- No route path: optimizer returns empty route data and still offers mode selection logic.
- LLM failure: in the local web app, demo mode prevents network dependencies; without demo mode, errors may bubble into the API response.

---

## 9. Model Documentation

| Item | Value |
| --- | --- |
| Model name | `XGBClassifier` |
| Version | Current artifact is not version-tagged in repo; model is saved as `models/congestion_xgb.pkl` |
| Provider | Local training pipeline using XGBoost |
| Context window | Not applicable to tabular model |
| Max tokens | Not applicable |
| Embedding dimension | Not applicable |
| Quantization | Not implemented |
| Parameters | `n_estimators=200`, `max_depth=4`, `learning_rate=0.05`, `scale_pos_weight` balancing |

### Why chosen
The project is structured as a small, explainable anomaly detector for traffic behavior rather than a deep neural network. XGBoost is appropriate for tabular location/time/weather data and supports local inference with low operational complexity.

### LLM model documentation

| Item | Value |
| --- | --- |
| Model name | `openai/gpt-oss-120b` by default |
| Provider | Groq |
| Temperature | `0.2` in context, `0.3` in synthesis |
| Max output | Not explicitly limited in code |
| Streaming | Not implemented in the current graph |
| Prompting pattern | Brief, state-driven prompts with route and weather context |

### Prompt and generation behavior
- The context node passes a short, factual prompt about location, congestion ratio, and weather.
- The synthesis node is instructed to write a short English message followed by the same message in Arabic.
- Structured outputs are validated via `pydantic.BaseModel`.

---

## 10. Token & Context Management

### Input token flow
- Request state is small and highly structured.
- Context prompt uses a compact textual summary rather than long retrieval chains.
- The system avoids large message histories since it is not built around a conversational memory layer.

### Context construction
The context prompt includes:
- location,
- congestion ratio,
- weather and rainfall,
- incidents,
- nearby events,
- social signal.

This is intentionally compact and designed to fit within a lightweight LLM context window without excessive token use.

### Chunking strategy
No chunking strategy is in place for this codebase. The data is processed at event-level granularity rather than as document chunks.

### Truncation logic
No custom truncation logic for LLMs is implemented. Prompt size is intentionally small enough to avoid hitting model limits in routine use.

### Conversation memory
There is no persistent conversational memory. Each request is a stateless state graph invocation.

```text
HTTPRequest payload
      │
      ▼
State dict created in memory
      │
      ▼
Graph nodes call LLMs with short prompt
      │
      ▼
Return response, no long-term memory retained
```

---

## 11. Data Specification

### Traffic data source
The application primarily consumes data from:
- TomTom flow segment API,
- OpenWeatherMap current weather API,
- Supabase `traffic_logs` table,
- locally generated synthetic rows.

### Database table patterns

| Table / dataset | Source | Fields | Notes |
| --- | --- | --- | --- |
| `traffic_logs` | Supabase | `location_name`, `current_speed`, `free_flow_speed`, `speed_ratio`, `weather_main`, `rain_mm`, `raw_data`, `created_at` | Main runtime signal |
| `reddit_chatter` | optional external feed | `reddit_id`, `title`, `score`, `post_url`, `published_at` | Not actively wired in the core graph |
| Synthetic training rows | generated in `train_model.py` | location-related features and anomaly labels | used for local validation |

### Feature schema for training

| Field | Type | Description |
| --- | --- | --- |
| `current_speed` | float | observed speed |
| `free_flow_speed` | float | expected uncongested speed |
| `congestion_ratio` | float | current speed / free flow |
| `temperature` | float | weather temperature |
| `is_raining` | bool | whether rain is present |
| `hour` | int | traffic hour bucket |
| `is_weekend` | bool | weekend indicator |
| `location` | str | district or corridor label |
| `timestamp` | datetime | row timestamp |
| `source` | str | `synthetic` or `real` |

### Data cleaning and validation
- Values are normalized from provider JSON into consistent keys.
- Missing weather temperature values fall back to a default of `30.0`.
- Synthetic rows are explicitly marked as `source="synthetic"`.
- Real rows are mapped into the same feature schema before training.

### Train / validation / test split
The split is time-based, not random:
- `split_idx = int(len(df) * 0.8)`
- earlier rows become training data, later rows become validation test data

This is done to avoid temporal leakage and to better simulate real-world deployment.

---

## 12. Feature Engineering

The feature list is centralized in `masar/features.py`.

```python
FEATURE_ORDER = [
    "current_speed",
    "free_flow_speed",
    "congestion_ratio",
    "temperature",
    "is_raining",
]
```

### Feature categories
- Numerical: speed, free-flow speed, temperature, congestion ratio
- Binary: `is_raining`
- Location-based: categorical labels like `Marina`, `Business Bay`
- Contextual aggregate: time-of-day bucket and weekend flag generated during training

### Transformation pipeline

```text
Raw traffic event
      │
      ▼
Normalize provider values
      │
      ▼
Map to feature order
      │
      ▼
Optional anomaly labeling
      │
      ▼
XGBClassifier training
```

### Notes
The model is not a heavy feature pipeline; it relies on a compact, interpretable feature set. This makes debugging easier and aligns with the project’s small local prototype design.

---

## 13. External APIs & Services

| Service | Purpose | Authentication | Status |
| --- | --- | --- | --- |
| TomTom Traffic API | Current speed and flow metadata | `TOMTOM_API_KEY` | Active in `live_data.py` |
| OpenWeatherMap | Weather and rainfall | `OPENWEATHER_API_KEY` | Active in `live_data.py` |
| Supabase | Persistence and traffic lookups | `SUPABASE_URL` + `SUPABASE_KEY` | Used for local data and dashboard |
| Groq | LLM reasoning | `GROQ_API_KEY` | Active in context and synthesis nodes |
| Reddit / PRAW | social chatter collection | credentials or app token | stubbed / pending |
| GitHub Actions | Traffic collector automation | repo secrets | configured in workflows |

### TomTom
Request pattern:
- URL: `https://api.tomtom.com/.../flowSegmentData/absolute/12/json`
- Uses `point`, `unit`, and API key
- Returns current and free-flow speed plus time metrics

### OpenWeatherMap
Request pattern:
- URL: `https://api.openweathermap.org/data/2.5/weather`
- Uses `lat`, `lon`, `units=metric`, API key
- Returns current weather, temperature, and rainfall

### Supabase
- Read-only selects are used in the training path.
- The app can insert live rows into `traffic_logs`.
- Data is expected to be shaped to `location_name`, `current_speed`, `speed_ratio`, and `raw_data` fields.

### Rate limits and retries
There is no formal retry policy or backoff schedule. The current system expects external APIs to be stable and fails gracefully on error.

---

## 14. Backend API Documentation

The project does not use FastAPI or Flask. It uses a lightweight `BaseHTTPRequestHandler` inside `web_app.py`.

### `GET /health`
Purpose: Health check endpoint.

Response:
```json
{"status": "ok"}
```

### `GET /`
Purpose: Serve the frontend page.

Response: HTML page from `frontend/index.html`

### `GET /api/traffic`
Purpose: Fetch recent traffic rows.

Response:
```json
{"rows": [{"location_name": "...", "current_speed": 32.0, "free_flow_speed": 90.0}]}
```

### `GET /api/locations`
Purpose: List map locations and coordinates.

Response:
```json
[{"label": "Marina", "lat": 25.08, "lon": 55.14}]
```

### `GET /api/traffic/latest?location=Marina`
Purpose: Fetch the most recent data point for a specific location.

Response:
```json
{"row": {...}, "fields": {...}, "age_seconds": 240.1}
```

### `POST /api/recommend`
Purpose: Run a recommendation and return the full graph output.

Request example:
```json
{
  "demo": true,
  "use_live_traffic": false,
  "origin": "Marina",
  "destination": "Business Bay",
  "current_speed": 25,
  "free_flow_speed": 90,
  "temperature": 36,
  "weather_condition": "Clear",
  "is_raining": false
}
```

Response example:
```json
{
  "result": {
    "predicted_congestion": 0.62,
    "prediction_confidence": 0.76,
    "recommended_mode": "drive_to_metro",
    "recommended_route": ["Marina", "Business Bay"],
    "message_en": "Use the recommended route and allow extra time for congestion.",
    "message_ar": "استخدم المسار الموصى به واترك وقتًا إضافيًا للازدحام."
  }
}
```

### API validation and status handling
- Request body is parsed as JSON.
- A fallback path re-runs the graph in demo mode if the regular mode fails.
- HTTP status handling uses `200`, `400`, `404`, `500`, and `503` depending on the case.

---

## 15. Runtime Execution Flow

### Startup sequence
1. Load environment variables with `python-dotenv`.
2. Initialize the standard HTTP server in `web_app.py`.
3. Determine bind address and port.
4. Serve root page and routes.
5. Optional: initialize a live traffic lookup if credentials are available.

### Inference sequence
1. Receive JSON request.
2. Build `MasarState` from payload.
3. Refresh live fields if `use_live_traffic` is enabled.
4. Invoke `build_graph()`.
5. Run `predictor_node()`.
6. Route through `router_node()`.
7. Optionally call `context_node()`.
8. Run `route_optimizer_node()`.
9. Call `synthesis_node()`.
10. Return bilingual result JSON.

```text
Start
  │
  ▼
Load env
  │
  ▼
HTTP server ready
  │
  ▼
POST /api/recommend
  │
  ▼
State assembly
  │
  ▼
Live data refresh (optional)
  │
  ▼
graph.invoke(state)
  │
  ├── predictor
  ├── router
  ├── context_agent (optional)
  ├── route_optimizer
  └── synthesis
  │
  ▼
JSON response
```

---

## 16. MLOps Pipeline

### Training lifecycle
- Synthetic generation is the default path in `train_model.py`.
- Real data may be included with `--include-real` or `--real-only`.
- The anomaly label is computed based on location-hour-weekend baselines.
- Model training is local and artifact-based.

### Reproducibility
- `seed=42` is used by default in synthetic generation.
- Time-based train/test split reduces leakage.
- Model artifact is persisted to `models/congestion_xgb.pkl`.

### Versioning and artifacts
| Artifact | Description |
| --- | --- |
| `models/congestion_xgb.pkl` | Serialized trained model |
| `reports/model_benchmark.json` | Benchmark output |
| `reports/model_benchmark.md` | Human-readable report |
| `tomtom_route_sample.json` | Sample route payload |

### Model registry status
There is no formal model registry or MLflow integration in the repo snapshot. The project is currently artifact-based rather than registry-based.

---

## 17. Monitoring & Observability

### Built-in observability
The project includes optional API call logging via `masar/api_report.py`.

- It redacts secret-like keys.
- It records call metadata in JSON lines.
- It writes to a path controlled by `MASAR_API_REPORT_PATH`.

### Logged metrics
- latency per external call,
- request payload metadata,
- status (success/error),
- response summary,
- error string if present.

### Current monitoring gaps
- no centralized metrics backend,
- no Prometheus or Grafana integration,
- no explicit trace IDs beyond the local report file,
- no alerting thresholds defined in code.

### Recommended future monitoring
- p95 latency for Groq / TomTom / OpenWeatherMap calls
- request volume and failure rate per endpoint
- model confidence distribution
- route recommendation coverage and fallback counts

---

## 18. Security & Compliance

### Current security posture
This repo is a local prototype and does not implement production identity or access control. It is best treated as a demo-grade application.

| Area | Current state |
| --- | --- |
| Authentication | None for the HTTP server |
| Authorization | None |
| RBAC | Not implemented |
| Encryption in transit | HTTPS is assumed by external providers |
| Secrets handling | `dotenv`-based local env variables |
| PII handling | Minimal; no explicit PII collection |
| Audit logs | Basic redacted API report logs |
| Prompt injection defense | Not formalized |

### Data protection recommendations
- Put credentials in a secure secret manager in production.
- Add request authentication for `/api/recommend`.
- Avoid exposing raw provider payloads outside controlled logs.
- Add rate limiting and request validation for inbound traffic.
- Keep benchmark and report files from containing sensitive request bodies.

### Compliance notes
- The repo does not yet declare GDPR, HIPAA, or SOC 2 controls.
- Use of external APIs should be reviewed against local data retention and privacy policy requirements.

---

## 19. Deployment Guide

### Local development

```bash
python -m venv VE
source VE/bin/activate
pip install -r requirements.txt
python train_model.py --synthetic-rows 6000
python web_app.py
```

On Windows PowerShell:

```powershell
.\VE\Scripts\Activate.ps1
python train_model.py --synthetic-rows 6000
python web_app.py
```

Open: http://127.0.0.1:8000

### Docker

```bash
docker build -t masar .
docker run --rm -p 8000:8000 --env-file .env masar
```

Or using Docker Compose:

```bash
docker compose up --build
```

The provided `Dockerfile` builds a Linux image and runs the web app on port 8000.

### Production guidance
- Use a real secret store instead of `.env` files.
- Add ingress authentication and a reverse proxy.
- Persist model artifacts in a managed artifact store.
- Add health checks and deployment rollback automation.
- Consider a proper service layer if the project expands beyond a single app server.

---

## 20. Testing

### Test suite overview
The repo contains unit tests and integration-style tests across the graph and web layer.

| Test file | Purpose |
| --- | --- |
| `tests/test_predictor.py` | model and fallback behavior |
| `tests/test_router.py` | fast vs deep routing |
| `tests/test_context.py` | context generation |
| `tests/test_optimizer.py` | route selection and corridor logic |
| `tests/test_synthesis.py` | bilingual synthesis |
| `tests/test_graph.py` | end-to-end graph execution |
| `tests/test_live_traffic.py` | live traffic normalization |
| `tests/test_web_app.py` | HTTP endpoint behavior |
| `tests/test_training_data.py` | data provenance and synthetic/real split |
| `tests/test_export_synthetic.py` | synthetic export logic |
| `tests/test_api_report.py` | redacted API call reporting |

### Running tests

```bash
python -m pytest -v
```

### Coverage expectations
The suite is designed to validate component-level behavior and graph-level flow. It is not a complete load or security suite and should be extended for production readiness.

---

## 21. Limitations

### Technical limitations
- No formal production authentication or user identity model.
- Model and route logic are intentionally simple and explainable rather than enterprise-grade.
- The route graph is not an official transportation topology and should not be treated as authoritative navigation geometry.
- Social and event signals are stubs rather than production integrators.
- No robust retries, exponential backoff, or queueing for provider calls.

### Performance bottlenecks
- LLM calls dominate latency in the deep path.
- Live provider calls add network latency to every recommendation.
- The current server is a single-threaded-style HTTP process with standard library utilities rather than an async ASGI app.

### Scaling limits
- No horizontal scaling logic.
- No DB connection pooling or request orchestration.
- No asynchronous queue for heavy tasks.

### Data limitations
- Real data may be sparse in early deployment states.
- The anomaly definition depends on throughput history and location-specific baselines.
- Location mapping is small and curated rather than a full city-wide transport model.

### Known assumptions
- Real-world commute analysis is approximated by a local graph and a heuristic route model.
- Weather and traffic are treated as the dominant factors, while broader social context remains intentionally limited.

---

## 22. Future Roadmap

### Short term
- add request validation and stronger API error handling,
- add retry logic for TomTom / OpenWeatherMap / Groq calls,
- add a formal `requirements.txt` freeze for reproducible deployments,
- expand tests for failure paths and rate-limit edge cases,
- add an optional `/api/recommend` auth layer.

### Medium term
- replace the corridor approximation with a richer route network,
- integrate proper events and social signals,
- support better model monitoring and drift analysis,
- add a lightweight model registry and versioning strategy,
- add a richer frontend dashboard with route comparison and time-series charts.

### Long term
- multi-agent planning with specialized expert nodes,
- better retrieval and memory systems for context-rich routing,
- distributed inference and asynchronous model serving,
- multimodal support for map imagery and route metadata,
- stronger MLOps workflow with CI/CD, experiment tracking, and artifact governance.

---

## 23. Facts, Assumptions, and Recommendations

### Facts
- The application is a local Python project built around a decision graph.
- It uses a trained XGBoost classifier and a fallback heuristic.
- It relies on Groq LLMs for context and synthesis.
- It reads live TomTom and OpenWeatherMap data when credentials are available.
- It exposes a simple HTTP API using the standard library.

### Assumptions
- The project is currently a prototype rather than a production-grade mobility platform.
- Real RTA routes and official transport feeds are not yet integrated.
- Social and events data are not fully wired to production services.
- The route graph is an approximation of urban corridors, not a full city network model.

### Recommendations
1. Add a persistent secret manager for production deployment.
2. Replace the synthetic or minimal graph with a verified city network and long-lived data pipeline.
3. Add formal MLOps tooling such as experiment tracking and model registry support.
4. Add stronger authentication and observability before any customer-facing launch.
5. Expand tests to cover failure modes, rate limits, and degraded network conditions.

---

## 24. Summary

Masar is a compact, explainable, Dubai-focused mobility intelligence system that combines a tabular anomaly model, graph-based route optimization, and LLM-based reasoning. It is structured for local experimentation, demo deployment, and extension into a more production-grade commuter intelligence platform. The repository is intentionally simple and testable, with a clear separation between traffic data, graph logic, training, and user-facing recommendation output.

The system is best understood as a working prototype for urban congestion and multimodal route guidance rather than as a complete operational transportation system. With additional integration work, stronger observability, and formal security controls, it can become a much more robust production product.

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
