# Masar (مسار) — Dubai Congestion Prediction & Commute Recommendation

## 📌 Overview
Masar is a prototype AI-driven urban mobility system for Dubai. It predicts congestion, recommends commute modes (drive, metro, drive-to-metro), and provides bilingual English/Arabic guidance. Built for experimentation, demo use, and as a foundation for production-ready travel intelligence.

## 🚦 Key Features
- **Congestion prediction** using XGBoost  
- **Dynamic graph routing** with fallback deep reasoning  
- **Context enrichment** for low-confidence forecasts  
- **Route optimization** with NetworkX  
- **Bilingual synthesis** (English + Arabic)  
- **Web demo app** with traffic integration  
- **Synthetic data generation** for training  

## 🛠 Tech Stack
- Python 3.11  
- LangGraph, LangChain-Groq  
- XGBoost, NetworkX  
- Supabase, TomTom, OpenWeatherMap  
- Local HTTP server (no FastAPI)  

## 📂 Repository Structure
```
Masar/
├── masar/           # Core package (graph, state, nodes, live data)
├── scripts/         # Automation (traffic collection, training, benchmarks)
├── tests/           # Unit & integration tests
├── frontend/        # Web UI
├── models/          # Trained model artifacts
├── data/            # Local datasets
├── docs/            # Extended documentation
```

## ⚙️ Setup
1. Clone repo & create virtual environment  
2. Install dependencies:  
   ```bash
   pip install -r requirements.txt
   ```
3. Configure `.env` with API keys (TomTom, OpenWeatherMap, Supabase, Groq)  
4. Run demo server:  
   ```bash
   python web_app.py
   ```

## 🔑 Environment Variables
- `SUPABASE_URL`, `SUPABASE_KEY` → Traffic DB  
- `TOMTOM_API_KEY`, `OPENWEATHER_API_KEY` → Live traffic/weather  
- `GROQ_API_KEY` → LLM reasoning  
- `MASAR_MODEL_PATH` → Trained model artifact  

## 📊 Workflow
1. **Predictor** → congestion score + confidence  
2. **Router** → fast vs deep path  
3. **Context Agent** (deep path only) → enrich reasoning  
4. **Optimizer** → route + mode selection  
5. **Synthesis** → bilingual recommendation  

## 🚫 Out of Scope
- Full turn-by-turn navigation  
- Official RTA integration  
- Production-grade authentication & deployment  

## 📖 Documentation
See [Full Documentation](Masar_Technical_Documentation.pdf) for detailed architecture, module breakdowns, and training instructions.