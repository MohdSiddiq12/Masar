# Masar (مسار) — Dubai Congestion Prediction & Commute Recommendation

> **Live AI-powered mobility intelligence platform for Dubai using live traffic, weather, machine learning, graph routing, and LLM-assisted reasoning.**

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Masar-brightgreen)](https://masar-ai.duckdns.org/)
[![Python](https://img.shields.io/badge/Python-3.11-blue)](https://www.python.org/)
[![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED)](https://www.docker.com/)
[![AWS](https://img.shields.io/badge/AWS-EC2-orange)](https://aws.amazon.com/ec2/)
[![HTTPS](https://img.shields.io/badge/HTTPS-Let's%20Encrypt-success)](https://letsencrypt.org/)

## 🚀 Live Demo

**Production:**
https://masar-ai.duckdns.org/

Masar is deployed on **AWS EC2** and exposed through **Nginx with HTTPS**, using a **DuckDNS domain** and an automatically renewed **Let's Encrypt TLS certificate**.

---

# 📌 Overview

Masar (مسار — *path / route*) is a Dubai-focused AI mobility intelligence platform that analyzes current traffic and weather conditions to predict congestion and recommend an appropriate commute mode.

Instead of acting as a conventional navigation application, Masar focuses on the **decision layer**:

* Is the current traffic condition significantly congested?
* Should the commuter drive, use the metro, or combine driving with metro?
* Which route is preferable under the current congestion state?
* When the prediction is uncertain, what additional contextual reasoning should be considered?

The system combines **machine learning, graph algorithms, LangGraph orchestration, live external APIs, and LLM reasoning** into a single deployed application.

---

# 🎯 Problem

Dubai commuters often need to make a mode-selection decision based on rapidly changing road conditions.

Traditional route applications primarily answer:

> "Where should I go?"

Masar focuses on:

> **"Given the current conditions, what is the better way to travel?"**

The platform therefore combines congestion prediction, route optimization, live traffic, weather context, and AI-assisted reasoning to produce a more actionable commute recommendation.

---

# 🚦 Key Features

* **Live congestion prediction** using a deployed XGBoost model
* **Live traffic ingestion** through TomTom Traffic API
* **Live weather ingestion** through OpenWeatherMap API
* **Confidence-based LangGraph routing**
* **LLM-assisted context enrichment** for uncertain cases
* **Graph-based route optimization** using NetworkX
* **Commute mode recommendation**

  * Drive
  * Drive to Metro
  * Metro
* **Bilingual recommendations**

  * English
  * Arabic
* **Supabase/PostgreSQL integration** for traffic data access
* **REST-style HTTP API**
* **Dockerized production deployment**
* **Nginx reverse proxy**
* **DuckDNS custom domain**
* **Let's Encrypt HTTPS**
* **Automatic TLS certificate renewal**
* **SSH key-based server administration**

---

# 🧠 AI Workflow

Masar uses a stateful **LangGraph** workflow to coordinate the recommendation process.

```text
User Request
     │
     ▼
Live Traffic / Weather Context
     │
     ▼
Feature Preparation
     │
     ▼
Congestion Predictor
     │
     ▼
Confidence Router
     │
     ├─────────────── High Confidence ───────────────┐
     │                                                │
     │                                                ▼
     │                                         Route Optimizer
     │                                                │
     └────── Low Confidence ──────► Context Agent    │
                                    │                 │
                                    ▼                 │
                              Groq LLM Reasoning      │
                                    │                 │
                                    └────────┬────────┘
                                             ▼
                                      Route Optimizer
                                             │
                                             ▼
                                         Synthesis
                                             │
                                  ┌──────────┴──────────┐
                                  ▼                     ▼
                               English                Arabic
```

## Core AI Components

### Predictor

Uses **XGBoost** to estimate congestion and prediction confidence from traffic and contextual features.

### Router

Uses **LangGraph conditional routing** to determine whether a request can proceed directly or requires additional reasoning.

### Context Agent

Uses **Groq LLM inference** to enrich low-confidence predictions with contextual reasoning.

### Route Optimizer

Uses **NetworkX graph algorithms** to evaluate available Dubai routes and commute options.

### Synthesis

Produces a concise recommendation in **English and Arabic**.

---

# 🌐 Live Data Pipeline

Production inference is based on **live runtime data**.

Synthetic data generation is no longer part of the production workflow and is intentionally excluded from the current deployment architecture.

```text
TomTom Traffic API ──────┐
                         │
                         ▼
                   Live Data Layer
                         │
OpenWeatherMap API ──────┤
                         │
                         ▼
                  Feature Processing
                         │
                         ▼
                  XGBoost Inference
                         │
                         ▼
                 LangGraph Workflow
                         │
                         ▼
                Route + Mode Decision
                         │
                         ▼
                  Groq Reasoning
                         │
                         ▼
              English + Arabic Output
```

## External Data Sources

| Service                   | Purpose                                        |
| ------------------------- | ---------------------------------------------- |
| **TomTom Traffic API**    | Current traffic speed and free-flow speed      |
| **OpenWeatherMap API**    | Current weather and rainfall conditions        |
| **Supabase / PostgreSQL** | Traffic data persistence and retrieval         |
| **Groq API**              | LLM-powered contextual reasoning and synthesis |

---

# 🏗️ Production Architecture

```text
                         ┌───────────────────┐
                         │   User / Browser  │
                         └─────────┬─────────┘
                                   │
                              HTTPS :443
                                   │
                                   ▼
                      ┌───────────────────────┐
                      │       DuckDNS         │
                      │ masar-ai.duckdns.org  │
                      └───────────┬───────────┘
                                  │
                                  ▼
                      ┌───────────────────────┐
                      │      AWS EC2          │
                      │    t3.micro           │
                      │   Ubuntu 26.04 LTS    │
                      └───────────┬───────────┘
                                  │
                                  ▼
                      ┌───────────────────────┐
                      │       Nginx           │
                      │ Reverse Proxy + TLS   │
                      └───────────┬───────────┘
                                  │
                           proxy → :8000
                                  │
                                  ▼
                      ┌───────────────────────┐
                      │   Docker Container    │
                      │    Masar Application  │
                      └───────────┬───────────┘
                                  │
                    ┌─────────────┼──────────────┐
                    ▼             ▼              ▼
                XGBoost       LangGraph       NetworkX
                    │             │              │
                    └─────────────┼──────────────┘
                                  │
                                  ▼
                               Groq
                                  │
              ┌───────────────────┼──────────────────┐
              ▼                   ▼                  ▼
           TomTom          OpenWeatherMap         Supabase
```

---

# ☁️ AWS Deployment

## Infrastructure

| Component         | Configuration             |
| ----------------- | ------------------------- |
| Cloud Provider    | AWS                       |
| Region            | `us-east-1` (N. Virginia) |
| Compute           | EC2 `t3.micro`            |
| Operating System  | Ubuntu 26.04 LTS          |
| Storage           | 8 GB gp3 EBS              |
| Container Runtime | Docker                    |
| Reverse Proxy     | Nginx                     |
| Application Port  | `8000`                    |
| HTTPS Port        | `443`                     |
| HTTP Port         | `80`                      |
| SSH Port          | `22`                      |

## Security Group

Current inbound access:

| Port  | Protocol | Purpose                       |
| ----- | -------- | ----------------------------- |
| `22`  | TCP      | SSH administration            |
| `80`  | TCP      | HTTP / certificate validation |
| `443` | TCP      | HTTPS application traffic     |

---

# 🔐 HTTPS / TLS

Masar is secured using **Let's Encrypt** with **Certbot**.

### Certificate

```text
Domain:
masar-ai.duckdns.org

Certificate Authority:
Let's Encrypt
```

### Certificate location

```text
/etc/letsencrypt/live/masar-ai.duckdns.org/
├── fullchain.pem
└── privkey.pem
```

### Certificate issuance

```bash
sudo certbot --nginx -d masar-ai.duckdns.org
```

Certbot configures Nginx and enables HTTPS for the domain.

### Renewal validation

```bash
sudo certbot renew --dry-run
```

Certbot has also configured automatic certificate renewal.

---

# 🌍 DNS

Masar uses **DuckDNS** for the production hostname.

```text
masar-ai.duckdns.org
        │
        ▼
32.199.176.101
        │
        ▼
AWS EC2
```

### Current domain

**https://masar-ai.duckdns.org**

DuckDNS was selected because it provides a free DNS hostname suitable for a small AWS-hosted portfolio/experimental deployment.

> Note: The EC2 public IPv4 address can change when the instance is stopped and restarted unless an Elastic IP is used. The DuckDNS record therefore needs to reflect the current EC2 public IP.

---

# 🐳 Docker Deployment

Masar is packaged as a Docker application.

## Build

```bash
docker build -t masar .
```

## Run

```bash
docker run --rm -p 8000:8000 --env-file .env masar
```

## Docker Compose

```bash
docker compose up -d --build
```

The container exposes the Masar service on:

```text
0.0.0.0:8000
```

Nginx receives public HTTPS traffic and proxies requests to the container.

---

# 🔁 Nginx Reverse Proxy

Current production routing:

```text
Internet
   │
 HTTPS :443
   │
   ▼
 Nginx
   │
   ▼
127.0.0.1:8000
   │
   ▼
Masar Docker Container
```

Example configuration:

```nginx
server {
    listen 80;
    server_name masar-ai.duckdns.org;

    location / {
        proxy_pass http://127.0.0.1:8000;

        proxy_http_version 1.1;

        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }
}
```

Certbot extends the Nginx configuration to support HTTPS and HTTP → HTTPS redirection.

---

# 🔌 API

Masar exposes a lightweight HTTP API from `web_app.py`.

## Health

```http
GET /health
```

Example:

```json
{
  "status": "ok"
}
```

## Locations

```http
GET /api/locations
```

Returns supported Dubai locations and coordinates.

## Traffic

```http
GET /api/traffic
```

Returns available traffic data.

## Latest traffic

```http
GET /api/traffic/latest?location=Marina
```

Returns the latest traffic state for a specified location.

## Recommendation

```http
POST /api/recommend
```

Example request:

```json
{
  "origin": "Marina",
  "destination": "Business Bay",
  "current_speed": 25,
  "free_flow_speed": 90,
  "temperature": 36,
  "weather_condition": "Clear",
  "is_raining": false,
  "use_live_traffic": true
}
```

The recommendation pipeline returns prediction, route, mode, and bilingual guidance.

---

# 🧩 Technology Stack

## Programming

* Python 3.11
* SQL
* HTML
* CSS
* JavaScript

## AI / Machine Learning

* XGBoost
* LangGraph
* LangChain
* Groq
* LLM inference
* Pydantic
* NumPy
* Pandas
* Joblib
* Scikit-learn

## Graph / Optimization

* NetworkX
* Dijkstra shortest-path routing

## Data

* Supabase
* PostgreSQL
* TomTom Traffic API
* OpenWeatherMap API

## Backend

* Python HTTP server
* REST-style API
* JSON
* `httpx`

## Infrastructure / DevOps

* AWS EC2
* Docker
* Docker Compose
* Nginx
* DuckDNS
* Let's Encrypt
* Certbot
* SSH / Ed25519

## Development

* Git
* GitHub
* GitHub Actions
* VS Code
* Pytest

---

# 📁 Repository Structure

```text
Masar/
│
├── masar/
│   ├── nodes/
│   │   ├── predictor.py
│   │   ├── router.py
│   │   ├── context.py
│   │   ├── optimizer.py
│   │   └── synthesis.py
│   │
│   ├── graph.py
│   ├── state.py
│   ├── features.py
│   ├── locations.py
│   ├── live_data.py
│   ├── live_traffic.py
│   └── api_report.py
│
├── frontend/
├── models/
├── tests/
├── scripts/
├── docs/
│
├── Dockerfile
├── docker-compose.yaml
├── requirements.txt
├── web_app.py
├── README.md
└── LICENSE
```

---

# ⚙️ Local Development

## Clone

```bash
git clone https://github.com/MohdSiddiq12/Masar.git
cd Masar
```

## Create environment

### Windows

```powershell
python -m venv VE
.\VE\Scripts\Activate.ps1
```

### Linux / macOS

```bash
python3 -m venv VE
source VE/bin/activate
```

## Install dependencies

```bash
pip install -r requirements.txt
```

## Configure environment

Create a `.env` file:

```env
SUPABASE_URL=...
SUPABASE_KEY=...
TOMTOM_API_KEY=...
OPENWEATHER_API_KEY=...
GROQ_API_KEY=...
GROQ_MODEL=...
MASAR_MODEL_PATH=models/congestion_xgb.pkl
PORT=8000
```

Never commit `.env` or API credentials to GitHub.

## Run

```bash
python web_app.py
```

Open:

```text
http://127.0.0.1:8000
```

---

# 🔄 Deployment Workflow

The current production deployment follows a Git + Docker workflow.

```text
Developer
    │
    ▼
GitHub Repository
    │
    ▼
Pull latest code on EC2
    │
    ▼
Docker Build
    │
    ▼
Docker Container
    │
    ▼
Nginx
    │
    ▼
HTTPS
    │
    ▼
Live Masar Application
```

Automated CI/CD deployment is a future enhancement rather than part of the current production deployment path.

---

# 📊 Runtime Flow

```text
1. User opens Masar
        │
        ▼
2. DuckDNS resolves domain
        │
        ▼
3. HTTPS request reaches EC2
        │
        ▼
4. Nginx terminates TLS
        │
        ▼
5. Request forwarded to Docker :8000
        │
        ▼
6. web_app.py validates request
        │
        ▼
7. Live traffic/weather data retrieved
        │
        ▼
8. MasarState constructed
        │
        ▼
9. LangGraph workflow starts
        │
        ▼
10. XGBoost predicts congestion
        │
        ▼
11. Confidence router selects path
        │
        ├── Fast path
        │
        └── Deep path → Groq context reasoning
        │
        ▼
12. NetworkX route optimization
        │
        ▼
13. Final bilingual synthesis
        │
        ▼
14. JSON/UI response returned
        │
        ▼
15. User receives commute recommendation
```

---

# 🧪 Testing & Validation

The application includes component and integration tests using **Pytest**.

Typical validation commands:

```bash
pytest
```

Infrastructure validation:

```bash
sudo nginx -t
```

Service validation:

```bash
docker ps
sudo systemctl status nginx
```

HTTPS validation:

```bash
curl -I https://masar-ai.duckdns.org
```

Certificate renewal validation:

```bash
sudo certbot renew --dry-run
```

System validation:

```bash
df -h
free -h
```

---

# 🔐 Security

Current security controls include:

* HTTPS with Let's Encrypt
* Automatic certificate renewal
* SSH Ed25519 key authentication
* Nginx reverse proxy
* AWS Security Group controls
* No API secrets committed to Git
* Environment-based credentials
* Redacted API reporting
* Containerized application runtime

Production secrets should remain outside the repository and be injected through environment configuration or a managed secret store.

---

# 📈 Current Production Status

| Component                   | Status |
| --------------------------- | ------ |
| AWS EC2                     | ✅ Live |
| Docker                      | ✅ Live |
| Masar application           | ✅ Live |
| Live traffic ingestion      | ✅      |
| Live weather ingestion      | ✅      |
| XGBoost inference           | ✅      |
| LangGraph workflow          | ✅      |
| NetworkX routing            | ✅      |
| Groq reasoning              | ✅      |
| Supabase integration        | ✅      |
| Nginx reverse proxy         | ✅      |
| DuckDNS                     | ✅      |
| HTTPS / TLS                 | ✅      |
| Let's Encrypt auto-renewal  | ✅      |
| SSH key-based access        | ✅      |
| Centralized monitoring      | 🚧     |
| Automated CI/CD deployment  | 🚧     |
| Auto scaling                | ❌      |
| Multi-region infrastructure | ❌      |

---

# ⚠️ Current Scope & Limitations

Masar is a **production-deployed MVP**, not a full-scale navigation platform.

Current limitations include:

* No turn-by-turn navigation
* No official RTA integration
* Single EC2 instance
* No automatic horizontal scaling
* No load balancer
* No centralized observability stack
* No production authentication/RBAC
* No multi-region deployment
* External API availability and rate limits affect runtime behavior
* EC2 public IP changes require DNS adjustment unless an Elastic IP is introduced

The model provides **decision support**, not guaranteed travel-time or navigation accuracy.

---

# 🗺️ Roadmap

## Short Term

* GitHub Actions CI/CD
* Automated Docker deployment
* Application health checks
* CloudWatch monitoring
* Centralized application logs
* Improved API retry/backoff handling

## Medium Term

* AWS Elastic IP or managed load balancing
* CloudWatch dashboards and alarms
* Prometheus / Grafana observability
* Improved model evaluation using accumulated real-world observations
* Authentication and API access control

## Long Term

* ECS deployment
* Application Load Balancer
* Horizontal scaling
* Multi-AZ architecture
* Advanced route intelligence
* Real-time event integration
* Production-grade mobility intelligence platform

---

# 📚 Documentation

Detailed engineering documentation covers:

* System architecture
* Application workflow
* LangGraph components
* Live data integrations
* API contracts
* Docker deployment
* AWS infrastructure
* Nginx configuration
* DNS
* TLS/SSL
* Security
* Testing
* Operations
* Troubleshooting
* Roadmap

**Technical Documentation:**
See [Full Documentation](Masar_Technical_Documentation.pdf) for detailed architecture, module breakdowns, and training instructions.
---

# 🌐 Production Endpoint

### Masar

**https://masar-ai.duckdns.org/**

### Repository

**https://github.com/MohdSiddiq12/Masar**

---

# 📜 License

Apache License 2.0
