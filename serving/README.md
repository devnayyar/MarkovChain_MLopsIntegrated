# Serving (Phase 7)

This folder contains the FastAPI application and helper utilities for serving Markov-chain predictions.

Quick start (local):

```bash
pip install -r requirements.txt
uvicorn serving.api.app:app --reload --port 8000
```

Endpoints:

- `GET /health` — health check
- `GET /current-regime` — returns latest regime from gold data
- `POST /predict-transition` — JSON body: `{ "current_state": "MODERATE_RISK" }`
- `POST /forecast-path` — JSON body: `{ "current_state": "MODERATE_RISK", "steps": 6 }`

Notes:

- The app builds a `MarkovChain` from `data/gold/markov_state_sequences.parquet` by default.
- If you register a model in MLflow as `markov_chain_model`, the API will attempt to use it via MLflow Model Registry.
- Inference requests are logged as MLflow runs under the `markov_chain_baseline` experiment.
