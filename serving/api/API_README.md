# FastAPI Serving for FINML

This folder contains a minimal FastAPI application exposing Markov-chain based endpoints.

Endpoints:
- `GET /health` — health check
- `GET /current-regime` — returns the most recent regime from gold data
- `POST /predict-transition` — body: `{ "current_state": "MODERATE_RISK" }` → next-state probabilities
- `POST /forecast-path` — body: `{ "current_state": "MODERATE_RISK", "steps": 6 }` → multi-step forecast

How it works:
- The app reads `data/gold/markov_state_sequences.parquet` and builds a `MarkovChain` using `serving.model_loader.build_markov_from_gold()`.
- Predictions are produced using the `MarkovChain` methods already implemented in `modeling/models/base_markov.py`.

Run locally (from project root):

```bash
pip install -r requirements.txt
uvicorn serving.api.app:app --reload --port 8000
```
