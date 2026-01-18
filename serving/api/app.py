from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import logging
from pathlib import Path

from serving.model_loader import build_markov_from_gold, load_model_from_registry
from serving.experiment_tracker import track_experiment

logger = logging.getLogger(__name__)

app = FastAPI(title="FINML Markov Serving", version="0.1")


class PredictRequest(BaseModel):
    current_state: str
    steps: Optional[int] = 1


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/current-regime")
def current_regime():
    # Return the most recent regime from gold data if available
    with track_experiment(experiment_name="markov_chain_baseline", run_name="api_current_regime") as tracker:
        mc = build_markov_from_gold()
        tracker.log_param("endpoint", "current_regime")
    if mc is None:
        raise HTTPException(status_code=404, detail="Gold regime data not available")
    # Last observed state label
    last_idx = mc.state_sequence[-1]
    state_label = mc.int_to_state.get(int(last_idx), str(last_idx))
    return {"current_regime": state_label}


@app.post("/predict-transition")
def predict_transition(req: PredictRequest):
    # Wrap in MLflow experiment for inference tracking
    with track_experiment(experiment_name="markov_chain_baseline", run_name="api_predict_transition") as tracker:
        tracker.log_param("endpoint", "predict_transition")
        tracker.log_param("requested_state", req.current_state)

        # Prefer a registered model if available (optional)
        try:
            model = load_model_from_registry("markov_chain_model")
        except Exception:
            model = None

        if model is not None:
            # Try to use pyfunc predict if available
            try:
                preds = model.predict([{"current_state": req.current_state}])
                tracker.log_metric("served_via_registry", 1)
                return {"current_state": req.current_state, "next_state_probs": preds}
            except Exception:
                tracker.log_metric("served_via_registry", 0)

        mc = build_markov_from_gold()
        if mc is None:
            raise HTTPException(status_code=404, detail="Gold regime data not available")

        # Ensure transition matrix is estimated
        if not hasattr(mc, 'transition_matrix'):
            mc.estimate_transition_matrix()

        if req.current_state not in mc.state_to_int:
            raise HTTPException(status_code=400, detail=f"Unknown state: {req.current_state}")

        probs = mc.predict_next_state(req.current_state)
        tracker.log_metric("num_next_states", len(probs))
        return {"current_state": req.current_state, "next_state_probs": probs}


@app.post("/forecast-path")
def forecast_path(req: PredictRequest):
    with track_experiment(experiment_name="markov_chain_baseline", run_name="api_forecast_path") as tracker:
        tracker.log_param("endpoint", "forecast_path")
        tracker.log_param("requested_state", req.current_state)

        mc = build_markov_from_gold()
        if mc is None:
            raise HTTPException(status_code=404, detail="Gold regime data not available")

        if not hasattr(mc, 'transition_matrix'):
            mc.estimate_transition_matrix()

        if req.current_state not in mc.state_to_int:
            raise HTTPException(status_code=400, detail=f"Unknown state: {req.current_state}")

        forecast = mc.forecast_path(req.current_state, steps=req.steps or 1)
        tracker.log_metric("forecast_steps", req.steps or 1)
        return {"forecast": forecast}
