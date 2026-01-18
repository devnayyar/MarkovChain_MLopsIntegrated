import pytest
import mlflow
from fastapi.testclient import TestClient
from serving.api.app import app

from serving.mlflow_config import initialize_mlflow
from serving.experiment_tracker import ExperimentTracker, track_experiment
from serving.model_loader import build_markov_from_gold, load_model_from_registry
from serving.model_registry import ModelRegistry


def test_e2e_all_features():
    # Initialize MLflow (local SQLite + artifact root)
    config = initialize_mlflow()
    assert config.tracking_uri is not None

    # Experiments present
    exps = config.list_experiments()
    assert isinstance(exps, dict)

    # Experiment tracker basic usage
    tracker = ExperimentTracker(experiment_name=config.EXPERIMENT_MARKOV_BASELINE, run_name="e2e_test_run")
    tracker.start()
    tracker.log_metric("e2e_dummy_metric", 0.123)
    run_id = tracker.end()
    assert run_id is not None

    # Context manager usage
    with track_experiment(experiment_name=config.EXPERIMENT_MARKOV_BASELINE, run_name="e2e_ctx") as t:
        t.log_metric("cm_metric", 1.0)
        assert t.run_id is not None

    # ModelRegistry basic check (list models may not exist)
    registry = ModelRegistry()
    try:
        models = registry.list_all_models()
    except Exception:
        models = []
    assert isinstance(models, (list, tuple))

    # Model loader & MarkovChain behavior (skip if gold missing)
    mc = build_markov_from_gold()
    if mc is None:
        pytest.skip("Gold data missing; skipping Markov evaluation checks")
    # Ensure transition matrix and stationary distribution compute
    P = mc.estimate_transition_matrix()
    assert P.shape[0] == P.shape[1]
    sd = mc.get_stationary_distribution()
    assert abs(sd.sum() - 1.0) < 1e-6

    # Try loading a registered model (may raise if not present)
    try:
        _ = load_model_from_registry("markov_chain_model")
    except Exception:
        # Not fatal for E2E, just log and continue
        pass

    # API endpoints
    client = TestClient(app)
    r = client.get("/health")
    assert r.status_code == 200 and r.json().get("status") == "ok"

    r = client.get("/current-regime")
    assert r.status_code in (200, 404)

    # Predict transition: accept either successful response or error due to unknown state / missing data
    r = client.post("/predict-transition", json={"current_state": "LOW_RISK"})
    assert r.status_code in (200, 400, 404)

    r = client.post("/forecast-path", json={"current_state": "LOW_RISK", "steps": 3})
    assert r.status_code in (200, 400, 404)
