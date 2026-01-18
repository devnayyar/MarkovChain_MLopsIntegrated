from fastapi.testclient import TestClient
from serving.api.app import app


client = TestClient(app)


def test_health():
    r = client.get('/health')
    assert r.status_code == 200
    assert r.json() == {'status': 'ok'}


def test_current_regime_or_404():
    r = client.get('/current-regime')
    # Either returns 200 with current_regime or 404 if gold data missing
    assert r.status_code in (200, 404)


def test_predict_transition_bad_state():
    r = client.post('/predict-transition', json={'current_state': 'UNKNOWN_STATE'})
    assert r.status_code in (400, 404)
