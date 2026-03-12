"""
Tests for POST /v2/calibrate/fit (hosted calibration endpoint).

Run:
    pytest tests/test_calibration_endpoint.py -v
"""

import base64
import pickle
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


# ── Minimal labeled runs fixture ───────────────────────────────────────────────

def make_runs(n: int = 40) -> list[dict]:
    steps = [
        {"thought": "Let me look this up.", "action_type": "Search",
         "action_arg": "query", "observation": "Found some info."},
        {"thought": "Done.", "action_type": "Finish",
         "action_arg": "answer", "observation": ""},
    ]
    runs = []
    for i in range(n):
        runs.append({
            "question": f"Question {i}?",
            "steps": steps,
            "final_answer": "answer",
            "correct": i % 3 != 0,  # 2/3 correct, 1/3 wrong
        })
    return runs


# ── Unit tests for calibration_service.fit_verifier() ────────────────────────

class TestFitVerifier:
    def test_returns_expected_keys(self):
        from qppg_service.calibration_service import fit_verifier
        result = fit_verifier(make_runs(40))
        assert "model_b64" in result
        assert "n_runs" in result
        assert "n_correct" in result
        assert "n_wrong" in result

    def test_model_b64_is_deserializable(self):
        from qppg_service.calibration_service import fit_verifier, load_verifier_from_b64
        result = fit_verifier(make_runs(40))
        verifier = load_verifier_from_b64(result["model_b64"])
        assert verifier is not None

    def test_auroc_between_zero_and_one_when_computed(self):
        from qppg_service.calibration_service import fit_verifier
        result = fit_verifier(make_runs(40))
        if result["auroc"] is not None:
            assert 0.0 <= result["auroc"] <= 1.0

    def test_counts_are_correct(self):
        from qppg_service.calibration_service import fit_verifier
        runs = make_runs(30)
        result = fit_verifier(runs)
        assert result["n_runs"] == 30
        assert result["n_correct"] + result["n_wrong"] == 30

    def test_too_few_raises_value_error(self):
        from qppg_service.calibration_service import fit_verifier
        with pytest.raises(ValueError, match="at least 5"):
            fit_verifier([{"question": "q", "steps": [], "correct": True}] * 3)

    def test_empty_raises_value_error(self):
        from qppg_service.calibration_service import fit_verifier
        with pytest.raises(ValueError):
            fit_verifier([])


# ── HTTP endpoint tests (using FastAPI TestClient) ────────────────────────────

@pytest.fixture(scope="module")
def client():
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient
    from qppg_service.server import create_app
    from qppg_service.service import QPPGService
    svc = QPPGService(domain_name="test_domain")
    app = create_app(service=svc)
    return TestClient(app)


class TestCalibrateFitEndpoint:
    def test_returns_200_with_valid_runs(self, client):
        runs = make_runs(40)
        resp = client.post("/v2/calibrate/fit", json={"labeled_runs": runs})
        assert resp.status_code == 200
        data = resp.json()
        assert "model_b64" in data
        assert data["n_runs"] == 40

    def test_returns_400_with_too_few_runs(self, client):
        runs = make_runs(3)
        resp = client.post("/v2/calibrate/fit", json={"labeled_runs": runs})
        assert resp.status_code == 400

    def test_returns_400_with_empty_runs(self, client):
        resp = client.post("/v2/calibrate/fit", json={"labeled_runs": []})
        assert resp.status_code == 400
