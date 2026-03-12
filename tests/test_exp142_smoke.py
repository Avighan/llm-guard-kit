"""
Smoke test for exp142: P(True) + Structural Verifier on exp141 chains.
Run with: pytest tests/test_exp142_smoke.py -v
"""

import json
import pytest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_summary_schema():
    """exp142 summary JSON has expected keys."""
    summary_path = ROOT / "results" / "exp142_summary.json"
    if not summary_path.exists():
        pytest.skip("exp142 not run yet")
    data = json.loads(summary_path.read_text())
    assert "results" in data
    assert "best_ptrue_ensemble" in data
    assert "confirmed" in data
    assert isinstance(data["confirmed"], bool)
    assert isinstance(data["best_ptrue_ensemble"], float)
    for r in data["results"]:
        for key in ("domain", "n", "n_wrong", "auroc_beh", "auroc_ptrue",
                    "auroc_ensemble", "auroc_structural"):
            assert key in r
