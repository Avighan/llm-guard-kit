"""Tests for domain-invariant LSTM step features in deep_verifier.py."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from llm_guard.deep_verifier import _extract_step_sequence

STEPS_GOOD = [
    {"thought": "Search for Einstein birthday.",
     "action_type": "Search", "action_arg": "Einstein birthday",
     "observation": "Albert Einstein was born on March 14, 1879 in Germany."},
    {"thought": "Einstein was born 1879.",
     "action_type": "Finish", "action_arg": "1879", "observation": ""},
]

STEPS_BAD_RETRIEVAL = [
    {"thought": "Search for Einstein birthday.",
     "action_type": "Search", "action_arg": "Einstein birthday",
     "observation": "No relevant results found."},
    {"thought": "I'll guess 1880.",
     "action_type": "Finish", "action_arg": "1880", "observation": ""},
]

def test_retrieval_conf_good():
    """Good retrieval: action_arg overlaps with observation."""
    seq, n = _extract_step_sequence(STEPS_GOOD, "1879")
    # Feature 0 = retrieval_conf; "Einstein birthday" overlaps with the observation
    # Jaccard("Einstein birthday", obs) = 1/11 ≈ 0.091 — above zero, showing retrieval hit
    assert seq[0, 0] > 0.05, f"Expected retrieval_conf > 0.05, got {seq[0,0]:.3f}"

def test_retrieval_conf_bad():
    """Bad retrieval: "Einstein birthday" has no overlap with "No relevant results found"."""
    seq, n = _extract_step_sequence(STEPS_BAD_RETRIEVAL, "1880")
    assert seq[0, 0] < 0.1, f"Expected retrieval_conf < 0.1, got {seq[0,0]:.3f}"

def test_semantic_gap_first_step_neutral():
    """No previous thought at step 0 → neutral 0.5."""
    seq, n = _extract_step_sequence(STEPS_GOOD, "1879")
    assert abs(seq[0, 1] - 0.5) < 1e-5, f"Expected 0.5 at step 0, got {seq[0,1]:.3f}"

def test_semantic_gap_divergent_steps():
    """Steps with very different thoughts → high semantic gap."""
    steps = [
        {"thought": "I need to find Einstein birthday.", "action_type": "Search",
         "action_arg": "Einstein birthday", "observation": "Born 1879."},
        {"thought": "Napoleon conquered Europe and changed history forever.", "action_type": "Search",
         "action_arg": "Napoleon Europe", "observation": "Napoleon lived 1769-1821."},
    ]
    seq, n = _extract_step_sequence(steps, "unknown")
    # High divergence between step 0 and step 1 thoughts
    assert seq[1, 1] > 0.3, f"Expected semantic_gap > 0.3 at step 1, got {seq[1,1]:.3f}"

def test_feature_shape():
    seq, n = _extract_step_sequence(STEPS_GOOD, "1879")
    assert seq.shape == (8, 6), f"Expected (8,6), got {seq.shape}"
    assert n == 2
