"""Tests for 3 answer-side domain-invariant features in LocalVerifier."""
import sys, numpy as np
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from llm_guard.local_verifier import FEATURE_NAMES, extract_features

GOOD_RUN_STEPS = [
    {"thought": "Search for Apple founder.", "action_type": "Search",
     "action_arg": "Apple Inc founder",
     "observation": "Apple Inc was co-founded by Steve Jobs, Steve Wozniak, and Ronald Wayne in 1976."},
    {"thought": "Steve Jobs.", "action_type": "Finish", "action_arg": "Steve Jobs", "observation": ""},
]
BAD_RUN_STEPS = [
    {"thought": "Search for Apple.", "action_type": "Search",
     "action_arg": "Apple",
     "observation": "Apple is a fruit. No relevant information about technology."},
    {"thought": "Probably Tim Cook.", "action_type": "Finish",
     "action_arg": "Tim Cook", "observation": ""},
]

def test_new_features_in_feature_names():
    assert "ans_entity_match"    in FEATURE_NAMES
    assert "ans_len_type_match"  in FEATURE_NAMES
    assert "obs_entity_coverage" in FEATURE_NAMES

def test_feature_vector_length():
    feats = extract_features("Who founded Apple Inc?", GOOD_RUN_STEPS, "Steve Jobs")
    assert len(feats) == len(FEATURE_NAMES)

def test_ans_entity_match_good():
    feats = extract_features("Who founded Apple Inc?", GOOD_RUN_STEPS, "Steve Jobs")
    idx = FEATURE_NAMES.index("ans_entity_match")
    # "Steve" and "Jobs" both appear in the observation → high match
    assert feats[idx] > 0.5, f"Expected > 0.5, got {feats[idx]:.3f}"

def test_ans_entity_match_bad():
    feats = extract_features("Who founded Apple Inc?", BAD_RUN_STEPS, "Tim Cook")
    idx = FEATURE_NAMES.index("ans_entity_match")
    # "Tim" and "Cook" do NOT appear in the observation → low match
    assert feats[idx] < 0.5, f"Expected < 0.5, got {feats[idx]:.3f}"

def test_obs_entity_coverage_good():
    feats = extract_features("Who founded Apple Inc?", GOOD_RUN_STEPS, "Steve Jobs")
    idx = FEATURE_NAMES.index("obs_entity_coverage")
    # "Apple" and "Inc" appear in observation → coverage > 0
    assert feats[idx] > 0.0, f"Expected > 0, got {feats[idx]:.3f}"


def test_ans_len_type_match_suspicious():
    """Factoid question + long answer (>8 words) → 1.0 (suspicious)."""
    long_answer = "The founder was a very important person who created many great things"
    feats = extract_features("Who founded Apple Inc?", GOOD_RUN_STEPS, long_answer)
    idx = FEATURE_NAMES.index("ans_len_type_match")
    assert feats[idx] == 1.0, f"Expected 1.0 (suspicious), got {feats[idx]:.3f}"


def test_ans_len_type_match_normal():
    """Factoid question + short answer (≤8 words) → 0.0 (normal)."""
    feats = extract_features("Who founded Apple Inc?", GOOD_RUN_STEPS, "Steve Jobs")
    idx = FEATURE_NAMES.index("ans_len_type_match")
    assert feats[idx] == 0.0, f"Expected 0.0 (normal), got {feats[idx]:.3f}"
