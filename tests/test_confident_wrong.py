"""Tests for confident_wrong failure mode detection and adapter routing."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from llm_guard.adapter_registry import AdapterRegistry

SHORT_STEPS = [
    {"thought": "I know the answer.", "action_type": "Search",
     "action_arg": "Eiffel Tower height", "observation": "The Eiffel Tower is 330m tall."},
    {"thought": "330m.", "action_type": "Finish", "action_arg": "330m", "observation": ""},
]

def test_confident_wrong_adapter_registered():
    registry = AdapterRegistry()
    assert "confident_wrong" in registry.list_adapters()

def test_confident_wrong_adapter_routes():
    registry = AdapterRegistry()
    result = registry.get("confident_wrong")
    assert result.activated
    assert not result.fallback
    assert result.config.search_strategy == "verify_claim"

def test_confident_wrong_detection():
    from llm_guard.agent_guard import AgentGuard
    guard = AgentGuard()
    mode = guard._detect_failure_mode(SHORT_STEPS, "330m")
    assert mode == "confident_wrong", f"Expected confident_wrong, got {mode!r}"

def test_non_confident_wrong_hedging():
    """Chains with hedging should NOT be flagged as confident_wrong."""
    from llm_guard.agent_guard import AgentGuard
    guard = AgentGuard()
    hedging_steps = [
        {"thought": "I'm not sure, might be wrong here.", "action_type": "Search",
         "action_arg": "query", "observation": "Some result."},
        {"thought": "Probably this answer.", "action_type": "Finish",
         "action_arg": "answer", "observation": ""},
    ]
    mode = guard._detect_failure_mode(hedging_steps, "answer")
    assert mode != "confident_wrong"
