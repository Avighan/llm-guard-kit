"""
Tests for FARL Phase 2: VictimPool, HunterRewardTracker, _diversity_score, FARLCycleRunner
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from unittest.mock import patch, MagicMock
from scripts.farl_phase2 import (
    VictimPool, HunterRewardTracker, FARLCycleRunner, _diversity_score,
)
from scripts.farl_hunt import FAILURE_MODES, DOMAIN_ROTATION


# ── _diversity_score ──────────────────────────────────────────────────────────

def test_victim_diversity_score_all_fail():
    """3/3 victims fail → score 1.0"""
    results = [
        {"is_failure": True},
        {"is_failure": True},
        {"is_failure": True},
    ]
    assert _diversity_score(results) == 1.0


def test_victim_diversity_score_gray_zone():
    """2/3 victims fail → score 0.67"""
    results = [
        {"is_failure": True},
        {"is_failure": True},
        {"is_failure": False},
    ]
    assert abs(_diversity_score(results) - 2 / 3) < 0.01


def test_victim_diversity_score_none_fail():
    """0/3 fail → score 0.0"""
    results = [{"is_failure": False}] * 3
    assert _diversity_score(results) == 0.0


def test_victim_diversity_score_one_of_three():
    """1/3 fail → score ~0.33"""
    results = [
        {"is_failure": True},
        {"is_failure": False},
        {"is_failure": False},
    ]
    assert abs(_diversity_score(results) - 1 / 3) < 0.01


def test_victim_diversity_score_empty():
    """Empty list → score 0.0"""
    assert _diversity_score([]) == 0.0


# ── VictimPool ────────────────────────────────────────────────────────────────

def test_victim_pool_has_three_victims():
    pool = VictimPool.__new__(VictimPool)
    pool.victims = [("A", "prompt A", None), ("B", "prompt B", None), ("C", "prompt C", None)]
    assert len(pool.victims) == 3


def test_victim_pool_configs_count():
    """VictimPool.VICTIM_CONFIGS has exactly 3 entries."""
    assert len(VictimPool.VICTIM_CONFIGS) == 3


def test_victim_pool_config_names():
    """Standard, confident, cautious victim names are present."""
    names = [name for name, _ in VictimPool.VICTIM_CONFIGS]
    assert "standard" in names
    assert "confident" in names
    assert "cautious" in names


def test_victim_pool_standard_has_none_system():
    """Standard victim uses None as system prompt (falls back to REACT_SYSTEM)."""
    standard_entry = next(
        (name, prompt) for name, prompt in VictimPool.VICTIM_CONFIGS if name == "standard"
    )
    assert standard_entry[1] is None


def test_victim_pool_confident_has_system():
    """Confident victim has a non-None system prompt."""
    confident_entry = next(
        (name, prompt) for name, prompt in VictimPool.VICTIM_CONFIGS if name == "confident"
    )
    assert confident_entry[1] is not None
    assert "confident" in confident_entry[1].lower() or "efficient" in confident_entry[1].lower()


def test_victim_pool_cautious_has_system():
    """Cautious victim has a non-None system prompt."""
    cautious_entry = next(
        (name, prompt) for name, prompt in VictimPool.VICTIM_CONFIGS if name == "cautious"
    )
    assert cautious_entry[1] is not None
    assert "cautious" in cautious_entry[1].lower() or "careful" in cautious_entry[1].lower()


# ── HunterRewardTracker ───────────────────────────────────────────────────────

def test_reward_tracker_init():
    tracker = HunterRewardTracker(FAILURE_MODES[:3], DOMAIN_ROTATION[:3])
    assert tracker.n_arms == 9  # 3 modes × 3 domains
    assert all(tracker.counts[arm] == 0 for arm in tracker.arms)


def test_reward_tracker_n_arms_full():
    """Full FAILURE_MODES × DOMAIN_ROTATION = 6×5 = 30 arms."""
    tracker = HunterRewardTracker(FAILURE_MODES, DOMAIN_ROTATION)
    assert tracker.n_arms == len(FAILURE_MODES) * len(DOMAIN_ROTATION)


def test_reward_tracker_select_untried_first():
    """UCB1 always tries untried arms first (count=0 → infinite UCB score)."""
    tracker = HunterRewardTracker(["retrieval_fail"], ["trivia", "multihop"])
    arm1 = tracker.select()
    tracker.update(arm1, reward=1.0)
    arm2 = tracker.select()
    assert arm2 != arm1  # picks the other untried arm


def test_reward_tracker_all_arms_tried_before_repeats():
    """All 4 arms tried before any is revisited."""
    tracker = HunterRewardTracker(["mode_a", "mode_b"], ["dom_x", "dom_y"])
    seen = set()
    for _ in range(4):
        arm = tracker.select()
        tracker.update(arm, reward=0.5)
        seen.add(arm)
    assert len(seen) == 4


def test_reward_tracker_favours_high_reward():
    """After enough trials, UCB1 favours arm with higher reward."""
    tracker = HunterRewardTracker(["mode_a", "mode_b"], ["domain_x"])
    # Give mode_a high reward
    for _ in range(10):
        tracker.update(("mode_a", "domain_x"), reward=1.0)
    for _ in range(10):
        tracker.update(("mode_b", "domain_x"), reward=0.1)
    # Select many times — mode_a should dominate
    selections = [tracker.select() for _ in range(20)]
    mode_a_count = sum(1 for s in selections if s[0] == "mode_a")
    assert mode_a_count > 12  # more than 60% of selections


def test_reward_tracker_reward_from_diversity():
    """reward_from_diversity maps correctly."""
    tracker = HunterRewardTracker(["m"], ["d"])
    assert tracker.reward_from_diversity(1.0) == 1.0
    assert tracker.reward_from_diversity(0.67) == 1.0
    assert tracker.reward_from_diversity(0.33) == 0.5
    assert tracker.reward_from_diversity(0.0) == 0.0


def test_reward_tracker_serialise():
    """State can be saved to and loaded from dict."""
    tracker = HunterRewardTracker(["m"], ["d"])
    tracker.update(("m", "d"), reward=0.8)
    state = tracker.to_dict()
    tracker2 = HunterRewardTracker.from_dict(state)
    assert tracker2.counts[("m", "d")] == tracker.counts[("m", "d")]
    assert abs(tracker2.rewards[("m", "d")] - tracker.rewards[("m", "d")]) < 1e-9
    assert tracker2._total == tracker._total


def test_reward_tracker_serialise_roundtrip_multiple_arms():
    """Serialise/deserialise with multiple arms preserves all state."""
    tracker = HunterRewardTracker(["mode_a", "mode_b"], ["dom_x", "dom_y"])
    tracker.update(("mode_a", "dom_x"), reward=1.0)
    tracker.update(("mode_b", "dom_y"), reward=0.5)
    state = tracker.to_dict()
    tracker2 = HunterRewardTracker.from_dict(state)
    assert tracker2.counts[("mode_a", "dom_x")] == 1
    assert tracker2.counts[("mode_b", "dom_y")] == 1
    assert tracker2._total == 2


def test_reward_tracker_summary_empty():
    """Summary returns empty dict when no arms tried yet."""
    tracker = HunterRewardTracker(["m"], ["d"])
    assert tracker.summary() == {}


def test_reward_tracker_summary_top5():
    """Summary returns at most 5 arms."""
    modes = ["m1", "m2", "m3"]
    domains = ["d1", "d2"]
    tracker = HunterRewardTracker(modes, domains)
    for m in modes:
        for d in domains:
            tracker.update((m, d), reward=0.5)
    s = tracker.summary()
    assert len(s) <= 5


# ── FARLCycleRunner ───────────────────────────────────────────────────────────

def test_cycle_runner_init():
    """FARLCycleRunner initialises without API call."""
    with patch("scripts.farl_phase2._load_api_key", return_value="test-key"), \
         patch("scripts.farl_phase2.AgentGuard"), \
         patch("scripts.farl_phase2.CachedLLMClient"):
        runner = FARLCycleRunner.__new__(FARLCycleRunner)
        runner.n_cycles = 3
        runner.n_per_cycle = 50
        runner.cycle_results = []
        assert runner.n_cycles == 3
        assert runner.n_per_cycle == 50


def test_diversity_score_in_log_entry():
    """Log entry must contain victim_diversity_score field."""
    entry = {
        "question": "q",
        "victim_results": [
            {"is_failure": True},
            {"is_failure": False},
            {"is_failure": True},
        ],
        "victim_diversity_score": _diversity_score([
            {"is_failure": True},
            {"is_failure": False},
            {"is_failure": True},
        ]),
    }
    assert abs(entry["victim_diversity_score"] - 2 / 3) < 0.01


def test_phase2_dir_constant():
    """PHASE2_DIR is inside the results directory."""
    from scripts.farl_phase2 import PHASE2_DIR
    assert "farl_phase2" in str(PHASE2_DIR)
    assert "results" in str(PHASE2_DIR)


def test_haiku_constant_used():
    """HAIKU constant is imported from farl_hunt (same model string)."""
    from scripts.farl_phase2 import HAIKU as P2_HAIKU
    from scripts.farl_hunt import HAIKU as P1_HAIKU
    assert P2_HAIKU == P1_HAIKU
