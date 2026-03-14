#!/usr/bin/env python3
"""
FARL Phase 2 — Multi-Agent Cooperative Shaping
===============================================

Extends Phase 1 (farl_hunt.py) with:
  1. VictimPool  — 3 diverse Haiku victims (standard/confident/cautious)
  2. HunterRewardTracker — UCB1 bandit: which (mode, domain) strategies work
  3. FARLCycleRunner — hunt → retrain MiniJudge → hunt harder, K cycles

Key new metric: victim_diversity_score
  = fraction of 3 victims that fail on a question
  0.33 = easy adversarial (only naive victim fails)
  0.67 = gray zone  ← most valuable for MiniJudge training
  1.00 = hard adversarial (all victims fail)

Inspired by: arXiv:2602.16301 — "Multi-agent cooperation through
in-context co-player inference" (Google, Weis et al., 2025)
The key insight: diverse co-player training produces better generalization
than homogeneous training. Applied here: diverse victim pool forces hunter
to find truly hard questions, not just format-specific failures.

Usage:
    python3 scripts/farl_phase2.py --cycles 3 --n 100 --budget 5.0 --search duckduckgo

Output:
    results/farl_phase2/
        cycle_{k}_taxonomy.json   — taxonomy after cycle k
        cycle_{k}_ablation.json   — ablation result after cycle k
        phase2_log.jsonl          — full iteration log
        phase2_summary.json       — final stats across all cycles
"""

import os, sys, json, time, argparse, warnings
warnings.filterwarnings("ignore")

import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from collections import defaultdict

QPPG_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(QPPG_ROOT))

from scripts.farl_hunt import (
    HuntLoop, HAIKU, FAILURE_MODES, DOMAIN_ROTATION,
    run_react_agent_real_search, _search_duckduckgo,
)
from experiments.exp156_live_crossdomain_generation import (
    CachedLLMClient, run_react_agent, _load_api_key,
)
from llm_guard import AgentGuard
from llm_guard.mini_judge import _extract_features as _mj_extract_features


# ── Victim system prompts ─────────────────────────────────────────────────────

VICTIM_A_SYSTEM = None  # uses REACT_SYSTEM from exp156 (standard)

VICTIM_B_SYSTEM = """You are a confident research assistant using the ReAct framework.
You answer questions efficiently and decisively. You make your best judgment quickly.
When you have enough information to form an answer, commit to it immediately.
Do not over-search. Prefer a direct answer in 1-2 steps.

Format EXACTLY:
Thought: [your reasoning]
Action: Search[your query]
...or...
Thought: [your reasoning]
Action: Finish[your answer]"""

VICTIM_C_SYSTEM = """You are a cautious research assistant using the ReAct framework.
You are careful about accuracy and hedge when uncertain.
If you cannot find a definitive source, say "I cannot confirm this with certainty."
Always verify facts with at least 2 searches before committing to an answer.

Format EXACTLY:
Thought: [your reasoning]
Action: Search[your query]
...or...
Thought: [your reasoning]
Action: Finish[your answer — or "I cannot confirm" if uncertain]"""


def _diversity_score(victim_results: List[dict]) -> float:
    """Fraction of victims that failed. 0.0 = none, 1.0 = all."""
    if not victim_results:
        return 0.0
    return sum(1 for r in victim_results if r.get("is_failure", False)) / len(victim_results)


class VictimPool:
    """
    Three Haiku victims with different system prompts.
    Runs the same question against all three and returns per-victim results.
    """

    VICTIM_CONFIGS = [
        ("standard",  VICTIM_A_SYSTEM),
        ("confident", VICTIM_B_SYSTEM),
        ("cautious",  VICTIM_C_SYSTEM),
    ]

    def __init__(self, api_key: str, search_fn=None, risk_threshold: float = 0.40):
        self.clients = {
            name: CachedLLMClient(api_key, model=HAIKU, domain_tag=f"farl_p2_{name}")
            for name, _ in self.VICTIM_CONFIGS
        }
        self.search_fn = search_fn
        self.risk_threshold = risk_threshold
        self.guard = AgentGuard()

        # Store victims list for testing
        self.victims = [(name, prompt, None) for name, prompt in self.VICTIM_CONFIGS]

    def run_all(self, question: str) -> List[dict]:
        """Run question against all 3 victims. Returns list of result dicts."""
        results = []
        for name, system_prompt in self.VICTIM_CONFIGS:
            llm = self.clients[name]

            # Run chain — use system prompt override for confident/cautious victims
            if self.search_fn is not None:
                # Real search: inject system prompt by temporarily patching
                chain = _run_victim_real_search(
                    question, llm, self.search_fn,
                    system_prompt=system_prompt, max_steps=7
                )
            else:
                chain = run_react_agent(question, llm, max_steps=6)

            # Judge
            try:
                scored = self.guard.score_chain(
                    question=question,
                    steps=chain["steps"],
                    final_answer=chain["final_answer"],
                )
                risk = scored.risk_score
                failure_mode = scored.failure_mode
                needs_alert = scored.needs_alert
            except Exception:
                risk, failure_mode, needs_alert = 0.0, None, False

            chain_stuck = not chain.get("finished", True) and len(chain["steps"]) >= 5
            confident_wrong = (failure_mode == "confident_wrong" and
                               chain.get("finished", False) and len(chain["steps"]) <= 3)
            is_failure = needs_alert or (risk > self.risk_threshold) or chain_stuck or confident_wrong

            results.append({
                "victim_name":   name,
                "final_answer":  chain["final_answer"],
                "steps":         len(chain["steps"]),
                "step_details":  chain["steps"],   # actual steps for feature extraction
                "finished":      chain.get("finished", False),
                "risk_score":    round(risk, 4),
                "failure_mode":  failure_mode,
                "needs_alert":   needs_alert,
                "is_failure":    is_failure,
            })
        return results

    @property
    def total_cost_usd(self) -> float:
        return sum(c.cost_usd for c in self.clients.values())


def _run_victim_real_search(
    question: str,
    llm: CachedLLMClient,
    search_fn,
    system_prompt: Optional[str],
    max_steps: int = 7,
) -> dict:
    """
    Like run_react_agent_real_search but allows a custom system prompt.
    Falls back to the standard REACT_SYSTEM if system_prompt is None.
    """
    from experiments.exp156_live_crossdomain_generation import (
        REACT_SYSTEM, parse_react_output,
    )
    import re as _re

    effective_system = system_prompt if system_prompt is not None else REACT_SYSTEM
    steps, seen_queries = [], set()
    conversation = f"Question: {question}"

    for _ in range(max_steps):
        output = llm.call(
            effective_system,
            conversation + "\n\n(Now output your next Thought and Action.)",
            max_tokens=300,
        )
        thought, action_type, action_arg = parse_react_output(output)

        if action_type == "Finish":
            steps.append({"thought": thought, "action_type": "Finish",
                          "action_arg": action_arg, "observation": ""})
            return {"final_answer": action_arg, "steps": steps, "finished": True}

        query_key = action_arg.strip().lower()
        if query_key in seen_queries:
            observation = "You already searched for this. Try a different, more specific query."
        else:
            seen_queries.add(query_key)
            observation = search_fn(action_arg)

        steps.append({"thought": thought, "action_type": "Search",
                      "action_arg": action_arg, "observation": observation})
        conversation += (f"\nThought: {thought}\nAction: Search[{action_arg}]"
                         f"\nObservation: {observation}")

    forced = llm.call(
        "Given the conversation so far, what is the best final answer? Reply with ONLY the answer.",
        conversation, max_tokens=100,
    )
    steps.append({"thought": "Forced finish", "action_type": "Finish",
                  "action_arg": forced, "observation": ""})
    return {"final_answer": forced, "steps": steps, "finished": False}


# ── HunterRewardTracker ───────────────────────────────────────────────────────

class HunterRewardTracker:
    """
    UCB1 bandit over (failure_mode, domain) strategy pairs.

    Reward signal:
      +1.0  if victim_diversity_score >= 0.67 (gray zone: 2+ victims fail)
      +0.5  if victim_diversity_score == 0.33 (easy: 1 victim fails)
       0.0  if no victim fails (question too easy)

    UCB1 formula: score(arm) = mean_reward(arm) + C * sqrt(ln(total) / count(arm))
    Untried arms: score = infinity (always tried first)
    """

    C = 1.0  # exploration constant

    def __init__(self, failure_modes: List[str], domains: List[str]):
        self.arms = [(m, d) for m in failure_modes for d in domains]
        self.n_arms = len(self.arms)
        self.counts = defaultdict(int)
        self.rewards = defaultdict(float)
        self._total = 0

    def select(self) -> Tuple[str, str]:
        """UCB1 arm selection. Returns (failure_mode, domain)."""
        untried = [a for a in self.arms if self.counts[a] == 0]
        if untried:
            return untried[0]
        scores = {}
        for arm in self.arms:
            mean = self.rewards[arm] / self.counts[arm]
            bonus = self.C * np.sqrt(np.log(self._total) / self.counts[arm])
            scores[arm] = mean + bonus
        return max(scores, key=scores.__getitem__)

    def update(self, arm: Tuple[str, str], reward: float):
        """Record reward for arm after an iteration."""
        self.counts[arm] += 1
        self.rewards[arm] += reward
        self._total += 1

    def reward_from_diversity(self, diversity_score: float) -> float:
        """Convert victim_diversity_score to bandit reward."""
        if diversity_score >= 0.67:
            return 1.0
        elif diversity_score >= 0.33:
            return 0.5
        return 0.0

    def summary(self) -> dict:
        """Top-5 arms by mean reward."""
        tried = {a: self.rewards[a] / self.counts[a]
                 for a in self.arms if self.counts[a] > 0}
        top = sorted(tried.items(), key=lambda x: -x[1])[:5]
        return {f"{m}/{d}": round(r, 3) for (m, d), r in top}

    def to_dict(self) -> dict:
        return {
            "arms":    [list(a) for a in self.arms],
            "counts":  {f"{k[0]}|{k[1]}": v for k, v in self.counts.items()},
            "rewards": {f"{k[0]}|{k[1]}": v for k, v in self.rewards.items()},
            "total":   self._total,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "HunterRewardTracker":
        modes = list(dict.fromkeys(a[0] for a in data["arms"]))
        domains = list(dict.fromkeys(a[1] for a in data["arms"]))
        tracker = cls(modes, domains)
        for key, val in data["counts"].items():
            m, d = key.split("|", 1)
            tracker.counts[(m, d)] = val
        for key, val in data["rewards"].items():
            m, d = key.split("|", 1)
            tracker.rewards[(m, d)] = val
        tracker._total = data["total"]
        return tracker


# ── FARLCycleRunner ───────────────────────────────────────────────────────────

PHASE2_DIR = QPPG_ROOT / "results" / "farl_phase2"


class FARLCycleRunner:
    """
    Multi-cycle FARL loop with cooperative shaping.

    Each cycle:
      1. Hunt n_per_cycle questions using VictimPool (3 victims each)
      2. Add gray-zone failures (diversity >= 0.33) to taxonomy
      3. Retrain MiniJudge on accumulated taxonomy + exp156
      4. Log cycle stats: victim pass rate improvement, taxonomy growth

    After all cycles:
      - Run ablation (Phase 1 baseline vs. Phase 2 taxonomy)
      - Save phase2_summary.json
    """

    def __init__(
        self,
        api_key: str,
        n_cycles: int = 3,
        n_per_cycle: int = 100,
        budget_usd: float = 5.0,
        search_backend: str = "duckduckgo",
        resume: bool = False,
        risk_threshold: float = 0.40,
    ):
        self.api_key = api_key
        self.n_cycles = n_cycles
        self.n_per_cycle = n_per_cycle
        self.budget_usd = budget_usd

        PHASE2_DIR.mkdir(parents=True, exist_ok=True)

        # Build search fn
        search_fn = None
        if search_backend == "duckduckgo":
            search_fn = _search_duckduckgo
            print("[search] DuckDuckGo")

        self.victim_pool = VictimPool(api_key, search_fn=search_fn, risk_threshold=risk_threshold)
        self.hunter_llm = CachedLLMClient(api_key, model=HAIKU, domain_tag="farl_p2_hunter")
        self.reward_tracker = HunterRewardTracker(FAILURE_MODES, DOMAIN_ROTATION)

        # Shared taxonomy (grows across all cycles)
        self.taxonomy: Dict[str, List[dict]] = defaultdict(list)
        self.known_features: List[np.ndarray] = []
        self.phase2_log: List[dict] = []
        self.cycle_results: List[dict] = []
        self.sigma = 1.0

        # Load program brief
        brief_path = QPPG_ROOT / "scripts" / "program_farl_phase2.md"
        if brief_path.exists():
            self.program_brief = brief_path.read_text()
        else:
            fallback = QPPG_ROOT / "scripts" / "program_farl.md"
            self.program_brief = fallback.read_text() if fallback.exists() else (
                "Generate a hard factual question that requires multi-step reasoning."
            )

        if resume:
            self._load_checkpoint()

    # ── Novelty ──────────────────────────────────────────────────────────────

    def compute_novelty(self, features: np.ndarray) -> float:
        if not self.known_features:
            return 1.0
        dists = [np.linalg.norm(features - kf) for kf in self.known_features]
        min_dist = min(dists)
        if len(self.known_features) >= 2:
            pairs = [np.linalg.norm(self.known_features[i] - self.known_features[j])
                     for i in range(len(self.known_features))
                     for j in range(i + 1, len(self.known_features))]
            self.sigma = max(float(np.median(pairs)), 0.01)
        return min(1.0, min_dist / self.sigma)

    # ── Hunter question generation ────────────────────────────────────────────

    def generate_question(self, target_mode: str, domain: str, iteration: int) -> str:
        """Ask Haiku to generate 1 adversarial question for (mode, domain)."""
        recent = []
        for entries in self.taxonomy.values():
            recent.extend(e["question"] for e in entries[-3:])

        reward_context = ""
        summary = self.reward_tracker.summary()
        if summary:
            reward_context = f"\nBest-performing strategies so far: {json.dumps(summary)}"

        user_prompt = (
            f"Target failure mode: {target_mode}\n"
            f"Domain: {domain}\n"
            f"Recent failures (avoid repeating):\n{json.dumps(recent[-10:], indent=2)}"
            f"{reward_context}\n\n"
            f"Return ONLY the question."
        )
        import anthropic
        import re as _re
        client = anthropic.Anthropic(api_key=self.api_key)
        try:
            resp = client.messages.create(
                model=HAIKU,
                max_tokens=150,
                system=self.program_brief,
                messages=[{"role": "user", "content": user_prompt}],
            )
            question = resp.content[0].text.strip().strip('"').strip("'")
            self.hunter_llm.total_input_tokens += resp.usage.input_tokens
            self.hunter_llm.total_output_tokens += resp.usage.output_tokens
        except Exception:
            question = self.hunter_llm.call(
                system=self.program_brief, user=user_prompt, max_tokens=150
            )
        question = _re.sub(
            r"^(here'?s?\s+(a|an|the)\s+(adversarial\s+)?question\s*:?\s*)",
            "", question.strip(), flags=_re.IGNORECASE
        ).strip().strip('"').strip("'")
        return question

    # ── One iteration ─────────────────────────────────────────────────────────

    def run_one_iteration(self, global_iter: int) -> dict:
        """Hunt cycle with VictimPool: 1 question → 3 victims → diversity score."""
        target_mode, domain = self.reward_tracker.select()

        question = self.generate_question(target_mode, domain, global_iter)

        # Run all 3 victims
        victim_results = self.victim_pool.run_all(question)
        diversity = _diversity_score(victim_results)

        # Use worst victim result for taxonomy (first failure found)
        worst = next((r for r in victim_results if r["is_failure"]), victim_results[0])
        failure_mode = worst["failure_mode"]
        risk_score = max(r["risk_score"] for r in victim_results)

        # Reward hunter
        reward = self.reward_tracker.reward_from_diversity(diversity)
        self.reward_tracker.update((target_mode, domain), reward)

        entry = {
            "global_iter":            global_iter,
            "target_mode":            target_mode,
            "domain":                 domain,
            "question":               question,
            "victim_results":         victim_results,
            "victim_diversity_score": round(diversity, 3),
            "failure_mode":           failure_mode,
            "risk_score":             round(risk_score, 4),
            "hunter_reward":          reward,
            "novel":                  False,
            "novelty_score":          0.0,
        }

        # Add to taxonomy if at least 1 victim failed
        any_failure = diversity > 0
        if any_failure:
            try:
                # Real novelty: extract MiniJudge features from worst victim's chain
                worst_steps = worst.get("step_details", [])
                feats = _mj_extract_features({
                    "question":     question,
                    "steps":        worst_steps,
                    "final_answer": worst.get("final_answer", ""),
                })
                novelty = self.compute_novelty(np.array(feats))
                entry["novelty_score"] = round(novelty, 3)

                if novelty > 0.3:
                    entry["novel"] = True
                    self.known_features.append(np.array(feats))
                    mode_key = failure_mode or "unknown"
                    self.taxonomy[mode_key].append({
                        **{k: v for k, v in entry.items() if k != "victim_results"},
                        "victim_diversity_score": diversity,
                        "step_details":           worst_steps,             # for feature extraction in ablation
                        "final_answer":           worst.get("final_answer", ""),  # required by retrain loader
                    })
            except Exception:
                pass

        return entry

    # ── One cycle ─────────────────────────────────────────────────────────────

    def run_cycle(self, cycle_idx: int) -> dict:
        """Run n_per_cycle iterations, return cycle stats."""
        print(f"\n{'='*60}")
        print(f"  FARL PHASE 2 — CYCLE {cycle_idx + 1}/{self.n_cycles}")
        print(f"  Taxonomy so far: {self._taxonomy_counts()}")
        print(f"{'='*60}")

        cycle_start = len(self.phase2_log)
        n_novel = 0

        for i in range(self.n_per_cycle):
            global_iter = len(self.phase2_log)
            entry = self.run_one_iteration(global_iter)
            self.phase2_log.append(entry)

            if entry["novel"]:
                n_novel += 1

            # Print progress every iteration when n is small; else every 10
            show = (self.n_per_cycle <= 5) or (i % 10 == 0)
            if show:
                cost = self._total_cost()
                print(f"  [{cycle_idx+1}] iter {i+1}/{self.n_per_cycle}  "
                      f"novel={n_novel}  "
                      f"victim_diversity_score={entry['victim_diversity_score']:.2f}  "
                      f"diversity_avg={self._avg_diversity():.2f}  "
                      f"cost=${cost:.3f}")
                self._save_checkpoint(cycle_idx)

            if self._total_cost() > self.budget_usd:
                print(f"  Budget ${self.budget_usd} reached — stopping.")
                break

        # Retrain MiniJudge after cycle
        auroc_after = self._retrain_mini_judge(cycle_idx)

        cycle_result = {
            "cycle":            cycle_idx + 1,
            "n_iterations":     len(self.phase2_log) - cycle_start,
            "n_novel":          n_novel,
            "taxonomy_total":   sum(len(v) for v in self.taxonomy.values()),
            "taxonomy_by_mode": self._taxonomy_counts(),
            "avg_diversity":    round(self._avg_diversity(), 3),
            "top_strategies":   self.reward_tracker.summary(),
            "mini_judge_auroc": auroc_after,
            "cost_usd":         round(self._total_cost(), 4),
        }
        self.cycle_results.append(cycle_result)

        print(f"\n  Cycle {cycle_idx+1} complete:")
        print(f"    Novel failures:  {n_novel}")
        print(f"    Taxonomy total:  {cycle_result['taxonomy_total']}")
        print(f"    Avg diversity:   {cycle_result['avg_diversity']:.2f}")
        print(f"    MiniJudge AUROC: {auroc_after:.4f}")
        print(f"    Top strategies:  {cycle_result['top_strategies']}")

        return cycle_result

    # ── MiniJudge retrain ─────────────────────────────────────────────────────

    def _retrain_mini_judge(self, cycle_idx: int) -> float:
        """Retrain MiniJudge on accumulated taxonomy. Returns test AUROC after retraining."""
        import subprocess
        n_chains = sum(len(v) for v in self.taxonomy.values())
        print(f"\n  [Cycle {cycle_idx+1}] Retraining MiniJudge on {n_chains} taxonomy chains...")
        report_path = QPPG_ROOT / "results" / "farl_hunt" / "retrain_report.json"
        try:
            tax_path = QPPG_ROOT / "results" / "farl_hunt" / "taxonomy.json"
            tax_path.parent.mkdir(parents=True, exist_ok=True)
            tax_path.write_text(json.dumps(
                {k: list(v) for k, v in self.taxonomy.items()}, indent=2
            ))
            result = subprocess.run(
                ["python3", str(QPPG_ROOT / "scripts" / "retrain_mini_judge_from_taxonomy.py")],
                capture_output=True, text=True, cwd=str(QPPG_ROOT)
            )
            if result.returncode != 0:
                print(f"  WARNING: retrain script exited {result.returncode}")
                print(result.stderr[-800:] if result.stderr else "")
                return 0.0
            # Read AUROC after retraining from JSON report (not stdout — avoids capturing "before" value)
            if report_path.exists():
                report = json.loads(report_path.read_text())
                return float(report.get("auroc_after_test", 0.0))
            return 0.0
        except Exception as e:
            print(f"  WARNING: retrain failed: {e}")
            return 0.0

    # ── Main run ──────────────────────────────────────────────────────────────

    def run(self):
        """Run all K cycles."""
        print(f"\n{'='*60}")
        print(f"  FARL PHASE 2 — MULTI-AGENT COOPERATIVE SHAPING")
        print(f"  Cycles: {self.n_cycles}  |  Per cycle: {self.n_per_cycle}")
        print(f"  Budget: ${self.budget_usd}  |  3 victims per question")
        print(f"  Theory: arXiv:2602.16301 (Google PPI — diverse co-players)")
        print(f"{'='*60}")

        for cycle_idx in range(self.n_cycles):
            if self._total_cost() > self.budget_usd:
                print(f"Budget exhausted before cycle {cycle_idx+1}.")
                break
            self.run_cycle(cycle_idx)

        self._save_final_summary()

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _total_cost(self) -> float:
        return self.hunter_llm.cost_usd + self.victim_pool.total_cost_usd

    def _taxonomy_counts(self) -> dict:
        return {k: len(v) for k, v in self.taxonomy.items() if v}

    def _avg_diversity(self) -> float:
        if not self.phase2_log:
            return 0.0
        scores = [e.get("victim_diversity_score", 0.0) for e in self.phase2_log[-50:]]
        return float(np.mean(scores))

    def _save_checkpoint(self, cycle_idx: int):
        log_path = PHASE2_DIR / "phase2_log.jsonl"
        log_path.write_text("\n".join(json.dumps(e, default=str) for e in self.phase2_log))
        tax_path = PHASE2_DIR / f"cycle_{cycle_idx+1}_taxonomy.json"
        tax_path.write_text(json.dumps(
            {k: list(v) for k, v in self.taxonomy.items()}, indent=2, default=str
        ))

    def _save_final_summary(self):
        summary = {
            "total_cycles":      len(self.cycle_results),
            "total_iterations":  len(self.phase2_log),
            "total_novel":       sum(c["n_novel"] for c in self.cycle_results),
            "taxonomy_final":    self._taxonomy_counts(),
            "mini_judge_auroc_by_cycle": [c["mini_judge_auroc"] for c in self.cycle_results],
            "avg_diversity_trend": [c["avg_diversity"] for c in self.cycle_results],
            "top_strategies_final": self.reward_tracker.summary(),
            "total_cost_usd":    round(self._total_cost(), 4),
            "cycle_results":     self.cycle_results,
        }
        (PHASE2_DIR / "phase2_summary.json").write_text(json.dumps(summary, indent=2))
        print(f"\n{'='*60}")
        print(f"  FARL PHASE 2 COMPLETE")
        print(f"  Cycles:     {summary['total_cycles']}")
        print(f"  Iterations: {summary['total_iterations']}")
        print(f"  Novel:      {summary['total_novel']}")
        print(f"  Taxonomy:   {summary['taxonomy_final']}")
        print(f"  AUROC trend: {[round(a, 4) for a in summary['mini_judge_auroc_by_cycle']]}")
        print(f"  Top strategies: {summary['top_strategies_final']}")
        print(f"  Total cost: ${summary['total_cost_usd']:.4f}")
        print(f"{'='*60}")
        print(f"\n  Results: {PHASE2_DIR}/")

    def _load_checkpoint(self):
        log_path = PHASE2_DIR / "phase2_log.jsonl"
        if log_path.exists():
            self.phase2_log = [json.loads(l) for l in log_path.read_text().splitlines() if l.strip()]
            print(f"Resumed Phase 2 log: {len(self.phase2_log)} iterations")
        tax_files = sorted(PHASE2_DIR.glob("cycle_*_taxonomy.json"))
        if tax_files:
            data = json.loads(tax_files[-1].read_text())
            self.taxonomy = defaultdict(list, {k: list(v) for k, v in data.items()})
            print(f"Resumed taxonomy: {self._taxonomy_counts()}")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FARL Phase 2 — Multi-Agent Cooperative Shaping")
    parser.add_argument("--cycles",  type=int,   default=3,            help="Number of FARL cycles")
    parser.add_argument("--n",       type=int,   default=100,          help="Hunt iterations per cycle")
    parser.add_argument("--budget",  type=float, default=5.0,          help="Total budget in USD")
    parser.add_argument("--search",  type=str,   default="duckduckgo", choices=["fake", "duckduckgo", "tavily"])
    parser.add_argument("--resume",    action="store_true",              help="Resume from checkpoint")
    parser.add_argument("--threshold", type=float, default=0.40,         help="Risk threshold for victim failure (default: 0.40)")
    args = parser.parse_args()

    api_key = _load_api_key()
    runner = FARLCycleRunner(
        api_key=api_key,
        n_cycles=args.cycles,
        n_per_cycle=args.n,
        budget_usd=args.budget,
        search_backend=args.search,
        resume=args.resume,
        risk_threshold=args.threshold,
    )
    runner.run()
