#!/usr/bin/env python3
"""
FARL Discovery Loop — CPU/API-only failure hunting with llm-guard-kit
======================================================================

Inspired by karpathy/autoresearch:
  program_farl.md  = the research brief (human writes this)
  farl_hunt.py     = the autonomous loop (agent runs this)
  novel_failure_rate = the metric (vs val_bpb in autoresearch)

Loop per iteration:
  1. HUNTER  — Claude Haiku reads program_farl.md, generates 1 adversarial question
  2. VICTIM  — Claude Haiku runs ReAct chain (up to 6 steps, fake search engine)
  3. JUDGE   — llm-guard-kit AgentGuard scores chain ($0, behavioral only)
  4. NOVELTY — L2 distance in 15-dim SC_OLD feature space (novel = diverse failure)
  5. TAXONOMY — novel failures stored by failure_mode

Cost: ~$0.0014/iteration (Haiku only, judge is free)
      50 iterations ≈ $0.07

Usage:
    # Quick smoke test (5 iterations, $0.10 budget)
    python3 scripts/farl_hunt.py --n 5 --budget 0.10

    # Full run (50 iterations)
    python3 scripts/farl_hunt.py --n 50 --budget 2.0

    # Self-evaluation demo (agents score their own output)
    python3 scripts/farl_hunt.py --demo --question "Who wrote Hamlet?"

    # Resume from existing taxonomy
    python3 scripts/farl_hunt.py --n 50 --resume
"""

import os, sys, json, time, hashlib, re, argparse, warnings
warnings.filterwarnings("ignore")

import numpy as np
from pathlib import Path
from typing import List, Dict, Optional
from collections import defaultdict

# ── Paths ─────────────────────────────────────────────────────────────────────

QPPG_ROOT    = Path(__file__).resolve().parent.parent
SCRIPTS_DIR  = QPPG_ROOT / "scripts"
RESULTS_DIR  = QPPG_ROOT / "results" / "farl_hunt"
CACHE_DIR    = RESULTS_DIR / "cache"
PROGRAM_PATH = SCRIPTS_DIR / "program_farl.md"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(QPPG_ROOT))

# ── Model constants ───────────────────────────────────────────────────────────

HAIKU = "claude-haiku-4-5-20251001"

# ── Import reusable components from exp156 ────────────────────────────────────

try:
    from experiments.exp156_live_crossdomain_generation import (
        CachedLLMClient,
        run_react_agent,
        _load_api_key,
    )
except ImportError as e:
    print(f"ERROR: Could not import from exp156: {e}")
    print("Make sure you run from the QPPG root directory.")
    sys.exit(1)

# ── Import llm-guard-kit ──────────────────────────────────────────────────────

try:
    from llm_guard import AgentGuard, QueryRewriter
    from llm_guard.local_verifier import extract_features, FEATURE_NAMES
except ImportError as e:
    print(f"ERROR: Could not import llm_guard: {e}")
    print("Install with: pip install llm-guard-kit")
    sys.exit(1)

# ── Failure mode target order ─────────────────────────────────────────────────

FAILURE_MODES = [
    "retrieval_fail",
    "repeated_query",
    "long_chain",
    "empty_answer",
    "factual_error",
]

DOMAIN_ROTATION = [
    "trivia",
    "multihop",
    "temporal",
    "numeric",
    "person_factual",
]


# ── Hunt Loop ─────────────────────────────────────────────────────────────────

class HuntLoop:
    def __init__(
        self,
        api_key: str,
        n_iterations: int = 50,
        risk_threshold: float = 0.65,
        novelty_threshold: float = 0.3,
        budget_usd: float = 2.0,
        resume: bool = False,
    ):
        self.api_key           = api_key
        self.n_iterations      = n_iterations
        self.risk_threshold    = risk_threshold
        self.novelty_threshold = novelty_threshold
        self.budget_usd        = budget_usd

        # Separate LLM clients for hunter and victim (separate caches)
        self.hunter_llm = CachedLLMClient(api_key, model=HAIKU, domain_tag="farl_hunter")
        self.victim_llm = CachedLLMClient(api_key, model=HAIKU, domain_tag="farl_victim")

        # llm-guard-kit judge (behavioral only, $0)
        self.guard = AgentGuard()

        # State
        self.taxonomy: Dict[str, List[dict]] = defaultdict(list)
        self.known_features: List[np.ndarray] = []
        self.hunt_log: List[dict] = []
        self.sigma: float = 1.0  # adaptive novelty scale

        # Load program brief once
        if PROGRAM_PATH.exists():
            self.program_brief = PROGRAM_PATH.read_text()
        else:
            self.program_brief = "Generate a hard factual question that requires multi-step reasoning."

        # Resume from existing taxonomy if requested
        if resume:
            self._load_checkpoint()

    # ── Novelty computation ───────────────────────────────────────────────────

    def compute_novelty(self, features: np.ndarray) -> float:
        """
        L2 distance to nearest known failure in feature space, normalized by σ.
        Returns 1.0 if no known failures yet (first failure is always novel).
        """
        if not self.known_features:
            return 1.0
        dists = [np.linalg.norm(features - kf) for kf in self.known_features]
        min_dist = min(dists)
        # Adaptive σ: median pairwise distance among known failures
        if len(self.known_features) >= 2:
            pairs = []
            for i in range(len(self.known_features)):
                for j in range(i + 1, len(self.known_features)):
                    pairs.append(np.linalg.norm(self.known_features[i] - self.known_features[j]))
            self.sigma = max(float(np.median(pairs)), 0.01)
        return min(1.0, min_dist / self.sigma)

    # ── Failure mode targeting ────────────────────────────────────────────────

    def weakest_failure_mode(self, iteration: int) -> str:
        """
        Return the failure mode with the fewest novel failures so far.
        Forces rotation every 3 iterations to prevent one mode dominating.
        """
        counts = {m: len(self.taxonomy[m]) for m in FAILURE_MODES}
        # Every 3rd iteration, force round-robin regardless of counts
        # This prevents repeated_query from monopolizing all iterations
        if iteration % 3 == 0:
            return FAILURE_MODES[iteration // 3 % len(FAILURE_MODES)]
        # Otherwise pick least-covered mode
        return min(FAILURE_MODES, key=lambda m: counts[m])

    def current_domain(self, iteration: int) -> str:
        return DOMAIN_ROTATION[iteration % len(DOMAIN_ROTATION)]

    # ── Hunter: generate adversarial question ─────────────────────────────────

    def generate_adversarial_question(
        self,
        target_mode: str,
        domain: str,
        iteration: int,
    ) -> str:
        """
        Ask Claude Haiku (hunter) to generate 1 adversarial question.
        Uses program_farl.md as the system prompt (the research brief).
        """
        # Build ALL known questions context to prevent repeats
        all_known_qs = []
        for mode_entries in self.taxonomy.values():
            all_known_qs.extend(e["question"] for e in mode_entries)
        # Also include recent non-novel failures from log (last 10)
        recent_log_qs = [e["question"] for e in self.hunt_log[-10:]]
        avoid_qs = list(dict.fromkeys(all_known_qs + recent_log_qs))  # deduplicated

        user_prompt = (
            f"Target failure mode: {target_mode}\n"
            f"Domain: {domain}\n"
            f"Iteration: {iteration + 1}\n"
            f"Random seed (use to vary your output): {iteration * 7 + hash(domain) % 100}\n"
        )
        if avoid_qs:
            # Show last 8 to avoid prompt bloat
            user_prompt += "\nQuestions already tried (DO NOT repeat or paraphrase these):\n"
            user_prompt += "\n".join(f"  - {q}" for q in avoid_qs[-8:])
        user_prompt += "\n\nGenerate ONE completely new adversarial question now (must be different topic/entity from all above):"

        # Use temperature=0.9 for hunter to ensure diversity (bypass cache)
        import anthropic as _anthropic
        api_key = self.hunter_llm.client.api_key if hasattr(self.hunter_llm.client, 'api_key') else self.api_key
        _client = _anthropic.Anthropic(api_key=self.api_key)
        try:
            resp = _client.messages.create(
                model=HAIKU,
                max_tokens=150,
                temperature=0.9,
                system=self.program_brief,
                messages=[{"role": "user", "content": user_prompt}],
            )
            response = resp.content[0].text
            self.hunter_llm.total_input_tokens  += resp.usage.input_tokens
            self.hunter_llm.total_output_tokens += resp.usage.output_tokens
        except Exception:
            # Fallback to cached client
            response = self.hunter_llm.call(
                system=self.program_brief,
                user=user_prompt,
                max_tokens=150,
            )
        # Clean up response — strip quotes, extra text
        question = response.strip().strip('"').strip("'")
        # Remove any "Here's a question:" prefix the model might add
        question = re.sub(r"^(here'?s?\s+(a|an|the)\s+(adversarial\s+)?question\s*:?\s*)", "", question, flags=re.IGNORECASE)
        question = question.strip().strip('"').strip("'")
        return question

    # ── One hunt iteration ────────────────────────────────────────────────────

    def run_one_iteration(self, iteration: int) -> dict:
        """
        Full hunt cycle: generate → victim runs → judge → novelty → taxonomy.
        Returns the log entry for this iteration.
        """
        target_mode = self.weakest_failure_mode(iteration)
        domain = self.current_domain(iteration)

        # 1. Hunter generates question
        question = self.generate_adversarial_question(target_mode, domain, iteration)

        # 2. Victim runs ReAct chain
        chain = run_react_agent(question, self.victim_llm, max_steps=6)

        # 3. Judge scores chain (llm-guard-kit, $0 behavioral)
        try:
            result = self.guard.score_chain(
                question=question,
                steps=chain["steps"],
                final_answer=chain["final_answer"],
            )
            risk_score      = result.risk_score
            confidence_tier = result.confidence_tier
            failure_mode    = result.failure_mode
            needs_alert     = result.needs_alert
            behavioral_components = getattr(result, "behavioral_components", {})
        except Exception as e:
            # Guard failed — treat as unknown risk
            risk_score      = 0.0
            confidence_tier = "UNKNOWN"
            failure_mode    = None
            needs_alert     = False
            behavioral_components = {}

        entry = {
            "iteration":           iteration,
            "question":            question,
            "final_answer":        chain["final_answer"],
            "steps":               len(chain["steps"]),
            "finished":            chain.get("finished", False),
            "domain":              domain,
            "target_mode":         target_mode,
            "risk_score":          round(risk_score, 4),
            "confidence_tier":     confidence_tier,
            "failure_mode":        failure_mode,
            "needs_alert":         needs_alert,
            "behavioral_components": behavioral_components,
            "novel":               False,
            "novelty_score":       0.0,
        }

        # 4. Novelty check — flag if:
        #    - guard says needs_alert, OR
        #    - risk above threshold, OR
        #    - chain didn't finish (hit max steps = agent got stuck)
        chain_stuck = not chain.get("finished", True) and len(chain["steps"]) >= 5
        is_failure = needs_alert or (risk_score > self.risk_threshold) or chain_stuck
        if is_failure:
            try:
                features = extract_features(question, chain["steps"], chain["final_answer"])
                novelty = self.compute_novelty(features)
                entry["novelty_score"] = round(novelty, 4)

                if novelty > self.novelty_threshold:
                    entry["novel"] = True
                    mode_key = failure_mode or "unknown"
                    # Store compact version (no full steps to keep taxonomy readable)
                    taxonomy_entry = {k: v for k, v in entry.items() if k != "behavioral_components"}
                    taxonomy_entry["step_details"] = chain["steps"]
                    self.taxonomy[mode_key].append(taxonomy_entry)
                    self.known_features.append(features)
            except Exception:
                pass  # feature extraction failed — skip novelty check

        return entry

    # ── Cost accounting ───────────────────────────────────────────────────────

    @property
    def total_cost_usd(self) -> float:
        return self.hunter_llm.cost_usd + self.victim_llm.cost_usd

    def failure_counts(self) -> dict:
        return {k: len(v) for k, v in self.taxonomy.items() if v}

    def total_novel_failures(self) -> int:
        return sum(len(v) for v in self.taxonomy.values())

    # ── Progress banner ───────────────────────────────────────────────────────

    def _print_banner(self, iteration: int, entry: dict):
        """autoresearch-style progress banner per iteration."""
        novel_marker = " ★ NOVEL" if entry["novel"] else ""
        print(f"\n{'=' * 60}")
        print(f"  Iteration {iteration + 1}/{self.n_iterations}")
        print(f"  Domain:     {entry['domain']}  →  target: {entry['target_mode']}")
        print(f"  Question:   {entry['question'][:75]}{'...' if len(entry['question']) > 75 else ''}")
        print(f"  Answer:     {entry['final_answer'][:60]}{'...' if len(entry['final_answer']) > 60 else ''}")
        print(f"  Steps:      {entry['steps']}  |  Finished: {entry['finished']}")
        print(f"  Risk:       {entry['risk_score']:.3f}  Tier: {entry['confidence_tier']}")
        print(f"  Mode:       {entry['failure_mode'] or 'none detected'}{novel_marker}")
        if entry["novel"]:
            print(f"  Novelty:    {entry['novelty_score']:.3f}")
        print(f"  Taxonomy:   {self.failure_counts()}")
        print(f"  Novel/Total:{self.total_novel_failures()}/{iteration + 1}  Cost: ${self.total_cost_usd:.4f}")

    # ── Persistence ───────────────────────────────────────────────────────────

    def _save_checkpoint(self):
        taxonomy_path = RESULTS_DIR / "taxonomy.json"
        log_path      = RESULTS_DIR / "hunt_log.jsonl"

        # Write taxonomy (all novel failures)
        serializable = {}
        for mode, entries in self.taxonomy.items():
            serializable[mode] = []
            for e in entries:
                entry_copy = dict(e)
                # Convert numpy types
                for k, v in entry_copy.items():
                    if isinstance(v, np.floating):
                        entry_copy[k] = float(v)
                    elif isinstance(v, np.integer):
                        entry_copy[k] = int(v)
                serializable[mode].append(entry_copy)
        taxonomy_path.write_text(json.dumps(serializable, indent=2))

        # Append latest log entry
        if self.hunt_log:
            with open(log_path, "a") as f:
                entry = self.hunt_log[-1]
                entry_copy = {}
                for k, v in entry.items():
                    if isinstance(v, np.floating):
                        entry_copy[k] = float(v)
                    elif isinstance(v, np.integer):
                        entry_copy[k] = int(v)
                    else:
                        entry_copy[k] = v
                f.write(json.dumps(entry_copy) + "\n")

    def _load_checkpoint(self):
        """Resume from existing taxonomy.json and hunt_log.jsonl."""
        taxonomy_path = RESULTS_DIR / "taxonomy.json"
        log_path      = RESULTS_DIR / "hunt_log.jsonl"

        if taxonomy_path.exists():
            data = json.loads(taxonomy_path.read_text())
            for mode, entries in data.items():
                self.taxonomy[mode] = entries
                for entry in entries:
                    # Reconstruct feature vectors from log if available
                    pass
            print(f"Resumed taxonomy: {self.failure_counts()}")

        if log_path.exists():
            with open(log_path) as f:
                self.hunt_log = [json.loads(line) for line in f if line.strip()]
            print(f"Resumed log: {len(self.hunt_log)} iterations")

    def _save_final_summary(self):
        n_total   = len(self.hunt_log)
        n_failed  = sum(1 for e in self.hunt_log if e["needs_alert"] or e["risk_score"] > self.risk_threshold)
        n_novel   = self.total_novel_failures()

        summary = {
            "total_iterations":       n_total,
            "total_failures_detected": n_failed,
            "novel_failures":          n_novel,
            "novel_failure_rate":      round(n_novel / max(n_total, 1), 4),
            "failure_detection_rate":  round(n_failed / max(n_total, 1), 4),
            "failure_modes":           self.failure_counts(),
            "total_cost_usd":          round(self.total_cost_usd, 5),
            "cost_per_novel_failure_usd": round(
                self.total_cost_usd / max(n_novel, 1), 5
            ),
            "hunter_stats":  self.hunter_llm.stats,
            "victim_stats":  self.victim_llm.stats,
        }

        summary_path = RESULTS_DIR / "summary.json"
        summary_path.write_text(json.dumps(summary, indent=2))

        print(f"\n{'=' * 60}")
        print("  FARL HUNT COMPLETE")
        print(f"{'=' * 60}")
        print(f"  Iterations:      {n_total}")
        print(f"  Failures found:  {n_failed}  ({100*n_failed//max(n_total,1)}%)")
        print(f"  Novel failures:  {n_novel}  ({100*n_novel//max(n_total,1)}%)")
        print(f"  Taxonomy:        {self.failure_counts()}")
        print(f"  Total cost:      ${self.total_cost_usd:.4f}")
        print(f"  Cost/novel fail: ${self.total_cost_usd/max(n_novel,1):.4f}")
        print(f"\n  Results saved to: {RESULTS_DIR}/")
        print(f"    taxonomy.json   — novel failures by mode")
        print(f"    hunt_log.jsonl  — all iterations")
        print(f"    summary.json    — final stats")

    # ── Main run loop ─────────────────────────────────────────────────────────

    def run(self):
        print(f"\n{'=' * 60}")
        print("  FARL DISCOVERY LOOP")
        print(f"{'=' * 60}")
        print(f"  Iterations:     {self.n_iterations}")
        print(f"  Budget:         ${self.budget_usd:.2f}")
        print(f"  Risk threshold: {self.risk_threshold}")
        print(f"  Novelty thresh: {self.novelty_threshold}")
        print(f"  Output:         {RESULTS_DIR}/")
        print(f"  Research brief: {PROGRAM_PATH.name}")

        for i in range(self.n_iterations):
            entry = self.run_one_iteration(i)
            self.hunt_log.append(entry)
            self._print_banner(i, entry)
            self._save_checkpoint()

            if self.total_cost_usd >= self.budget_usd:
                print(f"\n  Budget ${self.budget_usd:.2f} reached — stopping at iteration {i+1}.")
                break

        self._save_final_summary()


# ── Self-evaluation demo ──────────────────────────────────────────────────────

def run_guarded_agent_demo(question: str, api_key: str):
    """
    Demo: agent runs ReAct chain, then scores its own output with llm-guard-kit
    before committing. Like autoresearch's keep/discard — but for answer quality.

    This is the core pattern for llm-guard-kit self-evaluation:
      run → score → if flagged → activate adapter → rewrite query
    """
    print(f"\n{'=' * 60}")
    print("  FARL SELF-EVALUATION DEMO")
    print(f"{'=' * 60}")
    print(f"  Question: {question}")

    llm = CachedLLMClient(api_key, model=HAIKU, domain_tag="farl_demo")
    chain = run_react_agent(question, llm, max_steps=6)

    print(f"\n  Chain steps: {len(chain['steps'])}")
    for i, step in enumerate(chain["steps"]):
        print(f"  Step {i+1}: [{step['action_type']}] {step['action_arg'][:60]}")
        if step["observation"]:
            print(f"           obs: {step['observation'][:60]}...")
    print(f"\n  Final answer: {chain['final_answer']}")

    print(f"\n  Scoring with llm-guard-kit ...")
    guard = AgentGuard()
    result = guard.score_chain(question, chain["steps"], chain["final_answer"])

    print(f"\n  JUDGE VERDICT:")
    print(f"    risk_score:      {result.risk_score:.3f}")
    print(f"    confidence_tier: {result.confidence_tier}")
    print(f"    needs_alert:     {result.needs_alert}")
    print(f"    failure_mode:    {result.failure_mode or 'none'}")
    if hasattr(result, "behavioral_components") and result.behavioral_components:
        print(f"    top_signals:     {dict(list(result.behavioral_components.items())[:3])}")

    if result.needs_alert:
        print(f"\n  FLAGGED — activating recovery pipeline ...")

        # Activate adapter for this failure mode
        try:
            adapter = guard.activate_adapter(result.failure_mode)
            print(f"\n  ADAPTER ({result.failure_mode}):")
            print(f"    system_hint:     {adapter.config.system_hint[:80]}...")
            print(f"    search_strategy: {adapter.config.search_strategy[:60]}...")
        except Exception as e:
            print(f"    (adapter unavailable: {e})")

        # Rewrite query
        try:
            rewriter = QueryRewriter(api_key=api_key)
            variants = rewriter.rewrite(question, result)
            print(f"\n  QUERY REWRITER — 3 retry variants:")
            for j, v in enumerate(variants, 1):
                print(f"    {j}. {v}")
        except Exception as e:
            print(f"    (rewriter unavailable: {e})")

        print(f"\n  DECISION: Discard this answer. Retry with rewritten query.")
    else:
        print(f"\n  DECISION: Answer accepted (risk {result.risk_score:.3f} < threshold).")

    print(f"\n  Demo cost: ${llm.cost_usd:.4f}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="FARL Discovery Loop — autonomous failure hunting with llm-guard-kit"
    )
    parser.add_argument("--n",          type=int,   default=50,   help="Number of hunt iterations")
    parser.add_argument("--budget",     type=float, default=2.0,  help="Max spend in USD (default $2.00)")
    parser.add_argument("--threshold",  type=float, default=0.40, help="Risk score threshold for failure detection (default 0.40 for behavioral-only)")
    parser.add_argument("--novelty",    type=float, default=0.3,  help="Novelty threshold (0-1)")
    parser.add_argument("--resume",     action="store_true",      help="Resume from existing taxonomy.json")
    parser.add_argument("--demo",       action="store_true",      help="Run self-evaluation demo mode")
    parser.add_argument("--question",   type=str,   default="Who was the first person to walk on the moon, and what was the name of the mission?",
                        help="Question for --demo mode")
    args = parser.parse_args()

    api_key = _load_api_key()
    if not api_key:
        print("ERROR: No Anthropic API key found.")
        print("Set ANTHROPIC_API_KEY env var or add it to QPPG/.env")
        sys.exit(1)

    if args.demo:
        run_guarded_agent_demo(args.question, api_key)
        return

    loop = HuntLoop(
        api_key=api_key,
        n_iterations=args.n,
        risk_threshold=args.threshold,
        novelty_threshold=args.novelty,
        budget_usd=args.budget,
        resume=args.resume,
    )
    loop.run()


if __name__ == "__main__":
    main()
