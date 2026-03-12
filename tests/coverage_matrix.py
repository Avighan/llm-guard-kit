#!/usr/bin/env python3
"""
coverage_matrix.py — "Works Anywhere" coverage report for llm-guard-kit.

Run:
    python tests/coverage_matrix.py
    python tests/coverage_matrix.py --json          # machine-readable output
    python tests/coverage_matrix.py --promote-path  # show how to promote STRUCTURAL → VALIDATED

Exit codes:
    0 — report printed successfully
    1 — any VALIDATED entry is missing its AUROC number (data integrity check)
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from typing import List, Optional

# ── ANSI colour helpers ───────────────────────────────────────────────────────

_GREEN  = "\033[92m"
_YELLOW = "\033[93m"
_RED    = "\033[91m"
_CYAN   = "\033[96m"
_BOLD   = "\033[1m"
_DIM    = "\033[2m"
_RESET  = "\033[0m"

def _c(text: str, colour: str, no_colour: bool = False) -> str:
    if no_colour:
        return text
    return f"{colour}{text}{_RESET}"


# ── Matrix data ───────────────────────────────────────────────────────────────

STATUS_VALIDATED  = "VALIDATED"    # held-out AUROC measured
STATUS_STRUCTURAL = "STRUCTURAL"   # output valid, no AUROC claim
STATUS_NOT_TESTED = "NOT_TESTED"   # needs real data / API keys

@dataclass
class MatrixEntry:
    dimension: str
    condition: str
    status: str
    auroc: Optional[float] = None
    note: str = ""
    promote_steps: List[str] = field(default_factory=list)


MATRIX: List[MatrixEntry] = [

    # ── Datasets ──────────────────────────────────────────────────────────────
    MatrixEntry(
        dimension="Dataset",
        condition="HotpotQA (200 held-out chains)",
        status=STATUS_VALIDATED,
        auroc=0.812,
        note="SC_OLD 5-fold CV (exp88). Primary benchmark.",
    ),
    MatrixEntry(
        dimension="Dataset",
        condition="Natural Questions (NQ, 200 chains)",
        status=STATUS_VALIDATED,
        auroc=0.741,
        note="Cross-domain hold-out (exp89). Sonnet judge.",
    ),
    MatrixEntry(
        dimension="Dataset",
        condition="TriviaQA",
        status=STATUS_STRUCTURAL,
        note="Single-hop factual; step normalizer produces valid output. AUROC not measured.",
        promote_steps=[
            "1. Generate 200 TriviaQA ReAct chains (correct + incorrect mix).",
            "2. Run guard.score_chain() on each.",
            "3. Use ground-truth answer-match labels.",
            "4. Run sklearn roc_auc_score → add to table.",
        ],
    ),
    MatrixEntry(
        dimension="Dataset",
        condition="WebQ",
        status=STATUS_STRUCTURAL,
        note="Entity-centric; step normalizer produces valid output. AUROC not measured.",
        promote_steps=[
            "1. Generate 200 WebQ ReAct chains.",
            "2. Same pipeline as TriviaQA above.",
        ],
    ),
    MatrixEntry(
        dimension="Dataset",
        condition="FEVER (fact verification)",
        status=STATUS_STRUCTURAL,
        note="SUPPORTED/REFUTED chains; output is valid. AUROC not measured.",
        promote_steps=[
            "1. Generate 200 FEVER ReAct chains (balanced SUPPORTED/REFUTED).",
            "2. Use claim-level labels as ground truth.",
        ],
    ),

    # ── Agent frameworks ──────────────────────────────────────────────────────
    MatrixEntry(
        dimension="Agent Framework",
        condition="ReAct (native)",
        status=STATUS_VALIDATED,
        auroc=0.812,
        note="Native format. All AUROC claims refer to this format only.",
    ),
    MatrixEntry(
        dimension="Agent Framework",
        condition="OpenAI function-call",
        status=STATUS_STRUCTURAL,
        note="step_normalizer converts to canonical ReAct. Structural validity confirmed in test_cross_framework.py.",
        promote_steps=[
            "1. Run 200 GPT-4 tool-call chains on HotpotQA.",
            "2. Pass through score_chain(agent_format='openai').",
            "3. Measure AUROC vs answer-match labels.",
        ],
    ),
    MatrixEntry(
        dimension="Agent Framework",
        condition="LangGraph",
        status=STATUS_STRUCTURAL,
        note="step_normalizer converts node-graph steps. Structural validity confirmed.",
        promote_steps=[
            "1. Build LangGraph agent on HotpotQA (200 chains).",
            "2. Pass through score_chain(agent_format='langgraph').",
            "3. Measure AUROC.",
        ],
    ),
    MatrixEntry(
        dimension="Agent Framework",
        condition="AutoGen",
        status=STATUS_STRUCTURAL,
        note="step_normalizer converts sender/content dicts. Structural validity confirmed.",
        promote_steps=[
            "1. Run AutoGen multi-agent on 200 questions.",
            "2. Collect step dicts per chain.",
            "3. Measure AUROC.",
        ],
    ),
    MatrixEntry(
        dimension="Agent Framework",
        condition="LangChain AgentExecutor",
        status=STATUS_STRUCTURAL,
        note="step_normalizer converts tool+log format. Structural validity confirmed.",
        promote_steps=[
            "1. Run LangChain MRKL/ReAct agent on 200 HotpotQA questions.",
            "2. Measure AUROC.",
        ],
    ),
    MatrixEntry(
        dimension="Agent Framework",
        condition="CrewAI",
        status=STATUS_NOT_TESTED,
        note="No test coverage yet. CrewAI step format undocumented.",
        promote_steps=[
            "1. Inspect CrewAI task output structure.",
            "2. Add _norm_crewai() to step_normalizer.py.",
            "3. Add structural test to test_cross_framework.py.",
            "4. Collect 200 chains + measure AUROC.",
        ],
    ),

    # ── LLM backbones ─────────────────────────────────────────────────────────
    MatrixEntry(
        dimension="LLM Backbone",
        condition="Claude Sonnet (claude-sonnet-4-6)",
        status=STATUS_VALIDATED,
        auroc=0.812,
        note="All experiments run on Claude Sonnet chains. Primary validated backbone.",
    ),
    MatrixEntry(
        dimension="LLM Backbone",
        condition="GPT-4",
        status=STATUS_STRUCTURAL,
        note=(
            "SC_OLD measures step structure (count, backtracking, coherence), not content. "
            "Plausible backbone-agnostic, but AUROC not measured on GPT-4 chains."
        ),
        promote_steps=[
            "1. Run GPT-4 ReAct agent on 200 HotpotQA questions.",
            "2. score_chain() + roc_auc_score → confirm AUROC ≥ 0.70.",
        ],
    ),
    MatrixEntry(
        dimension="LLM Backbone",
        condition="Gemini (gemini-pro / flash)",
        status=STATUS_STRUCTURAL,
        note="Structural validity confirmed (synthetic Gemini-style chains). AUROC not measured.",
        promote_steps=[
            "1. Run Gemini ReAct agent on 200 HotpotQA questions.",
            "2. Measure AUROC.",
        ],
    ),
    MatrixEntry(
        dimension="LLM Backbone",
        condition="Llama-3-8B / 70B",
        status=STATUS_STRUCTURAL,
        note="Structural validity confirmed. AUROC not measured; format may differ.",
        promote_steps=[
            "1. Run Llama-3 ReAct agent on 200 questions.",
            "2. Check step format — may need step_normalizer additions.",
            "3. Measure AUROC.",
        ],
    ),

    # ── Step count distribution ───────────────────────────────────────────────
    MatrixEntry(
        dimension="Step Count",
        condition="2–8 steps (typical)",
        status=STATUS_VALIDATED,
        auroc=0.812,
        note="Primary validation range.",
    ),
    MatrixEntry(
        dimension="Step Count",
        condition="1 step (degenerate short)",
        status=STATUS_STRUCTURAL,
        note=(
            "score_chain() returns valid result with SC2 warning. "
            "AUROC meaningless — not enough steps for behavioral signal."
        ),
        promote_steps=[
            "1. Collect at least 100 single-step chains with ground-truth labels.",
            "2. Measure AUROC — expected LOW (scorer needs ≥2 steps).",
            "3. Document in docs/limitations.md.",
        ],
    ),
    MatrixEntry(
        dimension="Step Count",
        condition="10–15+ steps (long chains)",
        status=STATUS_STRUCTURAL,
        note=(
            "score_chain() runs without error on 10–15 step chains. "
            "SC2 (raw step count) saturates; other features still active. AUROC not measured."
        ),
        promote_steps=[
            "1. Collect 100 long-chain examples (>10 steps) with labels.",
            "2. Measure AUROC — expected similar or slightly lower than typical range.",
        ],
    ),

    # ── Language ──────────────────────────────────────────────────────────────
    MatrixEntry(
        dimension="Language",
        condition="English",
        status=STATUS_VALIDATED,
        auroc=0.812,
        note="All validation data is English.",
    ),
    MatrixEntry(
        dimension="Language",
        condition="French",
        status=STATUS_STRUCTURAL,
        note="Valid output + non-English warning emitted. AUROC not measured.",
        promote_steps=[
            "1. Collect 100 French ReAct chains with labels.",
            "2. Measure AUROC — SC_OLD is partly language-agnostic (step count, backtracking).",
        ],
    ),
    MatrixEntry(
        dimension="Language",
        condition="Spanish",
        status=STATUS_STRUCTURAL,
        note="Valid output + non-English warning emitted. AUROC not measured.",
    ),
    MatrixEntry(
        dimension="Language",
        condition="Chinese (Simplified)",
        status=STATUS_STRUCTURAL,
        note="Valid output + non-English warning emitted. AUROC not measured.",
    ),
    MatrixEntry(
        dimension="Language",
        condition="Arabic (RTL)",
        status=STATUS_STRUCTURAL,
        note="Valid output + non-English warning emitted. AUROC not measured.",
    ),
]


# ── Printing ──────────────────────────────────────────────────────────────────

def _status_cell(status: str, auroc: Optional[float], no_colour: bool) -> str:
    if status == STATUS_VALIDATED:
        label = f"VALIDATED  (AUROC {auroc:.3f})" if auroc is not None else "VALIDATED"
        return _c(label, _GREEN, no_colour)
    elif status == STATUS_STRUCTURAL:
        return _c("STRUCTURAL (valid output, no AUROC)", _YELLOW, no_colour)
    else:
        return _c("NOT TESTED", _RED, no_colour)


def print_matrix(entries: List[MatrixEntry], no_colour: bool = False) -> None:
    print()
    header = _c("  llm-guard-kit v0.8.0 — 'Works Anywhere' Coverage Matrix", _BOLD, no_colour)
    print(header)
    print(_c("  " + "─" * 78, _DIM, no_colour))

    current_dim = None
    for e in entries:
        if e.dimension != current_dim:
            current_dim = e.dimension
            print()
            print(_c(f"  [{e.dimension}]", _CYAN + _BOLD, no_colour))
        status_str = _status_cell(e.status, e.auroc, no_colour)
        cond_str   = f"    {e.condition:<42}"
        print(f"{cond_str}  {status_str}")
        if e.note:
            note_lines = _wrap(e.note, width=70, indent=8)
            for line in note_lines:
                print(_c(line, _DIM, no_colour))

    print()
    _print_legend(no_colour)
    _print_summary(entries, no_colour)


def _print_legend(no_colour: bool) -> None:
    print(_c("  Legend:", _BOLD, no_colour))
    print("  " + _c("VALIDATED ", _GREEN,  no_colour) + " held-out AUROC measured on real chains")
    print("  " + _c("STRUCTURAL", _YELLOW, no_colour) + " output is valid; AUROC not measured yet")
    print("  " + _c("NOT TESTED", _RED,    no_colour) + " no test coverage; needs implementation")
    print()


def _print_summary(entries: List[MatrixEntry], no_colour: bool) -> None:
    counts = {STATUS_VALIDATED: 0, STATUS_STRUCTURAL: 0, STATUS_NOT_TESTED: 0}
    for e in entries:
        counts[e.status] += 1
    total = len(entries)
    pct_v = 100 * counts[STATUS_VALIDATED]  // total
    pct_s = 100 * counts[STATUS_STRUCTURAL] // total
    pct_n = 100 * counts[STATUS_NOT_TESTED] // total
    print(_c("  Summary:", _BOLD, no_colour))
    print(f"    Total conditions : {total}")
    print(f"    {_c('VALIDATED ', _GREEN,  no_colour)}: {counts[STATUS_VALIDATED]:>2}  ({pct_v}%)")
    print(f"    {_c('STRUCTURAL', _YELLOW, no_colour)}: {counts[STATUS_STRUCTURAL]:>2}  ({pct_s}%)")
    print(f"    {_c('NOT TESTED', _RED,    no_colour)}: {counts[STATUS_NOT_TESTED]:>2}  ({pct_n}%)")
    print()


def print_promote_guide(entries: List[MatrixEntry], no_colour: bool = False) -> None:
    """Print step-by-step instructions to promote STRUCTURAL → VALIDATED."""
    print()
    print(_c("  Promotion Guide: STRUCTURAL → VALIDATED", _BOLD, no_colour))
    print(_c("  " + "─" * 60, _DIM, no_colour))
    for e in entries:
        if e.status == STATUS_STRUCTURAL and e.promote_steps:
            print()
            print(_c(f"  [{e.dimension}] {e.condition}", _CYAN, no_colour))
            for step in e.promote_steps:
                print(f"    {step}")
    print()


def _wrap(text: str, width: int = 72, indent: int = 6) -> List[str]:
    words = text.split()
    lines, line = [], []
    cur = 0
    for w in words:
        if cur + len(w) + 1 > width:
            lines.append(" " * indent + " ".join(line))
            line, cur = [w], len(w)
        else:
            line.append(w)
            cur += len(w) + 1
    if line:
        lines.append(" " * indent + " ".join(line))
    return lines


# ── JSON output ───────────────────────────────────────────────────────────────

def to_json(entries: List[MatrixEntry]) -> str:
    rows = []
    for e in entries:
        rows.append({
            "dimension":     e.dimension,
            "condition":     e.condition,
            "status":        e.status,
            "auroc":         e.auroc,
            "note":          e.note,
            "promote_steps": e.promote_steps,
        })
    return json.dumps({"entries": rows, "total": len(rows)}, indent=2)


# ── Data integrity check ──────────────────────────────────────────────────────

def check_integrity(entries: List[MatrixEntry]) -> List[str]:
    errors = []
    for e in entries:
        if e.status == STATUS_VALIDATED and e.auroc is None:
            errors.append(f"VALIDATED entry '{e.condition}' is missing auroc value.")
    return errors


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Print the 'works anywhere' coverage matrix for llm-guard-kit."
    )
    parser.add_argument("--json",         action="store_true", help="Output as JSON")
    parser.add_argument("--promote-path", action="store_true", help="Show how to promote STRUCTURAL → VALIDATED")
    parser.add_argument("--no-colour",    action="store_true", help="Disable ANSI colour output")
    args = parser.parse_args()

    errors = check_integrity(MATRIX)
    if errors:
        for err in errors:
            print(f"[DATA ERROR] {err}", file=sys.stderr)
        return 1

    if args.json:
        print(to_json(MATRIX))
        return 0

    print_matrix(MATRIX, no_colour=args.no_colour)

    if args.promote_path:
        print_promote_guide(MATRIX, no_colour=args.no_colour)

    return 0


if __name__ == "__main__":
    sys.exit(main())
