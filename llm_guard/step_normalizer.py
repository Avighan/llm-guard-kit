"""
step_normalizer.py — Multi-format agent step normalization for AgentGuard.
==========================================================================

Converts agent steps from any common format into the canonical ReAct dict:
    {"thought": str, "action_type": str, "action_arg": str, "observation": str}

Supported formats
-----------------
  "react"       Native {thought, action_type, action_arg, observation} — passthrough
  "openai"      OpenAI function-call / tool-call message list
  "langgraph"   LangGraph AIMessage / ToolMessage list (same wire shape as OpenAI)
  "autogen"     AutoGen ConversableAgent message dicts {sender, content, role}
  "langchain"   LangChain AgentAction dicts {tool, tool_input, log, observation}
  "auto"        (default) Attempt format detection then normalise

Claim validity warnings
-----------------------
SC_OLD was validated on ReAct chains (HotpotQA/NQ, English, 2-8 steps).
This module emits UserWarning when:
  - Chain has fewer than 2 Search steps (SC2 loses discriminative power)
  - Format is non-native ReAct (scores may be less calibrated)
  - Observation fields are mostly empty (grounding signals degrade)
  - Steps contain non-English text (Jaccard signals calibrated on English)

To suppress warnings:  import warnings; warnings.filterwarnings("ignore", category=UserWarning, module="llm_guard")
"""

from __future__ import annotations

import json
import re
import warnings
from typing import Any, Dict, List, Optional

# Canonical step keys
_STEP_KEYS = ("thought", "action_type", "action_arg", "observation")

# Heuristic for short-chain warning
_MIN_SEARCH_STEPS = 2

# Non-ASCII fraction threshold for non-English warning
_NON_ASCII_THRESHOLD = 0.15


def normalize_steps(
    steps: List[Any],
    agent_format: str = "auto",
    warn: bool = True,
) -> List[Dict[str, str]]:
    """
    Normalize a list of agent steps to canonical ReAct format.

    Parameters
    ----------
    steps : list
        Raw steps in any supported format.
    agent_format : str
        One of: "react", "openai", "langgraph", "autogen", "langchain", "auto".
        "auto" attempts format detection.
    warn : bool
        If True (default), emit UserWarning when claim conditions may not hold.

    Returns
    -------
    list of dict
        Each dict has keys: thought, action_type, action_arg, observation.
        All values are strings; missing fields default to "".

    Raises
    ------
    ValueError
        If agent_format is not a recognised value.
    """
    if not steps:
        return []

    fmt = agent_format.lower()
    if fmt == "auto":
        fmt = _detect_format(steps)

    if fmt == "react":
        normalised = _norm_react(steps)
    elif fmt in ("openai", "langgraph"):
        normalised = _norm_openai(steps)
    elif fmt == "autogen":
        normalised = _norm_autogen(steps)
    elif fmt == "langchain":
        normalised = _norm_langchain(steps)
    else:
        raise ValueError(
            f"Unknown agent_format '{agent_format}'. "
            "Choose from: react, openai, langgraph, autogen, langchain, auto."
        )

    if warn:
        _emit_warnings(normalised, fmt)

    return normalised


# ── Format detection ──────────────────────────────────────────────────────────

def _detect_format(steps: List[Any]) -> str:
    """Heuristically detect the step format from the first step."""
    if not steps:
        return "react"
    s = steps[0]
    if not isinstance(s, dict):
        return "react"  # best effort — let normaliser handle it

    keys = set(s.keys())

    # OpenAI / LangGraph: message dicts with "role" key
    if "role" in keys:
        return "openai"

    # AutoGen: dicts with "sender" key
    if "sender" in keys:
        return "autogen"

    # LangChain: AgentAction has "tool" and "tool_input" or "log"
    if "tool" in keys and ("tool_input" in keys or "log" in keys):
        return "langchain"

    # ReAct native (may have partial keys)
    return "react"


# ── Format normalisers ────────────────────────────────────────────────────────

def _make_step(
    thought: str = "",
    action_type: str = "Action",
    action_arg: str = "",
    observation: str = "",
) -> Dict[str, str]:
    return {
        "thought": str(thought or ""),
        "action_type": str(action_type or "Action"),
        "action_arg": str(action_arg or ""),
        "observation": str(observation or ""),
    }


def _norm_react(steps: List[Any]) -> List[Dict[str, str]]:
    """Pass-through with key normalisation for native ReAct dicts."""
    out = []
    for s in steps:
        if not isinstance(s, dict):
            out.append(_make_step(thought=str(s)))
            continue
        out.append(_make_step(
            thought      = s.get("thought", s.get("thinking", "")),
            action_type  = s.get("action_type", s.get("action", "Action")),
            action_arg   = s.get("action_arg", s.get("action_input", s.get("query", ""))),
            observation  = s.get("observation", s.get("result", s.get("output", ""))),
        ))
    return out


def _norm_openai(steps: List[Any]) -> List[Dict[str, str]]:
    """
    Normalise OpenAI function-call / tool-call message lists.

    Expected pattern (pairs or interleaved):
      {role: "assistant", content: "...", tool_calls: [{function: {name, arguments}}]}
      {role: "tool", content: "...", tool_call_id: "..."}

    Also handles plain assistant + user message pairs (no tool_calls).
    """
    out: List[Dict[str, str]] = []
    i = 0
    while i < len(steps):
        msg = steps[i]
        if not isinstance(msg, dict):
            i += 1
            continue

        role = msg.get("role", "")
        content = _extract_content(msg)

        if role == "assistant":
            thought = content
            action_type = "Action"
            action_arg = ""
            tool_calls = msg.get("tool_calls") or []

            if tool_calls:
                tc = tool_calls[0]
                fn = tc.get("function", tc) if isinstance(tc, dict) else {}
                action_type = str(fn.get("name", "Tool"))
                raw_args = fn.get("arguments", "")
                if isinstance(raw_args, str):
                    try:
                        parsed = json.loads(raw_args)
                        action_arg = (
                            parsed.get("query", parsed.get("input", raw_args))
                            if isinstance(parsed, dict) else raw_args
                        )
                    except (json.JSONDecodeError, ValueError):
                        action_arg = raw_args
                elif isinstance(raw_args, dict):
                    action_arg = str(raw_args.get("query", raw_args.get("input", raw_args)))
                else:
                    action_arg = str(raw_args)

            # Look ahead for the tool / user response
            observation = ""
            if i + 1 < len(steps):
                nxt = steps[i + 1]
                if isinstance(nxt, dict) and nxt.get("role") in ("tool", "user"):
                    observation = _extract_content(nxt)
                    i += 1  # consume the observation message

            out.append(_make_step(thought, action_type, action_arg, observation))

        elif role in ("tool", "user") and out:
            # Orphaned observation — attach to the last step
            out[-1]["observation"] = (out[-1]["observation"] + " " + content).strip()

        i += 1

    return out or [_make_step(thought=_extract_content(steps[0]) if steps else "")]


def _norm_autogen(steps: List[Any]) -> List[Dict[str, str]]:
    """
    Normalise AutoGen ConversableAgent message dicts.

    Common shape: {name/sender: str, role: str, content: str}
    AutoGen alternates assistant → user (tool result) → assistant …
    """
    out: List[Dict[str, str]] = []
    i = 0
    while i < len(steps):
        msg = steps[i]
        if not isinstance(msg, dict):
            i += 1
            continue
        sender = msg.get("sender", msg.get("name", ""))
        content = _extract_content(msg)
        role = msg.get("role", "")

        is_agent = role == "assistant" or (sender and sender.lower() not in ("user", "human", "tool"))

        if is_agent:
            # Try to parse "Action: X\nAction Input: Y" from content
            thought, action_type, action_arg = _parse_react_text(content)
            observation = ""
            if i + 1 < len(steps):
                nxt = steps[i + 1]
                if isinstance(nxt, dict):
                    nxt_sender = nxt.get("sender", nxt.get("name", ""))
                    nxt_role = nxt.get("role", "")
                    is_tool_response = (
                        nxt_role in ("tool", "function", "user")
                        or nxt_sender.lower() in ("user", "human", "tool", "userproxyagent")
                    )
                    if is_tool_response:
                        observation = _extract_content(nxt)
                        i += 1
            out.append(_make_step(thought, action_type, action_arg, observation))

        i += 1

    return out or [_make_step(thought=_extract_content(steps[0]) if steps else "")]


def _norm_langchain(steps: List[Any]) -> List[Dict[str, str]]:
    """
    Normalise LangChain AgentAction dicts.

    LangChain typically returns intermediate_steps as list of (AgentAction, str) tuples
    or dicts with keys: tool, tool_input, log, (observation appended separately).

    Handles both dict form and (action, observation) tuple form.
    """
    out: List[Dict[str, str]] = []
    for step in steps:
        # Tuple form: (AgentAction, observation_str)
        if isinstance(step, (tuple, list)) and len(step) == 2:
            action, obs = step
            if hasattr(action, "tool"):  # LangChain AgentAction object
                out.append(_make_step(
                    thought     = getattr(action, "log", ""),
                    action_type = getattr(action, "tool", "Tool"),
                    action_arg  = str(getattr(action, "tool_input", "")),
                    observation = str(obs),
                ))
            elif isinstance(action, dict):
                out.append(_make_step(
                    thought     = action.get("log", ""),
                    action_type = action.get("tool", "Tool"),
                    action_arg  = str(action.get("tool_input", "")),
                    observation = str(obs),
                ))
            continue

        # Dict form (already has observation key or not)
        if isinstance(step, dict):
            out.append(_make_step(
                thought     = step.get("log", step.get("thought", "")),
                action_type = step.get("tool", step.get("action_type", "Tool")),
                action_arg  = str(step.get("tool_input", step.get("action_arg", ""))),
                observation = step.get("observation", ""),
            ))
            continue

        # AgentAction object
        if hasattr(step, "tool"):
            out.append(_make_step(
                thought     = getattr(step, "log", ""),
                action_type = getattr(step, "tool", "Tool"),
                action_arg  = str(getattr(step, "tool_input", "")),
            ))

    return out or [_make_step()]


# ── Helper utilities ──────────────────────────────────────────────────────────

def _extract_content(msg: Dict) -> str:
    """Extract text content from a message dict."""
    c = msg.get("content", "")
    if isinstance(c, list):
        # OpenAI content blocks: [{type: "text", text: "..."}]
        parts = []
        for block in c:
            if isinstance(block, dict):
                parts.append(block.get("text", str(block)))
            else:
                parts.append(str(block))
        return " ".join(parts)
    return str(c or "")


def _parse_react_text(text: str):
    """
    Extract thought, action_type, action_arg from plain-text ReAct output.
    Returns (thought, action_type, action_arg).
    """
    thought = text
    action_type = "Action"
    action_arg = ""

    # Match "Action: SearchTool\nAction Input: query text"
    action_match = re.search(r"Action\s*:\s*(.+?)(?:\n|$)", text, re.IGNORECASE)
    input_match = re.search(r"Action\s*Input\s*:\s*(.+?)(?:\n|$)", text, re.IGNORECASE)
    thought_match = re.search(r"Thought\s*:\s*(.+?)(?:\nAction|$)", text, re.IGNORECASE | re.DOTALL)

    if thought_match:
        thought = thought_match.group(1).strip()
    if action_match:
        action_type = action_match.group(1).strip()
    if input_match:
        action_arg = input_match.group(1).strip()

    return thought, action_type, action_arg


# ── Claim-validity warnings ───────────────────────────────────────────────────

def _emit_warnings(steps: List[Dict[str, str]], fmt: str) -> None:
    """Emit UserWarnings when scoring conditions deviate from validated range."""

    if not steps:
        return

    # Warning 1: Non-native format
    if fmt != "react":
        warnings.warn(
            f"[llm-guard] Steps were normalised from '{fmt}' format to ReAct. "
            "SC_OLD behavioral signals were validated on native ReAct chains "
            "(HotpotQA/NQ). Scores on non-ReAct formats may be less calibrated. "
            "Validate on a sample of your own data before relying on AUROC claims.",
            UserWarning,
            stacklevel=4,
        )

    # Warning 2: Short chain — SC2 (step count) loses discriminative power
    search_steps = [s for s in steps if s.get("action_type", "").lower() not in ("finish", "final")]
    if len(search_steps) < _MIN_SEARCH_STEPS:
        warnings.warn(
            f"[llm-guard] Chain has only {len(search_steps)} non-Finish step(s). "
            "SC2 (step count, AUROC ~0.88 standalone) loses discriminative power "
            "on chains with < 2 Search steps. Behavioral AUROC may be lower than "
            "the validated 0.812. Consider using use_judge=True for short chains.",
            UserWarning,
            stacklevel=4,
        )

    # Warning 3: Empty observations — grounding signals degrade
    obs_list = [s.get("observation", "") for s in steps]
    empty_obs_frac = sum(1 for o in obs_list if not o.strip()) / max(len(obs_list), 1)
    if empty_obs_frac > 0.5:
        warnings.warn(
            f"[llm-guard] {empty_obs_frac:.0%} of steps have empty observations. "
            "SC9 (context utilisation), SC10 (coherence), and SC3 (thought variance) "
            "rely on observation content. Scores may underestimate risk for chains "
            "where tool results are not captured.",
            UserWarning,
            stacklevel=4,
        )

    # Warning 4: Non-English text
    all_text = " ".join(
        s.get("thought", "") + " " + s.get("observation", "") for s in steps
    )
    if all_text:
        non_ascii = sum(1 for c in all_text if ord(c) > 127)
        if non_ascii / len(all_text) > _NON_ASCII_THRESHOLD:
            warnings.warn(
                "[llm-guard] Detected non-ASCII characters (possible non-English text). "
                "Jaccard-based signals (SC3, SC6, SC9-12) were calibrated on English "
                "Wikipedia text. Cross-lingual accuracy is not validated.",
                UserWarning,
                stacklevel=4,
            )


# ── Convenience: validate a full batch ───────────────────────────────────────

def validate_step_coverage(
    steps: List[Dict[str, str]],
) -> Dict[str, Any]:
    """
    Return a coverage report for a normalised step list.
    Useful for debugging and for pre-flight checks before relying on claims.

    Returns dict with keys:
        n_steps, n_search_steps, n_finish_steps,
        obs_fill_rate, avg_thought_len, avg_obs_len,
        format_warnings: list of str
    """
    if not steps:
        return {"n_steps": 0, "format_warnings": ["empty step list"]}

    n = len(steps)
    search = [s for s in steps if s.get("action_type", "").lower() not in ("finish",)]
    finish = [s for s in steps if s.get("action_type", "").lower() == "finish"]
    obs_filled = [s for s in steps if s.get("observation", "").strip()]
    thought_lens = [len(s.get("thought", "")) for s in steps]
    obs_lens = [len(s.get("observation", "")) for s in steps]

    fw = []
    if len(search) < 2:
        fw.append(f"short_chain: only {len(search)} search step(s)")
    if len(obs_filled) / n < 0.5:
        fw.append(f"sparse_observations: {len(obs_filled)}/{n} steps have observations")

    return {
        "n_steps": n,
        "n_search_steps": len(search),
        "n_finish_steps": len(finish),
        "obs_fill_rate": round(len(obs_filled) / n, 3),
        "avg_thought_len": round(sum(thought_lens) / n, 1),
        "avg_obs_len": round(sum(obs_lens) / n, 1),
        "format_warnings": fw,
    }
