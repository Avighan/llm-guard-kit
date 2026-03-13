#!/usr/bin/env python3
"""
Retrain MiniJudge on FARL hunt taxonomy + existing correct chains.

Usage:
    python3 scripts/retrain_mini_judge_from_taxonomy.py

Reads:
    results/farl_hunt/taxonomy.json       — novel wrong chains (from farl_hunt.py)
    results/exp156_live_crossdomain/*.json — existing chains (for correct examples)

Writes:
    results/exp159_hybrid_mini_judge/hybrid_mini_judge.pkl  (in-place update)
    results/farl_hunt/retrain_report.json                   (AUROC before/after)
"""

import sys, json, statistics
from pathlib import Path

import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import roc_auc_score

QPPG_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(QPPG_ROOT))

from llm_guard.mini_judge import _extract_features   # 11-feature extractor

PKL_PATH       = QPPG_ROOT / "results" / "exp159_hybrid_mini_judge" / "hybrid_mini_judge.pkl"
TAXONOMY_PATH  = QPPG_ROOT / "results" / "farl_hunt" / "taxonomy.json"
EXP156_DIR     = QPPG_ROOT / "results" / "exp156_live_crossdomain"
REPORT_PATH    = QPPG_ROOT / "results" / "farl_hunt" / "retrain_report.json"


# ── Step-format normaliser ────────────────────────────────────────────────────

def _norm_steps(steps):
    """Convert exp156 step format (action_arg) to mini_judge format (action_args)."""
    normed = []
    for s in steps:
        normed.append({
            "thought":     s.get("thought", ""),
            "action_type": s.get("action_type", ""),
            "action_args": s.get("action_arg", s.get("action_args", "")),
            "observation": s.get("observation", ""),
        })
    return normed


def chain_to_features(chain: dict) -> np.ndarray:
    """Extract 11 MiniJudge features from any chain dict."""
    steps = _norm_steps(chain.get("steps", []))
    c = {
        "question":     chain.get("question", ""),
        "steps":        steps,
        "final_answer": chain.get("final_answer", ""),
    }
    return _extract_features(c)


# ── Load wrong chains from taxonomy ──────────────────────────────────────────

def load_taxonomy_chains():
    if not TAXONOMY_PATH.exists():
        print(f"No taxonomy found at {TAXONOMY_PATH}")
        return [], []
    taxonomy = json.loads(TAXONOMY_PATH.read_text())
    X, y = [], []
    for mode, entries in taxonomy.items():
        for entry in entries:
            if not entry.get("step_details"):
                continue
            chain = {
                "question":     entry["question"],
                "steps":        entry["step_details"],
                "final_answer": entry["final_answer"],
            }
            try:
                feat = chain_to_features(chain)
                X.append(feat)
                y.append(1)   # wrong chain
            except Exception as e:
                print(f"  Skip entry (feature error): {e}")
    print(f"Loaded {len(X)} wrong chains from taxonomy")
    return X, y


# ── Load correct + wrong chains from exp156 ──────────────────────────────────

def load_exp156_chains(max_per_domain=100):
    X, y = [], []
    domain_files = list(EXP156_DIR.glob("*.json"))
    for fpath in domain_files:
        if "cache" in fpath.name or "results" in fpath.name:
            continue
        try:
            chains = json.loads(fpath.read_text())
        except Exception:
            continue
        correct = [c for c in chains if c.get("correct", False)][:max_per_domain]
        wrong   = [c for c in chains if not c.get("correct", True)][:max_per_domain]
        for chain in correct + wrong:
            try:
                feat = chain_to_features(chain)
                label = 0 if chain.get("correct", False) else 1
                X.append(feat)
                y.append(label)
            except Exception:
                pass
        print(f"  {fpath.name}: {len(correct)} correct + {len(wrong)} wrong")
    print(f"Loaded {len(X)} chains from exp156 ({sum(y)} wrong, {len(y)-sum(y)} correct)")
    return X, y


# ── Evaluate current model (before retraining) ───────────────────────────────

def evaluate_current(X_all, y_all):
    obj = joblib.load(PKL_PATH)
    pipe = obj["pipeline"]
    probs = pipe.predict_proba(np.vstack(X_all))[:, 1]
    auroc = roc_auc_score(y_all, probs)
    print(f"Current MiniJudge AUROC (on combined dataset): {auroc:.4f}")
    return auroc


# ── Retrain ───────────────────────────────────────────────────────────────────

def retrain(X_all, y_all):
    X = np.vstack(X_all)
    y = np.array(y_all)

    print(f"\nTraining on {len(y)} chains ({sum(y)} wrong, {len(y)-sum(y)} correct)")

    new_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced")),
    ])

    # 5-fold stratified CV (or 3-fold if not enough samples per class)
    n_splits = min(5, max(2, min(sum(y), len(y) - sum(y)) // 2))
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = cross_val_score(new_pipe, X, y, cv=cv, scoring="roc_auc")
    cv_auroc = float(np.mean(scores))
    print(f"New pipeline {n_splits}-fold CV AUROC: {cv_auroc:.4f} ± {np.std(scores):.4f}")

    # Fit on full dataset
    new_pipe.fit(X, y)
    return new_pipe, cv_auroc


# ── Save updated pkl ──────────────────────────────────────────────────────────

def save_updated(pipe, cv_auroc):
    data = {
        "pipeline":    pipe,
        "feature_set": "SC_OLD + FARL taxonomy hardening",
        "embed_model": "all-MiniLM-L6-v2",
        "cv_auroc":    cv_auroc,
    }
    joblib.dump(data, PKL_PATH)
    print(f"Saved updated MiniJudge to {PKL_PATH}")
    print(f"  cv_auroc: {cv_auroc:.4f}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 55)
    print("  MiniJudge Retrain on FARL Taxonomy")
    print("=" * 55)

    # Load data
    X_tax, y_tax   = load_taxonomy_chains()
    X_exp, y_exp   = load_exp156_chains(max_per_domain=80)

    if not X_tax and not X_exp:
        print("ERROR: No training data found.")
        sys.exit(1)

    X_all = X_tax + X_exp
    y_all = y_tax + y_exp

    if len(set(y_all)) < 2:
        print("ERROR: Need both correct and wrong chains to train.")
        print(f"  y distribution: {sum(y_all)} wrong, {len(y_all)-sum(y_all)} correct")
        sys.exit(1)

    # Evaluate current model
    print("\n--- BEFORE retraining ---")
    auroc_before = evaluate_current(X_all, y_all)

    # Retrain
    print("\n--- Retraining ---")
    new_pipe, cv_auroc = retrain(X_all, y_all)

    # Evaluate new model
    print("\n--- AFTER retraining ---")
    X = np.vstack(X_all)
    y = np.array(y_all)
    probs_new = new_pipe.predict_proba(X)[:, 1]
    auroc_after_full = roc_auc_score(y, probs_new)
    print(f"New MiniJudge AUROC (full dataset, optimistic): {auroc_after_full:.4f}")
    print(f"CV AUROC (generalisation estimate):             {cv_auroc:.4f}")
    delta = cv_auroc - auroc_before
    print(f"\nImprovement (CV): {delta:+.4f}")

    # Save
    print("\n--- Saving ---")
    save_updated(new_pipe, cv_auroc)

    # Write report
    report = {
        "auroc_before":       round(auroc_before, 4),
        "auroc_after_cv":     round(cv_auroc, 4),
        "auroc_after_full":   round(auroc_after_full, 4),
        "delta_cv":           round(delta, 4),
        "n_taxonomy_wrong":   len(X_tax),
        "n_exp156_wrong":     sum(y_exp),
        "n_exp156_correct":   len(y_exp) - sum(y_exp),
        "n_total":            len(y_all),
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2))
    print(f"Report saved to {REPORT_PATH}")

    print("\n" + "=" * 55)
    print(f"  DONE  — AUROC {auroc_before:.4f} → {cv_auroc:.4f} ({delta:+.4f})")
    print("=" * 55)


if __name__ == "__main__":
    main()
