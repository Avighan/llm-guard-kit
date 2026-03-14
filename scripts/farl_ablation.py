#!/usr/bin/env python3
"""
FARL Ablation Study — proves FARL taxonomy contribution specifically.

Trains two MiniJudge models on the SAME exp156 chains:
  A) Baseline  — exp156 only (no FARL taxonomy)
  B) FARL      — exp156 + FARL taxonomy wrong chains

Both evaluated on the same held-out test set (30% of exp156, stratified).
Bootstrap CI confirms whether the delta is statistically significant.

Usage:
    python3 scripts/farl_ablation.py

Output:
    results/farl_hunt/ablation_report.json
"""

import sys, json
from pathlib import Path

import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.metrics import roc_auc_score

QPPG_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(QPPG_ROOT))

from llm_guard.mini_judge import _extract_features

TAXONOMY_PATH = QPPG_ROOT / "results" / "farl_hunt" / "taxonomy.json"
EXP156_DIR    = QPPG_ROOT / "results" / "exp156_live_crossdomain"
REPORT_PATH   = QPPG_ROOT / "results" / "farl_hunt" / "ablation_report.json"


def _norm_steps(steps):
    return [{
        "thought":     s.get("thought", ""),
        "action_type": s.get("action_type", ""),
        "action_args": s.get("action_arg", s.get("action_args", "")),
        "observation": s.get("observation", ""),
    } for s in steps]


def chain_to_features(chain):
    steps = _norm_steps(chain.get("steps", []))
    return _extract_features({
        "question":     chain.get("question", ""),
        "steps":        steps,
        "final_answer": chain.get("final_answer", ""),
    })


def bootstrap_auroc_ci(y_true, y_score, n_boot=1000, ci=0.95, seed=42):
    rng = np.random.RandomState(seed)
    y_true, y_score = np.array(y_true), np.array(y_score)
    n = len(y_true)
    boots = []
    for _ in range(n_boot):
        idx = rng.randint(0, n, n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        boots.append(roc_auc_score(y_true[idx], y_score[idx]))
    lo = np.percentile(boots, (1 - ci) / 2 * 100)
    hi = np.percentile(boots, (1 + ci) / 2 * 100)
    return float(np.mean(boots)), float(lo), float(hi)


def load_exp156():
    X, y = [], []
    for fpath in sorted(EXP156_DIR.glob("*.json")):
        if "cache" in fpath.name or "results" in fpath.name:
            continue
        chains = json.loads(fpath.read_text())
        c = sum(1 for ch in chains if ch.get("correct", False))
        w = len(chains) - c
        print(f"  {fpath.name}: {len(chains)} ({c} correct, {w} wrong)")
        for chain in chains:
            try:
                X.append(chain_to_features(chain))
                y.append(0 if chain.get("correct", False) else 1)
            except Exception:
                pass
    print(f"Loaded {len(X)} exp156 chains ({sum(y)} wrong, {len(y)-sum(y)} correct)")
    return X, y


def load_taxonomy():
    if not TAXONOMY_PATH.exists():
        return [], []
    taxonomy = json.loads(TAXONOMY_PATH.read_text())
    X, y = [], []
    for mode, entries in taxonomy.items():
        for entry in entries:
            # Accept both Phase 1 ("step_details") and Phase 2 ("steps") field names
            steps = entry.get("step_details") or entry.get("steps")
            if not steps:
                continue
            try:
                X.append(chain_to_features({
                    "question":     entry.get("question", ""),
                    "steps":        steps,
                    "final_answer": entry.get("final_answer", ""),  # defensive: Phase 2 entries may lack this
                }))
                y.append(1)
            except Exception:
                pass
    print(f"Loaded {len(X)} taxonomy wrong chains")
    return X, y


def train_and_eval(X_train, y_train, X_test, y_test, label):
    X = np.vstack(X_train)
    y = np.array(y_train)
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced")),
    ])
    n_splits = min(5, max(2, min(sum(y), len(y) - sum(y)) // 2))
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    cv_scores = cross_val_score(pipe, X, y, cv=cv, scoring="roc_auc")
    pipe.fit(X, y)

    probs = pipe.predict_proba(np.vstack(X_test))[:, 1]
    auroc = roc_auc_score(y_test, probs)
    mean_boot, lo, hi = bootstrap_auroc_ci(y_test, probs)

    print(f"\n  [{label}]")
    print(f"    Train: {len(y)} chains ({sum(y)} wrong, {len(y)-sum(y)} correct)")
    print(f"    CV AUROC:   {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")
    print(f"    Test AUROC: {auroc:.4f}  95% CI [{lo:.4f}, {hi:.4f}]")
    return auroc, lo, hi, float(np.mean(cv_scores))


def main():
    print("=" * 60)
    print("  FARL Ablation Study")
    print("  Baseline (exp156 only) vs FARL (exp156 + taxonomy)")
    print("=" * 60)

    print("\n--- Loading exp156 ---")
    X_exp, y_exp = load_exp156()

    print("\n--- Loading FARL taxonomy ---")
    X_tax, y_tax = load_taxonomy()

    if not X_tax:
        print("ERROR: No taxonomy found. Run farl_hunt.py first.")
        sys.exit(1)

    # Shared train/test split (same for both models — fair comparison)
    idx = list(range(len(y_exp)))
    idx_train, idx_test = train_test_split(
        idx, test_size=0.30, stratify=y_exp, random_state=42
    )
    X_train_base = [X_exp[i] for i in idx_train]
    y_train_base = [y_exp[i] for i in idx_train]
    X_test       = [X_exp[i] for i in idx_test]
    y_test       = [y_exp[i] for i in idx_test]

    print(f"\nShared test set: {len(y_test)} chains "
          f"({sum(y_test)} wrong, {len(y_test)-sum(y_test)} correct)")

    print("\n--- Training ---")

    # Model A: baseline (exp156 only)
    auroc_base, lo_base, hi_base, cv_base = train_and_eval(
        X_train_base, y_train_base, X_test, y_test,
        label="Baseline (exp156 only, no FARL)"
    )

    # Model B: FARL-hardened (exp156 + taxonomy)
    X_train_farl = X_tax + X_train_base
    y_train_farl = y_tax + y_train_base
    auroc_farl, lo_farl, hi_farl, cv_farl = train_and_eval(
        X_train_farl, y_train_farl, X_test, y_test,
        label=f"FARL-hardened (exp156 + {len(X_tax)} taxonomy chains)"
    )

    delta = auroc_farl - auroc_base
    # Non-overlapping 95% CIs: lower bound of FARL must exceed UPPER bound of baseline
    sig = "SIGNIFICANT" if lo_farl > hi_base else "CIs overlap — more taxonomy data needed"

    print("\n" + "=" * 60)
    print(f"  ABLATION RESULT")
    print(f"  Baseline:  {auroc_base:.4f}  [{lo_base:.4f}, {hi_base:.4f}]")
    print(f"  FARL:      {auroc_farl:.4f}  [{lo_farl:.4f}, {hi_farl:.4f}]")
    print(f"  Delta:     {delta:+.4f}")
    print(f"  Taxonomy:  {len(X_tax)} extra wrong chains from FARL hunt")
    print(f"  Signal:    {sig}")
    print("=" * 60)

    report = {
        "ablation": {
            "baseline_auroc":  round(auroc_base, 4),
            "baseline_ci":     [round(lo_base, 4), round(hi_base, 4)],
            "baseline_cv":     round(cv_base, 4),
            "farl_auroc":      round(auroc_farl, 4),
            "farl_ci":         [round(lo_farl, 4), round(hi_farl, 4)],
            "farl_cv":         round(cv_farl, 4),
            "delta":           round(delta, 4),
            "n_taxonomy":      len(X_tax),
            "n_test":          len(y_test),
            "n_test_wrong":    sum(y_test),
            "statistical_signal": sig,
        }
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2))
    print(f"\nReport saved to {REPORT_PATH}")


if __name__ == "__main__":
    main()
