#!/usr/bin/env python3
"""
Retrain MiniJudge on FARL hunt taxonomy + existing correct chains.
Uses full exp156 dataset (1600 chains) with held-out test split and bootstrap CI.

Usage:
    python3 scripts/retrain_mini_judge_from_taxonomy.py

Reads:
    results/farl_hunt/taxonomy.json       — novel wrong chains (from farl_hunt.py)
    results/exp156_live_crossdomain/*.json — existing chains (correct + wrong)

Writes:
    results/exp159_hybrid_mini_judge/hybrid_mini_judge.pkl  (in-place update)
    results/farl_hunt/retrain_report.json                   (AUROC before/after + bootstrap CI)
"""

import sys, json, random
from pathlib import Path

import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split
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


# ── Bootstrap CI ──────────────────────────────────────────────────────────────

def bootstrap_auroc_ci(y_true, y_score, n_boot=1000, ci=0.95, seed=42):
    """Compute bootstrap confidence interval for AUROC."""
    rng = np.random.RandomState(seed)
    y_true, y_score = np.array(y_true), np.array(y_score)
    n = len(y_true)
    boot_aurocs = []
    for _ in range(n_boot):
        idx = rng.randint(0, n, n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        boot_aurocs.append(roc_auc_score(y_true[idx], y_score[idx]))
    lo = np.percentile(boot_aurocs, (1 - ci) / 2 * 100)
    hi = np.percentile(boot_aurocs, (1 + ci) / 2 * 100)
    return float(np.mean(boot_aurocs)), float(lo), float(hi)


# ── Load wrong chains from taxonomy ──────────────────────────────────────────

def load_taxonomy_chains():
    if not TAXONOMY_PATH.exists():
        print(f"No taxonomy found at {TAXONOMY_PATH}")
        return [], []
    taxonomy = json.loads(TAXONOMY_PATH.read_text())
    X, y = [], []
    for mode, entries in taxonomy.items():
        for entry in entries:
            # Accept both Phase 1 ("step_details") and Phase 2 ("steps") field names
            steps = entry.get("step_details") or entry.get("steps")
            if not steps:
                continue
            chain = {
                "question":     entry.get("question", ""),
                "steps":        steps,
                "final_answer": entry.get("final_answer", ""),  # default empty — not all Phase 2 entries have this
            }
            try:
                feat = chain_to_features(chain)
                X.append(feat)
                y.append(1)   # wrong chain
            except Exception as e:
                print(f"  Skip entry (feature error): {e}")
    print(f"Loaded {len(X)} wrong chains from taxonomy")
    return X, y


# ── Load ALL chains from exp156 (no per-domain cap) ──────────────────────────

def load_exp156_chains():
    """Load all available chains; returns (X, y, raw_chains) for train/test splitting."""
    all_chains = []
    domain_files = sorted(EXP156_DIR.glob("*.json"))
    for fpath in domain_files:
        if "cache" in fpath.name or "results" in fpath.name:
            continue
        try:
            chains = json.loads(fpath.read_text())
        except Exception:
            continue
        domain = fpath.stem.replace("_chains", "")
        correct = sum(1 for c in chains if c.get("correct", False))
        wrong = len(chains) - correct
        print(f"  {fpath.name}: {len(chains)} chains ({correct} correct, {wrong} wrong)")
        all_chains.extend(chains)

    X, y = [], []
    valid_chains = []
    for chain in all_chains:
        try:
            feat = chain_to_features(chain)
            label = 0 if chain.get("correct", False) else 1
            X.append(feat)
            y.append(label)
            valid_chains.append(chain)
        except Exception:
            pass

    total_wrong = sum(y)
    print(f"Loaded {len(X)} chains from exp156 ({total_wrong} wrong, {len(y)-total_wrong} correct)")
    return X, y


# ── Evaluate model on a dataset ───────────────────────────────────────────────

def evaluate_model(pipe, X, y, label=""):
    probs = pipe.predict_proba(np.vstack(X))[:, 1]
    auroc = roc_auc_score(y, probs)
    mean_boot, lo, hi = bootstrap_auroc_ci(y, probs)
    if label:
        print(f"  {label}: AUROC={auroc:.4f}  95% CI [{lo:.4f}, {hi:.4f}]")
    return auroc, lo, hi


# ── Load original baseline model ─────────────────────────────────────────────

def load_original_baseline():
    """Load original HP-trained MiniJudge (before any retraining)."""
    orig_path = QPPG_ROOT / "results" / "exp159_hybrid_mini_judge" / "hybrid_mini_judge_original.pkl"
    if orig_path.exists():
        return joblib.load(orig_path)["pipeline"]
    # Fall back to current pkl
    return joblib.load(PKL_PATH)["pipeline"]


# ── Retrain ───────────────────────────────────────────────────────────────────

def retrain(X_train, y_train):
    X = np.vstack(X_train)
    y = np.array(y_train)

    print(f"\nTraining on {len(y)} chains ({sum(y)} wrong, {len(y)-sum(y)} correct)")

    new_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced")),
    ])

    # 5-fold stratified CV
    n_splits = min(5, max(2, min(sum(y), len(y) - sum(y)) // 2))
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = cross_val_score(new_pipe, X, y, cv=cv, scoring="roc_auc")
    cv_auroc = float(np.mean(scores))
    cv_std = float(np.std(scores))
    print(f"New pipeline {n_splits}-fold CV AUROC: {cv_auroc:.4f} ± {cv_std:.4f}")

    new_pipe.fit(X, y)
    return new_pipe, cv_auroc, cv_std


# ── Save updated pkl ──────────────────────────────────────────────────────────

def save_updated(pipe, cv_auroc, n_train):
    PKL_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Preserve original backup
    orig_path = QPPG_ROOT / "results" / "exp159_hybrid_mini_judge" / "hybrid_mini_judge_original.pkl"
    if PKL_PATH.exists() and not orig_path.exists():
        import shutil
        shutil.copy(PKL_PATH, orig_path)
        print(f"  Backed up original to {orig_path.name}")

    import llm_guard as _llm_guard
    data = {
        "pipeline":    pipe,
        "feature_set": f"SC_OLD + FARL taxonomy hardening (v{_llm_guard.__version__})",
        "embed_model": "all-MiniLM-L6-v2",
        "cv_auroc":    cv_auroc,
        "n_train":     n_train,
    }
    joblib.dump(data, PKL_PATH)
    print(f"Saved updated MiniJudge to {PKL_PATH}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  MiniJudge Retrain — Full Dataset + Statistical Evaluation")
    print("=" * 60)

    # ── 1. Load all exp156 chains ──
    print("\n--- Loading exp156 (full dataset, no cap) ---")
    X_exp, y_exp = load_exp156_chains()

    # ── 2. Train/test split on exp156 (stratified, 70/30) ──
    idx = list(range(len(y_exp)))
    idx_train, idx_test = train_test_split(
        idx, test_size=0.30, stratify=y_exp, random_state=42
    )
    X_exp_train = [X_exp[i] for i in idx_train]
    y_exp_train = [y_exp[i] for i in idx_train]
    X_exp_test  = [X_exp[i] for i in idx_test]
    y_exp_test  = [y_exp[i] for i in idx_test]
    print(f"\nTrain split: {len(y_exp_train)} chains ({sum(y_exp_train)} wrong, {len(y_exp_train)-sum(y_exp_train)} correct)")
    print(f"Test  split: {len(y_exp_test)} chains ({sum(y_exp_test)} wrong, {len(y_exp_test)-sum(y_exp_test)} correct)")

    # ── 3. Load taxonomy wrong chains (all go to train) ──
    print("\n--- Loading FARL taxonomy ---")
    X_tax, y_tax = load_taxonomy_chains()

    # ── 4. Combine train set ──
    X_train = X_tax + X_exp_train
    y_train = y_tax + y_exp_train
    print(f"\nFinal train set: {len(y_train)} chains ({sum(y_train)} wrong, {len(y_train)-sum(y_train)} correct)")

    if len(set(y_train)) < 2 or len(set(y_exp_test)) < 2:
        print("ERROR: Need both correct and wrong chains.")
        sys.exit(1)

    # ── 5. Evaluate BEFORE retraining ──
    print("\n--- Baseline (original HP-trained MiniJudge) on test set ---")
    orig_pipe = load_original_baseline()
    auroc_before, ci_lo_before, ci_hi_before = evaluate_model(
        orig_pipe, X_exp_test, y_exp_test, label="Original MiniJudge (cross-domain test)"
    )

    # ── 6. Retrain ──
    print("\n--- Retraining on train set ---")
    new_pipe, cv_auroc, cv_std = retrain(X_train, y_train)

    # ── 7. Evaluate AFTER on held-out test set ──
    print("\n--- Evaluation on held-out test set ---")
    auroc_after, ci_lo_after, ci_hi_after = evaluate_model(
        new_pipe, X_exp_test, y_exp_test, label="Retrained MiniJudge (cross-domain test)"
    )

    delta = auroc_after - auroc_before
    print(f"\n  Delta (test set): {delta:+.4f}")
    print(f"  CV AUROC (train): {cv_auroc:.4f} ± {cv_std:.4f}")

    # Statistical significance: non-overlapping 95% CIs (lower bound of after > UPPER bound of before)
    sig = "SIGNIFICANT" if ci_lo_after > ci_hi_before else "CIs overlap — more data needed"
    print(f"  Statistical signal: {sig}")

    # ── 8. Save ──
    print("\n--- Saving ---")
    save_updated(new_pipe, cv_auroc, len(y_train))

    # ── 9. Write report ──
    report = {
        "auroc_before":        round(auroc_before, 4),
        "auroc_before_ci":     [round(ci_lo_before, 4), round(ci_hi_before, 4)],
        "auroc_after_test":    round(auroc_after, 4),
        "auroc_after_ci":      [round(ci_lo_after, 4), round(ci_hi_after, 4)],
        "auroc_after_cv":      round(cv_auroc, 4),
        "auroc_after_cv_std":  round(cv_std, 4),
        "delta_test":          round(delta, 4),
        "n_taxonomy_wrong":    len(X_tax),
        "n_exp156_total":      len(y_exp),
        "n_train":             len(y_train),
        "n_test":              len(y_exp_test),
        "n_test_wrong":        sum(y_exp_test),
        "n_test_correct":      len(y_exp_test) - sum(y_exp_test),
        "statistical_signal":  sig,
        "bootstrap_n":         1000,
        "ci_level":            0.95,
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2))
    print(f"Report saved to {REPORT_PATH}")

    print("\n" + "=" * 60)
    print(f"  DONE  — Test AUROC {auroc_before:.4f} → {auroc_after:.4f} ({delta:+.4f})")
    print(f"         95% CI before: [{ci_lo_before:.4f}, {ci_hi_before:.4f}]")
    print(f"         95% CI after:  [{ci_lo_after:.4f}, {ci_hi_after:.4f}]")
    print(f"  {sig}")
    print("=" * 60)


if __name__ == "__main__":
    main()
