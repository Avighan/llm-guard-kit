"""
retrain_mini_judge.py — Monthly MiniJudge retraining from community labels.

Combines telemetry-collected labels.jsonl with existing exp159 training chains,
trains a new LogReg + MLP ensemble, and saves the model if AUROC improves.

Usage
-----
    python scripts/retrain_mini_judge.py --labels path/to/labels.jsonl

The script exits with code 0 in both the "improved" and "no improvement" cases.
It exits with code 1 only on a fatal error (e.g. missing required files).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List, Dict, Optional

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_QPPG_ROOT = Path(__file__).resolve().parent.parent
_EXISTING_CHAINS_PATH = (
    _QPPG_ROOT / "results" / "large_dataset" / "hotpot_qa_chains.json"
)
_PKL_DIR = _QPPG_ROOT / "results" / "exp159_hybrid_mini_judge"
_PKL_PATH = _PKL_DIR / "hybrid_mini_judge.pkl"


# ---------------------------------------------------------------------------
# Feature extraction (re-uses mini_judge._extract_features)
# ---------------------------------------------------------------------------

def _load_mini_judge_features():
    """Import _extract_features from llm_guard.mini_judge."""
    sys.path.insert(0, str(_QPPG_ROOT))
    from llm_guard.mini_judge import _extract_features  # noqa: PLC0415
    return _extract_features


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_jsonl(path: Path) -> List[dict]:
    """Read a .jsonl file, skipping blank lines and malformed entries."""
    records = []
    with open(path, "r", encoding="utf-8") as fh:
        for i, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                logger.warning("Skipping line %d in %s: %s", i, path, exc)
    return records


def _load_telemetry_labels(labels_path: Path, extract_features):
    """
    Load community-contributed labels from labels.jsonl.

    Each record format:
        {"f": [11 floats], "y": 0|1, "d": "a3f9bc", "v": "0.17.0"}

    Returns (X, y) numpy arrays.
    """
    records = _load_jsonl(labels_path)
    X_rows, y_rows = [], []
    for rec in records:
        f = rec.get("f")
        y = rec.get("y")
        if f is None or y is None:
            continue
        if len(f) != 11:
            logger.warning("Skipping telemetry record with %d features (expected 11)", len(f))
            continue
        X_rows.append(f)
        y_rows.append(int(y))
    logger.info("Loaded %d telemetry labels from %s", len(X_rows), labels_path)
    return np.array(X_rows, dtype=np.float64), np.array(y_rows, dtype=int)


def _load_existing_chains(chains_path: Path, extract_features):
    """
    Load exp159 HotpotQA training chains from hotpot_qa_chains.json.

    Expects a JSON list of chain dicts with keys:
        question, steps, final_answer, correct (bool/int)

    Returns (X, y) numpy arrays.
    """
    with open(chains_path, "r", encoding="utf-8") as fh:
        chains: List[Dict] = json.load(fh)
    X_rows, y_rows = [], []
    for c in chains:
        try:
            feats = extract_features(c)
            label = int(not bool(c.get("correct", True)))
            X_rows.append(feats)
            y_rows.append(label)
        except Exception as exc:
            logger.warning("Skipping chain: %s", exc)
    logger.info("Loaded %d chains from %s", len(X_rows), chains_path)
    return np.array(X_rows, dtype=np.float64), np.array(y_rows, dtype=int)


# ---------------------------------------------------------------------------
# Model training
# ---------------------------------------------------------------------------

def _train_and_evaluate(X: np.ndarray, y: np.ndarray, n_splits: int = 5):
    """
    Train a LogReg + MLP ensemble via 5-fold CV and return (pipeline, cv_auroc).

    The ensemble averages predicted probabilities from:
        - StandardScaler + LogisticRegression (exp159 baseline)
        - StandardScaler + MLPClassifier (hidden_layer_sizes=(64, 32))
    """
    from sklearn.linear_model import LogisticRegression  # type: ignore
    from sklearn.neural_network import MLPClassifier  # type: ignore
    from sklearn.pipeline import Pipeline  # type: ignore
    from sklearn.preprocessing import StandardScaler  # type: ignore
    from sklearn.model_selection import StratifiedKFold, cross_val_score  # type: ignore
    from sklearn.base import BaseEstimator, ClassifierMixin  # type: ignore

    class EnsemblePipeline(BaseEstimator, ClassifierMixin):
        """Average-probability ensemble of LogReg and MLP."""

        def __init__(self):
            self.lr = Pipeline([
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=1000, random_state=42)),
            ])
            self.mlp = Pipeline([
                ("scaler", StandardScaler()),
                ("clf", MLPClassifier(
                    hidden_layer_sizes=(64, 32),
                    max_iter=500,
                    random_state=42,
                    early_stopping=True,
                    validation_fraction=0.1,
                )),
            ])

        def fit(self, X, y):
            self.lr.fit(X, y)
            self.mlp.fit(X, y)
            self.classes_ = np.array([0, 1])
            return self

        def predict_proba(self, X):
            p_lr = self.lr.predict_proba(X)
            p_mlp = self.mlp.predict_proba(X)
            return (p_lr + p_mlp) / 2.0

        def predict(self, X):
            proba = self.predict_proba(X)
            return (proba[:, 1] >= 0.5).astype(int)

    pipeline = EnsemblePipeline()

    cv_auroc: Optional[float] = None
    if len(set(y)) == 2 and len(y) >= n_splits * 2:
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        scores = cross_val_score(pipeline, X, y, cv=skf, scoring="roc_auc")
        cv_auroc = float(scores.mean())
        logger.info(
            "5-fold CV AUROC: %.4f ± %.4f (%d folds, n=%d)",
            cv_auroc, scores.std(), n_splits, len(y),
        )
    else:
        logger.warning(
            "Dataset too small or single-class for CV (n=%d, classes=%s) — skipping CV",
            len(y), sorted(set(y)),
        )

    # Final fit on all data
    pipeline.fit(X, y)
    return pipeline, cv_auroc


# ---------------------------------------------------------------------------
# Current AUROC loading
# ---------------------------------------------------------------------------

def _load_current_auroc() -> Optional[float]:
    """Return the cv_auroc stored in the existing pkl, or None if not found."""
    if not _PKL_PATH.exists():
        return None
    try:
        import joblib  # type: ignore
        data = joblib.load(_PKL_PATH)
        return data.get("cv_auroc")
    except Exception as exc:
        logger.warning("Could not read current pkl AUROC: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Retrain MiniJudge from community labels + exp159 data."
    )
    parser.add_argument(
        "--labels",
        required=True,
        metavar="PATH",
        help="Path to labels.jsonl (telemetry-collected community labels)",
    )
    parser.add_argument(
        "--chains",
        default=str(_EXISTING_CHAINS_PATH),
        metavar="PATH",
        help="Path to existing hotpot_qa_chains.json (default: results/large_dataset/)",
    )
    parser.add_argument(
        "--min-improvement",
        type=float,
        default=0.005,
        metavar="DELTA",
        help="Minimum AUROC improvement to trigger model save (default: 0.005)",
    )
    parser.add_argument(
        "--n-splits",
        type=int,
        default=5,
        metavar="K",
        help="Number of CV folds (default: 5)",
    )
    args = parser.parse_args()

    labels_path = Path(args.labels)
    chains_path = Path(args.chains)

    if not labels_path.exists():
        logger.error("labels.jsonl not found: %s", labels_path)
        sys.exit(1)

    extract_features = _load_mini_judge_features()

    # --- Load telemetry labels ---
    X_tel, y_tel = _load_telemetry_labels(labels_path, extract_features)

    # --- Load existing exp159 chains (optional — skip if file missing) ---
    X_base = np.empty((0, 11), dtype=np.float64)
    y_base = np.empty(0, dtype=int)
    if chains_path.exists():
        X_base, y_base = _load_existing_chains(chains_path, extract_features)
    else:
        logger.warning(
            "Existing chains file not found: %s — training on telemetry labels only",
            chains_path,
        )

    # --- Combine datasets ---
    if X_tel.shape[0] == 0 and X_base.shape[0] == 0:
        logger.error("No training data available — aborting.")
        sys.exit(1)

    if X_tel.shape[0] > 0 and X_base.shape[0] > 0:
        X = np.vstack([X_base, X_tel])
        y = np.concatenate([y_base, y_tel])
    elif X_tel.shape[0] > 0:
        X, y = X_tel, y_tel
    else:
        X, y = X_base, y_base

    logger.info(
        "Combined dataset: n=%d (base=%d, telemetry=%d), pos_rate=%.2f",
        len(y), len(y_base), len(y_tel),
        y.mean() if len(y) > 0 else 0.0,
    )

    # --- Train ---
    pipeline, new_auroc = _train_and_evaluate(X, y, n_splits=args.n_splits)

    # --- Compare with current model ---
    current_auroc = _load_current_auroc()

    if new_auroc is None:
        logger.warning("Could not compute new AUROC — dataset may be too small.")
        print("RETRAIN: skipped (insufficient data for CV evaluation)")
        sys.exit(0)

    if current_auroc is None:
        # No existing model — always save
        should_save = True
        logger.info("No existing model found — saving new model unconditionally.")
    else:
        should_save = new_auroc > current_auroc + args.min_improvement

    if should_save:
        import joblib  # type: ignore
        _PKL_DIR.mkdir(parents=True, exist_ok=True)
        data = {
            "pipeline": pipeline,
            "feature_set": "SC_OLD only",
            "embed_model": "none",
            "cv_auroc": new_auroc,
            "training_n": int(len(y)),
            "telemetry_n": int(len(y_tel)),
        }
        joblib.dump(data, _PKL_PATH)
        logger.info("Saved new model to %s", _PKL_PATH)
        old_str = f"{current_auroc:.4f}" if current_auroc is not None else "N/A"
        print(
            f"RETRAIN: AUROC improved {old_str} → {new_auroc:.4f}, ready to ship"
        )
    else:
        old_str = f"{current_auroc:.4f}" if current_auroc is not None else "N/A"
        print(
            f"RETRAIN: no improvement ({new_auroc:.4f} vs {old_str}), keeping current model"
        )


if __name__ == "__main__":
    main()
