"""
MiniJudge — $0 local judge distilled from Sonnet judge decisions.

Drop-in replacement for the Haiku/Sonnet judge using a pre-trained sklearn
pipeline (StandardScaler + LogisticRegression on 11 SC_OLD features).

Validated (exp159, HP 200 chains, 5-fold CV):
    AUROC: 0.7471 ± 0.10  (vs Haiku 0.620, Sonnet 0.7735)
    Cost:  $0.000/chain   (vs Haiku ~$0.001, Sonnet ~$0.005)
"""

from __future__ import annotations

import os
import logging
import statistics
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# Default pkl path relative to QPPG project root
_QPPG_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_PKL = (
    _QPPG_ROOT / "results" / "exp159_hybrid_mini_judge" / "hybrid_mini_judge.pkl"
)


# ---------------------------------------------------------------------------
# Feature extraction (11 SC_OLD features, exact order from exp159)
# ---------------------------------------------------------------------------

def _extract_features(chain: dict) -> np.ndarray:
    """
    Extract the 11 SC_OLD features from a ReAct chain dict.

    Expected chain keys (mirrors AgentGuard / exp18 chain format):
        "question"     : str
        "steps"        : list of dicts with keys "thought", "action",
                         "action_type", "action_args", "observation"
        "final_answer" : str

    Returns
    -------
    np.ndarray of shape (11,)
    """
    question: str = chain.get("question", "")
    steps: List[Dict] = chain.get("steps", [])
    final_answer: str = chain.get("final_answer", "") or ""

    n_steps = max(1, len(steps))

    action_types: List[str] = [s.get("action_type", "") or "" for s in steps]
    observations: List[str] = [s.get("observation", "") or "" for s in steps]
    thoughts: List[str] = [s.get("thought", "") or "" for s in steps]

    # Search queries (action_args where action_type == "Search")
    queries: List[str] = [
        str(s.get("action_args", "") or "")
        for s in steps
        if (s.get("action_type", "") or "").strip().lower() == "search"
    ]

    # --- SC1: loop_rate ---
    sc1 = 1.0 - len(set(action_types)) / max(1, n_steps)

    # --- SC2: steps_norm ---
    sc2 = min(1.0, n_steps / 6.0)

    # --- SC3: empty_obs_rate ---
    sc3 = sum(1 for o in observations if len(o.strip()) < 15) / max(1, len(observations))

    # --- SC5: repeated_search ---
    sc5 = 1.0 - len(set(queries)) / max(1, len(queries)) if queries else 0.0

    # --- SC6: answer_gap ---
    q_len = max(1, len(question.split()))
    ans_len = len(final_answer.split())
    sc6 = max(0.0, 1.0 - ans_len / max(1, q_len * 0.5))

    # --- SC8: backtrack_rate ---
    n = len(action_types)
    if n > 1:
        sc8 = sum(
            1 for i in range(1, n) if action_types[i] == action_types[i - 1]
        ) / max(1, n - 1)
    else:
        sc8 = 0.0

    # --- SC9: obs_util ---
    obs_lens = [len(o.split()) for o in observations]
    avg_obs_len = statistics.mean(obs_lens) if obs_lens else 0.0
    sc9 = 1.0 - min(1.0, avg_obs_len / 50.0)

    # --- SC10: coherence_drop ---
    thought_lens = [len(t.split()) for t in thoughts]
    if len(thought_lens) >= 2:
        mean_tl = statistics.mean(thought_lens)
        stdev_tl = statistics.pstdev(thought_lens)
        sc10 = stdev_tl / max(1.0, mean_tl)
    else:
        sc10 = 0.0

    # --- SC11: ans_obs_mismatch ---
    answer_words = set(final_answer.lower().split())
    last_obs = observations[-1] if observations else ""
    last_obs_words = set(last_obs.lower().split())
    if answer_words or last_obs_words:
        overlap = len(answer_words & last_obs_words) / max(
            1, len(answer_words | last_obs_words)
        )
        sc11 = 1.0 - overlap
    else:
        sc11 = 1.0

    # --- search_count_norm ---
    search_count = sum(
        1 for at in action_types if at.strip().lower() == "search"
    )
    search_count_norm = search_count / max(1, n_steps)

    # --- avg_obs_norm ---
    avg_obs_norm = min(1.0, avg_obs_len / 100.0)

    return np.array(
        [sc1, sc2, sc3, sc5, sc6, sc8, sc9, sc10, sc11, search_count_norm, avg_obs_norm],
        dtype=np.float64,
    )


# ---------------------------------------------------------------------------
# MiniJudge class
# ---------------------------------------------------------------------------

class MiniJudge:
    """
    Local mini-judge distilled from Sonnet judge decisions.
    Drop-in replacement for Haiku/Sonnet judge at $0 inference cost.

    Validated (exp159, HP 200 chains, 5-fold CV):
        AUROC: 0.7471 ± 0.10  (vs Haiku 0.620, Sonnet 0.7735)
        Cost:  $0.000/chain   (vs Haiku ~$0.001, Sonnet ~$0.005)

    Usage::

        from llm_guard import MiniJudge

        judge = MiniJudge()        # auto-loads bundled model
        risk = judge.score(question, steps, final_answer)  # -> float [0,1]

        # Or retrain on your own labeled chains:
        judge = MiniJudge()
        judge.fit(labeled_chains)  # chains with "correct" field
        risk = judge.score(question, steps, final_answer)
    """

    def __init__(self, model_path: Optional[str] = None) -> None:
        """
        Load the pre-trained pipeline from *model_path* (or the default pkl).

        If the pkl file is not found, initialises as untrained; call
        :meth:`fit` before using :meth:`score`.

        Parameters
        ----------
        model_path:
            Explicit path to a ``hybrid_mini_judge.pkl`` file.  If *None*,
            uses the bundled default path.
        """
        self._pipeline = None
        self._cv_auroc: Optional[float] = None
        self._model_path: Path = Path(model_path) if model_path else _DEFAULT_PKL

        if self._model_path.exists():
            self._load_pkl(self._model_path)
        else:
            logger.warning(
                "MiniJudge: pkl not found at %s — call fit() before score().",
                self._model_path,
            )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_pkl(self, path: Path) -> None:
        try:
            import joblib  # type: ignore

            data = joblib.load(path)
            self._pipeline = data["pipeline"]
            self._cv_auroc = data.get("cv_auroc")
            logger.info(
                "MiniJudge: loaded pipeline from %s (cv_auroc=%s)",
                path,
                self._cv_auroc,
            )
        except Exception as exc:  # pragma: no cover
            logger.error("MiniJudge: failed to load pkl: %s", exc)
            self._pipeline = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def is_fitted(self) -> bool:
        """Return True if the underlying pipeline has been fitted."""
        return self._pipeline is not None

    @property
    def auroc(self) -> Optional[float]:
        """Cross-validated AUROC stored in the pkl, or None for a fresh fit."""
        return self._cv_auroc

    def score(
        self,
        question: str,
        steps: List[Dict],
        final_answer: str,
    ) -> float:
        """
        Score a single chain and return a risk float in [0, 1].

        A higher value indicates higher probability of a *wrong* final answer.

        Parameters
        ----------
        question:
            The original question posed to the agent.
        steps:
            List of ReAct step dicts (keys: thought, action, action_type,
            action_args, observation).
        final_answer:
            The agent's final answer string.

        Returns
        -------
        float
            Risk score in [0, 1].  1.0 = very likely wrong.

        Raises
        ------
        RuntimeError
            If the model has not been fitted/loaded yet.
        """
        if not self.is_fitted:
            raise RuntimeError(
                "MiniJudge is not fitted. Load a pkl or call fit() first."
            )

        chain = {
            "question": question,
            "steps": steps,
            "final_answer": final_answer,
        }
        feats = _extract_features(chain).reshape(1, -1)
        prob: float = float(self._pipeline.predict_proba(feats)[0, 1])
        return prob

    def score_chain(
        self,
        question: str,
        steps: List[Dict],
        final_answer: str,
    ) -> float:
        """Alias for :meth:`score` — drop-in compatibility with AgentGuard."""
        return self.score(question, steps, final_answer)

    def fit(
        self,
        chains: List[Dict],
        n_splits: int = 5,
    ) -> "MiniJudge":
        """
        Train a new LogReg pipeline on *chains* and save to the default pkl.

        Parameters
        ----------
        chains:
            List of chain dicts.  Each must contain ``"correct"`` (bool/int;
            0 = correct answer → risk label 0, 1 = wrong → risk label 1) plus
            the standard ``"question"``, ``"steps"``, ``"final_answer"`` keys.
        n_splits:
            Number of stratified CV folds used to compute cv_auroc.

        Returns
        -------
        MiniJudge
            *self*, to allow chaining.
        """
        from sklearn.linear_model import LogisticRegression  # type: ignore
        from sklearn.pipeline import Pipeline  # type: ignore
        from sklearn.preprocessing import StandardScaler  # type: ignore
        from sklearn.model_selection import StratifiedKFold, cross_val_score  # type: ignore
        import joblib  # type: ignore

        X = np.array([_extract_features(c) for c in chains])
        # label: 1 = wrong (high risk), 0 = correct (low risk)
        y = np.array([int(not bool(c.get("correct", True))) for c in chains])

        pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=1000, random_state=42)),
            ]
        )

        # Cross-validated AUROC
        cv_auroc: Optional[float] = None
        if len(set(y)) == 2 and len(chains) >= n_splits * 2:
            try:
                scores = cross_val_score(
                    pipeline,
                    X,
                    y,
                    cv=StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42),
                    scoring="roc_auc",
                )
                cv_auroc = float(scores.mean())
                logger.info(
                    "MiniJudge.fit: cv_auroc=%.4f ± %.4f over %d folds",
                    cv_auroc,
                    scores.std(),
                    n_splits,
                )
            except Exception as exc:
                logger.warning("MiniJudge.fit: CV scoring failed: %s", exc)

        pipeline.fit(X, y)
        self._pipeline = pipeline
        self._cv_auroc = cv_auroc

        # Persist
        self._model_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "pipeline": pipeline,
            "feature_set": "SC_OLD only",
            "embed_model": "none",
            "cv_auroc": cv_auroc,
        }
        joblib.dump(data, self._model_path)
        logger.info("MiniJudge: saved to %s", self._model_path)

        return self

    def save(self, path: str) -> None:
        """
        Serialise the fitted pipeline to *path* using joblib.

        Parameters
        ----------
        path:
            Destination file path (e.g. ``"/tmp/my_mini_judge.pkl"``).
        """
        if not self.is_fitted:
            raise RuntimeError("Cannot save an unfitted MiniJudge.")

        import joblib  # type: ignore

        data = {
            "pipeline": self._pipeline,
            "feature_set": "SC_OLD only",
            "embed_model": "none",
            "cv_auroc": self._cv_auroc,
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(data, path)
        logger.info("MiniJudge.save: written to %s", path)

    @classmethod
    def load(cls, path: str) -> "MiniJudge":
        """
        Load a serialised MiniJudge from *path*.

        Parameters
        ----------
        path:
            Path to a ``*.pkl`` file previously created by :meth:`save` or
            :meth:`fit`.

        Returns
        -------
        MiniJudge
            A ready-to-use, fitted instance.
        """
        instance = cls.__new__(cls)
        instance._pipeline = None
        instance._cv_auroc = None
        instance._model_path = Path(path)
        instance._load_pkl(Path(path))
        return instance
