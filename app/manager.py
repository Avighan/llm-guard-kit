"""
GuardManager — stateful wrapper around LLMGuard with a continuous learning loop.

Learning mechanisms (triggered automatically as feedback arrives):

  1. KNN expansion     — every verified-correct query is added to the calibration
                         pool and the KNN is re-fitted immediately (<100ms).

  2. Prompt Healing    — once ERROR_HEAL_THRESHOLD errors accumulate, calls
                         guard.learn_from_errors() to synthesise repair tools that
                         are auto-injected on future queries in those error clusters.

  3. QARA re-training  — once QARA_RETRAIN_THRESHOLD new labeled examples arrive
                         (with enough of each class), re-trains the cross-domain
                         adapter so risk scores stay accurate as the query
                         distribution shifts.

  4. Confidence tracking — logs predicted vs. actual correctness so you can audit
                           calibration accuracy over time.
"""

import uuid
import time
import pickle
import threading
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from llm_guard import LLMGuard

# ── Thresholds ──────────────────────────────────────────────────────────────────
QARA_RETRAIN_THRESHOLD = 50   # re-fit QARA after this many new labeled examples
ERROR_HEAL_THRESHOLD   = 5    # run learn_from_errors after this many new errors
MIN_QARA_PER_CLASS     = 10   # minimum correct + incorrect examples for QARA


# ── Query record ────────────────────────────────────────────────────────────────

@dataclass
class QueryRecord:
    query_id:       str
    question:       str
    answer:         str
    risk_score:     float
    confidence:     str
    timestamp:      float
    is_correct:     Optional[bool]  = None
    correct_answer: Optional[str]   = None


# ── Manager ─────────────────────────────────────────────────────────────────────

class GuardManager:
    """
    Manages one LLMGuard instance with an automatic online-learning loop.

    Typical lifecycle
    -----------------
    mgr = GuardManager(api_key="sk-ant-...")

    # 1. Calibrate once with known-correct examples
    mgr.calibrate(correct_questions)

    # 2. Process live queries
    record = mgr.query("What is 15% of 240?")
    # → record.query_id, record.answer, record.confidence, record.risk_score

    # 3. Submit feedback as ground truth arrives
    mgr.feedback(record.query_id, is_correct=True)
    mgr.feedback(other_id,        is_correct=False, correct_answer="42")
    # The guard silently improves after each feedback call.
    """

    def __init__(
        self,
        api_key:    str = None,
        model:      str = "claude-haiku-4-5-20251001",
        state_path: str = "guard_state.pkl",
    ):
        self.guard      = LLMGuard(api_key=api_key, model=model)
        self.state_path = state_path
        self._lock      = threading.Lock()

        # Query history (query_id → record)
        self._query_log: Dict[str, QueryRecord] = {}

        # Calibration pool — correct questions whose KNN will be re-fitted
        self._correct_questions: List[str] = []

        # Labeled pool — all feedback, used to periodically re-train QARA
        self._labeled_questions: List[str] = []
        self._labeled_labels:    List[int] = []

        # Error log — used by learn_from_errors (Prompt Healer)
        self._error_questions:       List[str] = []
        self._error_answers:         List[str] = []
        self._error_correct_answers: List[str] = []

        # Counters driving auto-triggers
        self._n_since_qara_fit: int = 0
        self._n_since_heal:     int = 0

        # Calibration accuracy tracking  (predicted_conf, actual_correct)
        self._conf_history: List[tuple] = []

        self._load_state()

    # ── Calibration ─────────────────────────────────────────────────────────────

    def calibrate(
        self,
        questions: List[str],
        labels:    Optional[List[int]] = None,
    ) -> dict:
        """
        Initial calibration.

        Parameters
        ----------
        questions : list of str
        labels    : list of int, optional
            1 = correct, 0 = incorrect. If omitted, all are treated as correct.
        """
        with self._lock:
            if labels is None:
                correct = list(questions)
                labs    = [1] * len(questions)
            else:
                correct = [q for q, l in zip(questions, labels) if l == 1]
                labs    = list(labels)

            if len(correct) < 6:
                raise ValueError(
                    f"Need at least 6 correct examples to calibrate "
                    f"(got {len(correct)})."
                )

            self._correct_questions = correct
            self._labeled_questions = list(questions)
            self._labeled_labels    = labs
            self.guard.fit(correct)
            self._save_state()

        return {
            "status":    "ok",
            "n_correct": len(correct),
            "n_total":   len(questions),
        }

    # ── Query ────────────────────────────────────────────────────────────────────

    def query(
        self,
        question:      str,
        system_prompt: Optional[str] = None,
    ) -> QueryRecord:
        """Run a query and return a QueryRecord (including query_id for feedback)."""
        result = self.guard.query(question, system_prompt=system_prompt)
        record = QueryRecord(
            query_id   = str(uuid.uuid4()),
            question   = question,
            answer     = result.answer,
            risk_score = result.risk_score,
            confidence = result.confidence,
            timestamp  = time.time(),
        )
        with self._lock:
            self._query_log[record.query_id] = record
        return record

    # ── Feedback + learning loop ─────────────────────────────────────────────────

    def feedback(
        self,
        query_id:       str,
        is_correct:     bool,
        correct_answer: Optional[str] = None,
    ) -> dict:
        """
        Submit correctness feedback for a previous query.

        This drives all three continuous learning mechanisms:
          • KNN expansion     (immediate, if correct)
          • Prompt Healing    (batched, every ERROR_HEAL_THRESHOLD errors)
          • QARA re-training  (batched, every QARA_RETRAIN_THRESHOLD examples)

        Returns a status dict showing the current state of the learning loop.
        """
        with self._lock:
            if query_id not in self._query_log:
                raise KeyError(f"query_id '{query_id}' not found")

            record                = self._query_log[query_id]
            record.is_correct     = is_correct
            record.correct_answer = correct_answer

            question = record.question
            triggered = []

            # ── 1. KNN expansion ─────────────────────────────────────────────
            if is_correct and question not in self._correct_questions:
                self._correct_questions.append(question)
                self.guard.fit(self._correct_questions)
                triggered.append("knn_expansion")

            # ── 2. Error accumulation + auto-heal ────────────────────────────
            if not is_correct:
                self._error_questions.append(question)
                self._error_answers.append(record.answer)
                self._error_correct_answers.append(correct_answer or "")
                self._n_since_heal += 1

                if self._n_since_heal >= ERROR_HEAL_THRESHOLD:
                    self.guard.learn_from_errors(
                        self._error_questions,
                        self._error_answers,
                        self._error_correct_answers,
                    )
                    self._n_since_heal = 0
                    triggered.append("prompt_healing")

            # ── 3. Labeled pool + QARA re-train ──────────────────────────────
            self._labeled_questions.append(question)
            self._labeled_labels.append(1 if is_correct else 0)
            self._n_since_qara_fit += 1

            qara_triggered = False
            if self._n_since_qara_fit >= QARA_RETRAIN_THRESHOLD:
                n_correct   = sum(self._labeled_labels)
                n_incorrect = len(self._labeled_labels) - n_correct
                if n_correct >= MIN_QARA_PER_CLASS and n_incorrect >= MIN_QARA_PER_CLASS:
                    self.guard.fit_qara([{
                        "name":      "live",
                        "questions": self._labeled_questions,
                        "labels":    self._labeled_labels,
                    }], verbose=False)
                    self._n_since_qara_fit = 0
                    qara_triggered = True
                    triggered.append("qara_retrain")

            # ── 4. Confidence tracking ────────────────────────────────────────
            self._conf_history.append((record.confidence, is_correct))

            self._save_state()

        return {
            "status":                 "ok",
            "triggered":              triggered,
            "calibration_pool_size":  len(self._correct_questions),
            "labeled_pool_size":      len(self._labeled_questions),
            "error_log_size":         len(self._error_questions),
            "examples_until_qara":    max(0, QARA_RETRAIN_THRESHOLD - self._n_since_qara_fit),
            "errors_until_heal":      max(0, ERROR_HEAL_THRESHOLD - self._n_since_heal),
            "qara_fitted":            self.guard._qara is not None,
        }

    # ── Manual triggers ──────────────────────────────────────────────────────────

    def fit_qara_now(self, epochs: int = 200) -> dict:
        """Manually trigger QARA re-training with all accumulated labeled data."""
        with self._lock:
            n_correct   = sum(self._labeled_labels)
            n_incorrect = len(self._labeled_labels) - n_correct
            if n_correct < MIN_QARA_PER_CLASS or n_incorrect < MIN_QARA_PER_CLASS:
                raise ValueError(
                    f"Need ≥{MIN_QARA_PER_CLASS} correct and incorrect examples. "
                    f"Have {n_correct} correct, {n_incorrect} incorrect."
                )
            result = self.guard.fit_qara([{
                "name":      "live",
                "questions": self._labeled_questions,
                "labels":    self._labeled_labels,
            }], epochs=epochs)
            self._n_since_qara_fit = 0
            self._save_state()
        return result

    def heal_now(self) -> dict:
        """Manually trigger Prompt Healer on accumulated errors."""
        with self._lock:
            if len(self._error_questions) < 5:
                raise ValueError(
                    f"Need at least 5 errors to diagnose (have {len(self._error_questions)})."
                )
            self.guard.learn_from_errors(
                self._error_questions,
                self._error_answers,
                self._error_correct_answers,
            )
            self._n_since_heal = 0
            self._save_state()
        return {"status": "ok", "n_errors_processed": len(self._error_questions)}

    def diagnose_now(self) -> list:
        """Return failure cluster analysis without modifying guard state."""
        with self._lock:
            if len(self._error_questions) < 5:
                raise ValueError("Need at least 5 errors to diagnose.")
            return self.guard.diagnose(
                self._error_questions,
                self._error_answers,
                self._error_correct_answers or None,
            )

    # ── Stats ────────────────────────────────────────────────────────────────────

    def get_stats(self) -> dict:
        with self._lock:
            guard_stats = self.guard.get_stats()

            # Calibration accuracy breakdown
            conf_acc = {}
            for conf, correct in self._conf_history:
                if conf not in conf_acc:
                    conf_acc[conf] = {"total": 0, "correct": 0}
                conf_acc[conf]["total"]   += 1
                conf_acc[conf]["correct"] += int(correct)
            for k in conf_acc:
                t = conf_acc[k]["total"]
                conf_acc[k]["accuracy"] = round(conf_acc[k]["correct"] / t, 3) if t else None

            guard_stats["learning"] = {
                "calibration_pool_size": len(self._correct_questions),
                "labeled_pool_size":     len(self._labeled_questions),
                "n_queries_logged":      len(self._query_log),
                "error_log_size":        len(self._error_questions),
                "n_tools":               len(self.guard._tools),
                "examples_until_qara":   max(0, QARA_RETRAIN_THRESHOLD - self._n_since_qara_fit),
                "errors_until_heal":     max(0, ERROR_HEAL_THRESHOLD - self._n_since_heal),
                "confidence_accuracy":   conf_acc,
            }
        return guard_stats

    # ── Persistence ──────────────────────────────────────────────────────────────

    def _save_state(self):
        """Persist manager + guard state to disk."""
        state = {
            "correct_questions":       self._correct_questions,
            "labeled_questions":       self._labeled_questions,
            "labeled_labels":          self._labeled_labels,
            "error_questions":         self._error_questions,
            "error_answers":           self._error_answers,
            "error_correct_answers":   self._error_correct_answers,
            "n_since_qara_fit":        self._n_since_qara_fit,
            "n_since_heal":            self._n_since_heal,
            "conf_history":            self._conf_history,
            # Guard internals
            "guard_cal_embs":          self.guard._cal_embs,
            "guard_qara":              self.guard._qara,
            "guard_tools":             self.guard._tools,
            "guard_cluster_centers":   self.guard._cluster_centers,
            "guard_risk_low":          self.guard._risk_low_threshold,
            "guard_risk_high":         self.guard._risk_high_threshold,
        }
        with open(self.state_path, "wb") as f:
            pickle.dump(state, f)

    def _load_state(self):
        """Restore manager + guard state from disk (no-op if file missing)."""
        try:
            with open(self.state_path, "rb") as f:
                state = pickle.load(f)

            self._correct_questions     = state.get("correct_questions",     [])
            self._labeled_questions     = state.get("labeled_questions",     [])
            self._labeled_labels        = state.get("labeled_labels",        [])
            self._error_questions       = state.get("error_questions",       [])
            self._error_answers         = state.get("error_answers",         [])
            self._error_correct_answers = state.get("error_correct_answers", [])
            self._n_since_qara_fit      = state.get("n_since_qara_fit",      0)
            self._n_since_heal          = state.get("n_since_heal",          0)
            self._conf_history          = state.get("conf_history",          [])

            # Restore guard internals
            self.guard._cal_embs            = state.get("guard_cal_embs")
            self.guard._qara                = state.get("guard_qara")
            self.guard._tools               = state.get("guard_tools", {})
            self.guard._cluster_centers     = state.get("guard_cluster_centers")
            self.guard._risk_low_threshold  = state.get("guard_risk_low")
            self.guard._risk_high_threshold = state.get("guard_risk_high")

            # Re-fit KNN in the correct space (adapted if QARA present)
            if self.guard._cal_embs is not None:
                self.guard._fit_knn(self.guard._cal_embs)

            print(
                f"[GuardManager] Loaded state: "
                f"{len(self._correct_questions)} correct, "
                f"{len(self._labeled_questions)} labeled, "
                f"{len(self._error_questions)} errors, "
                f"qara={'yes' if self.guard._qara else 'no'}"
            )
        except FileNotFoundError:
            pass
