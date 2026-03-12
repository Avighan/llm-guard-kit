"""
Tests for llm-guard-kit v0.3.0 new features
============================================
Run with:  pytest tests/test_v030.py -v
Run fast:  pytest tests/test_v030.py -v -m 'not slow'

Categories:
  Unit        — pure Python, no embedding model, no API
  Server      — FastAPI TestClient (requires httpx)
  Integration — loads sentence-transformers; marked @pytest.mark.slow
"""

from __future__ import annotations

import csv
import io
import json
import sys
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ── path setup ────────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — ChainStore (store.py)
# ═══════════════════════════════════════════════════════════════════════════════

class TestChainStore:
    """Unit tests for ChainStore: in-memory SQLite (:memory: or temp file)."""

    @pytest.fixture
    def store(self, tmp_path):
        from qppg_service.store import ChainStore
        return ChainStore(db_path=tmp_path / "test.db")

    def _chain(self, question="What year was X founded?", n_search=1):
        steps = [{"action_type": "Search", "action_arg": "X history",
                  "thought": "I need to look this up", "observation": "Founded 1990"}
                 for _ in range(n_search)]
        steps.append({"action_type": "Finish", "action_arg": "1990",
                      "thought": "I found the answer", "observation": ""})
        return {"question": question, "steps": steps,
                "final_answer": "1990", "finished": True}

    # ── add_chain + get_calibration_pool ──────────────────────────────────────

    def test_add_chain_returns_row_id(self, store):
        row_id = store.add_chain("prod", self._chain())
        assert isinstance(row_id, int) and row_id > 0

    def test_add_chain_increments_pool(self, store):
        for i in range(3):
            store.add_chain("prod", self._chain(question=f"Q{i}"))
        pool = store.get_calibration_pool("prod")
        assert len(pool) == 3

    def test_pool_respects_n_limit(self, store):
        for i in range(10):
            store.add_chain("prod", self._chain(question=f"Q{i}"))
        pool = store.get_calibration_pool("prod", n=5)
        assert len(pool) == 5

    def test_pool_preserves_chain_structure(self, store):
        store.add_chain("prod", self._chain(question="Test Q"))
        pool = store.get_calibration_pool("prod")
        assert pool[0]["question"] == "Test Q"
        assert isinstance(pool[0]["steps"], list)
        assert isinstance(pool[0]["finished"], bool)

    def test_domain_isolation(self, store):
        store.add_chain("domain_a", self._chain(question="A1"))
        store.add_chain("domain_b", self._chain(question="B1"))
        assert len(store.get_calibration_pool("domain_a")) == 1
        assert len(store.get_calibration_pool("domain_b")) == 1

    # ── get_domains + get_domain_stats ────────────────────────────────────────

    def test_get_domains_lists_all(self, store):
        store.add_chain("alpha", self._chain())
        store.add_chain("beta", self._chain())
        domains = store.get_domains()
        assert set(domains) == {"alpha", "beta"}

    def test_domain_stats_structure(self, store):
        store.add_chain("prod", self._chain(), risk_score=0.7, alert=True)
        store.add_chain("prod", self._chain(), risk_score=0.3, alert=False)
        stats = store.get_domain_stats("prod")
        assert stats["n_chains"] == 2
        assert stats["n_alerts"] == 1
        assert 0.0 <= stats["avg_risk"] <= 1.0

    def test_n_steps_counts_search_only(self, store):
        store.add_chain("prod", self._chain(n_search=3))
        rows = store.get_chains("prod")
        assert rows[0]["n_steps"] == 3

    # ── get_risk_window ───────────────────────────────────────────────────────

    def test_get_risk_window_returns_scores_in_range(self, store):
        t_now = time.time()
        # Use patch to control timestamp via direct SQL insert
        import sqlite3
        with sqlite3.connect(str(store.db_path)) as conn:
            conn.execute(
                "INSERT INTO chains (domain, question, steps_json, final_answer, "
                "finished, risk_score, alert_triggered, failure_mode, n_steps, "
                "model_name, timestamp) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                ("prod", "Q", "[]", "", 1, 0.8, 0, "", 0, "", t_now - 100)
            )
            conn.execute(
                "INSERT INTO chains (domain, question, steps_json, final_answer, "
                "finished, risk_score, alert_triggered, failure_mode, n_steps, "
                "model_name, timestamp) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                ("prod", "Q", "[]", "", 1, 0.2, 0, "", 0, "", t_now - 200)
            )
            conn.commit()
        risks = store.get_risk_window("prod", t_now - 150, t_now)
        assert risks == [pytest.approx(0.8)]

    # ── export_audit ──────────────────────────────────────────────────────────

    def test_export_audit_csv_format(self, store):
        store.add_chain("prod", self._chain(question="Audit Q"), risk_score=0.55)
        csv_str = store.export_audit("prod")
        reader = csv.DictReader(io.StringIO(csv_str))
        rows = list(reader)
        assert len(rows) == 1
        assert "risk_score" in rows[0]
        assert rows[0]["question_trunc"] == "Audit Q"

    def test_export_audit_json_format(self, store):
        store.add_chain("prod", self._chain(question="JSON Q"), risk_score=0.42)
        json_str = store.export_audit("prod", fmt="json")
        records = json.loads(json_str)
        assert len(records) == 1
        assert records[0]["risk_score"] == pytest.approx(0.42)

    def test_export_audit_empty_domain_returns_empty(self, store):
        result = store.export_audit("no_such_domain")
        assert result == ""

    def test_export_audit_time_filter(self, store):
        t_now = time.time()
        import sqlite3
        with sqlite3.connect(str(store.db_path)) as conn:
            conn.execute(
                "INSERT INTO chains (domain, question, steps_json, final_answer, "
                "finished, risk_score, alert_triggered, failure_mode, n_steps, "
                "model_name, timestamp) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                ("prod", "Old Q", "[]", "", 1, 0.5, 0, "", 0, "", t_now - 10000)
            )
            conn.execute(
                "INSERT INTO chains (domain, question, steps_json, final_answer, "
                "finished, risk_score, alert_triggered, failure_mode, n_steps, "
                "model_name, timestamp) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                ("prod", "New Q", "[]", "", 1, 0.5, 0, "", 0, "", t_now - 10)
            )
            conn.commit()
        csv_str = store.export_audit("prod", start=t_now - 100, end=t_now)
        reader = csv.DictReader(io.StringIO(csv_str))
        rows = list(reader)
        assert len(rows) == 1
        assert rows[0]["question_trunc"] == "New Q"

    # ── clear_domain ──────────────────────────────────────────────────────────

    def test_clear_domain_removes_chains(self, store):
        store.add_chain("prod", self._chain())
        store.add_chain("prod", self._chain())
        n = store.clear_domain("prod")
        assert n == 2
        assert store.get_calibration_pool("prod") == []

    def test_clear_domain_does_not_affect_other_domains(self, store):
        store.add_chain("a", self._chain())
        store.add_chain("b", self._chain())
        store.clear_domain("a")
        assert len(store.get_calibration_pool("b")) == 1

    # ── API key management ────────────────────────────────────────────────────

    def test_create_api_key_returns_string(self, store):
        key = store.create_api_key("acme", "prod")
        assert isinstance(key, str) and len(key) > 20

    def test_verify_api_key_valid(self, store):
        key = store.create_api_key("acme", "prod")
        info = store.verify_api_key(key)
        assert info is not None
        assert info["customer_id"] == "acme"

    def test_verify_api_key_invalid(self, store):
        result = store.verify_api_key("definitely-not-a-valid-key")
        assert result is None

    def test_verify_api_key_increments_request_count(self, store):
        key = store.create_api_key("acme", "prod")
        store.verify_api_key(key)
        store.verify_api_key(key)
        import hashlib
        key_hash = hashlib.sha256(key.encode()).hexdigest()
        stats = store.get_key_stats(key_hash)
        assert stats["request_count"] == 2

    def test_different_keys_are_unique(self, store):
        key1 = store.create_api_key("c1", "prod")
        key2 = store.create_api_key("c1", "prod")
        assert key1 != key2

    # ── log_calibration ───────────────────────────────────────────────────────

    def test_log_calibration_appears_in_stats(self, store):
        store.add_chain("prod", self._chain())
        store.log_calibration("prod", n_chains=50, auroc=0.87)
        stats = store.get_domain_stats("prod")
        assert stats["last_auroc"] == pytest.approx(0.87)

    # ── get_failure_mode_counts ───────────────────────────────────────────────

    def test_failure_mode_counts(self, store):
        store.add_chain("prod", self._chain(), alert=True, failure_mode="RETRIEVAL_FAILURE")
        store.add_chain("prod", self._chain(), alert=True, failure_mode="RETRIEVAL_FAILURE")
        store.add_chain("prod", self._chain(), alert=True, failure_mode="EXCESSIVE_SEARCH")
        counts = store.get_failure_mode_counts("prod")
        assert counts["RETRIEVAL_FAILURE"] == 2
        assert counts["EXCESSIVE_SEARCH"] == 1


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — DriftDetector (drift.py)
# ═══════════════════════════════════════════════════════════════════════════════

class TestDriftDetector:

    @pytest.fixture
    def store(self, tmp_path):
        from qppg_service.store import ChainStore
        return ChainStore(db_path=tmp_path / "drift.db")

    @pytest.fixture
    def detector(self):
        from qppg_service.drift import DriftDetector
        return DriftDetector(threshold=0.10, window_days=7, min_samples=5)

    def _insert_risks(self, store, domain, risks, offset_days=0):
        """Helper: insert risks at controlled timestamps."""
        import sqlite3
        t_base = time.time() - offset_days * 86400
        with sqlite3.connect(str(store.db_path)) as conn:
            for i, risk in enumerate(risks):
                conn.execute(
                    "INSERT INTO chains (domain, question, steps_json, final_answer, "
                    "finished, risk_score, alert_triggered, failure_mode, n_steps, "
                    "model_name, timestamp) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                    ("prod", f"Q{i}", "[]", "", 1, risk, 0, "", 0, "", t_base - i * 10)
                )
            conn.commit()

    def test_no_alert_when_too_few_samples(self, store, detector):
        self._insert_risks(store, "prod", [0.8, 0.9])  # only 2 < min_samples=5
        alert = detector.check("prod", store)
        assert alert is None

    def test_no_alert_when_delta_small(self, store, detector):
        # current window: mean=0.50, previous: mean=0.50 → delta=0
        self._insert_risks(store, "prod", [0.5]*8, offset_days=0)
        self._insert_risks(store, "prod", [0.5]*8, offset_days=10)
        alert = detector.check("prod", store)
        assert alert is None

    def test_drift_up_detected(self, store, detector):
        """Current mean 0.8, previous mean 0.3 → delta +0.5 > threshold 0.10."""
        self._insert_risks(store, "prod", [0.8]*10, offset_days=0)   # current window
        self._insert_risks(store, "prod", [0.3]*10, offset_days=10)  # previous window
        alert = detector.check("prod", store)
        assert alert is not None
        assert alert.direction == "UP"
        assert alert.delta > 0.10

    def test_drift_down_detected(self, store, detector):
        """Current mean 0.2, previous mean 0.8 → delta -0.6."""
        self._insert_risks(store, "prod", [0.2]*10, offset_days=0)
        self._insert_risks(store, "prod", [0.8]*10, offset_days=10)
        alert = detector.check("prod", store)
        assert alert is not None
        assert alert.direction == "DOWN"
        assert alert.delta < -0.10

    def test_alert_fields(self, store, detector):
        self._insert_risks(store, "prod", [0.9]*10, offset_days=0)
        self._insert_risks(store, "prod", [0.1]*10, offset_days=10)
        alert = detector.check("prod", store)
        assert alert.domain == "prod"
        assert alert.n_current == 10
        assert isinstance(alert.recommendation, str) and len(alert.recommendation) > 10

    def test_no_previous_data_no_alert(self, store, detector):
        """Only current data; previous window is empty → no alert (delta=0)."""
        self._insert_risks(store, "prod", [0.7]*10, offset_days=0)
        alert = detector.check("prod", store)
        # With no previous data, previous_mean = current_mean → delta=0
        assert alert is None


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — QppgMonitor with db_path (monitor.py changes)
# ═══════════════════════════════════════════════════════════════════════════════

class TestQppgMonitorPersistence:

    @pytest.fixture
    def monitored(self, tmp_path):
        from qppg_service.monitor import QppgMonitor
        from qppg_service.store import ChainStore
        db = tmp_path / "monitor.db"
        monitor = QppgMonitor(threshold=0.9, db_path=str(db))
        store = ChainStore(db_path=str(db))
        return monitor, store

    def _steps(self, n=1):
        return [{"action_type": "Search", "action_arg": "q",
                 "thought": "t", "observation": "o"}
                for _ in range(n)] + [
               {"action_type": "Finish", "action_arg": "42",
                "thought": "done", "observation": ""}]

    def test_track_persists_to_sqlite(self, monitored):
        monitor, store = monitored
        monitor.track("What is 2+2?", self._steps(), "4", finished=True)
        pool = store.get_calibration_pool("default")
        assert len(pool) == 1
        assert pool[0]["question"] == "What is 2+2?"

    def test_multiple_tracks_accumulate(self, monitored):
        monitor, store = monitored
        for i in range(5):
            monitor.track(f"Question {i}", self._steps(), f"Answer {i}", finished=True)
        pool = store.get_calibration_pool("default")
        assert len(pool) == 5

    def test_alert_flag_stored_on_high_risk(self, monitored, tmp_path):
        """When risk is high (above threshold), alert_triggered=1 in DB."""
        from qppg_service.monitor import QppgMonitor
        from qppg_service.store import ChainStore
        db = tmp_path / "alert_test.db"
        # Very low threshold → every chain triggers alert
        monitor = QppgMonitor(threshold=0.0, db_path=str(db))
        store = ChainStore(db_path=str(db))
        monitor.track("Q", self._steps(), "A", finished=True)
        stats = store.get_domain_stats("default")
        assert stats["n_alerts"] == 1

    def test_monitor_without_db_path_still_works(self):
        """Backward compat: QppgMonitor() without db_path should not crash."""
        from qppg_service.monitor import QppgMonitor
        monitor = QppgMonitor(threshold=0.65)
        alert = monitor.track("What is 1+1?", [], "2", finished=True)
        # Should not raise; alert may be None or QppgAlert
        assert alert is None or hasattr(alert, "risk_score")

    def test_domain_tag_stored(self, tmp_path):
        from qppg_service.monitor import QppgMonitor
        from qppg_service.store import ChainStore
        db = tmp_path / "domain_test.db"
        monitor = QppgMonitor(db_path=str(db), domain="my_domain")
        store = ChainStore(db_path=str(db))
        monitor.track("Q", [], "A", finished=True)
        domains = store.get_domains()
        assert "my_domain" in domains

    def test_recal_every_parameter_accepted(self):
        """recal_every kwarg should not raise."""
        from qppg_service.monitor import QppgMonitor
        monitor = QppgMonitor(recal_every=200)
        assert monitor.recal_every == 200


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — CLI argument parsing (cli.py)
# ═══════════════════════════════════════════════════════════════════════════════

class TestCLIParser:

    @pytest.fixture
    def parser(self):
        from qppg_service.cli import build_parser
        return build_parser()

    def test_status_command_parsed(self, parser):
        args = parser.parse_args(["status"])
        assert args.command == "status"
        assert args.domain is None

    def test_status_with_domain(self, parser):
        args = parser.parse_args(["status", "--domain", "prod"])
        assert args.domain == "prod"

    def test_calibrate_requires_domain(self, parser):
        with pytest.raises(SystemExit):
            parser.parse_args(["calibrate"])

    def test_calibrate_with_domain(self, parser):
        args = parser.parse_args(["calibrate", "--domain", "prod"])
        assert args.domain == "prod"
        assert args.chains == 25  # default

    def test_calibrate_source_domain(self, parser):
        args = parser.parse_args(["calibrate", "--domain", "prod",
                                   "--source-domain", "staging", "--chains", "50"])
        assert args.source_domain == "staging"
        assert args.chains == 50

    def test_score_question_arg(self, parser):
        args = parser.parse_args(["score", "--question", "What year?"])
        assert args.question == "What year?"

    def test_recalibrate_args(self, parser):
        args = parser.parse_args(["recalibrate", "--domain", "prod",
                                   "--new-model", "claude-opus-4-6"])
        assert args.domain == "prod"
        assert args.new_model == "claude-opus-4-6"

    def test_export_required_domain(self, parser):
        with pytest.raises(SystemExit):
            parser.parse_args(["export"])

    def test_export_format_choices(self, parser):
        args = parser.parse_args(["export", "--domain", "prod", "--format", "json"])
        assert args.format == "json"

    def test_export_invalid_format_rejected(self, parser):
        with pytest.raises(SystemExit):
            parser.parse_args(["export", "--domain", "prod", "--format", "xml"])

    def test_dashboard_defaults(self, parser):
        args = parser.parse_args(["dashboard"])
        assert args.port == 8080
        assert args.domain == "default"

    def test_serve_defaults(self, parser):
        args = parser.parse_args(["serve"])
        assert args.port == 8000
        assert args.host == "127.0.0.1"

    def test_global_db_flag(self, parser):
        args = parser.parse_args(["--db", "/tmp/test.db", "status"])
        assert args.db == "/tmp/test.db"


class TestCLIScore:
    """Test cmd_score with mocked scorer (no sentence-transformers needed)."""

    def test_score_from_steps_file(self, tmp_path, capsys):
        chain = {
            "question": "What is 2+2?",
            "steps": [{"action_type": "Finish", "action_arg": "4",
                        "thought": "", "observation": ""}],
            "final_answer": "4",
            "finished": True,
        }
        steps_file = tmp_path / "chain.json"
        steps_file.write_text(json.dumps(chain))

        from qppg_service.cli import build_parser, cmd_score
        parser = build_parser()
        args = parser.parse_args(["score", "--steps-file", str(steps_file)])

        mock_result = MagicMock()
        mock_result.risk_score = 0.25
        mock_result.needs_review = False
        mock_result.behavioral_score = 0.30
        mock_result.gmm_score = None

        mock_rq = {"mean_sim": 0.7, "min_sim": 0.5, "quality_label": "GOOD"}

        with patch("qppg_service.label_free_scorer.LabelFreeScorer") as MockScorer:
            instance = MockScorer.return_value
            instance.score.return_value = mock_result
            instance.retrieval_quality.return_value = mock_rq
            cmd_score(args)

        captured = capsys.readouterr()
        assert "Risk score" in captured.out
        assert "0.250" in captured.out


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — Framework Integrations (with mocks)
# ═══════════════════════════════════════════════════════════════════════════════

class TestLangChainIntegration:
    """Tests for QppgLangChainCallback using mocked langchain_core."""

    def _make_callback(self):
        """Create callback with mocked langchain import."""
        # Create a minimal BaseCallbackHandler mock
        class FakeBase:
            def __init__(self): pass

        # Patch the lazy import
        mock_module = MagicMock()
        mock_module.BaseCallbackHandler = FakeBase

        with patch.dict("sys.modules", {"langchain_core": mock_module,
                                         "langchain_core.callbacks": mock_module,
                                         "langchain_core.callbacks.base": mock_module}):
            from qppg_service.integrations.langchain_callback import (
                _normalize_tool_name, _extract_thought,
                QppgLangChainCallback,
            )
        return _normalize_tool_name, _extract_thought, QppgLangChainCallback

    def test_normalize_tool_name_search(self):
        from qppg_service.integrations import langchain_callback as lc
        assert lc._normalize_tool_name("tavily_search") == "Search"
        assert lc._normalize_tool_name("duckduckgo_search") == "Search"
        assert lc._normalize_tool_name("wikipedia") == "Search"
        assert lc._normalize_tool_name("retriever") == "Search"

    def test_normalize_tool_name_finish(self):
        from qppg_service.integrations import langchain_callback as lc
        assert lc._normalize_tool_name("final_answer") == "Finish"

    def test_normalize_tool_name_custom_preserved(self):
        from qppg_service.integrations import langchain_callback as lc
        assert lc._normalize_tool_name("calculator") == "calculator"

    def test_extract_thought_from_log(self):
        from qppg_service.integrations import langchain_callback as lc
        log = "Thought: I should search for this.\nAction: search_tool"
        thought = lc._extract_thought(log)
        assert "search" in thought.lower()

    def test_extract_thought_empty_log(self):
        from qppg_service.integrations import langchain_callback as lc
        assert lc._extract_thought("") == ""

    def test_import_error_without_langchain(self):
        """Should raise ImportError with helpful message if langchain_core missing."""
        import importlib
        # Remove any cached import
        mods_to_remove = [k for k in sys.modules if "langchain" in k]
        saved = {k: sys.modules.pop(k) for k in mods_to_remove}
        try:
            with patch.dict("sys.modules", {"langchain_core": None,
                                             "langchain_core.callbacks": None,
                                             "langchain_core.callbacks.base": None}):
                from qppg_service.integrations.langchain_callback import _lazy_base
                with pytest.raises(ImportError, match="langchain"):
                    _lazy_base()
        finally:
            sys.modules.update(saved)


class TestOpenAIAdapter:
    """Tests for score_assistants_run with mocked OpenAI client."""

    def test_score_assistants_run_basic(self):
        from qppg_service.integrations.openai_adapter import score_assistants_run

        # Mock OpenAI client
        mock_client = MagicMock()

        # Fake run steps: one tool_calls step + one message_creation step
        tool_step = MagicMock()
        tool_step.type = "tool_calls"
        tool_step.step_details = MagicMock()
        fn_call = MagicMock()
        fn_call.function.name = "tavily_search"
        fn_call.function.arguments = '{"query": "X history"}'
        tool_step.step_details.tool_calls = [fn_call]

        msg_step = MagicMock()
        msg_step.type = "message_creation"
        msg_step.step_details = MagicMock()
        msg_step.step_details.message_creation = MagicMock()
        msg_step.step_details.message_creation.message_id = "msg_abc"

        mock_client.beta.threads.runs.steps.list.return_value = [tool_step, msg_step]

        # Fake message content
        content_item = MagicMock()
        content_item.type = "text"
        content_item.text.value = "Founded in 1990"
        message = MagicMock()
        message.content = [content_item]
        mock_client.beta.threads.messages.retrieve.return_value = message

        mock_result = MagicMock()
        mock_result.risk_score = 0.3
        mock_result.needs_review = False
        mock_result.behavioral_score = 0.3
        mock_result.gmm_score = None
        mock_result.calibration_size = 0

        mock_scorer = MagicMock()
        mock_scorer.score.return_value = mock_result

        result = score_assistants_run(
            mock_client, "thread_123", "run_456",
            scorer=mock_scorer, question="When was X founded?"
        )
        assert result is not None
        mock_scorer.score.assert_called_once()

    def test_extract_question_from_thread(self):
        from qppg_service.integrations.openai_adapter import extract_question_from_thread

        mock_client = MagicMock()
        msg = MagicMock()
        msg.role = "user"
        content = MagicMock()
        content.type = "text"
        content.text.value = "What year was Python created?"
        msg.content = [content]
        # The code accesses messages.data (paginated response)
        mock_response = MagicMock()
        mock_response.data = [msg]
        mock_client.beta.threads.messages.list.return_value = mock_response

        question = extract_question_from_thread(mock_client, "thread_abc")
        assert "Python" in question


class TestHaystackIntegration:
    """Tests for QppgHaystackMonitor extraction helpers (no real Haystack needed)."""

    def _make_monitor(self):
        from qppg_service.integrations.haystack_callback import QppgHaystackMonitor
        mock_pipeline = MagicMock()
        mock_pipeline.run.return_value = {}

        # Haystack check is done in __init__; patch it
        with patch("qppg_service.integrations.haystack_callback._check_haystack"):
            monitor = QppgHaystackMonitor(mock_pipeline)
        return monitor

    def test_extract_question_from_inputs(self):
        monitor = self._make_monitor()
        inputs = {"query": {"query": "What year was X founded?"}}
        q = monitor._extract_question(inputs)
        assert q == "What year was X founded?"

    def test_extract_steps_retriever(self):
        monitor = self._make_monitor()
        doc = MagicMock()
        doc.content = "Founded in 1990"
        outputs = {
            "InMemoryBM25Retriever": {"documents": [doc]},
        }
        steps, final_answer = monitor._extract_steps({}, outputs)
        assert len(steps) == 1
        assert steps[0]["action_type"] == "Search"

    def test_extract_steps_generator(self):
        monitor = self._make_monitor()
        outputs = {
            "OpenAIGenerator": {"replies": ["The answer is 1990."]},
        }
        steps, final_answer = monitor._extract_steps({}, outputs)
        assert final_answer == "The answer is 1990."
        assert any(s["action_type"] == "Finish" for s in steps)

    def test_run_returns_none_when_empty(self):
        monitor = self._make_monitor()
        monitor._pipeline.run.return_value = {}
        outputs, result = monitor.run({"query": {"query": "test"}})
        assert result is None

    def test_get_last_result_initially_none(self):
        monitor = self._make_monitor()
        assert monitor.get_last_result() is None


class TestLlamaIndexIntegration:
    """Tests for QppgLlamaIndexCallback with mocked llama_index."""

    def test_import_error_without_llamaindex(self):
        """Should raise ImportError with helpful message if llama_index missing."""
        with patch.dict("sys.modules", {
            "llama_index": None,
            "llama_index.core": None,
            "llama_index.core.callbacks": None,
            "llama_index.core.callbacks.base_handler": None,
            "llama_index.core.callbacks.schema": None,
        }):
            from qppg_service.integrations.llamaindex_callback import _lazy_imports
            with pytest.raises(ImportError, match="llama"):
                _lazy_imports()

    def test_callback_get_last_result_initially_none(self):
        """get_last_result() returns None before any events (direct attribute check)."""
        from qppg_service.integrations.llamaindex_callback import QppgLlamaIndexCallback
        # Create a bare instance without calling __init__ (which needs llama_index)
        cb = object.__new__(QppgLlamaIndexCallback)
        cb._last_result = None
        assert cb.get_last_result() is None


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — FastAPI Server (server.py)
# ═══════════════════════════════════════════════════════════════════════════════

try:
    from fastapi.testclient import TestClient
    _HAS_HTTPX = True
except ImportError:
    _HAS_HTTPX = False

@pytest.mark.skipif(not _HAS_HTTPX, reason="httpx not installed")
class TestServerEndpoints:

    @pytest.fixture
    def client(self, tmp_path):
        from qppg_service.service import QPPGService
        from qppg_service.server import create_app

        svc = QPPGService(domain_name="test", save_dir=None)
        app = create_app(svc)
        return TestClient(app)

    @pytest.fixture
    def client_with_store(self, tmp_path):
        from qppg_service.service import QPPGService
        from qppg_service.server import create_app

        svc = QPPGService(domain_name="test", save_dir=None)
        db = str(tmp_path / "server_test.db")
        app = create_app(svc, db_path=db, admin_key="secret-admin-key")
        return TestClient(app), db

    # ── /status ───────────────────────────────────────────────────────────────

    def test_status_returns_200(self, client):
        resp = client.get("/status")
        assert resp.status_code == 200

    def test_status_has_domain_field(self, client):
        body = client.get("/status").json()
        assert "domain" in body

    # ── /score ────────────────────────────────────────────────────────────────

    def test_score_cold_start(self, client):
        payload = {
            "question": "What year was Python created?",
            "steps": [{"thought": "", "action_type": "Finish",
                        "action_arg": "1991", "observation": ""}],
        }
        resp = client.post("/score", json=payload)
        assert resp.status_code == 200
        body = resp.json()
        assert "needs_review" in body

    def test_score_empty_steps(self, client):
        payload = {"question": "Test?", "steps": []}
        resp = client.post("/score", json=payload)
        assert resp.status_code == 200

    # ── /calibrate ────────────────────────────────────────────────────────────

    def test_calibrate_correct_chain(self, client):
        payload = {
            "question": "What is 2+2?",
            "steps": [],
            "correct": True,
        }
        resp = client.post("/calibrate", json=payload)
        assert resp.status_code == 200

    # ── /bulk-calibrate ───────────────────────────────────────────────────────

    def test_bulk_calibrate(self, client):
        chains = [
            {"question": f"Q{i}", "steps": [], "correct": True}
            for i in range(3)
        ]
        resp = client.post("/bulk-calibrate", json={"chains": chains})
        assert resp.status_code == 200

    # ── /reset ────────────────────────────────────────────────────────────────

    def test_reset_without_confirm_rejected(self, client):
        resp = client.post("/reset")
        assert resp.status_code == 400

    def test_reset_with_confirm(self, client):
        resp = client.post("/reset?confirm=YES_RESET")
        assert resp.status_code == 200
        assert resp.json()["reset"] is True

    # ── /progress ────────────────────────────────────────────────────────────

    def test_progress_returns_string(self, client):
        body = client.get("/progress").json()
        assert "progress" in body
        assert isinstance(body["progress"], str)

    # ── SaaS: API key endpoints ────────────────────────────────────────────────

    def test_create_key_admin_auth(self, client_with_store):
        client, db = client_with_store
        payload = {"customer_id": "acme", "domain_prefix": "prod"}
        resp = client.post("/api/keys", json=payload,
                           headers={"X-Admin-Key": "secret-admin-key"})
        assert resp.status_code == 200
        body = resp.json()
        assert "api_key" in body
        assert body["customer_id"] == "acme"

    def test_create_key_wrong_admin_key_rejected(self, client_with_store):
        client, db = client_with_store
        resp = client.post("/api/keys",
                           json={"customer_id": "acme"},
                           headers={"X-Admin-Key": "wrong-key"})
        assert resp.status_code == 401

    def test_saas_score_requires_auth(self, client_with_store):
        client, db = client_with_store
        payload = {"question": "Test?", "steps": []}
        resp = client.post("/api/test-domain/score", json=payload)
        assert resp.status_code == 401

    def test_saas_score_invalid_token_rejected(self, client_with_store):
        client, db = client_with_store
        payload = {"question": "Test?", "steps": []}
        resp = client.post("/api/test-domain/score", json=payload,
                           headers={"Authorization": "Bearer invalid-token-xyz"})
        assert resp.status_code == 401

    def test_saas_score_with_valid_key(self, client_with_store):
        client, db = client_with_store
        # Create a key
        create_resp = client.post("/api/keys",
                                   json={"customer_id": "acme", "domain_prefix": "prod"},
                                   headers={"X-Admin-Key": "secret-admin-key"})
        api_key = create_resp.json()["api_key"]

        # Use it to score
        payload = {"question": "Test?", "steps": []}
        resp = client.post("/api/prod/score", json=payload,
                           headers={"Authorization": f"Bearer {api_key}"})
        assert resp.status_code == 200
        body = resp.json()
        assert "risk_score" in body

    def test_audit_log_with_valid_key(self, client_with_store):
        client, db = client_with_store
        create_resp = client.post("/api/keys",
                                   json={"customer_id": "acme"},
                                   headers={"X-Admin-Key": "secret-admin-key"})
        api_key = create_resp.json()["api_key"]

        resp = client.get("/api/prod/audit-log",
                          headers={"Authorization": f"Bearer {api_key}"})
        assert resp.status_code == 200


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 7 — Dashboard routes
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.skipif(not _HAS_HTTPX, reason="httpx not installed")
class TestDashboard:

    @pytest.fixture
    def dashboard_client(self, tmp_path):
        from fastapi import FastAPI
        from qppg_service.store import ChainStore
        from qppg_service.dashboard import add_dashboard_routes

        store = ChainStore(db_path=tmp_path / "dash.db")
        # Add some data
        store.add_chain("default", {"question": "Q1", "steps": [],
                                     "final_answer": "A1", "finished": True},
                        risk_score=0.7, alert=True)
        store.add_chain("default", {"question": "Q2", "steps": [],
                                     "final_answer": "A2", "finished": True},
                        risk_score=0.3)

        app = FastAPI()
        add_dashboard_routes(app, store)
        return TestClient(app)

    def test_dashboard_html_returns_200(self, dashboard_client):
        resp = dashboard_client.get("/dashboard")
        assert resp.status_code == 200
        assert "text/html" in resp.headers["content-type"]

    def test_dashboard_stats_api(self, dashboard_client):
        resp = dashboard_client.get("/dashboard/api/stats")
        assert resp.status_code == 200
        body = resp.json()
        assert "n_tracked" in body
        assert "alert_rate" in body
        assert body["n_tracked"] == 2

    def test_dashboard_timeseries_api(self, dashboard_client):
        resp = dashboard_client.get("/dashboard/api/timeseries")
        assert resp.status_code == 200
        body = resp.json()
        assert isinstance(body, list)

    def test_dashboard_chains_api(self, dashboard_client):
        resp = dashboard_client.get("/dashboard/api/chains")
        assert resp.status_code == 200
        chains = resp.json()
        assert isinstance(chains, list)
        assert len(chains) <= 20

    def test_dashboard_domain_filter(self, dashboard_client):
        resp = dashboard_client.get("/dashboard/api/stats?domain=default")
        assert resp.status_code == 200

    def test_dashboard_contains_chartjs(self, dashboard_client):
        resp = dashboard_client.get("/dashboard")
        assert "chart.js" in resp.text.lower() or "Chart" in resp.text


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 8 — Package exports (__init__.py)
# ═══════════════════════════════════════════════════════════════════════════════

class TestPackageExports:

    def test_all_exports_importable(self):
        import qppg_service
        for name in qppg_service.__all__:
            assert hasattr(qppg_service, name), f"Missing export: {name}"

    def test_chain_store_exported(self):
        from qppg_service import ChainStore
        assert ChainStore is not None

    def test_drift_detector_exported(self):
        from qppg_service import DriftDetector, DriftAlert
        assert DriftDetector is not None
        assert DriftAlert is not None

    def test_monitor_exported(self):
        from qppg_service import QppgMonitor, QppgAlert, MonitorStats
        assert QppgMonitor is not None

    def test_label_free_scorer_exported(self):
        from qppg_service import LabelFreeScorer, LabelFreeResult
        assert LabelFreeScorer is not None

    def test_integrations_package_importable(self):
        import qppg_service.integrations
        assert qppg_service.integrations is not None


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 9 — Rate limiter (server.py)
# ═══════════════════════════════════════════════════════════════════════════════

class TestRateLimiter:

    def test_allows_requests_under_limit(self):
        from qppg_service.server import _RateLimiter
        limiter = _RateLimiter(max_per_hour=5)
        for _ in range(5):
            assert limiter.check("key_abc") is True

    def test_rejects_at_limit(self):
        from qppg_service.server import _RateLimiter
        limiter = _RateLimiter(max_per_hour=3)
        for _ in range(3):
            limiter.check("key_xyz")
        assert limiter.check("key_xyz") is False

    def test_different_keys_independent(self):
        from qppg_service.server import _RateLimiter
        limiter = _RateLimiter(max_per_hour=2)
        for _ in range(2):
            limiter.check("key_a")
        # key_a is now at limit; key_b should still be allowed
        assert limiter.check("key_b") is True

    def test_old_timestamps_expire(self):
        from qppg_service.server import _RateLimiter
        limiter = _RateLimiter(max_per_hour=2)
        # Inject old timestamps manually
        limiter._counts["key_old"] = [time.time() - 7200, time.time() - 7200]
        # Old requests should have expired
        assert limiter.check("key_old") is True


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 14 — ChainStore new methods (get_key_by_prefix / revoke_key_by_prefix)
# ═══════════════════════════════════════════════════════════════════════════════

class TestChainStoreKeyMethods:
    """Tests for the key-lookup helper methods added in audit fix."""

    @pytest.fixture
    def store(self, tmp_path):
        from qppg_service.store import ChainStore
        return ChainStore(db_path=tmp_path / "keys.db")

    def test_get_key_by_prefix_returns_record(self, store):
        raw = store.create_api_key("acme", "prod")
        import hashlib
        prefix = hashlib.sha256(raw.encode()).hexdigest()[:12]
        rec = store.get_key_by_prefix(prefix)
        assert rec is not None
        assert rec["customer_id"] == "acme"
        assert "key_hash" not in rec      # hash is never exposed

    def test_get_key_by_prefix_unknown_returns_none(self, store):
        assert store.get_key_by_prefix("000000000000") is None

    def test_revoke_key_by_prefix_returns_true(self, store):
        raw = store.create_api_key("bob", "")
        import hashlib
        prefix = hashlib.sha256(raw.encode()).hexdigest()[:12]
        assert store.revoke_key_by_prefix(prefix) is True
        # Verify it's inactive
        assert store.verify_api_key(raw) is None

    def test_revoke_already_revoked_returns_false(self, store):
        raw = store.create_api_key("carol", "")
        import hashlib
        prefix = hashlib.sha256(raw.encode()).hexdigest()[:12]
        store.revoke_key_by_prefix(prefix)
        assert store.revoke_key_by_prefix(prefix) is False

    def test_revoke_unknown_prefix_returns_false(self, store):
        assert store.revoke_key_by_prefix("000000000000") is False


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 15 — Dashboard Admin endpoints (/dashboard/api/admin/*)
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.skipif(not _HAS_HTTPX, reason="httpx not installed")
class TestDashboardAdmin:
    """Tests for the 8 admin management endpoints added in the Admin UI feature."""

    @pytest.fixture
    def admin_client(self, tmp_path):
        from fastapi import FastAPI
        from qppg_service.store import ChainStore
        from qppg_service.dashboard import add_dashboard_routes

        store = ChainStore(db_path=tmp_path / "admin.db")
        # Seed two domains
        chain = {"question": "Q", "steps": [], "final_answer": "A", "finished": True}
        store.add_chain("dom-a", chain, risk_score=0.3)
        store.add_chain("dom-b", chain, risk_score=0.8, alert=True)

        app = FastAPI()
        add_dashboard_routes(app, store)
        return TestClient(app), store

    # ── GET /dashboard/api/admin/keys ─────────────────────────────────────────

    def test_list_keys_empty(self, admin_client):
        client, _ = admin_client
        resp = client.get("/dashboard/api/admin/keys")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_list_keys_after_create(self, admin_client):
        client, _ = admin_client
        client.post("/dashboard/api/admin/keys",
                    json={"customer_id": "team-x", "domain_prefix": "prod"})
        resp = client.get("/dashboard/api/admin/keys")
        assert resp.status_code == 200
        keys = resp.json()
        assert len(keys) == 1
        assert keys[0]["customer_id"] == "team-x"
        assert keys[0]["domain_prefix"] == "prod"
        assert "key_hash" not in keys[0]           # hash never exposed

    # ── POST /dashboard/api/admin/keys ────────────────────────────────────────

    def test_create_key_returns_api_key(self, admin_client):
        client, _ = admin_client
        resp = client.post("/dashboard/api/admin/keys",
                           json={"customer_id": "alice"})
        assert resp.status_code == 200
        body = resp.json()
        assert "api_key" in body
        assert len(body["api_key"]) > 20
        assert body["customer_id"] == "alice"

    def test_create_key_missing_customer_id_rejected(self, admin_client):
        client, _ = admin_client
        resp = client.post("/dashboard/api/admin/keys", json={"domain_prefix": "x"})
        assert resp.status_code == 400

    # ── POST /dashboard/api/admin/keys/{prefix}/revoke ────────────────────────

    def test_revoke_key(self, admin_client):
        client, store = admin_client
        resp = client.post("/dashboard/api/admin/keys",
                           json={"customer_id": "bob"})
        raw_key = resp.json()["api_key"]
        import hashlib
        prefix = hashlib.sha256(raw_key.encode()).hexdigest()[:12]

        rev = client.post(f"/dashboard/api/admin/keys/{prefix}/revoke")
        assert rev.status_code == 200
        assert rev.json()["revoked"] is True
        # Verify via store
        assert store.verify_api_key(raw_key) is None

    def test_revoke_unknown_key_404(self, admin_client):
        client, _ = admin_client
        resp = client.post("/dashboard/api/admin/keys/000000000000/revoke")
        assert resp.status_code == 404

    # ── GET /dashboard/api/admin/domains ─────────────────────────────────────

    def test_list_domains_returns_stats(self, admin_client):
        client, _ = admin_client
        resp = client.get("/dashboard/api/admin/domains")
        assert resp.status_code == 200
        data = resp.json()
        domains = {d["domain"] for d in data}
        assert "dom-a" in domains
        assert "dom-b" in domains
        b = next(d for d in data if d["domain"] == "dom-b")
        assert b["n_alerts"] == 1

    # ── DELETE /dashboard/api/admin/domains/{domain} ──────────────────────────

    def test_clear_domain(self, admin_client):
        client, store = admin_client
        resp = client.delete("/dashboard/api/admin/domains/dom-a")
        assert resp.status_code == 200
        assert resp.json()["deleted"] >= 1
        assert store.get_chains("dom-a") == []

    def test_clear_nonexistent_domain_returns_zero(self, admin_client):
        client, _ = admin_client
        resp = client.delete("/dashboard/api/admin/domains/no-such-domain")
        assert resp.status_code == 200
        assert resp.json()["deleted"] == 0

    # ── POST /dashboard/api/admin/score ──────────────────────────────────────

    def test_score_playground_returns_risk(self, admin_client):
        client, _ = admin_client
        payload = {
            "question": "What is 2+2?",
            "steps": [
                {"thought": "", "action_type": "Finish",
                 "action_arg": "4", "observation": ""},
            ],
            "domain": "dom-a",
        }
        resp = client.post("/dashboard/api/admin/score", json=payload)
        assert resp.status_code == 200
        body = resp.json()
        assert "risk_score" in body
        assert "needs_review" in body
        assert isinstance(body["risk_score"], float)

    # ── POST /dashboard/api/admin/seed ───────────────────────────────────────

    def test_seed_chains(self, admin_client):
        client, store = admin_client
        chains = [
            {"question": f"Q{i}", "steps": [], "final_answer": "A", "finished": True}
            for i in range(5)
        ]
        resp = client.post("/dashboard/api/admin/seed",
                           json={"domain": "new-dom", "chains": chains})
        assert resp.status_code == 200
        body = resp.json()
        assert body["seeded"] == 5
        assert body["domain"] == "new-dom"
        assert len(store.get_chains("new-dom")) == 5

    def test_seed_empty_chains_returns_zero(self, admin_client):
        client, _ = admin_client
        resp = client.post("/dashboard/api/admin/seed",
                           json={"domain": "x", "chains": []})
        assert resp.status_code == 200
        assert resp.json()["seeded"] == 0

    # ── POST /dashboard/api/admin/seed/cross ─────────────────────────────────

    def test_seed_cross_domain(self, admin_client):
        client, store = admin_client
        resp = client.post("/dashboard/api/admin/seed/cross",
                           json={"source_domain": "dom-a",
                                 "target_domain": "dom-copy", "n": 1})
        assert resp.status_code == 200
        body = resp.json()
        assert body["copied"] == 1
        assert body["source"] == "dom-a"
        assert body["target"] == "dom-copy"
        assert len(store.get_chains("dom-copy")) == 1

    def test_seed_cross_missing_source_returns_400(self, admin_client):
        client, _ = admin_client
        resp = client.post("/dashboard/api/admin/seed/cross",
                           json={"target_domain": "x"})
        assert resp.status_code == 400


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 16 — Dashboard auth (X-Dashboard-Key)
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.skipif(not _HAS_HTTPX, reason="httpx not installed")
class TestDashboardAuth:
    """Admin routes must return 401 when dashboard_key is set and header is wrong."""

    @pytest.fixture
    def auth_client(self, tmp_path):
        from fastapi import FastAPI
        from qppg_service.store import ChainStore
        from qppg_service.dashboard import add_dashboard_routes

        store = ChainStore(db_path=tmp_path / "auth.db")
        app   = FastAPI()
        add_dashboard_routes(app, store, dashboard_key="test-secret")
        return TestClient(app, raise_server_exceptions=False), store

    def test_admin_without_key_returns_401(self, auth_client):
        client, _ = auth_client
        resp = client.get("/dashboard/api/admin/keys")
        assert resp.status_code == 401

    def test_admin_with_wrong_key_returns_401(self, auth_client):
        client, _ = auth_client
        resp = client.get("/dashboard/api/admin/keys",
                          headers={"X-Dashboard-Key": "wrong"})
        assert resp.status_code == 401

    def test_admin_with_correct_key_returns_200(self, auth_client):
        client, _ = auth_client
        resp = client.get("/dashboard/api/admin/keys",
                          headers={"X-Dashboard-Key": "test-secret"})
        assert resp.status_code == 200

    def test_dashboard_page_not_gated(self, auth_client):
        """The /dashboard HTML page itself is never gated."""
        client, _ = auth_client
        resp = client.get("/dashboard")
        assert resp.status_code == 200

    def test_no_key_configured_allows_all(self, tmp_path):
        """When dashboard_key is None admin routes are open (backward compat)."""
        from fastapi import FastAPI
        from qppg_service.store import ChainStore
        from qppg_service.dashboard import add_dashboard_routes

        store  = ChainStore(db_path=tmp_path / "open.db")
        app    = FastAPI()
        add_dashboard_routes(app, store)  # no dashboard_key
        client = TestClient(app)
        resp   = client.get("/dashboard/api/admin/keys")
        assert resp.status_code == 200


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 17 — WebhookAlerter
# ═══════════════════════════════════════════════════════════════════════════════

class TestWebhookAlerter:
    """Unit tests for WebhookAlerter — no real HTTP calls made."""

    def test_import(self):
        from qppg_service.alerting import WebhookAlerter
        a = WebhookAlerter("http://example.com/hook")
        assert a.url == "http://example.com/hook"

    def test_below_threshold_does_not_fire(self):
        from qppg_service.alerting import WebhookAlerter
        a      = WebhookAlerter("http://x", alert_rate_threshold=0.5)
        before = dict(a._last_fired)
        a.fire_alert_rate("dom", 0.1, 100)
        assert a._last_fired == before  # nothing added

    def test_below_min_samples_does_not_fire(self):
        from qppg_service.alerting import WebhookAlerter
        a      = WebhookAlerter("http://x", alert_rate_threshold=0.1, min_samples=20)
        before = dict(a._last_fired)
        a.fire_alert_rate("dom", 0.9, 5)  # only 5 samples, need 20
        assert a._last_fired == before

    def test_cooldown_suppresses_second_fire(self):
        from qppg_service.alerting import WebhookAlerter
        a = WebhookAlerter("http://x", alert_rate_threshold=0.0,
                           min_samples=0, cooldown_seconds=9999)
        a._last_fired["rate:dom"] = time.time()  # simulate a recent fire
        before_len = len(a._last_fired)
        a.fire_alert_rate("dom", 0.99, 100)      # should be suppressed
        # _last_fired was not updated (still same entry)
        assert len(a._last_fired) == before_len

    def test_is_cooling(self):
        from qppg_service.alerting import WebhookAlerter
        a = WebhookAlerter("http://x", cooldown_seconds=3600)
        assert not a._is_cooling("k")
        a._last_fired["k"] = time.time()
        assert a._is_cooling("k")

    def test_fire_drift_calls_post(self, monkeypatch):
        from qppg_service.alerting import WebhookAlerter
        from qppg_service.drift import DriftAlert
        posts = []
        a = WebhookAlerter("http://x")
        monkeypatch.setattr(a, "_post", lambda payload, key: posts.append((payload, key)))
        da = DriftAlert(
            domain="prod", current_mean=0.7, previous_mean=0.5,
            delta=0.2, n_current=20, n_previous=20,
            direction="up", recommendation="Re-calibrate immediately",
        )
        a.fire_drift(da)
        assert len(posts) == 1
        assert posts[0][1] == "drift:prod"
        assert ":warning:" in posts[0][0]["text"]

    def test_fire_alert_rate_calls_post(self, monkeypatch):
        from qppg_service.alerting import WebhookAlerter
        posts = []
        a = WebhookAlerter("http://x", alert_rate_threshold=0.2, min_samples=5)
        monkeypatch.setattr(a, "_post", lambda payload, key: posts.append((payload, key)))
        a.fire_alert_rate("prod", 0.5, 50)
        assert len(posts) == 1
        assert posts[0][1] == "rate:prod"
        assert ":rotating_light:" in posts[0][0]["text"]

    def test_monitor_alerter_param_accepted(self):
        """QppgMonitor(alerter=...) is accepted without error."""
        from qppg_service.alerting import WebhookAlerter
        from qppg_service.monitor import QppgMonitor
        a = WebhookAlerter("http://x")
        monitor = QppgMonitor(threshold=0.65, alerter=a)
        assert monitor._alerter is a

    def test_monitor_fires_alert_rate_on_alert(self):
        """When risk triggers a QppgAlert, alerter.fire_alert_rate is called."""
        from qppg_service.alerting import WebhookAlerter
        from qppg_service.monitor import QppgMonitor
        fired = []

        class _TrackingAlerter(WebhookAlerter):
            def fire_alert_rate(self, domain, rate, n):
                fired.append((domain, rate, n))

        a       = _TrackingAlerter("http://x", alert_rate_threshold=0.0, min_samples=0)
        monitor = QppgMonitor(threshold=0.0, alerter=a, domain="test-dom")
        monitor.track("q", [], "", finished=True)
        assert len(fired) >= 1
        assert fired[0][0] == "test-dom"

    def test_exported_from_package(self):
        from qppg_service import WebhookAlerter  # noqa: F401


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 18 — Active learning review queue (store + endpoints)
# ═══════════════════════════════════════════════════════════════════════════════

class TestStoreReviewQueue:

    @pytest.fixture
    def store(self, tmp_path):
        from qppg_service.store import ChainStore
        return ChainStore(db_path=tmp_path / "rq.db")

    def test_get_review_queue_empty(self, store):
        assert store.get_review_queue("dom") == []

    def test_get_review_queue_returns_alerted_unreviewed(self, store):
        chain = {"question": "What?", "steps": [], "final_answer": "A", "finished": True}
        store.add_chain("dom", chain, risk_score=0.8, alert=True)
        queue = store.get_review_queue("dom")
        assert len(queue) == 1
        assert queue[0]["question"] == "What?"
        assert queue[0]["risk_score"] == pytest.approx(0.8)

    def test_get_review_queue_excludes_non_alerted(self, store):
        chain = {"question": "Q", "steps": [], "final_answer": "A", "finished": True}
        store.add_chain("dom", chain, risk_score=0.3, alert=False)
        assert store.get_review_queue("dom") == []

    def test_label_chain_accept(self, store):
        chain = {"question": "Q", "steps": [], "final_answer": "A", "finished": True}
        cid   = store.add_chain("dom", chain, risk_score=0.9, alert=True)
        assert store.label_chain(cid, "correct") is True
        # Chain should be removed from review queue (reviewed=1)
        assert store.get_review_queue("dom") == []

    def test_label_chain_reject(self, store):
        chain = {"question": "Q", "steps": [], "final_answer": "A", "finished": True}
        cid   = store.add_chain("dom", chain, risk_score=0.9, alert=True)
        assert store.label_chain(cid, "incorrect") is True
        assert store.get_review_queue("dom") == []

    def test_label_chain_nonexistent_returns_false(self, store):
        assert store.label_chain(99999, "correct") is False

    def test_review_queue_respects_n_limit(self, store):
        chain = {"question": "Q", "steps": [], "final_answer": "A", "finished": True}
        for _ in range(5):
            store.add_chain("dom", chain, risk_score=0.9, alert=True)
        assert len(store.get_review_queue("dom", n=3)) == 3


@pytest.mark.skipif(not _HAS_HTTPX, reason="httpx not installed")
class TestDashboardReviewEndpoints:

    @pytest.fixture
    def review_client(self, tmp_path):
        from fastapi import FastAPI
        from qppg_service.store import ChainStore
        from qppg_service.dashboard import add_dashboard_routes

        store = ChainStore(db_path=tmp_path / "review.db")
        chain = {"question": "Test Q?", "steps": [], "final_answer": "A", "finished": True}
        store.add_chain("dom", chain, risk_score=0.9, alert=True)
        app = FastAPI()
        add_dashboard_routes(app, store)
        return TestClient(app), store

    def test_review_queue_returns_alerted_chains(self, review_client):
        client, _ = review_client
        resp = client.get("/dashboard/api/admin/review?domain=dom")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 1
        assert data[0]["question"] == "Test Q?"

    def test_accept_chain_labels_correct(self, review_client):
        client, store = review_client
        cid  = store.get_review_queue("dom")[0]["chain_id"]
        resp = client.post(f"/dashboard/api/admin/review/{cid}/accept")
        assert resp.status_code == 200
        body = resp.json()
        assert body["labeled"] is True
        assert body["label"] == "correct"
        assert store.get_review_queue("dom") == []

    def test_reject_chain_labels_incorrect(self, review_client):
        client, store = review_client
        cid  = store.get_review_queue("dom")[0]["chain_id"]
        resp = client.post(f"/dashboard/api/admin/review/{cid}/reject")
        assert resp.status_code == 200
        assert resp.json()["label"] == "incorrect"

    def test_invalid_action_returns_400(self, review_client):
        client, store = review_client
        cid  = store.get_review_queue("dom")[0]["chain_id"]
        resp = client.post(f"/dashboard/api/admin/review/{cid}/badaction")
        assert resp.status_code == 400

    def test_unknown_chain_id_returns_404(self, review_client):
        client, _ = review_client
        resp = client.post("/dashboard/api/admin/review/99999/accept")
        assert resp.status_code == 404
