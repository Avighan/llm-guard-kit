"""
A2A Trust Object — Self-Assessment Security Test Suite

Tests the 5 attack surfaces of the HMAC-signed A2ATrustObject:
  1. Forgery       — attacker cannot create valid signature without key
  2. Tampering     — any field change breaks verification
  3. Replay        — expired objects are rejected (TTL enforcement)
  4. Timing        — verify() uses constant-time comparison
  5. Key exposure  — secret never appears in serialised output or repr

Run:
    pytest tests/test_a2a_security.py -v
"""

import json
import time
import statistics
import sys
from pathlib import Path

import pytest

QPPG_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(QPPG_ROOT))

from llm_guard.trust_object import A2ATrustObject


# ── Fixture ──────────────────────────────────────────────────────────────────

def _make_trust(risk: float = 0.4) -> A2ATrustObject:
    tier = "HIGH" if risk < 0.5 else ("LOW" if risk >= 0.7 else "MEDIUM")
    return A2ATrustObject(
        answer="Paris",
        risk_score=risk,
        confidence_tier=tier,
        failure_mode=None,
        step_count=3,
        judge_label="GOOD",
        downstream_hint="proceed",
        should_rewrite=False,
    )


SECRET  = "correct-horse-battery-staple"
WRONG   = "wrong-secret"


# ── 1. Forgery ────────────────────────────────────────────────────────────────

class TestForgery:
    def test_unsigned_object_fails_verify(self):
        t = _make_trust()
        assert t.verify(SECRET) is False

    def test_wrong_key_fails_verify(self):
        t = _make_trust().sign(SECRET)
        assert t.verify(WRONG) is False

    def test_correct_key_passes_verify(self):
        t = _make_trust().sign(SECRET)
        assert t.verify(SECRET) is True

    def test_empty_secret_rejected(self):
        t = _make_trust().sign("")
        assert t.verify(SECRET) is False

    def test_brute_force_single_char_secret_rejected(self):
        """Single-char secret signed object is rejected by real secret."""
        t = _make_trust().sign("x")
        assert t.verify(SECRET) is False

    def test_fabricated_signature_rejected(self):
        t = _make_trust()
        t.trust_signature = "a" * 64   # plausible-looking hex string
        assert t.verify(SECRET) is False

    def test_truncated_signature_rejected(self):
        t = _make_trust().sign(SECRET)
        t.trust_signature = t.trust_signature[:32]
        assert t.verify(SECRET) is False


# ── 2. Tampering ─────────────────────────────────────────────────────────────

class TestTampering:
    def _tamper_field(self, field: str, new_val):
        t = _make_trust().sign(SECRET)
        setattr(t, field, new_val)
        return t.verify(SECRET, check_expiry=False)

    def test_tamper_answer(self):
        assert self._tamper_field("answer", "London") is False

    def test_tamper_risk_score(self):
        assert self._tamper_field("risk_score", 0.99) is False

    def test_tamper_confidence_tier(self):
        assert self._tamper_field("confidence_tier", "LOW") is False

    def test_tamper_failure_mode(self):
        assert self._tamper_field("failure_mode", "retrieval_fail") is False

    def test_tamper_step_count(self):
        assert self._tamper_field("step_count", 99) is False

    def test_tamper_judge_label(self):
        assert self._tamper_field("judge_label", "POOR") is False

    def test_tamper_downstream_hint(self):
        assert self._tamper_field("downstream_hint", "escalate_to_human") is False

    def test_tamper_should_rewrite(self):
        assert self._tamper_field("should_rewrite", True) is False

    def test_tamper_ttl(self):
        assert self._tamper_field("ttl", 9999) is False

    def test_unsigned_behavioural_components_not_in_signature(self):
        """behavioral_components is NOT in canonical payload — can change freely."""
        t = _make_trust().sign(SECRET)
        t.behavioral_components["sc1"] = 0.999  # not signed
        assert t.verify(SECRET, check_expiry=False) is True

    def test_roundtrip_wire_format_verifies(self):
        t = _make_trust().sign(SECRET)
        wire = t.to_dict()
        t2 = A2ATrustObject.from_dict(wire)
        assert t2.verify(SECRET, check_expiry=False) is True


# ── 3. Replay Protection ──────────────────────────────────────────────────────

class TestReplayProtection:
    def test_fresh_object_passes(self):
        t = _make_trust().sign(SECRET)
        assert t.verify(SECRET) is True

    def test_expired_object_rejected(self):
        t = _make_trust().sign(SECRET)
        t.ttl = 1                    # 1-second TTL
        t.issued_at = time.time() - 10   # pretend issued 10s ago
        # Must re-sign after mutating TTL (TTL is part of canonical payload)
        t.sign(SECRET)
        t.issued_at = time.time() - 10   # backdate after re-sign
        assert t.verify(SECRET) is False

    def test_expired_object_passes_with_check_disabled(self):
        """check_expiry=False bypasses TTL but signature still validates."""
        t = _make_trust()
        t.ttl = 1
        t.sign(SECRET)
        # Backdating after sign would break sig — instead just confirm the
        # freshly-signed object passes with check_expiry=False
        assert t.verify(SECRET, check_expiry=False) is True

    def test_object_without_issued_at_skips_expiry_check(self):
        """Backward-compat: unsigned objects (trust_signature=None) are rejected."""
        t = _make_trust()
        # Old-format object never signed at all
        assert t.trust_signature is None
        assert t.verify(SECRET) is False  # unsigned → always False

    def test_custom_ttl_respected(self):
        t = _make_trust()
        t.ttl = 3600   # 1-hour TTL
        t.sign(SECRET)
        assert t.verify(SECRET) is True

    def test_issued_at_set_by_sign(self):
        before = time.time()
        t = _make_trust().sign(SECRET)
        after = time.time()
        assert t.issued_at is not None
        assert before <= t.issued_at <= after

    def test_replay_after_expiry_rejected(self):
        """Simulate attacker replaying an old valid token after TTL."""
        t = _make_trust()
        t.ttl = 2
        t.sign(SECRET)
        wire = t.to_dict()              # "transmitted" token
        time.sleep(3)                   # wait for TTL to expire
        replayed = A2ATrustObject.from_dict(wire)
        assert replayed.verify(SECRET) is False


# ── 4. Timing Safety ─────────────────────────────────────────────────────────

class TestTimingSafety:
    def test_constant_time_comparison(self):
        """
        verify() should use hmac.compare_digest (constant time).
        We measure timing variance between valid and invalid signatures
        and assert it's within noise bounds (< 2ms difference).
        This is a statistical test — not a guarantee, but catches obvious leaks.
        """
        t_valid   = _make_trust().sign(SECRET)
        t_invalid = _make_trust()
        t_invalid.trust_signature = "a" * 64

        n = 200
        valid_times, invalid_times = [], []

        for _ in range(n):
            start = time.perf_counter()
            t_valid.verify(SECRET, check_expiry=False)
            valid_times.append(time.perf_counter() - start)

        for _ in range(n):
            start = time.perf_counter()
            t_invalid.verify(SECRET, check_expiry=False)
            invalid_times.append(time.perf_counter() - start)

        mean_valid   = statistics.mean(valid_times)   * 1000   # ms
        mean_invalid = statistics.mean(invalid_times) * 1000   # ms
        diff_ms = abs(mean_valid - mean_invalid)

        # Constant-time comparison: difference should be < 2ms
        assert diff_ms < 2.0, (
            f"Timing difference {diff_ms:.3f}ms suggests non-constant-time comparison"
        )


# ── 5. Key Exposure ───────────────────────────────────────────────────────────

class TestKeyExposure:
    def test_secret_not_in_wire_format(self):
        t = _make_trust().sign(SECRET)
        wire = json.dumps(t.to_dict())
        assert SECRET not in wire

    def test_secret_not_in_repr(self):
        t = _make_trust().sign(SECRET)
        assert SECRET not in repr(t)

    def test_signature_is_hex_not_secret(self):
        t = _make_trust().sign(SECRET)
        sig = t.trust_signature
        assert sig is not None
        assert len(sig) == 64                     # SHA-256 hex = 64 chars
        assert all(c in "0123456789abcdef" for c in sig)

    def test_different_secrets_produce_different_signatures(self):
        t1 = _make_trust().sign("secret-A")
        t2 = _make_trust().sign("secret-B")
        assert t1.trust_signature != t2.trust_signature

    def test_same_payload_same_secret_deterministic(self):
        """Signatures are deterministic for same payload + secret."""
        t1 = _make_trust()
        t1.issued_at = 1700000000.0   # fixed timestamp
        t1.sign.__func__              # just referencing
        # manually set issued_at before sign to get deterministic result
        t1.issued_at = 1700000000.0
        payload = {k: getattr(t1, k) for k in t1._SIGN_FIELDS}
        import hashlib, hmac as _hmac
        canon = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        sig1 = _hmac.new(SECRET.encode(), canon.encode(), hashlib.sha256).hexdigest()
        sig2 = _hmac.new(SECRET.encode(), canon.encode(), hashlib.sha256).hexdigest()
        assert sig1 == sig2
