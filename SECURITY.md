# Security Policy — llm-guard-kit

## Supported Versions

| Version | Status |
|---------|--------|
| 0.19.x  | ✅ Supported |
| 0.18.x  | Maintenance only |
| < 0.18  | ❌ Not supported |

---

## A2A Trust Object Security

The `A2ATrustObject` uses **HMAC-SHA256** signing with the following properties:

| Property | Implementation |
|----------|---------------|
| Algorithm | HMAC-SHA256 (64-char hex digest) |
| Comparison | `hmac.compare_digest` (constant-time) |
| Replay protection | `issued_at` + `ttl` in canonical payload (default TTL: 300s) |
| Key exposure | Secret never serialised, never appears in `repr()` or wire format |
| Tamper detection | All 8 core trust fields in canonical payload |

### Verified Attack Resistance (31/31 tests pass)

- **Forgery**: Cannot create valid signature without key
- **Tampering**: Any field mutation fails verification
- **Replay**: Expired objects (age > ttl) rejected by default
- **Timing**: Constant-time comparison prevents timing side-channels
- **Key exposure**: Secret never leaks in wire format or repr

---

## Key Rotation Process

When rotating the shared HMAC secret:

1. **Generate new secret** — minimum 32 bytes of entropy:
   ```python
   import secrets
   new_secret = secrets.token_hex(32)
   ```

2. **Deploy with dual-secret window** — support both old and new secrets during rollover:
   ```python
   def verify_with_rotation(trust, primary_secret, old_secret=None):
       if trust.verify(primary_secret):
           return True
       if old_secret and trust.verify(old_secret, check_expiry=False):
           return True  # accept old-signed objects during rollover window
       return False
   ```

3. **Retire old secret** after all in-flight objects have expired (TTL elapsed):
   - Default TTL is 300s (5 min) — wait at least 10 min after deploying new secret
   - Set shorter TTL for faster rotation: `trust.ttl = 60`

4. **Store secrets** in environment variables or a secrets manager, never in code:
   ```bash
   export LLMGUARD_A2A_SECRET="$(python3 -c 'import secrets; print(secrets.token_hex(32))')"
   ```

---

## Dependency Vulnerability Scanning

Automated weekly scan via GitHub Actions (`security.yml`).
Scans only declared dependencies, not the full environment.

**Last scan result:** 0 known vulnerabilities in llm-guard-kit direct dependencies.

To run locally:
```bash
pip install pip-audit
pip-audit --desc -r requirements.txt
```

---

## Reporting Vulnerabilities

Please report security vulnerabilities **privately** via GitHub Security Advisories:
`https://github.com/Avighan/llm-guard-kit/security/advisories/new`

Do not open public issues for security vulnerabilities.

**Response SLA:** Acknowledgement within 48 hours, patch within 14 days for critical issues.

---

## Out of Scope

The following are intentionally out of scope for the current version:

- **Side-channel attacks on the Python runtime** — HMAC operations in CPython are not constant-time at the CPU instruction level; `hmac.compare_digest` is the best available mitigation in pure Python
- **LSTM model inversion** — the `LSTMRiskAccumulator` is a behavioural classifier; model weights are not considered secrets
- **LLM prompt injection via chain content** — chains are scored, not executed; prompt injection via observation text does not affect scoring
