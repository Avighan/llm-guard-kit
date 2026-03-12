"""
telemetry.py — Opt-in anonymous label contribution for model improvement

What is sent (11 floats + 1 bit, nothing else):
    {"f": [0.12, 0.33, ...], "y": 1, "d": "a3f9bc", "v": "0.17.0"}

    f = 11 SC_OLD feature values (no text, no questions, no answers)
    y = 0 (correct) or 1 (wrong)
    d = sha256(domain_tag)[:6] — anonymized domain
    v = llm_guard version

What is NEVER sent: question text, steps, final_answer, org identifiers

Backend: Option A (GitHub repository_dispatch webhook)
    POST https://api.github.com/repos/{owner}/{repo}/dispatches
    with Authorization: Bearer {github_token}

Falls back silently on any error — never breaks the main scoring path.
"""

from __future__ import annotations

import hashlib
import json
import threading
import urllib.request
from typing import List


class TelemetryClient:
    """
    Opt-in anonymous telemetry client for label contribution.

    Submits anonymized SC_OLD feature vectors and correctness labels to
    a GitHub repository via repository_dispatch webhook.  All submissions
    happen in a daemon thread so they never block the main scoring path.
    Any error is caught silently — telemetry never disrupts inference.

    Parameters
    ----------
    github_token:
        GitHub Personal Access Token (fine-grained, contents:write on
        the labels repository).
    repo:
        GitHub repository slug in "owner/repo" form where labels are
        stored (default: "amajumder/llm-guard-labels").
    enabled:
        Master switch.  If False, submit() is a no-op regardless of
        whether a token is present.
    """

    _DISPATCH_URL = "https://api.github.com/repos/{repo}/dispatches"

    def __init__(
        self,
        github_token: str,
        repo: str = "amajumder/llm-guard-labels",
        enabled: bool = True,
    ) -> None:
        self._token = github_token
        self._repo = repo
        self._enabled = enabled and bool(github_token)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def submit(
        self,
        features: List[float],
        label: int,
        domain: str = "",
        version: str = "",
    ) -> bool:
        """
        Fire-and-forget label submission.

        Builds the anonymized payload and dispatches a daemon thread to
        POST it to the GitHub API.  Returns immediately; never raises.

        Parameters
        ----------
        features:
            11 SC_OLD feature floats (exact order from _extract_features).
        label:
            0 = chain was correct, 1 = chain was incorrect (high risk).
        domain:
            Optional domain tag string.  Will be sha256-hashed to 6 hex
            chars before transmission — original text is never sent.
        version:
            llm_guard version string (e.g. "0.17.0").

        Returns
        -------
        bool
            True if the background thread was launched successfully.
            False if telemetry is disabled or an error occurred.
        """
        if not self._enabled:
            return False

        try:
            # Anonymize domain: sha256[:6] of the raw tag
            domain_hash = hashlib.sha256(domain.encode()).hexdigest()[:6] if domain else ""

            payload = {
                "f": [round(float(v), 6) for v in features],
                "y": int(label),
                "d": domain_hash,
                "v": str(version),
            }

            t = threading.Thread(
                target=self._send,
                args=(payload,),
                daemon=True,
                name="llm-guard-telemetry",
            )
            t.start()
            return True
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _send(self, payload: dict) -> None:
        """
        Perform the actual HTTPS POST to the GitHub repository_dispatch
        endpoint.  Called from a daemon thread; all exceptions are swallowed.

        Request body::

            {
                "event_type": "label_submission",
                "client_payload": <payload>
            }
        """
        try:
            url = self._DISPATCH_URL.format(repo=self._repo)
            body = json.dumps(
                {"event_type": "label_submission", "client_payload": payload}
            ).encode("utf-8")
            req = urllib.request.Request(
                url,
                data=body,
                method="POST",
                headers={
                    "Authorization": f"Bearer {self._token}",
                    "Accept": "application/vnd.github.v3+json",
                    "Content-Type": "application/json",
                },
            )
            with urllib.request.urlopen(req, timeout=3):
                pass  # 204 No Content on success — nothing to read
        except Exception:
            pass  # never propagate — telemetry must never break inference
