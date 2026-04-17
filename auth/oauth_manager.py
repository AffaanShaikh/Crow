"""
OAuth state manager - PKCE + CSRF state with TTL.
Handles the transient state required during OAuth flows:
  - CSRF state parameter (random nonce, verified on callback)
  - PKCE code_verifier / code_challenge pair (Spotify uses PKCE)

State entries expire after TTL_SECONDS to prevent stale-state attacks.
All state lives in memory - intentionally not persisted. If the server
restarts mid-flow, the user simply retries the login (harmless).

PKCE (Proof Key for Code Exchange)
PKCE prevents authorization code interception attacks:
  1. Client generates a random 'code_verifier'
  2. Client sends 'code_challenge = BASE64URL(SHA256(code_verifier))' to auth server
  3. On callback, client sends the original 'code_verifier' to token endpoint
  4. Auth server verifies: SHA256(received_verifier) == stored_challenge
  5. If an attacker intercepts the auth code, they can't exchange it without the verifier

Google supports but doesn't require PKCE for installed apps; we use it anyway.
Spotify requires PKCE for the Authorization Code with PKCE flow.
"""

from __future__ import annotations

import base64
import hashlib
import os
import time
from dataclasses import dataclass, field

from utils.logger import get_logger

log = get_logger(__name__)

TTL_SECONDS = 600 # 10 minutes - ample time to complete login no?


@dataclass
class OAuthState:
    state_token: str          # CSRF nonce - verified on callback
    provider: str
    code_verifier: str        # PKCE verifier (sent at token exchange)
    code_challenge: str       # PKCE challenge (sent at authorization)
    redirect_uri: str
    created_at: float = field(default_factory=time.time)

    def is_expired(self) -> bool:
        return time.time() - self.created_at > TTL_SECONDS


# In-memory state store: state_token -> OAuthState
_pending: dict[str, OAuthState] = {}


def create_state(provider: str, redirect_uri: str) -> OAuthState:
    """
    Generate a new OAuth state with PKCE for the given provider.
    Stores the state internally and returns it for use in the auth URL.
    """
    # Clean expired entries first (lazy expiry)
    _purge_expired()

    state_token    = _random_b64url(32)
    code_verifier  = _random_b64url(64)   # 64 bytes -> 86-char base64url string
    code_challenge = _sha256_b64url(code_verifier)

    entry = OAuthState(
        state_token=state_token,
        provider=provider,
        code_verifier=code_verifier,
        code_challenge=code_challenge,
        redirect_uri=redirect_uri,
    )
    _pending[state_token] = entry
    log.debug("oauth_state_created", provider=provider, state=state_token[:8] + "...")
    return entry


def consume_state(state_token: str) -> OAuthState | None:
    """
    Retrieve and remove a pending OAuth state by its token.
    Returns None if not found or expired (in both cases the callback is invalid).
    """
    entry = _pending.pop(state_token, None)
    if entry is None:
        log.warning("oauth_state_not_found", state=state_token[:8] + "...")
        return None
    if entry.is_expired():
        log.warning("oauth_state_expired", provider=entry.provider)
        return None
    log.debug("oauth_state_consumed", provider=entry.provider)
    return entry


def _purge_expired() -> None:
    expired = [k for k, v in _pending.items() if v.is_expired()]
    for k in expired:
        del _pending[k]
    if expired:
        log.debug("oauth_states_purged", count=len(expired))


def _random_b64url(n_bytes: int) -> str:
    """Generate a cryptographically random base64url-encoded string."""
    return base64.urlsafe_b64encode(os.urandom(n_bytes)).rstrip(b"=").decode()


def _sha256_b64url(value: str) -> str:
    """SHA-256 hash of a string, base64url-encoded (PKCE code_challenge)."""
    digest = hashlib.sha256(value.encode()).digest()
    return base64.urlsafe_b64encode(digest).rstrip(b"=").decode()
