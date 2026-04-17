"""
Token store - encrypted multi-provider OAuth token persistence.
Stores access tokens, refresh tokens, and expiry times per provider in an
encrypted JSON file. Each provider's data is isolated.

Security model:-
- Tokens are encrypted at rest using Fernet symmetric encryption
  (AES-128-CBC + HMAC-SHA256 via the 'cryptography' library).
- The encryption key is auto-generated on first run and stored in
  'tokens/.key' with mode 0o600 (owner-read only).
- The encrypted token file is stored at 'tokens/.tokens.enc'.
- Neither file should ever be committed to version control.

contract for provider:-
Each provider entry stores:
  {
    "access_token":  str,
    "refresh_token": str | None,
    "expires_at":    float | None,   # Unix timestamp
    "scope":         str | None,
    "token_type":    str,
    "extra":         dict            # provider-specific fields
  }

Thread safety: File I/O is synchronous; async wrappers run operations in the thread pool.
A module-level asyncio.Lock prevents concurrent writes corrupting the file.
"""

from __future__ import annotations

import asyncio
import json
import os
import stat
import time
from pathlib import Path
from typing import Any

from utils.logger import get_logger

log = get_logger(__name__)

# Paths

_TOKENS_DIR  = Path(os.getenv("TOKENS_DIR", "tokens"))
_KEY_FILE    = _TOKENS_DIR / ".key"
_TOKENS_FILE = _TOKENS_DIR / ".tokens.enc"

# Write lock (prevents concurrent file corruption)
_write_lock = asyncio.Lock()


# Lazy-loaded encryption key

def _get_or_create_key() -> bytes:
    """
    Load the Fernet key from disk, creating it if it doesn't exist.
    Sets file permissions to 0o600 (owner-read-write only).
    """
    _TOKENS_DIR.mkdir(parents=True, exist_ok=True)

    if _KEY_FILE.exists():
        key = _KEY_FILE.read_bytes().strip()
        log.debug("token_key_loaded")
        return key

    from cryptography.fernet import Fernet
    key = Fernet.generate_key()
    _KEY_FILE.write_bytes(key)
    _KEY_FILE.chmod(0o600)
    log.info("token_key_generated", path=str(_KEY_FILE))
    return key


def _get_fernet():
    from cryptography.fernet import Fernet
    return Fernet(_get_or_create_key())


# Core read / write

def _read_all() -> dict[str, Any]:
    """Read and decrypt the token store. Returns {} if file doesn't exist."""
    if not _TOKENS_FILE.exists():
        return {}
    try:
        f = _get_fernet()
        encrypted = _TOKENS_FILE.read_bytes()
        plain = f.decrypt(encrypted)
        return json.loads(plain)
    except Exception as exc:
        log.warning("token_store_read_failed", error=str(exc), action="returning_empty")
        return {}


def _write_all(data: dict[str, Any]) -> None:
    """Encrypt and persist the full token store."""
    _TOKENS_DIR.mkdir(parents=True, exist_ok=True)
    f = _get_fernet()
    plain = json.dumps(data, indent=2).encode()
    encrypted = f.encrypt(plain)
    _TOKENS_FILE.write_bytes(encrypted)
    _TOKENS_FILE.chmod(0o600)


# Public async API

async def save_token(provider: str, token_data: dict[str, Any]) -> None:
    """
    Persist an OAuth token for the given provider.

    Args:
        provider:   e.g. "google", "spotify"
        token_data: Dict containing at minimum "access_token".
                    Optional keys: refresh_token, expires_at, scope, token_type, extra.
    """
    loop = asyncio.get_event_loop()
    async with _write_lock:
        await loop.run_in_executor(None, _save_token_sync, provider, token_data)
    log.info("token_saved", provider=provider)


def _save_token_sync(provider: str, token_data: dict[str, Any]) -> None:
    store = _read_all()
    store[provider] = {
        "access_token":  token_data.get("access_token", ""),
        "refresh_token": token_data.get("refresh_token"),
        "expires_at":    token_data.get("expires_at"),
        "scope":         token_data.get("scope"),
        "token_type":    token_data.get("token_type", "Bearer"),
        "extra":         token_data.get("extra", {}),
        "saved_at":      time.time(),
    }
    _write_all(store)


async def load_token(provider: str) -> dict[str, Any] | None:
    """Return the stored token for a provider, or None if not found."""
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, _load_token_sync, provider)
    return result


def _load_token_sync(provider: str) -> dict[str, Any] | None:
    store = _read_all()
    return store.get(provider)


async def delete_token(provider: str) -> None:
    """Remove the stored token for a provider (logout)."""
    loop = asyncio.get_event_loop()
    async with _write_lock:
        await loop.run_in_executor(None, _delete_token_sync, provider)
    log.info("token_deleted", provider=provider)


def _delete_token_sync(provider: str) -> None:
    store = _read_all()
    store.pop(provider, None)
    _write_all(store)


async def is_authenticated(provider: str) -> bool:
    """Return True if the provider has a stored, apparently valid token."""
    token = await load_token(provider)
    if not token or not token.get("access_token"):
        return False
    expires_at = token.get("expires_at")
    if expires_at and time.time() > expires_at - 60:
        # expired token (or within 60s of expiry) - needs refresh
        return False
    return True


async def get_all_auth_status() -> dict[str, bool]:
    """Return auth status for all known providers."""
    providers = ["google", "spotify"]
    status = {}
    for p in providers:
        status[p] = await is_authenticated(p)
    return status
