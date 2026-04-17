"""
OAuth authentication routes:-
  GET  /auth/status              - auth status for all providers
  GET  /auth/google/login        - start Google OAuth flow
  GET  /auth/google/callback     - Google OAuth callback (browser redirect)
  DELETE /auth/google/logout     - revoke Google token
  GET  /auth/spotify/login       - start Spotify OAuth flow (PKCE)
  GET  /auth/spotify/callback    - Spotify OAuth callback
  DELETE /auth/spotify/logout    - revoke Spotify token

FLOW OVERVIEW:-
  1. Frontend opens /auth/{provider}/login in the current tab (or popup).
     The endpoint responds with a 302 redirect to the provider's auth page.

  2. User authorizes. Provider redirects to /auth/{provider}/callback?code=...&state=...

  3. Backend verifies state (CSRF), exchanges code for tokens (+ PKCE verifier),
     saves tokens to the encrypted store, then redirects browser to
     FRONTEND_ORIGIN/?auth={provider}&status=ok (or status=error).

  4. Frontend detects the ?auth= param on load (or polls /auth/status) and
     updates the UI to show the provider as connected.

SECURITY
─────────
  - CSRF protection: state parameter verified on every callback
  - PKCE used for Spotify (prevents code interception)
  - Tokens stored server-side only (never sent to frontend)
  - All tokens encrypted at rest (see auth/token_store.py)
  - Redirect URIs must exactly match what's registered in provider console

REDIRECT URI RULES:-
  Both redirect URIs are built from settings.backend_url (default: http://127.0.0.1:8000)
  and must be registered EXACTLY (string match) in each provider's console:

  Google Cloud Console -> APIs & Services -> Credentials -> your OAuth 2.0 Client ID
    -> "Web application" type (NOT Desktop)
    -> Authorised redirect URIs:
        http://127.0.0.1:8000/api/v1/auth/google/callback

  Spotify Developer Dashboard -> your app -> Edit Settings -> Redirect URIs:
        http://127.0.0.1:8000/api/v1/auth/spotify/callback
    NOTE: Spotify explicitly blocks the hostname "localhost".
          127.0.0.1 resolves to the same address but passes their allowlist check.

  If you change the port or deploy behind a proxy, update backend_url in .env:
    BACKEND_URL=https://yourserver.example.com

GOOGLE CREDENTIALS.JSON
──────────────────────────
  Download from Google Cloud Console -> OAuth 2.0 Client IDs -> your client -> ⬇
  Must be "Web application" type. The JSON will contain a "web" key.
  Place the file at the path set by GOOGLE_CREDENTIALS_PATH (default: credentials.json)

GOOGLE SETUP
─────────────
  credentials.json must exist in the backend root.
  For web credentials, add these to Authorized redirect URIs in Google Console:
    http://localhost:8000/api/v1/auth/google/callback
    
SPOTIFY SETUP
─────────────
  Set in .env:
    SPOTIFY_CLIENT_ID=your_client_id
    SPOTIFY_CLIENT_SECRET=your_client_secret   (not needed for PKCE-only, but useful)
  In Spotify Developer Dashboard -> your app -> Edit Settings -> Redirect URIs:
    http://localhost:8000/api/v1/auth/spotify/callback
"""

from __future__ import annotations

import os
import time

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import RedirectResponse

from auth import oauth_manager, token_store
from config import get_settings
from utils.logger import get_logger

log = get_logger(__name__)
settings = get_settings()
router = APIRouter(prefix="/auth", tags=["auth"])

# allow 'http://' for localhost OAuth callbacks (required by google-auth-oauthlib).
# Safe: only applies when backend_url uses http, i.e. local development
if settings.backend_url.startswith("http://"):
    os.environ.setdefault("OAUTHLIB_INSECURE_TRANSPORT", "1")
# where the browser is redirected after auth completes,
# this is the Vite dev. server (or production frontend URL)
_FRONTEND_ORIGIN = settings.cors_origins[0] if settings.cors_origins else "http://localhost:5173"

# Redirect URIs - built once from backend_url so there's one place to change
_GOOGLE_CALLBACK  = f"{settings.backend_url}/api/v1/auth/google/callback"
_SPOTIFY_CALLBACK = f"{settings.backend_url}/api/v1/auth/spotify/callback"



# Spotify OAuth constants

SPOTIFY_AUTH_URL  = "https://accounts.spotify.com/authorize"
SPOTIFY_TOKEN_URL = "https://accounts.spotify.com/api/token"

SPOTIFY_SCOPES = " ".join([
    "user-read-playback-state",
    "user-modify-playback-state",
    "user-read-currently-playing",
    "playlist-read-private",
    "playlist-read-collaborative",
    "user-library-read",
    "streaming",
])

# Status

@router.get("/status")
async def auth_status() -> dict:
    """Return authentication status for all providers."""
    status = await token_store.get_all_auth_status()
    return {"providers": status}

# # Google

# @router.get("/google/login")
# async def google_login():
#     """
#     Start the Google OAuth flow.
#     Redirects the browser to Google's authorization page.
#     """
#     redirect_uri = f"http://{settings.host}:{settings.port}/api/v1/auth/google/callback"
#     # For 0.0.0.0, use localhost for the redirect
#     redirect_uri = redirect_uri.replace("0.0.0.0", "localhost")

#     state = oauth_manager.create_state("google", redirect_uri)

#     try:
#         from google_auth_oauthlib.flow import Flow
#         import json
#         from pathlib import Path

#         creds_path = Path(settings.google_credentials_path)
#         if not creds_path.exists():
#             raise HTTPException(
#                 status_code=503,
#                 detail=f"credentials.json not found at '{creds_path}'. "
#                        "Download from Google Cloud Console.",
#             )

#         raw = json.loads(creds_path.read_text())
#         cred_type = "web" if "web" in raw else "installed"

#         scopes = ["https://www.googleapis.com/auth/calendar"]

#         if cred_type == "web":
#             flow = Flow.from_client_secrets_file(
#                 str(creds_path),
#                 scopes=scopes,
#                 redirect_uri=redirect_uri,
#             )
#         else:
#             # Desktop app - no redirect_uri needed for InstalledAppFlow,
#             # but we're doing web flow here, so use urn:ietf:wg:oauth:2.0:oob
#             # or localhost redirect. For simplicity treat as web flow:
#             flow = Flow.from_client_secrets_file(
#                 str(creds_path),
#                 scopes=scopes,
#                 redirect_uri=redirect_uri,
#             )

#         auth_url, _ = flow.authorization_url(
#             access_type="offline",
#             include_granted_scopes="true",
#             prompt="consent",
#             state=state.state_token,
#         )

#         log.info("google_oauth_redirect", redirect_uri=redirect_uri)
#         return RedirectResponse(url=auth_url)

#     except ImportError:
#         raise HTTPException(
#             status_code=503,
#             detail="google-auth-oauthlib not installed. Run: pip install google-auth-oauthlib",
#         )
#     except Exception as exc:
#         log.error("google_login_error", error=str(exc))
#         raise HTTPException(status_code=500, detail=str(exc))


# @router.get("/google/callback")
# async def google_callback(
#     code: str = Query(None),
#     state: str = Query(None),
#     error: str = Query(None),
# ):
#     """
#     Handle Google's OAuth callback.
#     Exchanges the authorization code for tokens and saves them.
#     """
#     if error:
#         log.warning("google_oauth_error", error=error)
#         return RedirectResponse(url=f"{_FRONTEND_ORIGIN}?auth=google&status=error&reason={error}")

#     if not code or not state:
#         return RedirectResponse(url=f"{_FRONTEND_ORIGIN}?auth=google&status=error&reason=missing_params")

#     # CSRF state verification
#     state_entry = oauth_manager.consume_state(state)
#     if not state_entry or state_entry.provider != "google":
#         log.warning("google_oauth_invalid_state")
#         return RedirectResponse(url=f"{_FRONTEND_ORIGIN}?auth=google&status=error&reason=invalid_state")

#     try:
#         from google_auth_oauthlib.flow import Flow
#         from pathlib import Path
#         import json

#         creds_path = Path(settings.google_credentials_path)
#         raw = json.loads(creds_path.read_text())
#         cred_type = "web" if "web" in raw else "installed"
#         scopes = ["https://www.googleapis.com/auth/calendar"]

#         flow = Flow.from_client_secrets_file(
#             str(creds_path),
#             scopes=scopes,
#             redirect_uri=state_entry.redirect_uri,
#             state=state,
#         )

#         # exchange code for tokens
#         import os
#         os.environ.setdefault("OAUTHLIB_INSECURE_TRANSPORT", "1") # allow http for localhost
#         flow.fetch_token(code=code)
#         credentials = flow.credentials

#         # save token to encrypted store
#         token_data = {
#             "access_token":  credentials.token,
#             "refresh_token": credentials.refresh_token,
#             "expires_at":    credentials.expiry.timestamp() if credentials.expiry else None,
#             "scope":         " ".join(credentials.scopes) if credentials.scopes else None,
#             "token_type":    "Bearer",
#             "extra": {
#                 "token_uri":    credentials.token_uri,
#                 "client_id":    credentials.client_id,
#                 "client_secret": credentials.client_secret,
#             },
#         }
#         await token_store.save_token("google", token_data)

#         # also writing token.json for the existing google_calendar.py compatibility
#         from pathlib import Path as P
#         P(settings.google_token_path).write_text(credentials.to_json())

#         # invalidate the cached service so it picks up the new token
#         from mcp.tools.google_calendar import GoogleCalendarClient
#         GoogleCalendarClient.invalidate()

#         log.info("google_auth_complete")
#         return RedirectResponse(url=f"{_FRONTEND_ORIGIN}?auth=google&status=ok")

#     except Exception as exc:
#         log.error("google_callback_error", error=str(exc))
#         return RedirectResponse(
#             url=f"{_FRONTEND_ORIGIN}?auth=google&status=error&reason=token_exchange_failed"
#         )


# @router.delete("/google/logout")
# async def google_logout():
#     """Revoke Google token and clear local storage."""
#     token = await token_store.load_token("google")
#     if token and token.get("access_token"):
#         # Best-effort revocation - Google revoke endpoint
#         try:
#             import httpx
#             async with httpx.AsyncClient() as client:
#                 await client.post(
#                     "https://oauth2.googleapis.com/revoke",
#                     params={"token": token["access_token"]},
#                 )
#         except Exception:
#             pass # revocation failure is non-fatal

#     await token_store.delete_token("google")
#     from mcp.tools.google_calendar import GoogleCalendarClient
#     GoogleCalendarClient.invalidate()

#     # Remove token.json
#     from pathlib import Path
#     tp = Path(settings.google_token_path)
#     if tp.exists():
#         tp.unlink()

#     log.info("google_logout_complete")
#     return {"status": "logged_out", "provider": "google"}
@router.get("/google/login")
async def google_login():
    """
    Start the Google OAuth 2.0 flow.

    credentials.json must be a Web Application credential (not Desktop) to support OAuth redirects,
    Download from Cloud Console -> APIs & Services -> Credentials -> your client
    """
    from pathlib import Path
    import json

    creds_path = Path(settings.google_credentials_path)
    if not creds_path.exists():
        raise HTTPException(
            status_code=503,
            detail=(
                f"credentials.json not found at '{creds_path}'. "
                "Download from Google Cloud Console (Web application type)."
            ),
        )

    raw = json.loads(creds_path.read_text())
    if "web" not in raw:
        raise HTTPException(
            status_code=503,
            detail=(
                "credentials.json is a Desktop App credential. "
                "OAuth redirects require a Web Application credential. "
                "In Google Cloud Console: create a new OAuth 2.0 Client ID "
                f"of type 'Web application', add '{_GOOGLE_CALLBACK}' as an "
                "Authorised redirect URI, then download and replace credentials.json."
            ),
        )

    try:
        from google_auth_oauthlib.flow import Flow

        state = oauth_manager.create_state("google", _GOOGLE_CALLBACK)
        flow = Flow.from_client_secrets_file(
            str(creds_path),
            scopes=["https://www.googleapis.com/auth/calendar"],
            redirect_uri=_GOOGLE_CALLBACK,
        )
        auth_url, _ = flow.authorization_url(
            access_type="offline",
            include_granted_scopes="true",
            prompt="consent",
            state=state.state_token,
        )
        log.info("google_oauth_start", redirect_uri=_GOOGLE_CALLBACK)
        return RedirectResponse(url=auth_url)

    except ImportError:
        raise HTTPException(
            status_code=503,
            detail="google-auth-oauthlib not installed: pip install google-auth-oauthlib",
        )
    except Exception as exc:
        log.error("google_login_error", error=str(exc))
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/google/callback")
async def google_callback(
    code: str = Query(None),
    state: str = Query(None),
    error: str = Query(None),
):
    """
    Handles Google OAuth callback. Exchanges code for tokens, saves them, redirects to frontend.
    """
    def _err(reason: str):
        return RedirectResponse(url=f"{_FRONTEND_ORIGIN}?auth=google&status=error&reason={reason}")

    if error:
        log.warning("google_oauth_denied", error=error)
        return _err(error)
    if not code or not state:
        return _err("missing_params")

    state_entry = oauth_manager.consume_state(state)
    if not state_entry or state_entry.provider != "google":
        log.warning("google_oauth_invalid_state")
        return _err("invalid_state")

    try:
        from google_auth_oauthlib.flow import Flow
        from pathlib import Path

        flow = Flow.from_client_secrets_file(
            settings.google_credentials_path,
            scopes=["https://www.googleapis.com/auth/calendar"],
            redirect_uri=_GOOGLE_CALLBACK,
            state=state,
        )
        flow.fetch_token(code=code)
        creds = flow.credentials

        await token_store.save_token("google", {
            "access_token":  creds.token,
            "refresh_token": creds.refresh_token,
            "expires_at":    creds.expiry.timestamp() if creds.expiry else None,
            "scope":         " ".join(creds.scopes) if creds.scopes else None,
            "token_type":    "Bearer",
            "extra": {
                "token_uri":     creds.token_uri,
                "client_id":     creds.client_id,
                "client_secret": creds.client_secret,
            },
        })

        # write token.json for GoogleCalendarClient compatibility
        Path(settings.google_token_path).write_text(creds.to_json())

        # invalidate cached service so it uses the new token immediately
        from mcp.tools.google_calendar import GoogleCalendarClient
        GoogleCalendarClient.invalidate()

        log.info("google_auth_complete")
        return RedirectResponse(url=f"{_FRONTEND_ORIGIN}?auth=google&status=ok")

    except Exception as exc:
        log.error("google_callback_error", error=str(exc))
        return _err("token_exchange_failed")


@router.delete("/google/logout")
async def google_logout():
    """Revoke the Google token and clear local storage."""
    token = await token_store.load_token("google")
    if token and token.get("access_token"):
        try:
            import httpx
            async with httpx.AsyncClient() as client:
                await client.post(
                    "https://oauth2.googleapis.com/revoke",
                    params={"token": token["access_token"]},
                )
        except Exception:
            pass # revocation failure is non-fatal - token is still deleted locally

    await token_store.delete_token("google")

    from mcp.tools.google_calendar import GoogleCalendarClient
    GoogleCalendarClient.invalidate()

    from pathlib import Path
    tp = Path(settings.google_token_path)
    if tp.exists():
        tp.unlink()

    log.info("google_logout_complete")
    return {"status": "logged_out", "provider": "google"}


# Spotify

# @router.get("/spotify/login")
# async def spotify_login():
#     """
#     Start the Spotify OAuth 2.0 + PKCE flow.
#     Redirects the browser to Spotify's authorization page.
#     """
#     if not settings.spotify_client_id:
#         raise HTTPException(
#             status_code=503,
#             detail="SPOTIFY_CLIENT_ID not set in .env",
#         )

#     redirect_uri = f"http://localhost:{settings.port}/api/v1/auth/spotify/callback"
#     state = oauth_manager.create_state("spotify", redirect_uri)

#     import urllib.parse
#     params = urllib.parse.urlencode({
#         "client_id":             settings.spotify_client_id,
#         "response_type":         "code",
#         "redirect_uri":          redirect_uri,
#         "state":                 state.state_token,
#         "scope":                 SPOTIFY_SCOPES,
#         "code_challenge_method": "S256",
#         "code_challenge":        state.code_challenge,
#     })
#     auth_url = f"{SPOTIFY_AUTH_URL}?{params}"
#     log.info("spotify_oauth_redirect", redirect_uri=redirect_uri)
#     return RedirectResponse(url=auth_url)


# @router.get("/spotify/callback")
# async def spotify_callback(
#     code: str = Query(None),
#     state: str = Query(None),
#     error: str = Query(None),
# ):
#     """
#     Handle Spotify's OAuth callback.
#     Exchanges the authorization code + PKCE verifier for tokens.
#     """
#     if error:
#         log.warning("spotify_oauth_error", error=error)
#         return RedirectResponse(url=f"{_FRONTEND_ORIGIN}?auth=spotify&status=error&reason={error}")

#     if not code or not state:
#         return RedirectResponse(url=f"{_FRONTEND_ORIGIN}?auth=spotify&status=error&reason=missing_params")

#     state_entry = oauth_manager.consume_state(state)
#     if not state_entry or state_entry.provider != "spotify":
#         log.warning("spotify_oauth_invalid_state")
#         return RedirectResponse(url=f"{_FRONTEND_ORIGIN}?auth=spotify&status=error&reason=invalid_state")

#     try:
#         import httpx
#         import base64

#         # Exchange authorization code for tokens using PKCE verifier
#         redirect_uri = state_entry.redirect_uri
#         data = {
#             "grant_type":    "authorization_code",
#             "code":          code,
#             "redirect_uri":  redirect_uri,
#             "code_verifier": state_entry.code_verifier, # PKCE
#         }
#         headers = {"Content-Type": "application/x-www-form-urlencoded"}

#         # If client_secret is set, use basic auth (more compatible with some apps)
#         if settings.spotify_client_secret:
#             creds = base64.b64encode(
#                 f"{settings.spotify_client_id}:{settings.spotify_client_secret}".encode()
#             ).decode()
#             headers["Authorization"] = f"Basic {creds}"
#         else:
#             data["client_id"] = settings.spotify_client_id

#         async with httpx.AsyncClient() as client:
#             resp = await client.post(SPOTIFY_TOKEN_URL, data=data, headers=headers)
#             resp.raise_for_status()
#             tokens = resp.json()

#         token_data = {
#             "access_token":  tokens["access_token"],
#             "refresh_token": tokens.get("refresh_token"),
#             "expires_at":    time.time() + tokens.get("expires_in", 3600),
#             "scope":         tokens.get("scope"),
#             "token_type":    tokens.get("token_type", "Bearer"),
#             "extra": {
#                 "client_id": settings.spotify_client_id,
#             },
#         }
#         await token_store.save_token("spotify", token_data)

#         log.info("spotify_auth_complete")
#         return RedirectResponse(url=f"{_FRONTEND_ORIGIN}?auth=spotify&status=ok")

#     except Exception as exc:
#         log.error("spotify_callback_error", error=str(exc))
#         return RedirectResponse(
#             url=f"{_FRONTEND_ORIGIN}?auth=spotify&status=error&reason=token_exchange_failed"
#         )


# @router.delete("/spotify/logout")
# async def spotify_logout():
#     """Clear the Spotify token."""
#     await token_store.delete_token("spotify")
#     log.info("spotify_logout_complete")
#     return {"status": "logged_out", "provider": "spotify"}
@router.get("/spotify/login")
async def spotify_login():
    """
    Start the Spotify OAuth 2.0 + PKCE flow.

    The redirect URI must be registered in Spotify Developer Dashboard exactly as:
        http://127.0.0.1:8000/api/v1/auth/spotify/callback
    Spotify rejects the hostname "localhost" - 127.0.0.1 must be used.
    """
    if not settings.spotify_client_id:
        raise HTTPException(
            status_code=503,
            detail="SPOTIFY_CLIENT_ID not set in .env",
        )

    import urllib.parse

    state = oauth_manager.create_state("spotify", _SPOTIFY_CALLBACK)
    params = urllib.parse.urlencode({
        "client_id":             settings.spotify_client_id,
        "response_type":         "code",
        "redirect_uri":          _SPOTIFY_CALLBACK,
        "state":                 state.state_token,
        "scope":                 SPOTIFY_SCOPES,
        "code_challenge_method": "S256",
        "code_challenge":        state.code_challenge,
    })
    log.info("spotify_oauth_start", redirect_uri=_SPOTIFY_CALLBACK)
    auth_url = f"{SPOTIFY_AUTH_URL}?{params}"
    log.info(
        "spotify_auth_url_built",
        client_id_preview=settings.spotify_client_id[:6] if settings.spotify_client_id else None,
        auth_url=auth_url,
    )
    return RedirectResponse(url=auth_url)
    # return RedirectResponse(url=f"{SPOTIFY_AUTH_URL}?{params}")


@router.get("/spotify/callback")
async def spotify_callback(
    code: str = Query(None),
    state: str = Query(None),
    error: str = Query(None),
):
    """
    Spotify OAuth callback. Exchanges code + PKCE verifier for tokens.
    """
    def _err(reason: str):
        return RedirectResponse(url=f"{_FRONTEND_ORIGIN}?auth=spotify&status=error&reason={reason}")

    if error:
        log.warning("spotify_oauth_denied", error=error)
        return _err(error)
    if not code or not state:
        return _err("missing_params")

    state_entry = oauth_manager.consume_state(state)
    if not state_entry or state_entry.provider != "spotify":
        log.warning("spotify_oauth_invalid_state")
        return _err("invalid_state")

    try:
        import httpx
        import base64

        data = {
            "grant_type":    "authorization_code",
            "code":          code,
            "redirect_uri":  _SPOTIFY_CALLBACK,
            "code_verifier": state_entry.code_verifier, # PKCE verifier
        }
        headers = {"Content-Type": "application/x-www-form-urlencoded"}

        if settings.spotify_client_secret:
            # basic auth (client_id:client_secret) - more compatible
            creds = base64.b64encode(
                f"{settings.spotify_client_id}:{settings.spotify_client_secret}".encode()
            ).decode()
            headers["Authorization"] = f"Basic {creds}"
        else:
            # PKCE-only - client_id in body, no secret needed
            data["client_id"] = settings.spotify_client_id

        async with httpx.AsyncClient() as client:
            resp = await client.post(SPOTIFY_TOKEN_URL, data=data, headers=headers, timeout=15.0)
            resp.raise_for_status()
            tokens = resp.json()

        await token_store.save_token("spotify", {
            "access_token":  tokens["access_token"],
            "refresh_token": tokens.get("refresh_token"),
            "expires_at":    time.time() + tokens.get("expires_in", 3600),
            "scope":         tokens.get("scope"),
            "token_type":    tokens.get("token_type", "Bearer"),
            "extra":         {"client_id": settings.spotify_client_id},
        })

        log.info("spotify_auth_complete")
        return RedirectResponse(url=f"{_FRONTEND_ORIGIN}?auth=spotify&status=ok")

    except Exception as exc:
        log.error("spotify_callback_error", error=str(exc))
        return _err("token_exchange_failed")


@router.delete("/spotify/logout")
async def spotify_logout():
    """Clear the Spotify token."""
    await token_store.delete_token("spotify")
    log.info("spotify_logout_complete")
    return {"status": "logged_out", "provider": "spotify"}
