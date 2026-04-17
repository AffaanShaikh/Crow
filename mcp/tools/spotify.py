"""
Spotify Tools - playback control, search, and library access.

Tools based on Spotify Web API v1 with OAuth 2.0 bearer tokens.
Tokens are read from the encrypted token store and auto-refreshed when expired.

TOOLS
─────
  spotify_get_playback       - current track + playback state
  spotify_play               - play / resume (optionally a track, album, playlist URI)
  spotify_pause              - pause playback
  spotify_next_track         - skip to next track
  spotify_previous_track     - go back to previous track / restart current
  spotify_set_repeat         - set repeat mode (off / track / context)
  spotify_set_shuffle        - toggle shuffle on or off
  spotify_set_volume         - set playback volume (0-100)
  spotify_search             - search tracks, albums, artists, playlists
  spotify_get_playlists      - list the user's playlists

ACTIVE DEVICE REQUIREMENT
──────────────────────────
Most playback commands require an active Spotify device (phone, desktop app,
web player). If no device is active, the tool returns a descriptive error.
The user must have Spotify open somewhere for commands to work.

SPOTIFY URI FORMAT
──────────────────
  spotify:track:<id>     e.g. spotify:track:4uLU6hMCjMI75M1A2tKUQC
  spotify:album:<id>
  spotify:playlist:<id>
  spotify:artist:<id>
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

import httpx

from auth import token_store
from config import get_settings
from mcp.schemas import ParameterProperty, ToolDefinition, ToolParameters
from mcp.tools.base import BaseTool
from utils.logger import get_logger

log = get_logger(__name__)
settings = get_settings()

SPOTIFY_API = "https://api.spotify.com/v1"
SPOTIFY_TOKEN_URL = "https://accounts.spotify.com/api/token"


# shared API client

async def _spotify_request(
    method: str,
    path: str,
    *,
    json_body: dict | None = None,
    params: dict | None = None,
    expect_json: bool = True,
) -> dict | None:
    """
    Make an authenticated request to the Spotify Web API.
    Auto-refreshes the access token if expired.
    Raises RuntimeError on auth failure or API errors.
    """
    token = await _get_valid_token()

    async with httpx.AsyncClient() as client:
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }
        resp = await client.request(
            method,
            f"{SPOTIFY_API}{path}",
            headers=headers,
            json=json_body,
            params=params,
            timeout=15.0,
        )

    # 204 No Content - success with no body (common for PUT/POST playback commands)
    if resp.status_code == 204:
        return {}

    if resp.status_code == 401:
        raise RuntimeError(
            "Spotify authentication expired. Please log in again via the Codex panel."
        )
    if resp.status_code == 403:
        raise RuntimeError(
            "Spotify access forbidden. Your account may need Spotify Premium for this action."
        )
    if resp.status_code == 404:
        raise RuntimeError(
            "Spotify resource not found. Ensure a Spotify app is open and active."
        )

    if not resp.is_success:
        try:
            detail = resp.json().get("error", {}).get("message", resp.text[:200])
        except Exception:
            detail = resp.text[:200]
        raise RuntimeError(f"Spotify API error {resp.status_code}: {detail}")

    if expect_json:
        return resp.json()
    return {}


async def _get_valid_token() -> str:
    """
    Return a valid Spotify access token, refreshing if necessary.
    Raises RuntimeError if not authenticated.
    """
    token = await token_store.load_token("spotify")
    if not token or not token.get("access_token"):
        raise RuntimeError(
            "Not authenticated with Spotify. Click 'Connect Spotify' in the Codex panel."
        )

    expires_at = token.get("expires_at")
    if expires_at and time.time() > expires_at - 60:
        # token is expired or about to expire -> refresh it
        token = await _refresh_token(token)

    return token["access_token"]


async def _refresh_token(token: dict) -> dict:
    """Exchange the refresh token for a new access token."""
    refresh_token = token.get("refresh_token")
    client_id = token.get("extra", {}).get("client_id") or settings.spotify_client_id

    if not refresh_token:
        raise RuntimeError("No Spotify refresh token. Please log in again.")

    data = {
        "grant_type":    "refresh_token",
        "refresh_token": refresh_token,
        "client_id":     client_id,
    }
    headers = {"Content-Type": "application/x-www-form-urlencoded"}

    if settings.spotify_client_secret:
        import base64
        creds = base64.b64encode(
            f"{client_id}:{settings.spotify_client_secret}".encode()
        ).decode()
        headers["Authorization"] = f"Basic {creds}"

    async with httpx.AsyncClient() as client:
        resp = await client.post(SPOTIFY_TOKEN_URL, data=data, headers=headers, timeout=10.0)

    if not resp.is_success:
        raise RuntimeError(
            f"Failed to refresh Spotify token: {resp.status_code}. Please log in again."
        )

    new_tokens = resp.json()
    updated = {
        **token,
        "access_token": new_tokens["access_token"],
        "expires_at":   time.time() + new_tokens.get("expires_in", 3600),
    }
    if "refresh_token" in new_tokens:
        updated["refresh_token"] = new_tokens["refresh_token"]

    await token_store.save_token("spotify", updated)
    log.info("spotify_token_refreshed")
    return updated


def _format_track(item: dict) -> dict:
    """Normalise a Spotify track object into a compact LLM-readable dict."""
    if not item:
        return {}
    artists = ", ".join(a["name"] for a in item.get("artists", []))
    album = item.get("album", {}).get("name", "")
    return {
        "id":       item.get("id"),
        "uri":      item.get("uri"),
        "title":    item.get("name"),
        "artists":  artists,
        "album":    album,
        "duration_ms": item.get("duration_ms"),
        "explicit": item.get("explicit", False),
    }


# Tool 1: Get current playback
class SpotifyGetPlaybackTool(BaseTool):

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="spotify_get_playback",
            description=(
                "Get the current Spotify playback state: what's playing, "
                "is it paused or playing, repeat mode, shuffle state, volume, "
                "and which device is active. Use before issuing playback commands "
                "to check the current state."
            ),
            parameters=ToolParameters(properties={}, required=[]),
        )

    async def _run(self) -> dict:
        data = await _spotify_request("GET", "/me/player")
        if not data:
            return {"is_playing": False, "message": "No active Spotify device found."}
        item = data.get("item") or {}
        return {
            "is_playing":    data.get("is_playing", False),
            "track":         _format_track(item),
            "repeat_state":  data.get("repeat_state", "off"),
            "shuffle_state": data.get("shuffle_state", False),
            "volume_percent": data.get("device", {}).get("volume_percent"),
            "device_name":   data.get("device", {}).get("name"),
            "progress_ms":   data.get("progress_ms"),
        }


# Tool 2: Play
class SpotifyPlayTool(BaseTool):

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="spotify_play",
            description=(
                "Start or resume Spotify playback. "
                "Call with no arguments to resume the current track. "
                "Pass a 'context_uri' (album or playlist Spotify URI) to play that context. "
                "Pass a 'track_uri' to play a specific track immediately. "
                "Pass 'track_uris' (list) to play a queue of tracks. "
                "Spotify URIs look like: spotify:track:..., spotify:album:..., spotify:playlist:... "
                "Use spotify_search first to find the URI for a song or playlist by name."
            ),
            parameters=ToolParameters(
                properties={
                    "context_uri": ParameterProperty(
                        type="string",
                        description="Spotify URI of album, artist, or playlist to play.",
                        default=None,
                    ),
                    "track_uri": ParameterProperty(
                        type="string",
                        description="Spotify URI of a single track to play.",
                        default=None,
                    ),
                    "track_uris": ParameterProperty(
                        type="array",
                        description="List of Spotify track URIs to play in order.",
                        default=None,
                        items={"type": "string"},
                    ),
                    "offset_position": ParameterProperty(
                        type="integer",
                        description="Position (0-indexed) in a context to start from.",
                        default=None,
                    ),
                    "device_id": ParameterProperty(
                        type="string",
                        description="Spotify device ID to target. Omit for active device.",
                        default=None,
                    ),
                },
                required=[],
            ),
        )

    async def _run(
        self,
        context_uri: str | None = None,
        track_uri: str | None = None,
        track_uris: list[str] | None = None,
        offset_position: int | None = None,
        device_id: str | None = None,
    ) -> dict:
        body: dict[str, Any] = {}

        if context_uri:
            body["context_uri"] = context_uri
            if offset_position is not None:
                body["offset"] = {"position": offset_position}
        elif track_uri:
            body["uris"] = [track_uri]
        elif track_uris:
            body["uris"] = track_uris

        params = {"device_id": device_id} if device_id else None
        await _spotify_request("PUT", "/me/player/play", json_body=body, params=params, expect_json=False)
        log.info("spotify_play", has_context=bool(context_uri), has_track=bool(track_uri or track_uris))
        return {"success": True, "action": "play"}


# Tool 3: Pause
class SpotifyPauseTool(BaseTool):

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="spotify_pause",
            description="Pause Spotify playback on the active device.",
            parameters=ToolParameters(
                properties={
                    "device_id": ParameterProperty(
                        type="string",
                        description="Spotify device ID. Omit for active device.",
                        default=None,
                    ),
                },
                required=[],
            ),
        )

    async def _run(self, device_id: str | None = None) -> dict:
        params = {"device_id": device_id} if device_id else None
        await _spotify_request("PUT", "/me/player/pause", params=params, expect_json=False)
        return {"success": True, "action": "pause"}


# Tool 4: Next track
class SpotifyNextTrackTool(BaseTool):

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="spotify_next_track",
            description="Skip to the next track in the Spotify playback queue.",
            parameters=ToolParameters(
                properties={
                    "device_id": ParameterProperty(
                        type="string", description="Spotify device ID. Omit for active device.",
                        default=None,
                    ),
                },
                required=[],
            ),
        )

    async def _run(self, device_id: str | None = None) -> dict:
        params = {"device_id": device_id} if device_id else None
        await _spotify_request("POST", "/me/player/next", params=params, expect_json=False)
        return {"success": True, "action": "next_track"}


# Tool 5: Previous track
class SpotifyPreviousTrackTool(BaseTool):

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="spotify_previous_track",
            description=(
                "Go to the previous track in Spotify. "
                "If called within the first 3 seconds of a track, restarts it. "
                "If called after that, goes to the previous track."
            ),
            parameters=ToolParameters(
                properties={
                    "device_id": ParameterProperty(
                        type="string", description="Spotify device ID. Omit for active device.",
                        default=None,
                    ),
                },
                required=[],
            ),
        )

    async def _run(self, device_id: str | None = None) -> dict:
        params = {"device_id": device_id} if device_id else None
        await _spotify_request("POST", "/me/player/previous", params=params, expect_json=False)
        return {"success": True, "action": "previous_track"}


# Tool 6: Set repeat mode
class SpotifySetRepeatTool(BaseTool):

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="spotify_set_repeat",
            description=(
                "Set Spotify repeat mode. "
                "'off' = no repeat, 'track' = loop current song, 'context' = loop playlist/album."
            ),
            parameters=ToolParameters(
                properties={
                    "state": ParameterProperty(
                        type="string",
                        description="Repeat mode: 'off', 'track', or 'context'.",
                        enum=["off", "track", "context"],
                    ),
                    "device_id": ParameterProperty(
                        type="string", description="Spotify device ID. Omit for active device.",
                        default=None,
                    ),
                },
                required=["state"],
            ),
        )

    async def _run(self, state: str, device_id: str | None = None) -> dict:
        params: dict[str, Any] = {"state": state}
        if device_id:
            params["device_id"] = device_id
        await _spotify_request("PUT", "/me/player/repeat", params=params, expect_json=False)
        return {"success": True, "repeat_state": state}


# Tool 7: Set shuffle
class SpotifySetShuffleTool(BaseTool):

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="spotify_set_shuffle",
            description="Turn Spotify shuffle on or off.",
            parameters=ToolParameters(
                properties={
                    "state": ParameterProperty(
                        type="boolean",
                        description="True to enable shuffle, False to disable.",
                    ),
                    "device_id": ParameterProperty(
                        type="string", description="Spotify device ID. Omit for active device.",
                        default=None,
                    ),
                },
                required=["state"],
            ),
        )

    async def _run(self, state: bool, device_id: str | None = None) -> dict:
        params: dict[str, Any] = {"state": str(state).lower()}
        if device_id:
            params["device_id"] = device_id
        await _spotify_request("PUT", "/me/player/shuffle", params=params, expect_json=False)
        return {"success": True, "shuffle_state": state}


# Tool 8: Set volume
class SpotifySetVolumeTool(BaseTool):

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="spotify_set_volume",
            description="Set the Spotify playback volume (0 = mute, 100 = max).",
            parameters=ToolParameters(
                properties={
                    "volume_percent": ParameterProperty(
                        type="integer",
                        description="Volume level from 0 to 100.",
                        minimum=0,
                        maximum=100,
                    ),
                    "device_id": ParameterProperty(
                        type="string", description="Spotify device ID. Omit for active device.",
                        default=None,
                    ),
                },
                required=["volume_percent"],
            ),
        )

    async def _run(self, volume_percent: int, device_id: str | None = None) -> dict:
        params: dict[str, Any] = {"volume_percent": max(0, min(100, volume_percent))}
        if device_id:
            params["device_id"] = device_id
        await _spotify_request("PUT", "/me/player/volume", params=params, expect_json=False)
        return {"success": True, "volume_percent": params["volume_percent"]}


# Tool 9: Search
class SpotifySearchTool(BaseTool):

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="spotify_search",
            description=(
                "Search Spotify for tracks, albums, artists, or playlists by name. "
                "Returns URIs you can pass to spotify_play. "
                "Use this to find the URI for a song or playlist before playing it."
            ),
            parameters=ToolParameters(
                properties={
                    "query": ParameterProperty(
                        type="string",
                        description="Search query (e.g. 'Bohemian Rhapsody', 'Dark Side of the Moon').",
                    ),
                    "type": ParameterProperty(
                        type="string",
                        description="What to search for.",
                        enum=["track", "album", "artist", "playlist"],
                        default="track",
                    ),
                    "limit": ParameterProperty(
                        type="integer",
                        description="Number of results to return (1-10). Default 5.",
                        default=5,
                        minimum=1,
                        maximum=10,
                    ),
                },
                required=["query"],
            ),
        )

    async def _run(self, query: str, type: str = "track", limit: int = 5) -> dict:
        data = await _spotify_request(
            "GET", "/search",
            params={"q": query, "type": type, "limit": limit},
        )
        results: list[dict] = []

        if type == "track":
            items = (data or {}).get("tracks", {}).get("items", [])
            results = [_format_track(t) for t in items]
        elif type == "album":
            items = (data or {}).get("albums", {}).get("items", [])
            results = [
                {"id": a["id"], "uri": a["uri"], "title": a["name"],
                 "artists": ", ".join(x["name"] for x in a.get("artists", []))}
                for a in items
            ]
        elif type == "playlist":
            items = (data or {}).get("playlists", {}).get("items", [])
            results = [
                {"id": p["id"], "uri": p["uri"], "name": p["name"],
                 "owner": p.get("owner", {}).get("display_name", ""),
                 "tracks": p.get("tracks", {}).get("total", 0)}
                for p in items if p
            ]
        elif type == "artist":
            items = (data or {}).get("artists", {}).get("items", [])
            results = [
                {"id": a["id"], "uri": a["uri"], "name": a["name"],
                 "genres": a.get("genres", [])[:3],
                 "popularity": a.get("popularity")}
                for a in items
            ]

        log.info("spotify_search", query=query, type=type, results=len(results))
        return {"query": query, "type": type, "results": results}


# Tool 10: Get user playlists
class SpotifyGetPlaylistsTool(BaseTool):

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="spotify_get_playlists",
            description=(
                "List the current user's Spotify playlists (owned and followed). "
                "Returns playlist names and URIs. "
                "Use the URI with spotify_play to play a specific playlist."
            ),
            parameters=ToolParameters(
                properties={
                    "limit": ParameterProperty(
                        type="integer",
                        description="Max playlists to return (1-50). Default 20.",
                        default=20,
                        minimum=1,
                        maximum=50,
                    ),
                },
                required=[],
            ),
        )

    async def _run(self, limit: int = 20) -> dict:
        data = await _spotify_request(
            "GET", "/me/playlists",
            params={"limit": limit},
        )
        items = (data or {}).get("items", [])
        playlists = [
            {
                "id":     p["id"],
                "uri":    p["uri"],
                "name":   p["name"],
                "owner":  p.get("owner", {}).get("display_name", ""),
                "tracks": p.get("tracks", {}).get("total", 0),
                "public": p.get("public"),
            }
            for p in items if p
        ]
        log.info("spotify_playlists", count=len(playlists))
        return {"playlists": playlists, "total": len(playlists)}
