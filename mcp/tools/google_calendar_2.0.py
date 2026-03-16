"""
Google Calendar Tools — full CRUD via Google Calendar API v3.

Authentication — OAuth2 with offline access (refresh token).

YOUR credentials.json IS TYPE "web" — that's fine, but Google requires
the redirect URI to be pre-registered. On first run this file opens a
local browser flow and listens on http://localhost:8085/callback.

SETUP (one-time):
  1. Go to console.cloud.google.com → your project → APIs & Services → Credentials
  2. Click your OAuth 2.0 Client ID (the web type you already have)
  3. Under "Authorised redirect URIs" add:  http://localhost:8085/
  4. Save. Wait ~5 minutes for Google to propagate.
  5. Re-download credentials.json and replace your current file.
  6. Start the backend — a browser tab will open for consent.
  7. token.json is created automatically. Future runs skip the browser.

  Alternatively: create a NEW credential of type "Desktop app" — those
  work with any localhost port without needing to register URIs.

Tools registered:
  list_calendar_events   — query events with time range + keyword filter
  get_calendar_event     — fetch one event by ID
  create_calendar_event  — create with full field support
  update_calendar_event  — partial update (only changed fields)
  delete_calendar_event  — permanent delete
"""

from __future__ import annotations

import asyncio
import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from mcp.schemas import ParameterProperty, ToolDefinition, ToolParameters
from mcp.tools.base import BaseTool
from utils.logger import get_logger

log = get_logger(__name__)

SCOPES = ["https://www.googleapis.com/auth/calendar"]
# Port for the local OAuth callback server.
# Must match the redirect URI registered in Google Console for web credentials.
OAUTH_REDIRECT_PORT = 8085


# ── Auth helper ───────────────────────────────────────────────────────────────

class GoogleCalendarClient:
    """
    Authenticated Google Calendar service — shared singleton across all tools.
    Handles both "installed" (desktop) and "web" credential types.
    """

    _service = None

    @classmethod
    def get_service(cls):
        if cls._service is not None:
            return cls._service

        from google.oauth2.credentials import Credentials
        from google.auth.transport.requests import Request
        from googleapiclient.discovery import build

        credentials_path = Path(os.getenv("GOOGLE_CREDENTIALS_PATH", "credentials.json"))
        token_path = Path(os.getenv("GOOGLE_TOKEN_PATH", "token.json"))

        if not credentials_path.exists():
            raise FileNotFoundError(
                f"credentials.json not found at '{credentials_path}'.\n"
                "Download it from Google Cloud Console → APIs & Services → Credentials."
            )

        # Detect credential type
        raw = json.loads(credentials_path.read_text())
        cred_type = "web" if "web" in raw else "installed"
        log.info("google_cred_type_detected", type=cred_type)

        creds = None

        # Load cached token if it exists
        if token_path.exists():
            try:
                creds = Credentials.from_authorized_user_file(str(token_path), SCOPES)
            except Exception as exc:
                log.warning("token_load_failed", error=str(exc), action="re-authenticating")
                creds = None

        # Refresh expired token
        if creds and creds.expired and creds.refresh_token:
            try:
                creds.refresh(Request())
                token_path.write_text(creds.to_json())
                log.info("google_token_refreshed")
            except Exception as exc:
                log.warning("token_refresh_failed", error=str(exc), action="re-authenticating")
                creds = None

        # Run OAuth flow if no valid token
        if not creds or not creds.valid:
            creds = cls._run_oauth_flow(credentials_path, token_path, cred_type)

        cls._service = build("calendar", "v3", credentials=creds)
        log.info("google_calendar_service_ready")
        return cls._service

    @classmethod
    def _run_oauth_flow(
        cls,
        credentials_path: Path,
        token_path: Path,
        cred_type: str,
    ):
        """
        Run the appropriate OAuth flow based on credential type.

        "installed" (Desktop app): any localhost port works — no setup needed.
        "web": needs http://localhost:{OAUTH_REDIRECT_PORT}/ registered in Console.
        """
        from google_auth_oauthlib.flow import InstalledAppFlow

        log.info("google_oauth_flow_start", cred_type=cred_type)

        if cred_type == "web":
            # Web credentials require a specific registered redirect URI.
            # InstalledAppFlow supports this since google-auth-oauthlib 1.0+
            # by passing redirect_uri explicitly.
            log.info(
                "google_oauth_web_flow",
                port=OAUTH_REDIRECT_PORT,
                hint=f"Make sure http://localhost:{OAUTH_REDIRECT_PORT}/ is in "
                     "Authorised redirect URIs in Google Cloud Console",
            )
            flow = InstalledAppFlow.from_client_secrets_file(
                str(credentials_path),
                SCOPES,
                redirect_uri=f"http://localhost:{OAUTH_REDIRECT_PORT}/",
            )
            creds = flow.run_local_server(
                port=OAUTH_REDIRECT_PORT,
                prompt="consent",
                access_type="offline",
                open_browser=True,
            )
        else:
            # Desktop / installed credentials: simplest flow, no URI registration needed
            flow = InstalledAppFlow.from_client_secrets_file(str(credentials_path), SCOPES)
            creds = flow.run_local_server(port=0, prompt="consent")

        token_path.write_text(creds.to_json())
        log.info("google_oauth_complete", token_saved=str(token_path))
        return creds

    @classmethod
    async def get_service_async(cls):
        """Async wrapper — auth runs in thread pool to avoid blocking the event loop."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, cls.get_service)

    @classmethod
    def invalidate(cls):
        """Force re-authentication on next call (useful for testing)."""
        cls._service = None


# ── Shared formatter ──────────────────────────────────────────────────────────

def _format_event(event: dict) -> dict:
    """
    Clean up the raw Google API event dict into a compact structure
    that is easier for the LLM to read and reason about.
    """
    start = event.get("start", {})
    end = event.get("end", {})
    return {
        "id": event.get("id"),
        "title": event.get("summary", "(No title)"),
        "description": (event.get("description") or "")[:500],  # cap length
        "location": event.get("location", ""),
        "start": start.get("dateTime") or start.get("date"),
        "end": end.get("dateTime") or end.get("date"),
        "is_all_day": "date" in start,
        "status": event.get("status"),
        "attendees": [a.get("email") for a in event.get("attendees", [])],
        "meet_link": event.get("hangoutLink", ""),
        "html_link": event.get("htmlLink", ""),
    }


def _to_rfc3339(dt_string: str) -> str:
    """
    Convert ISO 8601 date-time string → RFC 3339 with timezone.
    Required format for the Google Calendar API.

    Examples that work:
      "2025-06-15T14:00:00"          → assumes UTC
      "2025-06-15T14:00:00+02:00"   → preserves offset
      "2025-06-15"                   → treated as all-day (don't pass to this fn)
    """
    try:
        dt = datetime.fromisoformat(dt_string)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.isoformat()
    except ValueError:
        raise ValueError(
            f"Invalid datetime '{dt_string}'. "
            "Use ISO 8601, e.g. '2025-06-15T14:00:00' or '2025-06-15T16:00:00+02:00'"
        )


# ── Tool 1: List events ───────────────────────────────────────────────────────

class ListCalendarEventsTool(BaseTool):

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="list_calendar_events",
            description=(
                "List events from the user's Google Calendar. "
                "Use to check what's scheduled, find free slots, or search for specific events. "
                "Returns event IDs, titles, start/end times, locations, and attendees. "
                "Defaults to the next 7 days. Use search_query to filter by keyword."
            ),
            parameters=ToolParameters(
                properties={
                    "max_results": ParameterProperty(
                        type="integer",
                        description="Max events to return (1–50). Default 10.",
                        default=10, minimum=1, maximum=50,
                    ),
                    "time_min": ParameterProperty(
                        type="string",
                        description="Start of range (ISO 8601). Default: now.",
                        default=None,
                    ),
                    "time_max": ParameterProperty(
                        type="string",
                        description="End of range (ISO 8601). Default: 7 days from now.",
                        default=None,
                    ),
                    "search_query": ParameterProperty(
                        type="string",
                        description="Keyword search across event titles and descriptions.",
                        default=None,
                    ),
                    "calendar_id": ParameterProperty(
                        type="string",
                        description="Calendar ID. Use 'primary' for the main calendar.",
                        default="primary",
                    ),
                },
                required=[],
            ),
        )

    async def _run(
        self,
        max_results: int = 10,
        time_min: str | None = None,
        time_max: str | None = None,
        search_query: str | None = None,
        calendar_id: str = "primary",
    ) -> dict:
        service = await GoogleCalendarClient.get_service_async()
        now = datetime.now(timezone.utc)
        t_min = _to_rfc3339(time_min) if time_min else now.isoformat()
        t_max = _to_rfc3339(time_max) if time_max else (now + timedelta(days=7)).isoformat()

        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: service.events().list(
                calendarId=calendar_id,
                timeMin=t_min,
                timeMax=t_max,
                maxResults=max_results,
                singleEvents=True,
                orderBy="startTime",
                q=search_query,
            ).execute(),
        )

        events = [_format_event(e) for e in response.get("items", [])]
        log.info("calendar_list_ok", count=len(events))
        return {"events": events, "total": len(events), "range": {"from": t_min, "to": t_max}}


# ── Tool 2: Get event ─────────────────────────────────────────────────────────

class GetCalendarEventTool(BaseTool):

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="get_calendar_event",
            description=(
                "Fetch full details of one specific calendar event by its event ID. "
                "Use after list_calendar_events to get the complete description, "
                "all attendees, Google Meet link, and other extended fields."
            ),
            parameters=ToolParameters(
                properties={
                    "event_id": ParameterProperty(
                        type="string",
                        description="The event ID from a previous list_calendar_events result.",
                    ),
                    "calendar_id": ParameterProperty(
                        type="string",
                        description="Calendar ID containing the event.",
                        default="primary",
                    ),
                },
                required=["event_id"],
            ),
        )

    async def _run(self, event_id: str, calendar_id: str = "primary") -> dict:
        service = await GoogleCalendarClient.get_service_async()
        loop = asyncio.get_event_loop()
        event = await loop.run_in_executor(
            None,
            lambda: service.events().get(calendarId=calendar_id, eventId=event_id).execute(),
        )
        return _format_event(event)


# ── Tool 3: Create event ──────────────────────────────────────────────────────

class CreateCalendarEventTool(BaseTool):

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="create_calendar_event",
            description=(
                "Create a new event in the user's Google Calendar. "
                "Requires a title, start time, and end time. "
                "Times must be ISO 8601, e.g. '2025-06-15T14:00:00+00:00'. "
                "For all-day events use 'YYYY-MM-DD' format for start and end. "
                "Returns the new event's ID and calendar link."
            ),
            parameters=ToolParameters(
                properties={
                    "title": ParameterProperty(
                        type="string",
                        description="Event title.",
                    ),
                    "start_time": ParameterProperty(
                        type="string",
                        description="Start datetime (ISO 8601) or date (YYYY-MM-DD for all-day).",
                    ),
                    "end_time": ParameterProperty(
                        type="string",
                        description="End datetime (ISO 8601) or date (YYYY-MM-DD for all-day).",
                    ),
                    "description": ParameterProperty(
                        type="string",
                        description="Event notes or description.",
                        default="",
                    ),
                    "location": ParameterProperty(
                        type="string",
                        description="Physical address or video call URL.",
                        default="",
                    ),
                    "attendees": ParameterProperty(
                        type="array",
                        description="List of attendee email addresses.",
                        default=None,
                        items={"type": "string"},
                    ),
                    "add_google_meet": ParameterProperty(
                        type="boolean",
                        description="If true, attach a Google Meet link.",
                        default=False,
                    ),
                    "calendar_id": ParameterProperty(
                        type="string",
                        description="Target calendar ID.",
                        default="primary",
                    ),
                },
                required=["title", "start_time", "end_time"],
            ),
        )

    async def _run(
        self,
        title: str,
        start_time: str,
        end_time: str,
        description: str = "",
        location: str = "",
        attendees: list[str] | None = None,
        add_google_meet: bool = False,
        calendar_id: str = "primary",
    ) -> dict:
        service = await GoogleCalendarClient.get_service_async()

        is_all_day = len(start_time.strip()) == 10   # "YYYY-MM-DD"
        if is_all_day:
            time_field = lambda t: {"date": t.strip()}
        else:
            time_field = lambda t: {"dateTime": _to_rfc3339(t), "timeZone": "UTC"}

        body: dict[str, Any] = {
            "summary": title,
            "description": description,
            "location": location,
            "start": time_field(start_time),
            "end": time_field(end_time),
        }
        if attendees:
            body["attendees"] = [{"email": e} for e in attendees]
        if add_google_meet:
            import time as _time
            body["conferenceData"] = {
                "createRequest": {"requestId": f"meet-{int(_time.time())}"}
            }

        loop = asyncio.get_event_loop()
        insert_kwargs: dict[str, Any] = {
            "calendarId": calendar_id,
            "body": body,
            "sendUpdates": "all",
        }
        if add_google_meet:
            insert_kwargs["conferenceDataVersion"] = 1

        created = await loop.run_in_executor(
            None,
            lambda: service.events().insert(**insert_kwargs).execute(),
        )

        log.info("calendar_event_created", event_id=created.get("id"), title=title)
        return {
            "success": True,
            "event_id": created.get("id"),
            "title": created.get("summary"),
            "html_link": created.get("htmlLink"),
            "meet_link": created.get("hangoutLink", ""),
            "start": created.get("start"),
            "end": created.get("end"),
        }


# ── Tool 4: Update event ──────────────────────────────────────────────────────

class UpdateCalendarEventTool(BaseTool):

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="update_calendar_event",
            description=(
                "Modify an existing calendar event. Only the fields you provide are changed — "
                "omitted fields keep their current value. "
                "Requires an event_id from list_calendar_events or get_calendar_event. "
                "Use to reschedule (new times), rename (new title), or add details."
            ),
            parameters=ToolParameters(
                properties={
                    "event_id": ParameterProperty(
                        type="string",
                        description="ID of the event to update.",
                    ),
                    "title": ParameterProperty(
                        type="string",
                        description="New title. Omit to keep current.",
                        default=None,
                    ),
                    "start_time": ParameterProperty(
                        type="string",
                        description="New start datetime (ISO 8601). Omit to keep current.",
                        default=None,
                    ),
                    "end_time": ParameterProperty(
                        type="string",
                        description="New end datetime (ISO 8601). Omit to keep current.",
                        default=None,
                    ),
                    "description": ParameterProperty(
                        type="string",
                        description="New description. Omit to keep current.",
                        default=None,
                    ),
                    "location": ParameterProperty(
                        type="string",
                        description="New location. Omit to keep current.",
                        default=None,
                    ),
                    "calendar_id": ParameterProperty(
                        type="string",
                        description="Calendar ID containing the event.",
                        default="primary",
                    ),
                },
                required=["event_id"],
            ),
        )

    async def _run(
        self,
        event_id: str,
        title: str | None = None,
        start_time: str | None = None,
        end_time: str | None = None,
        description: str | None = None,
        location: str | None = None,
        calendar_id: str = "primary",
    ) -> dict:
        service = await GoogleCalendarClient.get_service_async()
        loop = asyncio.get_event_loop()

        patch: dict[str, Any] = {}
        if title is not None:
            patch["summary"] = title
        if description is not None:
            patch["description"] = description
        if location is not None:
            patch["location"] = location
        if start_time is not None:
            patch["start"] = {"dateTime": _to_rfc3339(start_time), "timeZone": "UTC"}
        if end_time is not None:
            patch["end"] = {"dateTime": _to_rfc3339(end_time), "timeZone": "UTC"}

        if not patch:
            return {"success": False, "message": "No fields provided to update."}

        updated = await loop.run_in_executor(
            None,
            lambda: service.events().patch(
                calendarId=calendar_id,
                eventId=event_id,
                body=patch,
                sendUpdates="all",
            ).execute(),
        )

        log.info("calendar_event_updated", event_id=event_id, fields=list(patch.keys()))
        return {
            "success": True,
            "event_id": updated.get("id"),
            "updated_fields": list(patch.keys()),
            "html_link": updated.get("htmlLink"),
        }


# ── Tool 5: Delete event ──────────────────────────────────────────────────────

class DeleteCalendarEventTool(BaseTool):

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="delete_calendar_event",
            description=(
                "Permanently delete a calendar event. This cannot be undone. "
                "Always confirm with the user before deleting. "
                "Use get_calendar_event first to verify you have the correct event. "
                "Requires the event_id."
            ),
            parameters=ToolParameters(
                properties={
                    "event_id": ParameterProperty(
                        type="string",
                        description="ID of the event to permanently delete.",
                    ),
                    "calendar_id": ParameterProperty(
                        type="string",
                        description="Calendar ID containing the event.",
                        default="primary",
                    ),
                },
                required=["event_id"],
            ),
        )

    async def _run(self, event_id: str, calendar_id: str = "primary") -> dict:
        service = await GoogleCalendarClient.get_service_async()
        loop = asyncio.get_event_loop()

        await loop.run_in_executor(
            None,
            lambda: service.events().delete(
                calendarId=calendar_id,
                eventId=event_id,
                sendUpdates="all",
            ).execute(),
        )

        log.info("calendar_event_deleted", event_id=event_id)
        return {"success": True, "deleted_event_id": event_id}