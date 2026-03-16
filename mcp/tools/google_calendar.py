"""
Google Calendar Tools - CRUD operations via the Google Calendar API v3.

Authentication:
  Uses OAuth2 with offline access (refresh token). On first run (first tool invocation), a browser
  window should open to authorise access. The token will be cached in 'token.json' and refreshed automatically on expiry.

  If credentials.json is type: "web" - fine, but Google requires
    the redirect URI to be pre-registered. On first run this file opens a
    local browser flow and listens on http://localhost:8085/callback
    SETUP for "web" token (one-time):-
        1. Go to console.cloud.google.com -> your project -> APIs & Services -> Credentials
        2. Click your OAuth 2.0 Client ID (the web type you already have)
        3. Under "Authorised redirect URIs" add:  http://localhost:8085/
        4. Save. Wait ~5 minutes for Google to propagate.
        5. Re-download credentials.json and replace your current file.
        6. Start the backend - a browser tab will open for consent.
        7. token.json is created automatically. Future runs skip the browser.

  Setup steps for "Desktop App" token (one-time):-
    1. Go to https://console.cloud.google.com
    2. Create a project -> Enable "Google Calendar API"
    3. Create OAuth2 credentials (Desktop app type)
    4. Download as 'credentials.json' -> place in backend/ root
    5. Run the backend once - browser opens for consent
    6. 'token.json' is created -> subsequent runs are automatic

  Required env var: GOOGLE_CREDENTIALS_PATH (default: credentials.json)
  Token cache path: GOOGLE_TOKEN_PATH (default: token.json)

Tools provided (five BaseTool subclasses):-
  list_calendar_events   - query events with time range + keyword filter
  get_calendar_event     - fetch one event by ID
  create_calendar_event  - create an event with all major fields support
  update_calendar_event  - modify any fields of an existing event (only changed fields)
  delete_calendar_event  - permanently remove an event
"""

from __future__ import annotations

import asyncio
import os
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from mcp.schemas import (
    ParameterProperty,
    ToolDefinition,
    ToolParameters,
)
from mcp.tools.base import BaseTool
from utils.logger import get_logger

log = get_logger(__name__)

SCOPES = ["https://www.googleapis.com/auth/calendar"]
# Port for the local OAuth callback server.
# Must match the redirect URI registered in Google Console for web credentials.
OAUTH_REDIRECT_PORT = 8085

# shared auth helper

class GoogleCalendarClient:
    """
    Thin wrapper around the Google Calendar API service object.
    Handles credential loading, OAuth flow, and token refresh,
    both "installed" (desktop) and "web" credential types.

    Authenticated Google Calendar service - shared singleton across all tools.
    """

    _service = None # cached across tool instances

    @classmethod
    def get_service(cls):
        """Return an authenticated Google Calendar service (cached)."""
        if cls._service is not None:
            return cls._service

        from google.oauth2.credentials import Credentials 
        from google.auth.transport.requests import Request
        from googleapiclient.discovery import build

        credentials_path = Path(os.getenv("GOOGLE_CREDENTIALS_PATH", "./credentials.json"))
        token_path = Path(os.getenv("GOOGLE_TOKEN_PATH", "./token.json"))


        if not credentials_path.exists():
            raise FileNotFoundError(
                f"credentials.json not found at '{credentials_path}'.\n"
                "Download it from Google Cloud Console -> APIs & Services -> Credentials."
            )

        # detect credential type
        raw = json.loads(credentials_path.read_text())
        cred_type = "web" if "web" in raw else "installed"
        log.info("google_cred_type_detected", type=cred_type)

        creds = None

        # load cached token if it exists
        if token_path.exists():
            try:
                creds = Credentials.from_authorized_user_file(str(token_path), SCOPES)
            except Exception as exc:
                log.warning("token_load_failed", error=str(exc), action="re-authenticating")
                creds = None

        # if no valid credentials -> run OAuth flow or refresh expired token
        if creds and creds.expired and creds.refresh_token:
            try:
                creds.refresh(Request())
                token_path.write_text(creds.to_json()) # save for next run
                log.info("google_token_refreshed")
            except Exception as exc:
                log.warning("token_refresh_failed", error=str(exc), action="re-authenticating")
                creds = None

        # run OAuth flow if no valid token
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
        Runs the appropriate OAuth flow based on credential type.

        "installed" (Desktop app): any localhost port works - no setup needed.
        "web": needs http://localhost:{OAUTH_REDIRECT_PORT}/ registered in Console.
        """
        from google_auth_oauthlib.flow import InstalledAppFlow

        log.info("google_oauth_flow_start", cred_type=cred_type)

        if cred_type == "web":
            # Web credentials require a specific registered redirect URI,
            # InstalledAppFlow supports this since google-auth-oauthlib 1.0+ by passing redirect_uri explicitly
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
        """Async wrapper - runs the blocking auth in a thread pool."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, cls.get_service)

    @classmethod
    def invalidate(cls):
        """Forces re-authentication on next call, for testing."""
        cls._service = None

# shared formatter

def _format_event(event: dict) -> dict:
    """
    Normalise a raw Google Calendar API event dict into a clean structure
    that's easier for the LLM to read and reason about.
    """
    start = event.get("start", {})
    end = event.get("end", {})
    return {
        "id": event.get("id"),
        "title": event.get("summary", "(No title)"),
        "description": event.get("description", ""), # (event.get("description") or "")[:500], # cap length
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
    Convert various date/time string formats to RFC 3339 (required by Google API).
    Accepts: ISO 8601, 'YYYY-MM-DD HH:MM', 'tomorrow 3pm' is NOT supported
    (we'll deal w/ natural language parsing later - using explicit dates for now).
    """
    try:
        # Try parsing with timezone
        dt = datetime.fromisoformat(dt_string)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.isoformat()
    except ValueError:
        raise ValueError(
            f"Invalid datetime: '{dt_string}'. "
            "Use ISO 8601 format, ex. '2025-06-15T14:00:00' or '2025-06-15T14:00:00+02:00'"
        )


# tool#1 lists events

class ListCalendarEventsTool(BaseTool):
    """List upcoming calendar events with optional time range and keyword filters."""

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="list_calendar_events",
            description=(
                "List upcoming events from the user's Google Calendar. "
                "Use this to check what's scheduled, find free time, or look for specific events. "
                "Returns event titles, times, locations, and IDs. "
                "Use 'search_query' to find events by keyword. "
                "Defaults to the next 8 days if no time range is given."
            ),
            parameters=ToolParameters(
                properties={
                    "max_results": ParameterProperty(
                        type="integer",
                        description="Maximum number of events to return (1-50).",
                        default=10, minimum=1, maximum=50,
                    ),
                    "time_min": ParameterProperty(
                        type="string",
                        description="Start of time range (ISO 8601). Defaults to now.",
                        default=None,
                        # format="date-time",
                    ),
                    "time_max": ParameterProperty(
                        type="string",
                        description="End of time range (ISO 8601). Defaults to 8 days from now.",
                        default=None,
                        # format="date-time",
                    ),
                    "search_query": ParameterProperty(
                        type="string",
                        description="Free-text search across event titles and descriptions.",
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
        t_max = _to_rfc3339(time_max) if time_max else (now + timedelta(days=8)).isoformat()

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
            ).execute()
        )

        events = [_format_event(e) for e in response.get("items", [])]
        log.info("calendar_list", count=len(events), calendar=calendar_id)
        return {"events": events, "total": len(events), "range": {"from": t_min, "to": t_max}} # "time_range": [t_min, t_max]}


# tool#2 fetches events

class GetCalendarEventTool(BaseTool):
    """Fetch a single calendar event by its ID."""

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="get_calendar_event",
            description=(
                "Fetch full details of a specific calendar event by its event ID. "
                "Use this after list_calendar_events to get complete information "
                "about a particular event (description, attendees, meet link, etc.)."
            ),
            parameters=ToolParameters(
                properties={
                    "event_id": ParameterProperty(
                        type="string",
                        description="The event ID from a previous list_calendar_events call.",
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
            lambda: service.events().get(calendarId=calendar_id, eventId=event_id).execute()
        )
        return _format_event(event)


# tool#3 creates events

class CreateCalendarEventTool(BaseTool):
    """Creates a new calendar event."""

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="create_calendar_event",
            description=(
                "Create a new event in the user's Google Calendar. "
                "Requires a title, start time, and end time. "
                "Times must be ISO 8601, ex. '2025-06-15T14:00:00+00:00'. "
                "For all-day events use 'YYYY-MM-DD' format for start and end. "
                "Returns the new event's ID and calendar link."
            ),
            parameters=ToolParameters(
                properties={
                    "title": ParameterProperty(
                        type="string",
                        description="Event title/summary.",
                    ),
                    "start_time": ParameterProperty(
                        type="string",
                        description="Start datetime (ISO 8601) or date (YYYY-MM-DD for all-day).",
                        # format="date-time",
                    ),
                    "end_time": ParameterProperty(
                        type="string",
                        description="End datetime (ISO 8601) or date (YYYY-MM-DD for all-day). "
                                    "For 1-hour events, add 1 hour to start_time.",
                        # format="date-time",
                    ),
                    "description": ParameterProperty(
                        type="string",
                        description="Detailed event description or notes.",
                        default="",
                    ),
                    "location": ParameterProperty(
                        type="string",
                        description="Physical location or video call URL.",
                        default="",
                    ),
                    "attendees": ParameterProperty(
                        type="array",
                        description="List of attendee email addresses.",
                        default=None,
                        items={"type": "string"}, # items={"type": "string", "format": "email"},
                    ),
                    "add_google_meet": ParameterProperty(
                        type="boolean",
                        description="If true, automatically create a Google Meet link.",
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

        # determine if all-day (date-only) or is a timed event
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
                "createRequest": {"requestId": f"meet-{int(_time.time())}"} # : f"meet-{datetime.now().timestamp():.0f}"}
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


# tool#4 update event

class UpdateCalendarEventTool(BaseTool):
    """Modify an existing calendar event."""

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="update_calendar_event",
            description=(
                "Modify an existing calendar event. Only the fields you provide are changed - "
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
                        description="New title (leave empty to keep current).",
                        default=None,
                    ),
                    "start_time": ParameterProperty(
                        type="string",
                        description="New start datetime (ISO 8601). Leave empty to keep current.",
                        default=None,
                    ),
                    "end_time": ParameterProperty(
                        type="string",
                        description="New end datetime (ISO 8601). Leave empty to keep current.",
                        default=None,
                    ),
                    "description": ParameterProperty(
                        type="string",
                        description="New description. Leave empty to keep current.",
                        default=None,
                    ),
                    "location": ParameterProperty(
                        type="string",
                        description="New location. Leave empty to keep current.",
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

        # # Fetch current event to merge changes
        # current = await loop.run_in_executor(
        #     None,
        #     lambda: service.events().get(calendarId=calendar_id, eventId=event_id).execute()
        # )

        patch: dict[str, Any] = {}
        if title:
            patch["summary"] = title
        if description is not None:
            patch["description"] = description
        if location is not None:
            patch["location"] = location
        if start_time:
            patch["start"] = {"dateTime": _to_rfc3339(start_time), "timeZone": "UTC"}
        if end_time:
            patch["end"] = {"dateTime": _to_rfc3339(end_time), "timeZone": "UTC"}

        if not patch:
            return {"success": False, "message": "No fields to update were provided."}

        updated = await loop.run_in_executor(
            None,
            lambda: service.events().patch(
                calendarId=calendar_id, eventId=event_id, body=patch, sendUpdates="all"
            ).execute()
        )

        log.info("calendar_event_updated", event_id=event_id, fields=list(patch.keys()))
        return {
            "success": True,
            "event_id": updated.get("id"),
            "updated_fields": list(patch.keys()),
            "html_link": updated.get("htmlLink"),
        }


# tool#5 deletes events

class DeleteCalendarEventTool(BaseTool):
    """Permanently delete a calendar event."""

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
                calendarId=calendar_id, eventId=event_id, sendUpdates="all"
            ).execute()
        )

        log.info("calendar_event_deleted", event_id=event_id)
        return {"success": True, "deleted_event_id": event_id}
