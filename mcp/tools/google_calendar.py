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
import re
import json
import time as _time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from mcp.schemas import ParameterProperty, ToolDefinition, ToolParameters
from mcp.tools.base import BaseTool
from mcp.tools.datetime_utils import _parse_utc, normalize_dt, validate_range
from utils.logger import get_logger

log = get_logger(__name__)

SCOPES = ["https://www.googleapis.com/auth/calendar"]
# Port for the local OAuth callback server.
# Must match the redirect URI registered in Google Console for web credentials.
OAUTH_REDIRECT_PORT = 8085

# a Google Calendar event_id(s): 26 lowercase alphanumeric chars (base32-ish),
    # anything containing spaces, mixed case words, or >60 chars is almost certainly
    # a title or hallucination
_REAL_EVENT_ID_RE = re.compile(r"^[a-z0-9_]{10,80}$")


# shared auth client

class GoogleCalendarClient:
    """
    Thin wrapper around the Google Calendar API service object.
    Handles credential loading, OAuth flow, and token refresh,
    both "installed" (desktop) and "web" credential types.

    Authenticated Google Calendar service - shared (cached) singleton across all tools.
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
                log.warning("token_load_failed", error=str(exc))#, action="re-authenticating")
                creds = None

        # if no valid credentials -> run OAuth flow or refresh expired token
        if creds and creds.expired and creds.refresh_token:
            try:
                creds.refresh(Request())
                token_path.write_text(creds.to_json()) # save for next run
                log.info("google_token_refreshed")
            except Exception as exc:
                log.warning("token_refresh_failed", error=str(exc))#, action="re-authenticating")
                creds = None

        # run OAuth flow if no valid token
        if not creds or not creds.valid:
            creds = cls._run_oauth_flow(credentials_path, token_path, cred_type)

        # 'cache_discovery=False' suppresses the noisy "file_cache is only supported
        # with oauth2client<4.0.0" warning - cosmetic issue, not a real error but annoying nonetheless
        cls._service = build("calendar", "v3", credentials=creds, cache_discovery=False)
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
    Normalises a raw Google Calendar API event dict into a clean structure
    that's easier for the LLM to read and reason about.
    """
    start = event.get("start", {})
    end = event.get("end", {})
    return {
        "id": event.get("id"),
        "title": event.get("summary", "(No title)"),
        "description": event.get("description") or "",#)[:400],
        "location": event.get("location", ""),
        "start": start.get("dateTime") or start.get("date"),
        "end": end.get("dateTime") or end.get("date"),
        "is_all_day": "date" in start,
        "status": event.get("status"),
        "attendees": [a.get("email") for a in event.get("attendees", [])],
        "meet_link": event.get("hangoutLink", ""),
        "html_link": event.get("htmlLink", ""),
    }

def _looks_like_real_id(event_id: str | None) -> bool:
    """
    Returns True if event_id appears to be a real Google Calendar event_id.
    
    Criteria:
        Real IDs: lowercase alphanumeric, no spaces, 10-80 chars.
        Fake IDs: contain spaces, mixed case words, or are obviously titles.
    """
    if not event_id:
        return False
    return bool(_REAL_EVENT_ID_RE.match(event_id.strip()))

async def _resolve_event_id(
    service,
    calendar_id: str,
    event_id: str | None,
    fallback_title: str | None = None,
) -> str:
    """
    Ensures we have a valid Google Calendar event_id before making a mutating call.

    Check order (zero extra LLM calls):
      1. If event_id looks real -> return it as-is (happy path, no API call)
      2. If event_id looks like a title (contains spaces, mixed words) ->
         search calendar for it, return first match's ID
      3. If event_id is None/empty but fallback_title is provided ->
         search by title, return first match's ID
      4. If nothing found -> raise ValueError with helpful message

    The search uses a 30-day window (past 7 days -> future 23 days) to catch
    both past and upcoming events the user might be referring to.
    """
    # case 1: looks real -> use directly
    if _looks_like_real_id(event_id):
        return event_id.strip()

    # determine search query
    search_query = None
    if event_id and event_id.strip():
        # event_id is present but looks fake - treat it as a title search
        search_query = event_id.strip()
        log.info(
            "event_id_resolve_by_search",
            fake_id=event_id[:60],
            reason="ID contains spaces or looks like a title",
        )
    elif fallback_title and fallback_title.strip():
        search_query = fallback_title.strip()
        log.info(
            "event_id_resolve_from_title",
            title=fallback_title[:60],
            reason="event_id was absent, searching by title",
        )

    if not search_query:
        raise ValueError(
            "event_id is required. Call list_calendar_events first to get the real ID, "
            "then pass it to this tool."
        )

    # search the calendar
    now = datetime.now(timezone.utc)
    time_min = (now - timedelta(days=7)).strftime("%Y-%m-%dT%H:%M:%S+00:00")
    time_max = (now + timedelta(days=23)).strftime("%Y-%m-%dT%H:%M:%S+00:00")

    loop = asyncio.get_event_loop()
    results = await loop.run_in_executor(
        None,
        lambda: service.events().list(
            calendarId=calendar_id,
            q=search_query,
            timeMin=time_min,
            timeMax=time_max,
            maxResults=5,
            singleEvents=True,
            orderBy="startTime",
        ).execute(),
    )

    items = results.get("items", [])
    if not items:
        # broaden search: try the whole past year
        results_broad = await loop.run_in_executor(
            None,
            lambda: service.events().list(
                calendarId=calendar_id,
                q=search_query,
                maxResults=5,
                singleEvents=True,
            ).execute(),
        )
        items = results_broad.get("items", [])

    if not items:
        raise ValueError(
            f"No event found matching '{search_query}'. "
            "Please call list_calendar_events first to find the correct event, "
            "then retry with its ID."
        )

    resolved_id = items[0]["id"]
    resolved_title = items[0].get("summary", "")
    log.info(
        "event_id_resolved",
        query=search_query[:60],
        resolved_id=resolved_id,
        resolved_title=resolved_title,
        candidates=len(items),
    )
    return resolved_id

def _is_virtual_meeting(title: str, location: str) -> bool:
    """
    True only if the event is clearly virtual/online.

    Uses word-boundary matching for short ambiguous keywords (meet, teams, remote, online,
    virtual) to avoid false positives like "Meeting at the office" matching "meet".
    Longer multi-word keywords (google meet, video call) use substring matching.
    """
    combined = f"{title} {location}".lower()

    # Multi-word / long keywords: substring match is safe (specific enough)
    _SUBSTR = {
        "zoom", "webinar", "video call", "video chat", "conference call",
        "google meet", "microsoft teams", "skype", "video meeting",
        "virtual meeting", "online meeting",
    }
    for kw in _SUBSTR:
        if kw in combined:
            return True

    # short keywords: require word boundaries to avoid "meeting" matching "meet"
    _WORD = {"meet", "teams", "remote", "virtual", "online"}
    for kw in _WORD:
        if re.search(r"\b" + re.escape(kw) + r"\b", combined):
            return True

    return False

def _filter_attendees(attendees: list | None) -> tuple[list[str], list]:
    """
    Returns:
        (valid_emails, invalid_items).
    Valid = contains @ and a dot after @.
    """
    if not attendees:
        return [], []
    valid = [e for e in attendees if isinstance(e, str) and "@" in e and "." in e.split("@")[-1]]
    invalid = [e for e in attendees if e not in valid]
    return valid, invalid


# tool#1 lists events

class ListCalendarEventsTool(BaseTool):
    """List upcoming calendar events with optional time range and keyword filters."""

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="list_calendar_events",
            description=(
                # "List upcoming events from the user's Google Calendar. "
                # "Use this to check what's scheduled, find free time, or look for specific events. "
                # "Returns event titles, times, locations, and IDs. "
                # "Use 'search_query' to find events by keyword. "
                # "Default to the next 8 days if no time range is given."
                # 
                # "List events from the user's Google Calendar within a time range. "
                # "Use to check schedule, find free slots, or search events. "
                # "time_min and time_max accept ISO 8601 OR natural language: "
                # "today, tomorrow, next week, in 3 days, March 16 2026, etc. "
                # "Leave both empty to get the next 7 days. "
                # "For 'next month' queries: set time_min=start of next month, "
                # "time_max=end of next month."
                "List events from the user's Google Calendar. "
                "Use to check schedule, find free slots, search events, or get event_id(s) "
                "before calling update_calendar_event or delete_calendar_event. "
                "time_min and time_max accept ISO 8601 datetimes - compute them from "
                "the current UTC time provided in your instructions. "
                "Defaults to the next 7 days if no range is given."
            ),
            parameters=ToolParameters(
                properties={
                    "max_results": ParameterProperty(
                        type="integer",
                        description="Maximum number of events to return (1-50). Default 10.",
                        default=10, minimum=1, maximum=50,
                    ),
                    "time_min": ParameterProperty(
                        type="string",
                        description="Range start (ISO 8601). Ex. '2026-02-02T00:00:00+00:00', defaults to now.",
                        default=None,
                        # format="date-time",
                    ),
                    "time_max": ParameterProperty(
                        type="string",
                        description="Range end (ISO 8601). Ex. '2026-10-08T23:59:59+00:00', MUST be after time_min, default: time_min + 7 days.",
                        default=None,
                        # format="date-time",
                    ),
                    "search_query": ParameterProperty(
                        type="string",
                        description="Keyword search across titles/descriptions. Omit to list all.",
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
        now = datetime.now(timezone.utc)
        # normalise and validate time range,
        # validate_range handles: None inputs, inverted ranges, natural language,
        # past time_max while time_min is in future, identical min==max.
        rng = validate_range(time_min, time_max, default_days=7, ref=now)
        # t_min = _to_rfc3339(time_min) if time_min else now.isoformat()
        # t_max = _to_rfc3339(time_max) if time_max else (now + timedelta(days=8)).isoformat()

        if rng.was_corrected:
            log.warning(
                "calendar_range_corrected",
                reason=rng.correction_reason,
                # original_min=time_min,
                # original_max=time_max,
                # resolved_min=rng.time_min,
                # resolved_max=rng.time_max,
                original=[time_min, time_max],
                resolved=[rng.time_min, rng.time_max],
            )

        # sanitise search_query - empty string behaves like None for Google API
        q = search_query.strip() if search_query and search_query.strip() else None
        calendar_id = calendar_id or "primary"

        service = await GoogleCalendarClient.get_service_async()
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: service.events().list(
                calendarId=calendar_id,
                timeMin=rng.time_min,   # t_min,
                timeMax=rng.time_max,   # t_max,
                maxResults=max_results,
                singleEvents=True,
                orderBy="startTime",
                q=q,
            ).execute()
        )

        events = [_format_event(e) for e in response.get("items", [])]
        log.info("calendar_list_ok", count=len(events), calendar=calendar_id)
        return {
            "events": events,
            "total": len(events),
            "range": {"from": rng.time_min, "to": rng.time_max}, # "range": {"from": t_min, "to": t_max}} # "time_range": [t_min, t_max]}
        }

# tool#2 fetches events

class GetCalendarEventTool(BaseTool):
    """Fetch a single calendar event by its ID."""

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="get_calendar_event",
            description=(
                "Fetch complete details of one calendar event by its real ID. "
                "Use after list_calendar_events to get full description, attendees, "
                "and Google Meet link. Do NOT call with a made-up or guessed ID."
            ),
            parameters=ToolParameters(
                properties={
                    "event_id": ParameterProperty(
                        type="string",
                        description="The event_id from a list_calendar_events result.",
                    ),
                    "calendar_id": ParameterProperty(
                        type="string",
                        description="Calendar ID of the containing event.",
                        default="primary",
                    ),
                },
                required=["event_id"],
            ),
        )

    async def _run(self, event_id: str, calendar_id: str = "primary") -> dict:
        calendar_id = calendar_id or "primary"
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
                # "Create a new event in the user's Google Calendar. "
                # "Requires a title, start time, and end time. "
                # "Times must be ISO 8601, ex. '2025-06-15T14:00:00+00:00'. "
                # "For all-day events use 'YYYY-MM-DD' format for start and end. "
                # "Returns the new event's ID and calendar link."
                #
                # "Create a new event in the user's Google Calendar. "
                # "start_time and end_time accept ISO 8601 or natural language. "
                # "For all-day events use 'YYYY-MM-DD' for both times. "
                # "attendees: only include if the user explicitly named email addresses. "
                # "add_google_meet: set True ONLY if the user explicitly asks for a "
                # "video call, virtual meeting, or Google Meet link. "
                # "NEVER set add_google_meet=True for physical location events. "
                # "Returns the new event_id and a link to open it."
                "Create a new event in the user's Google Calendar. "
                "start_time and end_time MUST be ISO 8601 - compute them from the current "
                "UTC time in your instructions. For all-day events use 'YYYY-MM-DD'. "
                "attendees: only include explicitly provided email addresses containing @. "
                "add_google_meet: True ONLY when user explicitly requests a video/online meeting. "
                "NEVER set add_google_meet=True for physical location events."
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
                        # format="date-time",
                    ),
                    "end_time": ParameterProperty(
                        type="string",
                        description=(
                            "End datetime (ISO 8601) or date (YYYY-MM-DD for all-day). "
                            #"For 1-hour events, simply add 1 hour to start_time. "
                            "Must be after start_time."
                        ),
                        # format="date-time",
                    ),
                    "description": ParameterProperty(
                        type="string",
                        description="Event description/notes. Omit if not provided by the user.",
                        default="",
                    ),
                    "location": ParameterProperty(
                        type="string",
                        description="Physical location or video call URL. Omit if not provided.",
                        default="",
                    ),
                    "attendees": ParameterProperty(
                        type="array",
                        description=(
                            "List of attendee email addresses. "
                            "Only include if the user explicitly provided email addresses. "
                            "Omit entirely if the user gave names without email addresses. "
                            "Do NOT include names without @-addresses."
                        ),
                        default=None,
                        items={"type": "string"}, # items={"type": "string", "format": "email"},
                    ),
                    "add_google_meet": ParameterProperty(
                        type="boolean",
                        description=(
                            "Adds a Google Meet link. "
                            "True ONLY when user says 'video call', 'Google Meet', 'online meeting'. "
                            "NEVER True for physical events."
                        ),
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
        calendar_id = calendar_id or "primary"
        now = datetime.now(timezone.utc)

        # safety: never add Meet link to physical-location events
        # over-rides model decision if location looks like a physical address
        if add_google_meet and location and not _is_virtual_meeting(title, location):
            log.info(
                "meet_link_suppressed",
                reason="Physical location detected..?",
                title=title,
                location=location,
            )
            add_google_meet = False

        # validate and filter attendees, Google API rejects attendees without a valid @ in the email
        # safe_attendees = None
        # if attendees:
        #     safe_attendees = [e for e in attendees if "@" in e and "." in e.split("@")[-1]]
        #     if len(safe_attendees) != len(attendees):
        #         invalid = set(attendees) - set(safe_attendees)
        #         log.warning("attendees_filtered", invalid=list(invalid))
        safe_attendees, invalid = _filter_attendees(attendees)
        if invalid:
            log.warning("attendees_filtered", invalid=invalid)
        

        # normalise times
        # is_all_day = (
        #     len(start_time.strip()) == 10
        #     and re.match(r"\d{4}-\d{2}-\d{2}$", start_time.strip())
        # )
        is_all_day = bool(re.match(r"^\d{4}-\d{2}-\d{2}$", start_time.strip()))  

        if is_all_day:
            start_spec: dict = {"date": start_time.strip()} # all-day events: date-only fields
            end_stripped = end_time.strip() # end_time for all-day must also be date-only
            # if len(end_stripped) > 10:
            if not re.match(r"^\d{4}-\d{2}-\d{2}$", end_stripped):
                # in case model passed a datetime for an all-day end: take date part only
                end_stripped = end_stripped[:10]
            end_spec: dict = {"date": end_stripped}
        else:
            # normalizing natural language / partial ISO
            start_norm = normalize_dt(start_time, now)
            end_norm = normalize_dt(end_time, now)
            t_start = _parse_utc(start_norm)
            t_end = _parse_utc(end_norm)
            if t_end <= t_start: # validate start < end
                # default to 1 hour if model gave bad range
                t_end = t_start + timedelta(hours=1)
                end_norm = t_end.strftime("%Y-%m-%dT%H:%M:%S+00:00")
                log.warning("create_end_fixed", original=end_time, fixed=end_norm)
            start_spec = {"dateTime": start_norm, "timeZone": "UTC"}
            end_spec = {"dateTime": end_norm, "timeZone": "UTC"} 

        # event request body:-
        body: dict[str, Any] = {
            "summary": title,
            "start": start_spec,
            "end": end_spec,
        }
        if description:
            body["description"] = description
        if location:
            body["location"] = location
        if safe_attendees:
            body["attendees"] = [{"email": e} for e in safe_attendees]
        if add_google_meet:
            body["conferenceData"] = {
                "createRequest": {"requestId": f"meet-{int(_time.time())}"}
            }

        service = await GoogleCalendarClient.get_service_async()
        loop = asyncio.get_event_loop()
        insert_kwargs: dict[str, Any] = {
            "calendarId": calendar_id, # required path param, never inside body
            "body": body,
            "sendUpdates": "all",
        }
        if add_google_meet:
            insert_kwargs["conferenceDataVersion"] = 1

        # capture insert_kwargs in closure properly (to avoid late-binding closure bug)
        kwargs_snapshot = dict(insert_kwargs)
        created = await loop.run_in_executor(
            None,
            lambda: service.events().insert(**kwargs_snapshot).execute(),
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
                # "Modify an existing calendar event. Only the fields you provide are changed - "
                # "omitted fields keep their current value. "
                # "Requires an event_id from list_calendar_events or get_calendar_event. "
                # "Use to reschedule (new times), rename (new title), or add details."
                #
                # "Modify an existing calendar event. Only provided fields are changed. "
                # "Requires event_id from list_calendar_events or get_calendar_event. "
                # "start_time and end_time accept ISO 8601 or natural language."
                "Modify an existing calendar event. Only provided fields are changed. "
                "event_id: the real alphanumeric ID from list_calendar_events. "
                "If you don't have the event_id yet, call list_calendar_events first. "
                "If you pass a title or description instead of a real ID, the tool will "
                "automatically search for the event by that text. "
                "start_time and end_time MUST be ISO 8601."
            ),
            parameters=ToolParameters(
                properties={
                    "event_id": ParameterProperty(
                        type="string",
                        description=(
                            "ID of the event to update. "
                            "Real Google Calendar event_id (26-char alphanumeric) from "
                            "list_calendar_events. If omitted, the tool searches by title."
                        ),
                        default=None # optional - resolved from title if absent
                    ),
                    "title": ParameterProperty(
                        type="string",
                        description="New title. Omit to keep current. If event_id is missing, this is used to look it up as an alternative.",
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
                # required=["event_id"],
                required=[], # keeping event_id optional - resolved automatically
            ),
        )

    async def _run(
        self,
        event_id: str | None = None,
        title: str | None = None,
        start_time: str | None = None,
        end_time: str | None = None,
        description: str | None = None,
        location: str | None = None,
        calendar_id: str = "primary",
    ) -> dict:
        calendar_id = calendar_id or "primary"
        now = datetime.now(timezone.utc)

        # resolving real event_id:
        # If event_id is absent or looks fake, search by title as fallback,
        # _resolve_event_id does one Google API list call only when needed.
        service = await GoogleCalendarClient.get_service_async()
        real_id = await _resolve_event_id(service, calendar_id, event_id, title)

        # buildin patch body
        patch: dict[str, Any] = {}
        if title is not None:
            patch["summary"] = title
        if description is not None:
            patch["description"] = description
        if location is not None:
            patch["location"] = location
        if start_time is not None:
            patch["start"] = {"dateTime": normalize_dt(start_time, now), "timeZone": "UTC"}
        if end_time is not None:
            patch["end"] = {"dateTime": normalize_dt(end_time, now), "timeZone": "UTC"}

        if not patch:
            return {"success": False, "message": "No fields to update were provided."}

        loop = asyncio.get_event_loop()
        kw = {
            "calendarId": calendar_id,
            "eventId": real_id,
            "body": patch,
            "sendUpdates": "all",
        }
        updated = await loop.run_in_executor(
            None,
            lambda: service.events().patch(**kw).execute(),
        )

        log.info("calendar_event_updated", event_id=real_id, fields=list(patch.keys()))
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
                "Use get_calendar_event first to verify you have the correct event (via event_id). "
                "event_id: real alphanumeric ID from list_calendar_events. " 
            ),
            parameters=ToolParameters(
                properties={
                    "event_id": ParameterProperty(
                        type="string",
                        description="ID of the event to permanently delete.",
                            # "If a title or description is passed instead, "
                            # "the tool will search for the matching event."
                    ),
                    "search_hint": ParameterProperty(
                        type="string",
                        description=(
                            "Event title or keyword to search by if you don't have the real ID. "
                            "Used only when event_id is missing or invalid."
                        ),
                        default=None,
                    ),
                    "calendar_id": ParameterProperty(
                        type="string",
                        description="Calendar ID containing the event.",
                        default="primary",
                    ),
                },
                required=[], # ["event_id"],
            ),
        )

    async def _run(
        self,
        event_id: str | None = None,
        search_hint: str | None = None,
        calendar_id: str = "primary",
    ) -> dict:
        calendar_id = calendar_id or "primary"
        service = await GoogleCalendarClient.get_service_async()
        
        # real event_id - uses event_id if real, searches by event_id
        # as a title if it looks fake, or searches by search_hint if event_id absent
        real_id = await _resolve_event_id(
            service, calendar_id, event_id, search_hint
        )

        loop = asyncio.get_event_loop()
        delete_kwargs = {
            "calendarId": calendar_id,
            "eventId": real_id,
            "sendUpdates": "all",
        }
        await loop.run_in_executor(
            None,
            lambda: service.events().delete(**delete_kwargs).execute(),
        )

        log.info("calendar_event_deleted", event_id=real_id)
        return {"success": True, "deleted_event_id": real_id}
