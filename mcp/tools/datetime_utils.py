"""
Datetime normalization for tool arguments. (stdlib only, no extra llm calls)

Since llms produce natural language dates instead of ISO 8601:-
    The model is given the current UTC time in its tool-calling prompt, so it
    *should* produce ISO 8601. This normalizer is the safety net for when it
    doesn't, handles every realistic llm date output format.

Coverage:
  Point tokens     : now, today, tomorrow, yesterday, tonight
  Week tokens      : this week, next week, end of week
  Month tokens     : this month, next month, end of month
  Weekday names    : this Friday, next Sunday, last Monday
  Weekday + time   : this Friday at 7pm, Monday at 14:30
  Day + month      : Friday 20th March, March 20th (infer year)
  Relative         : in 3 days, 2 weeks ago, in 1 month
  Time-only        : 2200, 1800hr, 7pm, 19:00, 7:30 pm (-> today at that time)
  Named month      : March 16 2026, 16th March 2026, March 2026
  ISO variants     : 2026-03-16, 2026-03-16T14:00, full ISO 8601
  Locale formats   : 03/16/2026, 16.03.2026 
"""

from __future__ import annotations

import re
import calendar
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

# weekday & month lookup
_WEEKDAY = {
    "monday": 0, "mon": 0,
    "tuesday": 1, "tue": 1, "tues": 1,
    "wednesday": 2, "wed": 2,
    "thursday": 3, "thu": 3, "thur": 3, "thurs": 3,
    "friday": 4, "fri": 4,
    "saturday": 5, "sat": 5,
    "sunday": 6, "sun": 6,
}
_MONTH_NAMES = {
    "january": 1, "jan": 1, "february": 2, "feb": 2,
    "march": 3, "mar": 3, "april": 4, "apr": 4,
    "may": 5, "june": 6, "jun": 6, "july": 7, "jul": 7,
    "august": 8, "aug": 8, "september": 9, "sep": 9, "sept": 9,
    "october": 10, "oct": 10, "november": 11, "nov": 11,
    "december": 12, "dec": 12,
}

def normalize_dt(value: str, ref: datetime | None = None) -> str:
    """
    Converts any datetime-like string to UTC ISO 8601.
    
    Raises:
        ValueError if the input cannot be parsed.
    Returns:
        "YYYY-MM-DDTHH:MM:SS+00:00"

    """
    if not value or not value.strip():
        raise ValueError("Empty datetime value")

    now = ref or datetime.now(timezone.utc)
    v = value.strip().lower()
    v_clean = re.sub(r"\b(\d+)(st|nd|rd|th)\b", r"\1", v) # strip ordinals

    # 1. ISO 8601 first, cheapest check, most common LLM output
    iso = _try_iso(value.strip())
    if iso:
        return _fmt(iso)

    # 2. Point tokens
    if v_clean in ("now", "current", "currently", "right now"):
        return _fmt(now)
    if v_clean in ("today", "start of today"):
        return _fmt(_sod(now))
    if v_clean in ("tonight", "end of today"):
        return _fmt(_sod(now).replace(hour=23, minute=59, second=59))
    if v_clean in ("tomorrow", "start of tomorrow"):
        return _fmt(_sod(now) + timedelta(days=1))
    if v_clean == "end of tomorrow":
        return _fmt(_sod(now) + timedelta(days=2) - timedelta(seconds=1))
    if v_clean in ("yesterday", "start of yesterday"):
        return _fmt(_sod(now) - timedelta(days=1))

    # 3. Week tokens
    mon = _sod(now) - timedelta(days=now.weekday()) # this Monday (start of week)
    if v_clean in ("this week", "start of week", "start of this week"):
        return _fmt(mon)
    if v_clean in ("end of week", "end of this week", "this weekend"):
        return _fmt(mon + timedelta(days=6, hours=23, minutes=59, seconds=59))
    if v_clean in ("next week", "start of next week"):
        return _fmt(mon + timedelta(weeks=1))
    if v_clean == "end of next week":
        return _fmt(mon + timedelta(weeks=1, days=6, hours=23, minutes=59, seconds=59))

    # 4. Month tokens
    if v_clean in ("this month", "start of month", "start of this month"):
        return _fmt(now.replace(day=1, hour=0, minute=0, second=0, microsecond=0))
    if v_clean in ("end of month", "end of this month"):
        return _fmt(_end_of_month(now))
    if v_clean in ("next month", "start of next month"):
        return _fmt(_start_of_next_month(now))
    if v_clean == "end of next month":
        return _fmt(_end_of_month(_start_of_next_month(now)))

    # 5. Time-only strings (apply to today, or tomorrow if past),
    # patterns like: "2200", "1800hr", "1800hrs", "18:00", "7pm", "7:30pm", "7:30 pm"
    time_only = _try_time_only(v_clean, now)
    if time_only:
        return _fmt(time_only)

    # 6. Relative: "in N unit" / "N unit ago" / "N unit from now"
    m = re.match(r"^in\s+(\d+)\s+(day|days|week|weeks|month|months)$", v_clean)
    if m:
        return _fmt(_add_unit(_sod(now), int(m.group(1)), m.group(2)))

    m = re.match(r"^(\d+)\s+(day|days|week|weeks|month|months)\s+ago$", v_clean)
    if m:
        return _fmt(_add_unit(_sod(now), -int(m.group(1)), m.group(2)))

    m = re.match(r"^(\d+)\s+(day|days|week|weeks)\s+from\s+now$", v_clean)
    if m:
        return _fmt(_add_unit(now, int(m.group(1)), m.group(2)))

    # 7. Weekday + optional time,
    # handles: "this Friday", "next Sunday", "last Monday",
    #          "this Friday at 7pm", "Monday at 14:30", "friday 7pm"
    wd_result = _try_weekday(v_clean, now)
    if wd_result:
        return _fmt(wd_result)

    # 8. Weekday + day + month  (no year -> infer)
    # "Friday 20th March", "Friday, March 20", "20 March Friday"
    wd_date = _try_weekday_date(v_clean, now)
    if wd_date:
        return _fmt(wd_date)

    # 9. Named month dates,
    # like "March 16 2026", "16 March 2026", "March 16" (infer year)
    named = _try_named_month(v_clean, now)
    if named:
        return _fmt(named)

    # 10. Locale formats
    for fmt in ("%m/%d/%Y", "%m/%d/%y", "%d.%m.%Y", "%d/%m/%Y"):
        try:
            dt = datetime.strptime(value.strip(), fmt).replace(tzinfo=timezone.utc)
            return _fmt(dt)
        except ValueError:
            pass

    raise ValueError(
        f"Unrecognised datetime: '{value}'. "
        "Expected ISO 8601 (2026-03-20T19:00:00+00:00), "
        "a weekday (this Friday at 7pm), or natural language (tomorrow, next week)."
    )


@dataclass
class RangeResult:
    time_min: str
    time_max: str
    was_corrected: bool = False
    correction_reason: str = ""


def validate_range(
    time_min: str | None,
    time_max: str | None,
    default_days: int = 7,
    ref: datetime | None = None,
) -> RangeResult:
    """
    Normalises and validates a time range. Ensures time_max > time_min.
    Corrects inverted ranges (coz Google API returns SSL error on min > max).
    """
    now = ref or datetime.now(timezone.utc)
    notes: list[str] = []

    def _resolve(val: str | None, fallback: datetime) -> datetime:
        if not val:
            return fallback
        try:
            return _parse_utc(normalize_dt(val, now))
        except ValueError as exc:
            notes.append(str(exc))
            return fallback

    t_min = _resolve(time_min, now)
    t_max = _resolve(time_max, t_min + timedelta(days=default_days))

    corrected = False
    if t_max <= t_min:
        notes.append(
            f"time_max ({t_max.isoformat()[:19]}) ≤ time_min, "
            f"set to min + {default_days}d."
        )
        t_max = t_min + timedelta(days=default_days)
        corrected = True

    return RangeResult(
        time_min=_fmt(t_min),
        time_max=_fmt(t_max),
        was_corrected=corrected or bool(notes),
        correction_reason=" | ".join(notes),
    )


# private parsers

def _try_iso(value: str) -> datetime | None:
    """Tries direct fromisoformat (handles full ISO and YYYY-MM-DD)."""
    try:
        dt = datetime.fromisoformat(value)
        return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt
    except ValueError:
        return None

def _try_time_only(v: str, now: datetime) -> datetime | None:
    """
    Parses time-only strings and attach to today (or tomorrow if already past).

    Handles:
      2200, 22:00, 22h, 22hr, 22hrs
      1800, 18:00, 1800hr, 1800hrs, 1800h
      7pm, 7:30pm, 7 pm, 7:30 pm
      19:00, 19, 7:00, 07:30
    """
    hour = minute = None

    # military / 24h compact: "2200", "1800"
    m = re.match(r"^(\d{3,4})(?:h|hr|hrs)?$", v)
    if m:
        t = m.group(1).zfill(4)
        hour, minute = int(t[:2]), int(t[2:])

    # "22:00", "7:30", "19:00"
    if hour is None:
        m = re.match(r"^(\d{1,2}):(\d{2})(?:h|hr|hrs)?$", v)
        if m:
            hour, minute = int(m.group(1)), int(m.group(2))

    # "7pm", "7:30pm", "7 pm", "7:30 pm"
    if hour is None:
        m = re.match(r"^(\d{1,2})(?::(\d{2}))?\s*(am|pm)$", v)
        if m:
            hour = int(m.group(1))
            minute = int(m.group(2)) if m.group(2) else 0
            if m.group(3) == "pm" and hour != 12:
                hour += 12
            elif m.group(3) == "am" and hour == 12:
                hour = 0

    if hour is None or not (0 <= hour <= 23) or not (0 <= minute <= 59):
        return None

    target = _sod(now).replace(hour=hour, minute=minute, second=0, microsecond=0)
    # if that time today is already in the past, we use tomorrow
    if target <= now:
        target += timedelta(days=1)
    return target


def _try_weekday(v: str, now: datetime) -> datetime | None:
    """
    Parses weekday references with optional time component.

    "this Friday"           -> coming Friday of current week
    "next Sunday"           -> Sunday of next week
    "last Monday"           -> previous Monday
    "friday"                -> soonest upcoming Friday (including today)
    "this Friday at 7pm"    -> coming Friday at 19:00
    "Monday at 14:30"       -> coming Monday at 14:30
    "friday 7pm"            -> coming Friday at 19:00
    """
    # try to extract: [this|next|last] <weekday> [at <time>] or <weekday> [at] <time>
    m = re.match(
        r"^(this\s+|next\s+|last\s+)?([a-z]+)(?:\s+at\s+|\s+)?([\d:apm ]+)?$", v
    )
    if not m:
        return None

    modifier = (m.group(1) or "").strip()
    day_name = m.group(2).strip()
    time_str = (m.group(3) or "").strip()

    target_wd = _WEEKDAY.get(day_name)
    if target_wd is None:
        return None

    today_wd = now.weekday()

    if modifier == "last":
        days_back = (today_wd - target_wd) % 7 or 7
        base = _sod(now) - timedelta(days=days_back)
    elif modifier == "next":
        days_fwd = (target_wd - today_wd) % 7 or 7
        base = _sod(now) + timedelta(days=days_fwd)
    elif modifier == "this":
        # "this Friday" = the Friday of the current calendar week,
        # if today IS Friday, "this Friday" = today
        days_fwd = (target_wd - today_wd) % 7
        base = _sod(now) + timedelta(days=days_fwd)
    else:
        # bare weekday: "friday" -> soonest upcoming (including today)
        days_fwd = (target_wd - today_wd) % 7
        base = _sod(now) + timedelta(days=days_fwd)

    # apply time component if present
    if time_str:
        t = _try_time_only(time_str.strip(), base) # base used as ref -> no single-day rollover
        if t is None:
            mt = re.match(r"^(\d{1,2}):(\d{2})$", time_str.strip())
            if mt:
                t = base.replace(hour=int(mt.group(1)), minute=int(mt.group(2)))
        if t:
            # for bare weekday (no this/next/last), if computed datetime is in the
            # real past, advance by 1 week. "Monday at 2pm" at 3pm on Monday = next Monday.
            if not modifier and t <= now:
                t += timedelta(weeks=1)
            return t

    return base


def _try_weekday_date(v: str, now: datetime) -> datetime | None:
    """
    Parses weekday + day + month (no year -> infer closest future date).

    "Friday 20th March", "Friday, March 20", "20 March", "March 20"
    """
    v = re.sub(r"\b(\d+)(st|nd|rd|th)\b", r"\1", v)
    v = v.replace(",", " ")

    # "Day_name? DD Month" or "Day_name? Month DD",
    # we ignore the day_name (Friday) and use the explicit date
    patterns = [
        r"(?:[a-z]+\s+)?(\d{1,2})\s+([a-z]+)$", # "Friday 20 March" / "20 March"
        r"(?:[a-z]+\s+)?([a-z]+)\s+(\d{1,2})$", # "Friday March 20" / "March 20"
    ]
    for pat in patterns:
        m = re.match(pat, v.strip())
        if not m:
            continue
        a, b = m.group(1), m.group(2)
        # to figure out which is day number and which is month name
        if a.isdigit():
            day_num, month_str = int(a), b
        elif b.isdigit():
            day_num, month_str = int(b), a
        else:
            continue
        month = _MONTH_NAMES.get(month_str.lower())
        if not month:
            continue
        # inferring year: use current year, bump to next if date is in the past
        for year_offset in (0, 1):
            year = now.year + year_offset
            try:
                dt = datetime(year, month, day_num, tzinfo=timezone.utc)
                if dt >= _sod(now):
                    return dt
            except ValueError:
                pass
    return None


def _try_named_month(v: str, now: datetime) -> datetime | None:
    """
    Parses "March 16 2026", "March 16", "March 2026".
    Infers the year if absent.
    """
    v = re.sub(r"\b(\d+)(st|nd|rd|th)\b", r"\1", v)

    # "Month DD YYYY"
    m = re.match(r"([a-z]+)\s+(\d{1,2})[,\s]+(\d{4})", v)
    if m:
        month = _MONTH_NAMES.get(m.group(1))
        if month:
            try:
                return datetime(int(m.group(3)), month, int(m.group(2)),
                                tzinfo=timezone.utc)
            except ValueError:
                pass

    # "DD Month YYYY"
    m = re.match(r"(\d{1,2})\s+([a-z]+)\s+(\d{4})", v)
    if m:
        month = _MONTH_NAMES.get(m.group(2))
        if month:
            try:
                return datetime(int(m.group(3)), month, int(m.group(1)),
                                tzinfo=timezone.utc)
            except ValueError:
                pass

    # "Month DD" (no year -> infer)
    m = re.match(r"([a-z]+)\s+(\d{1,2})$", v)
    if m:
        month = _MONTH_NAMES.get(m.group(1))
        if month:
            for year_offset in (0, 1):
                year = now.year + year_offset
                try:
                    dt = datetime(year, month, int(m.group(2)), tzinfo=timezone.utc)
                    if dt >= _sod(now):
                        return dt
                except ValueError:
                    pass

    # "Month YYYY" -> first of month
    m = re.match(r"([a-z]+)\s+(\d{4})$", v)
    if m:
        month = _MONTH_NAMES.get(m.group(1))
        if month:
            try:
                return datetime(int(m.group(2)), month, 1, tzinfo=timezone.utc)
            except ValueError:
                pass

    return None


# shared helpers

def _fmt(dt: datetime) -> str:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S+00:00")

def _sod(dt: datetime) -> datetime:
    """Start of day, UTC midnight."""
    return dt.replace(hour=0, minute=0, second=0, microsecond=0, tzinfo=timezone.utc)

def _parse_utc(iso: str) -> datetime:
    return datetime.fromisoformat(iso).astimezone(timezone.utc)

def _add_unit(base: datetime, n: int, unit: str) -> datetime:
    if "month" in unit:
        month = base.month + n
        year = base.year + (month - 1) // 12
        month = ((month - 1) % 12) + 1
        max_day = calendar.monthrange(year, month)[1]
        return base.replace(year=year, month=month, day=min(base.day, max_day),
                            hour=0, minute=0, second=0, microsecond=0)
    if "week" in unit:
        return base + timedelta(weeks=n)
    return base + timedelta(days=n)

def _start_of_next_month(dt: datetime) -> datetime:
    if dt.month == 12:
        return dt.replace(year=dt.year + 1, month=1, day=1,
                          hour=0, minute=0, second=0, microsecond=0)
    return dt.replace(month=dt.month + 1, day=1,
                      hour=0, minute=0, second=0, microsecond=0)

def _end_of_month(dt: datetime) -> datetime:
    return _start_of_next_month(dt) - timedelta(seconds=1)