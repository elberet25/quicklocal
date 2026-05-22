"""
Google Calendar API integration with OAuth2.

Shares credentials.json with Gmail but uses a separate token file
(calendar_token.json) to avoid scope conflicts.

Required files (not committed to git):
  credentials.json       — downloaded from Google Cloud Console (OAuth 2.0 Client ID)
  calendar_token.json    — created automatically after first authorization
"""

import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

try:
    from tools.base_tool import BaseTool
except ImportError:
    from base_tool import BaseTool


class CalendarBaseTool(BaseTool):
    """Shared auth and helpers for all Calendar tools."""

    category = "calendar"
    summarizable = True
    _SCOPES = [
        "https://www.googleapis.com/auth/calendar.readonly",
        "https://www.googleapis.com/auth/calendar.events",
    ]
    _CREDENTIALS_FILE = Path(os.getenv("GOOGLE_CREDENTIALS_PATH", "credentials.json"))
    _TOKEN_FILE = Path(os.getenv("CALENDAR_TOKEN_FILE", "calendar_token.json"))

    @classmethod
    def _get_calendar_service(cls):
        """Return an authorized Calendar API service client."""
        creds = None

        if cls._TOKEN_FILE.exists():
            creds = Credentials.from_authorized_user_file(str(cls._TOKEN_FILE), cls._SCOPES)

        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                if not cls._CREDENTIALS_FILE.exists():
                    raise FileNotFoundError(
                        f"credentials.json not found at '{cls._CREDENTIALS_FILE}'. "
                        "Download it from Google Cloud Console → APIs & Services → Credentials."
                    )
                flow = InstalledAppFlow.from_client_secrets_file(
                    str(cls._CREDENTIALS_FILE), cls._SCOPES
                )
                creds = flow.run_local_server(port=0)

            cls._TOKEN_FILE.write_text(creds.to_json())

        return build("calendar", "v3", credentials=creds)

    @staticmethod
    def _day_bounds(date_str: str) -> tuple[str, str]:
        """Return RFC 3339 start and end of a calendar day in local time."""
        dt = datetime.strptime(date_str, "%Y-%m-%d")
        local_tz = datetime.now(timezone.utc).astimezone().tzinfo
        start = datetime(dt.year, dt.month, dt.day, 0, 0, 0, tzinfo=local_tz)
        end = start + timedelta(days=1)
        return start.isoformat(), end.isoformat()

    @staticmethod
    def _parse_event(event: dict) -> dict:
        """Extract the key fields from a Calendar API event resource."""
        start = event.get("start", {})
        end = event.get("end", {})
        return {
            "id": event.get("id", ""),
            "summary": event.get("summary", "(no title)"),
            "start": start.get("dateTime") or start.get("date", ""),
            "end": end.get("dateTime") or end.get("date", ""),
            "description": event.get("description", ""),
            "location": event.get("location", ""),
            "all_day": "date" in start and "dateTime" not in start,
        }

    @classmethod
    def _list_events(cls, service, date_str: str) -> list[dict]:
        """Return all events on the given date, sorted by start time."""
        time_min, time_max = cls._day_bounds(date_str)
        result = service.events().list(
            calendarId="primary",
            timeMin=time_min,
            timeMax=time_max,
            singleEvents=True,
            orderBy="startTime",
        ).execute()
        return [cls._parse_event(e) for e in result.get("items", [])]


class GetScheduleTool(CalendarBaseTool):
    name = "get_schedule"

    def get_description(self) -> dict:
        return {
            "name": self.name,
            "description": (
                "Returns all calendar events for a specific date. "
                "Use when the user asks what they have scheduled, their agenda, "
                "or what's on their calendar for a particular day."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "date": {
                        "type": "string",
                        "description": "Date in YYYY-MM-DD format (e.g. '2024-01-15').",
                    },
                },
                "required": ["date"],
            },
        }

    def validate_input(self, **kwargs) -> bool:
        try:
            datetime.strptime(kwargs.get("date", ""), "%Y-%m-%d")
            return True
        except ValueError:
            return False

    def execute(self, **kwargs) -> dict:
        try:
            service = self._get_calendar_service()
            events = self._list_events(service, kwargs["date"])
            return {"result": json.dumps(events, ensure_ascii=False, indent=2)}
        except Exception as e:
            return self.handle_error(e)


class FindFreeTimeTool(CalendarBaseTool):
    name = "find_free_time"

    def get_description(self) -> dict:
        return {
            "name": self.name,
            "description": (
                "Finds free time slots on a specific date by checking gaps between "
                "existing calendar events. Only considers slots within working hours "
                "(default 09:00–18:00). Returns slots of at least 30 minutes."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "date": {
                        "type": "string",
                        "description": "Date in YYYY-MM-DD format (e.g. '2024-01-15').",
                    },
                    "start_hour": {
                        "type": "integer",
                        "description": "Start of working window (24h, default 9).",
                        "default": 9,
                    },
                    "end_hour": {
                        "type": "integer",
                        "description": "End of working window (24h, default 18).",
                        "default": 18,
                    },
                },
                "required": ["date"],
            },
        }

    def validate_input(self, **kwargs) -> bool:
        try:
            datetime.strptime(kwargs.get("date", ""), "%Y-%m-%d")
            return True
        except ValueError:
            return False

    def execute(self, **kwargs) -> dict:
        try:
            date_str = kwargs["date"]
            start_hour = kwargs.get("start_hour", 9)
            end_hour = kwargs.get("end_hour", 18)

            local_tz = datetime.now(timezone.utc).astimezone().tzinfo
            dt = datetime.strptime(date_str, "%Y-%m-%d")
            window_start = datetime(dt.year, dt.month, dt.day, start_hour, 0, tzinfo=local_tz)
            window_end = datetime(dt.year, dt.month, dt.day, end_hour, 0, tzinfo=local_tz)

            service = self._get_calendar_service()
            events = self._list_events(service, date_str)

            # Build list of busy intervals within the working window
            busy: list[tuple[datetime, datetime]] = []
            for ev in events:
                if ev["all_day"]:
                    continue
                ev_start = datetime.fromisoformat(ev["start"])
                ev_end = datetime.fromisoformat(ev["end"])
                # Clamp to working window
                b_start = max(ev_start, window_start)
                b_end = min(ev_end, window_end)
                if b_start < b_end:
                    busy.append((b_start, b_end))

            busy.sort(key=lambda x: x[0])

            # Find gaps
            free_slots = []
            cursor = window_start
            for b_start, b_end in busy:
                if b_start > cursor:
                    gap_minutes = int((b_start - cursor).total_seconds() / 60)
                    if gap_minutes >= 30:
                        free_slots.append({
                            "start": cursor.isoformat(),
                            "end": b_start.isoformat(),
                            "duration_minutes": gap_minutes,
                        })
                cursor = max(cursor, b_end)

            # Gap after last event
            if cursor < window_end:
                gap_minutes = int((window_end - cursor).total_seconds() / 60)
                if gap_minutes >= 30:
                    free_slots.append({
                        "start": cursor.isoformat(),
                        "end": window_end.isoformat(),
                        "duration_minutes": gap_minutes,
                    })

            return {"result": json.dumps(free_slots, ensure_ascii=False, indent=2)}
        except Exception as e:
            return self.handle_error(e)


class CreateEventTool(CalendarBaseTool):
    name = "create_event"
    requires_confirmation = True

    def get_confirmation_message(self, **kwargs) -> str:
        """Show event details before creating."""
        summary = kwargs.get("summary", "(untitled)")
        start = kwargs.get("start", "?")
        end = kwargs.get("end", "?")
        return f"Create calendar event: '{summary}' from {start} to {end}."

    def get_description(self) -> dict:
        return {
            "name": self.name,
            "description": (
                "Creates a new event on the user's primary Google Calendar. "
                "Use when the user asks to add, schedule, or book something on their calendar."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "summary": {
                        "type": "string",
                        "description": "Event title.",
                    },
                    "start": {
                        "type": "string",
                        "description": "Start datetime in ISO 8601 format, e.g. '2024-01-15T14:00:00'.",
                    },
                    "end": {
                        "type": "string",
                        "description": "End datetime in ISO 8601 format, e.g. '2024-01-15T15:00:00'.",
                    },
                    "description": {
                        "type": "string",
                        "description": "Optional event description or notes.",
                    },
                    "location": {
                        "type": "string",
                        "description": "Optional location string.",
                    },
                },
                "required": ["summary", "start", "end"],
            },
        }

    def validate_input(self, **kwargs) -> bool:
        try:
            datetime.fromisoformat(kwargs.get("start", ""))
            datetime.fromisoformat(kwargs.get("end", ""))
            return bool(kwargs.get("summary"))
        except ValueError:
            return False

    def execute(self, **kwargs) -> dict:
        try:
            local_tz = datetime.now(timezone.utc).astimezone().tzinfo

            def _to_rfc3339(dt_str: str) -> str:
                dt = datetime.fromisoformat(dt_str)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=local_tz)
                return dt.isoformat()

            event_body = {
                "summary": kwargs["summary"],
                "start": {"dateTime": _to_rfc3339(kwargs["start"])},
                "end": {"dateTime": _to_rfc3339(kwargs["end"])},
            }
            if kwargs.get("description"):
                event_body["description"] = kwargs["description"]
            if kwargs.get("location"):
                event_body["location"] = kwargs["location"]

            service = self._get_calendar_service()
            created = service.events().insert(calendarId="primary", body=event_body).execute()

            return {"result": json.dumps({
                "status": "Event created",
                "id": created["id"],
                "summary": created.get("summary"),
                "start": created["start"].get("dateTime"),
                "end": created["end"].get("dateTime"),
                "link": created.get("htmlLink", ""),
            }, indent=2)}
        except Exception as e:
            return self.handle_error(e)
