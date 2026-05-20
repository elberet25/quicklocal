"""
Unit tests for tools/calendar_tool.py

All tests mock the Calendar API — no real credentials or network calls needed.

Run from the project root:
    pytest tests/test_calendar_tool.py -v
"""

import json
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest
from tools.calendar_tool import (
    CalendarBaseTool,
    GetScheduleTool,
    FindFreeTimeTool,
    CreateEventTool,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _local_tz():
    return datetime.now(timezone.utc).astimezone().tzinfo


def _make_event(summary="Meeting", start_dt="2024-01-15T10:00:00",
                end_dt="2024-01-15T11:00:00", event_id="evt001",
                description="", location="", all_day=False):
    """Build a minimal Calendar API event dict."""
    if all_day:
        date_str = start_dt[:10]
        return {
            "id": event_id,
            "summary": summary,
            "start": {"date": date_str},
            "end": {"date": date_str},
            "description": description,
            "location": location,
        }
    return {
        "id": event_id,
        "summary": summary,
        "start": {"dateTime": start_dt},
        "end": {"dateTime": end_dt},
        "description": description,
        "location": location,
    }


def _make_service(events=None):
    """Return a mocked Calendar API service."""
    service = MagicMock()
    service.events().list().execute.return_value = {"items": events or []}
    return service


# ---------------------------------------------------------------------------
# CalendarBaseTool._day_bounds
# ---------------------------------------------------------------------------

class TestDayBounds:
    def test_returns_two_strings(self):
        start, end = CalendarBaseTool._day_bounds("2024-01-15")
        assert isinstance(start, str)
        assert isinstance(end, str)

    def test_start_is_midnight(self):
        start, _ = CalendarBaseTool._day_bounds("2024-01-15")
        dt = datetime.fromisoformat(start)
        assert dt.hour == 0 and dt.minute == 0 and dt.second == 0

    def test_end_is_next_day_midnight(self):
        _, end = CalendarBaseTool._day_bounds("2024-01-15")
        dt = datetime.fromisoformat(end)
        assert dt.day == 16

    def test_span_is_exactly_24_hours(self):
        start, end = CalendarBaseTool._day_bounds("2024-01-15")
        delta = datetime.fromisoformat(end) - datetime.fromisoformat(start)
        assert delta.total_seconds() == 86400


# ---------------------------------------------------------------------------
# CalendarBaseTool._parse_event
# ---------------------------------------------------------------------------

class TestParseEvent:
    def test_extracts_summary(self):
        result = CalendarBaseTool._parse_event(_make_event(summary="Standup"))
        assert result["summary"] == "Standup"

    def test_extracts_datetime_start_end(self):
        result = CalendarBaseTool._parse_event(
            _make_event(start_dt="2024-01-15T09:00:00", end_dt="2024-01-15T09:30:00")
        )
        assert result["start"] == "2024-01-15T09:00:00"
        assert result["end"] == "2024-01-15T09:30:00"

    def test_all_day_uses_date_field(self):
        result = CalendarBaseTool._parse_event(_make_event(start_dt="2024-01-15", all_day=True))
        assert result["start"] == "2024-01-15"
        assert result["all_day"] is True

    def test_timed_event_all_day_is_false(self):
        result = CalendarBaseTool._parse_event(_make_event())
        assert result["all_day"] is False

    def test_missing_summary_defaults(self):
        event = _make_event()
        del event["summary"]
        result = CalendarBaseTool._parse_event(event)
        assert result["summary"] == "(no title)"

    def test_extracts_description_and_location(self):
        result = CalendarBaseTool._parse_event(
            _make_event(description="Notes here", location="Room 42")
        )
        assert result["description"] == "Notes here"
        assert result["location"] == "Room 42"

    def test_extracts_id(self):
        result = CalendarBaseTool._parse_event(_make_event(event_id="abc123"))
        assert result["id"] == "abc123"


# ---------------------------------------------------------------------------
# CalendarBaseTool._list_events
# ---------------------------------------------------------------------------

class TestListEvents:
    def test_returns_parsed_events(self):
        service = _make_service([_make_event(summary="Sync", event_id="e1")])
        result = CalendarBaseTool._list_events(service, "2024-01-15")
        assert len(result) == 1
        assert result[0]["summary"] == "Sync"

    def test_returns_empty_list_when_no_events(self):
        service = _make_service([])
        result = CalendarBaseTool._list_events(service, "2024-01-15")
        assert result == []

    def test_calls_api_with_correct_params(self):
        service = _make_service()
        CalendarBaseTool._list_events(service, "2024-01-15")
        call_kwargs = service.events().list.call_args.kwargs
        assert call_kwargs["calendarId"] == "primary"
        assert call_kwargs["singleEvents"] is True
        assert call_kwargs["orderBy"] == "startTime"

    def test_time_bounds_match_date(self):
        service = _make_service()
        CalendarBaseTool._list_events(service, "2024-06-10")
        call_kwargs = service.events().list.call_args.kwargs
        assert "2024-06-10" in call_kwargs["timeMin"]
        assert "2024-06-11" in call_kwargs["timeMax"]


# ---------------------------------------------------------------------------
# GetScheduleTool
# ---------------------------------------------------------------------------

class TestGetScheduleTool:
    def setup_method(self):
        self.tool = GetScheduleTool()

    def test_returns_json_list(self):
        service = _make_service([_make_event(summary="Review")])
        with patch.object(CalendarBaseTool, "_get_calendar_service", return_value=service):
            result = self.tool.execute(date="2024-01-15")
        data = json.loads(result["result"])
        assert isinstance(data, list)
        assert data[0]["summary"] == "Review"

    def test_empty_day_returns_empty_list(self):
        service = _make_service([])
        with patch.object(CalendarBaseTool, "_get_calendar_service", return_value=service):
            result = self.tool.execute(date="2024-01-15")
        assert json.loads(result["result"]) == []

    def test_validate_input_accepts_valid_date(self):
        assert self.tool.validate_input(date="2024-01-15") is True

    def test_validate_input_rejects_invalid_date(self):
        assert self.tool.validate_input(date="not-a-date") is False
        assert self.tool.validate_input(date="") is False
        assert self.tool.validate_input(date="15-01-2024") is False

    def test_error_returned_on_api_failure(self):
        with patch.object(CalendarBaseTool, "_get_calendar_service", side_effect=Exception("API down")):
            result = self.tool.execute(date="2024-01-15")
        assert "error" in result
        assert "API down" in result["error"]


# ---------------------------------------------------------------------------
# FindFreeTimeTool
# ---------------------------------------------------------------------------

class TestFindFreeTimeTool:
    def setup_method(self):
        self.tool = FindFreeTimeTool()
        self.tz = _local_tz()

    def _dt(self, hour, minute=0):
        """Return a local-tz ISO string for 2024-01-15 at the given time."""
        return datetime(2024, 1, 15, hour, minute, tzinfo=self.tz).isoformat()

    def test_full_day_free_when_no_events(self):
        service = _make_service([])
        with patch.object(CalendarBaseTool, "_get_calendar_service", return_value=service):
            result = self.tool.execute(date="2024-01-15")
        slots = json.loads(result["result"])
        assert len(slots) == 1
        assert slots[0]["duration_minutes"] == 9 * 60  # 09:00–18:00

    def test_event_splits_day_into_two_slots(self):
        event = _make_event(start_dt=self._dt(11), end_dt=self._dt(12))
        service = _make_service([event])
        with patch.object(CalendarBaseTool, "_get_calendar_service", return_value=service):
            result = self.tool.execute(date="2024-01-15")
        slots = json.loads(result["result"])
        assert len(slots) == 2
        assert slots[0]["duration_minutes"] == 120  # 09:00–11:00
        assert slots[1]["duration_minutes"] == 360  # 12:00–18:00

    def test_all_day_event_does_not_block_slots(self):
        event = _make_event(start_dt="2024-01-15", all_day=True)
        service = _make_service([event])
        with patch.object(CalendarBaseTool, "_get_calendar_service", return_value=service):
            result = self.tool.execute(date="2024-01-15")
        slots = json.loads(result["result"])
        assert len(slots) == 1
        assert slots[0]["duration_minutes"] == 9 * 60

    def test_gap_under_30_minutes_excluded(self):
        # 09:00–10:00 and 10:20–18:00 leaves a 20-min gap
        e1 = _make_event(start_dt=self._dt(9), end_dt=self._dt(10), event_id="e1")
        e2 = _make_event(start_dt=self._dt(10, 20), end_dt=self._dt(18), event_id="e2")
        service = _make_service([e1, e2])
        with patch.object(CalendarBaseTool, "_get_calendar_service", return_value=service):
            result = self.tool.execute(date="2024-01-15")
        slots = json.loads(result["result"])
        assert slots == []

    def test_custom_working_hours(self):
        service = _make_service([])
        with patch.object(CalendarBaseTool, "_get_calendar_service", return_value=service):
            result = self.tool.execute(date="2024-01-15", start_hour=8, end_hour=16)
        slots = json.loads(result["result"])
        assert slots[0]["duration_minutes"] == 8 * 60

    def test_event_outside_window_ignored(self):
        # Event is 19:00–20:00, outside 09:00–18:00
        event = _make_event(start_dt=self._dt(19), end_dt=self._dt(20))
        service = _make_service([event])
        with patch.object(CalendarBaseTool, "_get_calendar_service", return_value=service):
            result = self.tool.execute(date="2024-01-15")
        slots = json.loads(result["result"])
        assert len(slots) == 1
        assert slots[0]["duration_minutes"] == 9 * 60

    def test_validate_input_accepts_valid_date(self):
        assert self.tool.validate_input(date="2024-01-15") is True

    def test_validate_input_rejects_invalid_date(self):
        assert self.tool.validate_input(date="bad") is False

    def test_error_returned_on_api_failure(self):
        with patch.object(CalendarBaseTool, "_get_calendar_service", side_effect=Exception("Auth error")):
            result = self.tool.execute(date="2024-01-15")
        assert "error" in result


# ---------------------------------------------------------------------------
# CreateEventTool
# ---------------------------------------------------------------------------

class TestCreateEventTool:
    def setup_method(self):
        self.tool = CreateEventTool()

    def _make_service(self, event_id="evt999", html_link="https://calendar.google.com/evt"):
        service = MagicMock()
        service.events().insert().execute.return_value = {
            "id": event_id,
            "summary": "Team Sync",
            "start": {"dateTime": "2024-01-15T14:00:00+00:00"},
            "end": {"dateTime": "2024-01-15T15:00:00+00:00"},
            "htmlLink": html_link,
        }
        return service

    def test_returns_json_with_status_created(self):
        service = self._make_service()
        with patch.object(CalendarBaseTool, "_get_calendar_service", return_value=service):
            result = self.tool.execute(
                summary="Team Sync",
                start="2024-01-15T14:00:00",
                end="2024-01-15T15:00:00",
            )
        data = json.loads(result["result"])
        assert data["status"] == "Event created"

    def test_returns_event_id_and_link(self):
        service = self._make_service(event_id="evt42", html_link="https://cal.example.com")
        with patch.object(CalendarBaseTool, "_get_calendar_service", return_value=service):
            result = self.tool.execute(
                summary="Sync",
                start="2024-01-15T14:00:00",
                end="2024-01-15T15:00:00",
            )
        data = json.loads(result["result"])
        assert data["id"] == "evt42"
        assert data["link"] == "https://cal.example.com"

    def test_calls_insert_with_correct_calendar(self):
        service = self._make_service()
        with patch.object(CalendarBaseTool, "_get_calendar_service", return_value=service):
            self.tool.execute(
                summary="Sync",
                start="2024-01-15T14:00:00",
                end="2024-01-15T15:00:00",
            )
        call_kwargs = service.events().insert.call_args.kwargs
        assert call_kwargs["calendarId"] == "primary"

    def test_event_body_contains_summary_start_end(self):
        service = self._make_service()
        with patch.object(CalendarBaseTool, "_get_calendar_service", return_value=service):
            self.tool.execute(
                summary="My Event",
                start="2024-01-15T14:00:00",
                end="2024-01-15T15:00:00",
            )
        body = service.events().insert.call_args.kwargs["body"]
        assert body["summary"] == "My Event"
        assert "dateTime" in body["start"]
        assert "dateTime" in body["end"]

    def test_optional_description_included_when_provided(self):
        service = self._make_service()
        with patch.object(CalendarBaseTool, "_get_calendar_service", return_value=service):
            self.tool.execute(
                summary="Sync",
                start="2024-01-15T14:00:00",
                end="2024-01-15T15:00:00",
                description="Bring slides",
            )
        body = service.events().insert.call_args.kwargs["body"]
        assert body["description"] == "Bring slides"

    def test_optional_description_absent_when_not_provided(self):
        service = self._make_service()
        with patch.object(CalendarBaseTool, "_get_calendar_service", return_value=service):
            self.tool.execute(
                summary="Sync",
                start="2024-01-15T14:00:00",
                end="2024-01-15T15:00:00",
            )
        body = service.events().insert.call_args.kwargs["body"]
        assert "description" not in body

    def test_optional_location_included_when_provided(self):
        service = self._make_service()
        with patch.object(CalendarBaseTool, "_get_calendar_service", return_value=service):
            self.tool.execute(
                summary="Sync",
                start="2024-01-15T14:00:00",
                end="2024-01-15T15:00:00",
                location="Conference Room A",
            )
        body = service.events().insert.call_args.kwargs["body"]
        assert body["location"] == "Conference Room A"

    def test_validate_input_accepts_valid_inputs(self):
        assert self.tool.validate_input(
            summary="Sync",
            start="2024-01-15T14:00:00",
            end="2024-01-15T15:00:00",
        ) is True

    def test_validate_input_rejects_missing_summary(self):
        assert self.tool.validate_input(
            summary="",
            start="2024-01-15T14:00:00",
            end="2024-01-15T15:00:00",
        ) is False

    def test_validate_input_rejects_invalid_datetime(self):
        assert self.tool.validate_input(
            summary="Sync",
            start="not-a-date",
            end="2024-01-15T15:00:00",
        ) is False

    def test_error_returned_on_api_failure(self):
        with patch.object(CalendarBaseTool, "_get_calendar_service", side_effect=Exception("Quota exceeded")):
            result = self.tool.execute(
                summary="Sync",
                start="2024-01-15T14:00:00",
                end="2024-01-15T15:00:00",
            )
        assert "error" in result
        assert "Quota exceeded" in result["error"]
