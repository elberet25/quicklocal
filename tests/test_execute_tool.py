"""
Unit tests for execute_tool() dispatcher behavior: validation, retry, confirmation gate.

Run from the project root:
    pytest tests/test_execute_tool.py -v
"""

from unittest.mock import patch

import pytest
from src.agent import execute_tool, tool_registry, MAX_TOOL_RETRIES


class TestValidateInputPreFlight:
    """validate_input() runs before confirmation and before execute()."""

    def test_invalid_input_returns_error(self):
        # CreateEventTool.validate_input rejects bad datetimes
        result = execute_tool("create_event", {
            "summary": "Sync",
            "start": "not-a-date",
            "end": "2026-05-23T15:00:00",
        })
        assert "Error" in result
        assert "invalid input" in result.lower()

    def test_invalid_input_does_not_call_execute(self, monkeypatch):
        monkeypatch.setattr("builtins.input", lambda _: "y")
        called = []
        with patch.object(tool_registry["create_event"], "execute",
                          side_effect=lambda **kw: called.append(True) or {"result": "ok"}):
            execute_tool("create_event", {
                "summary": "",
                "start": "2026-05-23T14:00:00",
                "end": "2026-05-23T15:00:00",
            })
        assert called == []

    def test_valid_input_passes_through(self, monkeypatch):
        monkeypatch.setattr("builtins.input", lambda _: "y")
        with patch.object(tool_registry["create_event"], "execute",
                          return_value={"result": "event created"}):
            result = execute_tool("create_event", {
                "summary": "Team Sync",
                "start": "2026-05-23T14:00:00",
                "end": "2026-05-23T15:00:00",
            })
        assert result == "event created"


class TestRetryBehavior:
    """Verify the retry loop checks result["retryable"], not exceptions."""

    def test_succeeds_on_second_attempt(self, monkeypatch):
        monkeypatch.setattr("time.sleep", lambda s: None)
        attempts = [0]

        def flaky(**kwargs):
            attempts[0] += 1
            if attempts[0] == 1:
                return {"error": "Rate limit hit", "retryable": True}
            return {"result": "success"}

        with patch.object(tool_registry["get_current_time"], "execute", side_effect=flaky):
            result = execute_tool("get_current_time", {})

        assert result == "success"
        assert attempts[0] == 2

    def test_does_not_retry_non_retryable_error(self, monkeypatch):
        monkeypatch.setattr("time.sleep", lambda s: None)
        attempts = [0]

        def always_fails(**kwargs):
            attempts[0] += 1
            return {"error": "Auth failed", "retryable": False}

        with patch.object(tool_registry["get_current_time"], "execute", side_effect=always_fails):
            result = execute_tool("get_current_time", {})

        assert "Auth failed" in result
        assert attempts[0] == 1

    def test_gives_up_after_max_retries(self, monkeypatch):
        monkeypatch.setattr("time.sleep", lambda s: None)
        attempts = [0]

        def always_flaky(**kwargs):
            attempts[0] += 1
            return {"error": "Flaky server", "retryable": True}

        with patch.object(tool_registry["get_current_time"], "execute", side_effect=always_flaky):
            result = execute_tool("get_current_time", {})

        assert "Flaky server" in result
        assert attempts[0] == MAX_TOOL_RETRIES

    def test_no_sleep_on_success(self, monkeypatch):
        slept = []
        monkeypatch.setattr("time.sleep", lambda s: slept.append(s))
        execute_tool("get_current_time", {})
        assert slept == []

    def test_no_retry_when_no_retryable_key(self, monkeypatch):
        monkeypatch.setattr("time.sleep", lambda s: None)
        attempts = [0]

        def normal_error(**kwargs):
            attempts[0] += 1
            return {"error": "Something broke"}  # no retryable key

        with patch.object(tool_registry["get_current_time"], "execute", side_effect=normal_error):
            execute_tool("get_current_time", {})

        assert attempts[0] == 1
