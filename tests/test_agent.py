"""
Unit tests for the agent and core tools.

Run from the project root:
    pytest tests/test_agent.py -v
"""

import re
from unittest.mock import patch

import pytest
from tools.time_tool import TimeTool
from tools.calculator_tool import CalculatorTool
from tools.calendar_tool import CreateEventTool
from tools.drive_tool import CreateDriveDocTool
from tools.notion_tool import CreateNotionPageTool
from src.agent import execute_tool, tool_registry


class TestTimeTool:
    def setup_method(self):
        self.tool = TimeTool()

    def test_returns_result_dict(self):
        result = self.tool.execute()
        assert "result" in result

    def test_not_empty(self):
        assert len(self.tool.execute()["result"]) > 0

    def test_expected_format(self):
        # Expected: "Monday, May 17 2026 — 03:45:22 PM"
        result = self.tool.execute()["result"]
        pattern = r"^\w+, \w+ \d{2} \d{4} — \d{2}:\d{2}:\d{2} (AM|PM)$"
        assert re.match(pattern, result), f"Unexpected format: {result!r}"


class TestCalculatorTool:
    def setup_method(self):
        self.tool = CalculatorTool()

    def test_add(self):
        assert self.tool.execute(operation="add", a=3, b=4)["result"] == "7"

    def test_subtract(self):
        assert self.tool.execute(operation="subtract", a=10, b=3)["result"] == "7"

    def test_multiply(self):
        assert self.tool.execute(operation="multiply", a=6, b=7)["result"] == "42"

    def test_divide(self):
        assert self.tool.execute(operation="divide", a=128, b=4)["result"] == "32.0"

    def test_divide_by_zero(self):
        result = self.tool.execute(operation="divide", a=5, b=0)["result"]
        assert "Error" in result
        assert "zero" in result

    def test_unknown_operation(self):
        result = self.tool.execute(operation="modulo", a=10, b=3)["result"]
        assert "Error" in result
        assert "modulo" in result

    def test_float_inputs(self):
        assert self.tool.execute(operation="add", a=1.5, b=2.5)["result"] == "4.0"

    def test_negative_numbers(self):
        assert self.tool.execute(operation="multiply", a=-3, b=4)["result"] == "-12"


class TestExecuteTool:
    def test_dispatches_get_current_time(self):
        result = execute_tool("get_current_time", {})
        assert isinstance(result, str)
        assert len(result) > 0

    def test_dispatches_calculate(self):
        result = execute_tool("calculate", {"operation": "add", "a": 2, "b": 3})
        assert result == "5"

    def test_unknown_tool_returns_error(self):
        result = execute_tool("nonexistent_tool", {})
        assert "Error" in result
        assert "nonexistent_tool" in result


class TestRequiresConfirmationAttributes:
    """Verify requires_confirmation flag and message content on write tools."""

    def test_create_event_requires_confirmation(self):
        assert CreateEventTool.requires_confirmation is True

    def test_create_drive_doc_requires_confirmation(self):
        assert CreateDriveDocTool.requires_confirmation is True

    def test_create_notion_page_requires_confirmation(self):
        assert CreateNotionPageTool.requires_confirmation is True

    def test_read_tools_do_not_require_confirmation(self):
        assert TimeTool.requires_confirmation is False

    def test_create_event_message_contains_summary_and_times(self):
        msg = CreateEventTool().get_confirmation_message(
            summary="Team Sync", start="2026-05-23T14:00:00", end="2026-05-23T15:00:00"
        )
        assert "Team Sync" in msg
        assert "2026-05-23T14:00:00" in msg
        assert "2026-05-23T15:00:00" in msg

    def test_create_drive_doc_message_contains_title(self):
        msg = CreateDriveDocTool().get_confirmation_message(title="Q4 Report", content="body")
        assert "Q4 Report" in msg

    def test_create_notion_page_message_contains_title(self):
        msg = CreateNotionPageTool().get_confirmation_message(title="Sprint Notes", content="body")
        assert "Sprint Notes" in msg


class TestConfirmationGate:
    """Verify execute_tool prompts and gates write actions correctly."""

    _event_input = {"summary": "Test", "start": "2026-05-23T14:00:00", "end": "2026-05-23T15:00:00"}

    def test_confirmed_executes_tool(self, monkeypatch):
        monkeypatch.setattr("builtins.input", lambda _: "y")
        with patch.object(tool_registry["create_event"], "execute", return_value={"result": "event created"}):
            result = execute_tool("create_event", self._event_input)
        assert result == "event created"

    def test_declined_cancels(self, monkeypatch):
        monkeypatch.setattr("builtins.input", lambda _: "n")
        result = execute_tool("create_event", self._event_input)
        assert result == "Action cancelled by user."

    def test_empty_answer_cancels(self, monkeypatch):
        monkeypatch.setattr("builtins.input", lambda _: "")
        result = execute_tool("create_event", self._event_input)
        assert result == "Action cancelled by user."

    def test_read_tools_skip_prompt(self, monkeypatch):
        prompted = []
        monkeypatch.setattr("builtins.input", lambda _: prompted.append(True) or "y")
        execute_tool("get_current_time", {})
        assert prompted == []
