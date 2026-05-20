"""
Unit tests for the agent and core tools.

Run from the project root:
    pytest tests/test_agent.py -v
"""

import re

import pytest
from tools.time_tool import TimeTool
from tools.calculator_tool import CalculatorTool
from src.agent import execute_tool


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
