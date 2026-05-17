"""
Unit tests for src/agent.py

Run from the project root:
    python -m pytest tests/test_agent.py -v
"""

import re
import pytest
from src.agent import execute_tool, get_current_time, calculate


class TestGetCurrentTime:
    def test_returns_string(self):
        result = get_current_time()
        assert isinstance(result, str)

    def test_not_empty(self):
        result = get_current_time()
        assert len(result) > 0

    def test_expected_format(self):
        # Expected: "Monday, May 17 2026 — 03:45:22 PM"
        result = get_current_time()
        pattern = r"^\w+, \w+ \d{2} \d{4} — \d{2}:\d{2}:\d{2} (AM|PM)$"
        assert re.match(pattern, result), f"Unexpected format: {result!r}"


class TestCalculate:
    def test_add(self):
        assert calculate("add", 3, 4) == "7"

    def test_subtract(self):
        assert calculate("subtract", 10, 3) == "7"

    def test_multiply(self):
        assert calculate("multiply", 6, 7) == "42"

    def test_divide(self):
        assert calculate("divide", 128, 4) == "32.0"

    def test_divide_by_zero(self):
        result = calculate("divide", 5, 0)
        assert "Error" in result
        assert "zero" in result

    def test_unknown_operation(self):
        result = calculate("modulo", 10, 3)
        assert "Error" in result
        assert "modulo" in result

    def test_float_inputs(self):
        assert calculate("add", 1.5, 2.5) == "4.0"

    def test_negative_numbers(self):
        assert calculate("multiply", -3, 4) == "-12"


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
