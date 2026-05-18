"""
Unit tests for conversation memory features in src/agent.py:
  - load_history / save_history / clear_history
  - _has_summary / _count_exchanges
  - summarize_old_exchanges

All summarization tests mock the Anthropic client — no real API calls.

Run from the project root:
    pytest tests/test_conversation_memory.py -v
"""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import src.agent as agent
from src.agent import (
    _count_exchanges,
    _has_summary,
    clear_history,
    load_history,
    save_history,
    summarize_old_exchanges,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_exchange(user_text="Hello", assistant_text="Hi there"):
    return [
        {"role": "user", "content": user_text},
        {"role": "assistant", "content": assistant_text},
    ]


def _make_conversation(n_exchanges):
    """Return a conversation with n plain user/assistant exchanges."""
    conv = []
    for i in range(n_exchanges):
        conv += _make_exchange(f"user message {i}", f"assistant reply {i}")
    return conv


def _make_summary_pair(text="Earlier summary."):
    return [
        {"role": "user", "content": f"Summary of our earlier conversation:\n{text}"},
        {"role": "assistant", "content": "Understood, I have the context from our earlier conversation."},
    ]


# ---------------------------------------------------------------------------
# load_history / save_history / clear_history
# ---------------------------------------------------------------------------

class TestPersistence:
    def test_load_returns_empty_list_when_no_file(self, tmp_path, monkeypatch):
        monkeypatch.setattr(agent, "HISTORY_FILE", tmp_path / "history.json")
        assert load_history() == []

    def test_save_and_load_roundtrip(self, tmp_path, monkeypatch):
        monkeypatch.setattr(agent, "HISTORY_FILE", tmp_path / "history.json")
        conv = _make_conversation(2)
        save_history(conv)
        assert load_history() == conv

    def test_load_returns_empty_on_corrupt_file(self, tmp_path, monkeypatch):
        history_file = tmp_path / "history.json"
        history_file.write_text("not valid json")
        monkeypatch.setattr(agent, "HISTORY_FILE", history_file)
        assert load_history() == []

    def test_clear_deletes_file(self, tmp_path, monkeypatch):
        history_file = tmp_path / "history.json"
        history_file.write_text("[]")
        monkeypatch.setattr(agent, "HISTORY_FILE", history_file)
        clear_history()
        assert not history_file.exists()

    def test_clear_does_not_raise_when_no_file(self, tmp_path, monkeypatch):
        monkeypatch.setattr(agent, "HISTORY_FILE", tmp_path / "nonexistent.json")
        clear_history()  # should not raise

    def test_save_writes_valid_json(self, tmp_path, monkeypatch):
        monkeypatch.setattr(agent, "HISTORY_FILE", tmp_path / "history.json")
        save_history(_make_conversation(1))
        content = (tmp_path / "history.json").read_text()
        assert isinstance(json.loads(content), list)


# ---------------------------------------------------------------------------
# _has_summary
# ---------------------------------------------------------------------------

class TestHasSummary:
    def test_detects_summary_pair(self):
        conv = _make_summary_pair() + _make_conversation(2)
        assert _has_summary(conv) is True

    def test_plain_conversation_has_no_summary(self):
        assert _has_summary(_make_conversation(3)) is False

    def test_empty_conversation_has_no_summary(self):
        assert _has_summary([]) is False

    def test_regular_user_message_not_detected_as_summary(self):
        conv = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
        ]
        assert _has_summary(conv) is False


# ---------------------------------------------------------------------------
# _count_exchanges
# ---------------------------------------------------------------------------

class TestCountExchanges:
    def test_counts_plain_exchanges(self):
        assert _count_exchanges(_make_conversation(5)) == 5

    def test_excludes_summary_pair_from_count(self):
        conv = _make_summary_pair() + _make_conversation(5)
        assert _count_exchanges(conv) == 5

    def test_empty_conversation_is_zero(self):
        assert _count_exchanges([]) == 0

    def test_only_summary_pair_is_zero(self):
        assert _count_exchanges(_make_summary_pair()) == 0


# ---------------------------------------------------------------------------
# summarize_old_exchanges
# ---------------------------------------------------------------------------

def _mock_summary_response(text="Mocked summary."):
    mock_block = MagicMock()
    mock_block.text = text
    mock_response = MagicMock()
    mock_response.content = [mock_block]
    return mock_response


class TestSummarizeOldExchanges:
    def _patch_client(self, summary_text="Mocked summary."):
        mock_response = _mock_summary_response(summary_text)
        return patch.object(agent.client.messages, "create", return_value=mock_response)

    def test_short_conversation_unchanged(self):
        conv = _make_conversation(agent.MAX_VERBATIM_EXCHANGES)
        with self._patch_client() as mock_create:
            result = summarize_old_exchanges(conv)
        mock_create.assert_not_called()
        assert result == conv

    def test_long_conversation_triggers_summary(self):
        conv = _make_conversation(agent.MAX_VERBATIM_EXCHANGES + 2)
        with self._patch_client("Summary of old stuff."):
            result = summarize_old_exchanges(conv)
        assert _has_summary(result)

    def test_verbatim_exchanges_preserved(self):
        n = agent.MAX_VERBATIM_EXCHANGES
        conv = _make_conversation(n + 3)
        verbatim_before = conv[-(n * 2):]
        with self._patch_client():
            result = summarize_old_exchanges(conv)
        # After summary pair, the verbatim messages should be unchanged.
        assert result[2:] == verbatim_before

    def test_summary_text_appears_in_output(self):
        conv = _make_conversation(agent.MAX_VERBATIM_EXCHANGES + 1)
        with self._patch_client("Key points from earlier."):
            result = summarize_old_exchanges(conv)
        assert "Key points from earlier." in result[0]["content"]

    def test_summary_replaced_when_already_present(self):
        old_summary = _make_summary_pair("Old summary.")
        conv = old_summary + _make_conversation(agent.MAX_VERBATIM_EXCHANGES + 1)
        with self._patch_client("New summary."):
            result = summarize_old_exchanges(conv)
        assert "New summary." in result[0]["content"]
        # Should still be exactly one summary pair at the start.
        assert result[1]["role"] == "assistant"
        assert result[2]["role"] == "user"

    def test_total_length_reduced(self):
        conv = _make_conversation(agent.MAX_VERBATIM_EXCHANGES + 5)
        original_len = len(conv)
        with self._patch_client():
            result = summarize_old_exchanges(conv)
        # summary pair (2) + verbatim (MAX * 2) < original length
        assert len(result) < original_len

    def test_exactly_one_over_limit_summarizes(self):
        conv = _make_conversation(agent.MAX_VERBATIM_EXCHANGES + 1)
        with self._patch_client():
            result = summarize_old_exchanges(conv)
        assert _has_summary(result)
