"""
Unit tests for conversation memory features in src/agent.py:
  - load_history / save_history / clear_history
  - _has_summary / _count_exchanges
  - summarize_old_exchanges
  - _strip_tool_blocks
  - _sanitize_history

All summarization tests mock the Anthropic client — no real API calls.

Run from the project root:
    pytest tests/test_conversation_memory.py -v
"""

import json
from unittest.mock import MagicMock, patch

import pytest
import src.agent as agent
from src.agent import (
    _count_exchanges,
    _format_tool_input,
    _has_summary,
    _sanitize_history,
    _strip_tool_blocks,
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


def _make_live_tool_exchange(tool_id="tid-001", tool_name="search_documents",
                             tool_input=None, result="some results"):
    """Return 4 messages for a full in-memory tool exchange.

    Structure: user question → assistant tool_use → user tool_result → assistant answer.
    Place this before enough plain exchanges to push it into the to_summarize portion.
    """
    if tool_input is None:
        tool_input = {"query": "test query"}
    return [
        {"role": "user", "content": "search my docs"},
        {"role": "assistant", "content": [
            {"type": "tool_use", "id": tool_id, "name": tool_name, "input": tool_input},
        ]},
        {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": tool_id, "content": result},
        ]},
        {"role": "assistant", "content": "Here is what I found."},
    ]


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

    def _capture_transcript(self, conv):
        """Run summarize_old_exchanges and return the transcript string sent to Claude."""
        mock_create = MagicMock(return_value=_mock_summary_response())
        with patch.object(agent.client.messages, "create", mock_create):
            summarize_old_exchanges(conv)
        return mock_create.call_args.kwargs["messages"][0]["content"]

    def test_summarizable_tool_use_annotated_in_transcript(self):
        """tool_use for a summarizable tool produces a [Used ...] line in the transcript."""
        conv = (
            _make_live_tool_exchange(tool_name="search_documents", tool_input={"query": "meeting notes"})
            + _make_conversation(agent.MAX_VERBATIM_EXCHANGES + 1)
        )
        transcript = self._capture_transcript(conv)
        assert "[Used search_documents:" in transcript

    def test_non_summarizable_tool_use_not_in_transcript(self):
        """tool_use for a non-summarizable tool is silently skipped."""
        conv = (
            _make_live_tool_exchange(tool_name="calculate", tool_input={"operation": "add", "a": 1, "b": 2})
            + _make_conversation(agent.MAX_VERBATIM_EXCHANGES + 1)
        )
        transcript = self._capture_transcript(conv)
        assert "[Used calculate:" not in transcript

    def test_summarizable_tool_result_annotated_in_transcript(self):
        """tool_result for a summarizable tool produces a [Result: ...] line in the transcript."""
        conv = (
            _make_live_tool_exchange(tool_name="search_documents", result="Found notes from the Q4 planning meeting.")
            + _make_conversation(agent.MAX_VERBATIM_EXCHANGES + 1)
        )
        transcript = self._capture_transcript(conv)
        assert "[Result:" in transcript
        assert "Q4 planning meeting" in transcript

    def test_tool_result_truncated_at_200_chars(self):
        """Results longer than 200 chars are truncated with '...' in the transcript."""
        long_result = "x" * 250
        conv = (
            _make_live_tool_exchange(tool_name="search_documents", result=long_result)
            + _make_conversation(agent.MAX_VERBATIM_EXCHANGES + 1)
        )
        transcript = self._capture_transcript(conv)
        assert "[Result: " + "x" * 200 + "..." in transcript
        assert "x" * 201 not in transcript

    def test_cutoff_does_not_split_tool_exchange(self):
        """Cutoff must never land between a tool_use and its tool_result.

        Build a conversation at exactly the verbatim limit, then append a full
        tool exchange (user → assistant:tool_use → user:tool_result →
        assistant:answer).  The summariser should keep the entire tool exchange
        in the verbatim tail so no orphan tool_result is produced.
        """
        conv = _make_conversation(agent.MAX_VERBATIM_EXCHANGES)
        tool_id = "test-tool-id-001"
        tool_exchange = [
            {"role": "user", "content": "search something"},
            {"role": "assistant", "content": [
                {"type": "tool_use", "id": tool_id, "name": "search_documents", "input": {"query": "test"}},
            ]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": tool_id, "content": "some results"},
            ]},
            {"role": "assistant", "content": "Here is what I found."},
        ]
        conv += tool_exchange

        with self._patch_client():
            result = summarize_old_exchanges(conv)

        # Collect all tool_use ids present in the result
        tool_use_ids = set()
        for msg in result:
            if isinstance(msg.get("content"), list):
                for block in msg["content"]:
                    if isinstance(block, dict) and block.get("type") == "tool_use":
                        tool_use_ids.add(block["id"])

        # Every tool_result must have a matching tool_use in the same conversation
        for msg in result:
            if isinstance(msg.get("content"), list):
                for block in msg["content"]:
                    if isinstance(block, dict) and block.get("type") == "tool_result":
                        assert block["tool_use_id"] in tool_use_ids, (
                            f"Orphaned tool_result with id {block['tool_use_id']!r}"
                        )


# ---------------------------------------------------------------------------
# _strip_tool_blocks
# ---------------------------------------------------------------------------

def _make_tool_exchange(tool_id="tool-abc", tool_name="search_documents"):
    return [
        {"role": "assistant", "content": [
            {"type": "tool_use", "id": tool_id, "name": tool_name, "input": {"query": "test"}},
        ]},
        {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": tool_id, "content": "results"},
        ]},
    ]


class TestStripToolBlocks:
    def test_plain_messages_pass_through_unchanged(self):
        conv = _make_conversation(2)
        assert _strip_tool_blocks(conv) == conv

    def test_pure_tool_result_user_message_dropped(self):
        conv = (
            [{"role": "user", "content": "search my docs"}]
            + _make_tool_exchange()
            + [{"role": "assistant", "content": "Here is what I found."}]
        )
        result = _strip_tool_blocks(conv)
        # Only the plain user message and final assistant text survive
        assert len(result) == 2
        assert result[0]["content"] == "search my docs"
        assert result[1]["content"] == "Here is what I found."

    def test_pure_tool_use_assistant_message_dropped(self):
        conv = _make_tool_exchange()
        result = _strip_tool_blocks(conv)
        assert result == []

    def test_mixed_assistant_message_keeps_text_only(self):
        conv = [{"role": "assistant", "content": [
            {"type": "text", "text": "Let me look that up."},
            {"type": "tool_use", "id": "x", "name": "foo", "input": {}},
        ]}]
        result = _strip_tool_blocks(conv)
        assert len(result) == 1
        assert result[0]["content"] == [{"type": "text", "text": "Let me look that up."}]

    def test_save_history_strips_tool_blocks(self, tmp_path, monkeypatch):
        monkeypatch.setattr(agent, "HISTORY_FILE", tmp_path / "history.json")
        conv = (
            [{"role": "user", "content": "search my docs"}]
            + _make_tool_exchange()
            + [{"role": "assistant", "content": "Here is what I found."}]
        )
        save_history(conv)
        saved = json.loads((tmp_path / "history.json").read_text())
        for msg in saved:
            assert isinstance(msg["content"], str), (
                f"Expected str content after stripping, got: {msg['content']!r}"
            )

    def test_multiple_tool_exchanges_all_stripped(self):
        conv = (
            [{"role": "user", "content": "first question"}]
            + _make_tool_exchange(tool_id="t1")
            + [{"role": "assistant", "content": "First answer."}]
            + [{"role": "user", "content": "second question"}]
            + _make_tool_exchange(tool_id="t2")
            + [{"role": "assistant", "content": "Second answer."}]
        )
        result = _strip_tool_blocks(conv)
        assert len(result) == 4
        assert all(isinstance(m["content"], str) for m in result)


# ---------------------------------------------------------------------------
# _sanitize_history
# ---------------------------------------------------------------------------

class TestSanitizeHistory:
    def test_clean_conversation_unchanged(self):
        conv = _make_conversation(3)
        assert _sanitize_history(conv) == conv

    def test_orphaned_tool_result_dropped(self):
        conv = [
            {"role": "user", "content": "hello"},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "ghost-id", "content": "stale result"},
            ]},
            {"role": "assistant", "content": "Done."},
        ]
        result = _sanitize_history(conv)
        # The orphaned tool_result user message should be gone
        assert all(isinstance(m["content"], str) for m in result)
        assert len(result) == 2

    def test_matched_tool_pair_preserved(self):
        tool_id = "real-id"
        conv = (
            [{"role": "user", "content": "search"}]
            + [{"role": "assistant", "content": [
                {"type": "tool_use", "id": tool_id, "name": "search_documents", "input": {}},
            ]}]
            + [{"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": tool_id, "content": "results"},
            ]}]
            + [{"role": "assistant", "content": "Found it."}]
        )
        result = _sanitize_history(conv)
        # All four messages should survive — the pair is intact
        assert len(result) == 4


# ---------------------------------------------------------------------------
# _format_tool_input
# ---------------------------------------------------------------------------

class TestFormatToolInput:
    def test_empty_dict_returns_no_input(self):
        assert _format_tool_input({}) == "no input"

    def test_single_key(self):
        assert _format_tool_input({"query": "emails"}) == "query='emails'"

    def test_multiple_keys(self):
        result = _format_tool_input({"a": 1, "b": 2})
        assert result == "a=1, b=2"

