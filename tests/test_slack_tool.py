"""
Unit tests for tools/slack_tool.py

All tests mock the Slack WebClient — no real credentials or network calls needed.

Run from the project root:
    pytest tests/test_slack_tool.py -v
"""

import json
from unittest.mock import MagicMock, patch

import pytest
from tools.slack_tool import (
    SlackBaseTool,
    GetChannelMessagesTool,
    SearchSlackTool,
    GetSlackUserInfoTool,
    DraftSlackMessageTool,
)


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

BOT_USER_ID = "UBOTAPP001"
HUMAN_USER_ID = "UHUMAN0001"


def _make_bot_message(text, user_id=BOT_USER_ID, ts="1716000000.000100"):
    """Minimal Slack message dict as posted by a bot."""
    return {
        "type": "message",
        "user": user_id,
        "bot_id": "BAPP000001",
        "app_id": "AAPP000001",
        "text": text,
        "ts": ts,
    }


def _make_human_message(text, user_id=HUMAN_USER_ID, ts="1716000001.000100"):
    """Minimal Slack message dict as posted by a real user."""
    return {
        "type": "message",
        "user": user_id,
        "text": text,
        "ts": ts,
    }


def _mock_bot_client(monkeypatch, messages=None, user_display_name="Marco", channel_id="C_GENERAL"):
    """
    Patch WebClient so _get_bot_client() returns a pre-configured mock.
    Returns the mock client for further assertion.
    """
    monkeypatch.setenv("SLACK_BOT_TOKEN", "xoxb-test")
    mock_client = MagicMock()
    mock_client.conversations_list.return_value = {
        "channels": [{"name": "general", "id": channel_id}],
        "response_metadata": {"next_cursor": ""},
    }
    if messages is not None:
        mock_client.conversations_history.return_value = {"messages": messages}
    mock_client.users_info.return_value = {
        "user": {
            "name": user_display_name.lower(),
            "is_bot": False,
            "profile": {
                "display_name": user_display_name,
                "real_name": user_display_name + " Rossi",
            },
        }
    }
    return mock_client


@pytest.fixture(autouse=True)
def clear_caches():
    """Clear class-level caches before and after each test to prevent pollution."""
    SlackBaseTool._user_cache.clear()
    SlackBaseTool._channel_cache.clear()
    yield
    SlackBaseTool._user_cache.clear()
    SlackBaseTool._channel_cache.clear()


# ---------------------------------------------------------------------------
# SlackBaseTool helpers
# ---------------------------------------------------------------------------

class TestFormatTs:
    def test_converts_slack_timestamp_to_readable_string(self):
        ts = "1716422400.000100"  # 2024-05-23 00:00:00 UTC
        result = SlackBaseTool._format_ts(ts)
        assert "2024-05-23" in result
        assert "UTC" in result

    def test_returns_original_on_invalid_input(self):
        assert SlackBaseTool._format_ts("not-a-timestamp") == "not-a-timestamp"

    def test_returns_original_on_empty_string(self):
        assert SlackBaseTool._format_ts("") == ""


class TestResolveMentions:
    def test_replaces_single_mention(self, monkeypatch):
        monkeypatch.setenv("SLACK_BOT_TOKEN", "xoxb-test")
        mock_client = MagicMock()
        mock_client.users_info.return_value = {
            "user": {"name": "marco", "is_bot": False, "profile": {"display_name": "Marco", "real_name": "Marco"}}
        }
        result = SlackBaseTool._resolve_mentions(f"Hey <@{HUMAN_USER_ID}> can you help?", mock_client)
        assert result == "Hey @Marco can you help?"

    def test_replaces_multiple_different_mentions(self, monkeypatch):
        monkeypatch.setenv("SLACK_BOT_TOKEN", "xoxb-test")
        mock_client = MagicMock()
        mock_client.users_info.side_effect = [
            {"user": {"name": "alice", "is_bot": False, "profile": {"display_name": "Alice", "real_name": "Alice"}}},
            {"user": {"name": "bob", "is_bot": False, "profile": {"display_name": "Bob", "real_name": "Bob"}}},
        ]
        result = SlackBaseTool._resolve_mentions("<@UALICE00001> and <@UBOB000001> are here", mock_client)
        assert "@Alice" in result
        assert "@Bob" in result

    def test_leaves_text_unchanged_when_no_mentions(self, monkeypatch):
        monkeypatch.setenv("SLACK_BOT_TOKEN", "xoxb-test")
        mock_client = MagicMock()
        text = "No mentions in this message"
        result = SlackBaseTool._resolve_mentions(text, mock_client)
        assert result == text
        mock_client.users_info.assert_not_called()

    def test_uses_cache_for_repeated_mentions(self, monkeypatch):
        monkeypatch.setenv("SLACK_BOT_TOKEN", "xoxb-test")
        mock_client = MagicMock()
        mock_client.users_info.return_value = {
            "user": {"name": "marco", "is_bot": False, "profile": {"display_name": "Marco", "real_name": "Marco"}}
        }
        SlackBaseTool._resolve_mentions(f"<@{HUMAN_USER_ID}> asked <@{HUMAN_USER_ID}> again", mock_client)
        # Second mention of the same user should hit the cache, not make a second API call.
        assert mock_client.users_info.call_count == 1


# ---------------------------------------------------------------------------
# GetChannelMessagesTool
# ---------------------------------------------------------------------------

class TestGetChannelMessagesTool:
    def _tool(self):
        return GetChannelMessagesTool()

    def test_validate_input_requires_channel(self):
        tool = self._tool()
        assert tool.validate_input(channel="general") is True
        assert tool.validate_input() is False
        assert tool.validate_input(channel="") is False

    def test_returns_json_list(self, monkeypatch):
        mock_client = _mock_bot_client(
            monkeypatch,
            messages=[_make_human_message("Hello team", user_id="U_HUMAN")],
        )
        with patch("tools.slack_tool.WebClient", return_value=mock_client):
            result = self._tool().execute(channel="general")

        assert "error" not in result
        data = json.loads(result["result"])
        assert isinstance(data, list)
        assert len(data) == 1

    def test_message_has_expected_fields(self, monkeypatch):
        mock_client = _mock_bot_client(
            monkeypatch,
            messages=[_make_human_message("Hello", user_id="U_HUMAN")],
        )
        with patch("tools.slack_tool.WebClient", return_value=mock_client):
            result = self._tool().execute(channel="general")

        msg = json.loads(result["result"])[0]
        assert "ts" in msg
        assert "user_id" in msg
        assert "username" in msg
        assert "is_bot" in msg
        assert "text" in msg

    def test_bot_messages_marked_is_bot_true(self, monkeypatch):
        mock_client = _mock_bot_client(
            monkeypatch,
            messages=[_make_bot_message("Bot says hello")],
        )
        with patch("tools.slack_tool.WebClient", return_value=mock_client):
            result = self._tool().execute(channel="general")

        msg = json.loads(result["result"])[0]
        assert msg["is_bot"] is True

    def test_human_messages_marked_is_bot_false(self, monkeypatch):
        mock_client = _mock_bot_client(
            monkeypatch,
            messages=[_make_human_message("Human says hello")],
        )
        with patch("tools.slack_tool.WebClient", return_value=mock_client):
            result = self._tool().execute(channel="general")

        msg = json.loads(result["result"])[0]
        assert msg["is_bot"] is False

    def test_messages_returned_in_chronological_order(self, monkeypatch):
        # Slack returns newest-first; tool should reverse to oldest-first.
        messages = [
            _make_human_message("Third", ts="1716000003.000000"),
            _make_human_message("Second", ts="1716000002.000000"),
            _make_human_message("First", ts="1716000001.000000"),
        ]
        mock_client = _mock_bot_client(monkeypatch, messages=messages)
        with patch("tools.slack_tool.WebClient", return_value=mock_client):
            result = self._tool().execute(channel="general")

        data = json.loads(result["result"])
        assert data[0]["text"] == "First"
        assert data[1]["text"] == "Second"
        assert data[2]["text"] == "Third"

    def test_skips_messages_with_empty_text(self, monkeypatch):
        messages = [
            _make_human_message("Real message"),
            {**_make_human_message(""), "text": "   "},
        ]
        mock_client = _mock_bot_client(monkeypatch, messages=messages)
        with patch("tools.slack_tool.WebClient", return_value=mock_client):
            result = self._tool().execute(channel="general")

        data = json.loads(result["result"])
        assert len(data) == 1
        assert data[0]["text"] == "Real message"

    def test_resolves_at_mentions_in_message_text(self, monkeypatch):
        mock_client = _mock_bot_client(
            monkeypatch,
            messages=[_make_human_message(f"Thanks <@{HUMAN_USER_ID}> for joining")],
            user_display_name="Marco",
        )
        with patch("tools.slack_tool.WebClient", return_value=mock_client):
            result = self._tool().execute(channel="general")

        msg = json.loads(result["result"])[0]
        assert "@Marco" in msg["text"]
        assert "<@U_HUMAN>" not in msg["text"]

    def test_returns_no_messages_found_when_channel_empty(self, monkeypatch):
        mock_client = _mock_bot_client(monkeypatch, messages=[])
        with patch("tools.slack_tool.WebClient", return_value=mock_client):
            result = self._tool().execute(channel="general")

        assert result["result"] == "No messages found."

    def test_caps_limit_at_50(self, monkeypatch):
        mock_client = _mock_bot_client(monkeypatch, messages=[])
        with patch("tools.slack_tool.WebClient", return_value=mock_client):
            self._tool().execute(channel="general", limit=999)

        _, call_kwargs = mock_client.conversations_history.call_args
        assert call_kwargs["limit"] == 50

    def test_returns_error_on_api_failure(self, monkeypatch):
        monkeypatch.setenv("SLACK_BOT_TOKEN", "xoxb-test")
        mock_client = MagicMock()
        mock_client.conversations_list.side_effect = Exception("Network error")
        with patch("tools.slack_tool.WebClient", return_value=mock_client):
            result = self._tool().execute(channel="general")

        assert "error" in result
        assert "Network error" in result["error"]

    def test_returns_error_on_unknown_channel(self, monkeypatch):
        monkeypatch.setenv("SLACK_BOT_TOKEN", "xoxb-test")
        mock_client = MagicMock()
        mock_client.conversations_list.return_value = {
            "channels": [],
            "response_metadata": {"next_cursor": ""},
        }
        with patch("tools.slack_tool.WebClient", return_value=mock_client):
            result = self._tool().execute(channel="nonexistent")

        assert "error" in result

    def test_description_has_required_fields(self):
        desc = self._tool().get_description()
        assert desc["name"] == "get_channel_messages"
        assert "channel" in desc["input_schema"]["properties"]
        assert "channel" in desc["input_schema"]["required"]

    def test_category_is_slack(self):
        assert GetChannelMessagesTool.category == "slack"

    def test_summarizable_is_true(self):
        assert GetChannelMessagesTool.summarizable is True


# ---------------------------------------------------------------------------
# SearchSlackTool
# ---------------------------------------------------------------------------

class TestSearchSlackTool:
    def _tool(self):
        return SearchSlackTool()

    def test_validate_input_requires_query(self):
        tool = self._tool()
        assert tool.validate_input(query="clustering") is True
        assert tool.validate_input() is False
        assert tool.validate_input(query="") is False

    def test_returns_formatted_results(self, monkeypatch):
        monkeypatch.setenv("SLACK_USER_TOKEN", "xoxp-test")
        mock_client = MagicMock()
        mock_client.search_messages.return_value = {
            "messages": {
                "matches": [
                    {
                        "username": "Marco",
                        "channel": {"name": "data-science"},
                        "ts": "1716000000.000100",
                        "text": "FastText is the way to go",
                    },
                    {
                        "username": "Anna",
                        "channel": {"name": "general"},
                        "ts": "1716000001.000100",
                        "text": "Agreed on FastText",
                    },
                ]
            }
        }
        with patch("tools.slack_tool.WebClient", return_value=mock_client):
            result = self._tool().execute(query="FastText")

        assert "error" not in result
        assert "Marco" in result["result"]
        assert "data-science" in result["result"]
        assert "FastText is the way to go" in result["result"]
        assert "Anna" in result["result"]

    def test_returns_no_results_message_when_empty(self, monkeypatch):
        monkeypatch.setenv("SLACK_USER_TOKEN", "xoxp-test")
        mock_client = MagicMock()
        mock_client.search_messages.return_value = {"messages": {"matches": []}}
        with patch("tools.slack_tool.WebClient", return_value=mock_client):
            result = self._tool().execute(query="nonexistent topic")

        assert "error" not in result
        assert "No Slack messages found" in result["result"]
        assert "nonexistent topic" in result["result"]

    def test_caps_max_results_at_25(self, monkeypatch):
        monkeypatch.setenv("SLACK_USER_TOKEN", "xoxp-test")
        mock_client = MagicMock()
        mock_client.search_messages.return_value = {"messages": {"matches": []}}
        with patch("tools.slack_tool.WebClient", return_value=mock_client):
            self._tool().execute(query="test", max_results=999)

        _, call_kwargs = mock_client.search_messages.call_args
        assert call_kwargs["count"] == 25

    def test_raises_when_user_token_missing(self, monkeypatch):
        monkeypatch.delenv("SLACK_USER_TOKEN", raising=False)
        result = self._tool().execute(query="test")
        assert "error" in result
        assert "SLACK_USER_TOKEN" in result["error"]

    def test_returns_error_on_api_failure(self, monkeypatch):
        monkeypatch.setenv("SLACK_USER_TOKEN", "xoxp-test")
        mock_client = MagicMock()
        mock_client.search_messages.side_effect = Exception("Search failed")
        with patch("tools.slack_tool.WebClient", return_value=mock_client):
            result = self._tool().execute(query="test")

        assert "error" in result
        assert "Search failed" in result["error"]

    def test_description_has_required_fields(self):
        desc = self._tool().get_description()
        assert desc["name"] == "search_slack"
        assert "query" in desc["input_schema"]["properties"]
        assert "query" in desc["input_schema"]["required"]

    def test_category_is_slack(self):
        assert SearchSlackTool.category == "slack"

    def test_summarizable_is_true(self):
        assert SearchSlackTool.summarizable is True


# ---------------------------------------------------------------------------
# GetSlackUserInfoTool
# ---------------------------------------------------------------------------

class TestGetSlackUserInfoTool:
    def _tool(self):
        return GetSlackUserInfoTool()

    def test_validate_input_requires_user_id(self):
        tool = self._tool()
        assert tool.validate_input(user_id="U123ABC") is True
        assert tool.validate_input() is False
        assert tool.validate_input(user_id="") is False

    def test_returns_user_profile(self, monkeypatch):
        monkeypatch.setenv("SLACK_BOT_TOKEN", "xoxb-test")
        mock_client = MagicMock()
        mock_client.users_info.return_value = {
            "user": {
                "name": "marco",
                "is_bot": False,
                "profile": {
                    "display_name": "Marco",
                    "real_name": "Marco Rossi",
                },
            }
        }
        with patch("tools.slack_tool.WebClient", return_value=mock_client):
            result = self._tool().execute(user_id="U_HUMAN")

        assert "error" not in result
        data = json.loads(result["result"])
        assert data["user_id"] == "U_HUMAN"
        assert data["display_name"] == "Marco"
        assert data["real_name"] == "Marco Rossi"
        assert data["is_bot"] is False

    def test_identifies_bot_accounts(self, monkeypatch):
        monkeypatch.setenv("SLACK_BOT_TOKEN", "xoxb-test")
        mock_client = MagicMock()
        mock_client.users_info.return_value = {
            "user": {
                "name": "quicklocal",
                "is_bot": True,
                "profile": {"display_name": "QuickLocal", "real_name": "QuickLocal"},
            }
        }
        with patch("tools.slack_tool.WebClient", return_value=mock_client):
            result = self._tool().execute(user_id="U_BOT")

        data = json.loads(result["result"])
        assert data["is_bot"] is True

    def test_returns_error_on_api_failure(self, monkeypatch):
        monkeypatch.setenv("SLACK_BOT_TOKEN", "xoxb-test")
        mock_client = MagicMock()
        mock_client.users_info.side_effect = Exception("User not found")
        with patch("tools.slack_tool.WebClient", return_value=mock_client):
            result = self._tool().execute(user_id="U_GONE")

        assert "error" in result
        assert "User not found" in result["error"]

    def test_description_has_required_fields(self):
        desc = self._tool().get_description()
        assert desc["name"] == "get_slack_user_info"
        assert "user_id" in desc["input_schema"]["properties"]
        assert "user_id" in desc["input_schema"]["required"]

    def test_summarizable_is_false(self):
        assert GetSlackUserInfoTool.summarizable is False

    def test_category_is_slack(self):
        assert GetSlackUserInfoTool.category == "slack"


# ---------------------------------------------------------------------------
# DraftSlackMessageTool
# ---------------------------------------------------------------------------

class TestDraftSlackMessageTool:
    def _tool(self):
        return DraftSlackMessageTool()

    def test_validate_input_requires_channel_and_text(self):
        tool = self._tool()
        assert tool.validate_input(channel="general", text="Hello") is True
        assert tool.validate_input(channel="general") is False
        assert tool.validate_input(text="Hello") is False
        assert tool.validate_input() is False

    def test_returns_preview_with_channel_and_text(self):
        result = self._tool().execute(channel="general", text="Meeting at 3pm")
        assert "error" not in result
        assert "general" in result["result"]
        assert "Meeting at 3pm" in result["result"]

    def test_strips_hash_from_channel_name(self):
        result = self._tool().execute(channel="#data-science", text="Hello")
        assert "#data-science" in result["result"]
        # Should not have double hash
        assert "##" not in result["result"]

    def test_preview_notes_message_not_sent(self):
        result = self._tool().execute(channel="general", text="Hello")
        assert "NOT been sent" in result["result"] or "NOT" in result["result"]

    def test_does_not_call_any_api(self, monkeypatch):
        monkeypatch.setenv("SLACK_BOT_TOKEN", "xoxb-test")
        mock_client = MagicMock()
        with patch("tools.slack_tool.WebClient", return_value=mock_client):
            self._tool().execute(channel="general", text="Hello")
        mock_client.assert_not_called()

    def test_description_mentions_confirmation(self):
        desc = self._tool().get_description()["description"]
        assert "confirm" in desc.lower()

    def test_description_says_does_not_post(self):
        desc = self._tool().get_description()["description"]
        assert "not post" in desc.lower() or "does not post" in desc.lower()

    def test_summarizable_is_false(self):
        assert DraftSlackMessageTool.summarizable is False

    def test_category_is_slack(self):
        assert DraftSlackMessageTool.category == "slack"
