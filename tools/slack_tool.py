"""
Slack integration using slack-sdk.

Auth:
  SLACK_BOT_TOKEN  — bot token (xoxb-...) for reading channel history and drafts.
  SLACK_USER_TOKEN — user token (xoxp-...) for search.messages (requires search:read scope).
  Both are read from the environment (.env file).
"""

import json
import os
import re
from datetime import datetime, timezone

from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError

try:
    from tools.base_tool import BaseTool
except ImportError:
    from base_tool import BaseTool


class SlackBaseTool(BaseTool):
    """Shared auth and helpers for all Slack tools."""

    category = "slack"
    summarizable = True

    # Class-level caches to avoid repeated API calls within a session.
    _user_cache: dict[str, str] = {}
    _channel_cache: dict[str, str] = {}

    @classmethod
    def _get_bot_client(cls) -> WebClient:
        """Return a WebClient authenticated with the bot token."""
        token = os.getenv("SLACK_BOT_TOKEN")
        if not token:
            raise EnvironmentError(
                "SLACK_BOT_TOKEN is not set. Add it to your .env file."
            )
        return WebClient(token=token)

    @classmethod
    def _get_user_client(cls) -> WebClient:
        """Return a WebClient authenticated with the user token (needed for search)."""
        token = os.getenv("SLACK_USER_TOKEN")
        if not token:
            raise EnvironmentError(
                "SLACK_USER_TOKEN is not set. Add it to your .env file. "
                "Get it from api.slack.com/apps → OAuth & Permissions → User OAuth Token."
            )
        return WebClient(token=token)

    @classmethod
    def _resolve_username(cls, user_id: str, client: WebClient) -> str:
        """Map a Slack user ID (e.g. U123ABC) to a display name. Cached per session."""
        if user_id in cls._user_cache:
            return cls._user_cache[user_id]
        try:
            resp = client.users_info(user=user_id)
            profile = resp["user"].get("profile", {})
            name = (
                profile.get("display_name")
                or profile.get("real_name")
                or resp["user"].get("name")
                or user_id
            )
        except SlackApiError:
            name = user_id
        cls._user_cache[user_id] = name
        return name

    @classmethod
    def _resolve_channel_id(cls, channel: str, client: WebClient) -> str:
        """
        Resolve a channel name (e.g. 'general' or '#general') to a channel ID.
        Falls through unchanged if it already looks like an ID (starts with 'C').
        """
        channel = channel.lstrip("#")
        if channel.upper().startswith("C") and len(channel) >= 9:
            return channel
        if channel in cls._channel_cache:
            return cls._channel_cache[channel]
        # Fetch all public channels and match by name.
        cursor = None
        while True:
            params = {"exclude_archived": True, "limit": 200, "types": "public_channel"}
            if cursor:
                params["cursor"] = cursor
            resp = client.conversations_list(**params)
            for ch in resp.get("channels", []):
                if ch["name"] == channel:
                    cls._channel_cache[channel] = ch["id"]
                    return ch["id"]
            cursor = resp.get("response_metadata", {}).get("next_cursor")
            if not cursor:
                break
        raise ValueError(
            f"Channel '{channel}' not found. Make sure the bot is invited to that channel."
        )

    @classmethod
    def _resolve_mentions(cls, text: str, client: WebClient) -> str:
        """Replace Slack @mention syntax (<@UXXXXX>) with @display_name."""
        def replace(match: re.Match) -> str:
            return "@" + cls._resolve_username(match.group(1), client)
        return re.sub(r"<@(U[A-Z0-9]+)>", replace, text)

    @staticmethod
    def _format_ts(ts: str) -> str:
        """Convert a Slack timestamp (e.g. '1716000000.000100') to a readable string."""
        try:
            dt = datetime.fromtimestamp(float(ts), tz=timezone.utc)
            return dt.strftime("%Y-%m-%d %H:%M UTC")
        except (ValueError, OSError):
            return ts


class GetChannelMessagesTool(SlackBaseTool):
    """Fetch recent messages from a Slack channel."""

    name = "get_channel_messages"

    def get_description(self) -> dict:
        return {
            "name": self.name,
            "description": (
                "Fetches the most recent messages from a Slack channel. "
                "Returns a JSON list of messages, each with ts, user_id, username, is_bot, and text. "
                "is_bot=true means the message was posted by an app or bot, not a human. "
                "Use get_slack_user_info to resolve any user_id to a full profile if needed. "
                "Use when the user asks what was discussed in a specific channel, "
                "or wants to see the latest activity in a channel."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "channel": {
                        "type": "string",
                        "description": "Channel name (e.g. 'general' or '#data-science').",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Number of messages to retrieve (default 10, max 50).",
                        "default": 10,
                    },
                },
                "required": ["channel"],
            },
        }

    def execute(self, **kwargs) -> dict:
        try:
            channel_name = kwargs["channel"]
            limit = min(int(kwargs.get("limit", 10)), 50)

            client = self._get_bot_client()
            channel_id = self._resolve_channel_id(channel_name, client)

            resp = client.conversations_history(channel=channel_id, limit=limit)
            messages = resp.get("messages", [])

            # Slack returns newest-first; reverse for chronological display.
            messages = list(reversed(messages))

            results = []
            for msg in messages:
                text = msg.get("text", "").strip()
                if not text:
                    continue
                user_id = msg.get("user", "")
                is_bot = "bot_id" in msg
                username = (
                    self._resolve_username(user_id, client) if user_id else "unknown"
                )
                results.append({
                    "ts": self._format_ts(msg.get("ts", "")),
                    "user_id": user_id,
                    "username": username,
                    "is_bot": is_bot,
                    "text": self._resolve_mentions(text, client),
                })

            if not results:
                return {"result": "No messages found."}
            return {"result": json.dumps(results, ensure_ascii=False, indent=2)}
        except (SlackApiError, ValueError) as e:
            return self.handle_error(e)
        except Exception as e:
            return self.handle_error(e)

    def validate_input(self, **kwargs) -> bool:
        return bool(kwargs.get("channel"))


class SearchSlackTool(SlackBaseTool):
    """Search Slack messages using the Slack search API (requires user token)."""

    name = "search_slack"

    def get_description(self) -> dict:
        return {
            "name": self.name,
            "description": (
                "Searches across all Slack messages the user can see using a keyword query. "
                "Returns matching messages with author, channel, timestamp, and text. "
                "Use when the user asks to find specific discussions, decisions, or mentions "
                "across channels (e.g. 'search Slack for Q4 planning' or 'find messages about clustering')."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query — keywords, phrases, or Slack search modifiers (e.g. 'in:#data-science clustering').",
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of results to return (default 10, max 25).",
                        "default": 10,
                    },
                },
                "required": ["query"],
            },
        }

    def execute(self, **kwargs) -> dict:
        try:
            query = kwargs["query"]
            max_results = min(int(kwargs.get("max_results", 10)), 25)

            client = self._get_user_client()
            resp = client.search_messages(query=query, count=max_results)

            matches = resp.get("messages", {}).get("matches", [])
            if not matches:
                return {"result": f"No Slack messages found for query: '{query}'"}

            formatted = []
            for match in matches:
                username = match.get("username") or match.get("user", "unknown")
                channel_name = match.get("channel", {}).get("name", "unknown-channel")
                ts = self._format_ts(match.get("ts", ""))
                text = match.get("text", "").strip()
                formatted.append(f"[{ts}] #{channel_name} — {username}: {text}")

            return {"result": "\n\n".join(formatted)}
        except SlackApiError as e:
            return self.handle_error(e)
        except Exception as e:
            return self.handle_error(e)

    def validate_input(self, **kwargs) -> bool:
        return bool(kwargs.get("query"))


class GetSlackUserInfoTool(SlackBaseTool):
    """Look up a Slack user's profile by their user ID."""

    name = "get_slack_user_info"
    summarizable = False  # User lookups are reference data, not worth summarizing.

    def get_description(self) -> dict:
        return {
            "name": self.name,
            "description": (
                "Returns the display name, real name, and account type (human or bot) "
                "for a Slack user ID. "
                "Use to resolve a user_id from get_channel_messages results into a human-readable name."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "user_id": {
                        "type": "string",
                        "description": "Slack user ID to look up (e.g. 'U0B46KGERRT').",
                    },
                },
                "required": ["user_id"],
            },
        }

    def execute(self, **kwargs) -> dict:
        try:
            user_id = kwargs["user_id"]
            client = self._get_bot_client()
            resp = client.users_info(user=user_id)
            user = resp["user"]
            profile = user.get("profile", {})
            return {
                "result": json.dumps({
                    "user_id": user_id,
                    "display_name": profile.get("display_name") or "",
                    "real_name": profile.get("real_name") or user.get("name") or "",
                    "is_bot": user.get("is_bot", False),
                }, ensure_ascii=False, indent=2)
            }
        except SlackApiError as e:
            return self.handle_error(e)
        except Exception as e:
            return self.handle_error(e)

    def validate_input(self, **kwargs) -> bool:
        return bool(kwargs.get("user_id"))


class DraftSlackMessageTool(SlackBaseTool):
    """
    Format a Slack message draft for user review — does NOT post to Slack.
    The user must explicitly confirm before any message is sent.
    """

    name = "draft_slack_message"
    summarizable = False  # Drafts that are never sent aren't worth summarizing.

    def get_description(self) -> dict:
        return {
            "name": self.name,
            "description": (
                "Formats a Slack message draft and returns it for the user to review. "
                "Does NOT post anything to Slack. "
                "Use this whenever the user asks to send or write a Slack message — "
                "always show the draft first and wait for explicit confirmation before taking any further action."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "channel": {
                        "type": "string",
                        "description": "Target channel name (e.g. 'general' or '#data-science').",
                    },
                    "text": {
                        "type": "string",
                        "description": "The message text to draft.",
                    },
                },
                "required": ["channel", "text"],
            },
        }

    def execute(self, **kwargs) -> dict:
        try:
            channel = kwargs["channel"].lstrip("#")
            text = kwargs["text"]
            preview = (
                f"--- Slack Message Draft ---\n"
                f"Channel : #{channel}\n"
                f"Message : {text}\n"
                f"---------------------------\n"
                f"This message has NOT been sent. Confirm to post it."
            )
            return {"result": preview}
        except Exception as e:
            return self.handle_error(e)

    def validate_input(self, **kwargs) -> bool:
        return bool(kwargs.get("channel")) and bool(kwargs.get("text"))
