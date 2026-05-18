"""
Unit tests for tools/gmail_tool.py

All tests mock the Gmail API — no real credentials or network calls needed.

Run from the project root:
    python -m pytest tests/test_gmail_tool.py -v
"""

import base64
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch, mock_open

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import gmail_tool
from gmail_tool import (
    _parse_message,
    _fetch_messages,
    preview_draft_reply,
    create_draft_reply,
    read_latest_emails,
    search_emails,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_raw_message(subject="Test Subject", from_="sender@example.com",
                      date="Mon, 1 Jan 2024 10:00:00 +0000",
                      body_text="Hello world", msg_id="msg001",
                      snippet="Hello world", mime_type="text/plain"):
    """Build a minimal Gmail API message dict."""
    encoded_body = base64.urlsafe_b64encode(body_text.encode()).decode()
    return {
        "id": msg_id,
        "threadId": "thread001",
        "snippet": snippet,
        "payload": {
            "mimeType": mime_type,
            "headers": [
                {"name": "Subject", "value": subject},
                {"name": "From", "value": from_},
                {"name": "Date", "value": date},
            ],
            "body": {"data": encoded_body},
            "parts": [],
        },
    }


def _make_multipart_message(subject="Multipart", body_text="Plain part"):
    """Build a multipart Gmail message where body is inside parts[]."""
    encoded_body = base64.urlsafe_b64encode(body_text.encode()).decode()
    return {
        "id": "msg002",
        "threadId": "thread002",
        "snippet": body_text[:100],
        "payload": {
            "mimeType": "multipart/alternative",
            "headers": [
                {"name": "Subject", "value": subject},
                {"name": "From", "value": "a@b.com"},
                {"name": "Date", "value": "Tue, 2 Jan 2024 10:00:00 +0000"},
            ],
            "body": {},
            "parts": [
                {"mimeType": "text/plain", "body": {"data": encoded_body}},
                {"mimeType": "text/html", "body": {"data": "ignored"}},
            ],
        },
    }


# ---------------------------------------------------------------------------
# _parse_message
# ---------------------------------------------------------------------------

class TestParseMessage:
    def test_extracts_headers(self):
        msg = _make_raw_message(subject="Hello", from_="a@b.com", date="2024-01-01")
        result = _parse_message(msg)
        assert result["subject"] == "Hello"
        assert result["from"] == "a@b.com"
        assert result["date"] == "2024-01-01"
        assert result["id"] == "msg001"

    def test_extracts_plain_body(self):
        msg = _make_raw_message(body_text="Body content here")
        result = _parse_message(msg)
        assert "Body content here" in result["body_preview"]

    def test_extracts_multipart_body(self):
        msg = _make_multipart_message(body_text="Multipart body text")
        result = _parse_message(msg)
        assert "Multipart body text" in result["body_preview"]

    def test_body_preview_capped_at_500_chars(self):
        long_body = "x" * 1000
        msg = _make_raw_message(body_text=long_body)
        result = _parse_message(msg)
        assert len(result["body_preview"]) == 500

    def test_missing_subject_defaults(self):
        msg = _make_raw_message()
        msg["payload"]["headers"] = []  # strip all headers
        result = _parse_message(msg)
        assert result["subject"] == "(no subject)"
        assert result["from"] == ""
        assert result["date"] == ""

    def test_snippet_included(self):
        msg = _make_raw_message(snippet="Short preview text")
        result = _parse_message(msg)
        assert result["snippet"] == "Short preview text"

    def test_no_body_data_returns_empty_string(self):
        msg = _make_raw_message()
        msg["payload"]["body"] = {}        # no data key
        msg["payload"]["parts"] = []
        result = _parse_message(msg)
        assert result["body_preview"] == ""


# ---------------------------------------------------------------------------
# _fetch_messages
# ---------------------------------------------------------------------------

class TestFetchMessages:
    def test_returns_parsed_list(self):
        raw = _make_raw_message(subject="Fetched", msg_id="abc")
        service = MagicMock()
        service.users().messages().list().execute.return_value = {
            "messages": [{"id": "abc"}]
        }
        service.users().messages().get().execute.return_value = raw

        results = _fetch_messages(service, query="in:inbox", max_results=5)
        assert len(results) == 1
        assert results[0]["subject"] == "Fetched"

    def test_empty_result(self):
        service = MagicMock()
        service.users().messages().list().execute.return_value = {}
        results = _fetch_messages(service, query="nothing", max_results=5)
        assert results == []

    def test_respects_max_results(self):
        service = MagicMock()
        service.users().messages().list().execute.return_value = {
            "messages": [{"id": str(i)} for i in range(3)]
        }
        service.users().messages().get().execute.return_value = _make_raw_message()
        _fetch_messages(service, query="in:inbox", max_results=3)
        assert service.users().messages().list.call_args.kwargs["maxResults"] == 3


# ---------------------------------------------------------------------------
# preview_draft_reply
# ---------------------------------------------------------------------------

class TestPreviewDraftReply:
    def test_returns_json_string(self):
        result = preview_draft_reply("to@example.com", "Re: Hello", "Thanks!")
        data = json.loads(result)
        assert isinstance(data, dict)

    def test_contains_preview_status(self):
        result = preview_draft_reply("to@example.com", "Re: Hello", "Thanks!")
        data = json.loads(result)
        assert "PREVIEW" in data["status"]
        assert "not saved" in data["status"]

    def test_fields_are_present(self):
        result = preview_draft_reply("to@example.com", "Re: Hello", "Thanks!", reply_to_id="msg123")
        data = json.loads(result)
        assert data["to"] == "to@example.com"
        assert data["subject"] == "Re: Hello"
        assert data["body"] == "Thanks!"
        assert data["reply_to_id"] == "msg123"

    def test_no_reply_to_id_shows_new_thread(self):
        result = preview_draft_reply("to@example.com", "Hi", "Body")
        data = json.loads(result)
        assert "new thread" in data["reply_to_id"]

    def test_does_not_call_gmail_api(self):
        with patch.object(gmail_tool, "get_gmail_service") as mock_service:
            preview_draft_reply("to@example.com", "Subject", "Body")
            mock_service.assert_not_called()


# ---------------------------------------------------------------------------
# create_draft_reply
# ---------------------------------------------------------------------------

class TestCreateDraftReply:
    def _make_service(self, draft_id="draft001"):
        service = MagicMock()
        service.users().drafts().create().execute.return_value = {"id": draft_id}
        return service

    def test_returns_json_with_draft_id(self):
        service = self._make_service("draft42")
        with patch.object(gmail_tool, "get_gmail_service", return_value=service):
            result = create_draft_reply("to@example.com", "Subject", "Body")
        data = json.loads(result)
        assert data["draft_id"] == "draft42"
        assert data["status"] == "Draft saved"

    def test_result_includes_to_and_subject(self):
        service = self._make_service()
        with patch.object(gmail_tool, "get_gmail_service", return_value=service):
            result = create_draft_reply("recv@example.com", "My Subject", "Body")
        data = json.loads(result)
        assert data["to"] == "recv@example.com"
        assert data["subject"] == "My Subject"

    def test_calls_drafts_create(self):
        service = self._make_service()
        with patch.object(gmail_tool, "get_gmail_service", return_value=service):
            create_draft_reply("to@example.com", "Subject", "Body")
        # Verify the real API call was made with userId and a body containing raw
        call_kwargs = service.users().drafts().create.call_args.kwargs
        assert call_kwargs["userId"] == "me"
        assert "raw" in call_kwargs["body"]["message"]

    def test_draft_body_contains_raw(self):
        service = self._make_service()
        with patch.object(gmail_tool, "get_gmail_service", return_value=service):
            create_draft_reply("to@example.com", "Subject", "Body")
        call_kwargs = service.users().drafts().create.call_args.kwargs
        assert "raw" in call_kwargs["body"]["message"]

    def test_reply_with_threading(self):
        service = self._make_service()
        service.users().messages().get().execute.return_value = {
            "threadId": "thread99",
            "payload": {
                "headers": [
                    {"name": "Message-ID", "value": "<orig@mail>"},
                    {"name": "References", "value": ""},
                ]
            },
        }
        with patch.object(gmail_tool, "get_gmail_service", return_value=service):
            result = create_draft_reply("to@example.com", "Re: Hi", "Body", reply_to_id="msg001")
        data = json.loads(result)
        assert data["status"] == "Draft saved"

    def test_reply_without_thread_id_still_works(self):
        service = self._make_service()
        with patch.object(gmail_tool, "get_gmail_service", return_value=service):
            result = create_draft_reply("to@example.com", "Subject", "Body", reply_to_id="")
        data = json.loads(result)
        assert data["status"] == "Draft saved"


# ---------------------------------------------------------------------------
# read_latest_emails / search_emails (integration with get_gmail_service)
# ---------------------------------------------------------------------------

class TestReadLatestEmails:
    def test_returns_json_list(self):
        service = MagicMock()
        service.users().messages().list().execute.return_value = {
            "messages": [{"id": "1"}]
        }
        service.users().messages().get().execute.return_value = _make_raw_message(msg_id="1")
        with patch.object(gmail_tool, "get_gmail_service", return_value=service):
            result = read_latest_emails(max_results=1)
        data = json.loads(result)
        assert isinstance(data, list)
        assert data[0]["id"] == "1"

    def test_uses_inbox_query(self):
        service = MagicMock()
        service.users().messages().list().execute.return_value = {}
        with patch.object(gmail_tool, "get_gmail_service", return_value=service):
            read_latest_emails()
        call_kwargs = service.users().messages().list.call_args.kwargs
        assert call_kwargs["q"] == "in:inbox"


class TestSearchEmails:
    def test_passes_query_to_api(self):
        service = MagicMock()
        service.users().messages().list().execute.return_value = {}
        with patch.object(gmail_tool, "get_gmail_service", return_value=service):
            search_emails("from:boss@example.com")
        call_kwargs = service.users().messages().list.call_args.kwargs
        assert call_kwargs["q"] == "from:boss@example.com"

    def test_returns_json_list(self):
        service = MagicMock()
        service.users().messages().list().execute.return_value = {
            "messages": [{"id": "x"}]
        }
        service.users().messages().get().execute.return_value = _make_raw_message(msg_id="x")
        with patch.object(gmail_tool, "get_gmail_service", return_value=service):
            result = search_emails("subject:invoice")
        data = json.loads(result)
        assert isinstance(data, list)
