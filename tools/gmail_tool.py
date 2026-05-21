"""
Gmail API integration with OAuth2.

OAuth flow:
  1. First run: opens a browser window to authorize the app.
     Google redirects back with an auth code; the library exchanges it for tokens.
  2. Tokens are saved to token.json so subsequent runs skip the browser.
  3. Expired access tokens are refreshed automatically from the refresh token.

Required files (not committed to git):
  credentials.json  — downloaded from Google Cloud Console (OAuth 2.0 Client ID)
  token.json        — created automatically after first authorization
"""

import base64
import email.mime.text
import json
import os
from pathlib import Path

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

try:
    from tools.base_tool import BaseTool
except ImportError:
    from base_tool import BaseTool


class GmailBaseTool(BaseTool):
    """Shared auth and helpers for all Gmail tools."""

    category = "gmail"
    summarizable = True
    _SCOPES = [
        "https://www.googleapis.com/auth/gmail.readonly",
        "https://www.googleapis.com/auth/gmail.compose",
    ]
    _CREDENTIALS_FILE = Path(os.getenv("GOOGLE_CREDENTIALS_PATH", "credentials.json"))
    _TOKEN_FILE = Path(os.getenv("GMAIL_TOKEN_FILE", "token.json"))

    @classmethod
    def _get_gmail_service(cls):
        """Return an authorized Gmail API service client."""
        creds = None

        if cls._TOKEN_FILE.exists():
            creds = Credentials.from_authorized_user_file(str(cls._TOKEN_FILE), cls._SCOPES)

        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                if not cls._CREDENTIALS_FILE.exists():
                    raise FileNotFoundError(
                        f"credentials.json not found at '{cls._CREDENTIALS_FILE}'. "
                        "Download it from Google Cloud Console → APIs & Services → Credentials."
                    )
                flow = InstalledAppFlow.from_client_secrets_file(
                    str(cls._CREDENTIALS_FILE), cls._SCOPES
                )
                creds = flow.run_local_server(port=0)

            cls._TOKEN_FILE.write_text(creds.to_json())

        return build("gmail", "v1", credentials=creds)

    @staticmethod
    def _parse_message(msg: dict) -> dict:
        """Extract subject, from, date, and snippet from a raw Gmail message."""
        headers = {h["name"]: h["value"] for h in msg["payload"].get("headers", [])}
        body = ""
        payload = msg.get("payload", {})

        if payload.get("mimeType") == "text/plain" and payload.get("body", {}).get("data"):
            body = base64.urlsafe_b64decode(payload["body"]["data"]).decode("utf-8", errors="replace")
        else:
            for part in payload.get("parts", []):
                if part.get("mimeType") == "text/plain" and part.get("body", {}).get("data"):
                    body = base64.urlsafe_b64decode(part["body"]["data"]).decode("utf-8", errors="replace")
                    break

        return {
            "id": msg["id"],
            "subject": headers.get("Subject", "(no subject)"),
            "from": headers.get("From", ""),
            "date": headers.get("Date", ""),
            "snippet": msg.get("snippet", ""),
            "body_preview": body[:500],
        }

    @classmethod
    def _fetch_messages(cls, service, query: str, max_results: int) -> list[dict]:
        """Run a Gmail search query and return parsed message dicts."""
        result = service.users().messages().list(
            userId="me", q=query, maxResults=max_results
        ).execute()

        messages = result.get("messages", [])
        parsed = []
        for m in messages:
            full = service.users().messages().get(
                userId="me", id=m["id"], format="full"
            ).execute()
            parsed.append(cls._parse_message(full))
        return parsed


class ReadLatestEmailsTool(GmailBaseTool):
    name = "read_latest_emails"

    def get_description(self) -> dict:
        return {
            "name": self.name,
            "description": (
                "Fetches the N most recent emails from the user's Gmail inbox. "
                "Use when the user asks to check, show, or read their latest/recent emails."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "max_results": {
                        "type": "integer",
                        "description": "Number of emails to return (default 5, max 20).",
                        "default": 5,
                    },
                },
                "required": [],
            },
        }

    def execute(self, **kwargs) -> dict:
        try:
            max_results = kwargs.get("max_results", 5)
            service = self._get_gmail_service()
            emails = self._fetch_messages(service, query="in:inbox", max_results=max_results)
            return {"result": json.dumps(emails, ensure_ascii=False, indent=2)}
        except Exception as e:
            return self.handle_error(e)

    def validate_input(self, **kwargs) -> bool:
        max_results = kwargs.get("max_results", 5)
        return isinstance(max_results, int) and 1 <= max_results <= 20


class SearchEmailsTool(GmailBaseTool):
    name = "search_emails"

    def get_description(self) -> dict:
        return {
            "name": self.name,
            "description": (
                "Searches Gmail using a search query string. Supports Gmail search operators: "
                "from:, to:, subject:, is:unread, has:attachment, after:YYYY/MM/DD, etc. "
                "Use when the user asks to find emails by sender, subject, keyword, or any filter."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": (
                            "Gmail search query, e.g. 'from:boss@example.com', "
                            "'subject:invoice is:unread', 'from:newsletter after:2024/01/01'."
                        ),
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of results to return (default 10).",
                        "default": 10,
                    },
                },
                "required": ["query"],
            },
        }

    def execute(self, **kwargs) -> dict:
        try:
            query = kwargs["query"]
            max_results = kwargs.get("max_results", 10)
            service = self._get_gmail_service()
            emails = self._fetch_messages(service, query=query, max_results=max_results)
            return {"result": json.dumps(emails, ensure_ascii=False, indent=2)}
        except Exception as e:
            return self.handle_error(e)

    def validate_input(self, **kwargs) -> bool:
        return bool(kwargs.get("query"))


class PreviewDraftReplyTool(GmailBaseTool):
    name = "preview_draft_reply"

    def get_description(self) -> dict:
        return {
            "name": self.name,
            "description": (
                "Formats a draft reply and returns it for the user to review. "
                "ALWAYS call this first and show the result to the user before calling create_draft_reply. "
                "Never skip this step — the user must confirm before any draft is saved."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "to": {"type": "string", "description": "Recipient email address."},
                    "subject": {"type": "string", "description": "Email subject line."},
                    "body": {"type": "string", "description": "Full email body text."},
                    "reply_to_id": {
                        "type": "string",
                        "description": "Gmail message ID of the email being replied to (for threading). Leave empty for a new thread.",
                    },
                },
                "required": ["to", "subject", "body"],
            },
        }

    def execute(self, **kwargs) -> dict:
        try:
            preview = {
                "status": "PREVIEW — not saved yet",
                "to": kwargs["to"],
                "subject": kwargs["subject"],
                "body": kwargs["body"],
                "reply_to_id": kwargs.get("reply_to_id") or "(new thread)",
            }
            return {"result": json.dumps(preview, ensure_ascii=False, indent=2)}
        except Exception as e:
            return self.handle_error(e)

    def validate_input(self, **kwargs) -> bool:
        return all(kwargs.get(f) for f in ("to", "subject", "body"))


class CreateDraftReplyTool(GmailBaseTool):
    name = "create_draft_reply"

    def get_description(self) -> dict:
        return {
            "name": self.name,
            "description": (
                "Saves a draft reply in Gmail. "
                "Only call this after preview_draft_reply has been shown to the user "
                "and they have explicitly confirmed (e.g. 'yes', 'looks good', 'save it')."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "to": {"type": "string", "description": "Recipient email address."},
                    "subject": {"type": "string", "description": "Email subject line."},
                    "body": {"type": "string", "description": "Full email body text."},
                    "reply_to_id": {
                        "type": "string",
                        "description": "Gmail message ID of the email being replied to (for threading). Leave empty for a new thread.",
                    },
                },
                "required": ["to", "subject", "body"],
            },
        }

    def execute(self, **kwargs) -> dict:
        try:
            to = kwargs["to"]
            subject = kwargs["subject"]
            body = kwargs["body"]
            reply_to_id = kwargs.get("reply_to_id", "")

            service = self._get_gmail_service()

            mime_msg = email.mime.text.MIMEText(body)
            mime_msg["To"] = to
            mime_msg["Subject"] = subject

            if reply_to_id:
                original = service.users().messages().get(
                    userId="me", id=reply_to_id, format="metadata",
                    metadataHeaders=["Message-ID", "References"]
                ).execute()
                orig_headers = {
                    h["name"]: h["value"]
                    for h in original.get("payload", {}).get("headers", [])
                }
                message_id = orig_headers.get("Message-ID", "")
                references = orig_headers.get("References", "")
                if message_id:
                    mime_msg["In-Reply-To"] = message_id
                    mime_msg["References"] = f"{references} {message_id}".strip()
                    mime_msg["threadId"] = original.get("threadId", "")

            raw = base64.urlsafe_b64encode(mime_msg.as_bytes()).decode("utf-8")
            draft_body: dict = {"message": {"raw": raw}}
            if reply_to_id:
                draft_body["message"]["threadId"] = original.get("threadId", "")

            draft = service.users().drafts().create(userId="me", body=draft_body).execute()
            return {"result": json.dumps({
                "status": "Draft saved",
                "draft_id": draft["id"],
                "to": to,
                "subject": subject,
            }, indent=2)}
        except Exception as e:
            return self.handle_error(e)

    def validate_input(self, **kwargs) -> bool:
        return all(kwargs.get(f) for f in ("to", "subject", "body"))
