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

SCOPES = [
    "https://www.googleapis.com/auth/gmail.readonly",
    "https://www.googleapis.com/auth/gmail.compose",
]

CREDENTIALS_FILE = Path(os.getenv("GOOGLE_CREDENTIALS_PATH", "credentials.json"))
TOKEN_FILE = Path(os.getenv("GMAIL_TOKEN_FILE", "token.json"))


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------

def get_gmail_service():
    """Return an authorized Gmail API service client."""
    creds = None

    if TOKEN_FILE.exists():
        creds = Credentials.from_authorized_user_file(str(TOKEN_FILE), SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            if not CREDENTIALS_FILE.exists():
                raise FileNotFoundError(
                    f"credentials.json not found at '{CREDENTIALS_FILE}'. "
                    "Download it from Google Cloud Console → APIs & Services → Credentials."
                )
            flow = InstalledAppFlow.from_client_secrets_file(
                str(CREDENTIALS_FILE), SCOPES
            )
            # Opens a local browser tab for the user to approve access.
            creds = flow.run_local_server(port=0)

        TOKEN_FILE.write_text(creds.to_json())

    return build("gmail", "v1", credentials=creds)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_message(msg: dict) -> dict:
    """Extract subject, from, date, and snippet from a raw Gmail message."""
    headers = {h["name"]: h["value"] for h in msg["payload"].get("headers", [])}
    body = ""
    payload = msg.get("payload", {})

    # Try to get the plain-text body.
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


def _fetch_messages(service, query: str, max_results: int) -> list[dict]:
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
        parsed.append(_parse_message(full))
    return parsed


# ---------------------------------------------------------------------------
# Tool functions (called by the agent)
# ---------------------------------------------------------------------------

def read_latest_emails(max_results: int = 5) -> str:
    """Return the N most recent emails from the inbox as a JSON string."""
    service = get_gmail_service()
    emails = _fetch_messages(service, query="in:inbox", max_results=max_results)
    return json.dumps(emails, ensure_ascii=False, indent=2)


def search_emails(query: str, max_results: int = 10) -> str:
    """
    Search emails using a Gmail search query and return results as JSON.

    Examples:
      query="from:boss@example.com"
      query="subject:invoice"
      query="from:newsletter@example.com subject:weekly"
    """
    service = get_gmail_service()
    emails = _fetch_messages(service, query=query, max_results=max_results)
    return json.dumps(emails, ensure_ascii=False, indent=2)


def preview_draft_reply(to: str, subject: str, body: str, reply_to_id: str = "") -> str:
    """
    Format a draft reply for the user to review — does NOT touch Gmail.
    Always call this before create_draft_reply and show the result to the user.
    Ask for explicit confirmation before proceeding to create_draft_reply.
    """
    preview = {
        "status": "PREVIEW — not saved yet",
        "to": to,
        "subject": subject,
        "body": body,
        "reply_to_id": reply_to_id or "(new thread)",
    }
    return json.dumps(preview, ensure_ascii=False, indent=2)


def create_draft_reply(to: str, subject: str, body: str, reply_to_id: str = "") -> str:
    """
    Save a draft reply in Gmail. Only call this after preview_draft_reply has been
    shown to the user and they have explicitly confirmed they want to save it.
    """
    service = get_gmail_service()

    mime_msg = email.mime.text.MIMEText(body)
    mime_msg["To"] = to
    mime_msg["Subject"] = subject

    if reply_to_id:
        # Fetch the original message to get threading headers.
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
    return json.dumps({
        "status": "Draft saved",
        "draft_id": draft["id"],
        "to": to,
        "subject": subject,
    }, indent=2)
