"""
Google Drive API integration with OAuth2.

Shares credentials.json with Gmail/Calendar but uses a separate token file
(drive_token.json) to avoid scope conflicts.

Supports:
  - Searching files by name/content query
  - Reading text content from Google Docs (native format)
  - Creating new Google Docs (preview-then-create two-step pattern)
  - Other file types (PDF, DOCX, etc.) return a clear unsupported message

Required files (not committed to git):
  credentials.json   — downloaded from Google Cloud Console (OAuth 2.0 Client ID)
  drive_token.json   — created automatically after first authorization

NOTE: If scopes change (e.g. adding drive.file), delete drive_token.json to
force re-authorization on next run.
"""

import json
import os
from pathlib import Path

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

try:
    from tools.base_tool import BaseTool
except ImportError:
    from base_tool import BaseTool


GOOGLE_DOC_MIME = "application/vnd.google-apps.document"


class DriveBaseTool(BaseTool):
    """Shared auth and helpers for all Drive tools."""

    category = "drive"
    summarizable = True
    _SCOPES = [
        "https://www.googleapis.com/auth/drive.readonly",
        "https://www.googleapis.com/auth/drive.file",
    ]
    _CREDENTIALS_FILE = Path(os.getenv("GOOGLE_CREDENTIALS_PATH", "credentials.json"))
    _TOKEN_FILE = Path(os.getenv("DRIVE_TOKEN_FILE", "drive_token.json"))

    @classmethod
    def _get_drive_service(cls):
        """Return an authorized Drive API v3 service client."""
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

        return build("drive", "v3", credentials=creds)

    @classmethod
    def _get_docs_service(cls):
        """Return an authorized Google Docs API v1 service client (reuses Drive credentials)."""
        creds = Credentials.from_authorized_user_file(str(cls._TOKEN_FILE), cls._SCOPES)
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
                cls._TOKEN_FILE.write_text(creds.to_json())
        return build("docs", "v1", credentials=creds)


class SearchDriveTool(DriveBaseTool):
    """Search Google Drive for files matching a query."""

    name = "search_drive"

    def get_description(self) -> dict:
        return {
            "name": self.name,
            "description": (
                "Searches Google Drive for files matching a query. "
                "Returns file names, types, links, and last-modified times. "
                "Use when the user asks to find documents, spreadsheets, or files in Drive."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": (
                            "Search query. Searches file names and content. "
                            "Examples: 'project proposal', 'Q4 planning', 'meeting notes'."
                        ),
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of files to return (default 5, max 20).",
                        "default": 5,
                    },
                },
                "required": ["query"],
            },
        }

    def execute(self, **kwargs) -> dict:
        try:
            query = kwargs["query"]
            max_results = min(kwargs.get("max_results", 5), 20)
            service = self._get_drive_service()

            # fullText search covers both file names and document content
            drive_query = f"fullText contains '{query}' and trashed = false"

            response = service.files().list(
                q=drive_query,
                pageSize=max_results,
                fields="files(id, name, mimeType, webViewLink, modifiedTime)",
            ).execute()

            files = response.get("files", [])
            results = [
                {
                    "id": f["id"],
                    "name": f.get("name", ""),
                    "mimeType": f.get("mimeType", ""),
                    "url": f.get("webViewLink", ""),
                    "modified": f.get("modifiedTime", ""),
                }
                for f in files
            ]
            return {"result": json.dumps(results, ensure_ascii=False, indent=2)}
        except HttpError as e:
            return self.handle_error(e)
        except Exception as e:
            return self.handle_error(e)

    def validate_input(self, **kwargs) -> bool:
        return bool(kwargs.get("query"))


class ReadDriveDocumentTool(DriveBaseTool):
    """Read the text content of a Google Doc by file ID."""

    name = "read_drive_document"

    def get_description(self) -> dict:
        return {
            "name": self.name,
            "description": (
                "Reads the full text content of a Google Doc. "
                "Use after search_drive to read the actual content of a document. "
                "Pass the 'id' field from search_drive results as file_id. "
                "Only works for native Google Docs — other file types (PDFs, DOCX) are not supported."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "file_id": {
                        "type": "string",
                        "description": "Google Drive file ID (from search_drive results).",
                    },
                },
                "required": ["file_id"],
            },
        }

    def execute(self, **kwargs) -> dict:
        try:
            file_id = kwargs["file_id"]
            drive_service = self._get_drive_service()

            # Check file type before attempting to read
            meta = drive_service.files().get(
                fileId=file_id, fields="id, name, mimeType, webViewLink"
            ).execute()

            mime_type = meta.get("mimeType", "")
            if mime_type != GOOGLE_DOC_MIME:
                return {
                    "result": json.dumps({
                        "name": meta.get("name", ""),
                        "mimeType": mime_type,
                        "content": None,
                        "note": (
                            f"Reading '{meta.get('name', '')}' is not supported — "
                            f"only native Google Docs can be read (got: {mime_type}). "
                            "Open it directly in Drive."
                        ),
                    }, indent=2)
                }

            docs_service = self._get_docs_service()
            doc = docs_service.documents().get(documentId=file_id).execute()
            content = self._extract_doc_text(doc)

            return {
                "result": json.dumps({
                    "name": meta.get("name", ""),
                    "url": meta.get("webViewLink", ""),
                    "content": content,
                }, ensure_ascii=False, indent=2)
            }
        except HttpError as e:
            return self.handle_error(e)
        except Exception as e:
            return self.handle_error(e)

    def validate_input(self, **kwargs) -> bool:
        return bool(kwargs.get("file_id"))

    @staticmethod
    def _extract_doc_text(doc: dict) -> str:
        """Extract plain text from a Google Docs API document object."""
        lines = []
        for element in doc.get("body", {}).get("content", []):
            paragraph = element.get("paragraph")
            if not paragraph:
                continue
            line_parts = []
            for pe in paragraph.get("elements", []):
                text_run = pe.get("textRun")
                if text_run:
                    line_parts.append(text_run.get("content", ""))
            line = "".join(line_parts).rstrip("\n")
            if line.strip():
                lines.append(line)
        return "\n".join(lines)


class PreviewDriveDocTool(DriveBaseTool):
    """Format a Google Doc for user review before creation — no API call."""

    name = "preview_drive_doc"

    def get_description(self) -> dict:
        return {
            "name": self.name,
            "description": (
                "Formats a new Google Doc and returns it for the user to review. "
                "ALWAYS call this first and show the result to the user before calling create_drive_doc. "
                "Never skip this step — the user must confirm before any document is created."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "title": {
                        "type": "string",
                        "description": "Title of the Google Doc to create.",
                    },
                    "content": {
                        "type": "string",
                        "description": "Full text content of the document.",
                    },
                },
                "required": ["title", "content"],
            },
        }

    def execute(self, **kwargs) -> dict:
        try:
            preview = {
                "status": "PREVIEW — not created yet",
                "title": kwargs["title"],
                "content": kwargs["content"],
            }
            return {"result": json.dumps(preview, ensure_ascii=False, indent=2)}
        except Exception as e:
            return self.handle_error(e)

    def validate_input(self, **kwargs) -> bool:
        return all(kwargs.get(f) for f in ("title", "content"))


class CreateDriveDocTool(DriveBaseTool):
    """Create a new Google Doc in Drive with the given title and content."""

    name = "create_drive_doc"
    requires_confirmation = True

    def get_confirmation_message(self, **kwargs) -> str:
        """Show doc title before creating."""
        title = kwargs.get("title", "(untitled)")
        return f"Create Google Doc: '{title}'."

    def get_description(self) -> dict:
        return {
            "name": self.name,
            "description": (
                "Creates a new Google Doc in Drive. "
                "Only call this after preview_drive_doc has been shown to the user "
                "and they have explicitly confirmed (e.g. 'yes', 'create it', 'looks good')."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "title": {
                        "type": "string",
                        "description": "Title of the Google Doc.",
                    },
                    "content": {
                        "type": "string",
                        "description": "Full text content of the document.",
                    },
                },
                "required": ["title", "content"],
            },
        }

    def execute(self, **kwargs) -> dict:
        try:
            title = kwargs["title"]
            content = kwargs["content"]

            drive_service = self._get_drive_service()

            # Create an empty Google Doc
            file_meta = {"name": title, "mimeType": GOOGLE_DOC_MIME}
            created = drive_service.files().create(
                body=file_meta,
                fields="id, webViewLink",
            ).execute()

            file_id = created["id"]
            url = created.get("webViewLink", "")

            # Insert content via Docs API
            docs_service = self._get_docs_service()
            docs_service.documents().batchUpdate(
                documentId=file_id,
                body={"requests": [{"insertText": {"location": {"index": 1}, "text": content}}]},
            ).execute()

            return {
                "result": json.dumps({
                    "status": "Document created",
                    "title": title,
                    "file_id": file_id,
                    "url": url,
                }, ensure_ascii=False, indent=2)
            }
        except HttpError as e:
            return self.handle_error(e)
        except Exception as e:
            return self.handle_error(e)

    def validate_input(self, **kwargs) -> bool:
        return all(kwargs.get(f) for f in ("title", "content"))
