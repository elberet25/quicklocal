"""
Unit tests for tools/drive_tool.py

All tests mock the Google Drive and Docs APIs — no real credentials needed.

Run from the project root:
    pytest tests/test_drive_tool.py -v
"""

import json
from unittest.mock import MagicMock, patch, PropertyMock

import pytest
from tools.drive_tool import (
    DriveBaseTool,
    SearchDriveTool,
    ReadDriveDocumentTool,
    GOOGLE_DOC_MIME,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_file(file_id="file-001", name="My Doc", mime=GOOGLE_DOC_MIME,
               url="https://docs.google.com/document/d/file-001",
               modified="2024-01-15T10:00:00.000Z"):
    return {
        "id": file_id,
        "name": name,
        "mimeType": mime,
        "webViewLink": url,
        "modifiedTime": modified,
    }


def _make_doc(paragraphs: list[str]) -> dict:
    """Build a minimal Google Docs API document object."""
    content = []
    for text in paragraphs:
        content.append({
            "paragraph": {
                "elements": [
                    {"textRun": {"content": text + "\n"}}
                ]
            }
        })
    return {"body": {"content": content}}


# ---------------------------------------------------------------------------
# ReadDriveDocumentTool._extract_doc_text
# ---------------------------------------------------------------------------

class TestExtractDocText:
    def test_extracts_paragraph_text(self):
        doc = _make_doc(["Hello world", "Second paragraph"])
        result = ReadDriveDocumentTool._extract_doc_text(doc)
        assert "Hello world" in result
        assert "Second paragraph" in result

    def test_joins_paragraphs_with_newlines(self):
        doc = _make_doc(["Line one", "Line two"])
        result = ReadDriveDocumentTool._extract_doc_text(doc)
        assert result == "Line one\nLine two"

    def test_skips_empty_paragraphs(self):
        doc = _make_doc(["Real content", "   ", "More content"])
        result = ReadDriveDocumentTool._extract_doc_text(doc)
        assert "Real content" in result
        assert "More content" in result
        # whitespace-only line should not appear as a blank line
        assert "\n\n" not in result

    def test_skips_non_paragraph_elements(self):
        doc = {"body": {"content": [{"sectionBreak": {}}]}}
        result = ReadDriveDocumentTool._extract_doc_text(doc)
        assert result == ""

    def test_concatenates_multiple_text_runs_in_one_paragraph(self):
        doc = {
            "body": {
                "content": [{
                    "paragraph": {
                        "elements": [
                            {"textRun": {"content": "Hello "}},
                            {"textRun": {"content": "world"}},
                        ]
                    }
                }]
            }
        }
        result = ReadDriveDocumentTool._extract_doc_text(doc)
        assert result == "Hello world"


# ---------------------------------------------------------------------------
# SearchDriveTool
# ---------------------------------------------------------------------------

class TestSearchDriveTool:
    def _tool(self):
        return SearchDriveTool()

    def test_validate_input_requires_query(self):
        tool = self._tool()
        assert tool.validate_input(query="meeting notes") is True
        assert tool.validate_input() is False
        assert tool.validate_input(query="") is False

    def test_returns_formatted_results(self):
        mock_service = MagicMock()
        mock_service.files().list().execute.return_value = {
            "files": [
                _make_file("f1", "Q4 Plan", GOOGLE_DOC_MIME),
                _make_file("f2", "Budget.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"),
            ]
        }
        with patch.object(DriveBaseTool, "_get_drive_service", return_value=mock_service):
            result = self._tool().execute(query="Q4")

        assert "error" not in result
        data = json.loads(result["result"])
        assert len(data) == 2
        assert data[0]["name"] == "Q4 Plan"
        assert data[0]["id"] == "f1"
        assert data[1]["name"] == "Budget.xlsx"

    def test_caps_max_results_at_20(self):
        mock_service = MagicMock()
        mock_service.files().list().execute.return_value = {"files": []}
        with patch.object(DriveBaseTool, "_get_drive_service", return_value=mock_service):
            self._tool().execute(query="test", max_results=999)

        call_kwargs = mock_service.files().list.call_args.kwargs
        assert call_kwargs["pageSize"] == 20

    def test_returns_empty_list_when_no_results(self):
        mock_service = MagicMock()
        mock_service.files().list().execute.return_value = {"files": []}
        with patch.object(DriveBaseTool, "_get_drive_service", return_value=mock_service):
            result = self._tool().execute(query="nonexistent")

        data = json.loads(result["result"])
        assert data == []

    def test_returns_error_on_api_failure(self):
        mock_service = MagicMock()
        mock_service.files().list().execute.side_effect = Exception("Drive API error")
        with patch.object(DriveBaseTool, "_get_drive_service", return_value=mock_service):
            result = self._tool().execute(query="test")

        assert "error" in result
        assert "Drive API error" in result["error"]

    def test_description_has_required_fields(self):
        desc = self._tool().get_description()
        assert desc["name"] == "search_drive"
        assert "query" in desc["input_schema"]["properties"]
        assert "query" in desc["input_schema"]["required"]

    def test_summarizable_is_true(self):
        assert SearchDriveTool.summarizable is True

    def test_category_is_drive(self):
        assert SearchDriveTool.category == "drive"


# ---------------------------------------------------------------------------
# ReadDriveDocumentTool
# ---------------------------------------------------------------------------

class TestReadDriveDocumentTool:
    def _tool(self):
        return ReadDriveDocumentTool()

    def test_validate_input_requires_file_id(self):
        tool = self._tool()
        assert tool.validate_input(file_id="abc-123") is True
        assert tool.validate_input() is False
        assert tool.validate_input(file_id="") is False

    def test_reads_google_doc_content(self):
        mock_drive = MagicMock()
        mock_drive.files().get().execute.return_value = _make_file(
            "f1", "My Doc", GOOGLE_DOC_MIME
        )
        mock_docs = MagicMock()
        mock_docs.documents().get().execute.return_value = _make_doc(
            ["Introduction", "Main content here"]
        )
        with patch.object(DriveBaseTool, "_get_drive_service", return_value=mock_drive), \
             patch.object(DriveBaseTool, "_get_docs_service", return_value=mock_docs):
            result = self._tool().execute(file_id="f1")

        assert "error" not in result
        data = json.loads(result["result"])
        assert data["name"] == "My Doc"
        assert "Introduction" in data["content"]
        assert "Main content here" in data["content"]

    def test_returns_unsupported_note_for_non_google_doc(self):
        mock_drive = MagicMock()
        mock_drive.files().get().execute.return_value = _make_file(
            "f2", "Report.pdf", "application/pdf"
        )
        with patch.object(DriveBaseTool, "_get_drive_service", return_value=mock_drive):
            result = self._tool().execute(file_id="f2")

        assert "error" not in result
        data = json.loads(result["result"])
        assert data["content"] is None
        assert "not supported" in data["note"]
        assert "application/pdf" in data["note"]

    def test_returns_error_on_api_failure(self):
        mock_drive = MagicMock()
        mock_drive.files().get().execute.side_effect = Exception("Not found")
        with patch.object(DriveBaseTool, "_get_drive_service", return_value=mock_drive):
            result = self._tool().execute(file_id="bad-id")

        assert "error" in result
        assert "Not found" in result["error"]

    def test_description_has_required_fields(self):
        desc = self._tool().get_description()
        assert desc["name"] == "read_drive_document"
        assert "file_id" in desc["input_schema"]["properties"]
        assert "file_id" in desc["input_schema"]["required"]

    def test_summarizable_is_true(self):
        assert ReadDriveDocumentTool.summarizable is True

    def test_category_is_drive(self):
        assert ReadDriveDocumentTool.category == "drive"
