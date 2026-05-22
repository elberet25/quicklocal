"""
Unit tests for tools/unified_search_tool.py

All sub-tool calls are mocked — no real API calls or RAG index needed.

Run from the project root:
    pytest tests/test_unified_search_tool.py -v
"""

import json
from unittest.mock import patch, MagicMock

import pytest
from tools.unified_search_tool import UnifiedSearchTool


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _notion_raw(titles=("Meeting Notes", "Project Plan")):
    pages = [{"title": t, "url": f"https://notion.so/{i}", "last_edited": "2024-01-01", "id": str(i)}
             for i, t in enumerate(titles)]
    return {"result": json.dumps(pages)}


def _drive_raw(names=("Q4 Plan.docx",)):
    files = [{"name": n, "url": f"https://drive.google.com/{i}",
              "mimeType": "application/vnd.google-apps.document",
              "modified": "2024-01-01", "id": str(i)}
             for i, n in enumerate(names)]
    return {"result": json.dumps(files)}


def _rag_raw(content="Local doc content about the query"):
    return {"result": content}


# ---------------------------------------------------------------------------
# UnifiedSearchTool
# ---------------------------------------------------------------------------

class TestUnifiedSearchTool:
    def _tool(self):
        return UnifiedSearchTool()

    def test_validate_input_requires_query(self):
        tool = self._tool()
        assert tool.validate_input(query="anything") is True
        assert tool.validate_input() is False
        assert tool.validate_input(query="") is False

    def test_merges_results_from_all_sources(self):
        with patch("tools.unified_search_tool.SearchDocumentsTool.execute", return_value=_rag_raw()), \
             patch("tools.unified_search_tool.SearchNotionTool.execute", return_value=_notion_raw()), \
             patch("tools.unified_search_tool.SearchDriveTool.execute", return_value=_drive_raw()):
            result = self._tool().execute(query="planning")

        assert "error" not in result
        data = json.loads(result["result"])
        results = data["results"]

        sources = {r["source"] for r in results}
        assert "local_files" in sources
        assert "notion" in sources
        assert "drive" in sources

    def test_notion_results_have_expected_fields(self):
        with patch("tools.unified_search_tool.SearchDocumentsTool.execute", return_value={"result": ""}), \
             patch("tools.unified_search_tool.SearchNotionTool.execute", return_value=_notion_raw(("My Page",))), \
             patch("tools.unified_search_tool.SearchDriveTool.execute", return_value=_drive_raw([])):
            result = self._tool().execute(query="test")

        data = json.loads(result["result"])
        notion_results = [r for r in data["results"] if r["source"] == "notion"]
        assert len(notion_results) == 1
        assert notion_results[0]["title"] == "My Page"
        assert "url" in notion_results[0]
        assert "id" in notion_results[0]

    def test_drive_results_have_expected_fields(self):
        with patch("tools.unified_search_tool.SearchDocumentsTool.execute", return_value={"result": ""}), \
             patch("tools.unified_search_tool.SearchNotionTool.execute", return_value=_notion_raw([])), \
             patch("tools.unified_search_tool.SearchDriveTool.execute", return_value=_drive_raw(("Report.docx",))):
            result = self._tool().execute(query="test")

        data = json.loads(result["result"])
        drive_results = [r for r in data["results"] if r["source"] == "drive"]
        assert len(drive_results) == 1
        assert drive_results[0]["name"] == "Report.docx"
        assert "url" in drive_results[0]
        assert "id" in drive_results[0]

    def test_skips_failed_source_and_adds_warning(self):
        with patch("tools.unified_search_tool.SearchDocumentsTool.execute", return_value=_rag_raw()), \
             patch("tools.unified_search_tool.SearchNotionTool.execute", side_effect=Exception("Notion down")), \
             patch("tools.unified_search_tool.SearchDriveTool.execute", return_value=_drive_raw()):
            result = self._tool().execute(query="test")

        assert "error" not in result
        data = json.loads(result["result"])
        sources = {r["source"] for r in data["results"]}
        assert "notion" not in sources
        assert "local_files" in sources or "drive" in sources
        assert any("notion" in w for w in data.get("warnings", []))

    def test_returns_no_warnings_when_all_sources_succeed(self):
        with patch("tools.unified_search_tool.SearchDocumentsTool.execute", return_value=_rag_raw()), \
             patch("tools.unified_search_tool.SearchNotionTool.execute", return_value=_notion_raw()), \
             patch("tools.unified_search_tool.SearchDriveTool.execute", return_value=_drive_raw()):
            result = self._tool().execute(query="test")

        data = json.loads(result["result"])
        assert "warnings" not in data

    def test_skips_empty_rag_results(self):
        with patch("tools.unified_search_tool.SearchDocumentsTool.execute",
                   return_value={"result": "No documents indexed yet. Use the index_documents tool first."}), \
             patch("tools.unified_search_tool.SearchNotionTool.execute", return_value=_notion_raw()), \
             patch("tools.unified_search_tool.SearchDriveTool.execute", return_value=_drive_raw()):
            result = self._tool().execute(query="test")

        data = json.loads(result["result"])
        local_results = [r for r in data["results"] if r["source"] == "local_files"]
        assert local_results == []

    def test_all_sources_fail_returns_empty_results_with_warnings(self):
        with patch("tools.unified_search_tool.SearchDocumentsTool.execute", side_effect=Exception("RAG error")), \
             patch("tools.unified_search_tool.SearchNotionTool.execute", side_effect=Exception("Notion error")), \
             patch("tools.unified_search_tool.SearchDriveTool.execute", side_effect=Exception("Drive error")):
            result = self._tool().execute(query="test")

        assert "error" not in result
        data = json.loads(result["result"])
        assert data["results"] == []
        assert len(data["warnings"]) == 3

    def test_description_has_required_fields(self):
        desc = self._tool().get_description()
        assert desc["name"] == "search_all_knowledge"
        assert "query" in desc["input_schema"]["properties"]
        assert "query" in desc["input_schema"]["required"]

    def test_summarizable_is_true(self):
        assert UnifiedSearchTool.summarizable is True

    def test_category_is_knowledge(self):
        assert UnifiedSearchTool.category == "knowledge"
