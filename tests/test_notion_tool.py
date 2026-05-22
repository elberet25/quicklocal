"""
Unit tests for tools/notion_tool.py

All tests mock the Notion API client — no real credentials or network calls needed.

Run from the project root:
    pytest tests/test_notion_tool.py -v
"""

import json
from unittest.mock import MagicMock, patch

import pytest
from tools.notion_tool import (
    NotionBaseTool,
    SearchNotionTool,
    GetNotionPageTool,
    CreateNotionPageTool,
)


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

def _make_page(page_id="page-001", title="My Page", url="https://notion.so/page-001",
               last_edited="2024-01-15T10:00:00.000Z"):
    """Build a minimal Notion page dict as returned by the search or retrieve API."""
    return {
        "id": page_id,
        "object": "page",
        "url": url,
        "last_edited_time": last_edited,
        "properties": {
            "title": {
                "type": "title",
                "title": [{"plain_text": title}],
            }
        },
    }


def _make_block(btype="paragraph", text="Hello world"):
    """Build a minimal Notion block dict."""
    return {
        "id": "block-001",
        "type": btype,
        btype: {
            "rich_text": [{"plain_text": text}]
        },
    }


# ---------------------------------------------------------------------------
# NotionBaseTool._extract_title
# ---------------------------------------------------------------------------

class TestExtractTitle:
    def test_extracts_title_from_title_property(self):
        page = _make_page(title="Project Plan")
        assert NotionBaseTool._extract_title(page) == "Project Plan"

    def test_concatenates_multiple_rich_text_parts(self):
        page = {
            "id": "p1",
            "properties": {
                "title": {
                    "type": "title",
                    "title": [
                        {"plain_text": "Part One"},
                        {"plain_text": " Part Two"},
                    ],
                }
            },
        }
        assert NotionBaseTool._extract_title(page) == "Part One Part Two"

    def test_falls_back_to_untitled_when_no_title(self):
        page = {"id": "p1", "properties": {}}
        assert NotionBaseTool._extract_title(page) == "(untitled)"

    def test_falls_back_to_child_page_title(self):
        page = {
            "id": "p1",
            "properties": {},
            "child_page": {"title": "Child Page Title"},
        }
        assert NotionBaseTool._extract_title(page) == "Child Page Title"


# ---------------------------------------------------------------------------
# NotionBaseTool._blocks_to_text
# ---------------------------------------------------------------------------

class TestBlocksToText:
    def test_extracts_paragraph_text(self):
        blocks = [_make_block("paragraph", "Hello world")]
        assert NotionBaseTool._blocks_to_text(blocks) == "Hello world"

    def test_extracts_heading_text(self):
        blocks = [_make_block("heading_1", "Chapter 1")]
        assert NotionBaseTool._blocks_to_text(blocks) == "Chapter 1"

    def test_extracts_bullet_items(self):
        blocks = [
            _make_block("bulleted_list_item", "Item A"),
            _make_block("bulleted_list_item", "Item B"),
        ]
        result = NotionBaseTool._blocks_to_text(blocks)
        assert "Item A" in result
        assert "Item B" in result

    def test_skips_empty_rich_text(self):
        block = {"id": "b1", "type": "paragraph", "paragraph": {"rich_text": []}}
        result = NotionBaseTool._blocks_to_text([block])
        assert result == ""

    def test_skips_unknown_block_types(self):
        block = {"id": "b1", "type": "image", "image": {}}
        result = NotionBaseTool._blocks_to_text([block])
        assert result == ""

    def test_includes_child_page_marker(self):
        block = {
            "id": "b1",
            "type": "child_page",
            "child_page": {"title": "Sub Page"},
        }
        result = NotionBaseTool._blocks_to_text([block])
        assert "Sub Page" in result

    def test_multiple_blocks_joined_by_newline(self):
        blocks = [
            _make_block("paragraph", "First paragraph"),
            _make_block("paragraph", "Second paragraph"),
        ]
        result = NotionBaseTool._blocks_to_text(blocks)
        assert result == "First paragraph\nSecond paragraph"


# ---------------------------------------------------------------------------
# NotionBaseTool._get_client
# ---------------------------------------------------------------------------

class TestGetClient:
    def test_raises_when_token_missing(self, monkeypatch):
        monkeypatch.delenv("NOTION_TOKEN", raising=False)
        with pytest.raises(EnvironmentError, match="NOTION_TOKEN"):
            NotionBaseTool._get_client()

    def test_returns_client_when_token_present(self, monkeypatch):
        monkeypatch.setenv("NOTION_TOKEN", "test-token-abc")
        client = NotionBaseTool._get_client()
        assert client is not None


# ---------------------------------------------------------------------------
# SearchNotionTool
# ---------------------------------------------------------------------------

class TestSearchNotionTool:
    def _tool(self):
        return SearchNotionTool()

    def test_validate_input_requires_query(self):
        tool = self._tool()
        assert tool.validate_input(query="meeting notes") is True
        assert tool.validate_input() is False
        assert tool.validate_input(query="") is False

    def test_returns_formatted_results(self, monkeypatch):
        monkeypatch.setenv("NOTION_TOKEN", "test-token")
        mock_client = MagicMock()
        mock_client.search.return_value = {
            "results": [
                _make_page("p1", "Meeting Notes", "https://notion.so/p1"),
                _make_page("p2", "Project Plan", "https://notion.so/p2"),
            ]
        }
        with patch("tools.notion_tool.Client", return_value=mock_client):
            result = self._tool().execute(query="meeting")

        assert "error" not in result
        data = json.loads(result["result"])
        assert len(data) == 2
        assert data[0]["title"] == "Meeting Notes"
        assert data[0]["url"] == "https://notion.so/p1"
        assert data[1]["title"] == "Project Plan"

    def test_caps_max_results_at_20(self, monkeypatch):
        monkeypatch.setenv("NOTION_TOKEN", "test-token")
        mock_client = MagicMock()
        mock_client.search.return_value = {"results": []}
        with patch("tools.notion_tool.Client", return_value=mock_client):
            self._tool().execute(query="test", max_results=999)
        _, call_kwargs = mock_client.search.call_args
        assert call_kwargs["page_size"] == 20

    def test_returns_empty_list_when_no_results(self, monkeypatch):
        monkeypatch.setenv("NOTION_TOKEN", "test-token")
        mock_client = MagicMock()
        mock_client.search.return_value = {"results": []}
        with patch("tools.notion_tool.Client", return_value=mock_client):
            result = self._tool().execute(query="nonexistent")

        data = json.loads(result["result"])
        assert data == []

    def test_returns_error_on_api_failure(self, monkeypatch):
        monkeypatch.setenv("NOTION_TOKEN", "test-token")
        mock_client = MagicMock()
        mock_client.search.side_effect = Exception("API error")
        with patch("tools.notion_tool.Client", return_value=mock_client):
            result = self._tool().execute(query="test")

        assert "error" in result
        assert "API error" in result["error"]

    def test_description_has_required_fields(self):
        desc = self._tool().get_description()
        assert desc["name"] == "search_notion"
        assert "query" in desc["input_schema"]["properties"]
        assert "query" in desc["input_schema"]["required"]

    def test_summarizable_is_true(self):
        assert SearchNotionTool.summarizable is True

    def test_category_is_notion(self):
        assert SearchNotionTool.category == "notion"


# ---------------------------------------------------------------------------
# GetNotionPageTool
# ---------------------------------------------------------------------------

class TestGetNotionPageTool:
    def _tool(self):
        return GetNotionPageTool()

    def test_validate_input_requires_page_id(self):
        tool = self._tool()
        assert tool.validate_input(page_id="abc-123") is True
        assert tool.validate_input() is False
        assert tool.validate_input(page_id="") is False

    def test_returns_title_url_and_content(self, monkeypatch):
        monkeypatch.setenv("NOTION_TOKEN", "test-token")
        mock_client = MagicMock()
        mock_client.pages.retrieve.return_value = _make_page("p1", "My Doc", "https://notion.so/p1")
        mock_client.blocks.children.list.return_value = {
            "results": [
                _make_block("paragraph", "First paragraph"),
                _make_block("heading_2", "A heading"),
            ],
            "has_more": False,
        }
        with patch("tools.notion_tool.Client", return_value=mock_client):
            result = self._tool().execute(page_id="p1")

        assert "error" not in result
        data = json.loads(result["result"])
        assert data["title"] == "My Doc"
        assert data["url"] == "https://notion.so/p1"
        assert "First paragraph" in data["content"]
        assert "A heading" in data["content"]

    def test_handles_pagination(self, monkeypatch):
        monkeypatch.setenv("NOTION_TOKEN", "test-token")
        mock_client = MagicMock()
        mock_client.pages.retrieve.return_value = _make_page()
        mock_client.blocks.children.list.side_effect = [
            {
                "results": [_make_block("paragraph", "Page 1 text")],
                "has_more": True,
                "next_cursor": "cursor-abc",
            },
            {
                "results": [_make_block("paragraph", "Page 2 text")],
                "has_more": False,
            },
        ]
        with patch("tools.notion_tool.Client", return_value=mock_client):
            result = self._tool().execute(page_id="p1")

        data = json.loads(result["result"])
        assert "Page 1 text" in data["content"]
        assert "Page 2 text" in data["content"]
        assert mock_client.blocks.children.list.call_count == 2

    def test_returns_error_on_api_failure(self, monkeypatch):
        monkeypatch.setenv("NOTION_TOKEN", "test-token")
        mock_client = MagicMock()
        mock_client.pages.retrieve.side_effect = Exception("Not found")
        with patch("tools.notion_tool.Client", return_value=mock_client):
            result = self._tool().execute(page_id="bad-id")

        assert "error" in result
        assert "Not found" in result["error"]

    def test_description_has_required_fields(self):
        desc = self._tool().get_description()
        assert desc["name"] == "get_notion_page"
        assert "page_id" in desc["input_schema"]["properties"]
        assert "page_id" in desc["input_schema"]["required"]

    def test_summarizable_is_true(self):
        assert GetNotionPageTool.summarizable is True

    def test_category_is_notion(self):
        assert GetNotionPageTool.category == "notion"


# ---------------------------------------------------------------------------
# CreateNotionPageTool._text_to_blocks
# ---------------------------------------------------------------------------

class TestTextToBlocks:
    def test_converts_single_paragraph(self):
        blocks = CreateNotionPageTool._text_to_blocks("Hello world")
        assert len(blocks) == 1
        assert blocks[0]["type"] == "paragraph"
        assert blocks[0]["paragraph"]["rich_text"][0]["text"]["content"] == "Hello world"

    def test_splits_on_blank_lines(self):
        blocks = CreateNotionPageTool._text_to_blocks("First para\n\nSecond para")
        assert len(blocks) == 2
        assert blocks[0]["paragraph"]["rich_text"][0]["text"]["content"] == "First para"
        assert blocks[1]["paragraph"]["rich_text"][0]["text"]["content"] == "Second para"

    def test_skips_empty_paragraphs(self):
        blocks = CreateNotionPageTool._text_to_blocks("Real\n\n\n\nContent")
        assert len(blocks) == 2

    def test_returns_empty_list_for_blank_input(self):
        blocks = CreateNotionPageTool._text_to_blocks("   \n\n   ")
        assert blocks == []


# ---------------------------------------------------------------------------
# CreateNotionPageTool
# ---------------------------------------------------------------------------

class TestCreateNotionPageTool:
    def _tool(self):
        return CreateNotionPageTool()

    def test_validate_input_requires_title_and_content(self):
        tool = self._tool()
        assert tool.validate_input(title="T", content="C") is True
        assert tool.validate_input(title="T") is False
        assert tool.validate_input(content="C") is False
        assert tool.validate_input() is False

    def test_creates_page_with_parent(self, monkeypatch):
        monkeypatch.setenv("NOTION_TOKEN", "test-token")
        mock_client = MagicMock()
        mock_client.pages.create.return_value = {
            "id": "new-page-123",
            "url": "https://notion.so/new-page-123",
        }
        with patch("tools.notion_tool.Client", return_value=mock_client):
            result = self._tool().execute(
                title="Meeting Summary",
                content="We discussed the project.",
                parent_page_id="parent-456",
            )

        assert "error" not in result
        data = json.loads(result["result"])
        assert data["status"] == "Page created"
        assert data["page_id"] == "new-page-123"
        assert data["title"] == "Meeting Summary"

        call_kwargs = mock_client.pages.create.call_args.kwargs
        assert call_kwargs["parent"] == {"type": "page_id", "page_id": "parent-456"}

    def test_creates_page_at_workspace_root_when_no_parent(self, monkeypatch):
        monkeypatch.setenv("NOTION_TOKEN", "test-token")
        mock_client = MagicMock()
        mock_client.pages.create.return_value = {
            "id": "root-page-789",
            "url": "https://notion.so/root-page-789",
        }
        with patch("tools.notion_tool.Client", return_value=mock_client):
            result = self._tool().execute(title="New Page", content="Content here")

        assert "error" not in result
        call_kwargs = mock_client.pages.create.call_args.kwargs
        assert call_kwargs["parent"] == {"type": "workspace", "workspace": True}

    def test_content_converted_to_blocks(self, monkeypatch):
        monkeypatch.setenv("NOTION_TOKEN", "test-token")
        mock_client = MagicMock()
        mock_client.pages.create.return_value = {"id": "p1", "url": ""}
        with patch("tools.notion_tool.Client", return_value=mock_client):
            self._tool().execute(
                title="T",
                content="First para\n\nSecond para",
                parent_page_id="parent-1",
            )

        call_kwargs = mock_client.pages.create.call_args.kwargs
        children = call_kwargs["children"]
        assert len(children) == 2
        assert children[0]["type"] == "paragraph"
        assert children[1]["type"] == "paragraph"

    def test_returns_error_on_api_failure(self, monkeypatch):
        monkeypatch.setenv("NOTION_TOKEN", "test-token")
        mock_client = MagicMock()
        mock_client.pages.create.side_effect = Exception("API error")
        with patch("tools.notion_tool.Client", return_value=mock_client):
            result = self._tool().execute(title="T", content="C", parent_page_id="p1")

        assert "error" in result

    def test_description_name_is_correct(self):
        assert self._tool().get_description()["name"] == "create_notion_page"

    def test_description_mentions_confirmation(self):
        desc = self._tool().get_description()["description"]
        assert "confirm" in desc.lower()

    def test_summarizable_is_true(self):
        assert CreateNotionPageTool.summarizable is True

    def test_category_is_notion(self):
        assert CreateNotionPageTool.category == "notion"
