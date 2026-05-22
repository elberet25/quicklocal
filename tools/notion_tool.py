"""
Notion API integration using the official notion-client library.

Auth:
  Reads NOTION_TOKEN from the environment (.env file).
  Token is created in Notion → Settings → Connections → Develop or manage integrations.
  Each page/database must be explicitly shared with the integration.
"""

import json
import os

from notion_client import Client
from notion_client.errors import APIResponseError

try:
    from tools.base_tool import BaseTool
except ImportError:
    from base_tool import BaseTool


class NotionBaseTool(BaseTool):
    """Shared auth and helpers for all Notion tools."""

    category = "notion"
    summarizable = True

    @classmethod
    def _get_client(cls) -> Client:
        """Return an authenticated Notion API client."""
        token = os.getenv("NOTION_TOKEN")
        if not token:
            raise EnvironmentError(
                "NOTION_TOKEN is not set. Add it to your .env file. "
                "Get it from Notion → Settings → Connections → Develop or manage integrations."
            )
        return Client(auth=token)

    @staticmethod
    def _extract_title(page: dict) -> str:
        """Pull the plain-text title out of a page or database object."""
        props = page.get("properties", {})
        # Page titles live under a 'title' property (type='title')
        for prop in props.values():
            if prop.get("type") == "title":
                parts = prop.get("title", [])
                return "".join(p.get("plain_text", "") for p in parts)
        # Fallback: child_page / child_database blocks expose title directly
        return page.get("child_page", {}).get("title") or page.get("child_database", {}).get("title") or "(untitled)"

    @staticmethod
    def _blocks_to_text(blocks: list[dict]) -> str:
        """
        Recursively extract plain text from a list of Notion block objects.
        Handles the most common block types; unknown types are skipped.
        """
        TEXT_BLOCK_TYPES = {
            "paragraph",
            "heading_1",
            "heading_2",
            "heading_3",
            "bulleted_list_item",
            "numbered_list_item",
            "quote",
            "callout",
            "toggle",
            "to_do",
        }
        lines = []
        for block in blocks:
            btype = block.get("type")
            if btype in TEXT_BLOCK_TYPES:
                rich_text = block.get(btype, {}).get("rich_text", [])
                text = "".join(r.get("plain_text", "") for r in rich_text)
                if text.strip():
                    lines.append(text)
            elif btype == "child_page":
                title = block.get("child_page", {}).get("title", "")
                if title:
                    lines.append(f"[child page: {title}]")
            # code, image, table, divider, etc. are intentionally skipped
        return "\n".join(lines)


class SearchNotionTool(NotionBaseTool):
    """Search across all Notion pages shared with the integration."""

    name = "search_notion"

    def get_description(self) -> dict:
        return {
            "name": self.name,
            "description": (
                "Searches across all Notion pages shared with the integration. "
                "Returns a list of matching pages with their titles, URLs, and last-edited times. "
                "Use when the user asks to find Notion pages, notes, or documentation by topic or keyword."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query — keywords or phrases to find in page titles and content.",
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of pages to return (default 5, max 20).",
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
            client = self._get_client()

            response = client.search(
                query=query,
                filter={"value": "page", "property": "object"},
                page_size=max_results,
            )

            results = []
            for page in response.get("results", []):
                results.append({
                    "id": page["id"],
                    "title": self._extract_title(page),
                    "url": page.get("url", ""),
                    "last_edited": page.get("last_edited_time", ""),
                })

            return {"result": json.dumps(results, ensure_ascii=False, indent=2)}
        except APIResponseError as e:
            return self.handle_error(e)
        except Exception as e:
            return self.handle_error(e)

    def validate_input(self, **kwargs) -> bool:
        return bool(kwargs.get("query"))


class GetNotionPageTool(NotionBaseTool):
    """Fetch and return the full text content of a Notion page."""

    name = "get_notion_page"

    def get_description(self) -> dict:
        return {
            "name": self.name,
            "description": (
                "Fetches the full text content of a specific Notion page by its ID. "
                "Use after search_notion to read the actual content of a page. "
                "Pass the 'id' field from search_notion results as page_id."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "page_id": {
                        "type": "string",
                        "description": "Notion page ID (from search_notion results).",
                    },
                },
                "required": ["page_id"],
            },
        }

    def execute(self, **kwargs) -> dict:
        try:
            page_id = kwargs["page_id"]
            client = self._get_client()

            page = client.pages.retrieve(page_id=page_id)
            title = self._extract_title(page)
            url = page.get("url", "")

            # Fetch all blocks (handles pagination)
            blocks = []
            cursor = None
            while True:
                params = {"block_id": page_id, "page_size": 100}
                if cursor:
                    params["start_cursor"] = cursor
                response = client.blocks.children.list(**params)
                blocks.extend(response.get("results", []))
                if not response.get("has_more"):
                    break
                cursor = response.get("next_cursor")

            content = self._blocks_to_text(blocks)

            return {
                "result": json.dumps(
                    {"title": title, "url": url, "content": content},
                    ensure_ascii=False,
                    indent=2,
                )
            }
        except APIResponseError as e:
            return self.handle_error(e)
        except Exception as e:
            return self.handle_error(e)

    def validate_input(self, **kwargs) -> bool:
        return bool(kwargs.get("page_id"))
