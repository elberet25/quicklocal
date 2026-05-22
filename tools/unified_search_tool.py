"""
Unified search across all knowledge sources: local files (RAG), Notion, and Google Drive.

Fans out a single query to all three sources in parallel and merges the results.
If one source fails (e.g. missing credentials), its results are skipped with a
warning — the other sources still return.
"""

import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    from tools.base_tool import BaseTool
    from tools.rag_tool import SearchDocumentsTool
    from tools.notion_tool import SearchNotionTool
    from tools.drive_tool import SearchDriveTool
except ImportError:
    from base_tool import BaseTool
    from rag_tool import SearchDocumentsTool
    from notion_tool import SearchNotionTool
    from drive_tool import SearchDriveTool

logger = logging.getLogger(__name__)


class UnifiedSearchTool(BaseTool):
    """Search local files, Notion, and Google Drive with a single query."""

    name = "search_all_knowledge"
    category = "knowledge"
    summarizable = True

    def get_description(self) -> dict:
        return {
            "name": self.name,
            "description": (
                "Searches all knowledge sources at once — local files, Notion pages, and Google Drive documents. "
                "Returns combined results labeled by source. "
                "Use this when the user asks a broad question and you're not sure which source has the answer, "
                "or when they explicitly want to search 'everything' or 'all sources'."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query to run across all knowledge sources.",
                    },
                    "max_results_per_source": {
                        "type": "integer",
                        "description": "Maximum results to return per source (default 3).",
                        "default": 3,
                    },
                },
                "required": ["query"],
            },
        }

    def execute(self, **kwargs) -> dict:
        try:
            query = kwargs["query"]
            n = kwargs.get("max_results_per_source", 3)

            results = []
            warnings = []

            sources = {
                "local_files": lambda: self._search_local(query, n),
                "notion": lambda: self._search_notion(query, n),
                "drive": lambda: self._search_drive(query, n),
            }

            with ThreadPoolExecutor(max_workers=3) as executor:
                futures = {executor.submit(fn): name for name, fn in sources.items()}
                for future in as_completed(futures):
                    source_name = futures[future]
                    try:
                        source_results = future.result()
                        results.extend(source_results)
                    except Exception as e:
                        msg = f"{source_name} unavailable: {e}"
                        warnings.append(msg)
                        logger.warning(msg)

            output = {"results": results}
            if warnings:
                output["warnings"] = warnings

            return {"result": json.dumps(output, ensure_ascii=False, indent=2)}
        except Exception as e:
            return self.handle_error(e)

    def validate_input(self, **kwargs) -> bool:
        return bool(kwargs.get("query"))

    # ------------------------------------------------------------------
    # Per-source search helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _search_local(query: str, n: int) -> list[dict]:
        """Run RAG search and normalize results to unified format."""
        tool = SearchDocumentsTool()
        raw = tool.execute(query=query, n_results=n)
        if "error" in raw:
            raise RuntimeError(raw["error"])

        text = raw.get("result", "")
        # SearchDocumentsTool returns formatted text, not JSON — wrap it as a single entry
        if not text or "No documents indexed" in text:
            return []
        return [{"source": "local_files", "content": text}]

    @staticmethod
    def _search_notion(query: str, n: int) -> list[dict]:
        """Run Notion search and normalize results to unified format."""
        tool = SearchNotionTool()
        raw = tool.execute(query=query, max_results=n)
        if "error" in raw:
            raise RuntimeError(raw["error"])

        pages = json.loads(raw["result"])
        return [
            {
                "source": "notion",
                "title": p.get("title", ""),
                "url": p.get("url", ""),
                "last_edited": p.get("last_edited", ""),
                "id": p.get("id", ""),
            }
            for p in pages
        ]

    @staticmethod
    def _search_drive(query: str, n: int) -> list[dict]:
        """Run Drive search and normalize results to unified format."""
        tool = SearchDriveTool()
        raw = tool.execute(query=query, max_results=n)
        if "error" in raw:
            raise RuntimeError(raw["error"])

        files = json.loads(raw["result"])
        return [
            {
                "source": "drive",
                "name": f.get("name", ""),
                "url": f.get("url", ""),
                "mimeType": f.get("mimeType", ""),
                "modified": f.get("modified", ""),
                "id": f.get("id", ""),
            }
            for f in files
        ]
