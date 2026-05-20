"""
Unit tests for tools/rag_tool.py.

RAGEngine is never instantiated — all tests that reach the tool layer mock
RAGEngine.get() so no ChromaDB or sentence-transformers is required.

Run from the project root:
    pytest tests/test_rag_tool.py -v
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from tools.rag_tool import (
    IndexDocumentsTool,
    SearchDocumentsTool,
    _is_allowed,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_engine(index_return=(5, ["file_a.txt"]), search_return=None):
    engine = MagicMock()
    engine.index_directory.return_value = index_return
    engine.search.return_value = (
        search_return if search_return is not None
        else [{"source": "file_a.txt", "text": "Some relevant text.", "score": 0.85}]
    )
    return engine


# ---------------------------------------------------------------------------
# _is_allowed
# ---------------------------------------------------------------------------

class TestIsAllowed:
    def test_exact_configured_dir_is_allowed(self, monkeypatch, tmp_path):
        data_dir = tmp_path / "mydata"
        monkeypatch.setenv("QUICKLOCAL_DATA_DIRS", str(data_dir))
        assert _is_allowed(data_dir) is True

    def test_subdirectory_of_configured_dir_is_allowed(self, monkeypatch, tmp_path):
        data_dir = tmp_path / "mydata"
        monkeypatch.setenv("QUICKLOCAL_DATA_DIRS", str(data_dir))
        assert _is_allowed(data_dir / "subdir") is True

    def test_unrelated_path_is_rejected(self, monkeypatch, tmp_path):
        data_dir = tmp_path / "mydata"
        monkeypatch.setenv("QUICKLOCAL_DATA_DIRS", str(data_dir))
        assert _is_allowed(tmp_path / "other") is False

    def test_system_path_is_rejected(self, monkeypatch, tmp_path):
        monkeypatch.setenv("QUICKLOCAL_DATA_DIRS", str(tmp_path / "mydata"))
        assert _is_allowed(Path("/etc")) is False

    def test_multiple_configured_dirs(self, monkeypatch, tmp_path):
        dir_a = tmp_path / "dir_a"
        dir_b = tmp_path / "dir_b"
        monkeypatch.setenv("QUICKLOCAL_DATA_DIRS", f"{dir_a},{dir_b}")
        assert _is_allowed(dir_a) is True
        assert _is_allowed(dir_b) is True
        assert _is_allowed(tmp_path / "dir_c") is False


# ---------------------------------------------------------------------------
# SearchDocumentsTool
# ---------------------------------------------------------------------------

class TestSearchDocumentsTool:
    def setup_method(self):
        self.tool = SearchDocumentsTool()

    def test_empty_query_returns_error(self):
        result = self.tool.execute(query="")
        assert "error" in result

    def test_whitespace_query_returns_error(self):
        result = self.tool.execute(query="   ")
        assert "error" in result

    def test_valid_query_returns_result(self, monkeypatch, tmp_path):
        monkeypatch.setenv("QUICKLOCAL_DATA_DIRS", str(tmp_path))
        with patch("tools.rag_tool.RAGEngine.get", return_value=_mock_engine()):
            result = self.tool.execute(query="meeting notes")
        assert "result" in result
        assert "file_a.txt" in result["result"]

    def test_n_results_capped_at_20(self, monkeypatch, tmp_path):
        monkeypatch.setenv("QUICKLOCAL_DATA_DIRS", str(tmp_path))
        engine = _mock_engine()
        with patch("tools.rag_tool.RAGEngine.get", return_value=engine):
            self.tool.execute(query="test", n_results=999)
        engine.search.assert_called_once_with("test", 20)

    def test_empty_index_returns_prompt_to_index(self, monkeypatch, tmp_path):
        monkeypatch.setenv("QUICKLOCAL_DATA_DIRS", str(tmp_path))
        engine = _mock_engine(search_return=[])
        with patch("tools.rag_tool.RAGEngine.get", return_value=engine):
            result = self.tool.execute(query="anything")
        assert "result" in result
        assert "index" in result["result"].lower()


# ---------------------------------------------------------------------------
# IndexDocumentsTool
# ---------------------------------------------------------------------------

class TestIndexDocumentsTool:
    def setup_method(self):
        self.tool = IndexDocumentsTool()

    def test_restricted_path_returns_error(self, monkeypatch, tmp_path):
        monkeypatch.setenv("QUICKLOCAL_DATA_DIRS", str(tmp_path / "mydata"))
        result = self.tool.execute(directory="/etc")
        assert "error" in result
        assert "restricted" in result["error"].lower()

    def test_allowed_path_indexes_successfully(self, monkeypatch, tmp_path):
        monkeypatch.setenv("QUICKLOCAL_DATA_DIRS", str(tmp_path))
        engine = _mock_engine(index_return=(3, ["note.txt"]))
        with patch("tools.rag_tool.RAGEngine.get", return_value=engine):
            result = self.tool.execute(directory=str(tmp_path))
        assert "result" in result
        assert "note.txt" in result["result"]

    def test_no_directory_indexes_all_configured(self, monkeypatch, tmp_path):
        monkeypatch.setenv("QUICKLOCAL_DATA_DIRS", str(tmp_path))
        engine = _mock_engine(index_return=(7, ["a.txt", "b.md"]))
        with patch("tools.rag_tool.RAGEngine.get", return_value=engine):
            result = self.tool.execute()
        assert "result" in result
        assert "2 file(s)" in result["result"]

    def test_no_files_found_returns_informative_message(self, monkeypatch, tmp_path):
        monkeypatch.setenv("QUICKLOCAL_DATA_DIRS", str(tmp_path))
        engine = _mock_engine(index_return=(0, []))
        with patch("tools.rag_tool.RAGEngine.get", return_value=engine):
            result = self.tool.execute()
        assert "result" in result
        assert "no" in result["result"].lower()
