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
    RAGEngine,
    CHUNKING_VERSION,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_engine(index_return=(5, ["file_a.txt"]), search_return=None):
    engine = MagicMock()
    engine.index_directory.return_value = index_return
    engine.search.return_value = (
        search_return if search_return is not None
        else [{"source": "file_a.txt", "text": "Some relevant text.", "score": 0.85, "doc_summary": "Summary of file_a."}]
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


    def test_rag_debug_prints_retrieval_info(self, monkeypatch, tmp_path, capsys):
        monkeypatch.setenv("QUICKLOCAL_DATA_DIRS", str(tmp_path))
        monkeypatch.setenv("RAG_DEBUG", "true")
        with patch("tools.rag_tool.RAGEngine.get", return_value=_mock_engine()):
            self.tool.execute(query="meeting notes")
        captured = capsys.readouterr()
        assert "[RAG_DEBUG]" in captured.out
        assert "file_a.txt" in captured.out
        assert "Document summaries" in captured.out
        assert "tokens" in captured.out


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


# ---------------------------------------------------------------------------
# Shared fixture: RAGEngine instance with mocked ChromaDB and embeddings
# ---------------------------------------------------------------------------

def _make_engine():
    """Create a RAGEngine instance with mocked external dependencies."""
    with patch("tools.rag_tool.chromadb.PersistentClient"), \
         patch("tools.rag_tool.SentenceTransformer"):
        RAGEngine._instance = None
        eng = RAGEngine.get()
    eng._collection = MagicMock()
    eng._model = MagicMock()
    # encode() returns a numpy-array-like: needs .tolist() to work in _index_image_file
    mock_embedding = MagicMock()
    mock_embedding.tolist.return_value = [[0.1, 0.2, 0.3]]
    eng._model.encode.return_value = mock_embedding
    return eng


def _png(tmp_path, name="diagram.png"):
    f = tmp_path / name
    f.write_bytes(b"\x89PNG\r\n\x1a\n")
    return f


def _vision_response(text):
    resp = MagicMock()
    resp.content = [MagicMock(text=text)]
    return resp


# ---------------------------------------------------------------------------
# RAGEngine: _describe_image
# ---------------------------------------------------------------------------

class TestDescribeImage:
    def setup_method(self):
        self.engine = _make_engine()

    def teardown_method(self):
        RAGEngine._instance = None

    def test_returns_description_on_success(self, tmp_path):
        img = _png(tmp_path)
        with patch("tools.rag_tool.anthropic.Anthropic") as mock_cls:
            mock_cls.return_value.messages.create.return_value = _vision_response("A flowchart showing steps.")
            result = self.engine._describe_image(img)
        assert result == "A flowchart showing steps."

    def test_exception_returns_empty_string(self, tmp_path):
        img = _png(tmp_path)
        with patch("tools.rag_tool.anthropic.Anthropic") as mock_cls:
            mock_cls.return_value.messages.create.side_effect = Exception("API error")
            result = self.engine._describe_image(img)
        assert result == ""

    def test_empty_api_response_returns_empty_string(self, tmp_path):
        """Vision API returns 200 but response text is empty — not an exception."""
        img = _png(tmp_path)
        with patch("tools.rag_tool.anthropic.Anthropic") as mock_cls:
            mock_cls.return_value.messages.create.return_value = _vision_response("")
            result = self.engine._describe_image(img)
        assert result == ""

    def test_whitespace_api_response_returns_empty_string(self, tmp_path):
        """Vision API returns whitespace-only text — treated the same as empty."""
        img = _png(tmp_path)
        with patch("tools.rag_tool.anthropic.Anthropic") as mock_cls:
            mock_cls.return_value.messages.create.return_value = _vision_response("   \n  ")
            result = self.engine._describe_image(img)
        assert result == ""

    def test_png_uses_correct_media_type(self, tmp_path):
        img = _png(tmp_path)
        with patch("tools.rag_tool.anthropic.Anthropic") as mock_cls:
            mock_client = mock_cls.return_value
            mock_client.messages.create.return_value = _vision_response("desc")
            self.engine._describe_image(img)
        messages = mock_client.messages.create.call_args.kwargs["messages"]
        image_block = next(b for b in messages[0]["content"] if b.get("type") == "image")
        assert image_block["source"]["media_type"] == "image/png"

    def test_jpg_uses_correct_media_type(self, tmp_path):
        img = tmp_path / "photo.jpg"
        img.write_bytes(b"\xff\xd8\xff")
        with patch("tools.rag_tool.anthropic.Anthropic") as mock_cls:
            mock_client = mock_cls.return_value
            mock_client.messages.create.return_value = _vision_response("desc")
            self.engine._describe_image(img)
        messages = mock_client.messages.create.call_args.kwargs["messages"]
        image_block = next(b for b in messages[0]["content"] if b.get("type") == "image")
        assert image_block["source"]["media_type"] == "image/jpeg"


# ---------------------------------------------------------------------------
# RAGEngine: _index_image_file
# ---------------------------------------------------------------------------

class TestIndexImageFile:
    def setup_method(self):
        self.engine = _make_engine()

    def teardown_method(self):
        RAGEngine._instance = None

    def test_success_returns_1(self, tmp_path):
        img = _png(tmp_path)
        with patch.object(self.engine, "_describe_image", return_value="A component diagram."):
            result = self.engine._index_image_file(img)
        assert result == 1

    def test_collection_add_called_on_success(self, tmp_path):
        img = _png(tmp_path)
        with patch.object(self.engine, "_describe_image", return_value="A component diagram."):
            self.engine._index_image_file(img)
        self.engine._collection.add.assert_called_once()

    def test_descriptor_text_contains_image_marker_path_and_description(self, tmp_path):
        img = _png(tmp_path)
        with patch.object(self.engine, "_describe_image", return_value="A flowchart."):
            self.engine._index_image_file(img)
        doc = self.engine._collection.add.call_args.kwargs["documents"][0]
        assert doc.startswith("[IMAGE]")
        assert img.name in doc
        assert str(img) in doc
        assert "A flowchart." in doc

    def test_metadata_level_chunk_type_image_source_path(self, tmp_path):
        img = _png(tmp_path)
        with patch.object(self.engine, "_describe_image", return_value="A chart."):
            self.engine._index_image_file(img)
        meta = self.engine._collection.add.call_args.kwargs["metadatas"][0]
        assert meta["level"] == "chunk"
        assert meta["type"] == "image"
        assert meta["source"] == str(img)

    def test_old_entry_deleted_before_new_add(self, tmp_path):
        img = _png(tmp_path)
        with patch.object(self.engine, "_describe_image", return_value="desc"):
            self.engine._index_image_file(img)
        self.engine._collection.delete.assert_called_once_with(where={"source": str(img)})

    def test_vision_exception_returns_0(self, tmp_path):
        img = _png(tmp_path)
        with patch.object(self.engine, "_describe_image", return_value=""):
            result = self.engine._index_image_file(img)
        assert result == 0

    def test_empty_description_from_api_returns_0(self, tmp_path):
        """Covers the case where Vision API returns 200 but an empty string — not an exception."""
        img = _png(tmp_path)
        with patch.object(self.engine, "_describe_image", return_value=""):
            result = self.engine._index_image_file(img)
        assert result == 0

    def test_vision_failure_does_not_call_collection_add(self, tmp_path):
        img = _png(tmp_path)
        with patch.object(self.engine, "_describe_image", return_value=""):
            self.engine._index_image_file(img)
        self.engine._collection.add.assert_not_called()


# ---------------------------------------------------------------------------
# RAGEngine: _sync_directory with image files
# ---------------------------------------------------------------------------

class TestSyncDirectoryImages:
    def setup_method(self):
        self.engine = _make_engine()

    def teardown_method(self):
        RAGEngine._instance = None

    def test_new_image_routed_to_index_image_file(self, tmp_path):
        img = _png(tmp_path)
        with patch.object(self.engine, "_load_manifest", return_value={"__chunking_version__": CHUNKING_VERSION}), \
             patch.object(self.engine, "_save_manifest"), \
             patch.object(self.engine, "_index_image_file", return_value=1) as mock_idx:
            self.engine._sync_directory(tmp_path)
        mock_idx.assert_called_once_with(img)

    def test_manifest_updated_when_image_indexed_successfully(self, tmp_path):
        img = _png(tmp_path)
        mock_save = MagicMock()
        with patch.object(self.engine, "_load_manifest", return_value={"__chunking_version__": CHUNKING_VERSION}), \
             patch.object(self.engine, "_save_manifest", mock_save), \
             patch.object(self.engine, "_index_image_file", return_value=1):
            self.engine._sync_directory(tmp_path)
        mock_save.assert_called_once()
        saved_manifest = mock_save.call_args[0][0]
        assert str(img) in saved_manifest

    def test_manifest_not_updated_when_vision_fails(self, tmp_path):
        """If _index_image_file returns 0, manifest is not written so the file retries next sync."""
        _png(tmp_path)
        mock_save = MagicMock()
        with patch.object(self.engine, "_load_manifest", return_value={"__chunking_version__": CHUNKING_VERSION}), \
             patch.object(self.engine, "_save_manifest", mock_save), \
             patch.object(self.engine, "_index_image_file", return_value=0):
            self.engine._sync_directory(tmp_path)
        mock_save.assert_not_called()

    def test_image_reindexed_when_mtime_changes(self, tmp_path):
        img = _png(tmp_path)
        stale_state = {"mtime": 0.0, "size": 0}
        with patch.object(self.engine, "_load_manifest",
                          return_value={"__chunking_version__": CHUNKING_VERSION, str(img): stale_state}), \
             patch.object(self.engine, "_save_manifest"), \
             patch.object(self.engine, "_index_image_file", return_value=1) as mock_idx:
            self.engine._sync_directory(tmp_path)
        mock_idx.assert_called_once_with(img)

    def test_image_not_reindexed_when_unchanged(self, tmp_path):
        img = _png(tmp_path)
        current_state = self.engine._file_state(img)
        with patch.object(self.engine, "_load_manifest",
                          return_value={"__chunking_version__": CHUNKING_VERSION, str(img): current_state}), \
             patch.object(self.engine, "_save_manifest"), \
             patch.object(self.engine, "_index_image_file") as mock_idx:
            self.engine._sync_directory(tmp_path)
        mock_idx.assert_not_called()

    def test_chunking_version_change_clears_old_entries_and_indexes_images(self, tmp_path):
        img = _png(tmp_path)
        old_manifest = {"__chunking_version__": CHUNKING_VERSION - 1, str(img): {"mtime": 0.0, "size": 0}}
        with patch.object(self.engine, "_load_manifest", return_value=old_manifest), \
             patch.object(self.engine, "_save_manifest"), \
             patch.object(self.engine, "_index_image_file", return_value=1) as mock_idx:
            self.engine._sync_directory(tmp_path)
        # Old entry deleted due to version change, image treated as new and indexed
        self.engine._collection.delete.assert_called()
        mock_idx.assert_called_once_with(img)
