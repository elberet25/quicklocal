from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from tools.vision_tool import AnalyzeImageTool


@pytest.fixture
def tool():
    return AnalyzeImageTool()


def _make_png(tmp_path, name="test.png"):
    f = tmp_path / name
    f.write_bytes(b"\x89PNG\r\n\x1a\n")
    return f


# ---------------------------------------------------------------------------
# Description / metadata
# ---------------------------------------------------------------------------

class TestAnalyzeImageToolDescription:
    def test_name(self, tool):
        assert tool.name == "analyze_image"

    def test_schema_has_required_image_path(self, tool):
        desc = tool.get_description()
        assert "image_path" in desc["input_schema"]["properties"]
        assert "image_path" in desc["input_schema"]["required"]

    def test_schema_has_optional_question(self, tool):
        desc = tool.get_description()
        assert "question" in desc["input_schema"]["properties"]
        assert "question" not in desc["input_schema"]["required"]

    def test_category(self, tool):
        assert tool.category == "vision"

    def test_summarizable(self, tool):
        assert tool.summarizable is True

    def test_requires_confirmation_false(self, tool):
        assert tool.requires_confirmation is False


# ---------------------------------------------------------------------------
# validate_input
# ---------------------------------------------------------------------------

class TestAnalyzeImageToolValidateInput:
    def test_valid_input(self, tool):
        assert tool.validate_input(image_path="/some/path.png") is True

    def test_missing_image_path(self, tool):
        assert tool.validate_input() is False

    def test_empty_image_path(self, tool):
        assert tool.validate_input(image_path="") is False


# ---------------------------------------------------------------------------
# execute — happy paths
# ---------------------------------------------------------------------------

class TestAnalyzeImageToolExecute:
    def _mock_response(self, text="A diagram showing components."):
        response = MagicMock()
        response.content = [MagicMock(text=text)]
        return response

    def test_successful_analysis(self, tool, tmp_path):
        img = _make_png(tmp_path)
        with patch("tools.vision_tool.anthropic.Anthropic") as mock_cls:
            mock_cls.return_value.messages.create.return_value = self._mock_response()
            result = tool.execute(image_path=str(img))
        assert result == {"result": "A diagram showing components."}

    def test_default_question_used_when_omitted(self, tool, tmp_path):
        img = _make_png(tmp_path)
        with patch("tools.vision_tool.anthropic.Anthropic") as mock_cls:
            mock_client = mock_cls.return_value
            mock_client.messages.create.return_value = self._mock_response()
            tool.execute(image_path=str(img))
        messages = mock_client.messages.create.call_args.kwargs["messages"]
        text_block = next(b for b in messages[0]["content"] if b.get("type") == "text")
        assert text_block["text"] == "Describe this image in detail."

    def test_custom_question_passed_to_api(self, tool, tmp_path):
        img = _make_png(tmp_path)
        with patch("tools.vision_tool.anthropic.Anthropic") as mock_cls:
            mock_client = mock_cls.return_value
            mock_client.messages.create.return_value = self._mock_response()
            tool.execute(image_path=str(img), question="What is the title?")
        messages = mock_client.messages.create.call_args.kwargs["messages"]
        text_block = next(b for b in messages[0]["content"] if b.get("type") == "text")
        assert text_block["text"] == "What is the title?"

    def test_image_sent_as_base64_block(self, tool, tmp_path):
        img = _make_png(tmp_path)
        with patch("tools.vision_tool.anthropic.Anthropic") as mock_cls:
            mock_client = mock_cls.return_value
            mock_client.messages.create.return_value = self._mock_response()
            tool.execute(image_path=str(img))
        messages = mock_client.messages.create.call_args.kwargs["messages"]
        image_block = next(b for b in messages[0]["content"] if b.get("type") == "image")
        assert image_block["source"]["type"] == "base64"
        assert image_block["source"]["media_type"] == "image/png"

    def test_media_type_jpeg_for_jpg(self, tool, tmp_path):
        f = tmp_path / "photo.jpg"
        f.write_bytes(b"\xff\xd8\xff")
        with patch("tools.vision_tool.anthropic.Anthropic") as mock_cls:
            mock_client = mock_cls.return_value
            mock_client.messages.create.return_value = self._mock_response()
            tool.execute(image_path=str(f))
        messages = mock_client.messages.create.call_args.kwargs["messages"]
        image_block = next(b for b in messages[0]["content"] if b.get("type") == "image")
        assert image_block["source"]["media_type"] == "image/jpeg"

    def test_media_type_jpeg_for_jpeg_extension(self, tool, tmp_path):
        f = tmp_path / "photo.jpeg"
        f.write_bytes(b"\xff\xd8\xff")
        with patch("tools.vision_tool.anthropic.Anthropic") as mock_cls:
            mock_client = mock_cls.return_value
            mock_client.messages.create.return_value = self._mock_response()
            tool.execute(image_path=str(f))
        messages = mock_client.messages.create.call_args.kwargs["messages"]
        image_block = next(b for b in messages[0]["content"] if b.get("type") == "image")
        assert image_block["source"]["media_type"] == "image/jpeg"

    def test_tilde_path_expanded(self, tool, tmp_path, monkeypatch):
        img = _make_png(tmp_path)
        monkeypatch.setenv("HOME", str(tmp_path))
        with patch("tools.vision_tool.anthropic.Anthropic") as mock_cls:
            mock_cls.return_value.messages.create.return_value = self._mock_response("ok")
            result = tool.execute(image_path="~/test.png")
        assert result == {"result": "ok"}

    # ---------------------------------------------------------------------------
    # execute — error paths
    # ---------------------------------------------------------------------------

    def test_unsupported_extension(self, tool, tmp_path):
        f = tmp_path / "doc.pdf"
        f.write_bytes(b"%PDF")
        result = tool.execute(image_path=str(f))
        assert "error" in result
        assert "Unsupported" in result["error"]

    def test_file_not_found(self, tool):
        result = tool.execute(image_path="/nonexistent/path/image.png")
        assert "error" in result
        assert "not found" in result["error"].lower()

    def test_empty_file(self, tool, tmp_path):
        f = tmp_path / "empty.png"
        f.write_bytes(b"")
        result = tool.execute(image_path=str(f))
        assert "error" in result
        assert "empty" in result["error"].lower()

    def test_io_error_on_read(self, tool, tmp_path):
        img = _make_png(tmp_path)
        with patch.object(Path, "read_bytes", side_effect=OSError("Permission denied")):
            result = tool.execute(image_path=str(img))
        assert "error" in result
