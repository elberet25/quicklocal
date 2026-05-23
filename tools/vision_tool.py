import base64
import os
from pathlib import Path

import anthropic
from dotenv import load_dotenv

try:
    from tools.base_tool import BaseTool
except ImportError:
    from base_tool import BaseTool  # type: ignore[no-redef]

load_dotenv()

_SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".webp"}

_MEDIA_TYPES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
}

_MODEL = "claude-sonnet-4-6"


class AnalyzeImageTool(BaseTool):
    """Analyze an image or screenshot using Claude's vision capabilities."""

    name = "analyze_image"
    category = "vision"
    summarizable = True

    def get_description(self) -> dict:
        return {
            "name": self.name,
            "description": (
                "Analyze an image or screenshot using Claude's vision capabilities. "
                "Use this when you need to reason about an image in detail — to read specific text, "
                "trace a particular flow, answer a precise question, or analyze visual elements "
                "beyond what an indexed description already covers. "
                "Note: image descriptions are indexed in the RAG store for discovery; "
                "call this tool only when deeper analysis of the specific image is needed "
                "to complete the task. "
                "Supported formats: PNG, JPG, JPEG, GIF, WebP."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "image_path": {
                        "type": "string",
                        "description": "Absolute path to the image file to analyze.",
                    },
                    "question": {
                        "type": "string",
                        "description": (
                            "Question or instruction about the image. "
                            "Defaults to a general description if not provided."
                        ),
                    },
                },
                "required": ["image_path"],
            },
        }

    def validate_input(self, **kwargs) -> bool:
        return bool(kwargs.get("image_path"))

    def execute(self, image_path: str, question: str = "Describe this image in detail.") -> dict:
        """Read the image at image_path and return an analysis via Claude Vision."""
        try:
            path = Path(image_path).expanduser().resolve()
            ext = path.suffix.lower()

            if ext not in _SUPPORTED_EXTENSIONS:
                return {
                    "error": (
                        f"Unsupported file type '{ext}'. "
                        f"Supported: {', '.join(sorted(_SUPPORTED_EXTENSIONS))}"
                    )
                }

            if not path.exists():
                return {"error": f"File not found: {image_path}"}

            image_bytes = path.read_bytes()
            if not image_bytes:
                return {"error": f"File is empty: {image_path}"}

            image_data = base64.standard_b64encode(image_bytes).decode("utf-8")
            client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
            response = client.messages.create(
                model=_MODEL,
                max_tokens=1024,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": _MEDIA_TYPES[ext],
                                    "data": image_data,
                                },
                            },
                            {"type": "text", "text": question},
                        ],
                    }
                ],
            )
            return {"result": response.content[0].text}

        except Exception as e:
            return self.handle_error(e)
