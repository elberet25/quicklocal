from tools.time_tool import TimeTool
from tools.calculator_tool import CalculatorTool
from tools.gmail_tool import (
    ReadLatestEmailsTool,
    SearchEmailsTool,
    PreviewDraftReplyTool,
    CreateDraftReplyTool,
)

ALL_TOOLS = [
    TimeTool(),
    CalculatorTool(),
    ReadLatestEmailsTool(),
    SearchEmailsTool(),
    PreviewDraftReplyTool(),
    CreateDraftReplyTool(),
]

__all__ = ["ALL_TOOLS"]
