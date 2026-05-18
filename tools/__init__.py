from tools.time_tool import TimeTool
from tools.calculator_tool import CalculatorTool
from tools.gmail_tool import (
    ReadLatestEmailsTool,
    SearchEmailsTool,
    PreviewDraftReplyTool,
    CreateDraftReplyTool,
)
from tools.calendar_tool import (
    GetScheduleTool,
    FindFreeTimeTool,
    CreateEventTool,
)

ALL_TOOLS = [
    TimeTool(),
    CalculatorTool(),
    ReadLatestEmailsTool(),
    SearchEmailsTool(),
    PreviewDraftReplyTool(),
    CreateDraftReplyTool(),
    GetScheduleTool(),
    FindFreeTimeTool(),
    CreateEventTool(),
]

__all__ = ["ALL_TOOLS"]
