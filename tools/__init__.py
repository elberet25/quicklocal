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
from tools.rag_tool import IndexDocumentsTool, SearchDocumentsTool
from tools.notion_tool import SearchNotionTool, GetNotionPageTool, CreateNotionPageTool
from tools.drive_tool import SearchDriveTool, ReadDriveDocumentTool, PreviewDriveDocTool, CreateDriveDocTool
from tools.unified_search_tool import UnifiedSearchTool
from tools.slack_tool import GetChannelMessagesTool, SearchSlackTool, GetSlackUserInfoTool, DraftSlackMessageTool

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
    IndexDocumentsTool(),
    SearchDocumentsTool(),
    SearchNotionTool(),
    GetNotionPageTool(),
    CreateNotionPageTool(),
    SearchDriveTool(),
    ReadDriveDocumentTool(),
    PreviewDriveDocTool(),
    CreateDriveDocTool(),
    UnifiedSearchTool(),
    GetChannelMessagesTool(),
    SearchSlackTool(),
    GetSlackUserInfoTool(),
    DraftSlackMessageTool(),
]

__all__ = ["ALL_TOOLS"]
