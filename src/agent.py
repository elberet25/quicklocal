"""
Simple AI Agent with Tool Use — using the Anthropic Claude API.

HOW IT WORKS (high level):
  1. We define tools that Claude can "call".
  2. The agent loop sends the user's message to Claude.
  3. If Claude decides it needs a tool, it returns a tool_use block
     instead of a final answer.  We execute the real Python function,
     send the result back, and Claude produces the final answer.
  4. We repeat until the user types "quit".

CONVERSATION MEMORY:
  - History is saved to HISTORY_FILE after every turn.
  - On startup the last session is restored automatically.
  - The last 10 exchanges are kept verbatim; older exchanges are
    replaced with a rolling Claude-generated summary so the context
    window stays manageable.
  - /clear  resets the conversation and deletes the history file.
"""

import json
import logging
import os
import sys
from pathlib import Path

import anthropic
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools import ALL_TOOLS

load_dotenv()

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# ---------------------------------------------------------------------------
# 1. CLIENT
# ---------------------------------------------------------------------------
client = anthropic.Anthropic(max_retries=3)

# ---------------------------------------------------------------------------
# 2. MODEL
# ---------------------------------------------------------------------------
MODEL = "claude-sonnet-4-6"

SYSTEM_PROMPT = """\
## Role
You are QuickLocal, a personal AI work assistant. You help the user manage their work across Gmail, Google Calendar, local documents, Notion, Google Drive, and Slack. You are concise, accurate, and always confirm before taking actions that write or modify data.

## Tool Guidance
- For email tasks (reading, searching, drafting replies) → use gmail_* tools
- For calendar tasks (schedule, free time, creating events) → use calendar_* tools
- For searching or indexing local documents → use rag_* tools
- For Notion tasks (finding pages, reading content, creating pages) → use search_notion, get_notion_page, create_notion_page
- For Google Drive tasks (finding files, reading Google Docs) → use search_drive, read_drive_document
- For creating a Google Doc → always call preview_drive_doc first, show the preview to the user, and only call create_drive_doc after explicit confirmation
- For Slack tasks (reading a channel, searching messages) → use get_channel_messages or search_slack; use get_slack_user_info to resolve any raw user ID you encounter into a display name
- For drafting a Slack message → always use draft_slack_message first and wait for explicit user confirmation before any further action
- For broad searches across all knowledge sources at once → use search_all_knowledge
- Always call preview_draft_reply before create_draft_reply — never skip the preview step
- When multiple tools could apply, prefer the most specific one

## Memory
Tool results in conversation history may be from a previous session and could be stale. \
For any question involving current state — documents, email, calendar, Notion pages, \
Drive files, or time — always call the relevant tool to get fresh data. \
Never answer from past tool results alone.\
"""

# ---------------------------------------------------------------------------
# 3. TOOL REGISTRY
# ---------------------------------------------------------------------------
tool_registry = {tool.name: tool for tool in ALL_TOOLS}
TOOLS = [tool.get_description() for tool in ALL_TOOLS]

# ---------------------------------------------------------------------------
# 4. HISTORY FILE
# ---------------------------------------------------------------------------
HISTORY_FILE = Path(os.getenv("CONVERSATION_HISTORY_FILE", "conversation_history.json"))

# How many recent exchanges (user+assistant pairs) to keep verbatim.
MAX_VERBATIM_EXCHANGES = 10


def _sanitize_history(conversation: list[dict]) -> list[dict]:
    """Last-resort guard: drop tool_result blocks with no matching tool_use in history.

    The summariser now aligns its cutoff to a clean user text message, so it
    should never split a tool_use/tool_result pair. This function exists as a
    safety net for history files written by older versions of the agent (before
    that fix) or any other edge case that produces an orphan. Without it, Claude
    returns a 400 because every tool_result must reference a tool_use present in
    the same conversation.
    """
    known_ids: set[str] = set()
    for msg in conversation:
        content = msg.get("content", "")
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "tool_use":
                    known_ids.add(block["id"])

    cleaned = []
    for msg in conversation:
        content = msg.get("content", "")
        if not isinstance(content, list):
            cleaned.append(msg)
            continue
        filtered = [
            b for b in content
            if not (
                isinstance(b, dict)
                and b.get("type") == "tool_result"
                and b.get("tool_use_id") not in known_ids
            )
        ]
        if filtered:
            cleaned.append({**msg, "content": filtered})

    return cleaned


def load_history() -> list[dict]:
    """Load conversation history from disk; return empty list if none exists."""
    if HISTORY_FILE.exists():
        try:
            history = json.loads(HISTORY_FILE.read_text(encoding="utf-8"))
            return _sanitize_history(history)
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Could not load history file: %s", e)
    return []


def _strip_tool_blocks(conversation: list[dict]) -> list[dict]:
    """Remove tool_use/tool_result blocks before persisting to disk.

    Tool results reflect external state (documents, email, calendar) at the
    moment of the call. Saving them causes Claude to trust stale data in the
    next session instead of re-running the tool. Claude's text responses
    already capture the outcome in human-readable form, so the raw API blocks
    add no cross-session value.

    Rules:
    - Pure tool_result user messages → dropped entirely.
    - Pure tool_use assistant messages → dropped entirely.
    - Mixed assistant messages (text + tool_use) → text blocks kept, tool_use dropped.
    """
    stripped = []
    for msg in conversation:
        content = msg.get("content", "")
        if not isinstance(content, list):
            stripped.append(msg)
            continue

        text_blocks = [b for b in content if isinstance(b, dict) and b.get("type") == "text"]
        has_tool_result = any(isinstance(b, dict) and b.get("type") == "tool_result" for b in content)

        if has_tool_result:
            continue  # pure tool_result user message — drop

        if text_blocks:
            stripped.append({**msg, "content": text_blocks})
        # else: pure tool_use assistant message — drop

    return stripped


def save_history(conversation: list[dict]) -> None:
    """Persist conversation history to disk, stripping ephemeral tool blocks."""
    try:
        HISTORY_FILE.write_text(
            json.dumps(_strip_tool_blocks(conversation), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except OSError as e:
        logger.warning("Could not save history file: %s", e)


def clear_history() -> None:
    """Delete the history file if it exists."""
    try:
        HISTORY_FILE.unlink(missing_ok=True)
    except OSError as e:
        logger.warning("Could not delete history file: %s", e)


# ---------------------------------------------------------------------------
# 5. SUMMARISATION
# ---------------------------------------------------------------------------

def _count_exchanges(conversation: list[dict]) -> int:
    """Return the number of user/assistant exchange pairs in the conversation.

    Skips the leading summary pair (if present) when counting.
    """
    messages = conversation[2:] if _has_summary(conversation) else conversation
    # Each exchange = 1 user message + 1 assistant message
    return len(messages) // 2


def _has_summary(conversation: list[dict]) -> bool:
    """Return True if the conversation starts with an injected summary pair."""
    return (
        len(conversation) >= 2
        and conversation[0].get("role") == "user"
        and isinstance(conversation[0].get("content"), str)
        and conversation[0]["content"].startswith("Summary of our earlier conversation:")
    )


def _format_tool_input(tool_input: dict) -> str:
    """Format a tool input dict as a brief human-readable string."""
    if not tool_input:
        return "no input"
    return ", ".join(f"{k}={v!r}" for k, v in tool_input.items())


def summarize_old_exchanges(conversation: list[dict]) -> list[dict]:
    """Replace exchanges older than MAX_VERBATIM_EXCHANGES with a summary.

    Returns the (possibly updated) conversation list.
    """
    has_summary = _has_summary(conversation)
    body = conversation[2:] if has_summary else conversation

    # Each exchange is 2 messages (user + assistant).
    if len(body) // 2 <= MAX_VERBATIM_EXCHANGES:
        return conversation

    cutoff = len(body) - MAX_VERBATIM_EXCHANGES * 2

    # Walk backward until verbatim starts at a real user text message.
    # A tool exchange is 3 messages (tool_use / tool_result / answer), so a
    # fixed 2-message cutoff can land between tool_use and tool_result.
    # User text messages always have str content; tool_result messages have list.
    while cutoff > 0 and not (
        body[cutoff].get("role") == "user"
        and isinstance(body[cutoff].get("content"), str)
    ):
        cutoff -= 1

    if cutoff == 0:
        return conversation  # nothing safe to summarize

    to_summarize = body[:cutoff]
    verbatim = body[cutoff:]

    # Build a readable transcript for Claude to summarize.
    # Summarizable tool calls are annotated; non-summarizable ones are skipped.
    summarizable_ids: set[str] = set()
    transcript_lines = []
    for msg in to_summarize:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if not isinstance(content, list):
            transcript_lines.append(f"{role.upper()}: {content}")
            continue
        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "tool_use":
                tool = tool_registry.get(block["name"])
                if tool and tool.summarizable:
                    summarizable_ids.add(block["id"])
                    input_desc = _format_tool_input(block.get("input", {}))
                    transcript_lines.append(f"[Used {block['name']}: {input_desc}]")
            elif block.get("type") == "tool_result":
                if block.get("tool_use_id") in summarizable_ids:
                    result_text = str(block.get("content", ""))
                    if len(result_text) > 200:
                        result_text = result_text[:200] + "..."
                    transcript_lines.append(f"[Result: {result_text}]")

    transcript = "\n\n".join(transcript_lines)

    logger.debug("Summarizing %d messages", len(to_summarize))
    summary_response = client.messages.create(
        model=MODEL,
        max_tokens=512,
        messages=[{
            "role": "user",
            "content": (
                "Summarize the following conversation excerpt concisely. "
                "Preserve key facts, decisions, and context that would help "
                "understand the rest of the conversation.\n\n"
                f"{transcript}"
            ),
        }],
    )
    summary_text = summary_response.content[0].text

    summary_pair = [
        {
            "role": "user",
            "content": f"Summary of our earlier conversation:\n{summary_text}",
        },
        {
            "role": "assistant",
            "content": "Understood, I have the context from our earlier conversation.",
        },
    ]

    return summary_pair + verbatim


# ---------------------------------------------------------------------------
# 6. TOOL DISPATCHER
# ---------------------------------------------------------------------------

def execute_tool(name: str, tool_input: dict) -> str:
    logger.debug("Tool called: %s | input: %s", name, tool_input)
    tool = tool_registry.get(name)
    if tool is None:
        return f"Error: unknown tool '{name}'"
    if tool.requires_confirmation:
        print(f"\n{tool.get_confirmation_message(**tool_input)}")
        answer = input("Proceed? (y/n): ").strip().lower()
        if answer != "y":
            return "Action cancelled by user."
    result = tool.execute(**tool_input)
    return result.get("result", result.get("error", str(result)))


# ---------------------------------------------------------------------------
# 7. CONTENT SERIALIZATION
# ---------------------------------------------------------------------------

def _serialize_block(block) -> dict:
    """Convert an Anthropic SDK content block to a plain API-compatible dict.

    model_dump() includes SDK-specific fields (e.g. citations=None) that the
    API rejects when the message is sent back in subsequent turns.
    """
    if block.type == "text":
        return {"type": "text", "text": block.text}
    if block.type == "tool_use":
        return {"type": "tool_use", "id": block.id, "name": block.name, "input": block.input}
    return block.model_dump(exclude_none=True)


# ---------------------------------------------------------------------------
# 8. AGENT LOOP (one turn)
# ---------------------------------------------------------------------------

def run_agent_turn(conversation: list[dict]) -> tuple[str, anthropic.types.Usage]:
    """Run one agent turn.

    Returns (answer_text, usage) where usage reflects the final API call.
    """
    last_usage = None
    while True:
        response = client.messages.create(
            model=MODEL,
            max_tokens=1024,
            tools=TOOLS,
            system=SYSTEM_PROMPT,
            messages=conversation,
        )
        last_usage = response.usage

        if response.stop_reason == "end_turn":
            for block in response.content:
                if block.type == "text":
                    return block.text, last_usage
            return "", last_usage

        if response.stop_reason == "tool_use":
            conversation.append({"role": "assistant", "content": [_serialize_block(b) for b in response.content]})
            tool_results = []
            for block in response.content:
                if block.type != "tool_use":
                    continue
                result_text = execute_tool(block.name, block.input)
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result_text,
                })
            conversation.append({"role": "user", "content": tool_results})
        else:
            raise RuntimeError(f"Unexpected stop_reason: {response.stop_reason!r}")


# ---------------------------------------------------------------------------
# 8. MAIN
# ---------------------------------------------------------------------------

def main() -> None:
    conversation = load_history()

    if conversation:
        exchanges = _count_exchanges(conversation)
        print(f"Resumed session ({exchanges} previous exchange(s) in memory).\n")
    else:
        print("AI Agent ready.  Type your message, or 'quit' to exit.\n")

    print("Commands: /clear — reset conversation\n")

    while True:
        user_input = input("You: ").strip()
        if not user_input:
            continue

        if user_input.lower() == "quit":
            print("Goodbye!")
            break

        if user_input.lower() == "/clear":
            conversation.clear()
            clear_history()
            print("Conversation cleared.\n")
            continue

        conversation.append({"role": "user", "content": user_input})

        answer, usage = run_agent_turn(conversation)
        conversation.append({"role": "assistant", "content": answer})

        # Summarize if the conversation has grown beyond the verbatim window.
        try:
            conversation = summarize_old_exchanges(conversation)
        except Exception as e:
            logger.warning("Summarization failed, continuing with full history: %s", e)

        save_history(conversation)

        total_tokens = usage.input_tokens + usage.output_tokens
        print(f"\nClaude: {answer}")
        print(f"[tokens: {total_tokens} ({usage.input_tokens} in / {usage.output_tokens} out)]\n")


if __name__ == "__main__":
    main()
