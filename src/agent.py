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
client = anthropic.Anthropic()

# ---------------------------------------------------------------------------
# 2. MODEL
# ---------------------------------------------------------------------------
MODEL = "claude-sonnet-4-6"

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


def load_history() -> list[dict]:
    """Load conversation history from disk; return empty list if none exists."""
    if HISTORY_FILE.exists():
        try:
            return json.loads(HISTORY_FILE.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Could not load history file: %s", e)
    return []


def save_history(conversation: list[dict]) -> None:
    """Persist conversation history to disk."""
    try:
        HISTORY_FILE.write_text(
            json.dumps(conversation, ensure_ascii=False, indent=2),
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
    to_summarize = body[:cutoff]
    verbatim = body[cutoff:]

    # Build a readable transcript for Claude to summarize.
    transcript_lines = []
    for msg in to_summarize:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if isinstance(content, list):
            # tool_use / tool_result blocks — skip, not useful in summary
            continue
        transcript_lines.append(f"{role.upper()}: {content}")

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
    result = tool.execute(**tool_input)
    return result.get("result", result.get("error", str(result)))


# ---------------------------------------------------------------------------
# 7. AGENT LOOP (one turn)
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
            messages=conversation,
        )
        last_usage = response.usage

        if response.stop_reason == "end_turn":
            for block in response.content:
                if block.type == "text":
                    return block.text, last_usage
            return "", last_usage

        if response.stop_reason == "tool_use":
            conversation.append({"role": "assistant", "content": response.content})
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
        conversation = summarize_old_exchanges(conversation)

        save_history(conversation)

        total_tokens = usage.input_tokens + usage.output_tokens
        print(f"\nClaude: {answer}")
        print(f"[tokens: {total_tokens} ({usage.input_tokens} in / {usage.output_tokens} out)]\n")


if __name__ == "__main__":
    main()
