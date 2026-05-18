"""
Simple AI Agent with Tool Use — using the Anthropic Claude API.

HOW IT WORKS (high level):
  1. We define a tool (get_current_time) that Claude can "call".
  2. The agent loop sends the user's message to Claude.
  3. If Claude decides it needs the time, it returns a tool_use block
     instead of a final answer.  We execute the real Python function,
     send the result back, and Claude produces the final answer.
  4. We repeat until the user types "quit".
"""

import datetime
import logging
import os

import anthropic
from dotenv import load_dotenv
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from gmail_tool import read_latest_emails, search_emails, preview_draft_reply, create_draft_reply

load_dotenv()  # loads .env from the current working directory

logging.basicConfig(
    level=logging.WARNING,  # suppress noisy third-party logs (httpx, anthropic, etc.)
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)  # our code logs at DEBUG and above

# ---------------------------------------------------------------------------
# 1. CLIENT
#    The Anthropic client reads ANTHROPIC_API_KEY from the environment.
#    Never hard-code keys in source files.
# ---------------------------------------------------------------------------
client = anthropic.Anthropic()

# ---------------------------------------------------------------------------
# 2. MODEL
#    claude-opus-4-7 is the current recommended default.
# ---------------------------------------------------------------------------
MODEL = "claude-sonnet-4-6"

# ---------------------------------------------------------------------------
# 3. TOOL DEFINITION
#    We describe our tool in JSON Schema so Claude knows:
#      • what the tool is called
#      • when to use it (description)
#      • what parameters it accepts (input_schema)
#
#    get_current_time needs no parameters — the schema is an empty object.
# ---------------------------------------------------------------------------
TOOLS = [
    {
        "name": "get_current_time",
        "description": (
            "Returns the current local date and time. "
            "Use this whenever the user asks what time or date it is."
        ),
        "input_schema": {
            "type": "object",
            "properties": {},   # no parameters needed
            "required": [],
        },
    },
    {
        "name": "read_latest_emails",
        "description": (
            "Fetches the N most recent emails from the user's Gmail inbox. "
            "Use when the user asks to check, show, or read their latest/recent emails."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "max_results": {
                    "type": "integer",
                    "description": "Number of emails to return (default 5, max 20).",
                    "default": 5,
                },
            },
            "required": [],
        },
    },
    {
        "name": "search_emails",
        "description": (
            "Searches Gmail using a search query string. Supports Gmail search operators: "
            "from:, to:, subject:, is:unread, has:attachment, after:YYYY/MM/DD, etc. "
            "Use when the user asks to find emails by sender, subject, keyword, or any filter."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "Gmail search query, e.g. 'from:boss@example.com', "
                        "'subject:invoice is:unread', 'from:newsletter after:2024/01/01'."
                    ),
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of results to return (default 10).",
                    "default": 10,
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "preview_draft_reply",
        "description": (
            "Formats a draft reply and returns it for the user to review. "
            "ALWAYS call this first and show the result to the user before calling create_draft_reply. "
            "Never skip this step — the user must confirm before any draft is saved."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "to": {"type": "string", "description": "Recipient email address."},
                "subject": {"type": "string", "description": "Email subject line."},
                "body": {"type": "string", "description": "Full email body text."},
                "reply_to_id": {
                    "type": "string",
                    "description": "Gmail message ID of the email being replied to (for threading). Leave empty for a new thread.",
                },
            },
            "required": ["to", "subject", "body"],
        },
    },
    {
        "name": "create_draft_reply",
        "description": (
            "Saves a draft reply in Gmail. "
            "Only call this after preview_draft_reply has been shown to the user "
            "and they have explicitly confirmed (e.g. 'yes', 'looks good', 'save it')."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "to": {"type": "string", "description": "Recipient email address."},
                "subject": {"type": "string", "description": "Email subject line."},
                "body": {"type": "string", "description": "Full email body text."},
                "reply_to_id": {
                    "type": "string",
                    "description": "Gmail message ID of the email being replied to (for threading). Leave empty for a new thread.",
                },
            },
            "required": ["to", "subject", "body"],
        },
    },
    {
        "name": "calculate",
        "description": (
            "Performs basic arithmetic: add, subtract, multiply, or divide two numbers. "
            "Use this whenever the user asks to calculate, compute, or do math."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "enum": ["add", "subtract", "multiply", "divide"],
                    "description": "The arithmetic operation to perform.",
                },
                "a": {
                    "type": "number",
                    "description": "The first number.",
                },
                "b": {
                    "type": "number",
                    "description": "The second number.",
                },
            },
            "required": ["operation", "a", "b"],
        },
    },
]

# ---------------------------------------------------------------------------
# 4. TOOL IMPLEMENTATION
#    This is the real Python function that runs when Claude calls the tool.
#    It must return a plain string (or something JSON-serialisable).
# ---------------------------------------------------------------------------
def get_current_time() -> str:
    now = datetime.datetime.now()
    return now.strftime("%A, %B %d %Y — %I:%M:%S %p")


def calculate(operation: str, a: float, b: float) -> str:
    if operation == "add":
        result = a + b
    elif operation == "subtract":
        result = a - b
    elif operation == "multiply":
        result = a * b
    elif operation == "divide":
        if b == 0:
            return "Error: division by zero"
        result = a / b
    else:
        return f"Error: unknown operation '{operation}'"
    return str(result)


# ---------------------------------------------------------------------------
# 5. TOOL DISPATCHER
#    Maps tool names → Python functions so we can call them generically.
#    If you add more tools later, register them here.
# ---------------------------------------------------------------------------
TOOL_REGISTRY = {
    "get_current_time": get_current_time,
    "calculate": calculate,
    "read_latest_emails": read_latest_emails,
    "search_emails": search_emails,
    "preview_draft_reply": preview_draft_reply,
    "create_draft_reply": create_draft_reply,
}

def execute_tool(name: str, tool_input: dict) -> str:
    """Look up and run the requested tool; return its string result."""
    logger.debug("Tool called: %s | input: %s", name, tool_input)
    fn = TOOL_REGISTRY.get(name)
    if fn is None:
        return f"Error: unknown tool '{name}'"
    return fn(**tool_input)          # pass Claude's parsed arguments to the function


# ---------------------------------------------------------------------------
# 6. AGENT LOOP (one turn)
#    Sends messages to Claude and handles the tool-use round trip.
#
#    Why a loop here?
#      Claude may call tools multiple times before giving a final answer.
#      Each time it does, we:
#        a) receive a response with stop_reason == "tool_use"
#        b) execute all requested tools
#        c) append the tool results as a new "user" message
#        d) send everything back to Claude
#      We exit only when stop_reason == "end_turn".
# ---------------------------------------------------------------------------
def run_agent_turn(conversation: list[dict]) -> str:
    """
    Run one agent turn.  May make multiple API calls if tools are needed.
    Returns Claude's final text answer.
    """
    while True:
        response = client.messages.create(
            model=MODEL,
            max_tokens=1024,
            tools=TOOLS,
            messages=conversation,
        )

        # ---- Case A: Claude is done — extract and return the text ----
        if response.stop_reason == "end_turn":
            for block in response.content:
                if block.type == "text":
                    return block.text
            return ""   # shouldn't happen, but safe fallback

        # ---- Case B: Claude wants to call one or more tools ----
        if response.stop_reason == "tool_use":
            # 6a. Append Claude's response (including tool_use blocks) to the
            #     conversation so the history stays consistent.
            conversation.append({"role": "assistant", "content": response.content})

            # 6b. Execute every tool Claude requested and collect the results.
            tool_results = []
            for block in response.content:
                if block.type != "tool_use":
                    continue

                result_text = execute_tool(block.name, block.input)

                # Each tool result must reference the matching tool_use id.
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result_text,
                })

            # 6c. Send all results back as a single "user" message.
            conversation.append({"role": "user", "content": tool_results})
            # Loop: Claude will now produce its final answer (or call more tools).

        else:
            # Unexpected stop reason — surface it so the developer can debug.
            raise RuntimeError(f"Unexpected stop_reason: {response.stop_reason!r}")


# ---------------------------------------------------------------------------
# 7. MAIN — conversation loop
#    Maintains the full conversation history across turns so Claude has
#    context for follow-up questions.
# ---------------------------------------------------------------------------
def main() -> None:
    print("AI Agent ready.  Type your message, or 'quit' to exit.\n")

    # The conversation is a list of {role, content} dicts.
    # We keep appending to it so Claude remembers previous exchanges.
    conversation: list[dict] = []

    while True:
        user_input = input("You: ").strip()
        if not user_input:
            continue
        if user_input.lower() == "quit":
            print("Goodbye!")
            break

        # Add the user's message to the conversation.
        conversation.append({"role": "user", "content": user_input})

        # Ask Claude (with tool-use support).
        answer = run_agent_turn(conversation)

        # Add Claude's final answer to the history for future context.
        conversation.append({"role": "assistant", "content": answer})

        print(f"\nClaude: {answer}\n")


if __name__ == "__main__":
    main()
