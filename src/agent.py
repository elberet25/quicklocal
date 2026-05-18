"""
Simple AI Agent with Tool Use — using the Anthropic Claude API.

HOW IT WORKS (high level):
  1. We define tools that Claude can "call".
  2. The agent loop sends the user's message to Claude.
  3. If Claude decides it needs a tool, it returns a tool_use block
     instead of a final answer.  We execute the real Python function,
     send the result back, and Claude produces the final answer.
  4. We repeat until the user types "quit".
"""

import logging
import os
import sys
from pathlib import Path

import anthropic
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools import ALL_TOOLS

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
# ---------------------------------------------------------------------------
MODEL = "claude-sonnet-4-6"

# ---------------------------------------------------------------------------
# 3. TOOL REGISTRY
#    Maps tool name → tool instance.  TOOLS is the schema list for the API.
# ---------------------------------------------------------------------------
tool_registry = {tool.name: tool for tool in ALL_TOOLS}
TOOLS = [tool.get_description() for tool in ALL_TOOLS]

# ---------------------------------------------------------------------------
# 4. TOOL DISPATCHER
# ---------------------------------------------------------------------------
def execute_tool(name: str, tool_input: dict) -> str:
    """Look up and run the requested tool; return its string result."""
    logger.debug("Tool called: %s | input: %s", name, tool_input)
    tool = tool_registry.get(name)
    if tool is None:
        return f"Error: unknown tool '{name}'"
    result = tool.execute(**tool_input)
    return result.get("result", result.get("error", str(result)))


# ---------------------------------------------------------------------------
# 5. AGENT LOOP (one turn)
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
            # Append Claude's response (including tool_use blocks) to the
            # conversation so the history stays consistent.
            conversation.append({"role": "assistant", "content": response.content})

            # Execute every tool Claude requested and collect the results.
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

            # Send all results back as a single "user" message.
            conversation.append({"role": "user", "content": tool_results})
            # Loop: Claude will now produce its final answer (or call more tools).

        else:
            # Unexpected stop reason — surface it so the developer can debug.
            raise RuntimeError(f"Unexpected stop_reason: {response.stop_reason!r}")


# ---------------------------------------------------------------------------
# 6. MAIN — conversation loop
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
