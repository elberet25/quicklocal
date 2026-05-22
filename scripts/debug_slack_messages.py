"""
Diagnostic script: print raw field values for the first 2 messages
from #general and #data-science to understand Slack message structure.

Run with venv activated:
    python scripts/debug_slack_messages.py
"""

import os
from slack_sdk import WebClient
from dotenv import load_dotenv

load_dotenv()

client = WebClient(token=os.getenv("SLACK_BOT_TOKEN"))

CHANNELS = ["general", "data-science"]

for channel_name in CHANNELS:
    print(f"\n{'='*60}")
    print(f"CHANNEL: #{channel_name}")
    print('='*60)

    # Resolve channel name to ID
    channel_id = None
    cursor = None
    while True:
        resp = client.conversations_list(
            exclude_archived=True, limit=200, types="public_channel", cursor=cursor
        )
        for ch in resp.get("channels", []):
            if ch["name"] == channel_name:
                channel_id = ch["id"]
                break
        if channel_id:
            break
        cursor = resp.get("response_metadata", {}).get("next_cursor")
        if not cursor:
            break

    if not channel_id:
        print(f"  Channel not found or bot not invited.")
        continue

    history = client.conversations_history(channel=channel_id, limit=3)
    messages = history.get("messages", [])

    for i, msg in enumerate(messages):
        print(f"\n--- Message {i+1} ---")
        for key in ("type", "user", "username", "bot_id", "app_id", "subtype", "text"):
            if key in msg:
                value = msg[key]
                if key == "text":
                    value = value[:80] + ("..." if len(value) > 80 else "")
                print(f"  {key}: {repr(value)}")
