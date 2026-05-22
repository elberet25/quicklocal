# QuickLocal

A personal AI assistant you run locally, built tool by tool to connect the apps you use every day: Gmail, Google Calendar, your own files and notes (Notion, Google Drive), and Slack. Ask questions, surface what matters, and take action — all from a single conversational interface.

## Vision

The goal is to keep adding tools until the assistant covers the main places where work happens:

- **Multi-tool orchestration** — email, calendar, files, Notion, Drive, Slack
- **Personalized knowledge** — RAG over local documents and cloud notes
- **Agentic reasoning** — multi-step task planning and execution
- **Multimodal understanding** — screenshots, documents, images
- **Action execution** — draft emails, create docs, schedule meetings
- **Conversation memory** — context persists across sessions

## What's Built

| Capability | Status |
|---|---|
| Gmail integration (read, search, draft) | Done |
| Google Calendar (read, create events) | Done |
| RAG over local files (hierarchical: chunk + doc summaries) | Done |
| Persistent conversation memory with rolling summarization | Done |
| Multi-tool orchestration via Claude tool use | Done |
| Notion (search pages, read content, create pages) | Done |
| Google Drive (search files, read Google Docs, create Google Docs) | Done |
| Unified search across local files + Notion + Drive | Done |
| Slack integration (read channels, search, draft messages) | Done |
| Write actions with confirmation gate (preview + y/n confirm for Calendar, Drive, Notion) | Done |
| Multimodal (screenshots, images) | Planned |

## Tech Stack

- **LLM**: Claude Sonnet (Anthropic API) with native tool use
- **RAG**: ChromaDB (vector store) + sentence-transformers (embeddings) + PyMuPDF (PDF extraction)
- **Google APIs**: Gmail, Calendar, Drive + Docs (OAuth 2.0)
- **Notion API**: `notion-client` library, integration token auth
- **Slack API**: `slack-sdk`, bot token (channel history) + user token (search)
- **Multimodal** — vision-language understanding for screenshots and documents (planned)
- **Memory**: rolling summarization — last 10 exchanges verbatim, older turns condensed by Claude

## Project Structure

```
quicklocal/
├── src/
│   └── agent.py          # Agent loop, conversation memory, tool dispatch
├── tools/
│   ├── base_tool.py          # Tool interface
│   ├── gmail_tool.py         # Gmail read/search/draft
│   ├── calendar_tool.py      # Google Calendar read/create
│   ├── rag_tool.py           # Local file RAG
│   ├── notion_tool.py        # Notion search/read/create
│   ├── drive_tool.py         # Google Drive search/read/create
│   ├── unified_search_tool.py # Search all sources at once
│   ├── slack_tool.py         # Slack read/search/draft
│   ├── error_utils.py        # Error classification + retryable flag
│   ├── time_tool.py          # Current time/date
│   └── calculator_tool.py
├── scripts/
│   ├── index_docs.py              # Index local documents into ChromaDB
│   ├── search_docs.py             # CLI: search the vector store directly
│   ├── check_chunking.py          # Diagnostic: inspect chunks for sampled files
│   ├── check_summaries.py         # Diagnostic: inspect stored document summaries
│   └── check_pdf_extraction.py    # Diagnostic: compare pypdf/pdfplumber/pymupdf output
├── tests/
├── config.py             # Data directory config (reads from .env)
├── requirements.txt
└── .env                  # API keys and config (not committed)
```

## Setup

### Prerequisites

- Python 3.11+
- Anthropic API key
- Google Cloud project with Gmail, Calendar, Drive, and Docs APIs enabled

### Install

```bash
python -m venv qenv
source qenv/bin/activate
pip install -r requirements.txt
```

### Configure

Copy `.env.example` to `.env` and fill in:

```env
ANTHROPIC_API_KEY=your_key_here
GOOGLE_CREDENTIALS_PATH=credentials.json   # OAuth client credentials from Google Cloud Console
NOTION_TOKEN=your_notion_integration_token # From Notion → Settings → Connections → Integrations
SLACK_BOT_TOKEN=xoxb-...                   # From api.slack.com/apps → OAuth & Permissions → Bot Token
SLACK_USER_TOKEN=xoxp-...                  # From api.slack.com/apps → OAuth & Permissions → User Token (needs search:read scope)
QUICKLOCAL_DATA_DIRS=~/path/to/your/docs   # comma-separated
CONVERSATION_HISTORY_FILE=conversation_history.json
```

For Google APIs, create a project in Google Cloud Console, enable Gmail, Calendar, Drive, and Docs APIs, and download the OAuth client credentials as `credentials.json`. On first run per service, the agent opens a browser for consent — separate token files are created for each service (`token.json`, `calendar_token.json`, `drive_token.json`).

For Notion, create an integration at notion.so/my-integrations, copy the token, and share each page with the integration from the Notion sidebar.

### Index your documents

```bash
python scripts/index_docs.py
```

Supports `.pdf`, `.txt`, and `.md` files. On first index (or after a chunking strategy change), the agent also generates a 3–5 sentence Claude summary per document stored alongside the chunks. Subsequent searches auto-sync only changed files.

To inspect retrieval at runtime:

```bash
RAG_DEBUG=true python src/agent.py
```

This prints the retrieved chunks (source, score, size), document summaries included, and an estimated token count for the retrieval context.

Diagnostic scripts:

```bash
python scripts/check_chunking.py    # show chunks for 1 sampled file per type
python scripts/check_summaries.py   # show up to 5 stored doc summaries from ChromaDB
```

### Run

```bash
python src/agent.py
```

## Usage

```
You: What meetings do I have tomorrow?
You: Summarize the last 3 emails from my manager
You: What do my notes say about the Q3 roadmap?
You: Search everything for meeting notes about the API redesign
You: Find my Notion pages about the ML project
You: Create a Google Doc summarising today's standup
You: Create a Notion page under my Meeting Notes with a summary of last week
You: What's the latest in #general?
You: Search Slack for messages about the recommender project
You: Draft a message to #data-science saying the model review is confirmed for Friday
You: /clear    ← reset conversation
```

Type `quit` to exit.
