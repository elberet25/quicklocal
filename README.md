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

## What's Built (MVP)

| Capability | Status |
|---|---|
| Gmail integration (read, search, draft) | Done |
| Google Calendar (read, create events) | Done |
| RAG over local files (ChromaDB + sentence-transformers) | Done |
| Persistent conversation memory with rolling summarization | Done |
| Multi-tool orchestration via Claude tool use | Done |
| Slack integration | Planned |
| Notion / Google Drive | Planned |
| Multimodal (screenshots, images) | Planned |

## Tech Stack

- **LLM**: Claude Sonnet (Anthropic API) with native tool use
- **RAG**: ChromaDB (vector store) + sentence-transformers (embeddings)
- **Google APIs**: Gmail, Calendar, Drive (OAuth 2.0)
- **Notion API** (planned)
- **Slack API** (planned)
- **Multimodal** — vision-language understanding for screenshots and documents (planned)
- **Memory**: rolling summarization — last 10 exchanges verbatim, older turns condensed by Claude

## Project Structure

```
quicklocal/
├── src/
│   └── agent.py          # Agent loop, conversation memory, tool dispatch
├── tools/
│   ├── base_tool.py      # Tool interface
│   ├── gmail_tool.py     # Gmail read/search/draft
│   ├── calendar_tool.py  # Google Calendar read/create
│   ├── rag_tool.py       # Local file RAG
│   ├── time_tool.py      # Current time/date
│   └── calculator_tool.py
├── scripts/
│   ├── index_docs.py     # Index local documents into ChromaDB
│   └── search_docs.py    # Debug: search the vector store directly
├── tests/
├── config.py             # Data directory config (reads from .env)
├── requirements.txt
└── .env                  # API keys and config (not committed)
```

## Setup

### Prerequisites

- Python 3.11+
- Anthropic API key
- Google Cloud project with Gmail and Calendar APIs enabled

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
QUICKLOCAL_DATA_DIRS=~/path/to/your/docs   # comma-separated
CONVERSATION_HISTORY_FILE=conversation_history.json
```

For Google APIs, create a project in Google Cloud Console, enable Gmail and Calendar APIs, and download the OAuth client credentials as `credentials.json`. On first run, the agent will open a browser for consent.

### Index your documents

```bash
python scripts/index_docs.py
```

### Run

```bash
python src/agent.py
```

## Usage

```
You: What meetings do I have tomorrow?
You: Summarize the last 3 emails from my manager
You: What does my notes say about the Q3 roadmap?
You: /clear    ← reset conversation
```

Type `quit` to exit.
