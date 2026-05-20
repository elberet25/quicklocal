# QuickLocal — Claude Code Context

## TL;DR

- **What**: Local AI work assistant (CLI) that orchestrates Gmail, Calendar, and local-file RAG — built as an interview portfolio project for an Amazon Quick Team role.
- **Last built (Session 4)**: RAG system with ChromaDB + sentence-transformers, manifest-based incremental reindexing, auto-sync on search, and path-restriction security.
- **Coming next (Session 5)**: Notion and Google Drive integrations, unified search across all knowledge sources.
- **Critical constraint**: Do not swap ChromaDB or change the paragraph-based chunking strategy without discussion — the incremental sync system is tightly coupled to both.

---

## End-of-Session Checklist

At the end of every session, update this file:
- **Session Log** — add a paragraph summarizing what was built and any key decisions made
- **Current Project State** — update the table (mark completed items, add new ones)
- **Architectural Decisions / Known Issues** — record anything new discovered during the session

---

## Project Vision & Purpose

**QuickLocal** is a local AI work assistant that connects to the user's digital workspace (Gmail, Calendar, local files) to help manage work, surface insights, and create deliverables. It is built as an **interview portfolio project** targeting an Amazon Quick Team role, demonstrating direct understanding of agentic AI, multi-tool orchestration, and personalized knowledge systems.

The user's background: strong Python and ML/AI foundations (Chip Huyen's book), new to building agentic systems, building with Claude Code assistance. The goal is a polished, technically deep project that can anchor interview conversations about Amazon Quick.

---

## Architecture Overview

```
User (CLI)
    │
    ▼
src/agent.py  ←  main()
│   ├── load_history()  ←→  conversation_history.json (disk)
│   ├── run_agent_turn()
│   │     └── anthropic.Client.messages.create(model, tools, messages)
│   │           │
│   │           ▼  stop_reason="tool_use"
│   │         execute_tool(name, input)
│   │           │
│   │           ▼
│   │         tool_registry  ←  {name: BaseTool instance}
│   │           └── tool.execute(**kwargs) → {"result": ...} | {"error": ...}
│   │
│   └── summarize_old_exchanges()  (rolling summary if > MAX_VERBATIM_EXCHANGES)
│
├── tools/
│   ├── BaseTool (ABC)         get_description() + execute() + validate_input() + handle_error()
│   ├── TimeTool               get current time
│   ├── CalculatorTool         basic arithmetic
│   ├── GmailBaseTool          shared OAuth2 + message parsing
│   │   ├── ReadLatestEmailsTool
│   │   ├── SearchEmailsTool
│   │   ├── PreviewDraftReplyTool   ← shows draft WITHOUT saving
│   │   └── CreateDraftReplyTool   ← saves draft (only after user confirmation)
│   ├── CalendarBaseTool       shared OAuth2 + event parsing
│   │   ├── GetScheduleTool
│   │   ├── FindFreeTimeTool
│   │   └── CreateEventTool
│   └── RAGEngine (singleton)  ChromaDB + sentence-transformers
│       ├── IndexDocumentsTool
│       └── SearchDocumentsTool
│
├── config.py                  get_data_dirs() — reads QUICKLOCAL_DATA_DIRS from .env
└── scripts/
    └── index_docs.py          CLI wrapper for indexing
```

**Data flow for a tool call:**
1. User types a message → appended to `conversation` list
2. `run_agent_turn()` calls Claude API with full conversation + tool schemas
3. If `stop_reason == "tool_use"`: extract tool block → `execute_tool()` → append result back to conversation as `tool_result` user message
4. Loop until `stop_reason == "end_turn"` → return final text
5. After each turn: `summarize_old_exchanges()` if needed, then `save_history()`

---

## Current Project State

### Fully Implemented (Sessions 1–4)

| Component | File(s) | Status |
|-----------|---------|--------|
| Agent loop + CLI | `src/agent.py` | Complete |
| Tool registry + BaseTool ABC | `tools/__init__.py`, `tools/base_tool.py` | Complete |
| TimeTool, CalculatorTool | `tools/time_tool.py`, `tools/calculator_tool.py` | Complete |
| Gmail (read, search, draft preview, draft create) | `tools/gmail_tool.py` | Complete |
| Google Calendar (schedule, free time, create event) | `tools/calendar_tool.py` | Complete |
| RAG system (index + search, ChromaDB, auto-sync) | `tools/rag_tool.py` | Complete |
| Conversation memory (persist, summarize, strip) | `src/agent.py` | Complete |
| Config (data dir management) | `config.py` | Complete |
| Unit tests (all tools + memory) | `tests/` | Complete |

### Not Yet Implemented (Sessions 5–10)

- **Session 5**: Notion integration, Google Drive integration, unified search across all sources
- **Session 6**: Slack integration (read channels, search, draft messages)
- **Session 7**: User confirmation system for write actions (create docs, send Slack), Google Doc creation
- **Session 8**: Multimodal — screenshot/image analysis via Claude Vision API
- **Session 9** (optional): Document OCR, Google Slides presentation builder
- **Session 10**: Streamlit UI, error handling improvements, demo video, GitHub polish

---

## Architectural Decisions & Rationale

**ChromaDB over FAISS**
ChromaDB's `PersistentClient` stores embeddings to `~/.quicklocal/chroma_db/` on disk. FAISS is in-memory by default and would require full re-embedding on every agent startup, which is expensive with `sentence-transformers`.

**Paragraph-based chunking (`text.split("\n\n")`)**
Paragraphs are natural semantic units — they preserve coherent ideas. Fixed-size word windows (e.g., 500 words) cut mid-sentence and break semantic coherence. Chunks shorter than `MIN_CHUNK_CHARS=50` are dropped as noise.

**Manifest-based incremental reindexing**
`~/.quicklocal/index_manifest.json` records `mtime + size` for each indexed file. `_sync_directory()` is called lazily on every search; only files whose mtime/size changed since last index are re-embedded. This avoids re-embedding an entire document corpus on every query while keeping the index fresh.

**Tool call stripping from saved history (`_strip_tool_blocks()`)**
Raw `tool_use`/`tool_result` blocks are stripped before writing `conversation_history.json`. Tool results reflect external state at the time of the call (e.g., specific emails, calendar events) — saving them would cause Claude to trust stale data in future sessions. Claude's final text responses capture the outcome in a session-agnostic way.

**Separate OAuth token files for Gmail vs Calendar**
Gmail uses `token.json`; Calendar uses `calendar_token.json`. Combining scopes into one token file can cause authorization failures if scopes are requested incrementally. Separate files allow each service to be independently authorized and refreshed.

**Two-tool draft confirmation pattern (email)**
`preview_draft_reply` formats the draft and returns it without any API call. `create_draft_reply` saves it to Gmail. Tool descriptions instruct Claude to always call preview first and wait for explicit user confirmation. The safety gate is enforced at the LLM instruction level, not in application code — this is intentional since it degrades gracefully if Claude ignores it (the user just sees an unexpected save).

**RAGEngine singleton**
`RAGEngine.get()` returns a single shared instance per process. Loading `all-MiniLM-L6-v2` via `sentence-transformers` takes ~1–2 seconds. A singleton avoids reloading the model on every tool call.

**Conversation summarization aligned to text message boundaries**
The summarizer finds the cutoff point by walking backward until it hits a message where `role == "user"` and `content` is a plain string (not a list). This prevents the cutoff from landing between a `tool_use` and its `tool_result` message, which would produce orphaned `tool_result` blocks that cause Claude API 400 errors.

**`_sanitize_history()` as a safety net on load**
On startup, orphaned `tool_result` blocks (missing corresponding `tool_use`) are dropped from loaded history. This handles history files written before the summarizer's alignment fix was added, and any edge case that produces orphans. Without it, the first API call in a resumed session would return a 400.

**`_serialize_block()` for SDK content blocks**
The Anthropic Python SDK's `model_dump()` includes internal fields (e.g., `citations=None`) that the API rejects when the message is sent back in subsequent turns. `_serialize_block()` explicitly constructs minimal dicts with only the fields the API accepts.

**Path restriction in RAG (`_is_allowed()`)**
Before indexing, the requested directory is checked against `QUICKLOCAL_DATA_DIRS`. This prevents prompt injection attacks where a malicious document could instruct the agent to index `/etc/passwd` or similar. Only configured dirs (and subdirs) are permitted.

---

## Project File Structure

```
quicklocal/
├── src/
│   ├── __init__.py
│   └── agent.py                    # CLI entry point; all conversation and agent logic
├── tools/
│   ├── __init__.py                 # ALL_TOOLS list — all tool instances registered here
│   ├── base_tool.py                # BaseTool ABC: get_description, execute, validate_input, handle_error
│   ├── time_tool.py                # TimeTool
│   ├── calculator_tool.py          # CalculatorTool
│   ├── gmail_tool.py               # GmailBaseTool + 4 Gmail tool classes
│   ├── calendar_tool.py            # CalendarBaseTool + 3 Calendar tool classes
│   └── rag_tool.py                 # RAGEngine singleton + IndexDocumentsTool + SearchDocumentsTool
├── tests/
│   ├── conftest.py                 # sys.path setup only
│   ├── test_agent.py               # TimeTool, CalculatorTool, execute_tool dispatcher
│   ├── test_calendar_tool.py       # All Calendar tools (API mocked)
│   ├── test_conversation_memory.py # load/save/clear, summarize, strip, sanitize
│   ├── test_gmail_tool.py          # All Gmail tools (API mocked)
│   ├── test_rag_tool.py            # IndexDocumentsTool, SearchDocumentsTool, _is_allowed
│   └── test_setup.py               # Basic environment/import checks
├── scripts/
│   ├── index_docs.py               # CLI: index all configured data dirs
│   ├── search_docs.py              # CLI: semantic search
│   └── populate_slack.py           # Utility: seed test Slack data
├── docs/
│   ├── quicklocal_project_overview.md    # Project vision, goals, interview strategy
│   └── quicklocal_implementation_guide.md # Session-by-session build plan
├── config.py                       # get_data_dirs() — reads QUICKLOCAL_DATA_DIRS from .env
├── requirements.txt
├── .env                            # API keys + config (not in git)
├── credentials.json                # Google OAuth client credentials (not in git)
├── token.json                      # Gmail OAuth tokens (not in git)
├── calendar_token.json             # Calendar OAuth tokens (not in git)
└── conversation_history.json       # Persisted conversation (not in git, regenerated)
```

**Key runtime paths (outside the repo):**
- `~/.quicklocal/chroma_db/` — ChromaDB vector store
- `~/.quicklocal/index_manifest.json` — RAG file-state tracking
- `~/quicklocal_test_data/` — default data directory (configurable via `QUICKLOCAL_DATA_DIRS`)

---

## Coding Conventions

**Tool structure**
- Every tool is a class inheriting `BaseTool`, with a `name: str` class attribute
- `get_description()` returns the exact dict Claude API expects for tool definitions
- `execute(**kwargs)` always returns `{"result": "..."}` or `{"error": "..."}` — never raises
- `validate_input()` is defined on `BaseTool` and tested in isolation, but is not yet wired into the dispatcher. It will be connected when the user-confirmation system is built (planned for Session 7).
- `handle_error(e)` is inherited from BaseTool and returns `{"error": str(e)}`

**Tool descriptions (for Claude)**
- Written to guide Claude's selection: include specific use cases and sometimes explicit "don't use unless" constraints
- The `preview_draft_reply` description says "ALWAYS call this first" — LLM-level enforcement of the safety pattern

**Auth pattern (Google tools)**
- Auth helpers are `@classmethod` on the base class (`GmailBaseTool`, `CalendarBaseTool`), shared across all subtools of that service
- Credentials path and token path are configurable via env vars (`GOOGLE_CREDENTIALS_PATH`, `GMAIL_TOKEN_FILE`, `CALENDAR_TOKEN_FILE`)

**Error handling**
- All tool `execute()` methods wrap their logic in `try/except Exception` → `self.handle_error(e)`
- Logging uses Python `logging` module: `logger.debug` for tool dispatch, `logger.warning` for non-fatal failures (I/O, manifest write)
- Agent-level errors (unexpected `stop_reason`) raise `RuntimeError` intentionally

**Testing**
- All tests mock external APIs; no real Google/Anthropic calls in the test suite
- `monkeypatch.setenv("QUICKLOCAL_DATA_DIRS", ...)` is used to isolate config in RAG tests
- `patch.object(BaseTool, "_get_service", ...)` pattern for Google API mocking
- Test class names: `TestToolNameBehavior` (e.g., `TestFindFreeTimeTool`, `TestStripToolBlocks`)

**Naming**
- snake_case everywhere (Python standard)
- Tool names in `get_description()` match the class attribute `name` exactly (these are sent to Claude API)
- History file is configurable via `CONVERSATION_HISTORY_FILE` env var (useful for testing)

**venv**
The virtual environment is named `qenv` (not `.venv`). Activate with `source qenv/bin/activate`.

**pytest config**
`tests/conftest.py` handles sys.path setup. There is NO root-level `conftest.py` — do not create one.

---

## Constraints & Guardrails

These should NOT be changed without explicit discussion:

1. **Do not swap ChromaDB** — the manifest sync system uses ChromaDB's `where={"source": ...}` filter API for targeted chunk deletion. Switching to FAISS or another store requires redesigning the incremental sync.

2. **Do not change the paragraph-based chunking strategy** without evaluating retrieval quality. The split is `text.split("\n\n")` with a `MIN_CHUNK_CHARS=50` filter. Changing chunk boundaries invalidates the entire existing index.

3. **Do not modify the conversation memory format** (summary pair sentinel text, strip rules, sanitize logic) without updating all related tests in `test_conversation_memory.py`. The tests assert on specific content strings.

4. **Do not add direct email sending** — `CreateDraftReplyTool` saves to Gmail Drafts only. Sending requires explicit human action in Gmail. This is intentional for safety.

5. **Do not remove the preview → create two-step for email drafts.** The pattern is intentional user-safety architecture.

6. **Do not commit credentials, tokens, or .env** — `credentials.json`, `token.json`, `calendar_token.json`, `.env` are in `.gitignore`.

7. **Do not remove `_sanitize_history()`** — it protects against corrupt/legacy history files that would otherwise cause Claude API 400 errors on session resume.

8. **Do not create a root-level `conftest.py`** — pytest configuration and sys.path setup belongs in `tests/conftest.py` only. A root conftest was removed; do not re-add it.

---

## Known Issues & Tech Debt

- **RAG quality is basic**: paragraph cosine similarity only. No re-ranking, no hybrid keyword+semantic search, no metadata filtering by date or source type. Retrieval fails on short queries and ambiguous terms.

- **`validate_input()` is not yet wired into the dispatcher**: `execute_tool()` calls `tool.execute()` directly. This is intentional — `validate_input()` will be connected when the user-confirmation system is built in Session 7.

- **`CreateEventTool` lacks a confirmation preview step**: Calendar events are created immediately. This is inconsistent with the email draft pattern. A `PreviewEventTool` → `CreateEventTool` two-step should be added for consistency with the user-safety architecture.

- **Conversation summarization skips tool call content**: When building the transcript sent to Claude for summarization, `tool_use`/`tool_result` blocks are omitted (they are list-content messages). This means that meaningful tool results (specific email subjects, calendar details) are lost in summaries. A proper solution would include a human-readable description of tool calls in the transcript.

- **No async/parallel tool execution**: All tools execute sequentially even when they're independent (e.g., searching Gmail and Calendar at the same time). The agent loop in `run_agent_turn()` is synchronous. This is a Session 10 enhancement candidate.

- **No retry/backoff on API failures**: Google API calls and the Anthropic API call have no exponential backoff. A rate-limited or transient failure surfaces directly as an error to the user.

- **RAG auto-sync runs on every search**: `_sync_directory()` does a full directory scan (stat every file) on every `SearchDocumentsTool` call. On large directories this adds latency. A file-system watcher or TTL cache would be better.

- **Structural chunking mismatch for txt files**: `\n\n`-based splitting works well for prose (markdown, PDFs) but produces poor chunks for structured txt files like meeting notes, which organize content around speaker turns, agenda sections, and explicit separators (`------`). The semantic unit in these files is "everything in a section" not "a paragraph between blank lines." Potential improvement: structure-aware chunking that detects section delimiters and speaker/header patterns and uses those as chunk boundaries instead of blank lines. Deferred — current cleaning pass (strip separators, normalize whitespace) mitigates the worst artifacts, and hierarchical document summaries (Step 3) compensate for poor chunk boundaries at query time.

- **`_serialize_block()` is a workaround for SDK behavior**: It exists because `response.content[i].model_dump()` includes `citations=None` and other fields the API rejects in subsequent turns. This may break if the SDK changes its response format.

---

## Session Log

**Session 1** — Built the core agent loop using the Anthropic Python SDK (`claude-sonnet-4-6`). Implemented `BaseTool` ABC, `ALL_TOOLS` registry in `tools/__init__.py`, and two simple tools: `TimeTool` and `CalculatorTool`. Established the `tool_use` → `tool_result` conversation loop pattern and confirmed the agent dispatches tools correctly.

**Session 2** — Added full Gmail integration via Google OAuth2. Implemented `GmailBaseTool` shared base class (auth + `_parse_message` + `_fetch_messages`), and four tools: `ReadLatestEmailsTool`, `SearchEmailsTool`, `PreviewDraftReplyTool`, `CreateDraftReplyTool`. Introduced the two-step draft preview-before-save safety pattern enforced at the LLM description level.

**Session 3** — Added Google Calendar integration: `CalendarBaseTool` with separate `calendar_token.json`, and three tools: `GetScheduleTool` (events for a date), `FindFreeTimeTool` (gaps ≥30min within working hours), `CreateEventTool`. Implemented the full conversation memory system: JSON persistence, `_strip_tool_blocks()` to avoid stale tool data across sessions, rolling summarization with `summarize_old_exchanges()` aligned to text-message boundaries, `_sanitize_history()` as an orphan-protection safety net. Added comprehensive unit tests for all memory behaviors.

**Session 4** — Built the local RAG system. Implemented `RAGEngine` singleton using ChromaDB `PersistentClient` and `sentence-transformers` (`all-MiniLM-L6-v2`). Chunking is paragraph-based (`\n\n` split, MIN_CHUNK_CHARS=50). Manifest-based incremental reindexing tracks mtime+size to avoid re-embedding unchanged files. Auto-sync runs on every search query across all configured data directories. Path restriction (`_is_allowed()`) prevents indexing outside configured dirs. Added `IndexDocumentsTool`, `SearchDocumentsTool`, `config.py` (QUICKLOCAL_DATA_DIRS), and `scripts/index_docs.py`. Full unit test coverage with mocked ChromaDB.
