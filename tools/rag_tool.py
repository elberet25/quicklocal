import hashlib
import json
import logging
import sys
from pathlib import Path

import anthropic
import chromadb
import os
from sentence_transformers import SentenceTransformer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import get_data_dirs

try:
    from tools.base_tool import BaseTool
except ImportError:
    from base_tool import BaseTool

logger = logging.getLogger(__name__)

CHROMA_PATH = Path.home() / ".quicklocal" / "chroma_db"
MANIFEST_PATH = Path.home() / ".quicklocal" / "index_manifest.json"
COLLECTION_NAME = "local_docs"
EMBED_MODEL = "all-MiniLM-L6-v2"
MIN_CHUNK_CHARS = 50
MIN_MERGE_CHARS = 150  # chunks shorter than this are merged into the following chunk
SUPPORTED = {".pdf", ".txt", ".md"}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".webp"}

DOC_SUMMARY_MIN_CHARS = 300  # documents shorter than this skip the Claude call

# Increment whenever the chunking strategy or indexed file types change (triggers automatic full reindex)
CHUNKING_VERSION = 5


def _is_allowed(path: Path) -> bool:
    """Return True if path is within any configured data directory."""
    return any(path == d or path.is_relative_to(d) for d in get_data_dirs())


# ---------------------------------------------------------------------------
# Internal engine (shared singleton across both tools)
# ---------------------------------------------------------------------------

class RAGEngine:
    _instance = None

    @classmethod
    def get(cls) -> "RAGEngine":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        # suppress "unauthenticated HF Hub" noise — not needed for public cached models
        logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

        CHROMA_PATH.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=str(CHROMA_PATH))
        self._collection = self._client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        logger.debug("Loading embedding model %s", EMBED_MODEL)
        self._model = SentenceTransformer(EMBED_MODEL)

    # ------------------------------------------------------------------
    # Manifest helpers
    # ------------------------------------------------------------------

    def _load_manifest(self) -> dict:
        try:
            if MANIFEST_PATH.exists():
                return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass
        return {}

    def _save_manifest(self, manifest: dict) -> None:
        manifest["__chunking_version__"] = CHUNKING_VERSION
        try:
            MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        except OSError as e:
            logger.warning("Could not save manifest: %s", e)

    def _file_state(self, path: Path) -> dict:
        stat = path.stat()
        return {"mtime": stat.st_mtime, "size": stat.st_size}

    # ------------------------------------------------------------------
    # Chunking
    # ------------------------------------------------------------------

    def _clean_text(self, text: str) -> str:
        import re
        # Remove horizontal rule lines (3+ dashes, underscores, equals, asterisks)
        text = re.sub(r"^\s*[-_=*]{3,}\s*$", "", text, flags=re.MULTILINE)
        # Collapse 3+ consecutive blank lines into one paragraph break
        text = re.sub(r"\n{3,}", "\n\n", text)
        # Strip trailing whitespace from each line
        text = "\n".join(line.rstrip() for line in text.split("\n"))
        return text.strip()

    def _read_file(self, path: Path) -> str:
        suffix = path.suffix.lower()
        if suffix == ".pdf":
            try:
                # PyMuPDF (fitz) correctly maps custom font glyphs (e.g. bullet symbols)
                # to proper Unicode characters. pypdf renders them as \x7f (DEL), which
                # corrupts bullet-point lists in the index. Verified via check_pdf_extraction.py.
                import fitz
                doc = fitz.open(str(path))
                text = "\n\n".join(page.get_text() for page in doc)
                return self._clean_text(text)
            except Exception as e:
                logger.warning("Could not read PDF %s: %s", path, e)
                return ""
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
            return self._clean_text(text)
        except Exception as e:
            logger.warning("Could not read %s: %s", path, e)
            return ""

    def _chunk(self, text: str) -> list[str]:
        raw = [p.strip() for p in text.split("\n\n") if len(p.strip()) >= MIN_CHUNK_CHARS]

        # Forward pass: merge short chunks into the following chunk to prevent
        # isolated bullet items and transitional fragments.
        merged = []
        i = 0
        while i < len(raw):
            chunk = raw[i]
            while len(chunk) < MIN_MERGE_CHARS and i + 1 < len(raw):
                i += 1
                chunk = chunk + "\n\n" + raw[i]
            merged.append(chunk)
            i += 1

        # Backward pass: if the last chunk is a short trailing fragment (e.g. a
        # single bullet split off by a page break), absorb it into the previous chunk.
        if len(merged) > 1 and len(merged[-1]) < MIN_MERGE_CHARS * 2:
            merged[-2] = merged[-2] + "\n\n" + merged[-1]
            merged.pop()

        return merged

    # ------------------------------------------------------------------
    # Document summarization
    # ------------------------------------------------------------------

    def _summarize_document(self, text: str, filename: str) -> str:
        if len(text) < DOC_SUMMARY_MIN_CHARS:
            return text
        try:
            client = anthropic.Anthropic()
            truncated = text[:8000]
            response = client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=200,
                messages=[{
                    "role": "user",
                    "content": (
                        f"Summarize this document in 3-5 sentences, focusing on its main topics "
                        f"and key information. Document: {filename}\n\n{truncated}"
                    ),
                }],
            )
            return response.content[0].text.strip()
        except Exception as e:
            logger.warning("Could not summarize %s: %s", filename, e)
            return ""

    def _describe_image(self, file: Path) -> str:
        """Return a rich text description of an image via Claude Vision (Haiku).

        Used at index time so images are discoverable via semantic search without
        requiring a Vision API call on every query. Returns "" on any failure or
        if the API returns an empty response — caller should skip indexing in that case.
        """
        import base64
        try:
            media_types = {
                ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
                ".png": "image/png", ".gif": "image/gif", ".webp": "image/webp",
            }
            media_type = media_types.get(file.suffix.lower(), "image/png")
            image_data = base64.standard_b64encode(file.read_bytes()).decode("utf-8")
            client = anthropic.Anthropic()
            response = client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=300,
                messages=[{
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {"type": "base64", "media_type": media_type, "data": image_data},
                        },
                        {
                            "type": "text",
                            "text": (
                                "Describe this image in detail for the purpose of making it searchable. "
                                "Cover what it shows, key concepts, any visible text, and the content "
                                "type (diagram, chart, screenshot, photo, etc.)."
                            ),
                        },
                    ],
                }],
            )
            description = response.content[0].text.strip()
            if not description:
                logger.warning("Vision API returned empty description for %s", file.name)
                return ""
            return description
        except Exception as e:
            logger.warning("Could not describe image %s: %s", file.name, e)
            return ""

    # ------------------------------------------------------------------
    # Core indexing (single file)
    # ------------------------------------------------------------------

    def _index_file(self, file: Path) -> int:
        """Delete existing chunks for file, re-embed and insert. Returns chunk count."""
        text = self._read_file(file)
        chunks = self._chunk(text)
        if not chunks:
            return 0

        self._collection.delete(where={"source": str(file)})

        ids, docs, metas = [], [], []
        for i, chunk in enumerate(chunks):
            chunk_id = hashlib.md5(f"{file}::{i}::{chunk[:80]}".encode()).hexdigest()
            ids.append(chunk_id)
            docs.append(chunk)
            metas.append({"source": str(file), "chunk_index": i, "level": "chunk"})

        embeddings = self._model.encode(docs, show_progress_bar=False).tolist()
        self._collection.add(ids=ids, documents=docs, metadatas=metas, embeddings=embeddings)

        summary = self._summarize_document(text, file.name)
        if summary:
            summary_id = hashlib.md5(f"{file}::doc_summary".encode()).hexdigest()
            summary_embedding = self._model.encode([summary], show_progress_bar=False).tolist()
            self._collection.add(
                ids=[summary_id],
                documents=[summary],
                metadatas=[{"source": str(file), "level": "document"}],
                embeddings=summary_embedding,
            )

        return len(chunks)

    def _index_image_file(self, file: Path) -> int:
        """Describe and index a single image file. Returns 1 on success, 0 on failure.

        A return value of 0 means the manifest should NOT be updated so the file
        is retried on the next sync (unlike empty text files, a Vision API failure
        is transient and worth retrying).
        """
        self._collection.delete(where={"source": str(file)})
        description = self._describe_image(file)
        if not description:
            return 0

        doc = f"[IMAGE] {file.name}\nPath: {file}\nDescription: {description}"
        doc_id = hashlib.md5(f"{file}::image".encode()).hexdigest()
        embedding = self._model.encode([doc], show_progress_bar=False).tolist()
        self._collection.add(
            ids=[doc_id],
            documents=[doc],
            metadatas=[{"source": str(file), "level": "chunk", "type": "image"}],
            embeddings=embedding,
        )
        return 1

    # ------------------------------------------------------------------
    # Auto-sync (called lazily on search)
    # ------------------------------------------------------------------

    def _sync_directory(self, directory: Path) -> None:
        """Index new/modified files and drop chunks for deleted files.

        Compares each file's mtime + size against the manifest. Only files
        that changed since the last index run are re-embedded, so repeated
        searches over an unchanged directory cost just a directory scan.

        If CHUNKING_VERSION has changed since the manifest was written, all
        files in this directory are force-reindexed with the new strategy.
        """
        manifest = self._load_manifest()

        if manifest.get("__chunking_version__") != CHUNKING_VERSION:
            logger.debug("Chunking version changed — forcing reindex of %s", directory)
            for key in list(manifest.keys()):
                if key.startswith("__"):
                    continue
                if Path(key).is_relative_to(directory):
                    self._collection.delete(where={"source": key})
            manifest = {"__chunking_version__": CHUNKING_VERSION}

        current_files = {
            f: self._file_state(f)
            for f in directory.rglob("*")
            if f.suffix.lower() in SUPPORTED or f.suffix.lower() in IMAGE_EXTENSIONS
        }

        changed = False

        for file, state in current_files.items():
            if manifest.get(str(file)) != state:
                logger.debug("Auto-indexing changed file: %s", file.name)
                if file.suffix.lower() in IMAGE_EXTENSIONS:
                    n = self._index_image_file(file)
                    if n > 0:
                        manifest[str(file)] = state
                        changed = True
                else:
                    self._index_file(file)
                    manifest[str(file)] = state
                    changed = True

        for key in list(manifest.keys()):
            if key.startswith("__"):
                continue
            path = Path(key)
            if path.is_relative_to(directory) and not path.exists():
                logger.debug("Removing deleted file from index: %s", path.name)
                self._collection.delete(where={"source": key})
                del manifest[key]
                changed = True

        if changed:
            self._save_manifest(manifest)

    # ------------------------------------------------------------------
    # Explicit full-directory index (called by IndexDocumentsTool)
    # ------------------------------------------------------------------

    def index_directory(self, directory: Path) -> tuple[int, list[str]]:
        """Force-index all supported files in directory. Updates manifest."""
        files = [
            f for f in directory.rglob("*")
            if f.suffix.lower() in SUPPORTED or f.suffix.lower() in IMAGE_EXTENSIONS
        ]
        manifest = self._load_manifest()

        total_chunks = 0
        indexed_files = []

        for file in files:
            if file.suffix.lower() in IMAGE_EXTENSIONS:
                n = self._index_image_file(file)
            else:
                n = self._index_file(file)
            if n > 0:
                manifest[str(file)] = self._file_state(file)
                total_chunks += n
                indexed_files.append(file.name)

        # Clean up stale manifest entries for files removed from this directory
        for key in list(manifest.keys()):
            if key.startswith("__"):
                continue
            path = Path(key)
            if path.is_relative_to(directory) and not path.exists():
                self._collection.delete(where={"source": key})
                del manifest[key]

        self._save_manifest(manifest)
        return total_chunks, indexed_files

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(self, query: str, n_results: int = 3) -> list[dict]:
        """Auto-sync all configured data dirs, then return best-matching chunks with doc summaries."""
        if not query or not query.strip():
            raise ValueError("query must not be empty")

        for directory in get_data_dirs():
            if directory.exists():
                self._sync_directory(directory)

        chunk_ids = self._collection.get(where={"level": "chunk"}, include=[])["ids"]
        chunk_count = len(chunk_ids)
        if chunk_count == 0:
            return []

        n_results = min(n_results, chunk_count)
        query_embedding = self._model.encode([query], show_progress_bar=False).tolist()
        results = self._collection.query(
            query_embeddings=query_embedding,
            n_results=n_results,
            where={"level": "chunk"},
            include=["documents", "metadatas", "distances"],
        )

        # Fetch document summaries for each unique source file in results
        unique_source_paths = list(dict.fromkeys(
            meta["source"] for meta in results["metadatas"][0]
        ))
        doc_summaries: dict[str, str] = {}
        for source_path in unique_source_paths:
            sr = self._collection.get(
                where={"$and": [{"source": {"$eq": source_path}}, {"level": {"$eq": "document"}}]},
                include=["documents"],
            )
            if sr["documents"]:
                doc_summaries[Path(source_path).name] = sr["documents"][0]

        hits = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            source_name = Path(meta["source"]).name
            hits.append({
                "source": source_name,
                "text": doc,
                "score": round(1 - dist, 3),  # cosine distance → similarity
                "doc_summary": doc_summaries.get(source_name, ""),
            })

        return hits


# ---------------------------------------------------------------------------
# Tool: index_documents
# ---------------------------------------------------------------------------

class IndexDocumentsTool(BaseTool):
    name = "index_documents"
    category = "rag"

    def get_description(self) -> dict:
        allowed = ", ".join(str(d) for d in get_data_dirs())
        return {
            "name": self.name,
            "description": (
                "Index local documents (PDF, TXT, MD) and images (PNG, JPG, JPEG, GIF, WebP) "
                "into the RAG vector store. Images are described via Claude Vision at index time "
                "so they are searchable by content. "
                f"Allowed directories: {allowed}. Any other path will be rejected. "
                "Omit directory to index all configured directories."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "directory": {
                        "type": "string",
                        "description": (
                            "Directory to index. Must be one of the configured data "
                            f"directories (or a subdirectory): {allowed}. "
                            "Omit to index all configured directories."
                        ),
                    },
                },
                "required": [],
            },
        }

    def execute(self, **kwargs) -> dict:
        try:
            raw_dir = kwargs.get("directory")
            if raw_dir is not None:
                requested = Path(raw_dir).expanduser().resolve()
                if not _is_allowed(requested):
                    allowed = ", ".join(str(d) for d in get_data_dirs())
                    return {"error": f"Access restricted. Allowed directories: {allowed}"}
                dirs_to_index = [requested]
            else:
                dirs_to_index = [d for d in get_data_dirs() if d.exists()]

            total_chunks, all_files = 0, []
            for directory in dirs_to_index:
                n, files = RAGEngine.get().index_directory(directory)
                total_chunks += n
                all_files.extend(files)

            if not all_files:
                return {"result": "No PDF/TXT/MD/image files found in configured directories."}
            return {
                "result": (
                    f"Indexed {len(all_files)} file(s), {total_chunks} chunk(s) total.\n"
                    f"Files: {', '.join(all_files)}"
                )
            }
        except Exception as e:
            return self.handle_error(e)


# ---------------------------------------------------------------------------
# Tool: search_documents
# ---------------------------------------------------------------------------

class SearchDocumentsTool(BaseTool):
    name = "search_documents"
    category = "rag"
    summarizable = True

    def get_description(self) -> dict:
        return {
            "name": self.name,
            "description": (
                "Search indexed local documents using semantic similarity. "
                "Returns the most relevant text chunks from PDF, TXT, and MD files, "
                "plus any indexed images (marked [IMAGE] in results). "
                "Image results include a Vision-generated description and the full file path. "
                "Use the indexed description to answer general questions about an image; "
                "call analyze_image with the path only when you need deeper analysis "
                "beyond what the description already covers. "
                "Use this to answer questions about the user's documents."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The natural-language search query.",
                    },
                    "n_results": {
                        "type": "integer",
                        "description": "Number of results to return (default 3, max 20).",
                        "default": 3,
                    },
                },
                "required": ["query"],
            },
        }

    def execute(self, **kwargs) -> dict:
        try:
            query = kwargs.get("query", "").strip()
            if not query:
                return {"error": "query is required"}
            n_results = min(int(kwargs.get("n_results", 3)), 20)
            hits = RAGEngine.get().search(query, n_results)
            if not hits:
                return {"result": "No documents indexed yet. Use the index_documents tool first."}

            parts = []

            # Prepend deduplicated document summaries as context
            seen: dict[str, str] = {}
            for h in hits:
                if h["doc_summary"] and h["source"] not in seen:
                    seen[h["source"]] = h["doc_summary"]
            if seen:
                context_lines = [f"[{src}]\n{summary}" for src, summary in seen.items()]
                parts.append("=== Document Context ===\n\n" + "\n\n".join(context_lines))

            chunk_parts = [f"[{h['source']}] (score: {h['score']})\n{h['text']}" for h in hits]
            parts.append("=== Relevant Chunks ===\n\n" + "\n\n---\n\n".join(chunk_parts))

            if os.environ.get("RAG_DEBUG", "").lower() == "true":
                print(f"\n[RAG_DEBUG] Retrieved {len(hits)} chunk(s):")
                for h in hits:
                    print(f"  [{h['source']}] score={h['score']} ({len(h['text'])} chars)")
                unique_summaries = {h["source"]: h["doc_summary"] for h in hits if h["doc_summary"]}
                print(f"\n[RAG_DEBUG] Document summaries ({len(unique_summaries)}):")
                for src, summary in unique_summaries.items():
                    print(f"  [{src}] {summary[:120]}…")
                total_chars = sum(len(h["text"]) for h in hits) + sum(len(s) for s in unique_summaries.values())
                print(f"\n[RAG_DEBUG] Estimated retrieval context: ~{total_chars // 4} tokens ({total_chars} chars)\n")

            return {"result": "\n\n".join(parts)}
        except Exception as e:
            return self.handle_error(e)


# ---------------------------------------------------------------------------
# Module-level convenience functions (for scripts / direct use)
# ---------------------------------------------------------------------------

def index_documents() -> int:
    """Index all PDF/TXT/MD files across all configured data dirs. Returns total chunk count."""
    total = 0
    for directory in get_data_dirs():
        if directory.exists():
            n, _ = RAGEngine.get().index_directory(directory)
            total += n
    return total


def search_documents(query: str, n: int = 3) -> list[dict]:
    """Search indexed documents. Returns list of {source, text, score, doc_summary} dicts."""
    return RAGEngine.get().search(query, n_results=n)
