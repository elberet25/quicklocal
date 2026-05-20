import hashlib
import json
import logging
import sys
from pathlib import Path

import chromadb
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
SUPPORTED = {".pdf", ".txt", ".md"}


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

    def _read_file(self, path: Path) -> str:
        suffix = path.suffix.lower()
        if suffix == ".pdf":
            try:
                from pypdf import PdfReader
                reader = PdfReader(str(path))
                return "\n\n".join(page.extract_text() or "" for page in reader.pages)
            except Exception as e:
                logger.warning("Could not read PDF %s: %s", path, e)
                return ""
        try:
            return path.read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            logger.warning("Could not read %s: %s", path, e)
            return ""

    def _chunk(self, text: str) -> list[str]:
        raw = [p.strip() for p in text.split("\n\n")]
        return [p for p in raw if len(p) >= MIN_CHUNK_CHARS]

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
            metas.append({"source": str(file), "chunk_index": i})

        embeddings = self._model.encode(docs, show_progress_bar=False).tolist()
        self._collection.add(ids=ids, documents=docs, metadatas=metas, embeddings=embeddings)
        return len(chunks)

    # ------------------------------------------------------------------
    # Auto-sync (called lazily on search)
    # ------------------------------------------------------------------

    def _sync_directory(self, directory: Path) -> None:
        """Index new/modified files and drop chunks for deleted files.

        Compares each file's mtime + size against the manifest. Only files
        that changed since the last index run are re-embedded, so repeated
        searches over an unchanged directory cost just a directory scan.
        """
        manifest = self._load_manifest()
        current_files = {
            f: self._file_state(f)
            for f in directory.rglob("*")
            if f.suffix.lower() in SUPPORTED
        }

        changed = False

        for file, state in current_files.items():
            if manifest.get(str(file)) != state:
                logger.debug("Auto-indexing changed file: %s", file.name)
                self._index_file(file)
                manifest[str(file)] = state
                changed = True

        for key in list(manifest.keys()):
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
        files = [f for f in directory.rglob("*") if f.suffix.lower() in SUPPORTED]
        manifest = self._load_manifest()

        total_chunks = 0
        indexed_files = []

        for file in files:
            n = self._index_file(file)
            if n > 0:
                manifest[str(file)] = self._file_state(file)
                total_chunks += n
                indexed_files.append(file.name)

        # Clean up stale manifest entries for files removed from this directory
        for key in list(manifest.keys()):
            path = Path(key)
            if path.is_relative_to(directory) and not path.exists():
                self._collection.delete(where={"source": key})
                del manifest[key]

        self._save_manifest(manifest)
        return total_chunks, indexed_files

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(self, query: str, n_results: int = 5) -> list[dict]:
        """Auto-sync all configured data dirs, then return best-matching chunks."""
        if not query or not query.strip():
            raise ValueError("query must not be empty")

        for directory in get_data_dirs():
            if directory.exists():
                self._sync_directory(directory)

        count = self._collection.count()
        if count == 0:
            return []

        n_results = min(n_results, count)
        query_embedding = self._model.encode([query], show_progress_bar=False).tolist()
        results = self._collection.query(
            query_embeddings=query_embedding,
            n_results=n_results,
            include=["documents", "metadatas", "distances"],
        )

        hits = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            hits.append({
                "source": Path(meta["source"]).name,
                "text": doc,
                "score": round(1 - dist, 3),  # cosine distance → similarity
            })
        return hits


# ---------------------------------------------------------------------------
# Tool: index_documents
# ---------------------------------------------------------------------------

class IndexDocumentsTool(BaseTool):
    name = "index_documents"

    def get_description(self) -> dict:
        allowed = ", ".join(str(d) for d in get_data_dirs())
        return {
            "name": self.name,
            "description": (
                "Index local documents (PDF, TXT, MD) into the RAG vector store. "
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
                return {"result": "No PDF/TXT/MD files found in configured directories."}
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

    def get_description(self) -> dict:
        return {
            "name": self.name,
            "description": (
                "Search indexed local documents using semantic similarity. "
                "Returns the most relevant text chunks from files in the RAG store. "
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
                        "description": "Number of results to return (default 5, max 20).",
                        "default": 5,
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
            n_results = min(int(kwargs.get("n_results", 5)), 20)
            hits = RAGEngine.get().search(query, n_results)
            if not hits:
                return {"result": "No documents indexed yet. Use the index_documents tool first."}
            parts = [f"[{h['source']}] (score: {h['score']})\n{h['text'][:400]}" for h in hits]
            return {"result": "\n\n---\n\n".join(parts)}
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


def search_documents(query: str, n: int = 5) -> list[dict]:
    """Search indexed documents. Returns list of {source, text, score} dicts."""
    return RAGEngine.get().search(query, n_results=n)
