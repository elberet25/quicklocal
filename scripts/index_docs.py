"""Index local documents into the RAG store.

Indexes all directories configured in QUICKLOCAL_DATA_DIRS (.env).
No path arguments are accepted — add folders to .env to expand the index.

Usage:
    python scripts/index_docs.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import get_data_dirs
from tools.rag_tool import index_documents

if len(sys.argv) > 1:
    print(f"Error: no path arguments accepted. Add folders to QUICKLOCAL_DATA_DIRS in .env instead.")
    sys.exit(1)

dirs = get_data_dirs()
print(f"Indexing documents in: {', '.join(str(d) for d in dirs)}")
num_chunks = index_documents()

if num_chunks == 0:
    print("No documents found or no text could be extracted.")
else:
    print(f"Indexed {num_chunks} document chunks.")
