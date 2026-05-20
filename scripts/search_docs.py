"""Search indexed documents by semantic similarity.

Usage:
    python scripts/search_docs.py "your query here" [n_results]

Examples:
    python scripts/search_docs.py "meeting notes"
    python scripts/search_docs.py "user segmentation" 3
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.rag_tool import search_documents

if len(sys.argv) < 2:
    print("Usage: python scripts/search_docs.py \"your query\" [n_results]")
    sys.exit(1)

query = sys.argv[1].strip()
if not query:
    print("Error: query must not be empty.")
    sys.exit(1)

n = int(sys.argv[2]) if len(sys.argv) > 2 else 3

print(f'Searching for: "{query}" (top {n} results)\n')
try:
    results = search_documents(query, n=n)
except ValueError as e:
    print(f"Error: {e}")
    sys.exit(1)

if not results:
    print("No results found. Run scripts/index_docs.py first.")
    sys.exit(0)

for i, result in enumerate(results, 1):
    print(f"Result {i} — {result['source']} (score: {result['score']})")
    print(f"{result['text'][:200]}...")
    print()
