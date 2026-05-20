"""Diagnostic: print stored document-level summaries from ChromaDB.

Samples up to 5 random summaries from the index using a fixed seed so the
same files are chosen on every run. Run this after indexing to verify that
summaries were generated and stored correctly.

Usage:
    python scripts/check_summaries.py
"""

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.rag_tool import RAGEngine

SEPARATOR = "─" * 70
SAMPLE_SEED = 42
SAMPLE_CAP = 5


def main() -> None:
    engine = RAGEngine.get()

    results = engine._collection.get(
        where={"level": "document"},
        include=["documents", "metadatas"],
    )

    if not results["documents"]:
        print("No document summaries found. Index your documents first.")
        sys.exit(0)

    total = len(results["documents"])
    pairs = list(zip(results["documents"], results["metadatas"]))
    sample = random.Random(SAMPLE_SEED).sample(pairs, min(SAMPLE_CAP, total))

    print(f"Found {total} document summary/ies — showing {len(sample)} (seed={SAMPLE_SEED})\n")

    for doc, meta in sample:
        source = Path(meta["source"])
        print(SEPARATOR)
        print(f"FILE: {source.name}")
        print(f"PATH: {source}")
        print(SEPARATOR)
        print(f"  {doc}")
        print()


if __name__ == "__main__":
    main()
