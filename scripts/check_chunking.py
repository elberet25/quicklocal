"""Diagnostic: print chunks produced from one sampled file per supported type.

Samples 1 file per extension (.md, .pdf, .txt) using a fixed random seed so
the same files are chosen on every run. Shows chunk index, character count,
and full text so you can verify the chunking + merge logic.

Usage:
    python scripts/check_chunking.py
"""

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import get_data_dirs
from tools.rag_tool import RAGEngine, SUPPORTED, MIN_MERGE_CHARS, MIN_CHUNK_CHARS

SEPARATOR = "─" * 70
SAMPLE_SEED = 42


def _sample_files(data_dirs) -> list[Path]:
    """Return one randomly selected file per supported extension."""
    by_ext: dict[str, list[Path]] = {}
    for d in data_dirs:
        if d.exists():
            for f in d.rglob("*"):
                ext = f.suffix.lower()
                if ext in SUPPORTED:
                    by_ext.setdefault(ext, []).append(f)

    rng = random.Random(SAMPLE_SEED)
    return [rng.choice(file_list) for _, file_list in sorted(by_ext.items(), key=lambda x: x[0])]


def main() -> None:
    engine = RAGEngine.get()
    data_dirs = get_data_dirs()
    sample = _sample_files(data_dirs)

    if not sample:
        print("No supported files found in configured data directories.")
        sys.exit(0)

    print(f"Chunking strategy: split on \\n\\n, merge chunks < {MIN_MERGE_CHARS} chars, "
          f"drop chunks < {MIN_CHUNK_CHARS} chars")
    print(f"Sampled {len(sample)} file(s) (1 per type, seed={SAMPLE_SEED})\n")

    for file in sample:
        print(SEPARATOR)
        print(f"FILE: {file.name}")
        print(SEPARATOR)

        text = engine._read_file(file)
        if not text.strip():
            print("  (no text extracted)\n")
            continue

        chunks = engine._chunk(text)
        print(f"  {len(chunks)} chunk(s)\n")

        for i, chunk in enumerate(chunks):
            flag = " ⚠️  short" if len(chunk) < MIN_MERGE_CHARS else ""
            print(f"  Chunk {i + 1} ({len(chunk)} chars){flag}")
            print(f"  {chunk!r}")
            print()


if __name__ == "__main__":
    main()
