"""Diagnostic script: compare pypdf vs pdfplumber extraction quality on all PDFs
in the configured data directories.

Usage:
    python scripts/check_pdf_extraction.py

For each PDF found:
  - Extracts text page by page using pypdf (current backend)
  - Reports total pages, total chars, avg chars/page
  - Flags any page under 100 chars as potentially problematic
  - If any page is flagged, also runs pdfplumber and shows a side-by-side
    comparison of the first flagged page so you can judge which backend is better
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import get_data_dirs

SEPARATOR = "─" * 70
FLAG_THRESHOLD = 100  # chars — pages below this are flagged


def extract_with_pypdf(pdf_path: Path) -> list[str]:
    from pypdf import PdfReader
    reader = PdfReader(str(pdf_path))
    return [page.extract_text() or "" for page in reader.pages]


def extract_with_pdfplumber(pdf_path: Path) -> list[str]:
    import pdfplumber
    pages = []
    with pdfplumber.open(str(pdf_path)) as pdf:
        for page in pdf.pages:
            pages.append(page.extract_text() or "")
    return pages


def extract_with_pymupdf(pdf_path: Path) -> list[str]:
    import fitz
    doc = fitz.open(str(pdf_path))
    return [page.get_text() for page in doc]


def diagnose_pdf(pdf_path: Path) -> None:
    print(f"\n{SEPARATOR}")
    print(f"FILE: {pdf_path.name}")
    print(f"PATH: {pdf_path}")
    print(SEPARATOR)

    try:
        pages = extract_with_pypdf(pdf_path)
    except Exception as e:
        print(f"  [ERROR] pypdf failed: {e}")
        return

    total_pages = len(pages)
    total_chars = sum(len(p) for p in pages)
    avg_chars = total_chars / total_pages if total_pages else 0
    flagged = [i for i, p in enumerate(pages) if len(p) < FLAG_THRESHOLD]

    print(f"  Total pages   : {total_pages}")
    print(f"  Total chars   : {total_chars:,}")
    print(f"  Avg chars/page: {avg_chars:.0f}")
    if flagged:
        print(f"  ⚠️  Flagged pages (< {FLAG_THRESHOLD} chars): {[i + 1 for i in flagged]}")
    else:
        print(f"  ✓  No pages flagged (all ≥ {FLAG_THRESHOLD} chars)")

    print()
    for i, text in enumerate(pages):
        print(f"  --- Page {i + 1} ({len(text)} chars) ---")
        preview = text[:300].replace("\n", " ").strip()
        print(f"  {preview!r}")
        if len(text) > 300:
            print(f"  ... [{len(text) - 300} more chars]")
        print()

    # Check for garbled characters across all pages
    garbled = [i for i, p in enumerate(pages) if "\x7f" in p or "\x00" in p]
    if garbled:
        print(f"  ⚠️  Pages with garbled chars (\\x7f or \\x00): {[i + 1 for i in garbled]}")

    # Always run three-way comparison on every page
    try:
        plumber_pages = extract_with_pdfplumber(pdf_path)
    except Exception as e:
        plumber_pages = []
        print(f"  pdfplumber failed: {e}")

    try:
        mupdf_pages = extract_with_pymupdf(pdf_path)
    except Exception as e:
        mupdf_pages = []
        print(f"  pymupdf failed: {e}")

    for page_idx in range(total_pages):
        print(f"\n{SEPARATOR}")
        print(f"THREE-WAY COMPARISON — Page {page_idx + 1} of {total_pages}")
        print(SEPARATOR)

        pypdf_text   = pages[page_idx]
        plumber_text = plumber_pages[page_idx] if page_idx < len(plumber_pages) else "(unavailable)"
        mupdf_text   = mupdf_pages[page_idx]   if page_idx < len(mupdf_pages)   else "(unavailable)"

        for label, text in [("pypdf", pypdf_text), ("pdfplumber", plumber_text), ("pymupdf", mupdf_text)]:
            garbled_count = text.count("\x7f") + text.count("\x00")
            flag = f" ⚠️  {garbled_count} garbled chars" if garbled_count else " ✓"
            # Show garbled chars visibly in the preview
            clean_preview = text[:400].replace("\x7f", "[\\x7f]").replace("\x00", "[\\x00]")
            print(f"\n  [{label}] {len(text)} chars{flag}")
            print(f"  {clean_preview!r}")

    print(f"\n{SEPARATOR}")
    print("VERDICT")
    print(SEPARATOR)
    # Score each backend: garbled chars (lower is better), then total chars (higher is better)
    for label, page_list in [("pypdf", pages),
                              ("pdfplumber", plumber_pages),
                              ("pymupdf", mupdf_pages)]:
        if not page_list:
            print(f"  {label:12} — unavailable")
            continue
        garbled_total = sum(p.count("\x7f") + p.count("\x00") for p in page_list)
        total = sum(len(p) for p in page_list)
        print(f"  {label:12} {garbled_total} garbled chars, {total:,} total chars")
    print()


def main() -> None:
    data_dirs = get_data_dirs()
    pdfs = []
    for d in data_dirs:
        if d.exists():
            pdfs.extend(d.rglob("*.pdf"))

    if not pdfs:
        print(f"No PDFs found in configured data directories: {', '.join(str(d) for d in data_dirs)}")
        sys.exit(0)

    print(f"Found {len(pdfs)} PDF(s) across {len(data_dirs)} data director(y/ies)")

    for pdf in sorted(pdfs):
        diagnose_pdf(pdf)

    print(f"\n{SEPARATOR}")
    print("Diagnostic complete.")


if __name__ == "__main__":
    main()
