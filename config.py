import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def get_data_dirs() -> list[Path]:
    """Return the list of directories the RAG system is allowed to index.

    Reads QUICKLOCAL_DATA_DIRS from .env — comma-separated paths, ~ supported.
    Falls back to ~/quicklocal_test_data if the variable is not set.
    """
    raw = os.getenv("QUICKLOCAL_DATA_DIRS", str(Path.home() / "quicklocal_test_data"))
    return [
        Path(p.strip()).expanduser().resolve()
        for p in raw.split(",")
        if p.strip()
    ]
