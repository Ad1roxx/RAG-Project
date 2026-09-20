"""Canonical on-disk locations for the pipeline.

Everything under data/ is *derived* — it is rebuilt by running the ingestion
scripts, and is gitignored. The repo stays small and the corpus stays
reproducible, which is why the plan says "a fetch script, not checked-in PDFs".
"""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"

# Stage 1 output: exactly what arXiv gave us, unmodified.
RAW_DIR = DATA_DIR / "raw"
PDF_DIR = RAW_DIR / "pdfs"
METADATA_DIR = RAW_DIR / "metadata"

# Stage 2 output: cleaned text + structure, one JSON per paper.
PROCESSED_DIR = DATA_DIR / "processed"


def ensure_dirs() -> None:
    """Create every data directory we write to. Safe to call repeatedly."""
    for directory in (PDF_DIR, METADATA_DIR, PROCESSED_DIR):
        directory.mkdir(parents=True, exist_ok=True)
