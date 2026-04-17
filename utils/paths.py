"""
Central data directory management. Single source of truth for where persistent data lives.
FOR PACKAGING:  When I package the backend to an .exe with PyInstaller, I need to change where data is stored.
DEV MODE:  ./data/  (relative to backend root)
EXE MODE:  ~/AppData/Local/Crow/  (Windows)
            ~/.local/share/Crow/   (Linux)
            ~/Library/Application Support/Crow/  (macOS)

Every component that needs persistent storage (vector DB, token store, downloaded models, RAG documents)
should call just get_data_dir() rather than hardcoding a path. Later when I package to .exe with PyInstaller, only this
function will need changes, rest of the codebase should be unaffected.

The sys.frozen attribute is set by PyInstaller on the packaged executable.
In dev mode it is absent, so we fall back to a local ./data/ directory.

subdirectory layout:-
  data/
    tokens/       <- encrypted OAuth tokens   (auth/token_store.py)
    chromadb/     <- vector store             (rag/vector_store.py)
    documents/    <- original uploaded files  (rag/ingester.py)
    models/       <- future: local model cache
"""

from __future__ import annotations

import sys
from pathlib import Path
from functools import lru_cache


@lru_cache(maxsize=1)
def get_data_dir() -> Path:
    """
    Return the root data directory, creating it if necessary.

    Cached so filesystem operations only happen once per process lifetime.
    """
    if getattr(sys, "frozen", False):
        # running as a PyInstaller executable
        try:
            from platformdirs import user_data_dir
            p = Path(user_data_dir("Crow", "Crow"))
        except ImportError:
            # platformdirs not available - we fall back to home dir
            p = Path.home() / ".crow"
    else:
        # in development mode - use ./data/ next to the backend root
        p = Path(__file__).parent.parent / "data"

    p.mkdir(parents=True, exist_ok=True)
    return p


def get_subdir(name: str) -> Path:
    """Return a named subdirectory of the data dir, creating it if needed."""
    p = get_data_dir() / name
    p.mkdir(parents=True, exist_ok=True)
    return p


# accessors

def tokens_dir() -> Path:
    return get_subdir("tokens")

def chromadb_dir() -> Path:
    return get_subdir("chromadb")

def documents_dir() -> Path:
    return get_subdir("documents")

def models_dir() -> Path:
    return get_subdir("models")