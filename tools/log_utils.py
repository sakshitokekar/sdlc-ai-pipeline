# Logging — two destinations, two audiences:
#   1. Terminal: short status lines / truncated tails, for a human watching live
#   2. Log file: full untruncated detail with timestamps, for debugging after
#      the fact — this is the persistent record the terminal alone can't give you
import os
from datetime import datetime
from pathlib import Path

# Silence noisy third-party progress bars/warnings (HuggingFace Hub,
# transformers) BEFORE they get imported anywhere downstream. This must be
# set as early as possible in the import chain — pipeline.py imports this
# module first, before any agent (which transitively imports Embedder ->
# sentence_transformers) gets imported.
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

_LOG_PATH = Path("codebase_search_rag/data/pipeline.log")


def _write_to_file(level: str, message: str) -> None:
    """Appends one timestamped, untruncated line to the log file. Never
    raises — a logging failure should never crash the pipeline itself."""
    try:
        _LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().isoformat(timespec="seconds")
        with open(_LOG_PATH, "a") as f:
            f.write(f"[{timestamp}] {level}: {message}\n")
    except OSError:
        pass  # logging is best-effort; a disk issue shouldn't take down the pipeline


def step(message: str) -> None:
    """A step currently in progress, e.g. 'Indexing repository...'"""
    print(f"→ {message}")
    _write_to_file("STEP", message)


def success(message: str) -> None:
    print(f"✓ {message}")
    _write_to_file("SUCCESS", message)


def warn(message: str) -> None:
    print(f"⚠ {message}")
    _write_to_file("WARN", message)


def error(message: str) -> None:
    print(f"✗ {message}")
    _write_to_file("ERROR", message)


def full(message: str) -> None:
    """Writes FULL, untruncated content to the log file only — never
    printed to terminal. Use this for large blobs (pytest output, docker
    build logs, raw LLM responses) that you want preserved for later
    debugging without flooding the live terminal view."""
    _write_to_file("DETAIL", message)


def truncate(text: str, max_chars: int = 500) -> str:
    """Truncates long text for terminal display, keeping the TAIL — the
    most relevant part of a traceback, pytest failure, or build error is
    almost always at the end, not the beginning. The untruncated version
    should separately go to full() so it's not lost."""
    if not text:
        return "(no output)"
    text = text.strip()
    if len(text) <= max_chars:
        return text
    hidden = len(text) - max_chars
    return f"... [{hidden} earlier characters hidden — see codebase_search_rag/data/pipeline.log for full output] ...\n{text[-max_chars:]}"