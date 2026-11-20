import json
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass, field, asdict
from pathlib import Path

try:
    import fcntl  # POSIX
    _HAS_FCNTL = True
except ImportError:  # Windows
    import msvcrt
    _HAS_FCNTL = False

STATE_ROOT = Path.home() / ".claude" / "state" / "traceroot"


@dataclass
class SessionState:
    emitted_ask_keys: list = field(default_factory=list)
    git_repo: str | None = None
    git_ref: str | None = None
    snapshots: dict = field(default_factory=dict)  # parent tool_use_id -> snapshot path


def _dir(session_id: str) -> Path:
    d = STATE_ROOT / session_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def _lock(fh) -> None:
    if _HAS_FCNTL:
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
    else:
        # Lock 1 byte at offset 0; retry until acquired (LK_LOCK alone times out).
        while True:
            fh.seek(0)
            try:
                msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)
                return
            except OSError:
                time.sleep(0.05)


def _unlock(fh) -> None:
    if _HAS_FCNTL:
        fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
    else:
        fh.seek(0)
        msvcrt.locking(fh.fileno(), msvcrt.LK_UNLCK, 1)


@contextmanager
def session_lock(session_id: str):
    lock = _dir(session_id) / ".lock"
    fh = open(lock, "a+")
    try:
        _lock(fh)
        try:
            yield
        finally:
            _unlock(fh)
    finally:
        fh.close()


def _state_path(session_id: str) -> Path:
    return _dir(session_id) / "state.json"


def load_state(session_id: str) -> SessionState:
    """Load persisted state, or a fresh SessionState if none/invalid.

    OSError covers a missing/unreadable file; ValueError covers bad JSON
    (JSONDecodeError) and non-UTF-8 bytes (UnicodeDecodeError).
    """
    path = _state_path(session_id)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return SessionState()
    if not isinstance(data, dict):
        return SessionState()
    # Ignore unknown keys so older/newer state files don't crash loading.
    known = {k: v for k, v in data.items() if k in SessionState.__dataclass_fields__}
    try:
        return SessionState(**known)
    except TypeError:
        return SessionState()


def save_state(session_id: str, st: SessionState) -> None:
    """Atomically persist state (write temp file, then replace)."""
    path = _state_path(session_id)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(asdict(st)), encoding="utf-8")
    os.replace(tmp, path)