import json
import os
import fcntl
from contextlib import contextmanager
from dataclasses import dataclass, field, asdict
from pathlib import Path

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


@contextmanager
def session_lock(session_id: str):
    lock = _dir(session_id) / ".lock"
    fh = open(lock, "a+")
    try:
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
        fh.close()


def load_state(session_id: str) -> SessionState:
    f = _dir(session_id) / "state.json"
    if not f.exists():
        return SessionState()
    try:
        data = json.loads(f.read_text())
        # Legacy migration: emitted_turns -> emitted_asks (old name)
        if "emitted_turns" in data and "emitted_asks" not in data:
            data["emitted_asks"] = data.pop("emitted_turns")
        # Migration: emitted_asks (int) -> emitted_ask_keys (list)
        # We can't recover which keys were emitted; risk re-emitting once for
        # mid-flight sessions — acceptable (fail-safe = re-emit, not skip).
        if "emitted_asks" in data and "emitted_ask_keys" not in data:
            data.pop("emitted_asks")
            data["emitted_ask_keys"] = []
        return SessionState(**data)
    except Exception:
        return SessionState()


def save_state(session_id: str, st: SessionState) -> None:
    f = _dir(session_id) / "state.json"
    tmp = f.with_suffix(".tmp")
    tmp.write_text(json.dumps(asdict(st)))
    os.replace(tmp, f)
