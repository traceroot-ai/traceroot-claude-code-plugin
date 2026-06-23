"""
Optional debug logging for the TraceRoot Claude Code plugin.

When TRACEROOT_PLUGIN_DEBUG (or CLAUDE_PLUGIN_OPTION_TRACEROOT_PLUGIN_DEBUG) is truthy,
timestamped diagnostic lines are appended to <STATE_ROOT>/debug.log.

All public functions are FAIL-OPEN: they never raise, never propagate exceptions.
When the flag is off, they return immediately without I/O.
"""

import os
import traceback
from datetime import datetime, timezone


def _env_opt(name: str) -> str:
    return os.environ.get(f"CLAUDE_PLUGIN_OPTION_{name}") or os.environ.get(name) or ""


_TRUTHY = {"true", "1", "yes", "on"}


def debug_enabled() -> bool:
    """Return True if TRACEROOT_PLUGIN_DEBUG is set to a truthy value."""
    return _env_opt("TRACEROOT_PLUGIN_DEBUG").strip().lower() in _TRUTHY


def debug_log(msg: str) -> None:
    """Append a timestamped line to <STATE_ROOT>/debug.log if debug is enabled.

    Derives STATE_ROOT at call-time so tests that monkeypatch state.STATE_ROOT
    redirect the log automatically.  Swallows all exceptions — never propagates.
    """
    if not debug_enabled():
        return
    try:
        from . import state
        log_path = state.STATE_ROOT / "debug.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).isoformat()
        with open(log_path, "a") as fh:
            fh.write(f"{ts} {msg}\n")
    except Exception:
        pass


def debug_exc(prefix: str) -> None:
    """Log prefix + current exception traceback if debug is enabled.

    Should be called from inside an except block.  Swallows all exceptions.
    """
    if not debug_enabled():
        return
    try:
        debug_log(prefix + ": " + traceback.format_exc())
    except Exception:
        pass
