"""
Tests for traceroot_observability.debug — opt-in TRACEROOT_PLUGIN_DEBUG logging.
"""
import os
import stat

import pytest

from traceroot_observability import debug, state


# ---------------------------------------------------------------------------
# debug_enabled() parsing
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("value,expected", [
    ("true", True),
    ("True", True),
    ("TRUE", True),
    ("1", True),
    ("yes", True),
    ("on", True),
    ("false", False),
    ("no", False),
    ("0", False),
    ("", False),
    ("off", False),
])
def test_debug_enabled_parsing(monkeypatch, value, expected):
    monkeypatch.setenv("TRACEROOT_PLUGIN_DEBUG", value)
    assert debug.debug_enabled() == expected


def test_debug_enabled_plugin_option_prefix(monkeypatch):
    monkeypatch.delenv("TRACEROOT_PLUGIN_DEBUG", raising=False)
    monkeypatch.setenv("CLAUDE_PLUGIN_OPTION_TRACEROOT_PLUGIN_DEBUG", "true")
    assert debug.debug_enabled() is True


def test_debug_enabled_false_when_unset(monkeypatch):
    monkeypatch.delenv("TRACEROOT_PLUGIN_DEBUG", raising=False)
    monkeypatch.delenv("CLAUDE_PLUGIN_OPTION_TRACEROOT_PLUGIN_DEBUG", raising=False)
    assert debug.debug_enabled() is False


# ---------------------------------------------------------------------------
# debug_log() — writes when enabled
# ---------------------------------------------------------------------------

def test_debug_log_writes_line(tmp_path, monkeypatch):
    monkeypatch.setenv("TRACEROOT_PLUGIN_DEBUG", "true")
    monkeypatch.setattr(state, "STATE_ROOT", tmp_path)

    debug.debug_log("hello world")

    log = tmp_path / "debug.log"
    assert log.exists(), "debug.log should be created"
    content = log.read_text()
    assert "hello world" in content
    # Should contain a timestamp (ISO-8601 format, starts with year)
    assert content.strip()[0:4].isdigit()


def test_debug_log_appends(tmp_path, monkeypatch):
    monkeypatch.setenv("TRACEROOT_PLUGIN_DEBUG", "true")
    monkeypatch.setattr(state, "STATE_ROOT", tmp_path)

    debug.debug_log("first")
    debug.debug_log("second")

    lines = (tmp_path / "debug.log").read_text().splitlines()
    assert len(lines) == 2
    assert "first" in lines[0]
    assert "second" in lines[1]


# ---------------------------------------------------------------------------
# debug_log() — no-op when disabled
# ---------------------------------------------------------------------------

def test_debug_log_noop_when_disabled(tmp_path, monkeypatch):
    monkeypatch.delenv("TRACEROOT_PLUGIN_DEBUG", raising=False)
    monkeypatch.delenv("CLAUDE_PLUGIN_OPTION_TRACEROOT_PLUGIN_DEBUG", raising=False)
    monkeypatch.setattr(state, "STATE_ROOT", tmp_path)

    debug.debug_log("nope")

    assert not (tmp_path / "debug.log").exists(), "log should not be created when debug is off"


def test_debug_log_noop_explicit_false(tmp_path, monkeypatch):
    monkeypatch.setenv("TRACEROOT_PLUGIN_DEBUG", "false")
    monkeypatch.setattr(state, "STATE_ROOT", tmp_path)

    debug.debug_log("nope")

    assert not (tmp_path / "debug.log").exists()


# ---------------------------------------------------------------------------
# Fail-open: bogus STATE_ROOT must not raise
# ---------------------------------------------------------------------------

def test_debug_log_fail_open_bogus_path(monkeypatch):
    monkeypatch.setenv("TRACEROOT_PLUGIN_DEBUG", "true")
    # Point STATE_ROOT at a path that cannot be written to
    from pathlib import Path
    monkeypatch.setattr(state, "STATE_ROOT", Path("/dev/null/impossible/path"))

    # Must not raise, ever
    debug.debug_log("this should silently fail")


def test_debug_log_fail_open_readonly_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("TRACEROOT_PLUGIN_DEBUG", "true")
    # Make the directory read-only so open(..., "a") fails
    tmp_path.chmod(stat.S_IRUSR | stat.S_IXUSR)
    monkeypatch.setattr(state, "STATE_ROOT", tmp_path)
    try:
        debug.debug_log("should not raise")
    finally:
        # Restore so pytest can clean up tmp_path
        tmp_path.chmod(stat.S_IRWXU)


# ---------------------------------------------------------------------------
# debug_exc() — logs traceback
# ---------------------------------------------------------------------------

def test_debug_exc_logs_traceback(tmp_path, monkeypatch):
    monkeypatch.setenv("TRACEROOT_PLUGIN_DEBUG", "true")
    monkeypatch.setattr(state, "STATE_ROOT", tmp_path)

    try:
        raise ValueError("test error")
    except ValueError:
        debug.debug_exc("myprefix")

    content = (tmp_path / "debug.log").read_text()
    assert "myprefix" in content
    assert "ValueError" in content
    assert "test error" in content


def test_debug_exc_noop_when_disabled(tmp_path, monkeypatch):
    monkeypatch.delenv("TRACEROOT_PLUGIN_DEBUG", raising=False)
    monkeypatch.setattr(state, "STATE_ROOT", tmp_path)

    try:
        raise RuntimeError("x")
    except RuntimeError:
        debug.debug_exc("prefix")

    assert not (tmp_path / "debug.log").exists()
