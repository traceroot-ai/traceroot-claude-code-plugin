"""load_state must fall back to a fresh state on any unreadable state file."""

import pytest

from traceroot_observability import state


@pytest.fixture(autouse=True)
def _isolated_state_root(tmp_path, monkeypatch):
    monkeypatch.setattr(state, "STATE_ROOT", tmp_path)


def _write(session_id: str, data: bytes) -> None:
    (state._dir(session_id) / "state.json").write_bytes(data)


def test_missing_file_gives_fresh_state():
    assert state.load_state("missing") == state.SessionState()


@pytest.mark.parametrize("raw", [
    b"\xff\xfe\x00\x80 not utf-8",  # UnicodeDecodeError (a ValueError)
    b"{not json",                    # JSONDecodeError
    b"[1, 2, 3]",                    # valid JSON, not a dict
    b'{"emitted_ask_keys": 1, "snapshots": {}}'[:0],  # empty file
])
def test_bad_file_gives_fresh_state(raw):
    _write("bad", raw)
    assert state.load_state("bad") == state.SessionState()


def test_unknown_keys_are_ignored():
    _write("unk", b'{"git_repo": "o/r", "future_field": 1}')
    st = state.load_state("unk")
    assert st.git_repo == "o/r"
    assert st.emitted_ask_keys == []


def test_save_then_load_roundtrip():
    st = state.SessionState(emitted_ask_keys=["a"], git_repo="o/r", git_ref="main")
    state.save_state("rt", st)
    assert state.load_state("rt") == st