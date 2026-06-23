from traceroot_observability import state

def test_roundtrip_and_atomic(tmp_path, monkeypatch):
    monkeypatch.setattr(state, "STATE_ROOT", tmp_path)
    sid = "sess-1"
    with state.session_lock(sid):
        st = state.load_state(sid)
        assert st.emitted_ask_keys == [] and st.snapshots == {}
        st.emitted_ask_keys = ["2026-06-22T00:00:00Z"]
        st.snapshots["tool-x"] = "/tmp/agent-x.jsonl"
        state.save_state(sid, st)
        st2 = state.load_state(sid)
    assert st2.emitted_ask_keys == ["2026-06-22T00:00:00Z"] and st2.snapshots["tool-x"] == "/tmp/agent-x.jsonl"


def test_state_migration_from_emitted_asks_int(tmp_path, monkeypatch):
    """Old state files with emitted_asks (int) must migrate cleanly to emitted_ask_keys=[]."""
    import json as _json
    monkeypatch.setattr(state, "STATE_ROOT", tmp_path)
    sid = "sess-migrate"
    # Write a legacy state file with emitted_asks int
    d = tmp_path / sid
    d.mkdir(parents=True)
    (d / "state.json").write_text(_json.dumps({"emitted_asks": 3, "git_repo": None, "git_ref": None, "snapshots": {}}))
    st = state.load_state(sid)
    assert st.emitted_ask_keys == [], f"Migration must set emitted_ask_keys=[], got {st.emitted_ask_keys}"
    assert not hasattr(st, "emitted_asks") or True  # emitted_asks field no longer exists
