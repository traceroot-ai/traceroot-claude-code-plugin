import json
from traceroot_observability import transcript


def _line(**kw): return json.dumps(kw)


# --- read_all_records tests ---

def test_missing_file_returns_empty():
    """Missing transcript file should return empty list, not raise."""
    recs = transcript.read_all_records("/nonexistent/path/t.jsonl")
    assert recs == []


def test_malformed_json_lines_skipped(tmp_path):
    """Malformed JSON lines are skipped; valid lines are still returned."""
    f = tmp_path / "t.jsonl"
    f.write_text(
        "NOT VALID JSON\n"
        + _line(type="user", message={"role": "user", "content": "valid"}) + "\n"
        + "{broken\n"
    )
    recs = transcript.read_all_records(str(f))
    assert len(recs) == 1
    assert recs[0]["type"] == "user"


# --- assemble_turns tests ---

def test_build_turns_groups_user_assistant_and_tool_results():
    recs = [
        {"type": "user", "message": {"role": "user", "content": "do it"}},
        {"type": "assistant", "message": {"role": "assistant", "id": "m1",
            "content": [{"type": "tool_use", "id": "t1", "name": "Read", "input": {"file_path": "a.py"}}]}},
        {"type": "user", "message": {"role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "ok"}]}},
    ]
    turns = transcript.assemble_turns(recs)
    assert len(turns) == 1
    assert turns[0].tool_results["t1"]["content"] == "ok"
    # Successful result: is_error defaults to False.
    assert turns[0].tool_results["t1"]["is_error"] is False


def test_build_turns_captures_tool_result_is_error_flag():
    """A tool_result with is_error=true must surface as is_error=True on the turn."""
    recs = [
        {"type": "user", "message": {"role": "user", "content": "run bad cmd"}},
        {"type": "assistant", "message": {"role": "assistant", "id": "m1",
            "content": [{"type": "tool_use", "id": "t1", "name": "Bash", "input": {"command": "exit 1"}}]}},
        {"type": "user", "message": {"role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "t1",
                         "content": "exit code 1", "is_error": True}]}},
    ]
    turns = transcript.assemble_turns(recs)
    assert turns[0].tool_results["t1"]["is_error"] is True


def test_isMeta_rows_are_skipped():
    """isMeta phantom rows must NOT create a new turn or split an existing one."""
    recs = [
        {"type": "user", "message": {"role": "user", "content": "hello"}},
        # isMeta injection that would look like a user message without the flag
        {"type": "user", "isMeta": True, "message": {"role": "user", "content": "/skill injected"}},
        {"type": "assistant", "message": {"role": "assistant", "id": "m1", "content": "world"}},
    ]
    turns = transcript.assemble_turns(recs)
    assert len(turns) == 1, "isMeta row must not produce a phantom turn"
    assert turns[0].user_msg["message"]["content"] == "hello"


def test_assistant_frames_all_retained():
    """
    All assistant rows sharing a message.id are retained in order.

    Claude Code writes one row per content block (thinking/text/tool_use), all
    sharing the same message.id and requestId but with different timestamps.
    assemble_turns must NOT deduplicate them — tokens.llm_calls groups by requestId
    and needs the first frame's timestamp for start_ts and the last frame's
    timestamp for end_ts to produce a non-zero LLM span duration.
    """
    recs = [
        {"type": "user", "message": {"role": "user", "content": "q"}},
        {"type": "assistant", "timestamp": "2026-06-22T10:00:00Z",
         "message": {"role": "assistant", "id": "m1", "content": [{"type": "thinking", "thinking": "let me think"}]}},
        {"type": "assistant", "timestamp": "2026-06-22T10:00:01Z",
         "message": {"role": "assistant", "id": "m1", "content": [{"type": "text", "text": "the answer"}]}},
    ]
    turns = transcript.assemble_turns(recs)
    assert len(turns) == 1
    # Both frames must be retained so llm_calls gets start_ts and end_ts from different rows.
    assert len(turns[0].assistant_msgs) == 2
    assert turns[0].assistant_msgs[0]["timestamp"] == "2026-06-22T10:00:00Z"
    assert turns[0].assistant_msgs[1]["timestamp"] == "2026-06-22T10:00:01Z"


def test_tool_result_without_assistant_is_ignored():
    """A tool_result row with no prior user turn should not crash."""
    recs = [
        {"type": "user", "message": {"role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "t99", "content": "orphan"}]}},
    ]
    # No assistant messages means no completed turns
    turns = transcript.assemble_turns(recs)
    assert turns == []


# --- Bug A regression test ---

def test_cross_read_turn_split_regression(tmp_path, monkeypatch):
    """
    Bug A regression: the byte-offset tailer could read a USER message in one Stop
    and the ASSISTANT reply in the next, dropping the turn forever (offset advanced
    past the user row, assemble_turns dropped the incomplete turn).

    The fix: whole-transcript re-parse indexed by completed-turn count.
    Simulate: 2 complete turns + a 3rd that is USER-ONLY (no assistant yet).
    First handle_stop must emit 2 turns, emitted_turns==2.
    Then append the 3rd assistant row; second handle_stop must emit the 3rd turn
    (exactly once), emitted_turns==3, without re-emitting turns 1 and 2.
    """
    import json
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
    from traceroot_observability import emit, state, spans

    monkeypatch.setattr(state, "STATE_ROOT", tmp_path)
    monkeypatch.setenv("TRACEROOT_API_KEY", "test-key")

    tpath = tmp_path / "sess.jsonl"

    def _j(**kw): return json.dumps(kw)

    # Write 2 complete turns + user-only 3rd turn (no assistant yet)
    lines = [
        _j(type="user", message={"role": "user", "content": "turn1 user"}, timestamp="2026-06-23T10:00:00Z"),
        _j(type="assistant", requestId="r1", timestamp="2026-06-23T10:00:01Z",
           message={"id": "r1", "model": "claude-opus-4-8",
                    "content": [{"type": "text", "text": "turn1 reply"}],
                    "usage": {"input_tokens": 5, "output_tokens": 3}}),
        _j(type="user", message={"role": "user", "content": "turn2 user"}, timestamp="2026-06-23T10:00:02Z"),
        _j(type="assistant", requestId="r2", timestamp="2026-06-23T10:00:03Z",
           message={"id": "r2", "model": "claude-opus-4-8",
                    "content": [{"type": "text", "text": "turn2 reply"}],
                    "usage": {"input_tokens": 5, "output_tokens": 3}}),
        _j(type="user", message={"role": "user", "content": "turn3 user"}, timestamp="2026-06-23T10:00:04Z"),
        # NO assistant row yet — 3rd turn is incomplete
    ]
    tpath.write_text("\n".join(lines) + "\n")

    exp = InMemorySpanExporter()
    def fake_tracer(cfg, git):
        prov = TracerProvider(); prov.add_span_processor(SimpleSpanProcessor(exp))
        return prov.get_tracer("t"), (lambda: None)
    monkeypatch.setattr(emit, "_tracer", fake_tracer)
    monkeypatch.setattr(emit, "_resolve_git", lambda payload, st: ("", ""))

    # First Stop: 2 complete turns, 3rd (user-only) dropped by assemble_turns
    emit.handle_stop({"hook_event_name": "Stop", "session_id": "sess-cross",
                      "transcript_path": str(tpath)})

    agent_spans_after_stop1 = [s for s in exp.get_finished_spans()
                                if s.attributes.get(spans.OI["KIND"]) == "AGENT"]
    assert len(agent_spans_after_stop1) == 2, (
        f"Expected 2 AGENT spans after first Stop, got {len(agent_spans_after_stop1)}"
    )

    st = state.load_state("sess-cross")
    assert len(st.emitted_ask_keys) == 2, f"emitted_ask_keys should have 2 entries, got {st.emitted_ask_keys}"

    # Now append the 3rd turn's assistant row (simulating the next Stop reading it)
    with open(tpath, "a") as fh:
        fh.write(_j(type="assistant", requestId="r3", timestamp="2026-06-23T10:00:05Z",
                    message={"id": "r3", "model": "claude-opus-4-8",
                             "content": [{"type": "text", "text": "turn3 reply"}],
                             "usage": {"input_tokens": 5, "output_tokens": 3}}) + "\n")

    # Second Stop: 3rd turn is now complete — must emit exactly once
    emit.handle_stop({"hook_event_name": "Stop", "session_id": "sess-cross",
                      "transcript_path": str(tpath)})

    agent_spans_after_stop2 = [s for s in exp.get_finished_spans()
                                if s.attributes.get(spans.OI["KIND"]) == "AGENT"]
    assert len(agent_spans_after_stop2) == 3, (
        f"Expected 3 AGENT spans total after second Stop, got {len(agent_spans_after_stop2)}; "
        "3rd turn was either not emitted or re-emitted"
    )

    st2 = state.load_state("sess-cross")
    assert len(st2.emitted_ask_keys) == 3, f"emitted_ask_keys should have 3 entries, got {st2.emitted_ask_keys}"
