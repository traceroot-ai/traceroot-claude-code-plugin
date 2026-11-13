"""
Stop can fire before Claude Code has written the turn's final assistant rows to
the transcript. handle_stop must wait for them (the payload's
last_assistant_message says what the last row should contain) instead of
emitting a truncated trace and marking the ask as done.
"""
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from traceroot_observability import emit, spans, state


def _user(text, ts):
    return {"type": "user", "timestamp": ts, "message": {"role": "user", "content": text}}


def _assistant(rid, ts, *blocks):
    """One assistant row. A str block is text; a dict block is used as is."""
    content = [{"type": "text", "text": b} if isinstance(b, str) else b for b in blocks]
    return {"type": "assistant", "requestId": rid, "timestamp": ts,
            "message": {"id": rid, "model": "claude-opus-4-8", "content": content,
                        "usage": {"input_tokens": 5, "output_tokens": 3}}}


TOOL_USE = {"type": "tool_use", "id": "tu-1", "name": "Read", "input": {}}
TOOL_RESULT = {"type": "user", "timestamp": "2026-06-22T00:00:02Z",
               "message": {"role": "user",
                           "content": [{"type": "tool_result", "tool_use_id": "tu-1", "content": "data"}]}}

RECORDS = [
    _user("read the file", "2026-06-22T00:00:00Z"),
    _assistant("r1", "2026-06-22T00:00:01Z", "Reading it.", TOOL_USE),
    TOOL_RESULT,
    _assistant("r2", "2026-06-22T00:00:03Z", "final answer"),
]


def _setup(tmp_path, monkeypatch, reads):
    """Wire handle_stop to an in-memory exporter; `reads` is what each successive
    transcript read returns (the last entry repeats)."""
    monkeypatch.setattr(state, "STATE_ROOT", tmp_path)
    monkeypatch.setenv("TRACEROOT_API_KEY", "test-key")
    exp = InMemorySpanExporter()

    def fake_tracer(cfg, git):
        prov = TracerProvider()
        prov.add_span_processor(SimpleSpanProcessor(exp))
        return prov.get_tracer("t"), (lambda: None)

    calls = []

    def fake_read(path):
        calls.append(path)
        return reads[min(len(calls), len(reads)) - 1]

    monkeypatch.setattr(emit, "_tracer", fake_tracer)
    monkeypatch.setattr(emit, "_resolve_git", lambda payload, st: ("", ""))
    monkeypatch.setattr(emit._t, "read_all_records", fake_read)
    monkeypatch.setattr(emit.time, "sleep", lambda s: None)
    return exp, calls


def _stop(tmp_path, sid, last_assistant_message=None):
    payload = {"hook_event_name": "Stop", "session_id": sid,
               "transcript_path": str(tmp_path / "t.jsonl")}
    if last_assistant_message is not None:
        payload["last_assistant_message"] = last_assistant_message
    emit.handle_stop(payload)


def _root_outputs(exp):
    return [s.attributes.get(spans.OI["OUTPUT"]) for s in exp.get_finished_spans()
            if s.attributes.get(spans.OI["KIND"]) == "AGENT"]


def test_stop_waits_for_final_assistant_rows(tmp_path, monkeypatch):
    # First two reads lack the final assistant row; the third has it.
    exp, calls = _setup(tmp_path, monkeypatch, [RECORDS[:3], RECORDS[:3], RECORDS])
    _stop(tmp_path, "sess-lag", "final answer")
    assert len(calls) == 3
    assert _root_outputs(exp) == ["final answer"]


def test_stop_reads_once_without_last_assistant_message(tmp_path, monkeypatch):
    # Older Claude Code sends no last_assistant_message: nothing to wait for.
    exp, calls = _setup(tmp_path, monkeypatch, [RECORDS])
    _stop(tmp_path, "sess-nolam")
    assert len(calls) == 1
    assert _root_outputs(exp) == ["final answer"]


def test_stop_ignores_matching_text_from_an_earlier_turn(tmp_path, monkeypatch):
    # The previous turn ended with the same words. That row must not count as
    # this turn's final answer.
    earlier = [_user("first question", "2026-06-21T00:00:00Z"),
               _assistant("r0", "2026-06-21T00:00:01Z", "final answer")]
    this_turn = [_user("read the file", "2026-06-22T00:00:00Z"),
                 _assistant("r1", "2026-06-22T00:00:01Z", TOOL_USE),
                 TOOL_RESULT]
    final = _assistant("r2", "2026-06-22T00:00:03Z", "final answer")
    exp, calls = _setup(tmp_path, monkeypatch,
                        [earlier + this_turn, earlier + this_turn + [final]])
    _stop(tmp_path, "sess-earlier", "final answer")
    assert len(calls) == 2
    assert _root_outputs(exp) == ["final answer", "final answer"]


def test_stop_ignores_earlier_text_that_only_ends_the_same(tmp_path, monkeypatch):
    # "Done." is on disk from the first call; the final answer merely ends with it.
    partial = [_user("fix it", "2026-06-22T00:00:00Z"),
               _assistant("r1", "2026-06-22T00:00:01Z", "Done.", TOOL_USE),
               TOOL_RESULT]
    final = _assistant("r2", "2026-06-22T00:00:03Z", "All good. Done.")
    exp, calls = _setup(tmp_path, monkeypatch, [partial, partial + [final]])
    _stop(tmp_path, "sess-suffix", "All good. Done.")
    assert len(calls) == 2
    assert _root_outputs(exp) == ["All good. Done."]


def test_stop_accepts_final_message_split_across_text_rows(tmp_path, monkeypatch):
    # Claude Code writes one row per content block and joins the blocks with a
    # newline in last_assistant_message, so the last row is only the last block.
    records = [_user("explain", "2026-06-22T00:00:00Z"),
               _assistant("r1", "2026-06-22T00:00:01Z", "Part one"),
               _assistant("r1", "2026-06-22T00:00:02Z", "Part two")]
    exp, calls = _setup(tmp_path, monkeypatch, [records])
    _stop(tmp_path, "sess-blocks", "Part one\nPart two")
    assert len(calls) == 1


def test_stop_gives_up_after_the_deadline_and_logs_it(tmp_path, monkeypatch):
    # The final row never arrives: emit what is on disk and say so in the debug log.
    exp, calls = _setup(tmp_path, monkeypatch, [RECORDS[:3]])
    clock = iter([0.0, 0.0, 3.0])  # deadline set at 0, checked at 0, then past 2s
    monkeypatch.setattr(emit.time, "monotonic", lambda: next(clock))
    logged = []
    monkeypatch.setattr(emit, "debug_log", logged.append)

    _stop(tmp_path, "sess-timeout", "final answer")

    assert len(calls) == 2
    assert _root_outputs(exp) == ["Reading it."]
    assert any("still missing" in line for line in logged)


def test_session_end_cannot_overtake_a_waiting_stop(tmp_path, monkeypatch):
    # While Stop waits for the final rows it must hold the session lock. Otherwise
    # a SessionEnd arriving in that window reads the half-written transcript,
    # emits the ask truncated and marks it emitted, and Stop then skips it.
    import fcntl
    exp, calls = _setup(tmp_path, monkeypatch, [RECORDS[:3], RECORDS])
    probes = []

    def sleep_and_probe(seconds):
        # What a concurrent SessionEnd would hit: the lock must already be taken.
        with open(state._dir("sess-race") / ".lock", "a+") as fh:
            try:
                fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
                probes.append("free")
            except BlockingIOError:
                probes.append("held")

    monkeypatch.setattr(emit.time, "sleep", sleep_and_probe)

    _stop(tmp_path, "sess-race", "final answer")

    assert probes == ["held"]
    assert _root_outputs(exp) == ["final answer"]
