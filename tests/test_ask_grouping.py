"""
Tests for ask-level grouping and emission (one trace per ask).

Tests:
  1. test_group_turns_into_asks_basic — grouping logic for two real-world patterns
  2. test_is_task_notification_detects_correctly — helper detection
  3. test_emit_ask_groups_query_and_subagent_turns — build_ask_spans puts both turns under one root
  4. test_handle_stop_emits_complete_asks_only — pending sub-agent defers emission
"""
import json
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from traceroot_observability import spans, emit, state
from traceroot_observability.transcript import Turn, group_turns_into_asks, is_task_notification


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _tracer():
    prov = TracerProvider()
    exp = InMemorySpanExporter()
    prov.add_span_processor(SimpleSpanProcessor(exp))
    return prov.get_tracer("traceroot.claude-code"), exp


def _human_turn(user_text: str, ts: str, req_id: str, model: str = "claude-opus-4-8",
                tool_use_name: str | None = None, tool_use_id: str | None = None) -> Turn:
    """Build a minimal human Turn (optionally with a Task tool_use)."""
    content: list = []
    if tool_use_name and tool_use_id:
        content.append({"type": "tool_use", "id": tool_use_id, "name": tool_use_name,
                        "input": {"description": "do something"}})
    else:
        content.append({"type": "text", "text": "ok"})
    return Turn(
        user_msg={"type": "user", "timestamp": ts,
                  "message": {"role": "user", "content": user_text}},
        assistant_msgs=[{"type": "assistant", "requestId": req_id, "timestamp": ts,
                         "message": {"id": req_id, "model": model,
                                     "content": content,
                                     "usage": {"input_tokens": 5, "output_tokens": 3}}}],
        tool_results={tool_use_id: {"content": "launched", "timestamp": ts}} if tool_use_id else {},
    )


def _notif_turn(task_id: str, ts: str, req_id: str, model: str = "claude-opus-4-8") -> Turn:
    """Build a minimal task-notification Turn."""
    notif_text = f"<task-notification>\n<task-id>{task_id}</task-id>\n<result>done</result>\n</task-notification>"
    return Turn(
        user_msg={"type": "user", "timestamp": ts,
                  "message": {"role": "user", "content": notif_text}},
        assistant_msgs=[{"type": "assistant", "requestId": req_id, "timestamp": ts,
                         "message": {"id": req_id, "model": model,
                                     "content": [{"type": "text", "text": "ok notif"}],
                                     "usage": {"input_tokens": 5, "output_tokens": 3}}}],
        tool_results={},
    )


# ---------------------------------------------------------------------------
# 1. test_group_turns_into_asks
# ---------------------------------------------------------------------------

def test_group_turns_into_asks_basic():
    """
    [HUMAN, HUMAN+Tasks, NOTIF, NOTIF] → asks [[0],[1,2,3]]
    [HUMAN, HUMAN+Task, NOTIF, HUMAN, HUMAN] → [[0],[1,2],[3],[4]]
    """
    # Case 1: single human, then human+task with 2 notifs
    T0 = _human_turn("hello", "2026-06-22T00:00:00Z", "r0")
    T1 = _human_turn("run tasks", "2026-06-22T00:01:00Z", "r1",
                     tool_use_name="Task", tool_use_id="tu1")
    T2 = _notif_turn("task1", "2026-06-22T00:02:00Z", "r2")
    T3 = _notif_turn("task2", "2026-06-22T00:03:00Z", "r3")

    asks = group_turns_into_asks([T0, T1, T2, T3])
    assert len(asks) == 2, f"Expected 2 asks, got {len(asks)}"
    assert asks[0] == [T0]
    assert asks[1] == [T1, T2, T3]

    # Case 2: human, human+task, notif, human, human
    A = _human_turn("q1", "2026-06-22T00:10:00Z", "rA")
    B = _human_turn("q2", "2026-06-22T00:11:00Z", "rB",
                    tool_use_name="Task", tool_use_id="tuB")
    C = _notif_turn("taskB", "2026-06-22T00:12:00Z", "rC")
    D = _human_turn("q3", "2026-06-22T00:13:00Z", "rD")
    E = _human_turn("q4", "2026-06-22T00:14:00Z", "rE")

    asks2 = group_turns_into_asks([A, B, C, D, E])
    assert len(asks2) == 4, f"Expected 4 asks, got {len(asks2)}"
    assert asks2[0] == [A]
    assert asks2[1] == [B, C]
    assert asks2[2] == [D]
    assert asks2[3] == [E]


def test_is_task_notification_detects_correctly():
    """is_task_notification returns True only for task-notification turns."""
    notif = _notif_turn("t1", "2026-06-22T00:00:00Z", "r1")
    human = _human_turn("hello", "2026-06-22T00:01:00Z", "r2")
    assert is_task_notification(notif)
    assert not is_task_notification(human)


# ---------------------------------------------------------------------------
# 2. test_emit_ask_groups_query_and_subagent_turns
# ---------------------------------------------------------------------------

def test_emit_ask_groups_query_and_subagent_turns():
    """
    An ask = [human turn that launches a Task, task-notification turn].
    build_ask_spans must produce exactly ONE root AGENT span; both turns' LLM spans are
    direct children of that root; all spans share the same trace_id.
    """
    tracer, exp = _tracer()

    sub_llm_start = 1_700_000_001_000_000_000
    sub_llm_end   = 1_700_000_002_000_000_000
    captured = []

    def fake_emitter(t, ctx, tool_use_id, sid, mc, parent_start_ns=None):
        captured.append(tool_use_id)
        sub_span = t.start_span("claude-haiku-4-5", start_time=sub_llm_start, context=ctx,
                                 attributes={spans.OI["KIND"]: "LLM",
                                             spans.OI["SESSION"]: sid,
                                             spans.OI["MODEL"]: "claude-haiku-4-5",
                                             spans.OI["TOK_PROMPT"]: 5,
                                             spans.OI["TOK_COMPLETION"]: 10,
                                             spans.OI["TOK_TOTAL"]: 15,
                                             spans.OI["CACHE_READ"]: 0,
                                             spans.OI["CACHE_CREATE"]: 0})
        sub_span.end(end_time=sub_llm_end)
        return sub_llm_end

    turn1 = _human_turn("launch a task", "2026-06-22T01:00:00Z", "r-ask1",
                        tool_use_name="Task", tool_use_id="tu-ask1")
    turn2 = _notif_turn("ask1-task", "2026-06-22T01:01:00Z", "r-ask2")

    spans.build_ask_spans(tracer, [turn1, turn2], session_id="sess-ask", git=("o/r", "abc"),
                   tags=["claude-code"], max_chars=20000, subagent_emitter=fake_emitter)

    finished = exp.get_finished_spans()

    # Exactly ONE root AGENT span
    agent_spans = [s for s in finished if s.attributes.get(spans.OI["KIND"]) == "AGENT"]
    assert len(agent_spans) == 1, f"Expected 1 AGENT root span, got {len(agent_spans)}"
    root = agent_spans[0]

    # All spans share the same trace_id
    trace_ids = {s.context.trace_id for s in finished}
    assert len(trace_ids) == 1, f"All spans must share trace_id; got {len(trace_ids)} distinct"

    # Both turns' LLM spans are direct children of the root
    llm_under_root = [s for s in finished if s.attributes.get(spans.OI["KIND"]) == "LLM"
                      and s.parent is not None and s.parent.span_id == root.context.span_id]
    assert len(llm_under_root) >= 2, (
        f"Expected ≥2 main-session LLM spans as direct children of root, got {len(llm_under_root)}"
    )

    # sub-agent emitter was called
    assert "tu-ask1" in captured, "sub-agent emitter was not called for tu-ask1"

    # task-notification turn's LLM spans are in the SAME trace as the root
    notif_llm = [s for s in finished
                 if s.attributes.get(spans.OI["KIND"]) == "LLM"
                 and s.context.trace_id == root.context.trace_id]
    assert notif_llm, "task-notification turn LLM spans must share the ask root's trace_id"


# ---------------------------------------------------------------------------
# 3. test_handle_stop_emits_complete_asks_only
# ---------------------------------------------------------------------------

def test_handle_stop_emits_complete_asks_only(tmp_path, monkeypatch):
    """
    A human turn that launches a sub-agent but the notification hasn't arrived yet
    → handle_stop must NOT emit that ask.
    Once the notification turn is appended → emit exactly once at the next Stop.
    """
    monkeypatch.setattr(state, "STATE_ROOT", tmp_path)
    monkeypatch.setenv("TRACEROOT_API_KEY", "test-key")

    tpath = tmp_path / "sess.jsonl"

    def _j(**kw): return json.dumps(kw)

    lines = [
        # Turn 0: plain human turn (no sub-agent) — completes immediately
        _j(type="user", message={"role": "user", "content": "plain question"},
           timestamp="2026-06-22T00:00:00Z"),
        _j(type="assistant", requestId="r0", timestamp="2026-06-22T00:00:01Z",
           message={"id": "r0", "model": "claude-opus-4-8",
                    "content": [{"type": "text", "text": "answer"}],
                    "usage": {"input_tokens": 5, "output_tokens": 3}}),
        # Turn 1: human turn that launches a Task (pending notification)
        _j(type="user", message={"role": "user", "content": "run a task"},
           timestamp="2026-06-22T00:01:00Z"),
        _j(type="assistant", requestId="r1", timestamp="2026-06-22T00:01:01Z",
           message={"id": "r1", "model": "claude-opus-4-8",
                    "content": [{"type": "tool_use", "id": "tu-pending",
                                 "name": "Task", "input": {"description": "do work"}}],
                    "usage": {"input_tokens": 5, "output_tokens": 3}}),
        _j(type="user", timestamp="2026-06-22T00:01:02Z",
           message={"role": "user",
                    "content": [{"type": "tool_result", "tool_use_id": "tu-pending",
                                 "content": "Task launched"}]}),
        # NO task-notification turn yet
    ]
    tpath.write_text("\n".join(lines) + "\n")

    exp = InMemorySpanExporter()
    def fake_tracer(cfg, git):
        prov = TracerProvider(); prov.add_span_processor(SimpleSpanProcessor(exp))
        return prov.get_tracer("t"), (lambda: None)
    monkeypatch.setattr(emit, "_tracer", fake_tracer)
    monkeypatch.setattr(emit, "_resolve_git", lambda payload, st: ("", ""))

    sid = "sess-pending"

    # First Stop: ask-0 (plain) is complete, ask-1 (pending task) is NOT
    emit.handle_stop({"hook_event_name": "Stop", "session_id": sid,
                      "transcript_path": str(tpath)})

    agent_spans_1 = [s for s in exp.get_finished_spans()
                     if s.attributes.get(spans.OI["KIND"]) == "AGENT"]
    assert len(agent_spans_1) == 1, (
        f"After first Stop (pending sub-agent): expected 1 AGENT span (ask-0 only), "
        f"got {len(agent_spans_1)}"
    )
    st1 = state.load_state(sid)
    assert len(st1.emitted_ask_keys) == 1, f"emitted_ask_keys should have 1 entry, got {st1.emitted_ask_keys}"

    # Append the task-notification turn (sub-agent completed)
    notif_text = "<task-notification>\n<task-id>tu-pending</task-id>\n<result>done</result>\n</task-notification>"
    with open(tpath, "a") as fh:
        fh.write(_j(type="user", timestamp="2026-06-22T00:02:00Z",
                    message={"role": "user", "content": notif_text}) + "\n")
        fh.write(_j(type="assistant", requestId="r2", timestamp="2026-06-22T00:02:01Z",
                    message={"id": "r2", "model": "claude-opus-4-8",
                             "content": [{"type": "text", "text": "sub-agent done"}],
                             "usage": {"input_tokens": 5, "output_tokens": 3}}) + "\n")

    # Second Stop: ask-1 now has notifs >= launched → emit it
    emit.handle_stop({"hook_event_name": "Stop", "session_id": sid,
                      "transcript_path": str(tpath)})

    agent_spans_2 = [s for s in exp.get_finished_spans()
                     if s.attributes.get(spans.OI["KIND"]) == "AGENT"]
    assert len(agent_spans_2) == 2, (
        f"After second Stop (notification arrived): expected 2 AGENT spans total, "
        f"got {len(agent_spans_2)}"
    )
    st2 = state.load_state(sid)
    assert len(st2.emitted_ask_keys) == 2, f"emitted_ask_keys should have 2 entries, got {st2.emitted_ask_keys}"


# ---------------------------------------------------------------------------
# 4. test_interleaved_id_match
# ---------------------------------------------------------------------------

def test_interleaved_id_match(tmp_path):
    """
    With id-matching: NOTIF for T1 attaches to Q1's ask even though Q2 was the last human turn.
    Without subagents_dir (positional): NOTIF attaches to Q2 (current open ask).

    Turns: [Q1 (launches Task T1 with toolUseId 'tu-T1'), Q2 (no subagent), NOTIF(task-id='agent-abc')]
    meta.json: agent-abc.meta.json -> {"toolUseId": "tu-T1"}
    """
    import os
    # Set up subagents dir with meta.json
    subagents_dir = str(tmp_path / "subagents")
    os.makedirs(subagents_dir)
    meta = {"toolUseId": "tu-T1"}
    with open(os.path.join(subagents_dir, "agent-abc.meta.json"), "w") as f:
        import json as _json
        _json.dump(meta, f)

    # Q1 launches a Task with tool_use_id "tu-T1"
    Q1 = _human_turn("do something", "2026-06-22T10:00:00Z", "rQ1",
                     tool_use_name="Task", tool_use_id="tu-T1")
    # Q2: a new human question with no sub-agent
    Q2 = _human_turn("unrelated question", "2026-06-22T10:01:00Z", "rQ2")
    # NOTIF: notification for task-id "abc" (maps to tu-T1 via meta.json)
    notif_text = "<task-notification>\n<task-id>abc</task-id>\n<result>done</result>\n</task-notification>"
    NOTIF = Turn(
        user_msg={"type": "user", "timestamp": "2026-06-22T10:02:00Z",
                  "message": {"role": "user", "content": notif_text}},
        assistant_msgs=[{"type": "assistant", "requestId": "rN", "timestamp": "2026-06-22T10:02:01Z",
                         "message": {"id": "rN", "model": "claude-opus-4-8",
                                     "content": [{"type": "text", "text": "ack"}],
                                     "usage": {"input_tokens": 5, "output_tokens": 3}}}],
        tool_results={},
    )

    # ID-matched: NOTIF should attach to Q1's ask (index 0)
    asks_id = group_turns_into_asks([Q1, Q2, NOTIF], subagents_dir=subagents_dir)
    assert len(asks_id) == 2, f"Expected 2 asks (id-matched), got {len(asks_id)}"
    assert asks_id[0] == [Q1, NOTIF], "NOTIF must attach to Q1's ask (id-matched)"
    assert asks_id[1] == [Q2], "Q2 ask must be [Q2] only"

    # Positional (no subagents_dir): NOTIF attaches to Q2's ask (was the open ask)
    asks_pos = group_turns_into_asks([Q1, Q2, NOTIF])
    assert len(asks_pos) == 2, f"Expected 2 asks (positional), got {len(asks_pos)}"
    assert asks_pos[0] == [Q1], "Q1 ask must be [Q1] only (positional)"
    assert asks_pos[1] == [Q2, NOTIF], "NOTIF must attach to Q2's ask (positional fallback)"


# ---------------------------------------------------------------------------
# 5. test_out_of_order_emission
# ---------------------------------------------------------------------------

def test_out_of_order_emission(tmp_path, monkeypatch):
    """
    Q1 launches a sub-agent (pending) → Q2 has no sub-agent (complete immediately).
    After first Stop: Q2 emits even though Q1 is still pending (out-of-order safe).
    After NOTIF for Q1 arrives: Q1 emits.
    Each ask emitted exactly once (by key).
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

    lines = [
        # Q1: human turn that launches a Task (pending notification)
        _j(type="user", message={"role": "user", "content": "run a task"},
           timestamp="2026-06-22T10:00:00Z"),
        _j(type="assistant", requestId="rQ1", timestamp="2026-06-22T10:00:01Z",
           message={"id": "rQ1", "model": "claude-opus-4-8",
                    "content": [{"type": "tool_use", "id": "tu-ooo",
                                 "name": "Task", "input": {"description": "work"}}],
                    "usage": {"input_tokens": 5, "output_tokens": 3}}),
        _j(type="user", timestamp="2026-06-22T10:00:02Z",
           message={"role": "user",
                    "content": [{"type": "tool_result", "tool_use_id": "tu-ooo",
                                 "content": "Task launched"}]}),
        # Q2: new human question (no sub-agent) — should emit at first Stop
        _j(type="user", message={"role": "user", "content": "another question"},
           timestamp="2026-06-22T10:01:00Z"),
        _j(type="assistant", requestId="rQ2", timestamp="2026-06-22T10:01:01Z",
           message={"id": "rQ2", "model": "claude-opus-4-8",
                    "content": [{"type": "text", "text": "sure"}],
                    "usage": {"input_tokens": 5, "output_tokens": 3}}),
    ]
    tpath.write_text("\n".join(lines) + "\n")

    exp = InMemorySpanExporter()
    def fake_tracer(cfg, git):
        prov = TracerProvider(); prov.add_span_processor(SimpleSpanProcessor(exp))
        return prov.get_tracer("t"), (lambda: None)
    monkeypatch.setattr(emit, "_tracer", fake_tracer)
    monkeypatch.setattr(emit, "_resolve_git", lambda payload, st: ("", ""))

    sid = "sess-ooo"

    # First Stop: Q1 is pending, Q2 is complete — Q2 should emit (not Q1)
    emit.handle_stop({"hook_event_name": "Stop", "session_id": sid,
                      "transcript_path": str(tpath)})

    agent_spans_1 = [s for s in exp.get_finished_spans()
                     if s.attributes.get(spans.OI["KIND"]) == "AGENT"]
    assert len(agent_spans_1) == 1, (
        f"First Stop: expected 1 AGENT span (Q2 only), got {len(agent_spans_1)}"
    )
    # Verify Q1 not yet emitted (Q2 was emitted, Q1 is pending)
    st1 = state.load_state(sid)
    assert len(st1.emitted_ask_keys) == 1, (
        f"Expected 1 emitted key, got {len(st1.emitted_ask_keys)}"
    )

    # Set up meta.json so the notification routes to Q1 via id-matching
    import os
    subagents_dir = os.path.join(os.path.dirname(str(tpath)), sid, "subagents")
    os.makedirs(subagents_dir, exist_ok=True)
    import json as _json
    with open(os.path.join(subagents_dir, "agent-ooo-task.meta.json"), "w") as fh:
        _json.dump({"toolUseId": "tu-ooo"}, fh)

    notif_text = "<task-notification>\n<task-id>ooo-task</task-id>\n<result>done</result>\n</task-notification>"
    with open(tpath, "a") as fh:
        fh.write(_j(type="user", timestamp="2026-06-22T10:02:00Z",
                    message={"role": "user", "content": notif_text}) + "\n")
        fh.write(_j(type="assistant", requestId="rN", timestamp="2026-06-22T10:02:01Z",
                    message={"id": "rN", "model": "claude-opus-4-8",
                             "content": [{"type": "text", "text": "done"}],
                             "usage": {"input_tokens": 5, "output_tokens": 3}}) + "\n")

    # Second Stop: Q1 now has notification → should emit Q1 (Q2 already emitted, not re-emitted)
    emit.handle_stop({"hook_event_name": "Stop", "session_id": sid,
                      "transcript_path": str(tpath)})

    agent_spans_2 = [s for s in exp.get_finished_spans()
                     if s.attributes.get(spans.OI["KIND"]) == "AGENT"]
    assert len(agent_spans_2) == 2, (
        f"Second Stop: expected 2 total AGENT spans (Q1+Q2), got {len(agent_spans_2)}"
    )
    st2 = state.load_state(sid)
    assert len(st2.emitted_ask_keys) == 2, (
        f"Expected 2 emitted keys total, got {len(st2.emitted_ask_keys)}"
    )
    # Verify no duplicate emission
    assert len(set(st2.emitted_ask_keys)) == 2, "Keys must be unique (no duplicates)"
