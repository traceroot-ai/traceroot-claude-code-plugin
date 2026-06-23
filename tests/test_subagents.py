import json
import os
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from traceroot_observability import subagents, state, spans
from traceroot_observability.transcript import Turn


def _make_tracer():
    prov = TracerProvider()
    exp = InMemorySpanExporter()
    prov.add_span_processor(SimpleSpanProcessor(exp))
    return prov.get_tracer("x"), exp


def _assistant_row(req_id, ts, model, text, usage=None):
    return {
        "type": "assistant",
        "requestId": req_id,
        "timestamp": ts,
        "message": {
            "id": req_id,
            "model": model,
            "content": [{"type": "text", "text": text}],
            "usage": usage or {"input_tokens": 5, "output_tokens": 10},
        },
    }


def test_subagent_spans_nest_under_parent_with_parent_session(tmp_path, monkeypatch):
    monkeypatch.setattr(state, "STATE_ROOT", tmp_path)
    agent = tmp_path / "agent-x.jsonl"
    agent.write_text(
        json.dumps({"type": "user", "timestamp": "2026-06-22T00:00:00Z",
                    "message": {"role": "user", "content": "sub-agent task"}}) + "\n" +
        json.dumps({"type": "assistant", "requestId": "a1", "timestamp": "2026-06-22T00:00:01Z",
            "message": {"id": "a1", "model": "claude-haiku-4-5", "content": [{"type": "text", "text": "sub"}],
                        "usage": {"input_tokens": 3, "output_tokens": 7}}}) + "\n"
    )
    sid = "sess"
    with state.session_lock(sid):
        subagents.handle_subagent_stop({"session_id": sid, "agent_transcript_path": str(agent),
                                        "tool_use_id": "t-task"})
    prov = TracerProvider(); exp = InMemorySpanExporter()
    prov.add_span_processor(SimpleSpanProcessor(exp)); tracer = prov.get_tracer("x")
    parent = tracer.start_span("Task")
    emit = subagents.make_emitter(sid, str(agent))
    emit(tracer, trace.set_span_in_context(parent), "t-task", sid, 20000)
    parent.end()
    llm = [s for s in exp.get_finished_spans() if s.attributes.get(spans.OI["KIND"]) == "LLM"]
    assert llm and llm[0].attributes[spans.OI["MODEL"]] == "claude-haiku-4-5"
    assert llm[0].attributes[spans.OI["SESSION"]] == "sess"   # parent session, not the sub-agent's


# ---------------------------------------------------------------------------
# Bug A — Disk-based resolution: state.snapshots is EMPTY but meta.json exists
# ---------------------------------------------------------------------------

def test_disk_resolution_when_snapshots_empty(tmp_path, monkeypatch):
    """
    Prod-bug regression: snapshots={} but meta.json + .jsonl on disk.
    make_emitter should find the sub-agent transcript via the .meta.json sidecar
    and emit the nested LLM span under the Agent TOOL span.

    Directory layout mirrors real Claude Code:
      <projects_dir>/<session_id>.jsonl          ← main transcript
      <projects_dir>/<session_id>/subagents/agent-X.jsonl
      <projects_dir>/<session_id>/subagents/agent-X.meta.json
    """
    monkeypatch.setattr(state, "STATE_ROOT", tmp_path)

    session_id = "sess-disk"
    projects_dir = tmp_path / "projects"
    projects_dir.mkdir()

    # Main transcript at <projects_dir>/<session_id>.jsonl
    main_transcript = projects_dir / f"{session_id}.jsonl"
    main_transcript.write_text("")  # content irrelevant; used for path derivation

    # Subagents directory at <projects_dir>/<session_id>/subagents/
    subagents_dir = projects_dir / session_id / "subagents"
    subagents_dir.mkdir(parents=True)

    agent_jsonl = subagents_dir / "agent-abc.jsonl"
    agent_jsonl.write_text(
        json.dumps({"type": "user", "timestamp": "2026-06-22T01:00:00Z",
                    "message": {"role": "user", "content": "disk task"}}) + "\n" +
        json.dumps(_assistant_row("r1", "2026-06-22T01:00:01Z", "claude-haiku-4-5", "disk result",
                                  usage={"input_tokens": 5, "output_tokens": 10})) + "\n" +
        json.dumps(_assistant_row("r2", "2026-06-22T01:00:03Z", "claude-haiku-4-5", "disk result end",
                                  usage={"input_tokens": 5, "output_tokens": 10})) + "\n"
    )

    meta_path = subagents_dir / "agent-abc.meta.json"
    meta_path.write_text(json.dumps({
        "agentType": "subagent",
        "description": "Explore smolvm repo",
        "toolUseId": "toolu_DISK",
    }))

    # DO NOT seed snapshots — this is the prod-bug scenario
    # state.snapshots = {} (default)

    tracer, exp = _make_tracer()
    parent = tracer.start_span("Agent")

    emit = subagents.make_emitter(session_id, str(main_transcript))
    emit(tracer, trace.set_span_in_context(parent), "toolu_DISK", session_id, 20000)
    parent.end()

    finished = exp.get_finished_spans()
    llm = [s for s in finished if s.attributes.get(spans.OI["KIND"]) == "LLM"]
    assert llm, "Expected nested LLM span from disk-resolved sub-agent transcript"
    assert llm[0].attributes[spans.OI["MODEL"]] == "claude-haiku-4-5"
    assert llm[0].attributes[spans.OI["SESSION"]] == session_id


# ---------------------------------------------------------------------------
# Bug B — Containment: Agent TOOL span must contain its sub-agent children
# ---------------------------------------------------------------------------

def test_agent_tool_span_contains_subagent_children(tmp_path, monkeypatch):
    """
    The Agent TOOL span's end_time must be >= max child end_time.
    Bug B: previously the TOOL span ended at launch-ack (0.2s), while
    sub-agent LLM spans ran 45s later — violating OTEL parent containment.
    """
    monkeypatch.setattr(state, "STATE_ROOT", tmp_path)

    session_id = "sess-contain"
    projects_dir = tmp_path / "projects"
    projects_dir.mkdir()

    # Main transcript layout mirrors real Claude Code
    main_transcript = projects_dir / f"{session_id}.jsonl"
    main_transcript.write_text("")

    subagents_dir = projects_dir / session_id / "subagents"
    subagents_dir.mkdir(parents=True)

    # Two sub-agent LLM calls with timestamps well AFTER the tool launch-ack
    agent_jsonl = subagents_dir / "agent-contain.jsonl"
    agent_jsonl.write_text(
        json.dumps({"type": "user", "timestamp": "2026-06-22T01:00:05Z",
                    "message": {"role": "user", "content": "containment task"}}) + "\n" +
        json.dumps(_assistant_row("r1", "2026-06-22T01:00:10Z", "claude-haiku-4-5", "first")) + "\n" +
        json.dumps(_assistant_row("r1", "2026-06-22T01:00:20Z", "claude-haiku-4-5", "first end")) + "\n" +
        json.dumps(_assistant_row("r2", "2026-06-22T01:00:25Z", "claude-haiku-4-5", "second")) + "\n" +
        json.dumps(_assistant_row("r2", "2026-06-22T01:00:35Z", "claude-haiku-4-5", "second end")) + "\n"
    )

    meta_path = subagents_dir / "agent-contain.meta.json"
    meta_path.write_text(json.dumps({
        "agentType": "subagent",
        "description": "Containment test agent",
        "toolUseId": "toolu_CONTAIN",
    }))

    # Build a Turn where the Agent tool's ack timestamp is BEFORE the sub-agent ends
    tool_use_id = "toolu_CONTAIN"
    user_row = {
        "type": "user",
        "timestamp": "2026-06-22T01:00:00Z",
        "message": {"role": "user", "content": "do something async"},
    }
    assistant_row = {
        "type": "assistant",
        "requestId": "main-r1",
        "timestamp": "2026-06-22T01:00:05Z",
        "message": {
            "id": "main-r1",
            "model": "claude-opus-4-8",
            "content": [{"type": "tool_use", "id": tool_use_id, "name": "Agent",
                          "input": {"description": "Containment test agent"}}],
            "usage": {"input_tokens": 100, "output_tokens": 5},
        },
    }
    tool_result_row = {
        "type": "user",
        "timestamp": "2026-06-22T01:00:06Z",  # launch-ack at T+6s; sub-agent finishes at T+35s
        "message": {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": tool_use_id,
                         "content": "Async agent launched"}],
        },
    }

    from traceroot_observability.transcript import assemble_turns
    turn = assemble_turns([user_row, assistant_row, tool_result_row])[0]

    tracer, exp = _make_tracer()

    emit = subagents.make_emitter(session_id, str(main_transcript))

    from traceroot_observability import spans as _spans
    _spans.build_turn_spans(
        tracer, turn, session_id, (None, None), [], 20000,
        subagent_emitter=emit,
    )

    finished = exp.get_finished_spans()
    tool_span = next((s for s in finished if s.attributes.get(_spans.OI["KIND"]) == "TOOL"), None)
    assert tool_span is not None, "Expected TOOL span for Agent"

    llm_spans = [s for s in finished if s.attributes.get(_spans.OI["KIND"]) == "LLM"
                 and s.attributes.get(_spans.OI["SESSION"]) == session_id
                 and s.parent is not None
                 and s.parent.span_id == tool_span.context.span_id]
    assert llm_spans, "Expected nested sub-agent LLM spans under TOOL span"

    max_child_end = max(s.end_time for s in llm_spans)
    assert tool_span.end_time >= max_child_end, (
        f"TOOL span end_time {tool_span.end_time} < max child end_time {max_child_end}"
    )


# ---------------------------------------------------------------------------
# Bug A (hardening) — _parent_tool_use_id reads meta sidecar when
# tool_use_id absent from payload but agent_transcript_path is present
# ---------------------------------------------------------------------------

def test_parent_tool_use_id_via_meta_sidecar(tmp_path, monkeypatch):
    """
    When SubagentStop payload has agent_transcript_path but NO tool_use_id,
    _parent_tool_use_id should read the sibling .meta.json and return its toolUseId.
    """
    monkeypatch.setattr(state, "STATE_ROOT", tmp_path)

    # Write transcript + sidecar meta
    transcript = tmp_path / "agent-sidecar.jsonl"
    transcript.write_text("")
    meta = tmp_path / "agent-sidecar.meta.json"
    meta.write_text(json.dumps({
        "agentType": "subagent",
        "description": "sidecar test",
        "toolUseId": "toolu_SIDECAR",
    }))

    payload = {
        "agent_transcript_path": str(transcript),
        # deliberately no "tool_use_id" key
    }

    result = subagents._parent_tool_use_id(payload)
    assert result == "toolu_SIDECAR", f"Expected toolu_SIDECAR, got {result!r}"


# ---------------------------------------------------------------------------
# Change (c): full parity — sub-agent TOOL spans + LLM input + containment
# ---------------------------------------------------------------------------

def test_subagent_full_parity_tool_spans(tmp_path, monkeypatch):
    """
    After change (c), make_emitter routes the sub-agent transcript through
    _build_turn_spans_into, so sub-agent TOOL spans (Bash/Read) appear alongside LLM
    spans. Asserts: (i) ≥2 sub-agent LLM spans, (ii) Bash TOOL span with correct
    input/output, (iii) first sub-agent LLM has non-empty input.value (task prompt),
    (iv) no negative-duration spans, (v) Agent TOOL span contains all descendants.
    """
    monkeypatch.setattr(state, "STATE_ROOT", tmp_path)

    session_id = "sess-parity"
    T0 = "2026-06-22T09:00:00Z"
    T1 = "2026-06-22T09:00:01Z"
    T2 = "2026-06-22T09:00:02Z"
    T3 = "2026-06-22T09:00:03Z"

    # Build sub-agent JSONL:
    #  row 0: user task prompt
    #  row 1: assistant call 1 — Bash tool_use (requestId sa-r1)
    #  row 2: user tool_result for Bash
    #  row 3: assistant call 2 — text only (requestId sa-r2)
    sa_records = [
        # Task prompt
        {"type": "user", "timestamp": T0,
         "message": {"role": "user", "content": "List files in /tmp"}},
        # LLM call 1: issues Bash
        {"type": "assistant", "requestId": "sa-r1", "timestamp": T1,
         "message": {
             "id": "sa-r1", "model": "claude-haiku-4-5",
             "content": [
                 {"type": "tool_use", "id": "sa-bash-1", "name": "Bash",
                  "input": {"command": "ls /tmp"}}
             ],
             "usage": {"input_tokens": 20, "output_tokens": 8},
         }},
        # Tool result for Bash
        {"type": "user", "timestamp": T2,
         "message": {"role": "user", "content": [
             {"type": "tool_result", "tool_use_id": "sa-bash-1",
              "content": "file1.txt\nfile2.txt"}
         ]}},
        # LLM call 2: final text response
        {"type": "assistant", "requestId": "sa-r2", "timestamp": T3,
         "message": {
             "id": "sa-r2", "model": "claude-haiku-4-5",
             "content": [{"type": "text", "text": "Done: found 2 files"}],
             "usage": {"input_tokens": 30, "output_tokens": 12},
         }},
    ]

    # Write sub-agent transcript and register it via handle_subagent_stop
    sa_jsonl = tmp_path / "agent-parity.jsonl"
    sa_jsonl.write_text("\n".join(json.dumps(r) for r in sa_records) + "\n")

    agent_tid = "toolu_PARITY"
    with state.session_lock(session_id):
        subagents.handle_subagent_stop({
            "session_id": session_id,
            "agent_transcript_path": str(sa_jsonl),
            "tool_use_id": agent_tid,
        })

    # Build a parent turn that spawns the sub-agent via an Agent tool
    parent_user_row = {
        "type": "user", "timestamp": "2026-06-22T08:59:55Z",
        "message": {"role": "user", "content": "explore /tmp for me"},
    }
    parent_asst_row = {
        "type": "assistant", "requestId": "main-r1", "timestamp": "2026-06-22T08:59:58Z",
        "message": {
            "id": "main-r1", "model": "claude-opus-4-8",
            "content": [
                {"type": "tool_use", "id": agent_tid, "name": "Agent",
                 "input": {"description": "List files in /tmp"}}
            ],
            "usage": {"input_tokens": 50, "output_tokens": 5},
        },
    }
    parent_result_row = {
        "type": "user", "timestamp": "2026-06-22T08:59:59Z",
        "message": {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": agent_tid,
             "content": "Agent launched"}
        ]},
    }

    from traceroot_observability.transcript import assemble_turns
    parent_turn = assemble_turns([parent_user_row, parent_asst_row, parent_result_row])[0]

    tracer, exp = _make_tracer()
    emit_fn = subagents.make_emitter(session_id, str(sa_jsonl))
    spans.build_turn_spans(
        tracer, parent_turn, session_id, (None, None), [], 20000,
        subagent_emitter=emit_fn,
    )

    finished = exp.get_finished_spans()

    # Find the Agent TOOL span
    agent_tool_span = next(
        (s for s in finished
         if s.attributes.get(spans.OI["KIND"]) == "TOOL"
         and s.attributes.get(spans.OI["TOOL_NAME"]) == "Agent"),
        None,
    )
    assert agent_tool_span is not None, "Expected Agent TOOL span in finished spans"

    # (i) Sub-agent LLM spans — direct children of the Agent TOOL span
    sub_llm_spans = [
        s for s in finished
        if s.attributes.get(spans.OI["KIND"]) == "LLM"
        and s.parent is not None
        and s.parent.span_id == agent_tool_span.context.span_id
    ]
    assert len(sub_llm_spans) >= 2, (
        f"Expected ≥2 sub-agent LLM spans under Agent TOOL span, got {len(sub_llm_spans)}"
    )

    # (ii) Sub-agent Bash TOOL span — child of a sub-agent LLM span
    sub_llm_ids = {s.context.span_id for s in sub_llm_spans}
    bash_tool_spans = [
        s for s in finished
        if s.attributes.get(spans.OI["KIND"]) == "TOOL"
        and s.attributes.get(spans.OI["TOOL_NAME"]) == "Bash"
        and s.parent is not None
        and s.parent.span_id in sub_llm_ids
    ]
    assert len(bash_tool_spans) >= 1, (
        "Expected ≥1 Bash TOOL span nested under a sub-agent LLM span"
    )
    bash = bash_tool_spans[0]
    bash_inp = bash.attributes.get(spans.OI["INPUT"]) or ""
    bash_out = bash.attributes.get(spans.OI["OUTPUT"]) or ""
    assert "ls /tmp" in bash_inp, f"Bash TOOL input must contain 'ls /tmp', got: {bash_inp!r}"
    assert "file1.txt" in bash_out, f"Bash TOOL output must contain 'file1.txt', got: {bash_out!r}"

    # (iii) First sub-agent LLM span must have non-empty input.value (task prompt)
    first_sub_llm = min(sub_llm_spans, key=lambda s: s.start_time)
    first_inp = first_sub_llm.attributes.get(spans.OI["INPUT"]) or ""
    assert first_inp, (
        "First sub-agent LLM span must have non-empty input.value (task prompt); "
        "delta-input fix now applies to sub-agents too"
    )
    assert "List files in /tmp" in first_inp, (
        f"First sub-agent LLM input must contain the task prompt, got: {first_inp!r}"
    )

    # (iv) No negative-duration spans among sub-agent spans
    all_sub_spans = sub_llm_spans + bash_tool_spans
    neg = [s for s in all_sub_spans if s.end_time < s.start_time]
    assert not neg, (
        f"Sub-agent spans with negative duration: "
        + str([(s.name, s.start_time, s.end_time) for s in neg])
    )

    # (v) Containment: Agent TOOL span end >= max end of all sub-agent descendant spans
    max_desc_end = max(s.end_time for s in all_sub_spans)
    assert agent_tool_span.end_time >= max_desc_end, (
        f"Agent TOOL span end_time {agent_tool_span.end_time} < "
        f"max descendant end_time {max_desc_end}; containment violated"
    )
