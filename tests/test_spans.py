"""
Tests for spans.py: backdated OTEL span tree builder.

Uses the OpenTelemetry in-memory exporter to assert the span tree structure,
attribute values, and correct token accounting.
"""

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import StatusCode

from traceroot_observability import spans
from traceroot_observability.transcript import Turn, assemble_turns


def _tracer():
    prov = TracerProvider()
    exp = InMemorySpanExporter()
    prov.add_span_processor(SimpleSpanProcessor(exp))
    return prov.get_tracer("traceroot.claude-code"), exp


def _parse_ns_spans(ts: str) -> int:
    from datetime import datetime
    return int(datetime.fromisoformat(ts.replace("Z", "+00:00")).timestamp() * 1e9)


# ---------------------------------------------------------------------------
# Brief's required tests (verbatim from brief Step 1)
# ---------------------------------------------------------------------------

def test_emit_turn_builds_llm_and_tool_spans_with_session_and_tokens():
    tracer, exp = _tracer()
    turn = Turn(
        user_msg={"message": {"role": "user", "content": "read a.py"}, "timestamp": "2026-06-22T00:00:00Z"},
        assistant_msgs=[{"type": "assistant", "requestId": "r1", "timestamp": "2026-06-22T00:00:01Z",
            "message": {"id": "r1", "model": "claude-opus-4-8",
                "content": [{"type": "tool_use", "id": "t1", "name": "Read", "input": {"file_path": "a.py"}}],
                "usage": {"input_tokens": 2, "output_tokens": 10, "cache_read_input_tokens": 50}}}],
        tool_results={"t1": {"content": "file body", "timestamp": "2026-06-22T00:00:02Z"}})
    spans.build_turn_spans(tracer, turn, session_id="sess-1", git=("o/r", "abc"), tags=["claude-code"], max_chars=20000)
    got = {s.attributes.get(spans.OI["KIND"]): s for s in exp.get_finished_spans()}
    assert "AGENT" in got and "LLM" in got and "TOOL" in got
    llm = got["LLM"]
    assert llm.attributes[spans.OI["MODEL"]] == "claude-opus-4-8"
    assert llm.attributes[spans.OI["TOK_PROMPT"]] == 52      # 2 + 50 cache
    assert got["AGENT"].attributes[spans.OI["SESSION"]] == "sess-1"
    assert got["LLM"].attributes[spans.OI["SESSION"]] == "sess-1"
    assert got["TOOL"].attributes[spans.OI["SESSION"]] == "sess-1"


def test_middle_truncate_keeps_head_and_tail():
    out = spans.middle_truncate("A" * 100 + "B" * 100, 40)
    assert out.startswith("A") and out.endswith("B") and len(out) < 210


# ---------------------------------------------------------------------------
# Extra tests (beyond the brief)
# ---------------------------------------------------------------------------

def test_middle_truncate_records_original_length_and_elides_middle():
    """middle_truncate must embed the original length and an elision marker."""
    original = "A" * 100 + "M" * 200 + "B" * 100
    out = spans.middle_truncate(original, 60)
    # Original length reported
    assert "400" in out, "original length must appear in truncated output"
    # Middle section elided
    assert "M" * 200 not in out, "middle content must be elided"
    # Head and tail preserved
    assert out.startswith("A")
    assert out.endswith("B")
    # Total output is shorter than the original
    assert len(out) < len(original)


def test_tool_span_captures_input_args_and_result():
    """TOOL span must carry the tool call input as INPUT and tool result as OUTPUT."""
    tracer, exp = _tracer()
    turn = Turn(
        user_msg={"message": {"role": "user", "content": "write file"}, "timestamp": "2026-06-22T00:01:00Z"},
        assistant_msgs=[{"type": "assistant", "requestId": "r2", "timestamp": "2026-06-22T00:01:01Z",
            "message": {"id": "r2", "model": "claude-haiku-4",
                "content": [
                    {"type": "tool_use", "id": "tw1", "name": "Write",
                     "input": {"file_path": "out.txt", "content": "hello"}}
                ],
                "usage": {"input_tokens": 5, "output_tokens": 3}}}],
        tool_results={"tw1": {"content": "Written successfully", "timestamp": "2026-06-22T00:01:02Z"}})
    spans.build_turn_spans(tracer, turn, session_id="sess-2", git=("", ""), tags=[], max_chars=20000)
    finished = exp.get_finished_spans()
    tool_spans = [s for s in finished if s.attributes.get(spans.OI["KIND"]) == "TOOL"]
    assert len(tool_spans) == 1
    ts = tool_spans[0]
    # Tool name stored in both keys
    assert ts.attributes[spans.OI["TOOL_NAME"]] == "Write"
    assert ts.attributes[spans.OI["GEN_TOOL"]] == "Write"
    # Input contains the args JSON
    inp = ts.attributes[spans.OI["INPUT"]]
    assert "out.txt" in inp
    # Output contains the result
    assert ts.attributes[spans.OI["OUTPUT"]] == "Written successfully"


def test_git_and_session_attrs_on_agent_span():
    """AGENT span must carry git repo/ref and session.id."""
    tracer, exp = _tracer()
    turn = Turn(
        user_msg={"message": {"role": "user", "content": "hello"}, "timestamp": "2026-06-22T00:02:00Z"},
        assistant_msgs=[{"type": "assistant", "requestId": "r3", "timestamp": "2026-06-22T00:02:01Z",
            "message": {"id": "r3", "model": "claude-sonnet-4",
                "content": [{"type": "text", "text": "hi there"}],
                "usage": {"input_tokens": 1, "output_tokens": 2}}}],
        tool_results={})
    spans.build_turn_spans(tracer, turn, session_id="my-session", git=("owner/repo", "sha123"), tags=["t1"], max_chars=20000)
    finished = exp.get_finished_spans()
    agent_spans = [s for s in finished if s.attributes.get(spans.OI["KIND"]) == "AGENT"]
    assert len(agent_spans) == 1
    a = agent_spans[0]
    assert a.attributes[spans.OI["SESSION"]] == "my-session"
    assert a.attributes[spans.OI["GIT_REPO"]] == "owner/repo"
    assert a.attributes[spans.OI["GIT_REF"]] == "sha123"


def test_llm_span_backdates_to_turn_start_when_no_assistant_timestamps():
    """
    Regression: when an LLM call has no start_ts / end_ts the LLM span must
    backdate to the turn start (user_msg timestamp), not fall through to the
    OTEL SDK's live wall-clock.
    """
    tracer, exp = _tracer()
    user_ts = "2026-06-22T00:00:00Z"
    turn = Turn(
        user_msg={"message": {"role": "user", "content": "hello"}, "timestamp": user_ts},
        # assistant message deliberately has NO "timestamp" field
        assistant_msgs=[{"type": "assistant", "requestId": "r-no-ts",
            "message": {"id": "r-no-ts", "model": "claude-haiku-4",
                "content": [{"type": "text", "text": "hi"}],
                "usage": {"input_tokens": 1, "output_tokens": 1}}}],
        tool_results={})
    spans.build_turn_spans(tracer, turn, session_id="sess-ts", git=("", ""), tags=[], max_chars=20000)

    finished = exp.get_finished_spans()
    llm_spans = [s for s in finished if s.attributes.get(spans.OI["KIND"]) == "LLM"]
    assert len(llm_spans) == 1

    from datetime import datetime, timezone
    expected_ns = int(datetime.fromisoformat(user_ts.replace("Z", "+00:00")).timestamp() * 1e9)
    assert llm_spans[0].start_time == expected_ns, (
        f"LLM span start_time {llm_spans[0].start_time} != turn start {expected_ns}; "
        "span fell through to wall-clock instead of backdating to turn start"
    )


def test_subagent_emitter_called_for_task_tool():
    """subagent_emitter must be invoked for Task/Agent tool spans."""
    tracer, exp = _tracer()
    calls = []

    def fake_emitter(t, ctx, tool_use_id, sid, mc, parent_start_ns=None):
        calls.append((tool_use_id, sid))

    turn = Turn(
        user_msg={"message": {"role": "user", "content": "run task"}, "timestamp": "2026-06-22T00:03:00Z"},
        assistant_msgs=[{"type": "assistant", "requestId": "r4", "timestamp": "2026-06-22T00:03:01Z",
            "message": {"id": "r4", "model": "claude-opus-4",
                "content": [
                    {"type": "tool_use", "id": "agent1", "name": "Task",
                     "input": {"description": "do something"}}
                ],
                "usage": {"input_tokens": 3, "output_tokens": 4}}}],
        tool_results={"agent1": {"content": "task done", "timestamp": "2026-06-22T00:03:02Z"}})
    spans.build_turn_spans(tracer, turn, session_id="sess-3", git=("", ""), tags=[], max_chars=20000,
                    subagent_emitter=fake_emitter)
    assert len(calls) == 1
    assert calls[0] == ("agent1", "sess-3")


# ---------------------------------------------------------------------------
# Bug B regression test — negative tool span durations
# ---------------------------------------------------------------------------

def test_tool_spans_no_negative_duration_multi_tool_same_request():
    """
    Bug B regression: when a single LLM response issues MULTIPLE tool_use blocks
    at DIFFERENT timestamps (Bash@T1, Agent@T2>T1), build_turn_spans was stamping EVERY
    tool span's start as call.end_ts (the response's LAST frame, T2), so the
    earlier Bash result@T1+0.6s got start=T2 end=T1+0.6s = NEGATIVE duration.

    Fix: each tool span's start = its OWN tool_use frame's timestamp (tc["ts"]).
    Assert: Bash start == T1 (not T2), Bash end == Bash_result_ts, NO span has end < start.
    """
    T1 = "2026-06-23T07:39:48.440Z"   # Bash tool_use frame timestamp
    T2 = "2026-06-23T07:39:50.459Z"   # Agent tool_use frame timestamp (= response last frame)
    BASH_RESULT_TS = "2026-06-23T07:39:49.074Z"   # Bash tool result timestamp
    AGENT_RESULT_TS = "2026-06-23T07:39:50.495Z"  # Agent tool result timestamp

    T1_ns = _parse_ns_spans(T1)
    BASH_RESULT_NS = _parse_ns_spans(BASH_RESULT_TS)

    turn = Turn(
        user_msg={"type": "user", "timestamp": "2026-06-23T07:39:43.961Z",
                  "message": {"role": "user", "content": "explore"}},
        assistant_msgs=[
            # Frame 1: Bash tool_use (arrives at T1)
            {"type": "assistant", "requestId": "req-multi", "timestamp": T1,
             "message": {"id": "req-multi", "model": "claude-opus-4-8",
                         "content": [{"type": "tool_use", "id": "bash-1",
                                      "name": "Bash", "input": {"command": "ls"}}],
                         "usage": {"input_tokens": 100, "output_tokens": 10}}},
            # Frame 2: Agent tool_use (arrives at T2, which is AFTER T1)
            {"type": "assistant", "requestId": "req-multi", "timestamp": T2,
             "message": {"id": "req-multi", "model": "claude-opus-4-8",
                         "content": [{"type": "tool_use", "id": "agent-1",
                                      "name": "Agent", "input": {"task": "explore"}}],
                         "usage": {"input_tokens": 100, "output_tokens": 20}}},
        ],
        tool_results={
            "bash-1": {"content": "file list", "timestamp": BASH_RESULT_TS},
            "agent-1": {"content": "agent done", "timestamp": AGENT_RESULT_TS},
        },
    )

    tracer, exp = _tracer()
    spans.build_turn_spans(tracer, turn, session_id="sess-neg", git=("", ""), tags=[], max_chars=20000)

    finished = exp.get_finished_spans()
    tool_spans = [s for s in finished if s.attributes.get(spans.OI["KIND"]) == "TOOL"]
    assert len(tool_spans) == 2, f"Expected 2 TOOL spans, got {len(tool_spans)}"

    bash_spans = [s for s in tool_spans if s.attributes.get(spans.OI["TOOL_NAME"]) == "Bash"]
    assert len(bash_spans) == 1, "Expected exactly 1 Bash TOOL span"
    bash = bash_spans[0]

    # Bash span must start at T1 (its own frame timestamp), NOT T2 (the response's last frame)
    assert bash.start_time == T1_ns, (
        f"Bash span start_time {bash.start_time} != T1 {T1_ns}; "
        "span was stamped with the response's last frame instead of Bash's own frame"
    )
    # Bash span end must be its result timestamp
    assert bash.end_time == BASH_RESULT_NS, (
        f"Bash span end_time {bash.end_time} != BASH_RESULT_NS {BASH_RESULT_NS}"
    )

    # No tool span may have negative duration
    for s in tool_spans:
        assert s.end_time >= s.start_time, (
            f"TOOL span '{s.name}' has negative duration: "
            f"start={s.start_time} end={s.end_time} delta={s.end_time - s.start_time}ns"
        )


# ---------------------------------------------------------------------------
# Nesting tests — sync tools nested under LLM gen, async Agent at turn root
# ---------------------------------------------------------------------------

def test_sync_tool_nested_under_llm_span():
    """
    Sync tool spans (Bash, Read, Write, etc.) must be children of the LLM span
    that issued them, NOT the turn root. The LLM span's end_time must also be
    >= the tool's end_time (generation contains its sync tools).
    """
    tracer, exp = _tracer()
    turn = Turn(
        user_msg={"message": {"role": "user", "content": "run bash"}, "timestamp": "2026-06-22T00:10:00Z"},
        assistant_msgs=[{"type": "assistant", "requestId": "r-nest", "timestamp": "2026-06-22T00:10:01Z",
            "message": {"id": "r-nest", "model": "claude-opus-4-8",
                "content": [{"type": "tool_use", "id": "bash-nest", "name": "Bash",
                             "input": {"command": "ls"}}],
                "usage": {"input_tokens": 10, "output_tokens": 5}}}],
        tool_results={"bash-nest": {"content": "file1\nfile2", "timestamp": "2026-06-22T00:10:03Z"}})
    spans.build_turn_spans(tracer, turn, session_id="sess-nest", git=("", ""), tags=[], max_chars=20000)

    finished = exp.get_finished_spans()
    llm_spans = [s for s in finished if s.attributes.get(spans.OI["KIND"]) == "LLM"]
    tool_spans = [s for s in finished if s.attributes.get(spans.OI["KIND"]) == "TOOL"]
    root_spans = [s for s in finished if s.attributes.get(spans.OI["KIND"]) == "AGENT"]

    assert len(llm_spans) == 1, f"Expected 1 LLM span, got {len(llm_spans)}"
    assert len(tool_spans) == 1, f"Expected 1 TOOL span, got {len(tool_spans)}"
    assert len(root_spans) == 1, f"Expected 1 AGENT span, got {len(root_spans)}"

    llm = llm_spans[0]
    bash = tool_spans[0]
    root = root_spans[0]

    # Bash TOOL span must be nested under the LLM span (not the turn root)
    assert bash.parent is not None, "Bash TOOL span must have a parent"
    assert bash.parent.span_id == llm.context.span_id, (
        f"Bash TOOL span parent {bash.parent.span_id!r} != LLM span id {llm.context.span_id!r}; "
        "sync tool must be nested under the LLM generation that issued it"
    )
    # Must NOT be parented to the turn root
    assert bash.parent.span_id != root.context.span_id, (
        "Bash TOOL span must NOT be a direct child of the turn root"
    )

    # LLM span must contain (end >= ) the Bash tool's end time
    assert llm.end_time >= bash.end_time, (
        f"LLM span end_time {llm.end_time} < Bash tool end_time {bash.end_time}; "
        "LLM generation must contain its sync tool children"
    )


def test_async_agent_tool_stays_at_turn_root():
    """
    Task/Agent tool spans must stay at the turn root level (child of AGENT root),
    NOT nested under the LLM span. Sub-agent LLM spans are still nested under the
    Agent TOOL span (containment preserved).
    """
    tracer, exp = _tracer()

    fake_sub_llm_start = _parse_ns_spans("2026-06-22T00:20:05Z")
    fake_sub_llm_end = _parse_ns_spans("2026-06-22T00:20:10Z")

    captured_ctx = {}

    def fake_emitter(t, ctx, tool_use_id, sid, mc, parent_start_ns=None):
        # Emit a sub-agent LLM span under the provided ctx (which should be the Agent TOOL span)
        captured_ctx["ctx"] = ctx
        sub_span = t.start_span(
            "claude-haiku-4-5",
            start_time=fake_sub_llm_start,
            context=ctx,
            attributes={spans.OI["KIND"]: "LLM", spans.OI["SESSION"]: sid,
                        spans.OI["MODEL"]: "claude-haiku-4-5",
                        spans.OI["TOK_PROMPT"]: 5, spans.OI["TOK_COMPLETION"]: 10,
                        spans.OI["TOK_TOTAL"]: 15, spans.OI["CACHE_READ"]: 0,
                        spans.OI["CACHE_CREATE"]: 0},
        )
        sub_span.end(end_time=fake_sub_llm_end)
        return fake_sub_llm_end

    user_row = {"type": "user", "timestamp": "2026-06-22T00:20:00Z",
                "message": {"role": "user", "content": "spawn agent"}}
    assistant_row = {"type": "assistant", "requestId": "r-async", "timestamp": "2026-06-22T00:20:01Z",
        "message": {"id": "r-async", "model": "claude-opus-4-8",
            "content": [{"type": "tool_use", "id": "agent-async", "name": "Agent",
                         "input": {"description": "do async work"}}],
            "usage": {"input_tokens": 8, "output_tokens": 3}}}
    tool_result_row = {"type": "user", "timestamp": "2026-06-22T00:20:02Z",
        "message": {"role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "agent-async",
                         "content": "Agent launched"}]}}

    turn = assemble_turns([user_row, assistant_row, tool_result_row])[0]
    spans.build_turn_spans(tracer, turn, session_id="sess-async", git=("", ""), tags=[], max_chars=20000,
                    subagent_emitter=fake_emitter)

    finished = exp.get_finished_spans()
    llm_spans = [s for s in finished if s.attributes.get(spans.OI["KIND"]) == "LLM"]
    tool_spans = [s for s in finished if s.attributes.get(spans.OI["KIND"]) == "TOOL"]
    root_spans = [s for s in finished if s.attributes.get(spans.OI["KIND"]) == "AGENT"]

    assert len(root_spans) == 1
    root = root_spans[0]

    agent_tool_spans = [s for s in tool_spans if s.attributes.get(spans.OI["TOOL_NAME"]) == "Agent"]
    assert len(agent_tool_spans) == 1, f"Expected 1 Agent TOOL span, got {len(agent_tool_spans)}"
    agent_tool = agent_tool_spans[0]

    # Agent TOOL span must be a child of the TURN ROOT (not the LLM span)
    assert agent_tool.parent is not None, "Agent TOOL span must have a parent"
    assert agent_tool.parent.span_id == root.context.span_id, (
        f"Agent TOOL span parent {agent_tool.parent.span_id!r} != turn root id {root.context.span_id!r}; "
        "async Agent tool must stay at turn root level"
    )

    # The sub-agent LLM spans must be nested under the Agent TOOL span
    sub_llm_spans = [s for s in llm_spans
                     if s.parent is not None and s.parent.span_id == agent_tool.context.span_id]
    assert sub_llm_spans, "Sub-agent LLM spans must be nested under the Agent TOOL span"

    # Agent TOOL span must contain its sub-agent children (containment)
    max_sub_end = max(s.end_time for s in sub_llm_spans)
    assert agent_tool.end_time >= max_sub_end, (
        f"Agent TOOL span end_time {agent_tool.end_time} < max sub-agent end {max_sub_end}"
    )


def test_llm_spans_remain_sequential_non_overlapping():
    """
    When there are multiple LLM calls in one turn, each LLM span must start
    at or after the previous LLM call's boundary (no overlapping gen spans).
    """
    tracer, exp = _tracer()
    turn = Turn(
        user_msg={"message": {"role": "user", "content": "two calls"}, "timestamp": "2026-06-22T00:30:00Z"},
        assistant_msgs=[
            {"type": "assistant", "requestId": "rA", "timestamp": "2026-06-22T00:30:01Z",
             "message": {"id": "rA", "model": "claude-opus-4-8",
                "content": [{"type": "tool_use", "id": "b1", "name": "Bash",
                             "input": {"command": "ls"}}],
                "usage": {"input_tokens": 5, "output_tokens": 3}}},
            {"type": "assistant", "requestId": "rB", "timestamp": "2026-06-22T00:30:05Z",
             "message": {"id": "rB", "model": "claude-opus-4-8",
                "content": [{"type": "text", "text": "done"}],
                "usage": {"input_tokens": 8, "output_tokens": 4}}},
        ],
        tool_results={"b1": {"content": "output", "timestamp": "2026-06-22T00:30:03Z"}})
    spans.build_turn_spans(tracer, turn, session_id="sess-seq", git=("", ""), tags=[], max_chars=20000)

    finished = exp.get_finished_spans()
    llm_spans = sorted(
        [s for s in finished if s.attributes.get(spans.OI["KIND"]) == "LLM"],
        key=lambda s: s.start_time,
    )
    assert len(llm_spans) == 2, f"Expected 2 LLM spans, got {len(llm_spans)}"

    # Second LLM span must start at or after the first LLM span ends
    first, second = llm_spans
    assert second.start_time >= first.start_time, (
        f"LLM spans overlap: second.start={second.start_time} < first.start={first.start_time}"
    )


# ---------------------------------------------------------------------------
# Delta-input tests (Change 1: delta inputs — new messages only, not full history)
# ---------------------------------------------------------------------------

def test_delta_input_first_call_has_user_text_only():
    """
    First LLM call in a turn: input.value must contain the user text.
    It must NOT contain any assistant or tool_result content.
    """
    tracer, exp = _tracer()
    turn = Turn(
        user_msg={"message": {"role": "user", "content": "hello from user"}, "timestamp": "2026-06-22T01:00:00Z"},
        assistant_msgs=[{"type": "assistant", "requestId": "r-first", "timestamp": "2026-06-22T01:00:01Z",
            "message": {"id": "r-first", "model": "claude-opus-4-8",
                "content": [{"type": "text", "text": "hi"}],
                "usage": {"input_tokens": 5, "output_tokens": 3}}}],
        tool_results={})
    spans.build_turn_spans(tracer, turn, session_id="sess-delta-1", git=("", ""), tags=[], max_chars=20000)

    finished = exp.get_finished_spans()
    llm_spans = [s for s in finished if s.attributes.get(spans.OI["KIND"]) == "LLM"]
    assert len(llm_spans) == 1

    inp = llm_spans[0].attributes.get(spans.OI["INPUT"])
    assert inp is not None, "First LLM call must have input.value"
    assert "hello from user" in inp, "First call input must contain the user text"
    # Must NOT contain assistant or tool_result role markers
    assert '"role": "assistant"' not in inp, "First call input must NOT contain assistant turns"
    assert '"role": "tool"' not in inp, "First call input must NOT contain tool_result turns"


def test_delta_input_second_call_has_tool_results_not_user_prompt():
    """
    When call-1 issues a tool and call-2 follows, call-2's input.value must be
    the tool result delta (role=tool, tool_results=[...]), and must NOT repeat
    the original user prompt.
    """
    tracer, exp = _tracer()
    turn = Turn(
        user_msg={"message": {"role": "user", "content": "ORIGINAL USER PROMPT"}, "timestamp": "2026-06-22T01:10:00Z"},
        assistant_msgs=[
            # Call 1: issues a Bash tool
            {"type": "assistant", "requestId": "rD1", "timestamp": "2026-06-22T01:10:01Z",
             "message": {"id": "rD1", "model": "claude-opus-4-8",
                "content": [{"type": "tool_use", "id": "td1", "name": "Bash",
                             "input": {"command": "ls"}}],
                "usage": {"input_tokens": 10, "output_tokens": 5}}},
            # Call 2: follows after tool result
            {"type": "assistant", "requestId": "rD2", "timestamp": "2026-06-22T01:10:05Z",
             "message": {"id": "rD2", "model": "claude-opus-4-8",
                "content": [{"type": "text", "text": "done"}],
                "usage": {"input_tokens": 15, "output_tokens": 4}}},
        ],
        tool_results={"td1": {"content": "TOOL OUTPUT CONTENT", "timestamp": "2026-06-22T01:10:03Z"}})
    spans.build_turn_spans(tracer, turn, session_id="sess-delta-2", git=("", ""), tags=[], max_chars=20000)

    finished = exp.get_finished_spans()
    llm_spans = sorted(
        [s for s in finished if s.attributes.get(spans.OI["KIND"]) == "LLM"],
        key=lambda s: s.start_time,
    )
    assert len(llm_spans) == 2, f"Expected 2 LLM spans, got {len(llm_spans)}"

    first, second = llm_spans

    # First call: must have user text
    first_inp = first.attributes.get(spans.OI["INPUT"])
    assert first_inp is not None, "First LLM call must have input.value"
    assert "ORIGINAL USER PROMPT" in first_inp, "First call input must contain user text"

    # Second call: must have the tool result (delta), NOT the original user prompt
    second_inp = second.attributes.get(spans.OI["INPUT"])
    assert second_inp is not None, "Second LLM call must have input.value (tool result delta)"
    assert "TOOL OUTPUT CONTENT" in second_inp, (
        "Second call input must contain the tool result from call-1"
    )
    assert "ORIGINAL USER PROMPT" not in second_inp, (
        "Second call input must NOT repeat the original user prompt (delta only)"
    )
    # The delta must use the tool role shape
    assert "tool_results" in second_inp, (
        "Second call input must use {role: tool, tool_results: [...]} shape"
    )
    assert "td1" in second_inp, "Second call delta must reference the tool_call_id"


def test_delta_input_no_tool_second_call_omits_input():
    """
    When call-1 produces text (no tools), call-2 should omit input.value
    rather than setting it to an empty object.
    """
    tracer, exp = _tracer()
    turn = Turn(
        user_msg={"message": {"role": "user", "content": "hi"}, "timestamp": "2026-06-22T01:20:00Z"},
        assistant_msgs=[
            # Call 1: text only, no tools
            {"type": "assistant", "requestId": "rN1", "timestamp": "2026-06-22T01:20:01Z",
             "message": {"id": "rN1", "model": "claude-opus-4-8",
                "content": [{"type": "text", "text": "thinking..."}],
                "usage": {"input_tokens": 5, "output_tokens": 3}}},
            # Call 2: follows call-1 which had NO tools
            {"type": "assistant", "requestId": "rN2", "timestamp": "2026-06-22T01:20:02Z",
             "message": {"id": "rN2", "model": "claude-opus-4-8",
                "content": [{"type": "text", "text": "done"}],
                "usage": {"input_tokens": 7, "output_tokens": 2}}},
        ],
        tool_results={})
    spans.build_turn_spans(tracer, turn, session_id="sess-delta-3", git=("", ""), tags=[], max_chars=20000)

    finished = exp.get_finished_spans()
    llm_spans = sorted(
        [s for s in finished if s.attributes.get(spans.OI["KIND"]) == "LLM"],
        key=lambda s: s.start_time,
    )
    assert len(llm_spans) == 2, f"Expected 2 LLM spans, got {len(llm_spans)}"

    _, second = llm_spans
    # Second call with no prior tools must omit input.value
    assert spans.OI["INPUT"] not in second.attributes, (
        "Second call with no prior tools must omit input.value (not set empty object)"
    )


# ---------------------------------------------------------------------------
# SDK name/version tests (Change 2)
# ---------------------------------------------------------------------------

def test_sdk_name_and_version_on_agent_span():
    """
    The root AGENT span must carry traceroot.sdk.name == SDK_NAME and
    traceroot.sdk.version == SDK_VERSION, overriding any 'traceroot-py' value
    set by the transport's on_start callback.
    """
    tracer, exp = _tracer()
    turn = Turn(
        user_msg={"message": {"role": "user", "content": "sdk test"}, "timestamp": "2026-06-22T02:00:00Z"},
        assistant_msgs=[{"type": "assistant", "requestId": "r-sdk", "timestamp": "2026-06-22T02:00:01Z",
            "message": {"id": "r-sdk", "model": "claude-opus-4-8",
                "content": [{"type": "text", "text": "ok"}],
                "usage": {"input_tokens": 1, "output_tokens": 1}}}],
        tool_results={})
    spans.build_turn_spans(tracer, turn, session_id="sess-sdk", git=("", ""), tags=[], max_chars=20000)

    finished = exp.get_finished_spans()
    agent_spans = [s for s in finished if s.attributes.get(spans.OI["KIND"]) == "AGENT"]
    assert len(agent_spans) == 1

    a = agent_spans[0]
    assert a.attributes.get("traceroot.sdk.name") == spans.SDK_NAME, (
        f"AGENT span traceroot.sdk.name={a.attributes.get('traceroot.sdk.name')!r}, "
        f"expected {spans.SDK_NAME!r} (must NOT be 'traceroot-py')"
    )
    assert a.attributes.get("traceroot.sdk.version") == spans.SDK_VERSION, (
        f"AGENT span traceroot.sdk.version={a.attributes.get('traceroot.sdk.version')!r}, "
        f"expected {spans.SDK_VERSION!r}"
    )
    # Explicitly verify it is not the default transport value
    assert a.attributes.get("traceroot.sdk.name") != "traceroot-py", (
        "AGENT span sdk.name must be overridden from traceroot-py to the plugin name"
    )


def test_sdk_name_and_version_on_llm_and_tool_spans():
    """
    LLM and TOOL spans must also carry traceroot.sdk.name == SDK_NAME and
    traceroot.sdk.version == SDK_VERSION.
    """
    tracer, exp = _tracer()
    turn = Turn(
        user_msg={"message": {"role": "user", "content": "sdk all spans"}, "timestamp": "2026-06-22T02:10:00Z"},
        assistant_msgs=[{"type": "assistant", "requestId": "r-sdk2", "timestamp": "2026-06-22T02:10:01Z",
            "message": {"id": "r-sdk2", "model": "claude-opus-4-8",
                "content": [{"type": "tool_use", "id": "ts1", "name": "Read",
                             "input": {"file_path": "x.py"}}],
                "usage": {"input_tokens": 3, "output_tokens": 2}}}],
        tool_results={"ts1": {"content": "file content", "timestamp": "2026-06-22T02:10:02Z"}})
    spans.build_turn_spans(tracer, turn, session_id="sess-sdk2", git=("", ""), tags=[], max_chars=20000)

    finished = exp.get_finished_spans()
    for s in finished:
        kind = s.attributes.get(spans.OI["KIND"])
        if kind in ("AGENT", "LLM", "TOOL"):
            assert s.attributes.get("traceroot.sdk.name") == spans.SDK_NAME, (
                f"{kind} span '{s.name}' traceroot.sdk.name="
                f"{s.attributes.get('traceroot.sdk.name')!r}, expected {spans.SDK_NAME!r}"
            )
            assert s.attributes.get("traceroot.sdk.version") == spans.SDK_VERSION, (
                f"{kind} span '{s.name}' traceroot.sdk.version="
                f"{s.attributes.get('traceroot.sdk.version')!r}, expected {spans.SDK_VERSION!r}"
            )


def test_cache_write_uses_backend_key_for_correct_pricing():
    """Cache-creation tokens must be emitted under the OpenInference
    `prompt_details.cache_write` key — the one the TraceRoot backend reads for its
    cache-write price bucket. Emitting `prompt_details.cache_creation` (the Anthropic
    name) silently mis-prices them as uncached input (~8% cost error)."""
    tracer, exp = _tracer()
    turn = Turn(
        user_msg={"message": {"role": "user", "content": "hi"},
                  "timestamp": "2026-06-23T00:00:00Z"},
        assistant_msgs=[{
            "type": "assistant", "requestId": "r1", "timestamp": "2026-06-23T00:00:02Z",
            "message": {"id": "r1", "model": "claude-opus-4-8",
                        "content": [{"type": "text", "text": "ok"}],
                        "usage": {"input_tokens": 100, "output_tokens": 50,
                                  "cache_read_input_tokens": 16088,
                                  "cache_creation_input_tokens": 2689}}}],
        tool_results={})
    spans.build_turn_spans(tracer, turn, "s", ("o/r", "abc"), ["claude-code"], 20000)
    llm = [s for s in exp.get_finished_spans()
           if s.attributes.get(spans.OI["KIND"]) == "LLM"][0]
    # The backend reads this exact key for the cache-write bucket.
    assert llm.attributes.get("llm.token_count.prompt_details.cache_write") == 2689
    # The Anthropic-named key must NOT be the one we emit (backend ignores it).
    assert "llm.token_count.prompt_details.cache_creation" not in llm.attributes
    assert llm.attributes.get("llm.token_count.prompt_details.cache_read") == 16088


# ---------------------------------------------------------------------------
# Change (a): tool_calls in LLM output.value
# ---------------------------------------------------------------------------

def test_llm_output_includes_tool_calls_when_no_text():
    """
    A pure-tool-use LLM call (no text) must produce a non-empty output.value
    that includes the tool name and input args.
    """
    tracer, exp = _tracer()
    turn = Turn(
        user_msg={"message": {"role": "user", "content": "run bash"}, "timestamp": "2026-06-22T10:00:00Z"},
        assistant_msgs=[{
            "type": "assistant", "requestId": "r-tc1", "timestamp": "2026-06-22T10:00:01Z",
            "message": {
                "id": "r-tc1", "model": "claude-opus-4-8",
                "content": [
                    {"type": "tool_use", "id": "tc-bash-1", "name": "Bash",
                     "input": {"command": "ls /tmp", "description": "list files"}}
                ],
                "usage": {"input_tokens": 10, "output_tokens": 5},
            },
        }],
        tool_results={"tc-bash-1": {"content": "output", "timestamp": "2026-06-22T10:00:02Z"}},
    )
    spans.build_turn_spans(tracer, turn, session_id="sess-tc1", git=("", ""), tags=[], max_chars=20000)

    finished = exp.get_finished_spans()
    llm_spans = [s for s in finished if s.attributes.get(spans.OI["KIND"]) == "LLM"]
    assert len(llm_spans) == 1
    out = llm_spans[0].attributes.get(spans.OI["OUTPUT"])
    assert out, "Pure-tool-use LLM call must have non-empty output.value"
    assert "Bash" in out, f"output.value must contain tool name 'Bash', got: {out!r}"
    assert "ls /tmp" in out, f"output.value must contain tool input args, got: {out!r}"


def test_llm_output_includes_tool_calls_and_text():
    """
    When an LLM call emits both text and a tool_use, output.value must include both.
    """
    tracer, exp = _tracer()
    turn = Turn(
        user_msg={"message": {"role": "user", "content": "read and tell me"}, "timestamp": "2026-06-22T10:01:00Z"},
        assistant_msgs=[
            {
                "type": "assistant", "requestId": "r-tc2", "timestamp": "2026-06-22T10:01:01Z",
                "message": {
                    "id": "r-tc2", "model": "claude-opus-4-8",
                    "content": [{"type": "text", "text": "Let me read that file."}],
                    "usage": {"input_tokens": 8, "output_tokens": 6},
                },
            },
            {
                "type": "assistant", "requestId": "r-tc2", "timestamp": "2026-06-22T10:01:02Z",
                "message": {
                    "id": "r-tc2", "model": "claude-opus-4-8",
                    "content": [
                        {"type": "tool_use", "id": "tc-read-1", "name": "Read",
                         "input": {"file_path": "notes.txt"}}
                    ],
                    "usage": {"input_tokens": 8, "output_tokens": 10},
                },
            },
        ],
        tool_results={"tc-read-1": {"content": "note contents", "timestamp": "2026-06-22T10:01:03Z"}},
    )
    spans.build_turn_spans(tracer, turn, session_id="sess-tc2", git=("", ""), tags=[], max_chars=20000)

    finished = exp.get_finished_spans()
    llm_spans = [s for s in finished if s.attributes.get(spans.OI["KIND"]) == "LLM"]
    assert len(llm_spans) == 1
    out = llm_spans[0].attributes.get(spans.OI["OUTPUT"])
    assert out, "LLM call with text+tool must have non-empty output.value"
    assert "Let me read that file." in out, f"output.value must contain the text, got: {out!r}"
    assert "Read" in out, f"output.value must contain tool name 'Read', got: {out!r}"


# ---------------------------------------------------------------------------
# Error-status surfacing — failed tool_result (is_error) → span status ERROR
# ---------------------------------------------------------------------------

def test_failed_tool_result_sets_span_status_error():
    """
    A tool_result carrying is_error=true (failed Bash, missing file, etc.) must
    set the TOOL span's OTEL status to ERROR so the backend stores it as an error
    span (status.code==2) and the trace's error_count reflects the failure.
    """
    tracer, exp = _tracer()
    turn = Turn(
        user_msg={"message": {"role": "user", "content": "run bad cmd"}, "timestamp": "2026-06-22T03:00:00Z"},
        assistant_msgs=[{"type": "assistant", "requestId": "r-err", "timestamp": "2026-06-22T03:00:01Z",
            "message": {"id": "r-err", "model": "claude-opus-4-8",
                "content": [{"type": "tool_use", "id": "bad-1", "name": "Bash",
                             "input": {"command": "exit 1"}}],
                "usage": {"input_tokens": 5, "output_tokens": 3}}}],
        tool_results={"bad-1": {"content": "bash: command failed with exit code 1",
                                "timestamp": "2026-06-22T03:00:02Z", "is_error": True}})
    spans.build_turn_spans(tracer, turn, session_id="sess-err", git=("", ""), tags=[], max_chars=20000)

    tool_spans = [s for s in exp.get_finished_spans()
                  if s.attributes.get(spans.OI["KIND"]) == "TOOL"]
    assert len(tool_spans) == 1
    ts = tool_spans[0]
    assert ts.status.status_code == StatusCode.ERROR, (
        f"Failed tool span status {ts.status.status_code} != ERROR"
    )
    # Status message should carry (a prefix of) the error content for triage.
    assert ts.status.description and "command failed" in ts.status.description


def test_successful_tool_result_is_not_error():
    """A normal tool_result (no is_error) must NOT set the span status to ERROR."""
    tracer, exp = _tracer()
    turn = Turn(
        user_msg={"message": {"role": "user", "content": "run ok cmd"}, "timestamp": "2026-06-22T03:10:00Z"},
        assistant_msgs=[{"type": "assistant", "requestId": "r-ok", "timestamp": "2026-06-22T03:10:01Z",
            "message": {"id": "r-ok", "model": "claude-opus-4-8",
                "content": [{"type": "tool_use", "id": "ok-1", "name": "Bash",
                             "input": {"command": "echo hi"}}],
                "usage": {"input_tokens": 5, "output_tokens": 3}}}],
        tool_results={"ok-1": {"content": "hi", "timestamp": "2026-06-22T03:10:02Z"}})
    spans.build_turn_spans(tracer, turn, session_id="sess-ok", git=("", ""), tags=[], max_chars=20000)

    tool_spans = [s for s in exp.get_finished_spans()
                  if s.attributes.get(spans.OI["KIND"]) == "TOOL"]
    assert len(tool_spans) == 1
    assert tool_spans[0].status.status_code != StatusCode.ERROR, (
        "A successful tool result must not produce an ERROR span"
    )
