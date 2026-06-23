"""
End-to-end fixture replay test.

Drives the real plugin pipeline over a synthetic-but-realistic Claude Code
transcript fixture and asserts the full span tree.

Fixture: tests/fixtures/simple-session/
  transcript.jsonl          — 2 user turns; turn 2 fires a Task sub-agent
  subagents/agent-sub001.jsonl — sub-agent's own transcript

The fixture is SYNTHETIC (see Fixture shape note below), modelled exactly on the
real Claude Code JSONL schema.  A real captured fixture will replace it once
manual end-to-end verification against a live Claude Code session is done.

Pipeline under test:
  1. emit.handle_subagent_stop  — snapshots the sub-agent transcript
  2. emit.handle_stop           — reads main transcript, builds turns, emits spans

Assertions:
  A. Two root AGENT spans (one per user turn).
  B. Every span carries a non-empty session.id.
  C. At least one LLM span has a non-zero llm.token_count.total.
  D. A nested sub-agent LLM span exists under the Task TOOL span:
       - its openinference.span.kind == "LLM"
       - it carries the PARENT session.id (not the sub-agent's own)
       - it is a child of the Task TOOL span (verified via parent_span_id)
"""

from pathlib import Path
import json

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from traceroot_observability import emit, state, spans

FIXTURES = Path(__file__).parent / "fixtures" / "simple-session"
SESSION_ID = "fix-sess-0001-0000-0000-000000000001"
SUBAGENT_ID = "sub001"
TOOL_USE_ID = "tu-task-001"


def _build_tracer_and_exporter():
    exp = InMemorySpanExporter()
    prov = TracerProvider()
    prov.add_span_processor(SimpleSpanProcessor(exp))
    tracer = prov.get_tracer("traceroot.claude-code")

    def fake_tracer(cfg, git):
        return tracer, lambda: None

    return tracer, exp, fake_tracer


def test_fixture_replay_full_pipeline(tmp_path, monkeypatch):
    """
    End-to-end: SubagentStop snapshot + Stop emission produce a correct span tree.
    """
    monkeypatch.setattr(state, "STATE_ROOT", tmp_path)
    monkeypatch.setenv("TRACEROOT_API_KEY", "test-key")

    tracer, exp, fake_tracer = _build_tracer_and_exporter()
    monkeypatch.setattr(emit, "_tracer", fake_tracer)
    monkeypatch.setattr(emit, "_resolve_git", lambda payload, st: ("o/r", "abc"))

    # --- Step 1: SubagentStop for the Task sub-agent -------------------------
    subagent_tpath = str(FIXTURES / "subagents" / "agent-sub001.jsonl")
    emit.handle_subagent_stop({
        "hook_event_name": "SubagentStop",
        "session_id": SESSION_ID,
        "agent_transcript_path": subagent_tpath,
        "tool_use_id": TOOL_USE_ID,
    })

    # --- Step 2: SessionEnd for the main session (flushes Task-launching ask) ---
    main_tpath = str(FIXTURES / "transcript.jsonl")
    emit.handle_session_end({
        "hook_event_name": "SessionEnd",
        "session_id": SESSION_ID,
        "transcript_path": main_tpath,
        "cwd": "/tmp/my-project",
    })

    finished = exp.get_finished_spans()
    assert finished, "Expected at least one span to be emitted"

    # -------------------------------------------------------------------------
    # A. Two root AGENT spans (one per user turn)
    # -------------------------------------------------------------------------
    agent_spans = [s for s in finished if s.attributes.get(spans.OI["KIND"]) == "AGENT"]
    assert len(agent_spans) == 2, (
        f"Expected 2 AGENT root spans (one per turn), got {len(agent_spans)}: "
        f"{[s.name for s in agent_spans]}"
    )

    # -------------------------------------------------------------------------
    # B. Every span carries a non-empty session.id
    # -------------------------------------------------------------------------
    for s in finished:
        sid = s.attributes.get(spans.OI["SESSION"])
        assert sid, (
            f"Span '{s.name}' (kind={s.attributes.get(spans.OI['KIND'])}) "
            f"is missing session.id"
        )

    # -------------------------------------------------------------------------
    # C. At least one LLM span has a non-zero llm.token_count.total
    # -------------------------------------------------------------------------
    llm_spans = [s for s in finished if s.attributes.get(spans.OI["KIND"]) == "LLM"]
    assert llm_spans, "Expected at least one LLM span"
    nonzero_tok = [
        s for s in llm_spans
        if (s.attributes.get(spans.OI["TOK_TOTAL"]) or 0) > 0
    ]
    assert nonzero_tok, (
        f"Expected at least one LLM span with non-zero token total; "
        f"got totals: {[s.attributes.get(spans.OI['TOK_TOTAL']) for s in llm_spans]}"
    )

    # -------------------------------------------------------------------------
    # D. Nested sub-agent LLM span under the Task TOOL span
    # -------------------------------------------------------------------------
    # Find the Task TOOL span
    tool_spans = [
        s for s in finished
        if s.attributes.get(spans.OI["KIND"]) == "TOOL"
        and s.attributes.get(spans.OI["TOOL_NAME"]) == "Task"
    ]
    assert tool_spans, "Expected a TOOL span with tool.name='Task'"
    task_span = tool_spans[0]
    task_span_id = task_span.context.span_id

    # Sub-agent LLM spans are children of the Task TOOL span
    sub_llm_spans = [
        s for s in finished
        if s.attributes.get(spans.OI["KIND"]) == "LLM"
        and s.parent is not None
        and s.parent.span_id == task_span_id
    ]
    assert sub_llm_spans, (
        f"Expected at least one LLM span parented under the Task TOOL span "
        f"(task_span_id={task_span_id:#x}). All LLM spans and parents: "
        + str([
            (s.name, hex(s.parent.span_id) if s.parent else None)
            for s in llm_spans
        ])
    )

    # Sub-agent LLM spans must carry the PARENT session.id (not their own)
    for s in sub_llm_spans:
        assert s.attributes.get(spans.OI["SESSION"]) == SESSION_ID, (
            f"Sub-agent LLM span '{s.name}' carries session.id="
            f"{s.attributes.get(spans.OI['SESSION'])!r}, expected {SESSION_ID!r}"
        )


def test_fixture_isMeta_rows_skipped(tmp_path, monkeypatch):
    """
    isMeta rows in the transcript must not produce extra turns or spans.
    The fixture contains one isMeta row between turns 1 and 2.
    We still expect exactly 2 AGENT spans (not 3).
    """
    monkeypatch.setattr(state, "STATE_ROOT", tmp_path)
    monkeypatch.setenv("TRACEROOT_API_KEY", "test-key")

    _, exp, fake_tracer = _build_tracer_and_exporter()
    monkeypatch.setattr(emit, "_tracer", fake_tracer)
    monkeypatch.setattr(emit, "_resolve_git", lambda payload, st: ("o/r", "abc"))

    emit.handle_session_end({
        "hook_event_name": "SessionEnd",
        "session_id": SESSION_ID + "-meta",
        "transcript_path": str(FIXTURES / "transcript.jsonl"),
        "cwd": "/tmp/my-project",
    })

    agent_spans = [
        s for s in exp.get_finished_spans()
        if s.attributes.get(spans.OI["KIND"]) == "AGENT"
    ]
    assert len(agent_spans) == 2, (
        f"isMeta row must be ignored; expected 2 AGENT spans, got {len(agent_spans)}"
    )


def test_second_stop_emits_nothing_new(tmp_path, monkeypatch):
    """
    Calling handle_stop a second time with no new bytes must emit zero spans
    (byte-offset idempotency).
    """
    monkeypatch.setattr(state, "STATE_ROOT", tmp_path)
    monkeypatch.setenv("TRACEROOT_API_KEY", "test-key")

    _, exp, fake_tracer = _build_tracer_and_exporter()
    monkeypatch.setattr(emit, "_tracer", fake_tracer)
    monkeypatch.setattr(emit, "_resolve_git", lambda payload, st: ("o/r", "abc"))

    session_end_payload = {
        "hook_event_name": "SessionEnd",
        "session_id": SESSION_ID + "-idem",
        "transcript_path": str(FIXTURES / "transcript.jsonl"),
    }
    emit.handle_session_end(session_end_payload)
    count_after_first = len(exp.get_finished_spans())
    assert count_after_first > 0

    emit.handle_session_end(session_end_payload)
    assert len(exp.get_finished_spans()) == count_after_first, (
        "Second SessionEnd with no new bytes must not emit additional spans"
    )
