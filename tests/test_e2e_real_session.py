"""
Local end-to-end regression test — real Claude Code session 428ddacf.

This test replays the ACTUAL session that exposed three bugs and asserts
all fixed behaviours.  No mocking of transcript content; the real JSONL
files are the ground truth.

Bugs locked in by this regression:
  Bug 1 — Timeline zero: every LLM span had zero duration (start_ts missing).
  Bug 2 — Input empty: every LLM span had empty input.value.
  Bug 3 — Sub-agent orphaned: disk-resolution path broken; sub-agent LLM
           spans were not nested under the Task/Agent TOOL span.
  Bug 4 — Containment: the Agent TOOL span ended before its sub-agent children.
  Bug pass-3 — Latency accuracy: LLM spans measured frame gap (~0ms for single-
           frame), not real generation latency.  Fix: prev-boundary model.

Fixture: tests/fixtures/real-session-428ddacf/
  transcript.jsonl                            — 4-turn main session
  subagents/agent-ae71f29fe45ac2886.jsonl     — Explore sub-agent (3 LLM calls)
  subagents/agent-ae71f29fe45ac2886.meta.json — {"toolUseId": "toolu_01KZ7axWyWL6ZcgghjpkkSY3"}

Session details (from the real data):
  Turn 0: user@05:40:41.965Z, response last-frame@05:40:51.479Z → LLM duration=9.514s
  Turn 1: user@05:42:20.797Z, response@05:42:25.333Z (single-frame) → LLM=4.536s
  Turn 2: user@05:45:10.860Z, call1 last-frame@05:45:16.022Z → LLM1=5.162s;
           tool_result@05:45:16.219Z, call2@05:45:25.609Z → LLM2=9.390s
  Turn 3: user@05:46:01.894Z, response last-frame@05:46:22.528Z → LLM=20.634s

Scrub: no secrets (tr-, sk-, third-party API key prefixes, AWS-key patterns) found in
       any of the three fixture files.
"""

import shutil
from datetime import datetime, timezone
from pathlib import Path

from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from traceroot_observability import emit, state, spans

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SESSION = "428ddacf-6ec4-4821-99f5-c42f39dd9d06"
TOOL_USE_ID = "toolu_01KZ7axWyWL6ZcgghjpkkSY3"   # Agent tool call in turn 3
FIXTURES = Path(__file__).parent / "fixtures" / "real-session-428ddacf"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_tracer_and_exporter(monkeypatch):
    """Wire up an in-memory exporter and monkeypatch emit._tracer."""
    exp = InMemorySpanExporter()
    prov = TracerProvider()
    prov.add_span_processor(SimpleSpanProcessor(exp))
    tracer = prov.get_tracer("traceroot.claude-code")

    def fake_tracer(cfg, git):
        return tracer, lambda: None

    monkeypatch.setattr(emit, "_tracer", fake_tracer)
    return exp


def _reconstruct_on_disk(tmp_path: Path) -> Path:
    """
    Lay out the Claude Code on-disk structure in tmp_path so the
    disk-resolution path (resolve_subagent_transcript) can find the
    sub-agent sidecar files.

    Layout:
      tmp_path/<SESSION>.jsonl                         ← main transcript
      tmp_path/<SESSION>/subagents/agent-*.jsonl
      tmp_path/<SESSION>/subagents/agent-*.meta.json
    """
    # Main transcript
    main_dst = tmp_path / f"{SESSION}.jsonl"
    shutil.copy(FIXTURES / "transcript.jsonl", main_dst)

    # Sub-agent files
    subagents_dst = tmp_path / SESSION / "subagents"
    subagents_dst.mkdir(parents=True)
    for f in (FIXTURES / "subagents").iterdir():
        shutil.copy(f, subagents_dst / f.name)

    return main_dst


# ---------------------------------------------------------------------------
# The regression test
# ---------------------------------------------------------------------------

def test_e2e_real_session_428ddacf(tmp_path, monkeypatch):
    """
    Full pipeline replay of the real 428ddacf session.

    state.snapshots is EMPTY (we never call handle_subagent_stop), so the
    sub-agent must be found via disk-resolution from the meta.json sidecar.
    This exercises the exact production path.
    """
    monkeypatch.setattr(state, "STATE_ROOT", tmp_path)
    monkeypatch.setenv("TRACEROOT_API_KEY", "test-key")
    monkeypatch.setattr(emit, "_resolve_git", lambda payload, st: ("o/r", "abc"))

    exp = _build_tracer_and_exporter(monkeypatch)

    # Reconstruct on-disk layout; transcript_path drives disk resolution
    main_transcript = _reconstruct_on_disk(tmp_path)

    emit.handle_stop({
        "hook_event_name": "Stop",
        "session_id": SESSION,
        "transcript_path": str(main_transcript),
        "cwd": "/tmp/claude-code-plugin-tests",
    })

    finished = exp.get_finished_spans()
    assert finished, "Expected spans to be emitted"

    # ------------------------------------------------------------------
    # 1. Turns: ≥ 4 root AGENT spans
    # ------------------------------------------------------------------
    agent_spans = [
        s for s in finished
        if s.attributes.get(spans.OI["KIND"]) == "AGENT"
    ]
    assert len(agent_spans) >= 3, (
        f"Expected ≥3 AGENT root spans (one per ask; turns 2+3 are grouped into one ask), got {len(agent_spans)}: "
        + str([s.name for s in agent_spans])
    )

    # Every AGENT span must carry this session.id and a non-empty input.value
    for ag in agent_spans:
        assert ag.attributes.get(spans.OI["SESSION"]) == SESSION, (
            f"AGENT span '{ag.name}' has wrong session.id: "
            f"{ag.attributes.get(spans.OI['SESSION'])!r}"
        )
        inp = ag.attributes.get(spans.OI["INPUT"]) or ""
        assert inp, (
            f"AGENT span '{ag.name}' has empty input.value"
        )

    # ------------------------------------------------------------------
    # 2. Timeline fixed: EXACT latency assertions (prev-boundary model)
    #    Every main-session LLM span must have end_time > start_time.
    #    Turn 0 LLM span must have duration == response_last_frame_ts − user_ts.
    # ------------------------------------------------------------------
    llm_spans = [
        s for s in finished
        if s.attributes.get(spans.OI["KIND"]) == "LLM"
    ]
    assert llm_spans, "Expected at least one LLM span"

    def _ts_ns(iso: str) -> int:
        return int(datetime.fromisoformat(iso.replace("Z", "+00:00")).timestamp() * 1e9)

    # Collect main-session LLM spans (direct children of AGENT spans).
    agent_span_ids = {ag.context.span_id for ag in agent_spans}
    main_session_llm = [
        s for s in llm_spans
        if s.attributes.get(spans.OI["SESSION"]) == SESSION
        and s.parent is not None
        and s.parent.span_id in agent_span_ids
    ]
    assert main_session_llm, "Expected at least one main-session LLM span"

    # Every main-session LLM span must be strictly positive duration.
    zero_dur = [s for s in main_session_llm if (s.end_time - s.start_time) <= 0]
    assert not zero_dur, (
        "Pass-3 regression: some main-session LLM spans have zero/negative duration: "
        + str([(s.name, s.start_time, s.end_time) for s in zero_dur])
    )

    # Turn 0 exact latency check.
    # user@05:40:41.965Z → last frame@05:40:51.479Z = 9.514s
    # The LLM span in turn-0 must have:
    #   start = _ts_ns("2026-06-23T05:40:41.965Z")  (user_ts = prev-boundary)
    #   end   = _ts_ns("2026-06-23T05:40:51.479Z")  (last content-block frame)
    turn0_user_ns   = _ts_ns("2026-06-23T05:40:41.965Z")
    turn0_resp_ns   = _ts_ns("2026-06-23T05:40:51.479Z")
    expected_dur_ns = turn0_resp_ns - turn0_user_ns  # 9,514,000,000 ns

    # Find the turn-0 LLM span: it is the first-emitted main-session LLM span
    # (lowest start_time), should start at turn0_user_ns.
    turn0_llm = min(main_session_llm, key=lambda s: s.start_time)
    assert turn0_llm.start_time == turn0_user_ns, (
        f"Turn-0 LLM span start {turn0_llm.start_time} != user_ts {turn0_user_ns} "
        f"(expected prev-boundary = user message timestamp)"
    )
    assert turn0_llm.end_time == turn0_resp_ns, (
        f"Turn-0 LLM span end {turn0_llm.end_time} != last_frame_ts {turn0_resp_ns} "
        f"(expected end = last content-block timestamp)"
    )
    actual_dur_ns = turn0_llm.end_time - turn0_llm.start_time
    assert actual_dur_ns == expected_dur_ns, (
        f"Turn-0 LLM span duration {actual_dur_ns / 1e9:.3f}s != "
        f"expected {expected_dur_ns / 1e9:.3f}s (response_ts − user_ts)"
    )

    # Turn 1 single-frame check: user@05:42:20.797Z → resp@05:42:25.333Z = 4.536s
    # This was the zero-duration case under the old frame-gap model.
    turn1_user_ns = _ts_ns("2026-06-23T05:42:20.797Z")
    turn1_resp_ns = _ts_ns("2026-06-23T05:42:25.333Z")
    turn1_llm_candidates = [
        s for s in main_session_llm
        if s.start_time == turn1_user_ns
    ]
    assert turn1_llm_candidates, (
        f"Expected a main-session LLM span starting at turn1_user_ts "
        f"({turn1_user_ns}); none found. "
        f"(Pass-3: single-frame response must not be 0-duration)"
    )
    turn1_llm = turn1_llm_candidates[0]
    assert turn1_llm.end_time == turn1_resp_ns, (
        f"Turn-1 single-frame LLM span end {turn1_llm.end_time} != resp_ts {turn1_resp_ns}"
    )
    assert (turn1_llm.end_time - turn1_llm.start_time) > 0, (
        "Turn-1 single-frame LLM span is zero-duration (Pass-3 regression)"
    )

    # ------------------------------------------------------------------
    # 3. Input: main-session first-call LLM spans must have non-empty input.value
    #    (delta-input design: first call = user message; sub-agent spans omit input
    #    because only assistant rows are available in the sub-agent snapshot)
    # ------------------------------------------------------------------
    # First-call LLM spans are direct children of AGENT spans (same parent chain)
    # and are the first (lowest start_time) LLM child of each AGENT span.
    agent_to_first_llm: dict = {}
    for s in main_session_llm:
        parent_id = s.parent.span_id if s.parent else None
        if parent_id not in agent_to_first_llm or s.start_time < agent_to_first_llm[parent_id].start_time:
            agent_to_first_llm[parent_id] = s

    empty_first_call_llm = [
        s for s in agent_to_first_llm.values()
        if not (s.attributes.get(spans.OI["INPUT"]) or "")
    ]
    assert not empty_first_call_llm, (
        "Bug 2 (input) regression: first-call main-session LLM spans with empty input.value: "
        + str([s.name for s in empty_first_call_llm])
    )

    # ------------------------------------------------------------------
    # 4. Sub-agent nested via disk resolution (state.snapshots was EMPTY)
    # ------------------------------------------------------------------
    tool_spans = [
        s for s in finished
        if s.attributes.get(spans.OI["KIND"]) == "TOOL"
        and s.attributes.get(spans.OI["TOOL_NAME"]) == "Agent"
    ]
    assert tool_spans, (
        "Expected a TOOL span with tool.name='Agent' (the Explore sub-agent call)"
    )
    task_span = tool_spans[0]
    task_span_id = task_span.context.span_id

    sub_llm_spans = [
        s for s in finished
        if s.attributes.get(spans.OI["KIND"]) == "LLM"
        and s.parent is not None
        and s.parent.span_id == task_span_id
    ]
    assert sub_llm_spans, (
        "Bug 3 (disk-resolution) regression: no sub-agent LLM spans nested under "
        f"the Agent TOOL span (task_span_id={task_span_id:#x}). "
        "state.snapshots was empty so disk-resolution must have found the sidecar. "
        "All LLM spans + parents: "
        + str([
            (s.name, hex(s.parent.span_id) if s.parent else None)
            for s in llm_spans
        ])
    )

    # Sub-agent LLM spans carry the PARENT session.id (not sub-agent's own)
    for s in sub_llm_spans:
        assert s.attributes.get(spans.OI["SESSION"]) == SESSION, (
            f"Sub-agent LLM span '{s.name}' carries session.id="
            f"{s.attributes.get(spans.OI['SESSION'])!r}, expected {SESSION!r}"
        )

    # ------------------------------------------------------------------
    # 5. Containment: Agent TOOL span end_time >= max child end_time
    # ------------------------------------------------------------------
    max_child_end = max(s.end_time for s in sub_llm_spans)
    assert task_span.end_time >= max_child_end, (
        f"Bug 4 (containment) regression: Agent TOOL span end_time "
        f"{task_span.end_time} < max sub-agent child end_time {max_child_end}. "
        f"Difference: {(max_child_end - task_span.end_time) // 1_000_000} ms"
    )

    # ------------------------------------------------------------------
    # 6. Tokens: sub-agent LLM spans carry non-zero llm.token_count.total
    # ------------------------------------------------------------------
    nonzero_tok = [
        s for s in sub_llm_spans
        if (s.attributes.get(spans.OI["TOK_TOTAL"]) or 0) > 0
    ]
    assert nonzero_tok, (
        "Expected sub-agent LLM spans to carry non-zero token totals. "
        f"Got: {[s.attributes.get(spans.OI['TOK_TOTAL']) for s in sub_llm_spans]}"
    )

    # ------------------------------------------------------------------
    # 7. Every span carries a non-empty session.id
    # ------------------------------------------------------------------
    missing_session = [
        s for s in finished
        if not (s.attributes.get(spans.OI["SESSION"]) or "")
    ]
    assert not missing_session, (
        f"Spans missing session.id: "
        + str([(s.name, s.attributes.get(spans.OI["KIND"])) for s in missing_session])
    )

    # ------------------------------------------------------------------
    # 8. Change (c): sub-agent TOOL spans appear under sub-agent LLM spans
    #    (real fixture has Bash/Read tool calls in the sub-agent transcript)
    # ------------------------------------------------------------------
    sub_llm_ids = {s.context.span_id for s in sub_llm_spans}
    sub_tool_spans = [
        s for s in finished
        if s.attributes.get(spans.OI["KIND"]) == "TOOL"
        and s.parent is not None
        and s.parent.span_id in sub_llm_ids
    ]
    assert sub_tool_spans, (
        "Change (c): expected sub-agent TOOL spans (Bash/Read) nested under sub-agent "
        "LLM spans. The real fixture sub-agent issues Bash and Read tool calls."
    )

    # Sub-agent TOOL spans must have non-empty input.value and output.value
    for st in sub_tool_spans:
        assert st.attributes.get(spans.OI["INPUT"]), (
            f"Sub-agent TOOL span '{st.name}' missing input.value"
        )
        assert st.attributes.get(spans.OI["OUTPUT"]), (
            f"Sub-agent TOOL span '{st.name}' missing output.value"
        )

    # No containment violations across all sub-agent spans (LLM + TOOL)
    all_sub = sub_llm_spans + sub_tool_spans
    max_sub_end = max(s.end_time for s in all_sub)
    assert task_span.end_time >= max_sub_end, (
        f"Change (c) containment: Agent TOOL span end {task_span.end_time} < "
        f"max sub-agent span end {max_sub_end}"
    )
