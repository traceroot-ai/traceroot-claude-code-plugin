"""
Regression tests pinning bugs fixed in pass-1 and pass-3:

  Bug 1: All LLM spans had zero duration (assemble_turns dedup collapsed multi-frame
         requestIds to a single timestamp).
  Bug 2: input.value was absent on all LLM spans.
  Bug 3d: thinking blocks were silently dropped.
  Bug pass-3: LLM span timing measured frame gap (~0–60ms), not real generation
         latency.  Fix: prev-boundary model — start = user_ts (or last tool-result
         ts for subsequent calls), end = response ts.  Single-frame responses must
         NOT produce 0-duration spans.

The fixture below models a single assistant response that Claude Code writes as
TWO transcript rows sharing the same requestId:
  - Frame 0 (t0): thinking block
  - Frame 1 (t1): text block

Timestamps chosen to mirror the real session-428ddacf fixture:
  user_msg  @ 2026-06-23T05:40:41.965Z
  frame 0   @ 2026-06-23T05:40:50.336Z  (thinking)
  frame 1   @ 2026-06-23T05:40:51.479Z  (text / last frame)
  Expected LLM duration (prev-boundary model): frame1 − user_msg = 9.514s
"""

from datetime import datetime, timezone

from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from traceroot_observability import spans
from traceroot_observability.transcript import Turn, assemble_turns


# ---------------------------------------------------------------------------
# Shared fixture helpers
# ---------------------------------------------------------------------------

_USER_TS = "2026-06-23T05:40:41.965Z"   # user message timestamp
_T0 = "2026-06-23T05:40:50.336Z"        # thinking frame timestamp
_T1 = "2026-06-23T05:40:51.479Z"        # text frame timestamp (last frame)
_MODEL = "claude-opus-4-8"
_REQUEST_ID = "req_011CcKeyT2Kppf8TbaPgRBXZ"
_MSG_ID = "msg_01DJ4xyzABCDEF"
_USER_TEXT = "this is the first traceroot cc test!"
_THINKING_TEXT = "Let me think about this carefully."
_ANSWER_TEXT = "Hey! The plugin is working."


def _parse_ns(ts: str) -> int:
    """Parse an ISO-8601 timestamp string to integer nanoseconds since epoch."""
    return int(datetime.fromisoformat(ts.replace("Z", "+00:00")).timestamp() * 1e9)


def _make_two_frame_turn() -> Turn:
    """
    Build a Turn that mirrors real Claude Code transcript output:
    one assistant response emitted as two rows sharing a requestId.
    """
    user_msg = {
        "type": "user",
        "timestamp": _USER_TS,
        "message": {"role": "user", "content": _USER_TEXT},
    }
    # Frame 0: thinking block (arrives first, earlier timestamp)
    frame0 = {
        "type": "assistant",
        "requestId": _REQUEST_ID,
        "timestamp": _T0,
        "message": {
            "id": _MSG_ID,
            "model": _MODEL,
            "role": "assistant",
            "content": [{"type": "thinking", "thinking": _THINKING_TEXT}],
            "usage": {
                "input_tokens": 27896,
                "output_tokens": 0,
                "cache_read_input_tokens": 0,
                "cache_creation_input_tokens": 0,
            },
        },
    }
    # Frame 1: text block (arrives second, later timestamp)
    frame1 = {
        "type": "assistant",
        "requestId": _REQUEST_ID,
        "timestamp": _T1,
        "message": {
            "id": _MSG_ID,
            "model": _MODEL,
            "role": "assistant",
            "content": [{"type": "text", "text": _ANSWER_TEXT}],
            "usage": {
                "input_tokens": 27896,
                "output_tokens": 174,
                "cache_read_input_tokens": 0,
                "cache_creation_input_tokens": 0,
            },
        },
    }
    return Turn(user_msg=user_msg, assistant_msgs=[frame0, frame1], tool_results={})


def _tracer():
    prov = TracerProvider()
    exp = InMemorySpanExporter()
    prov.add_span_processor(SimpleSpanProcessor(exp))
    return prov.get_tracer("traceroot.test"), exp


# ---------------------------------------------------------------------------
# Test: assemble_turns retains all frames
# ---------------------------------------------------------------------------

def test_build_turns_retains_both_frames_for_same_request_id():
    """
    Regression for Bug 1: assemble_turns must append every assistant row, not dedup
    by message.id. Two frames sharing _MSG_ID must both appear in assistant_msgs.
    """
    records = [
        {
            "type": "user",
            "timestamp": "2026-06-23T05:40:41.965Z",
            "message": {"role": "user", "content": _USER_TEXT},
        },
        {
            "type": "assistant",
            "requestId": _REQUEST_ID,
            "timestamp": _T0,
            "message": {
                "id": _MSG_ID,
                "role": "assistant",
                "model": _MODEL,
                "content": [{"type": "thinking", "thinking": _THINKING_TEXT}],
                "usage": {"input_tokens": 27896, "output_tokens": 0},
            },
        },
        {
            "type": "assistant",
            "requestId": _REQUEST_ID,
            "timestamp": _T1,
            "message": {
                "id": _MSG_ID,
                "role": "assistant",
                "model": _MODEL,
                "content": [{"type": "text", "text": _ANSWER_TEXT}],
                "usage": {"input_tokens": 27896, "output_tokens": 174},
            },
        },
    ]
    turns = assemble_turns(records)
    assert len(turns) == 1
    assert len(turns[0].assistant_msgs) == 2, (
        "Both frames must be retained; dedup collapsed them to 1"
    )
    assert turns[0].assistant_msgs[0]["timestamp"] == _T0
    assert turns[0].assistant_msgs[1]["timestamp"] == _T1


# ---------------------------------------------------------------------------
# Test: build_turn_spans produces correct LLM span
# ---------------------------------------------------------------------------

def test_llm_span_duration_is_nonzero():
    """Bug 1 regression: LLM span must have duration > 0 when frames span two timestamps."""
    tracer, exp = _tracer()
    turn = _make_two_frame_turn()
    spans.build_turn_spans(tracer, turn, session_id="sess-reg", git=("", ""), tags=[], max_chars=20000)

    llm_spans = [s for s in exp.get_finished_spans() if s.attributes.get(spans.OI["KIND"]) == "LLM"]
    assert len(llm_spans) == 1, f"Expected 1 LLM span, got {len(llm_spans)}"
    llm = llm_spans[0]

    duration_ns = llm.end_time - llm.start_time
    assert duration_ns > 0, (
        f"LLM span duration is 0 ns — start_ts and end_ts were the same "
        f"(Bug 1: assemble_turns dedup collapsed frames)"
    )


def test_llm_span_latency_accuracy_prev_boundary_model():
    """
    Pass-3 regression: LLM span duration must equal response_ts − user_ts,
    NOT the ~0–60ms gap between content-block frames.

    With the prev-boundary model:
      start = user_ts  (= 2026-06-23T05:40:41.965Z)
      end   = last frame ts (= 2026-06-23T05:40:51.479Z)
      expected duration = 9.514s (exactly, modulo float precision)

    The old code used frame-gap (T1 − T0 = 1.143s), which underestimates latency.
    A single-frame response would have produced 0s under the old model.
    """
    tracer, exp = _tracer()
    turn = _make_two_frame_turn()
    spans.build_turn_spans(tracer, turn, session_id="sess-latency", git=("", ""), tags=[], max_chars=20000)

    llm_spans = [s for s in exp.get_finished_spans() if s.attributes.get(spans.OI["KIND"]) == "LLM"]
    assert len(llm_spans) == 1
    llm = llm_spans[0]

    expected_start_ns = _parse_ns(_USER_TS)  # prev-boundary = user message
    expected_end_ns   = _parse_ns(_T1)       # last content-block frame

    assert llm.start_time == expected_start_ns, (
        f"LLM span start_time {llm.start_time} != user_ts {expected_start_ns} "
        f"(expected prev-boundary = user message timestamp)"
    )
    assert llm.end_time == expected_end_ns, (
        f"LLM span end_time {llm.end_time} != last_frame_ts {expected_end_ns} "
        f"(expected end = last content-block timestamp)"
    )
    expected_duration_ns = expected_end_ns - expected_start_ns
    actual_duration_ns = llm.end_time - llm.start_time
    assert actual_duration_ns == expected_duration_ns, (
        f"LLM span duration {actual_duration_ns / 1e9:.3f}s != "
        f"expected {expected_duration_ns / 1e9:.3f}s "
        f"(response_ts − user_ts = 9.514s)"
    )


def test_llm_span_prev_boundary_threading_two_calls():
    """
    Pass-3 regression: when a turn has TWO LLM calls (call1 → tool result → call2),
    the SECOND call's start must equal the tool-result timestamp (prev-boundary),
    not the frame ts of the second response.

    Pass-5 update: LLM1's end_time now extends to contain its sync tool child (Bash),
    so llm1.end_time == tool_result_ts (not resp1_ts).

    Fixture:
      user_msg   @ T=00:00.000
      call1 resp @ T=00:02.000  (LLM1 generation ends; Bash child extends it to T+2.1)
      tool result@ T=00:02.100  (LLM1 end_time = tool_result_ts; LLM2 start = this)
      call2 resp @ T=00:05.500  (LLM2 end_time = resp2_ts)
    """
    USER_TS     = "2026-06-23T10:00:00.000Z"
    RESP1_TS    = "2026-06-23T10:00:02.000Z"
    TOOL_RES_TS = "2026-06-23T10:00:02.100Z"
    RESP2_TS    = "2026-06-23T10:00:05.500Z"
    TOOL_ID     = "t-abc"

    user_ns      = _parse_ns(USER_TS)
    resp1_ns     = _parse_ns(RESP1_TS)
    tool_res_ns  = _parse_ns(TOOL_RES_TS)
    resp2_ns     = _parse_ns(RESP2_TS)

    turn = Turn(
        user_msg={"type": "user", "timestamp": USER_TS,
                  "message": {"role": "user", "content": "do something"}},
        assistant_msgs=[
            # call1: has a tool_use
            {"type": "assistant", "requestId": "req-1", "timestamp": RESP1_TS,
             "message": {"id": "req-1", "model": "claude-haiku-4",
                         "content": [{"type": "tool_use", "id": TOOL_ID,
                                       "name": "Bash", "input": {"command": "ls"}}],
                         "usage": {"input_tokens": 10, "output_tokens": 5}}},
            # call2: plain text response after tool result
            {"type": "assistant", "requestId": "req-2", "timestamp": RESP2_TS,
             "message": {"id": "req-2", "model": "claude-haiku-4",
                         "content": [{"type": "text", "text": "done"}],
                         "usage": {"input_tokens": 15, "output_tokens": 3}}},
        ],
        tool_results={
            TOOL_ID: {"content": "file list", "timestamp": TOOL_RES_TS},
        },
    )

    tracer, exp = _tracer()
    spans.build_turn_spans(tracer, turn, session_id="sess-2call", git=("", ""), tags=[], max_chars=20000)

    llm_spans = sorted(
        [s for s in exp.get_finished_spans() if s.attributes.get(spans.OI["KIND"]) == "LLM"],
        key=lambda s: s.start_time,
    )
    assert len(llm_spans) == 2, f"Expected 2 LLM spans, got {len(llm_spans)}"

    llm1, llm2 = llm_spans

    # LLM1: start = user_ts, end = tool_result_ts (extended to contain sync Bash child)
    assert llm1.start_time == user_ns, (
        f"LLM1 start {llm1.start_time} != user_ts {user_ns}"
    )
    assert llm1.end_time == tool_res_ns, (
        f"LLM1 end {llm1.end_time} != tool_res_ns {tool_res_ns}; "
        "LLM span must be extended to contain its sync Bash child (pass-5 nesting)"
    )

    # LLM2: start = tool_result_ts (prev-boundary after call1's tool), end = resp2_ts
    assert llm2.start_time == tool_res_ns, (
        f"LLM2 start {llm2.start_time} != tool_result_ts {tool_res_ns}; "
        f"expected prev-boundary threading to advance past tool result"
    )
    assert llm2.end_time == resp2_ns, (
        f"LLM2 end {llm2.end_time} != resp2_ts {resp2_ns}"
    )


def test_single_frame_response_is_not_zero_duration():
    """
    Pass-3 regression: a response with exactly ONE content-block frame must have
    non-zero duration.  Old code produced 0s (end_ts = start_ts for single-frame).
    New code: start = user_ts (prev-boundary), end = response frame ts.
    """
    USER_TS = "2026-06-23T11:00:00.000Z"
    RESP_TS = "2026-06-23T11:00:04.536Z"

    user_ns = _parse_ns(USER_TS)
    resp_ns = _parse_ns(RESP_TS)

    turn = Turn(
        user_msg={"type": "user", "timestamp": USER_TS,
                  "message": {"role": "user", "content": "hello"}},
        assistant_msgs=[
            # Single frame: start_ts == end_ts under old model → 0s duration
            {"type": "assistant", "requestId": "req-sf", "timestamp": RESP_TS,
             "message": {"id": "req-sf", "model": "claude-haiku-4",
                         "content": [{"type": "text", "text": "hi there"}],
                         "usage": {"input_tokens": 5, "output_tokens": 2}}},
        ],
        tool_results={},
    )

    tracer, exp = _tracer()
    spans.build_turn_spans(tracer, turn, session_id="sess-sf", git=("", ""), tags=[], max_chars=20000)

    llm_spans = [s for s in exp.get_finished_spans() if s.attributes.get(spans.OI["KIND"]) == "LLM"]
    assert len(llm_spans) == 1
    llm = llm_spans[0]

    assert llm.start_time == user_ns, (
        f"Single-frame LLM span start {llm.start_time} != user_ts {user_ns}"
    )
    assert llm.end_time == resp_ns, (
        f"Single-frame LLM span end {llm.end_time} != resp_ts {resp_ns}"
    )
    assert llm.end_time > llm.start_time, (
        f"Single-frame LLM span has zero duration! start={llm.start_time} end={llm.end_time}"
    )


def test_llm_span_input_value_present():
    """Bug 2 regression: LLM span must have input.value set and contain the user text."""
    tracer, exp = _tracer()
    turn = _make_two_frame_turn()
    spans.build_turn_spans(tracer, turn, session_id="sess-reg", git=("", ""), tags=[], max_chars=20000)

    llm_spans = [s for s in exp.get_finished_spans() if s.attributes.get(spans.OI["KIND"]) == "LLM"]
    assert len(llm_spans) == 1
    llm = llm_spans[0]

    inp = llm.attributes.get(spans.OI["INPUT"])
    assert inp, "input.value is absent or empty on LLM span (Bug 2)"
    assert _USER_TEXT in inp, (
        f"input.value does not contain the user text '{_USER_TEXT}'. Got: {inp[:200]}"
    )


def test_llm_span_thinking_attribute_present():
    """Bug 3d regression: gen_ai.thinking must be set and contain the thinking text."""
    tracer, exp = _tracer()
    turn = _make_two_frame_turn()
    spans.build_turn_spans(tracer, turn, session_id="sess-reg", git=("", ""), tags=[], max_chars=20000)

    llm_spans = [s for s in exp.get_finished_spans() if s.attributes.get(spans.OI["KIND"]) == "LLM"]
    assert len(llm_spans) == 1
    llm = llm_spans[0]

    thinking = llm.attributes.get("gen_ai.thinking")
    assert thinking, "gen_ai.thinking is absent or empty on LLM span (Bug 3d)"
    assert _THINKING_TEXT in thinking, (
        f"gen_ai.thinking does not contain expected text '{_THINKING_TEXT}'. Got: {thinking[:200]}"
    )
