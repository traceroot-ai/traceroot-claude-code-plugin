from traceroot_observability import tokens


def _am(rid, model, inp, out, cr=0, cc=0, text="", ts="2026-06-22T00:00:00Z"):
    return {"type": "assistant", "requestId": rid, "timestamp": ts,
            "message": {"id": rid, "model": model, "content": [{"type": "text", "text": text}],
                        "usage": {"input_tokens": inp, "output_tokens": out,
                                  "cache_read_input_tokens": cr, "cache_creation_input_tokens": cc}}}


# --- Brief Step 1 tests ---

def test_same_requestid_counts_input_once_and_takes_output_max():
    msgs = [_am("r1", "opus", 10, 100, cr=50, cc=5),
            _am("r1", "opus", 10, 175, cr=50, cc=5)]  # streamed: output grows, input identical
    calls = tokens.llm_calls(msgs)
    assert len(calls) == 1
    c = calls[0]
    assert c.completion == 175           # max, not 275
    assert c.cache_read == 50 and c.cache_creation == 5
    assert c.prompt == 10 + 50 + 5       # input + cache, counted once


def test_distinct_requestids_are_separate_calls():
    calls = tokens.llm_calls([_am("r1", "opus", 10, 20), _am("r2", "opus", 30, 40)])
    assert [c.completion for c in calls] == [20, 40]


# --- Extra tests for text/tool_calls accumulation across frames ---

def _am_tool(rid, model, inp, out, tool_name, tool_id, tool_input, ts="2026-06-22T00:00:00Z"):
    """Build an assistant message with a tool_use content block."""
    return {
        "type": "assistant",
        "requestId": rid,
        "timestamp": ts,
        "message": {
            "id": rid,
            "model": model,
            "content": [{"type": "tool_use", "id": tool_id, "name": tool_name, "input": tool_input}],
            "usage": {"input_tokens": inp, "output_tokens": out,
                      "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0},
        },
    }


def test_text_accumulates_across_frames_same_requestid():
    """
    When multiple frames share a requestId, text from each frame is concatenated.
    This exercises the "text" field accumulation in llm_calls (streaming scenario
    where partial text arrives in successive lines before tool_use appears).
    """
    msgs = [
        _am("r1", "claude-sonnet", 100, 50, text="Hello "),
        _am("r1", "claude-sonnet", 100, 110, text="world"),  # same rid, more output
    ]
    calls = tokens.llm_calls(msgs)
    assert len(calls) == 1
    c = calls[0]
    # Text from both frames should be joined
    assert "Hello" in c.text and "world" in c.text
    # Token accounting: input once, output = max(110)
    assert c.prompt == 100
    assert c.completion == 110


def test_tool_calls_accumulate_across_frames_same_requestid():
    """
    Multiple tool_use blocks can appear in separate frames of the same requestId.
    All should appear in the resulting LlmCall.tool_calls list.
    """
    msgs = [
        _am_tool("r2", "claude-sonnet", 200, 80,
                 tool_name="read_file", tool_id="t1", tool_input={"path": "/foo"},
                 ts="2026-06-22T00:00:01Z"),
        _am_tool("r2", "claude-sonnet", 200, 120,
                 tool_name="write_file", tool_id="t2", tool_input={"path": "/bar", "content": "x"},
                 ts="2026-06-22T00:00:02Z"),
    ]
    calls = tokens.llm_calls(msgs)
    assert len(calls) == 1
    c = calls[0]
    assert len(c.tool_calls) == 2
    names = {tc["name"] for tc in c.tool_calls}
    assert names == {"read_file", "write_file"}
    # Token accounting
    assert c.prompt == 200
    assert c.completion == 120


def test_order_of_first_appearance_preserved():
    """Calls appear in the order the requestIds were first seen."""
    msgs = [
        _am("r3", "haiku", 5, 10),
        _am("r1", "haiku", 5, 20),
        _am("r3", "haiku", 5, 15),  # continuation of r3, not a new call
    ]
    calls = tokens.llm_calls(msgs)
    assert len(calls) == 2
    assert calls[0].completion == 15   # r3 max
    assert calls[1].completion == 20   # r1


def test_missing_usage_defaults_to_zero():
    """Messages without a 'usage' field should produce a zero-token LlmCall (never raise)."""
    msg = {
        "type": "assistant",
        "requestId": "r_no_usage",
        "timestamp": "2026-06-22T00:00:00Z",
        "message": {"id": "r_no_usage", "model": "claude-haiku", "content": []},
    }
    calls = tokens.llm_calls([msg])
    assert len(calls) == 1
    c = calls[0]
    assert c.prompt == 0 and c.completion == 0 and c.cache_read == 0 and c.cache_creation == 0


def test_start_and_end_timestamps():
    """start_ts = first frame's timestamp, end_ts = last frame's timestamp."""
    msgs = [
        _am("r1", "opus", 10, 50, ts="2026-06-22T10:00:00Z"),
        _am("r1", "opus", 10, 90, ts="2026-06-22T10:00:05Z"),
    ]
    calls = tokens.llm_calls(msgs)
    assert calls[0].start_ts == "2026-06-22T10:00:00Z"
    assert calls[0].end_ts == "2026-06-22T10:00:05Z"
