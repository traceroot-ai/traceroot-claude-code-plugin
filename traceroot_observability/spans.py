"""
Backdated OTEL span tree builder for Claude Code turns and asks.

Provides:
  OI              — OpenInference attribute-key constants dict.
  SDK_NAME        — Plugin SDK name (overrides traceroot-py set by the transport).
  SDK_VERSION     — Plugin SDK version (bump this when you bump .claude-plugin/plugin.json).
  middle_truncate — Keep head + tail of a string, elide the middle.
  build_turn_spans       — Build the backdated span tree for one Turn (root AGENT span,
                    child LLM spans, child TOOL spans) and write to the tracer.
  build_ask_spans        — Build the backdated span tree for one Ask (a human turn + its
                    task-notification continuation turns) under ONE root AGENT span.
"""

import json
from datetime import datetime, timezone

from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode

from .tokens import llm_calls

# ---------------------------------------------------------------------------
# SDK identity constants
# Keep SDK_VERSION in sync with .claude-plugin/plugin.json "version" field.
# ---------------------------------------------------------------------------

SDK_NAME = "traceroot-claude-code-plugin"
SDK_VERSION = "0.1.0"

# ---------------------------------------------------------------------------
# OpenInference / gen-ai attribute-key constants
# ---------------------------------------------------------------------------

OI: dict[str, str] = {
    "KIND":         "openinference.span.kind",
    "INPUT":        "input.value",
    "OUTPUT":       "output.value",
    "MODEL":        "llm.model_name",
    "GEN_MODEL":    "gen_ai.response.model",
    "TOK_PROMPT":   "llm.token_count.prompt",
    "TOK_COMPLETION": "llm.token_count.completion",
    "TOK_TOTAL":    "llm.token_count.total",
    "CACHE_READ":   "llm.token_count.prompt_details.cache_read",
    # NOTE: the backend's cache-WRITE bucket reads `...prompt_details.cache_write`
    # (the OpenInference key), NOT `...cache_creation` (the Anthropic name). Emitting
    # cache_creation here mis-prices cache-write tokens (they fall into uncached input).
    "CACHE_CREATE": "llm.token_count.prompt_details.cache_write",
    "TOOL_NAME":    "tool.name",
    "GEN_TOOL":     "gen_ai.tool.name",
    "TOOL_CALL_ID": "gen_ai.tool.call.id",
    "SESSION":      "session.id",
    "GIT_REPO":     "traceroot.git.repo",
    "GIT_REF":      "traceroot.git.ref",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ns(ts: str | None) -> int | None:
    """Convert an ISO-8601 timestamp string to integer nanoseconds, or None."""
    if not ts:
        return None
    try:
        return int(datetime.fromisoformat(ts.replace("Z", "+00:00")).timestamp() * 1e9)
    except Exception:
        return None


def middle_truncate(s: str, max_chars: int) -> str:
    """
    Return s unchanged if len(s) <= max_chars.  Otherwise keep the first half
    and last half of the budget, elide the middle, and embed the original
    character count so the reader knows what was omitted.
    """
    if s is None:
        return ""
    if len(s) <= max_chars:
        return s
    half = max_chars // 2
    return f"{s[:half]}\n…[{len(s)} chars, middle elided]…\n{s[-half:]}"


def _text(content) -> str:
    """Extract concatenated text blocks from a content field (str or list)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            x.get("text", "")
            for x in content
            if isinstance(x, dict) and x.get("type") == "text"
        )
    return ""


def _stamp_sdk(span) -> None:
    """
    Override the sdk.name/version attrs on a span.

    The transport's on_start callback sets traceroot.sdk.name="traceroot-py"
    during start_span().  Calling this AFTER start_span returns overwrites it
    with the plugin's own identity, so traces are attributed correctly.
    """
    span.set_attribute("traceroot.sdk.name", SDK_NAME)
    span.set_attribute("traceroot.sdk.version", SDK_VERSION)


# ---------------------------------------------------------------------------
# Internal: per-turn span builder (no root span)
# ---------------------------------------------------------------------------

def _build_turn_spans_into(
    tracer,
    parent_ctx,
    turn,
    session_id: str,
    max_chars: int,
    subagent_emitter=None,
    start_floor_ns: int | None = None,
) -> tuple:
    """
    Build and emit the LLM/tool/sub-agent span tree for one Turn, parented to parent_ctx.

    Does NOT create a root span.  All top-level spans (LLM spans and async Task/Agent
    TOOL spans) are children of parent_ctx.  Sync tool spans nest under their LLM span.

    Args:
        tracer:           OTEL tracer
        parent_ctx:       OTEL context to parent all top-level spans of this turn under
        turn:             Turn object
        session_id:       session.id attribute value
        max_chars:        max characters for truncation
        subagent_emitter: optional callback for Task/Agent tool spans
        start_floor_ns:   floor for LLM start times (e.g. ask root start) to avoid
                          spans before the root span window.

    Returns:
        (last_text: str, max_end_ns: int | None)
        last_text   — the last non-empty assistant text from this turn
        max_end_ns  — max end timestamp (ns) of everything emitted
                      (incl. async Agent/Task ends from sub-agent containment)
    """
    user_text = middle_truncate(
        _text((turn.user_msg.get("message") or {}).get("content")),
        max_chars,
    )
    start_ns = _ns(turn.user_msg.get("timestamp"))

    # --- LLM + TOOL child spans ------------------------------------------
    calls = llm_calls(turn.assistant_msgs)
    last_text = ""
    agent_tool_ends: list = []  # async Agent/Task tool ends (sub-agent may outlive the turn)

    # Delta-input tracking (only the new messages since the last call, not full history):
    #   first LLM call  → input = the user message
    #   subsequent calls → input = tool results from the PRIOR call's tools
    #                      (or None if the prior call had no tools)
    # carried_tool_results carries {"tool_call_id": tid, "content": output_str} entries
    # from the PREVIOUS iteration, to be injected as this call's input.
    # None means "first call" (use user message); [] means "prior call had no tools".
    carried_tool_results: list[dict] | None = None  # None = first call sentinel

    # prev_ts_ns tracks the "previous boundary" timestamp in nanoseconds.
    # An LLM span's start = the moment the model could have begun generating,
    # which is the user message time (first call) or the last tool-result time
    # (subsequent calls).  This gives real generation latency rather than the
    # ~0ms gap between content-block frames.
    # Floor at start_floor_ns (ask root start) to ensure no span predates the root.
    prev_ts_ns: int | None = start_ns
    if start_floor_ns is not None:
        if prev_ts_ns is not None:
            prev_ts_ns = max(prev_ts_ns, start_floor_ns)
        else:
            prev_ts_ns = start_floor_ns

    is_first_call = True

    for call in calls:
        # Build delta input for this LLM span.
        if is_first_call:
            # First call: input is the user message.
            input_delta: dict | None = {"role": "user", "content": user_text}
        elif carried_tool_results:
            # Subsequent call: input is the tool results from the previous call.
            input_delta = {"role": "tool", "tool_results": carried_tool_results}
        else:
            # Subsequent call but prior call had no tools — omit input.
            input_delta = None

        llm_attrs = {
            OI["KIND"]:           "LLM",
            OI["SESSION"]:        session_id,
            OI["MODEL"]:          call.model,
            OI["GEN_MODEL"]:      call.model,
            OI["TOK_PROMPT"]:     call.prompt,
            OI["TOK_COMPLETION"]: call.completion,
            OI["TOK_TOTAL"]:      call.prompt + call.completion,
            OI["CACHE_READ"]:     call.cache_read,
            OI["CACHE_CREATE"]:   call.cache_creation,
            # output.value: JSON {role,content,tool_calls} when the call made
            # tool_calls (so tool-only calls aren't empty), else the plain text.
            # Asymmetric by design — don't assume json.loads() works on every value.
            OI["OUTPUT"]:         (
                                    middle_truncate(
                                        json.dumps({
                                            "role": "assistant",
                                            "content": call.text,
                                            "tool_calls": [
                                                {"name": tc["name"], "input": tc.get("input")}
                                                for tc in call.tool_calls
                                            ],
                                        }, default=str),
                                        max_chars,
                                    )
                                    if call.tool_calls
                                    else middle_truncate(call.text, max_chars)
                                ),
        }
        if input_delta is not None:
            llm_attrs[OI["INPUT"]] = middle_truncate(json.dumps(input_delta, default=str), max_chars)
        if call.thinking:
            llm_attrs["gen_ai.thinking"] = middle_truncate(call.thinking, max_chars)

        # LLM span timing: prev-boundary → this response.
        # start = prev boundary (user msg or last tool result); never frame gap.
        # end   = last content-block frame for this requestId; floor at start.
        llm_start = prev_ts_ns if prev_ts_ns is not None else (_ns(call.start_ts) or start_ns)
        llm_end   = _ns(call.end_ts) or _ns(call.start_ts) or llm_start
        if llm_end is not None and llm_start is not None and llm_end < llm_start:
            llm_end = llm_start  # no negative durations

        llm_span = tracer.start_span(
            call.model,
            start_time=llm_start,
            context=parent_ctx,
            attributes=llm_attrs,
        )
        _stamp_sdk(llm_span)
        # Do NOT end llm_span yet — sync tools will be nested under it, and we
        # need to extend llm_span's end to contain them.
        llm_ctx = trace.set_span_in_context(llm_span)

        is_first_call = False

        if call.text:
            last_text = call.text

        # One TOOL span per tool_use block.
        # Sync tools (all except Task/Agent) nest under the LLM span (llm_ctx).
        # Async Task/Agent tools stay at the turn root (parent_ctx) because they
        # contain an independent sub-agent timeline.
        #
        # Two separate lists:
        #   tool_result_ends: the tool_result timestamps (when the main session got
        #     the result back) — used to advance prev_ts_ns for the next LLM call.
        #   sync_tool_ends: end times of sync-tool spans — used to extend llm_end
        #     so the LLM span contains all its sync tool children.
        tool_results_for_next: list[dict] = []  # {"tool_call_id": tid, "content": output_str}
        tool_result_ends: list[int] = []   # for prev_ts_ns advancement
        sync_tool_ends: list[int] = []     # for llm_end extension

        response_end = _ns(call.end_ts) or _ns(call.start_ts) or llm_start

        for tc in call.tool_calls:
            tid = str(tc.get("id") or "")
            res = turn.tool_results.get(tid) or {}
            result_content = res.get("content")
            if isinstance(result_content, str):
                output_str = result_content
            else:
                output_str = json.dumps(result_content, default=str)

            tool_attrs = {
                OI["KIND"]:         "TOOL",
                OI["SESSION"]:      session_id,
                OI["TOOL_NAME"]:    tc.get("name"),
                OI["GEN_TOOL"]:     tc.get("name"),
                OI["TOOL_CALL_ID"]: tid,
                OI["INPUT"]:        middle_truncate(
                                        json.dumps(tc.get("input"), default=str),
                                        max_chars,
                                    ),
                OI["OUTPUT"]:       middle_truncate(output_str, max_chars),
            }
            tool_start = _ns(tc.get("ts")) or (llm_end if llm_end is not None else (
                _ns(call.end_ts) or _ns(call.start_ts) or start_ns
            ))
            # tool_result_end: when main session received the result; floor at tool_start to avoid negative
            _res_ns = _ns(res.get("timestamp"))
            tool_result_end = max(_res_ns, tool_start) if _res_ns is not None else tool_start
            tool_end = tool_result_end  # may be extended by sub-agent below

            is_async = tc.get("name") in ("Task", "Agent")

            # Sync tools nest under the LLM span; async Task/Agent stay at parent_ctx
            span_ctx = parent_ctx if is_async else llm_ctx

            tool_span = tracer.start_span(
                str(tc.get("name") or "tool"),
                start_time=tool_start,
                context=span_ctx,
                attributes=tool_attrs,
            )
            _stamp_sdk(tool_span)

            # Surface tool failures: Claude Code marks a failed tool_result with
            # is_error=true. Set OTEL status=ERROR so the backend stores it as an
            # error span (status.code==2) and the trace's error_count reflects it.
            if res.get("is_error"):
                tool_span.set_status(
                    Status(StatusCode.ERROR, middle_truncate(output_str, 200))
                )

            # Hook for subagent traces (Task 7 wires the real emitter).
            # Capture the max sub-agent end time and extend tool_end (span containment)
            # but do NOT let the sub-agent's async end advance prev_ts_ns — the main
            # session's next LLM call may have already started before sub-agent finishes.
            if is_async and subagent_emitter:
                sub_end = subagent_emitter(
                    tracer,
                    trace.set_span_in_context(tool_span),
                    tid,
                    session_id,
                    max_chars,
                    parent_start_ns=tool_start,
                )
                tool_end = max(
                    (x for x in (tool_end, sub_end) if x is not None),
                    default=tool_end,
                )

            tool_span.end(end_time=tool_end)
            if tool_result_end is not None:
                tool_result_ends.append(tool_result_end)
            if not is_async and tool_end is not None:
                sync_tool_ends.append(tool_end)
            elif is_async and tool_end is not None:
                agent_tool_ends.append(tool_end)
            tool_results_for_next.append({"tool_call_id": tid, "content": output_str})

        # Extend llm_end to contain all sync tool children, then close the LLM span.
        llm_end = max([response_end] + sync_tool_ends)
        if llm_end < llm_start:
            llm_end = llm_start
        llm_span.end(end_time=llm_end)

        # Advance prev_ts_ns: next LLM call starts after this response + its tool RESULTS
        # (use tool_result_ends, not the sub-agent-extended tool_end).
        boundary_candidates = [t for t in ([llm_end] + tool_result_ends) if t is not None]
        prev_ts_ns = max(boundary_candidates) if boundary_candidates else llm_end

        # Advance delta carry: next LLM call gets THIS call's tool results as its input
        # (empty list if no tools were called — downstream logic omits input.value in that case).
        carried_tool_results = tool_results_for_next

    # --- compute max end time of everything in this turn -------------------
    end_ns = _ns(calls[-1].end_ts) if calls else start_ns
    # agent_tool_ends: async Agent/Task spans may outlive the LLM response
    all_candidates = [x for x in ([end_ns] + agent_tool_ends) if x is not None]
    max_end_ns = max(all_candidates) if all_candidates else end_ns

    return last_text, max_end_ns


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_turn_spans(
    tracer,
    turn,
    session_id: str,
    git: tuple,
    tags: list,
    max_chars: int,
    subagent_emitter=None,
) -> None:
    """
    Build and emit a backdated OTEL span tree for one Turn.

    Tree shape:
      AGENT  "Claude Code Turn"          (root, spans the full turn)
        LLM  <model>                     (one per requestId, child of root)
        TOOL <tool_name>                 (sync: child of LLM; async: child of root)

    All spans carry session.id.  The AGENT span also carries git attrs and tags.
    LLM spans carry token counts.  TOOL spans carry input args and result.

    subagent_emitter(tracer, ctx, tool_use_id, session_id, max_chars) is called
    for Task/Agent tool spans so Task 7 can attach subagent traces.
    """
    repo, ref = git

    # --- user message input text ------------------------------------------
    user_text = middle_truncate(
        _text((turn.user_msg.get("message") or {}).get("content")),
        max_chars,
    )
    start_ns = _ns(turn.user_msg.get("timestamp"))

    # --- root AGENT span attrs --------------------------------------------
    root_attrs: dict = {
        OI["KIND"]: "AGENT",
        OI["SESSION"]: session_id,
        OI["INPUT"]: user_text,
    }
    if repo:
        root_attrs[OI["GIT_REPO"]] = repo
    if ref:
        root_attrs[OI["GIT_REF"]] = ref
    for tag in tags:
        root_attrs[f"tag.{tag}"] = True

    root = tracer.start_span("Claude Code Turn", start_time=start_ns, attributes=root_attrs)
    _stamp_sdk(root)
    root_ctx = trace.set_span_in_context(root)

    # --- emit per-turn LLM/tool spans into the root context ---------------
    last_text, max_end_ns = _build_turn_spans_into(
        tracer, root_ctx, turn, session_id, max_chars, subagent_emitter,
        start_floor_ns=start_ns,
    )

    # --- finalise root span -----------------------------------------------
    if last_text:
        root.set_attribute(OI["OUTPUT"], middle_truncate(last_text, max_chars))

    # max_end_ns already covers the last LLM call's end plus any async tool ends.
    root.end(end_time=max_end_ns)


def build_ask_spans(
    tracer,
    ask_turns: list,
    session_id: str,
    git: tuple,
    tags: list,
    max_chars: int,
    subagent_emitter=None,
) -> None:
    """
    Build and emit a backdated OTEL span tree for one Ask.

    An Ask is a human turn plus any immediately following task-notification continuation
    turns (where the user message starts with '<task-notification>').

    Creates ONE root AGENT span:
      - input.value  = the FIRST turn's user text (the human prompt)
      - output.value = the last non-empty assistant text across all turns
      - start_time   = the first turn's user_ts
      - end_time     = max end timestamp of all spans emitted across all turns

    All turns' LLM/tool/sub-agent spans are parented to this single root, so a
    sub-agent launched in turn 1 and whose result-processing arrives in the
    task-notification turn now live in the SAME trace.
    """
    if not ask_turns:
        return

    repo, ref = git
    first_turn = ask_turns[0]

    user_text = middle_truncate(
        _text((first_turn.user_msg.get("message") or {}).get("content")),
        max_chars,
    )
    start_ns = _ns(first_turn.user_msg.get("timestamp"))

    root_attrs: dict = {
        OI["KIND"]: "AGENT",
        OI["SESSION"]: session_id,
        OI["INPUT"]: user_text,
    }
    if repo:
        root_attrs[OI["GIT_REPO"]] = repo
    if ref:
        root_attrs[OI["GIT_REF"]] = ref
    for tag in tags:
        root_attrs[f"tag.{tag}"] = True

    root = tracer.start_span("Claude Code Turn", start_time=start_ns, attributes=root_attrs)
    _stamp_sdk(root)
    root_ctx = trace.set_span_in_context(root)

    ask_last_text = ""
    ask_max_end_ns: int | None = start_ns

    for turn in ask_turns:
        last_text, turn_max_end = _build_turn_spans_into(
            tracer, root_ctx, turn, session_id, max_chars, subagent_emitter,
            start_floor_ns=start_ns,
        )
        if last_text:
            ask_last_text = last_text
        if turn_max_end is not None:
            if ask_max_end_ns is None or turn_max_end > ask_max_end_ns:
                ask_max_end_ns = turn_max_end

    if ask_last_text:
        root.set_attribute(OI["OUTPUT"], middle_truncate(ask_last_text, max_chars))

    root.end(end_time=ask_max_end_ns)
